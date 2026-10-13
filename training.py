"""GSV-XS training loop and resumable checkpoint handling."""

import hashlib
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.train_dataset import PlaceDataset, group_gsv_images
from datasets.transforms import make_transform
from evaluation import validation_recall
from models import ResNetDescriptor
from utils import contrastive_loss, save_checkpoint


def _restore_rng_state(checkpoint):
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state"]]
        )


def _write_history(path, data):
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(data, indent=2) + "\n")
    temporary_path.replace(path)


def train(args, device):
    places = group_gsv_images(args.train_dir, args.min_images_per_place)
    all_ids = sorted(places)
    if len(all_ids) < 2:
        raise ValueError("GSV-XS must contain at least two eligible places.")
    rng = np.random.default_rng(args.seed)
    shuffled_ids = rng.permutation(all_ids)
    val_count = max(1, round(len(all_ids) * args.val_fraction))
    if val_count >= len(all_ids):
        raise ValueError("The validation split leaves no training places.")
    validation_ids = sorted(shuffled_ids[:val_count].tolist())
    training_ids = sorted(shuffled_ids[val_count:].tolist())
    split_payload = json.dumps(
        {"train": training_ids, "validation": validation_ids},
        separators=(",", ":"),
    )
    args.training_split_hash = hashlib.sha256(split_payload.encode()).hexdigest()

    too_small_train = [
        place_id
        for place_id in training_ids
        if len(places[place_id]) < args.images_per_place
    ]
    too_small_validation = [
        place_id for place_id in validation_ids if len(places[place_id]) < 2
    ]
    if too_small_train or too_small_validation:
        raise ValueError(
            "Not enough images per place for the requested training/validation "
            "sampling; lower the minimum-image threshold or adjust "
            "--images-per-place."
        )
    if len(training_ids) < args.batch_size:
        raise ValueError(
            f"Only {len(training_ids)} training places are available, fewer than "
            f"--batch-size={args.batch_size}."
        )

    print(
        f"GSV-XS split: {len(training_ids)} train places, "
        f"{len(validation_ids)} validation places; split seed={args.seed}"
    )
    training_dataset = PlaceDataset(
        {place_id: places[place_id] for place_id in training_ids},
        make_transform(training=True),
        args.images_per_place,
    )
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        drop_last=True,
    )

    model = ResNetDescriptor(
        "gem",
        pretrained=args.resume is None,
        backbone_training=args.backbone_training,
        backbone_name=args.backbone,
    ).to(device)
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, best_recall = 1, -1.0
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("backbone_training") != args.backbone_training:
            raise ValueError("Resume checkpoint backbone mode does not match --backbone-training.")
        if checkpoint.get("backbone", "resnet50") != args.backbone:
            raise ValueError("Resume checkpoint architecture does not match --backbone.")
        if checkpoint.get("similarity_margin") != args.similarity_margin:
            raise ValueError("Resume checkpoint similarity margin does not match --similarity-margin.")
        saved_config = checkpoint.get("training_config")
        if saved_config is not None:
            current_config = {
                "batch_size": args.batch_size,
                "images_per_place": args.images_per_place,
                "min_images_per_place": args.min_images_per_place,
                "val_fraction": args.val_fraction,
                "steps_per_epoch": args.steps_per_epoch,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
            }
            mismatched = [
                key
                for key, value in saved_config.items()
                if current_config.get(key) != value
            ]
            if mismatched:
                raise ValueError(
                    "Resume checkpoint training settings do not match: "
                    + ", ".join(mismatched)
                )
        saved_split_hash = checkpoint.get("training_split_hash")
        if saved_split_hash and saved_split_hash != args.training_split_hash:
            raise ValueError(
                "Resume checkpoint was created with a different GSV-XS place split."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_recall = checkpoint["best_validation_recall_at_1"]
        _restore_rng_state(checkpoint)
        print(f"Resuming at epoch {start_epoch} from {args.resume}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "training_history.json"
    previous_history = (
        json.loads(history_path.read_text())
        if args.resume and history_path.is_file()
        else {}
    )
    history = previous_history.get("history", [])
    best_epoch = previous_history.get("best_epoch", start_epoch - 1)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        loss_total, steps = 0.0, 0
        for images, labels in training_loader:
            images = images.flatten(0, 1).to(device, non_blocking=True)
            labels = labels.flatten().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                descriptors = model(images)
            loss = contrastive_loss(descriptors, labels, args.similarity_margin)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_total += loss.item()
            steps += 1
            if steps >= args.steps_per_epoch:
                break

        if steps == 0:
            raise RuntimeError("The training data loader produced no batches.")
        val_recalls = validation_recall(model, places, validation_ids, device, args)
        current_recall = val_recalls[0]
        improved = current_recall >= best_recall
        if improved:
            best_recall = current_recall
            best_epoch = epoch
        mean_loss = loss_total / steps
        history.append(
            {
                "epoch": epoch,
                "mean_train_loss": mean_loss,
                "validation_R@1": val_recalls[0],
                "validation_R@5": val_recalls[1],
                "validation_R@10": val_recalls[2],
                "validation_R@20": val_recalls[3],
            }
        )
        print(
            f"Epoch {epoch}/{args.epochs}: loss={mean_loss:.4f}, "
            f"validation R@1/R@5/R@10/R@20={val_recalls}"
        )
        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            scaler,
            epoch,
            best_recall,
            args,
        )
        if improved:
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                scaler,
                epoch,
                best_recall,
                args,
            )
        _write_history(
            history_path,
            {
                "backbone_training": args.backbone_training,
                "backbone": args.backbone,
                "similarity_margin": args.similarity_margin,
                "training_args": {
                    "epochs": args.epochs,
                    "steps_per_epoch": args.steps_per_epoch,
                    "batch_size_places": args.batch_size,
                    "images_per_place": args.images_per_place,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "seed": args.seed,
                },
                "history": history,
                "best_epoch": best_epoch,
                "best_validation_R@1": best_recall,
            },
        )
