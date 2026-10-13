"""Main command-line entry point for the VPR course-project experiments."""

import json

import torch

from evaluation import evaluate_benchmarks
from models import ResNetDescriptor
from parser import parse_args
from training import train
from utils import seed_everything


def _validate_args(args):
    if args.num_workers < 0 or args.eval_batch_size < 1:
        raise ValueError("Worker count must be non-negative and evaluation batch size positive.")
    if args.epochs < 1 or args.steps_per_epoch < 1 or args.batch_size < 1:
        raise ValueError("Epochs, steps per epoch, and batch size must be positive.")
    if args.images_per_place < 2 or args.min_images_per_place < 1:
        raise ValueError("--images-per-place must be at least two and minimum images positive.")
    if not -1.0 <= args.similarity_margin <= 1.0:
        raise ValueError("--similarity-margin must be between -1 and 1.")
    if args.positive_dist_threshold < 0:
        raise ValueError("--positive-dist-threshold must be non-negative.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative.")
    if args.mode == "train" and not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between zero and one.")
    if args.mode == "train" and args.checkpoint is not None:
        raise ValueError("--checkpoint is for evaluation; use --resume when training.")
    if args.mode == "evaluate" and args.resume is not None:
        raise ValueError("--resume is only valid with --mode train.")


def _write_results(path, results):
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(results, indent=2) + "\n")
    temporary_path.replace(path)


def main(argv=None):
    args = parse_args(argv)
    _validate_args(args)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; input resolution: 224x224")

    if args.mode == "train":
        train(args, device)
        return

    checkpoint = (
        torch.load(args.checkpoint, map_location=device, weights_only=False)
        if args.checkpoint
        else None
    )
    backbone_name = checkpoint.get("backbone", args.backbone) if checkpoint else args.backbone
    models_to_evaluate = [
        (
            f"ImageNet {backbone_name} + GAP",
            ResNetDescriptor(
                "avg",
                pretrained=True,
                backbone_training="frozen",
                backbone_name=backbone_name,
            ).to(device),
        )
    ]
    if checkpoint is not None:
        if "backbone_training" not in checkpoint or "model_state_dict" not in checkpoint:
            raise ValueError("The checkpoint is missing model or backbone metadata.")
        tuned = ResNetDescriptor(
            "gem",
            backbone_training=checkpoint["backbone_training"],
            backbone_name=backbone_name,
        ).to(device)
        tuned.load_state_dict(checkpoint["model_state_dict"])
        models_to_evaluate.append(
            (
                f"GSV-XS {backbone_name} + GeM "
                f"({checkpoint['backbone_training']} backbone)",
                tuned,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for model_name, model in models_to_evaluate:
        print(f"\nEvaluating {model_name}")
        all_results[model_name] = evaluate_benchmarks(
            model, args.data_root, device, args
        )
    result_path = args.output_dir / "results.json"
    _write_results(result_path, all_results)
    print(f"Saved metrics to {result_path}")


if __name__ == "__main__":
    main()
