"""Training helpers, reproducibility, and checkpoint persistence."""

import random

import numpy as np
import torch


def seed_everything(seed):
    """Set seed for reproducibility across common Python ML libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Ensure deterministic behavior in cuDNN (at the cost of performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def contrastive_loss(descriptors, labels, margin=0.5):
    """TODO: implement the project's contrastive objective and return a scalar loss."""
    raise NotImplementedError(
        "Students should implement the contrastive loss using descriptors, labels, and margin."
    )


def save_checkpoint(path, model, optimizer, scaler, epoch, best_recall, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_validation_recall_at_1": best_recall,
        "seed": args.seed,
        "backbone_training": args.backbone_training,
        "backbone": args.backbone,
        "similarity_margin": args.similarity_margin,
        "training_split_hash": getattr(args, "training_split_hash", None),
        "training_config": {
            "batch_size": args.batch_size,
            "images_per_place": args.images_per_place,
            "min_images_per_place": args.min_images_per_place,
            "val_fraction": args.val_fraction,
            "steps_per_epoch": args.steps_per_epoch,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)
