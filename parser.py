"""Command-line arguments for training and evaluating VPR models."""

import argparse
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train or evaluate visual place recognition models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=("evaluate", "train"), required=True)
    parser.add_argument("--backbone", choices=("resnet18", "resnet50"), default="resnet50")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--train-dir", type=Path, default=Path("data/gsv_xs/train"))
    parser.add_argument("--checkpoint", type=Path, help="trained checkpoint to evaluate")
    parser.add_argument("--resume", type=Path, help="checkpoint to resume training from")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/resnet50_gem"),
    )
    parser.add_argument("--positive-dist-threshold", type=float, default=25.0)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32, help="places per training batch")
    parser.add_argument("--images-per-place", type=int, default=4)
    parser.add_argument("--min-images-per-place", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--backbone-training",
        choices=("frozen", "full"),
        default="full",
        help="freeze the backbone or fine-tune all its parameters",
    )
    parser.add_argument(
        "--similarity-margin",
        type=float,
        default=0.5,
        help="negative-pair cosine-similarity margin",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)
