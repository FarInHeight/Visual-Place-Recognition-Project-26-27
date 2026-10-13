"""Shared image transforms and path helpers."""

from pathlib import Path

from torchvision import transforms

IMAGE_SIZE = (224, 224)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def make_transform(training=False):
    steps = [transforms.Resize(IMAGE_SIZE, antialias=True)]
    if training:
        steps.append(transforms.RandAugment(num_ops=3))
    steps.extend((transforms.ToTensor(), transforms.Normalize(MEAN, STD)))
    return transforms.Compose(steps)


def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    paths = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No images found under {folder}")
    return paths
