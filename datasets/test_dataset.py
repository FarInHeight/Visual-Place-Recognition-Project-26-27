"""Image loading and benchmark folder definitions for retrieval evaluation."""

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from datasets.transforms import make_transform


class PathDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = list(paths)
        self.transform = transform or make_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), index


def coordinates_from_paths(paths):
    coordinates = []
    for path in paths:
        fields = Path(path).name.split("@")
        if len(fields) < 3:
            raise ValueError(f"Cannot read UTM coordinates from filename: {path}")
        try:
            coordinates.append((float(fields[1]), float(fields[2])))
        except ValueError as exc:
            raise ValueError(f"Invalid UTM coordinates in filename: {path}") from exc
    return np.asarray(coordinates, dtype=np.float64)


def benchmark_sets(data_root):
    root = Path(data_root)
    svox_root = root / "svox/images/test"
    return [
        (
            "SF-XS",
            root / "sf_xs/test/database",
            [("test", root / "sf_xs/test/queries")],
        ),
        (
            "Tokyo-XS",
            root / "tokyo_xs/test/database",
            [("test", root / "tokyo_xs/test/queries")],
        ),
        (
            "SVOX",
            svox_root / "gallery",
            [
                ("snow", svox_root / "queries_snow"),
                ("rain", svox_root / "queries_rain"),
                ("sun", svox_root / "queries_sun"),
                ("night", svox_root / "queries_night"),
                ("overcast", svox_root / "queries_overcast"),
            ],
        ),
    ]
