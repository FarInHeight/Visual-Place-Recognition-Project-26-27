"""GSV-XS place-grouped data for contrastive training."""

import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.transforms import list_images


def place_id_from_path(path):
    fields = Path(path).name.split("@")
    if len(fields) < 3 or not fields[-2]:
        raise ValueError(f"Cannot read the place ID from GSV-XS filename: {path}")
    return fields[-2]


def group_gsv_images(train_dir, min_images_per_place):
    places = defaultdict(list)
    for path in list_images(train_dir):
        places[place_id_from_path(path)].append(path)
    return {
        place_id: paths
        for place_id, paths in places.items()
        if len(paths) >= min_images_per_place
    }


class PlaceDataset(Dataset):
    def __init__(self, places, transform, images_per_place):
        if images_per_place < 1:
            raise ValueError("images_per_place must be positive")
        self.place_ids = list(places)
        self.places = places
        self.transform = transform
        self.images_per_place = images_per_place

    def __len__(self):
        return len(self.place_ids)

    def __getitem__(self, index):
        place_id = self.place_ids[index]
        paths = random.sample(self.places[place_id], self.images_per_place)
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(self.transform(image.convert("RGB")))
        labels = torch.full((self.images_per_place,), index, dtype=torch.long)
        return torch.stack(images), labels
