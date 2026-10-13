"""Descriptor extraction and recall evaluation."""

import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from datasets.test_dataset import (
    PathDataset,
    benchmark_sets,
    coordinates_from_paths,
)
from datasets.transforms import IMAGE_SUFFIXES, make_transform

RECALL_VALUES = (1, 5, 10, 20)


def extract_descriptors(model, paths, device, batch_size, num_workers):
    loader = DataLoader(
        PathDataset(paths, make_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    descriptors = np.empty((len(paths), model.output_dim), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for images, indices in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                batch_descriptors = model(images)
            batch_descriptors = F.normalize(batch_descriptors.float(), dim=1)
            descriptors[indices.numpy()] = batch_descriptors.cpu().numpy()
    return descriptors


def recall_at_n(
    database_descriptors,
    query_descriptors,
    database_paths,
    query_paths,
    threshold,
):
    if len(database_paths) == 0 or len(query_paths) == 0:
        raise ValueError("Recall evaluation requires non-empty database and query sets.")
    neighbors = NearestNeighbors(n_jobs=-1)
    neighbors.fit(coordinates_from_paths(database_paths))
    positives = neighbors.radius_neighbors(
        coordinates_from_paths(query_paths),
        radius=threshold,
        return_distance=False,
    )
    no_positive_queries = sum(len(indices) == 0 for indices in positives)
    k = min(max(RECALL_VALUES), len(database_paths))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    database = torch.from_numpy(database_descriptors).to(device)
    queries = torch.from_numpy(query_descriptors)
    hits = np.zeros(len(RECALL_VALUES), dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(queries), 256):
            query_batch = queries[start : start + 256].to(device)
            predictions = (query_batch @ database.T).topk(k, dim=1).indices.cpu().numpy()
            for offset, ranked_indices in enumerate(predictions):
                positive_indices = positives[start + offset]
                for value_index, recall_value in enumerate(RECALL_VALUES):
                    if np.intersect1d(ranked_indices[:recall_value], positive_indices).size:
                        hits[value_index] += 1
    recalls = (hits / len(query_paths) * 100).round(2).tolist()
    return {
        **{
            f"R@{value}": recall
            for value, recall in zip(RECALL_VALUES, recalls)
        },
        "queries": len(query_paths),
        "queries_without_positives": int(no_positive_queries),
        "positive_distance_threshold_m": threshold,
    }


def evaluate_benchmarks(model, data_root, device, args):
    results = {}
    for dataset_name, database_folder, query_sets in benchmark_sets(data_root):
        database_paths = sorted(
            path
            for path in database_folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not database_paths:
            raise FileNotFoundError(f"No images found under {database_folder}")
        database_descriptors = extract_descriptors(
            model, database_paths, device, args.eval_batch_size, args.num_workers
        )
        results[dataset_name] = {}
        for subset_name, query_folder in query_sets:
            query_paths = sorted(
                path
                for path in query_folder.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not query_paths:
                raise FileNotFoundError(f"No images found under {query_folder}")
            query_descriptors = extract_descriptors(
                model, query_paths, device, args.eval_batch_size, args.num_workers
            )
            metrics = recall_at_n(
                database_descriptors,
                query_descriptors,
                database_paths,
                query_paths,
                args.positive_dist_threshold,
            )
            results[dataset_name][subset_name] = metrics
            print(f"{dataset_name} / {subset_name}: {metrics}")
    return results


def validation_recall(model, places, validation_ids, device, args):
    rng = random.Random(args.seed)
    gallery_paths, query_paths = [], []
    gallery_ids = []
    for place_id in validation_ids:
        reference, query = rng.sample(places[place_id], 2)
        gallery_paths.append(reference)
        query_paths.append(query)
        gallery_ids.append(place_id)
    gallery_indices = {place_id: index for index, place_id in enumerate(gallery_ids)}
    gallery = extract_descriptors(
        model, gallery_paths, device, args.eval_batch_size, args.num_workers
    )
    queries = extract_descriptors(
        model, query_paths, device, args.eval_batch_size, args.num_workers
    )
    gallery = torch.from_numpy(gallery).to(device)
    queries = torch.from_numpy(queries).to(device)
    hits = np.zeros(len(RECALL_VALUES), dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(queries), 256):
            rankings = (queries[start : start + 256] @ gallery.T).topk(
                min(max(RECALL_VALUES), len(gallery)), dim=1
            ).indices.cpu().numpy()
            for offset, ranked_indices in enumerate(rankings):
                query_id = gallery_ids[start + offset]
                positive = gallery_indices[query_id]
                for value_index, recall_value in enumerate(RECALL_VALUES):
                    if positive in ranked_indices[:recall_value]:
                        hits[value_index] += 1
    return (hits / len(query_paths) * 100).round(2).tolist()
