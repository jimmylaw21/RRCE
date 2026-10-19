import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from rrce.io import manifest_images, read_manifest


def find_images(directory):
    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    paths = sorted(
        path
        for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
    if not paths:
        raise ValueError(f"No images found in {directory}")
    return paths


def image_to_tensor(
    image, resolution=None, center_crop=False, interpolation=Image.Resampling.BICUBIC
):
    image = image.convert("RGB")
    if center_crop:
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
    if resolution is not None:
        image = image.resize((resolution, resolution), interpolation)
    array = np.asarray(image, dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def read_image(path, resolution, center_crop=False, interpolation=Image.Resampling.BICUBIC):
    with Image.open(path) as image:
        return image_to_tensor(image, resolution, center_crop, interpolation)


class ImageDataset(Dataset):
    def __init__(self, paths, resolution):
        self.paths = list(paths)
        self.resolution = resolution

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return {"pixels": read_image(self.paths[index], self.resolution)}


class LatentDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        value = torch.load(self.paths[index], map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            value = value["latents"]
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 3 or not torch.isfinite(value).all():
            raise ValueError("Expected one finite latent tensor per retain file")
        return {"latents": value.float()}


def retain_datasets(manifest, concepts, resolution):
    rows = read_manifest(manifest, required=("prompt", "image_path"))
    paths = manifest_images(manifest, rows)
    groups = {}
    for row, path in zip(rows, paths):
        groups.setdefault(row["prompt"], []).append(path)
    if set(groups) != set(concepts.retain + [""]):
        raise ValueError("Retain data must cover the configured 20 concepts and the empty prompt")
    datasets = {}
    for prompt, group in groups.items():
        latent_files = [path.suffix == ".pt" for path in group]
        if any(latent_files) and not all(latent_files):
            raise ValueError("A retain concept cannot mix images and cached latents")
        datasets[prompt] = (
            LatentDataset(group) if all(latent_files) else ImageDataset(group, resolution)
        )
    return datasets


class RecoveryDataset(Dataset):
    def __init__(self, paths, tokenizer, token, templates, config, resolution, round_index, rounds):
        size = max(1, len(paths) // rounds)
        start = (round_index * size) % len(paths)
        selected = [paths[(start + index) % len(paths)] for index in range(size)]
        self.images = [read_image(path, resolution, config.center_crop) for path in selected]
        self.length = len(paths) * config.repeats
        self.flip_probability = config.flip_probability
        self.token_ids = [
            tokenizer(
                template.format(token),
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
            for template in templates
        ]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        token_ids = random.choice(self.token_ids).clone()
        pixels = self.images[index % len(self.images)].clone()
        if random.random() < self.flip_probability:
            pixels = pixels.flip(-1)
        return {"pixels": pixels, "input_ids": token_ids}
