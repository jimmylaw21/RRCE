from pathlib import Path

import torch
from tqdm.auto import tqdm

from rrce.io import write_manifest
from rrce.runtime import make_generator


def sampling_arguments(config, sampling):
    return {
        "height": config.model.resolution,
        "width": config.model.resolution,
        "num_inference_steps": sampling.steps,
        "guidance_scale": sampling.guidance_scale,
    }


@torch.no_grad()
def generate_images(pipeline, rows, destination, config, sampling):
    destination = Path(destination)
    image_directory = destination / "images"
    image_directory.mkdir(parents=True, exist_ok=False)
    generated = []
    for start in tqdm(range(0, len(rows), sampling.batch_size), desc="Sampling"):
        batch = rows[start : start + sampling.batch_size]
        generators = [make_generator(pipeline.device, row["seed"]) for row in batch]
        images = pipeline(
            prompt=[row["prompt"] for row in batch],
            generator=generators,
            **sampling_arguments(config, sampling),
        ).images
        for offset, (row, image) in enumerate(zip(batch, images)):
            relative = f"images/{start + offset:06d}.png"
            image.save(destination / relative)
            generated.append({**row, "image_path": relative})
        write_manifest(destination / "images.partial.csv", generated)
    write_manifest(destination / "images.csv", generated)
    (destination / "images.partial.csv").unlink()
    return generated
