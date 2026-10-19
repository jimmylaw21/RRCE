from pathlib import Path

import torch

from rrce.data.manifests import evaluation_manifest
from rrce.evaluation.sampling import generate_images, sampling_arguments
from rrce.io import read_manifest, write_json, write_manifest
from rrce.models.diffusion import load_inference_pipeline
from rrce.runtime import make_generator


def prepare_retain_latents(config):
    destination = Path(config.data.retain_manifest).parent
    destination.mkdir(parents=True, exist_ok=False)
    sampling = config.assets.retain_sampling
    pipeline = load_inference_pipeline(config, sampling)
    rows = []
    for concept_index, prompt in enumerate([*config.concepts.retain, ""]):
        for start in range(0, len(config.assets.retain_seeds), sampling.batch_size):
            seeds = config.assets.retain_seeds[start : start + sampling.batch_size]
            with torch.no_grad():
                latents = pipeline(
                    [prompt] * len(seeds),
                    output_type="latent",
                    generator=[make_generator(pipeline.device, seed) for seed in seeds],
                    **sampling_arguments(config, sampling),
                ).images
            for seed, latent in zip(seeds, latents):
                filename = f"{concept_index:02d}_{seed}.pt"
                torch.save(latent.detach().float().cpu(), destination / filename)
                rows.append({"prompt": prompt, "image_path": filename, "seed": seed})
    write_manifest(config.data.retain_manifest, rows)
    write_json(destination / "config.json", config.to_dict())


def prepare_cce_gallery(config):
    from rrce.evaluation.detectors import nudity_detections

    destination = Path(config.data.gallery_manifest).parent
    destination.mkdir(parents=True, exist_ok=False)
    manifest = evaluation_manifest(config, "gallery")
    rows = read_manifest(manifest, expected=config.cce.gallery_size)
    pipeline = load_inference_pipeline(config, config.assets.gallery_sampling)
    generated = generate_images(pipeline, rows, destination, config, config.assets.gallery_sampling)
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if config.task == "nudity":
        detections = nudity_detections(
            [destination / row["image_path"] for row in generated], config
        )
        generated = [row for row, result in zip(generated, detections) if result["hit"]]
        if not generated:
            raise ValueError("No gallery images passed the configured nudity threshold")
    write_manifest(config.data.gallery_manifest, generated)
    write_json(destination / "config.json", config.to_dict())
