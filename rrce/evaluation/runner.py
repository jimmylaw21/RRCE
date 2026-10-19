import gc
import json
from pathlib import Path

import torch

from rrce.data.manifests import evaluation_manifest
from rrce.evaluation.detectors import detect_target
from rrce.evaluation.quality import evaluate_quality
from rrce.evaluation.retention import evaluate_retention
from rrce.evaluation.sampling import generate_images
from rrce.io import file_digest, manifest_images, read_manifest, write_json, write_manifest
from rrce.models.diffusion import load_inference_pipeline


def evaluate_images(config, split, manifest):
    required = ("sample_id", "prompt", "seed", "image_path")
    if split == "retain":
        required += ("target_label",)
    rows = read_manifest(manifest, required=required, expected=config.evaluation.counts[split])
    paths = manifest_images(manifest, rows)
    if split == "coco":
        summary, details = evaluate_quality(paths, rows, config)
    else:
        details = (
            evaluate_retention(paths, rows, config)
            if split == "retain"
            else detect_target(paths, config)
        )
        metric = {"erased": "Erased", "retain": "Retain", "ud": "UD", "cce": "CCE"}[split]
        summary = {metric: 100 * sum(item["hit"] for item in details) / len(details)}
    records = [{**row, **detail} for row, detail in zip(rows, details)]
    write_manifest(Path(manifest).parent / "per_image.csv", records)
    write_json(Path(manifest).parent / "metrics.json", {"samples": len(rows), **summary})
    return summary


def run_evaluation(config, splits, checkpoint=None, metrics_only=False):
    from rrce.attacks.cce import run_cce
    from rrce.attacks.unlearndiff import run_unlearndiff

    checkpoint = Path(checkpoint or Path(config.training.output) / "unet.pt")
    checkpoint_hash = file_digest(checkpoint)
    output = Path(config.evaluation.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config.to_dict())
    completed = {}
    for split in splits:
        manifest = evaluation_manifest(config, split)
        rows = read_manifest(
            manifest,
            required=("sample_id", "prompt", "seed"),
            expected=config.evaluation.counts[split],
        )
        context = {
            "checkpoint_sha256": checkpoint_hash,
            "manifest_sha256": file_digest(manifest),
            "config": config.to_dict(),
        }
        if split == "cce":
            context["gallery_sha256"] = file_digest(config.data.gallery_manifest)
        destination = output / split
        metadata_path = destination / "generation.json"
        if destination.exists():
            if not metadata_path.is_file() or json.loads(metadata_path.read_text()) != context:
                raise ValueError(
                    f"Existing {split} output is incomplete or belongs to different inputs; select a new evaluation.output"
                )
        elif metrics_only:
            raise FileNotFoundError(f"No completed generation for {split}")
        else:
            destination.mkdir()
            if split == "cce":
                run_cce(config, checkpoint, manifest, destination)
            elif split == "ud":
                run_unlearndiff(config, checkpoint, manifest, destination)
            else:
                sampling = config.evaluation.sampling[split]
                pipeline = load_inference_pipeline(config, sampling, checkpoint)
                generate_images(pipeline, rows, destination, config, sampling)
                del pipeline
            write_json(metadata_path, context)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        completed.update(evaluate_images(config, split, destination / "images.csv"))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(
        output / "results.json",
        {
            "task": config.task,
            "checkpoint_sha256": checkpoint_hash,
            "evaluated_splits": splits,
            **completed,
        },
    )
    return completed
