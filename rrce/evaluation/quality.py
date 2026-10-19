import gc
from pathlib import Path

import torch

from rrce.data.images import find_images
from rrce.evaluation.detectors import load_rgb_images


@torch.inference_mode()
def paired_clip_scores(paths, rows, config):
    from transformers import CLIPModel, CLIPProcessor

    settings = config.evaluation
    processor = CLIPProcessor.from_pretrained(settings.quality_clip_model)
    model = CLIPModel.from_pretrained(settings.quality_clip_model).to(config.runtime.device).eval()
    scores = []
    for start in range(0, len(paths), settings.clip_batch_size):
        batch_rows = rows[start : start + settings.clip_batch_size]
        inputs = processor(
            text=[row["prompt"] for row in batch_rows],
            images=load_rgb_images(paths[start : start + settings.clip_batch_size]),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(config.runtime.device)
        scores.extend(model(**inputs).logits_per_image.diag().cpu().tolist())
    return scores


def evaluate_quality(paths, rows, config):
    from cleanfid import fid

    if len(find_images(config.data.coco_reference)) != config.evaluation.counts["coco"]:
        raise ValueError("COCO reference image count disagrees with the evaluation protocol")
    image_directory = paths[0].parent
    if set(find_images(image_directory)) != set(paths):
        raise ValueError("The FID image directory contains images outside the manifest")
    scores = paired_clip_scores(paths, rows, config)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    fid_score = fid.compute_fid(
        str(image_directory),
        str(Path(config.data.coco_reference)),
        mode=config.evaluation.fid_mode,
        batch_size=config.evaluation.fid_batch_size,
        num_workers=config.runtime.num_workers,
        device=torch.device(config.runtime.device),
    )
    summary = {"CLIP": sum(scores) / len(scores), "FID": float(fid_score)}
    return summary, [{"clip_logit": score} for score in scores]
