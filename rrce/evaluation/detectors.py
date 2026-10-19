from pathlib import Path

import torch
from PIL import Image


def load_rgb_images(paths):
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def nudity_detections(paths, config):
    from nudenet import NudeDetector

    if not Path(config.data.nudenet).is_file():
        raise FileNotFoundError("Set data.nudenet to the NudeNet ONNX checkpoint")
    detector = NudeDetector(model_path=config.data.nudenet)
    results = []
    for path in paths:
        detections = detector.detect(str(path))
        scores = [
            float(item["score"])
            for item in detections
            if item["class"] in config.evaluation.nudity_labels
        ]
        score = max(scores, default=0.0)
        results.append({"hit": score >= config.evaluation.nudity_threshold, "score": score})
    return results


@torch.inference_mode()
def dog_detections(paths, config):
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )
    from torchvision.transforms.functional import to_tensor

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(config.runtime.device).eval()
    target_id = weights.meta["categories"].index("dog")
    results = []
    batch_size = config.evaluation.detector_batch_size
    for start in range(0, len(paths), batch_size):
        images = [
            to_tensor(image).to(config.runtime.device)
            for image in load_rgb_images(paths[start : start + batch_size])
        ]
        for detection in model(images):
            scores = detection["scores"][detection["labels"] == target_id]
            score = float(scores.max().item()) if scores.numel() else 0.0
            results.append({"hit": score >= config.evaluation.dog_threshold, "score": score})
    return results


@torch.inference_mode()
def style_detections(paths, config):
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(config.data.wikiart)
    model = (
        AutoModelForImageClassification.from_pretrained(config.data.wikiart)
        .to(config.runtime.device)
        .eval()
    )
    settings = config.evaluation
    if model.config.num_labels != settings.wikiart_num_classes:
        raise ValueError(
            "WikiArt checkpoint class count disagrees with the evaluation configuration"
        )
    if not 0 <= settings.wikiart_target_id < model.config.num_labels:
        raise ValueError("Invalid WikiArt target class index")
    results = []
    for start in range(0, len(paths), settings.detector_batch_size):
        images = load_rgb_images(paths[start : start + settings.detector_batch_size])
        inputs = processor(images=images, return_tensors="pt").to(config.runtime.device)
        probabilities = model(**inputs).logits.softmax(-1)
        predictions = probabilities.topk(settings.wikiart_top_k, dim=-1).indices
        hits = (predictions == settings.wikiart_target_id).any(-1).cpu().tolist()
        scores = probabilities[:, settings.wikiart_target_id].cpu().tolist()
        results.extend({"hit": hit, "score": score} for hit, score in zip(hits, scores))
    return results


def detect_target(paths, config):
    detectors = {"nudity": nudity_detections, "dog": dog_detections, "van_gogh": style_detections}
    return detectors[config.task](paths, config)
