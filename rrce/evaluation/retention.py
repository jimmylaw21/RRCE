import torch
import torch.nn.functional as functional

from rrce.evaluation.detectors import load_rgb_images


@torch.inference_mode()
def evaluate_retention(paths, rows, config):
    from transformers import CLIPModel, CLIPProcessor

    settings = config.evaluation
    texts = [settings.retain_template.format(label) for label in config.concepts.retain]
    texts.extend(settings.coco_label_template.format(label) for label in config.coco_labels)
    labels = [*config.concepts.retain, *[f"coco:{label}" for label in config.coco_labels]]
    processor = CLIPProcessor.from_pretrained(settings.retain_clip_model)
    model = CLIPModel.from_pretrained(settings.retain_clip_model).to(config.runtime.device).eval()
    inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt").to(
        config.runtime.device
    )
    text_features = functional.normalize(model.get_text_features(**inputs), dim=-1)
    results = []
    for start in range(0, len(paths), settings.clip_batch_size):
        batch_paths = paths[start : start + settings.clip_batch_size]
        inputs = processor(images=load_rgb_images(batch_paths), return_tensors="pt").to(
            config.runtime.device
        )
        image_features = functional.normalize(model.get_image_features(**inputs), dim=-1)
        predictions = (image_features @ text_features.T).argmax(-1).cpu().tolist()
        for row, prediction in zip(rows[start : start + settings.clip_batch_size], predictions):
            expected = config.concepts.retain.index(row["target_label"])
            results.append({"hit": prediction == expected, "predicted_label": labels[prediction]})
    return results
