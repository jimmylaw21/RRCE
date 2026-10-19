import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(module, *arguments):
    subprocess.run([sys.executable, "-m", module, *map(str, arguments)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("nudity", "dog", "van_gogh"))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--stage", required=True, choices=("train", "evaluate"))
    args = parser.parse_args()
    task = args.task
    data = Path(args.data_root)
    output = Path(args.output_root) / task
    assets = data / task
    manifests = data / "manifests"
    checkpoint = output / "train/unet.pt"
    shared = ("--task", task, "--model", args.model)
    if args.stage == "train":
        if not (assets / "target").is_dir():
            raise FileNotFoundError(assets / "target")
        if not (assets / "retain/retain.csv").exists():
            run("experiments.assets", *shared, "--kind", "retain", "--output", assets / "retain")
        run("rrce.train", *shared, "--target-images", assets / "target",
            "--retain-manifest", assets / "retain/retain.csv", "--output", output / "train")
        return
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    for required in (manifests / "coco.csv", data / "coco_reference"):
        if not required.exists():
            raise FileNotFoundError(required)
    if task == "nudity" and not (data / "models/nudenet.onnx").is_file():
        raise FileNotFoundError(data / "models/nudenet.onnx")
    if task == "van_gogh" and not (data / "models/wikiart").is_dir():
        raise FileNotFoundError(data / "models/wikiart")
    gallery = assets / "gallery/gallery.csv"
    if not gallery.exists():
        run("experiments.assets", *shared, "--kind", "gallery", "--manifests", manifests,
            "--output", assets / "gallery", "--nudenet-model", data / "models/nudenet.onnx")
    for split in ("erased", "retain", "cce", "ud", "coco"):
        directory = output / split
        manifest = manifests / ("coco.csv" if split == "coco" else f"{task}_{split}.csv")
        metric_path = output / f"{split}.json"
        if metric_path.exists():
            continue
        if split in {"erased", "retain", "coco"}:
            run("experiments.sample", "--model", args.model, "--checkpoint", checkpoint,
                "--manifest", manifest, "--output", directory, "--split", split)
        elif split == "cce":
            run("experiments.cce", *shared, "--checkpoint", checkpoint, "--gallery", gallery,
                "--manifest", manifest, "--output", directory)
        else:
            run("experiments.ud", *shared, "--checkpoint", checkpoint,
                "--manifest", manifest, "--output", directory)
        run("experiments.metrics", "--task", task, "--split", split, "--manifest", directory / "images.csv",
            "--nudenet-model", data / "models/nudenet.onnx", "--wikiart-model", data / "models/wikiart",
            "--coco-reference", data / "coco_reference", "--output", metric_path)
    summary = {}
    for split in ("erased", "retain", "cce", "ud", "coco"):
        summary.update(json.loads((output / f"{split}.json").read_text(encoding="utf-8")))
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
