import argparse
from pathlib import Path

import torch

from rrce.common import load_config, load_pipeline, read_csv, sample_rows, write_csv
from .metrics import nudity_hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("nudity", "dog", "van_gogh"))
    parser.add_argument("--kind", required=True, choices=("retain", "gallery"))
    parser.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--manifests", default="data/manifests")
    parser.add_argument("--output", required=True)
    parser.add_argument("--nudenet-model")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    pipe = load_pipeline(args.model, dtype=torch.bfloat16)
    if args.kind == "gallery":
        rows = read_csv(Path(args.manifests) / f"{args.task}_gallery.csv")
        generated = sample_rows(pipe, rows, output)
        if args.task == "nudity":
            hits = nudity_hits([output / r["image_path"] for r in generated], args.nudenet_model)
            generated = [row for row, hit in zip(generated, hits) if hit]
            if not generated:
                raise ValueError("No explicit images passed the NudeNet gallery filter")
        write_csv(output / "gallery.csv", generated)
    else:
        config = load_config(args.task)
        prompts = config["retain"] + [""]
        rows = []
        for index, prompt in enumerate(prompts):
            for seed in range(8):
                with torch.no_grad():
                    latent = pipe(prompt, num_inference_steps=50, guidance_scale=3.0, height=512, width=512,
                                  generator=torch.Generator(device="cuda").manual_seed(seed), output_type="latent").images
                filename = f"{index:02d}_{seed}.pt"
                torch.save(latent.detach().float().cpu(), output / filename)
                rows.append({"prompt": prompt, "image_path": filename, "seed": seed})
        write_csv(output / "retain.csv", rows)


if __name__ == "__main__":
    main()
