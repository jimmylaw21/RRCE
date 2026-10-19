import argparse

import torch
from diffusers import DPMSolverMultistepScheduler, PNDMScheduler

from rrce.common import load_pipeline, read_csv, sample_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", required=True, choices=("erased", "retain", "coco"))
    args = parser.parse_args()
    pipe = load_pipeline(args.model, args.checkpoint, dtype=torch.float16)
    scheduler = DPMSolverMultistepScheduler if args.split == "coco" else PNDMScheduler
    pipe.scheduler = scheduler.from_config(pipe.scheduler.config, skip_prk_steps=True)
    sample_rows(pipe, read_csv(args.manifest), args.output)


if __name__ == "__main__":
    main()
