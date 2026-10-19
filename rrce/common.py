import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(task):
    path = Path(__file__).resolve().parents[1] / "configs" / f"{task}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_pipeline(model, checkpoint=None, dtype=torch.float32):
    pipe = StableDiffusionPipeline.from_pretrained(
        model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        pipe.unet.load_state_dict(state, strict=True)
    for module in (pipe.unet, pipe.vae, pipe.text_encoder):
        module.requires_grad_(False).eval()
    return pipe


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def image_files(folder):
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not files:
        raise ValueError(f"No images in {folder}")
    return files


def image_tensor(path, flip=False):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
        array = np.array(image, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    if flip and random.random() < 0.5:
        tensor = tensor.flip(-1)
    return tensor.to("cuda")


@torch.no_grad()
def text_embedding(pipe, prompt):
    ids = pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.to("cuda")
    return pipe.text_encoder(ids)[0].detach()


@torch.no_grad()
def noisy_latent(pipe, path, flip=False, timestep_limit=1000):
    if Path(path).suffix == ".pt":
        latent = torch.load(path, map_location="cuda", weights_only=True)
        if isinstance(latent, dict):
            latent = latent["latents"]
        if latent.ndim == 3:
            latent = latent.unsqueeze(0)
    else:
        pixels = image_tensor(path, flip).to(pipe.vae.dtype)
        latent = pipe.vae.encode(pixels).latent_dist.sample() * pipe.vae.config.scaling_factor
    noise = torch.randn_like(latent)
    timestep = torch.randint(0, timestep_limit, (latent.shape[0],), device="cuda")
    return pipe.scheduler.add_noise(latent, noise, timestep), timestep, noise


def sample_rows(pipe, rows, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "images.csv").exists():
        raise FileExistsError(destination / "images.csv")
    output = []
    for index, row in enumerate(rows):
        generator = torch.Generator(device="cuda").manual_seed(int(row["seed"]))
        with torch.no_grad():
            image = pipe(row["prompt"], num_inference_steps=50, guidance_scale=7.5,
                         height=512, width=512, generator=generator).images[0]
        name = f"{index:06d}.png"
        image.save(destination / name)
        output.append({**row, "image_path": name})
    write_csv(destination / "images.csv", output)
    return output
