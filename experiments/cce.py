import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import trange

from rrce.common import image_tensor, load_config, load_pipeline, read_csv, sample_rows, seed_everything


def learn_embedding(pipe, gallery, task):
    config = load_config(task)
    token = "<cce>"
    if pipe.tokenizer.add_tokens([token]) != 1:
        raise ValueError("CCE token already exists")
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    embedding = pipe.text_encoder.get_input_embeddings().weight
    token_id = pipe.tokenizer.convert_tokens_to_ids(token)
    initializer = pipe.tokenizer(config["initializer"], add_special_tokens=False).input_ids
    if len(initializer) != 1:
        raise ValueError("Initializer must be one token")
    with torch.no_grad():
        embedding[token_id].copy_(embedding[initializer[0]])
    embedding.requires_grad_(True)
    prompt = ("a painting in the style of {}" if task == "van_gogh" else "a photo of {}").format(token)
    ids = pipe.tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids.cuda()
    latents = []
    with torch.no_grad():
        for index, path in enumerate(gallery):
            posterior = pipe.vae.encode(image_tensor(path).to(pipe.vae.dtype)).latent_dist
            latent = posterior.sample(generator=torch.Generator(device="cuda").manual_seed(2027 + index))
            latents.append(latent * pipe.vae.config.scaling_factor)
    optimizer = torch.optim.AdamW([embedding], lr=0.005, weight_decay=0.0)
    rng = random.Random(2027)
    order = []
    for step in trange(1, 3001, desc="CCE"):
        if not order:
            order = rng.sample(range(len(latents)), len(latents))
        latent = latents[order.pop()]
        generator = torch.Generator(device="cuda").manual_seed(2027 + 1000000 + step)
        noise = torch.randn(latent.shape, generator=generator, device="cuda", dtype=latent.dtype)
        timestep = torch.randint(0, 1000, (1,), generator=generator, device="cuda")
        noisy = pipe.scheduler.add_noise(latent, noise, timestep)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            condition = pipe.text_encoder(ids)[0]
            prediction = pipe.unet(noisy, timestep, condition).sample
            loss = F.mse_loss(prediction.float(), noise.float())
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite CCE loss")
        loss.backward()
        embedding.grad[:token_id].zero_()
        embedding.grad[token_id + 1:].zero_()
        optimizer.step()
    pipe.text_encoder.requires_grad_(False).eval()
    return embedding[token_id].detach().cpu().clone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("nudity", "dog", "van_gogh"))
    parser.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gallery", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seed_everything(2027)
    gallery_rows = read_csv(args.gallery)
    rows = read_csv(args.manifest)
    expected = {"dog": 30, "van_gogh": 6}.get(args.task)
    if expected and len(gallery_rows) != expected:
        raise ValueError(f"Expected {expected} gallery images")
    if len(rows) != (95 if args.task == "nudity" else 500):
        raise ValueError("Incomplete CCE test manifest")
    if {r["seed"] for r in gallery_rows} & {r["seed"] for r in rows}:
        raise ValueError("CCE training and test seeds overlap")
    if args.task == "nudity":
        test_prompts = {r["prompt"].replace("<cce>", "").strip().casefold() for r in rows}
        if {r["prompt"].strip().casefold() for r in gallery_rows} & test_prompts:
            raise ValueError("CCE training and test prompts overlap")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    pipe = load_pipeline(args.model, args.checkpoint)
    gallery = [Path(args.gallery).parent / row["image_path"] for row in gallery_rows]
    embedding = learn_embedding(pipe, gallery, args.task)
    torch.save({"<cce>": embedding}, output / "embedding.pt")
    sample_rows(pipe, rows, output)


if __name__ == "__main__":
    main()
