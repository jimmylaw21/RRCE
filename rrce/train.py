import argparse
import copy
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import trange

from .common import image_files, load_config, load_pipeline, noisy_latent, read_csv, seed_everything, text_embedding, write_csv
from .inversion import invert
from .optimizer import AdaptiveAdamW


def train(args):
    seed_everything(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    config = load_config(args.task)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    pipe = load_pipeline(args.model)
    teacher = copy.deepcopy(pipe.unet).requires_grad_(False).eval()
    targets = image_files(args.target_images)
    if len(targets) < 2:
        raise ValueError("Recovery requires at least two target images")
    retain_rows = read_csv(args.retain_manifest)
    retain_groups = {}
    for row in retain_rows:
        path = Path(args.retain_manifest).parent / row["image_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        retain_groups.setdefault(row["prompt"], []).append(path)
    params = [p for name, p in pipe.unet.named_parameters() if args.task != "van_gogh" or "attn2" in name]
    optimizers = {name: AdaptiveAdamW(params, lr=2e-5, beta_min=0.1, beta_max=0.9,
                                     beta2=0.999, eps=1e-8, weight_decay=0.0) for name in ("erase", "retain")}
    prompts = list(retain_groups)
    if set(prompts) != set(config["retain"] + [""]):
        raise ValueError("Retain manifest must contain the 20 configured concepts and an empty prompt")
    conditions = {prompt: text_embedding(pipe, prompt) for prompt in prompts + config["anchors"] + [""]}
    active = config["target"]
    history = []
    for block in range(3):
        if block:
            seed_everything(42)
            active = f"<rrce_{block}>"
            size = max(1, len(targets) // 2)
            subset = targets[(block - 1) * size:block * size]
            embedding = invert(pipe, subset, active, config["initializer"], config["inversion_templates"])
            torch.save({active: embedding}, output / f"recovery_{block}.pt")
        for parameter in params:
            parameter.requires_grad_(True)
        pipe.unet.train()
        target_condition = text_embedding(pipe, active)
        retain_order = []
        anchor_order = []
        for step in trange(500, desc=f"Defense {block + 1}/3"):
            if not retain_order:
                retain_order = random.sample(prompts, len(prompts))
            retain_prompt = retain_order.pop()
            if not anchor_order:
                anchor_order = random.sample(config["anchors"], len(config["anchors"]))
            anchor_condition = conditions[anchor_order.pop()]
            erase_x, erase_t, _ = noisy_latent(pipe, random.choice(targets))
            retain_x, retain_t, _ = noisy_latent(pipe, random.choice(retain_groups[retain_prompt]))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                unconditional = teacher(erase_x, erase_t, conditions[""]).sample
                target = teacher(erase_x, erase_t, target_condition).sample
                anchor = teacher(erase_x, erase_t, anchor_condition).sample
                erase_target = unconditional + 3.0 * (anchor - target)
                retain_target = teacher(retain_x, retain_t, conditions[retain_prompt]).sample
            optimizers["erase"].zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = pipe.unet(erase_x, erase_t, target_condition).sample
                erase_loss = F.mse_loss(prediction.float(), erase_target.float())
            if not torch.isfinite(erase_loss):
                raise RuntimeError("Non-finite erase loss")
            erase_loss.backward()
            optimizers["erase"].step()
            optimizers["retain"].zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = pipe.unet(retain_x, retain_t, conditions[retain_prompt]).sample
                retain_loss = F.mse_loss(prediction.float(), retain_target.float())
            if not torch.isfinite(retain_loss):
                raise RuntimeError("Non-finite retain loss")
            retain_loss.backward()
            optimizers["retain"].step()
            history.append({"block": block, "step": step, "erase": erase_loss.item(), "retain": retain_loss.item()})
    torch.save({name: value.detach().cpu() for name, value in pipe.unet.state_dict().items()}, output / "unet.pt")
    write_csv(output / "losses.csv", history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("nudity", "dog", "van_gogh"), required=True)
    parser.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--target-images", required=True)
    parser.add_argument("--retain-manifest", required=True)
    parser.add_argument("--output", required=True)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
