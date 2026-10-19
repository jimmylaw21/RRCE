import random

import torch
import torch.nn.functional as F
from tqdm import trange

from .common import noisy_latent


def invert(pipe, paths, token, initializer, templates):
    pipe.unet.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False)
    tokenizer = pipe.tokenizer
    if tokenizer.add_tokens([token]) != 1:
        raise ValueError(f"Token already exists: {token}")
    pipe.text_encoder.resize_token_embeddings(len(tokenizer))
    weight = pipe.text_encoder.get_input_embeddings().weight
    token_id = tokenizer.convert_tokens_to_ids(token)
    initializer_ids = tokenizer(initializer, add_special_tokens=False).input_ids
    if len(initializer_ids) != 1:
        raise ValueError("Initializer must be a single token")
    with torch.no_grad():
        weight[token_id].copy_(weight[initializer_ids[0]])
    original = weight.detach().clone()
    weight.requires_grad_(True)
    mask = torch.arange(len(tokenizer), device=weight.device) != token_id
    optimizer = torch.optim.AdamW([weight], lr=0.005)
    ids = [tokenizer(template.format(token), padding="max_length", max_length=77,
                     truncation=True, return_tensors="pt").input_ids.to("cuda") for template in templates]
    for step in trange(3000, desc="Recovery"):
        optimizer.zero_grad(set_to_none=True)
        latent, timestep, noise = noisy_latent(pipe, random.choice(paths), flip=True, timestep_limit=999)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            condition = pipe.text_encoder(random.choice(ids))[0]
            prediction = pipe.unet(latent, timestep, condition).sample
            loss = F.mse_loss(prediction.float(), noise.float())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite recovery loss at step {step}")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            weight[mask] = original[mask]
    pipe.text_encoder.requires_grad_(False).eval()
    return weight[token_id].detach().cpu().clone()
