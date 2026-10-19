import os
import random
from pathlib import Path

import numpy as np
import torch

from rrce.models.diffusion import add_learned_token


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def capture_random_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def cpu_state_dict(module):
    return {name: value.detach().cpu() for name, value in module.state_dict().items()}


def save_training_checkpoint(path, model, optimizer, tokens, next_block, config):
    atomic_torch_save(
        {
            "format": "rrce_training_v2",
            "unet": cpu_state_dict(model.pipeline.unet),
            "optimizer": optimizer.state_dict(),
            "tokens": tokens,
            "next_block": next_block,
            "random_state": capture_random_state(),
            "config": config.to_dict(),
        },
        path,
    )


def restore_training_checkpoint(path, model, optimizer, config):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "rrce_training_v2" or state["config"] != config.to_dict():
        raise ValueError("Resume requires a matching RRCE checkpoint and resolved configuration")
    model.pipeline.unet.load_state_dict(state["unet"], strict=True)
    for token, value in state["tokens"].items():
        token_id = add_learned_token(model.pipeline, token, config.concepts.initializer)
        with torch.no_grad():
            model.pipeline.text_encoder.get_input_embeddings().weight[token_id].copy_(value)
    model.pipeline.text_encoder.requires_grad_(False).eval()
    optimizer.load_state_dict(state["optimizer"])
    restore_random_state(state["random_state"])
    return state["next_block"], state["tokens"]
