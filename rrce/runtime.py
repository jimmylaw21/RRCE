import random
from contextlib import nullcontext

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_runtime(config):
    seed_everything(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select runtime.device=cpu for CPU checks")


def torch_dtype(precision):
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]


def autocast_context(device, precision):
    device = torch.device(device)
    if device.type == "cuda" and precision != "fp32":
        return torch.autocast(device_type="cuda", dtype=torch_dtype(precision))
    return nullcontext()


def make_generator(device, seed):
    return torch.Generator(device=device).manual_seed(int(seed) % (2**63 - 1))


def check_loss(loss, stage):
    if not torch.isfinite(loss) or not loss.requires_grad:
        raise RuntimeError(f"Invalid {stage} loss: finite gradients are required")
