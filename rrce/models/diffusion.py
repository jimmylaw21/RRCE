import copy

import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
)
from transformers import CLIPTextModel, CLIPTokenizer

from rrce.runtime import autocast_context, torch_dtype


def freeze_pipeline(pipeline):
    for module in (pipeline.unet, pipeline.vae, pipeline.text_encoder):
        module.requires_grad_(False).eval()
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def set_scheduler(pipeline, scheduler):
    classes = {"ddim": DDIMScheduler, "pndm": PNDMScheduler, "dpm": DPMSolverMultistepScheduler}
    options = {"skip_prk_steps": True} if scheduler == "pndm" else {}
    pipeline.scheduler = classes[scheduler].from_config(pipeline.scheduler.config, **options)


def load_inference_pipeline(config, sampling, checkpoint=None):
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.model.pretrained,
        torch_dtype=torch_dtype(sampling.precision),
        safety_checker=None,
        requires_safety_checker=False,
    )
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        pipeline.unet.load_state_dict(state, strict=True)
    set_scheduler(pipeline, sampling.scheduler)
    return freeze_pipeline(pipeline.to(config.runtime.device))


def select_trainable_parameters(unet, scope):
    if scope not in ("full", "cross_attention"):
        raise ValueError(f"Unknown parameter scope: {scope}")
    unet.requires_grad_(False)
    parameters = []
    names = []
    candidates = {"Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"}
    for module_name, module in unet.named_modules():
        if module.__class__.__name__ not in candidates:
            continue
        if scope == "cross_attention" and "attn2" not in module_name:
            continue
        for parameter_name, parameter in module.named_parameters(recurse=False):
            parameter.requires_grad_(True)
            parameters.append(parameter)
            names.append(f"{module_name}.{parameter_name}")
    if not parameters:
        raise ValueError("No trainable U-Net parameters were selected")
    return parameters, names


def add_learned_token(pipeline, token, initializer):
    initializer_ids = pipeline.tokenizer(initializer, add_special_tokens=False).input_ids
    if len(initializer_ids) != 1:
        raise ValueError("The initializer must map to exactly one tokenizer token")
    if pipeline.tokenizer.add_tokens([token]) != 1:
        raise ValueError(f"Token already exists: {token}")
    pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer))
    token_id = pipeline.tokenizer.convert_tokens_to_ids(token)
    embeddings = pipeline.text_encoder.get_input_embeddings().weight
    with torch.no_grad():
        embeddings[token_id].copy_(embeddings[initializer_ids[0]])
    return token_id


def tokenize_prompts(pipeline, prompts):
    return pipeline.tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(pipeline.device)


@torch.no_grad()
def encode_prompt(pipeline, prompts):
    return pipeline.text_encoder(tokenize_prompts(pipeline, prompts))[0].detach()


class DiffusionModel:
    def __init__(self, pipeline, runtime):
        self.pipeline = freeze_pipeline(pipeline)
        self.runtime = runtime
        self.device = pipeline.device

    @classmethod
    def from_config(cls, config):
        tokenizer = CLIPTokenizer.from_pretrained(config.model.text_encoder)
        text_encoder = CLIPTextModel.from_pretrained(config.model.text_encoder)
        pipeline = StableDiffusionPipeline.from_pretrained(
            config.model.pretrained,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(config.runtime.device)
        set_scheduler(pipeline, "ddim")
        return cls(pipeline, config.runtime)

    def autocast(self):
        return autocast_context(self.device, self.runtime.precision)

    def make_teacher(self):
        return copy.deepcopy(self.pipeline.unet).requires_grad_(False).eval()

    @torch.no_grad()
    def encode_prompts(self, prompts):
        with self.autocast():
            return encode_prompt(self.pipeline, prompts)

    @torch.no_grad()
    def encode_pixels(self, pixels):
        pixels = pixels.to(device=self.device, dtype=self.pipeline.vae.dtype)
        posterior = self.pipeline.vae.encode(pixels).latent_dist
        return posterior.sample() * self.pipeline.vae.config.scaling_factor

    @torch.no_grad()
    def noisy_training_batch(self, batch):
        if "latents" in batch:
            latent = batch["latents"].to(device=self.device, dtype=self.pipeline.unet.dtype)
        else:
            latent = self.encode_pixels(batch["pixels"])
        upper = self.pipeline.scheduler.config.num_train_timesteps
        timestep = torch.randint(0, upper, (latent.shape[0],), device=self.device)
        noise = torch.randn_like(latent)
        return self.pipeline.scheduler.add_noise(latent, noise, timestep), timestep

    def predict_noise(self, latent, timestep, condition, network=None):
        network = self.pipeline.unet if network is None else network
        with self.autocast():
            return network(latent, timestep, encoder_hidden_states=condition).sample
