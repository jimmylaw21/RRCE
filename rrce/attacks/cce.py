import random
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image
from tqdm.auto import trange

from rrce.data.images import read_image
from rrce.data.manifests import validate_gallery_separation
from rrce.evaluation.sampling import generate_images
from rrce.io import MetricLogger, manifest_images, read_manifest
from rrce.models.diffusion import add_learned_token, load_inference_pipeline, tokenize_prompts
from rrce.runtime import check_loss, make_generator, seed_everything
from rrce.training.checkpoints import atomic_torch_save


class ConceptEmbeddingAttack:
    def __init__(self, pipeline, config, logger):
        self.pipeline = pipeline
        self.config = config
        self.settings = config.cce
        self.logger = logger

    @torch.no_grad()
    def encode_gallery(self, paths):
        latents = []
        pipeline = self.pipeline
        for index, path in enumerate(paths):
            pixels = read_image(
                path,
                self.config.model.resolution,
                center_crop=True,
                interpolation=Image.Resampling.LANCZOS,
            )
            pixels = pixels.unsqueeze(0).to(device=pipeline.device, dtype=pipeline.vae.dtype)
            posterior = pipeline.vae.encode(pixels).latent_dist
            latent = posterior.sample(
                generator=make_generator(pipeline.device, self.settings.seed + index)
            )
            latents.append(latent * pipeline.vae.config.scaling_factor)
        return latents

    def fit(self, gallery_paths):
        pipeline = self.pipeline
        settings = self.settings
        seed_everything(settings.seed)
        pipeline.text_encoder.to(dtype=torch.float32)
        token_id = add_learned_token(pipeline, settings.token, self.config.concepts.initializer)
        embedding = pipeline.text_encoder.get_input_embeddings().weight
        embedding.requires_grad_(True)
        prompt = settings.prompt_template.format(settings.token)
        input_ids = tokenize_prompts(pipeline, prompt)
        latents = self.encode_gallery(gallery_paths)
        optimizer = torch.optim.AdamW(
            [embedding],
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
            betas=tuple(settings.betas),
            eps=settings.epsilon,
        )
        original = embedding.detach().clone()
        frozen_rows = torch.arange(embedding.shape[0], device=embedding.device) != token_id
        order = list(range(len(latents)))
        rng = random.Random(settings.seed)
        cursor = len(order)
        for step in trange(1, settings.steps + 1, desc="CCE"):
            if cursor == len(order):
                order = list(range(len(latents)))
                rng.shuffle(order)
                cursor = 0
            latent = latents[order[cursor]]
            cursor += 1
            generator = make_generator(
                pipeline.device, settings.seed + settings.noise_seed_offset + step
            )
            noise = torch.randn(
                latent.shape, device=latent.device, dtype=latent.dtype, generator=generator
            )
            timestep = torch.randint(
                0,
                pipeline.scheduler.config.num_train_timesteps,
                (latent.shape[0],),
                device=latent.device,
                generator=generator,
            )
            noisy = pipeline.scheduler.add_noise(latent, noise, timestep)
            optimizer.zero_grad(set_to_none=True)
            condition = pipeline.text_encoder(input_ids)[0].to(pipeline.unet.dtype)
            prediction = pipeline.unet(noisy, timestep, encoder_hidden_states=condition).sample
            loss = functional.mse_loss(prediction.float(), noise.float())
            check_loss(loss, "CCE")
            loss.backward()
            embedding.grad[frozen_rows] = 0
            optimizer.step()
            with torch.no_grad():
                embedding[frozen_rows] = original[frozen_rows]
            if step % self.config.runtime.log_every == 0 or step == settings.steps:
                self.logger.write(
                    stage="cce", step=step, loss=loss.item(), gallery_index=order[cursor - 1]
                )
        result = embedding[token_id].detach().cpu().clone()
        pipeline.text_encoder.requires_grad_(False).eval().to(dtype=pipeline.unet.dtype)
        return result


def run_cce(config, checkpoint, manifest, destination):
    settings = config.cce
    gallery = read_manifest(config.data.gallery_manifest, required=("prompt", "seed", "image_path"))
    rows = read_manifest(manifest, expected=config.evaluation.counts["cce"])
    validate_gallery_separation(config, gallery, rows)
    paths = manifest_images(config.data.gallery_manifest, gallery)
    destination = Path(destination)
    pipeline = load_inference_pipeline(config, settings.sampling, checkpoint)
    attack = ConceptEmbeddingAttack(pipeline, config, MetricLogger(destination / "attack.jsonl"))
    embedding = attack.fit(paths)
    atomic_torch_save({settings.token: embedding}, destination / "embedding.pt")
    return generate_images(pipeline, rows, destination, config, settings.sampling)
