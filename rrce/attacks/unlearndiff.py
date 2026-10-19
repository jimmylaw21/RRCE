import hashlib
from pathlib import Path

import torch
import torch.nn.functional as functional
from tqdm.auto import tqdm

from rrce.data.images import image_to_tensor
from rrce.evaluation.sampling import sampling_arguments
from rrce.io import MetricLogger, read_manifest, write_manifest
from rrce.models.diffusion import encode_prompt, load_inference_pipeline
from rrce.runtime import check_loss, make_generator, seed_everything


def attack_seed(*parts):
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def encode_token_embeddings(encoder, embeddings):
    text_model = encoder.text_model
    hidden = text_model.embeddings(inputs_embeds=embeddings)
    length = hidden.shape[1]
    causal_mask = torch.full(
        (length, length), torch.finfo(hidden.dtype).min, device=hidden.device, dtype=hidden.dtype
    ).triu(1)
    output = text_model.encoder(
        inputs_embeds=hidden,
        attention_mask=None,
        causal_attention_mask=causal_mask[None, None].expand(hidden.shape[0], -1, -1, -1),
        return_dict=True,
    )
    return text_model.final_layer_norm(output.last_hidden_state)


@torch.no_grad()
def project_probability_simplex(probabilities, iterations):
    lower = (probabilities - 1).amin(dim=-1, keepdim=True)
    upper = probabilities.amax(dim=-1, keepdim=True)
    for _ in range(iterations):
        middle = (lower + upper) / 2
        mass = (probabilities - middle).clamp(0, 1).sum(-1, keepdim=True)
        lower = torch.where(mass > 1, middle, lower)
        upper = torch.where(mass > 1, upper, middle)
    result = (probabilities - upper).clamp(0, 1)
    return result / result.sum(-1, keepdim=True).clamp_min(torch.finfo(result.dtype).eps)


class StraightThroughTokenSelection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probabilities, seed):
        flat = probabilities.reshape(-1, probabilities.shape[-1])
        normalized = flat / flat.sum(-1, keepdim=True).clamp_min(torch.finfo(flat.dtype).eps)
        token_ids = torch.multinomial(
            normalized, 1, generator=make_generator(probabilities.device, seed)
        ).squeeze(-1)
        return (
            functional.one_hot(token_ids, flat.shape[-1])
            .reshape_as(probabilities)
            .to(probabilities.dtype)
        )

    @staticmethod
    def backward(ctx, gradient):
        return functional.hardtanh(gradient), None


def prefix_embedding_layout(pipeline, prompt, prefix_length):
    tokenizer = pipeline.tokenizer
    max_length = tokenizer.model_max_length
    if prefix_length > max_length - 2:
        raise ValueError("The attack prefix leaves no room for BOS and EOS")
    text_ids = tokenizer(prompt, add_special_tokens=False, truncation=False).input_ids[
        : max_length - 2 - prefix_length
    ]
    ids = [tokenizer.bos_token_id, *text_ids]
    ids.extend([tokenizer.eos_token_id] * (max_length - len(ids)))
    table = pipeline.text_encoder.get_input_embeddings().weight.detach()
    original = table[torch.tensor([ids], device=table.device)]
    text_length = len(text_ids)

    def insert_prefix(attack_embeddings):
        return torch.cat(
            (
                original[:, :1],
                attack_embeddings,
                original[:, 1 : text_length + 1],
                original[:, text_length + prefix_length + 1 :],
            ),
            dim=1,
        )

    return table, insert_prefix, text_length


class UnlearnDiffAttack:
    def __init__(self, pipeline, config, logger):
        self.pipeline = pipeline
        self.config = config
        self.settings = config.ud
        self.logger = logger

    def optimize(self, row, reference_latent):
        pipeline = self.pipeline
        settings = self.settings
        table, insert_prefix, text_length = prefix_embedding_layout(
            pipeline, row["prompt"], settings.prefix_tokens
        )
        vocabulary = table.shape[0]
        probabilities = torch.nn.Parameter(
            torch.full(
                (1, settings.prefix_tokens, vocabulary),
                1 / vocabulary,
                device=table.device,
                dtype=torch.float32,
            )
        )
        optimizer = torch.optim.AdamW(
            [probabilities],
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
            betas=tuple(settings.betas),
            eps=settings.epsilon,
        )
        alphas = pipeline.scheduler.alphas_cumprod.to(
            device=table.device, dtype=reference_latent.dtype
        )
        if max(settings.timesteps) >= len(alphas):
            raise ValueError("UD timesteps exceed the diffusion schedule")
        for timestep in settings.timesteps:
            losses = []
            for iteration in range(1, settings.iterations_per_timestep + 1):
                optimizer.zero_grad(set_to_none=True)
                material = (settings.seed, row["sample_id"], timestep, iteration)
                selected = StraightThroughTokenSelection.apply(
                    probabilities, attack_seed(*material, "categorical")
                )
                embeddings = insert_prefix(selected.to(table.dtype) @ table)
                condition = encode_token_embeddings(pipeline.text_encoder, embeddings)
                noise = torch.randn(
                    reference_latent.shape,
                    device=table.device,
                    dtype=reference_latent.dtype,
                    generator=make_generator(table.device, attack_seed(*material, "noise")),
                )
                alpha = alphas[timestep]
                noisy = reference_latent * alpha.sqrt() + noise * (1 - alpha).sqrt()
                prediction = pipeline.unet(noisy, timestep, encoder_hidden_states=condition).sample
                loss = functional.mse_loss(prediction.float(), noise.float())
                check_loss(loss, "UD")
                loss.backward()
                if probabilities.grad is None or not torch.isfinite(probabilities.grad).all():
                    raise RuntimeError("UD did not produce finite token-probability gradients")
                optimizer.step()
                with torch.no_grad():
                    probabilities.copy_(
                        project_probability_simplex(probabilities, settings.projection_iterations)
                    )
                losses.append(loss.item())
            self.logger.write(
                sample_id=row["sample_id"], timestep=timestep, mean_loss=sum(losses) / len(losses)
            )
        with torch.no_grad():
            token_ids = probabilities.argmax(-1)
            condition = encode_token_embeddings(
                pipeline.text_encoder, insert_prefix(table[token_ids])
            )
        return condition, token_ids[0].cpu().tolist(), text_length


@torch.no_grad()
def reference_image_latent(pipeline, row, config):
    image = pipeline(
        row["prompt"],
        generator=make_generator(pipeline.device, row["seed"]),
        **sampling_arguments(config, config.ud.sampling),
    ).images[0]
    pixels = (
        image_to_tensor(image).unsqueeze(0).to(device=pipeline.device, dtype=pipeline.vae.dtype)
    )
    latent = pipeline.vae.encode(pixels).latent_dist.mean * pipeline.vae.config.scaling_factor
    return image, latent.detach()


def run_unlearndiff(config, checkpoint, manifest, destination):
    settings = config.ud
    seed_everything(settings.seed)
    rows = read_manifest(
        manifest, required=("sample_id", "prompt", "seed"), expected=config.evaluation.counts["ud"]
    )
    destination = Path(destination)
    (destination / "images").mkdir(parents=True, exist_ok=False)
    (destination / "references").mkdir()
    reference = load_inference_pipeline(config, settings.sampling)
    target = load_inference_pipeline(config, settings.sampling, checkpoint)
    attack = UnlearnDiffAttack(target, config, MetricLogger(destination / "attack.jsonl"))
    generated = []
    for index, row in enumerate(tqdm(rows, desc="UnlearnDiff")):
        image, latent = reference_image_latent(reference, row, config)
        image.save(destination / "references" / f"{index:06d}.png")
        condition, token_ids, text_length = attack.optimize(row, latent)
        with torch.no_grad():
            image = target(
                prompt_embeds=condition,
                negative_prompt_embeds=encode_prompt(target, ""),
                generator=make_generator(target.device, row["seed"]),
                **sampling_arguments(config, settings.sampling),
            ).images[0]
        relative = f"images/{index:06d}.png"
        image.save(destination / relative)
        generated.append(
            {
                **row,
                "image_path": relative,
                "attack_tokens": " ".join(map(str, token_ids)),
                "retained_text_tokens": text_length,
            }
        )
        write_manifest(destination / "images.partial.csv", generated)
    write_manifest(destination / "images.csv", generated)
    (destination / "images.partial.csv").unlink()
    return generated
