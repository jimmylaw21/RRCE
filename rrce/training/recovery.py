import torch
import torch.nn.functional as functional
from tqdm.auto import trange

from rrce.data.images import RecoveryDataset
from rrce.data.streams import InfiniteLoader, make_loader
from rrce.models.diffusion import add_learned_token
from rrce.runtime import check_loss, seed_everything


class ConceptRecovery:
    def __init__(self, model, config, logger):
        self.model = model
        self.config = config
        self.logger = logger

    def discover(self, paths, round_index):
        config = self.config
        settings = config.recovery
        seed_everything(config.runtime.seed)
        pipeline = self.model.pipeline
        pipeline.unet.requires_grad_(False).eval()
        pipeline.text_encoder.requires_grad_(False).train()
        token = f"<{settings.token_prefix}_{round_index + 1}>"
        token_id = add_learned_token(pipeline, token, config.concepts.initializer)
        embeddings = pipeline.text_encoder.get_input_embeddings().weight
        original = embeddings.detach().clone()
        embeddings.requires_grad_(True)
        frozen_rows = torch.arange(len(pipeline.tokenizer), device=embeddings.device) != token_id
        dataset = RecoveryDataset(
            paths,
            pipeline.tokenizer,
            token,
            config.concepts.inversion_templates,
            settings,
            config.model.resolution,
            round_index,
            config.training.defense_blocks - 1,
        )
        batches = InfiniteLoader(make_loader(dataset, settings.batch_size, config.runtime))
        optimizer = torch.optim.AdamW(
            [embeddings],
            lr=settings.learning_rate,
            betas=tuple(settings.betas),
            eps=settings.epsilon,
            weight_decay=settings.weight_decay,
        )
        if settings.timestep_upper > pipeline.scheduler.config.num_train_timesteps:
            raise ValueError("Recovery timestep upper bound exceeds the diffusion schedule")
        for step in trange(settings.steps, desc=f"Recovery {round_index + 1}"):
            batch = batches.next()
            optimizer.zero_grad(set_to_none=True)
            with self.model.autocast():
                latent = self.model.encode_pixels(batch["pixels"])
            noise = torch.randn_like(latent)
            timestep = torch.randint(
                0, settings.timestep_upper, (latent.shape[0],), device=latent.device
            )
            noisy = pipeline.scheduler.add_noise(latent, noise, timestep)
            with self.model.autocast():
                condition = pipeline.text_encoder(batch["input_ids"].to(self.model.device))[0]
                prediction = pipeline.unet(noisy, timestep, encoder_hidden_states=condition).sample
            loss = functional.mse_loss(prediction.float(), noise.float())
            check_loss(loss, "recovery")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                embeddings[frozen_rows] = original[frozen_rows]
            if (step + 1) % config.runtime.log_every == 0 or step + 1 == settings.steps:
                self.logger.write(
                    stage="recovery", round=round_index + 1, step=step + 1, loss=loss.item()
                )
        pipeline.text_encoder.requires_grad_(False).eval()
        return token, embeddings[token_id].detach().cpu().clone()
