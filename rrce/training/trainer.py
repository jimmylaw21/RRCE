from pathlib import Path

from tqdm.auto import trange

from rrce.data.images import ImageDataset, find_images, retain_datasets
from rrce.data.streams import BalancedConceptStream, InfiniteLoader, make_loader
from rrce.io import MetricLogger, write_json
from rrce.models.diffusion import DiffusionModel, select_trainable_parameters
from rrce.optim.erase_retain import EraseRetainOptimizer
from rrce.runtime import configure_runtime
from rrce.training.checkpoints import (
    atomic_torch_save,
    cpu_state_dict,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from rrce.training.objectives import ErasureRetentionObjective
from rrce.training.recovery import ConceptRecovery


class RRCETrainer:
    def __init__(self, config, model=None):
        self.config = config
        configure_runtime(config.runtime)
        self.target_paths = find_images(config.data.target_images)
        if len(self.target_paths) < config.training.defense_blocks - 1:
            raise ValueError("At least one target image per recovery round is required")
        self.retain_data = retain_datasets(
            config.data.retain_manifest, config.concepts, config.model.resolution
        )
        self.model = model if model is not None else DiffusionModel.from_config(config)
        self.teacher = self.model.make_teacher()
        parameters, self.parameter_names = select_trainable_parameters(
            self.model.pipeline.unet, config.training.parameter_scope
        )
        self.optimizer = EraseRetainOptimizer(parameters, config.training.optimizer)
        self.objective = ErasureRetentionObjective(self.model, self.teacher, config.training)

    def train_defense_block(self, block, active_concept, logger):
        config = self.config
        select_trainable_parameters(self.model.pipeline.unet, config.training.parameter_scope)
        self.model.pipeline.unet.train()
        target_batches = InfiniteLoader(
            make_loader(
                ImageDataset(self.target_paths, config.model.resolution),
                config.training.batch_size,
                config.runtime,
            )
        )
        retain_batches = {
            prompt: InfiniteLoader(make_loader(dataset, config.training.batch_size, config.runtime))
            for prompt, dataset in self.retain_data.items()
        }
        anchors = BalancedConceptStream(config.concepts.anchors)
        retains = BalancedConceptStream(list(self.retain_data))
        prompts = list(
            dict.fromkeys([active_concept, "", *config.concepts.anchors, *self.retain_data])
        )
        conditions = {prompt: self.model.encode_prompts(prompt) for prompt in prompts}

        def expand_condition(prompt, batch_size):
            return conditions[prompt].expand(batch_size, -1, -1)

        for step in trange(config.training.pairs_per_block, desc=f"Defense {block + 1}"):
            anchor = anchors.next()
            erase_latent, erase_timestep = self.model.noisy_training_batch(target_batches.next())
            batch_size = erase_latent.shape[0]
            erase_loss = self.objective.erase_loss(
                erase_latent,
                erase_timestep,
                expand_condition(active_concept, batch_size),
                expand_condition(anchor, batch_size),
                expand_condition("", batch_size),
            )
            erase_statistics = self.optimizer.update("erase", erase_loss)
            retain_prompt = retains.next()
            retain_latent, retain_timestep = self.model.noisy_training_batch(
                retain_batches[retain_prompt].next()
            )
            retain_loss = self.objective.retain_loss(
                retain_latent,
                retain_timestep,
                expand_condition(retain_prompt, retain_latent.shape[0]),
            )
            retain_statistics = self.optimizer.update("retain", retain_loss)
            if (
                step + 1
            ) % config.runtime.log_every == 0 or step + 1 == config.training.pairs_per_block:
                logger.write(
                    stage="defense",
                    block=block + 1,
                    pair=step + 1,
                    erase_loss=erase_loss.item(),
                    retain_loss=retain_loss.item(),
                    erase_beta=erase_statistics["last_beta1"],
                    retain_beta=retain_statistics["last_beta1"],
                    erase_cosine=erase_statistics["last_rho"],
                    retain_cosine=retain_statistics["last_rho"],
                )
        self.model.pipeline.unet.requires_grad_(False).eval()

    def run(self, resume=None):
        config = self.config
        output = Path(config.training.output)
        if resume is None:
            output.mkdir(parents=True, exist_ok=False)
            first_block, tokens = 0, {}
        else:
            if not output.is_dir():
                raise ValueError("Resume output directory does not exist")
            first_block, tokens = restore_training_checkpoint(
                resume, self.model, self.optimizer, config
            )
        write_json(output / "config.json", config.to_dict())
        write_json(output / "trainable_parameters.json", self.parameter_names)
        logger = MetricLogger(output / "training.jsonl")
        recovery = ConceptRecovery(self.model, config, logger)
        try:
            for block in range(first_block, config.training.defense_blocks):
                active_concept = config.concepts.target
                if block:
                    write_json(output / "status.json", {"stage": "recovery", "round": block})
                    active_concept, embedding = recovery.discover(self.target_paths, block - 1)
                    tokens[active_concept] = embedding
                    atomic_torch_save({active_concept: embedding}, output / f"recovery_{block}.pt")
                write_json(output / "status.json", {"stage": "defense", "block": block + 1})
                self.train_defense_block(block, active_concept, logger)
                if (
                    (block + 1) % config.training.checkpoint_every_blocks == 0
                    or block + 1 == config.training.defense_blocks
                ):
                    save_training_checkpoint(
                        output / "latest.pt", self.model, self.optimizer, tokens, block + 1, config
                    )
            atomic_torch_save(cpu_state_dict(self.model.pipeline.unet), output / "unet.pt")
            atomic_torch_save(tokens, output / "recovery_tokens.pt")
            write_json(
                output / "status.json",
                {"stage": "complete", "defense_blocks": config.training.defense_blocks},
            )
        except Exception as error:
            write_json(
                output / "status.json", {"stage": "failed", "error_type": type(error).__name__}
            )
            raise
