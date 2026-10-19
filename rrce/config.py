from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import types
from dataclasses import dataclass
from pathlib import Path
from typing import get_args, get_origin, get_type_hints


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    precision: str
    seed: int
    allow_tf32: bool
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int
    log_every: int


@dataclass(frozen=True)
class ModelConfig:
    pretrained: str
    text_encoder: str
    resolution: int


@dataclass(frozen=True)
class DataConfig:
    target_images: str
    retain_manifest: str
    gallery_manifest: str
    manifests: str
    i2p: str
    coco: str
    coco_reference: str
    nudenet: str
    wikiart: str


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float
    beta_min: float
    beta_max: float
    beta2: float
    epsilon: float
    weight_decay: float


@dataclass(frozen=True)
class TrainingConfig:
    output: str
    parameter_scope: str
    defense_blocks: int
    pairs_per_block: int
    batch_size: int
    guidance_scale: float
    projection_weight: float
    retain_weight: float
    checkpoint_every_blocks: int
    optimizer: OptimizerConfig


@dataclass(frozen=True)
class RecoveryConfig:
    steps: int
    learning_rate: float
    weight_decay: float
    betas: list[float]
    epsilon: float
    batch_size: int
    repeats: int
    flip_probability: float
    center_crop: bool
    timestep_upper: int
    token_prefix: str


@dataclass(frozen=True)
class SamplingConfig:
    scheduler: str
    precision: str
    steps: int
    guidance_scale: float
    batch_size: int


@dataclass(frozen=True)
class AssetConfig:
    retain_seeds: list[int]
    retain_sampling: SamplingConfig
    gallery_sampling: SamplingConfig


@dataclass(frozen=True)
class EvaluationConfig:
    output: str
    sampling: dict[str, SamplingConfig]
    counts: dict[str, int]
    retain_per_concept: int
    retain_template: str
    coco_label_template: str
    retain_clip_model: str
    quality_clip_model: str
    clip_batch_size: int
    detector_batch_size: int
    nudity_threshold: float
    dog_threshold: float
    nudity_labels: list[str]
    wikiart_target_id: int
    wikiart_num_classes: int
    wikiart_top_k: int
    fid_mode: str
    fid_batch_size: int


@dataclass(frozen=True)
class CCEConfig:
    token: str
    seed: int
    steps: int
    learning_rate: float
    weight_decay: float
    betas: list[float]
    epsilon: float
    noise_seed_offset: int
    gallery_size: int | None
    prompt_template: str
    sampling: SamplingConfig


@dataclass(frozen=True)
class UDConfig:
    seed: int
    prefix_tokens: int
    learning_rate: float
    weight_decay: float
    betas: list[float]
    epsilon: float
    timesteps: list[int]
    iterations_per_timestep: int
    projection_iterations: int
    sampling: SamplingConfig


@dataclass(frozen=True)
class ProtocolConfig:
    seed_namespace: str
    explicit_threshold: float
    nudity_erased_prompts: int
    nudity_erased_repeats: int
    nudity_attack_prompts: int
    coco_labels_file: str
    ud_prompts_file: str


@dataclass(frozen=True)
class ConceptConfig:
    task: str
    target: str
    initializer: str
    template: str
    inversion_templates: list[str]
    retain: list[str]
    anchors: list[str]


@dataclass(frozen=True)
class ExperimentConfig:
    task: str
    concepts_file: str
    runtime: RuntimeConfig
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    recovery: RecoveryConfig
    assets: AssetConfig
    evaluation: EvaluationConfig
    cce: CCEConfig
    ud: UDConfig
    protocol: ProtocolConfig
    concepts: ConceptConfig
    coco_labels: list[str]
    ud_prompts: list[str]

    def to_dict(self):
        return dataclasses.asdict(self)


def merge_config(base, updates):
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def read_config_tree(path, ancestors=()):
    path = Path(path).resolve()
    if path in ancestors:
        raise ValueError("Circular configuration inheritance")
    data = json.loads(path.read_text(encoding="utf-8"))
    parent = data.pop("extends", None)
    if parent is not None:
        data = merge_config(read_config_tree(path.parent / parent, (*ancestors, path)), data)
    return data


def apply_overrides(data, overrides):
    data = copy.deepcopy(data)
    for override in overrides:
        key, separator, raw_value = override.partition("=")
        if not separator:
            raise ValueError(f"Expected KEY=VALUE, received {override!r}")
        target = data
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown configuration key: {key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown configuration key: {key}")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        target[parts[-1]] = value
    return data


def construct_config(expected, value, location="config"):
    if dataclasses.is_dataclass(expected):
        fields = get_type_hints(expected)
        if not isinstance(value, dict):
            raise ValueError(f"{location}: expected an object")
        if set(value) != set(fields):
            missing = set(fields) - set(value)
            extra = set(value) - set(fields)
            raise ValueError(f"{location}: missing={sorted(missing)}, unknown={sorted(extra)}")
        return expected(
            **{
                key: construct_config(kind, value[key], f"{location}.{key}")
                for key, kind in fields.items()
            }
        )
    origin = get_origin(expected)
    if origin is types.UnionType:
        if value is None and type(None) in get_args(expected):
            return None
        return construct_config(get_args(expected)[0], value, location)
    if origin is list:
        if not isinstance(value, list):
            raise ValueError(f"{location}: expected a list")
        return [construct_config(get_args(expected)[0], item, location) for item in value]
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{location}: expected an object")
        return {
            key: construct_config(get_args(expected)[1], item, f"{location}.{key}")
            for key, item in value.items()
        }
    if expected is float and type(value) in (int, float):
        return float(value)
    if type(value) is not expected:
        raise ValueError(f"{location}: expected {expected.__name__}")
    return value


def validate_config(config):
    if config.task not in ("nudity", "dog", "van_gogh") or config.concepts.task != config.task:
        raise ValueError("Task and concept configuration do not match")
    if config.training.parameter_scope not in ("full", "cross_attention"):
        raise ValueError("parameter_scope must be full or cross_attention")
    if config.runtime.precision not in ("fp32", "bf16"):
        raise ValueError("Training supports fp32 or bf16")
    positive = [
        config.model.resolution,
        config.runtime.log_every,
        config.runtime.prefetch_factor,
        config.training.defense_blocks,
        config.training.pairs_per_block,
        config.training.batch_size,
        config.training.checkpoint_every_blocks,
        config.recovery.steps,
        config.recovery.batch_size,
        config.recovery.repeats,
        config.recovery.timestep_upper,
        config.cce.steps,
        config.ud.prefix_tokens,
        config.ud.iterations_per_timestep,
        config.ud.projection_iterations,
        config.evaluation.clip_batch_size,
        config.evaluation.detector_batch_size,
        config.evaluation.fid_batch_size,
        *config.evaluation.counts.values(),
    ]
    if min(positive) <= 0 or config.runtime.num_workers < 0 or config.model.resolution % 8:
        raise ValueError("Invalid size, step count, worker count, or image resolution")
    optimizer = config.training.optimizer
    if not 0 <= optimizer.beta_min <= optimizer.beta_max < 1 or not 0 <= optimizer.beta2 < 1:
        raise ValueError("Invalid adaptive momentum bounds")
    for section in (optimizer, config.recovery, config.cce, config.ud):
        if section.learning_rate <= 0 or section.epsilon <= 0 or section.weight_decay < 0:
            raise ValueError("Invalid optimizer hyperparameters")
    for section in (config.recovery, config.cce, config.ud):
        if len(section.betas) != 2 or any(not 0 <= beta < 1 for beta in section.betas):
            raise ValueError("Adam betas must contain two values in [0, 1)")
    sampling = [
        *config.evaluation.sampling.values(),
        config.assets.retain_sampling,
        config.assets.gallery_sampling,
        config.cce.sampling,
        config.ud.sampling,
    ]
    for settings in sampling:
        if settings.scheduler not in ("ddim", "pndm", "dpm") or settings.precision not in (
            "fp32",
            "bf16",
            "fp16",
        ):
            raise ValueError("Unsupported sampling scheduler or precision")
        if min(settings.steps, settings.batch_size, settings.guidance_scale) <= 0:
            raise ValueError("Invalid sampling configuration")
    if set(config.evaluation.counts) != {"erased", "retain", "ud", "cce", "coco"}:
        raise ValueError("Expected all five evaluation splits")
    if set(config.evaluation.sampling) != {"erased", "retain", "coco"}:
        raise ValueError("Expected erased, retain, and coco sampling settings")
    if len(config.concepts.retain) != 20 or len(set(config.concepts.retain)) != 20:
        raise ValueError("Each task requires 20 distinct retain concepts")
    if not config.concepts.anchors or len(set(config.concepts.anchors)) != len(
        config.concepts.anchors
    ):
        raise ValueError("Anchor concepts must be nonempty and unique")
    if len(config.coco_labels) != 80 or not config.concepts.inversion_templates:
        raise ValueError("Missing protocol labels or recovery templates")
    if (
        len(config.assets.retain_seeds) != len(set(config.assets.retain_seeds))
        or not config.assets.retain_seeds
    ):
        raise ValueError("Retain seeds must be nonempty and unique")
    if (
        not config.ud.timesteps
        or min(config.ud.timesteps) < 0
        or len(set(config.ud.timesteps)) != len(config.ud.timesteps)
    ):
        raise ValueError("UD timesteps must be nonnegative and distinct")
    if not 0 <= config.recovery.flip_probability <= 1:
        raise ValueError("Invalid recovery flip probability")
    if config.cce.gallery_size is not None and config.cce.gallery_size <= 0:
        raise ValueError("CCE gallery size must be positive")
    if (
        config.evaluation.counts["retain"]
        != len(config.concepts.retain) * config.evaluation.retain_per_concept
    ):
        raise ValueError("Retain sample count disagrees with the concept protocol")


def load_config(path, overrides=()):
    path = Path(path)
    data = apply_overrides(read_config_tree(path), overrides)
    if "concepts" not in data:
        data["concepts"] = json.loads(
            (path.parent / data["concepts_file"]).read_text(encoding="utf-8")
        )
        data["coco_labels"] = json.loads(
            (path.parent / data["protocol"]["coco_labels_file"]).read_text(encoding="utf-8")
        )
        prompt_file = data["protocol"]["ud_prompts_file"]
        data["ud_prompts"] = (
            json.loads((path.parent / prompt_file).read_text(encoding="utf-8"))
            if prompt_file
            else []
        )
    config = construct_config(ExperimentConfig, data)
    validate_config(config)
    return config


def config_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path, help="Task JSON configuration")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a configuration value; repeat for multiple values",
    )
    parser.add_argument(
        "--print-config", action="store_true", help="Print the resolved configuration and exit"
    )
    return parser


def config_from_args(args):
    config = load_config(args.config, args.set)
    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2))
        raise SystemExit(0)
    return config
