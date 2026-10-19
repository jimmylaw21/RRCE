import json
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from rrce.models.diffusion import freeze_pipeline


def make_tiny_pipeline(directory):
    torch.manual_seed(123)
    torch.set_num_threads(1)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tokens = ["<|startoftext|>", "<|endoftext|>"]
    tokens.extend(chr(code) for code in range(ord("a"), ord("z") + 1))
    tokens.extend(chr(code) + "</w>" for code in range(ord("a"), ord("z") + 1))
    (directory / "vocab.json").write_text(
        json.dumps({token: index for index, token in enumerate(tokens)})
    )
    (directory / "merges.txt").write_text("")
    tokenizer = CLIPTokenizer(
        str(directory / "vocab.json"), str(directory / "merges.txt"), model_max_length=16
    )
    text_encoder = CLIPTextModel(
        CLIPTextConfig(
            vocab_size=len(tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            max_position_embeddings=16,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    )
    unet = UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(16, 32),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32,
        attention_head_dim=4,
        norm_num_groups=8,
    )
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(16, 32),
        layers_per_block=1,
        norm_num_groups=8,
        sample_size=16,
    )
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        steps_offset=1,
        clip_sample=False,
    )
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    return freeze_pipeline(pipeline.to("cpu"))
