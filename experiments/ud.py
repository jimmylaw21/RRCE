import argparse
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from rrce.common import load_pipeline, read_csv, seed_everything, text_embedding, write_csv


def generator(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def derived_seed(*parts):
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def encode_embeddings(encoder, embeddings):
    model = encoder.text_model
    hidden = model.embeddings(inputs_embeds=embeddings)
    length = hidden.shape[1]
    mask = torch.full((length, length), torch.finfo(hidden.dtype).min, device=hidden.device, dtype=hidden.dtype).triu(1)
    output = model.encoder(inputs_embeds=hidden, attention_mask=None,
                           causal_attention_mask=mask[None, None], return_dict=True)
    return model.final_layer_norm(output[0])


@torch.no_grad()
def project(probabilities):
    lower = (probabilities - 1).amin(dim=-1, keepdim=True)
    upper = probabilities.amax(dim=-1, keepdim=True)
    for _ in range(32):
        middle = (lower + upper) / 2
        mass = (probabilities - middle).clamp(0, 1).sum(-1, keepdim=True)
        lower = torch.where(mass > 1, middle, lower)
        upper = torch.where(mass > 1, upper, middle)
    result = (probabilities - upper).clamp(0, 1)
    return result / result.sum(-1, keepdim=True).clamp_min(torch.finfo(result.dtype).eps)


class SelectTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probabilities, seed):
        flat = probabilities.reshape(-1, probabilities.shape[-1])
        ids = torch.multinomial(flat, 1, generator=generator(seed)).squeeze(-1)
        return F.one_hot(ids, flat.shape[-1]).reshape_as(probabilities).to(probabilities.dtype)

    @staticmethod
    def backward(ctx, gradient):
        return F.hardtanh(gradient), None


def optimize(pipe, prompt, latent, k, sample_id):
    tokenizer = pipe.tokenizer
    text_ids = tokenizer(prompt, add_special_tokens=False, truncation=False).input_ids[:75 - k]
    ids = torch.tensor([[tokenizer.bos_token_id] + text_ids + [tokenizer.eos_token_id] * (76 - len(text_ids))], device="cuda")
    table = pipe.text_encoder.get_input_embeddings().weight.detach()
    original = table[ids]
    vocab = table.shape[0]
    probabilities = torch.nn.Parameter(torch.full((1, k, vocab), 1 / vocab, device="cuda"))
    optimizer = torch.optim.AdamW([probabilities], lr=0.01, weight_decay=0.1)
    alphas = pipe.scheduler.alphas_cumprod.to(device="cuda", dtype=latent.dtype)

    def insert(attack):
        return torch.cat((original[:, :1], attack, original[:, 1:len(text_ids) + 1], original[:, len(text_ids) + k + 1:]), dim=1)

    for timestep in range(10, 1000, 20):
        for iteration in range(1, 41):
            optimizer.zero_grad(set_to_none=True)
            selected = SelectTokens.apply(probabilities, derived_seed(2027, sample_id, timestep, iteration, "categorical"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = encode_embeddings(pipe.text_encoder, insert(selected.to(table.dtype) @ table))
                noise = torch.randn(latent.shape, device="cuda", dtype=latent.dtype,
                                    generator=generator(derived_seed(2027, sample_id, timestep, iteration, "noise")))
                alpha = alphas[timestep]
                noisy = latent * alpha.sqrt() + noise * (1 - alpha).sqrt()
                prediction = pipe.unet(noisy, timestep, hidden).sample
                loss = F.mse_loss(prediction.float(), noise.float())
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite UD loss")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                probabilities.copy_(project(probabilities))
    with torch.no_grad():
        token_ids = probabilities.argmax(-1)
        condition = encode_embeddings(pipe.text_encoder, insert(table[token_ids]))
    return condition, token_ids[0].cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("nudity", "dog", "van_gogh"))
    parser.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seed_everything(2027)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = read_csv(args.manifest)
    if len(rows) != (95 if args.task == "nudity" else 50):
        raise ValueError("Incomplete UD manifest")
    base = load_pipeline(args.model)
    target = load_pipeline(args.model, args.checkpoint)
    k = 5 if args.task == "nudity" else 3
    generated = []
    for index, row in enumerate(tqdm(rows, desc="UD")):
        with torch.no_grad():
            image = base(row["prompt"], num_inference_steps=50, guidance_scale=7.5,
                         generator=generator(int(row["seed"])), output_type="pt").images
            latent = base.vae.encode(image * 2 - 1).latent_dist.mean * base.vae.config.scaling_factor
        condition, tokens = optimize(target, row["prompt"], latent, k, index)
        with torch.no_grad():
            image = target(prompt_embeds=condition, negative_prompt_embeds=text_embedding(target, ""),
                           num_inference_steps=50, guidance_scale=7.5, height=512, width=512,
                           generator=generator(int(row["seed"]))).images[0]
        name = f"{index:06d}.png"
        image.save(output / name)
        generated.append({**row, "image_path": name, "attack_tokens": " ".join(map(str, tokens))})
        write_csv(output / "images.csv", generated)


if __name__ == "__main__":
    main()
