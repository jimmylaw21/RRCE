import argparse
import hashlib
from pathlib import Path

from transformers import CLIPTokenizer

from rrce.common import load_config, read_csv, write_csv
from .protocol import DOG_UD_PROMPTS, VAN_GOGH_UD_PROMPTS


USED_SEEDS = set()


def seeds(namespace, count):
    result = []
    nonce = 0
    while len(result) < count:
        digest = hashlib.sha256(f"aaai2027-eval-v2:seed:{namespace}:{nonce}".encode()).digest()
        value = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        if value not in USED_SEEDS:
            result.append(value)
            USED_SEEDS.add(value)
        nonce += 1
    return result


def rows_for(prompts, namespace, repeats=1):
    values = seeds(namespace, len(prompts) * repeats)
    return [{"prompt": prompt, "seed": values[i * repeats + j]}
            for i, prompt in enumerate(prompts) for j in range(repeats)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--i2p", required=True)
    parser.add_argument("--coco", required=True)
    parser.add_argument("--output", default="data/manifests")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    eligible = []
    all_i2p = read_csv(args.i2p)
    for row in all_i2p:
        score = float(row["nudity_percentage"])
        score = score / 100 if score > 1 else score
        prompt = " ".join(row["prompt"].split())
        if score > 0.5:
            eligible.append(prompt)
    eligible.sort(key=lambda p: hashlib.sha256(f"aaai2027-eval-v2:prompt:{p.casefold()}".encode()).hexdigest())
    erased = [p for p in eligible if len(tokenizer(p, truncation=False).input_ids) < 77][:88]
    attack = eligible[:95]
    if len(erased) != 88 or len(attack) != 95:
        raise ValueError("I2P requires at least 88 short explicit prompts and 95 attack prompts")
    write_csv(output / "nudity_erased.csv", rows_for(erased, "nudity_erased", 5))
    write_csv(output / "nudity_ud.csv", rows_for(attack, "nudity_ud"))
    write_csv(output / "nudity_cce.csv", rows_for([f"<cce> {p}" for p in attack], "nudity:cce_test"))
    excluded = {p.casefold() for p in erased + attack}
    gallery = [" ".join(row["prompt"].split()) for row in all_i2p
               if " ".join(row["prompt"].split()).casefold() not in excluded]
    if not gallery:
        raise ValueError("No disjoint I2P gallery prompts")
    write_csv(output / "nudity_gallery.csv", rows_for(gallery, "nudity:cce_gallery"))
    for task, prompts, count in (("dog", DOG_UD_PROMPTS, 30), ("van_gogh", VAN_GOGH_UD_PROMPTS, 6)):
        config = load_config(task)
        prompt = config["template"].format(config["target"])
        write_csv(output / f"{task}_erased.csv", rows_for([prompt], f"{task}:erased", 500))
        write_csv(output / f"{task}_ud.csv", rows_for(prompts, f"{task}:ud"))
        write_csv(output / f"{task}_cce.csv", rows_for([config["template"].format("<cce>")], f"{task}:cce_test", 500))
        write_csv(output / f"{task}_gallery.csv", rows_for([prompt], f"{task}:cce_gallery", count))
    for task in ("nudity", "dog", "van_gogh"):
        config = load_config(task)
        template = "A painting in the style of {}" if task == "van_gogh" else "A photo of {}"
        retain = rows_for([template.format(label) for label in config["retain"]], f"{task}:retain", 25)
        for index, row in enumerate(retain):
            row["target_label"] = config["retain"][index // 25]
        write_csv(output / f"{task}_retain.csv", retain)
    coco = read_csv(args.coco)
    if len(coco) != 30000 or len({r["case_number"] for r in coco}) != 30000:
        raise ValueError("COCO must contain exactly 30,000 unique cases")
    write_csv(output / "coco.csv", [{"prompt": r["prompt"], "seed": int(r["evaluation_seed"]), "coco_id": r["coco_id"]} for r in coco])


if __name__ == "__main__":
    main()
