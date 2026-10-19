import hashlib
from pathlib import Path

from rrce.io import read_manifest, write_manifest


class SeedAllocator:
    def __init__(self, namespace):
        self.namespace = namespace
        self.used = set()

    def allocate(self, split, count):
        values = []
        nonce = 0
        while len(values) < count:
            material = f"{self.namespace}:seed:{split}:{nonce}"
            digest = hashlib.sha256(material.encode()).digest()
            value = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
            if value not in self.used:
                values.append(value)
                self.used.add(value)
            nonce += 1
        return values


def repeated_rows(prompts, seeds, split, repeats=1):
    if len(seeds) != len(prompts) * repeats:
        raise ValueError("Seed count does not match the prompt expansion")
    return [
        {
            "sample_id": f"{split}_{prompt_index:03d}_{image_index:02d}",
            "prompt": prompt,
            "seed": seeds[prompt_index * repeats + image_index],
        }
        for prompt_index, prompt in enumerate(prompts)
        for image_index in range(repeats)
    ]


def nudity_manifests(config, tokenizer, allocator):
    settings = config.protocol
    raw_rows = read_manifest(config.data.i2p, required=("prompt", "nudity_percentage"))
    all_prompts = [" ".join(row["prompt"].split()) for row in raw_rows]
    eligible = []
    for row, prompt in zip(raw_rows, all_prompts):
        score = float(row["nudity_percentage"])
        if score > 1:
            score /= 100
        if score > settings.explicit_threshold:
            eligible.append(prompt)
    eligible.sort(
        key=lambda prompt: hashlib.sha256(
            f"{settings.seed_namespace}:prompt:{prompt.casefold()}".encode(),
        ).hexdigest()
    )
    short = [
        prompt
        for prompt in eligible
        if len(tokenizer(prompt, truncation=False).input_ids) < tokenizer.model_max_length
    ]
    erased_prompts = short[: settings.nudity_erased_prompts]
    attack_prompts = eligible[: settings.nudity_attack_prompts]
    if (
        len(erased_prompts) != settings.nudity_erased_prompts
        or len(attack_prompts) != settings.nudity_attack_prompts
    ):
        raise ValueError("The I2P input does not cover the configured evaluation protocol")
    erased_count = settings.nudity_erased_prompts * settings.nudity_erased_repeats
    erased = repeated_rows(
        erased_prompts,
        allocator.allocate("nudity_erased", erased_count),
        "nudity_erased",
        settings.nudity_erased_repeats,
    )
    ud = repeated_rows(
        attack_prompts, allocator.allocate("nudity_ud", len(attack_prompts)), "nudity_ud"
    )
    cce_prompts = [f"{config.cce.token} {prompt}" for prompt in attack_prompts]
    cce = repeated_rows(
        cce_prompts, allocator.allocate("nudity:cce_test", len(cce_prompts)), "nudity_cce_test"
    )
    for index, row in enumerate(cce):
        row["sample_id"] = f"nudity_cce_test_{index:03d}"
    excluded = {prompt.casefold() for prompt in erased_prompts + attack_prompts}
    gallery_prompts = [prompt for prompt in all_prompts if prompt.casefold() not in excluded]
    if not gallery_prompts:
        raise ValueError("No disjoint I2P prompts remain for the CCE gallery")
    gallery = repeated_rows(
        gallery_prompts,
        allocator.allocate("nudity:cce_gallery", len(gallery_prompts)),
        "nudity_cce_gallery",
    )
    return {"erased": erased, "ud": ud, "cce": cce, "gallery": gallery}


def concept_manifests(config, allocator):
    task = config.task
    prompt = config.concepts.template.format(config.concepts.target)
    manifests = {}
    for split, prompts, repeats, namespace in (
        ("erased", [prompt], config.evaluation.counts["erased"], f"{task}:erased"),
        ("ud", config.ud_prompts, 1, f"{task}:ud"),
        (
            "cce",
            [config.concepts.template.format(config.cce.token)],
            config.evaluation.counts["cce"],
            f"{task}:cce_test",
        ),
        ("gallery", [prompt], config.cce.gallery_size, f"{task}:cce_gallery"),
    ):
        rows = repeated_rows(
            prompts,
            allocator.allocate(namespace, len(prompts) * repeats),
            f"{task}_{split}",
            repeats,
        )
        identifier = {"cce": "cce_test", "gallery": "cce_gallery"}.get(split, split)
        for index, row in enumerate(rows):
            row["sample_id"] = f"{task}_{identifier}_{index:03d}"
        manifests[split] = rows
    return manifests


def prepare_manifests(config):
    from transformers import CLIPTokenizer

    allocator = SeedAllocator(config.protocol.seed_namespace)
    if config.task == "nudity":
        tokenizer = CLIPTokenizer.from_pretrained(config.model.text_encoder)
        manifests = nudity_manifests(config, tokenizer, allocator)
    else:
        manifests = concept_manifests(config, allocator)
    repeat = config.evaluation.retain_per_concept
    retain_prompts = [
        config.evaluation.retain_template.format(label) for label in config.concepts.retain
    ]
    retain = repeated_rows(
        retain_prompts,
        allocator.allocate(f"{config.task}:retain", len(retain_prompts) * repeat),
        f"{config.task}_retain",
        repeat,
    )
    for index, row in enumerate(retain):
        row["target_label"] = config.concepts.retain[index // repeat]
    manifests["retain"] = retain
    coco_count = config.evaluation.counts["coco"]
    coco = read_manifest(
        config.data.coco,
        required=("case_number", "prompt", "evaluation_seed", "coco_id"),
        expected=coco_count,
    )
    if len({row["case_number"] for row in coco}) != coco_count:
        raise ValueError("COCO case numbers must be unique")
    manifests["coco"] = [
        {
            "sample_id": f"coco_{row['case_number']}",
            "prompt": row["prompt"],
            "seed": int(row["evaluation_seed"]),
            "coco_id": row["coco_id"],
        }
        for row in coco
    ]
    for split, expected in config.evaluation.counts.items():
        if len(manifests[split]) != expected:
            raise ValueError(f"{split} manifest disagrees with evaluation.counts")
    output = Path(config.data.manifests)
    destinations = {split: output / f"{config.task}_{split}.csv" for split in manifests}
    for path in destinations.values():
        if path.exists():
            raise FileExistsError(f"Refusing to replace an existing manifest: {path}")
    for split, rows in manifests.items():
        write_manifest(destinations[split], rows)


def evaluation_manifest(config, split):
    return Path(config.data.manifests) / f"{config.task}_{split}.csv"


def validate_gallery_separation(config, gallery, test):
    if config.cce.gallery_size is not None and len(gallery) != config.cce.gallery_size:
        raise ValueError("CCE gallery size disagrees with the task configuration")
    if {int(row["seed"]) for row in gallery} & {int(row["seed"]) for row in test}:
        raise ValueError("CCE gallery and test seeds overlap")
    if config.task == "nudity":
        test_prompts = {
            row["prompt"].replace(config.cce.token, "").strip().casefold() for row in test
        }
        if {row["prompt"].strip().casefold() for row in gallery} & test_prompts:
            raise ValueError("CCE gallery and test prompts overlap")
