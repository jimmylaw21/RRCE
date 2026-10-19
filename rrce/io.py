import csv
import hashlib
import json
import os
from pathlib import Path


def read_manifest(path, required=("prompt", "seed"), expected=None):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns {required}: {path}")
        rows = list(reader)
    if not rows or (expected is not None and len(rows) != expected):
        raise ValueError(f"Unexpected manifest size {len(rows)}; expected {expected}: {path}")
    if "sample_id" in rows[0] and len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate sample identifiers: {path}")
    return rows


def write_manifest(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty manifest")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    os.replace(temporary, path)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_images(manifest, rows):
    root = Path(manifest).parent.resolve()
    paths = [(root / row["image_path"]).resolve() for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Image paths are duplicated")
    for path in paths:
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing image or path outside manifest directory: {path.name}")
    return paths


class MetricLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **metrics):
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(metrics, allow_nan=False) + "\n")
