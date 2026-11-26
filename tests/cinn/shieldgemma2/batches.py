"""Validated image-policy batch format shared by preparation and execution."""

import hashlib
import json
from pathlib import Path

import numpy as np


INPUT_NAMES = ("input_ids", "pixel_values", "attention_mask", "position_ids", "token_type_ids")
POLICIES = ("dangerous", "sexual", "violence")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_for_record(record):
    if record.get("policy") not in POLICIES:
        raise ValueError("policy must be dangerous, sexual, or violence")
    if type(record.get("violates")) is not bool:
        raise ValueError("violates must be an explicit JSON boolean, not a guessed label")
    return 0 if record["violates"] else 1


def validate_batch(batch):
    required = set(INPUT_NAMES) | {"labels"}
    if set(batch) != required:
        raise ValueError(f"batch keys must be {sorted(required)}")
    input_ids = batch["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("expected one image-policy pair per batch with nonempty sequence")
    for name in ("input_ids", "attention_mask", "position_ids", "token_type_ids"):
        if batch[name].dtype != np.int64 or batch[name].shape != input_ids.shape:
            raise ValueError(f"{name} must be int64 and match input_ids shape")
    attention_mask = batch["attention_mask"]
    if not np.isin(attention_mask, (0, 1)).all() or attention_mask[0, -1] != 1:
        raise ValueError("attention_mask must be binary with an unpadded last token")
    if np.any(np.diff(attention_mask, axis=1) < 0):
        raise ValueError("classification requires contiguous left padding")
    if np.any(input_ids < 0) or np.any(batch["position_ids"] < 0):
        raise ValueError("token and position indices must be nonnegative")
    if not np.isin(batch["token_type_ids"], (0, 1)).all():
        raise ValueError("token_type_ids must be binary")
    pixels = batch["pixel_values"]
    if pixels.ndim != 4 or pixels.shape[:2] != (1, 3) or min(pixels.shape[2:]) < 1:
        raise ValueError("pixel_values must have shape [1, 3, height, width]")
    if pixels.dtype != np.float32 or not np.isfinite(pixels).all():
        raise ValueError("pixel_values must be finite float32")
    labels = batch["labels"]
    if labels.dtype != np.int64 or labels.shape != (1,) or labels[0] not in (0, 1):
        raise ValueError("labels must be int64 [1], with 0=Yes/violates, 1=No/complies")


def load_batches(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("label_order") != ["Yes/violates", "No/complies"]:
        raise ValueError("unrecognized label mapping in batch manifest")
    if not manifest.get("batches"):
        raise ValueError("manifest must contain at least one batch")
    batches = []
    for entry in manifest["batches"]:
        path = (directory / entry["file"]).resolve()
        if not path.is_relative_to(directory.resolve()) or path.suffix != ".npz":
            raise ValueError("batch path must be an NPZ inside the batch directory")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"batch checksum mismatch: {path.name}")
        with np.load(path, allow_pickle=False) as stored:
            batch = {name: stored[name] for name in stored.files}
        validate_batch(batch)
        batches.append(batch)
    return batches, manifest
