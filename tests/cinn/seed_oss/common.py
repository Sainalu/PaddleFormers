import hashlib
import json
from pathlib import Path


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError(f"refusing to overwrite result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def checked_file(path, expected_hash=None):
    path = Path(path).resolve()
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"missing or empty asset: {path}")
    if expected_hash and sha256(path) != expected_hash:
        raise ValueError(f"asset hash mismatch: {path}")
    return path


def load_batches(directory, checkpoint):
    directory = Path(directory).resolve()
    manifest = json.loads(checked_file(directory / "manifest.json").read_text())
    if manifest.get("checkpoint_config_sha256") != sha256(Path(checkpoint) / "config.json"):
        raise ValueError("batch and checkpoint configs differ")
    batches = []
    for entry in manifest["batches"]:
        path = checked_file(directory / entry["file"], entry["sha256"])
        import numpy as np
        with np.load(path, allow_pickle=False) as archive:
            batch = {name: archive[name] for name in archive.files}
        shape = (1, manifest["sequence_length"])
        if set(batch) != {"input_ids", "labels", "attention_mask", "position_ids"}:
            raise ValueError("unexpected batch keys")
        if any(value.dtype != "int64" or value.shape != shape for value in batch.values()):
            raise ValueError("unexpected batch shape or dtype")
        if not (batch["attention_mask"] == 1).all():
            raise ValueError("only unpadded fixed inputs are accepted")
        batches.append(batch)
    if not batches:
        raise ValueError("empty batch directory")
    return batches, manifest
