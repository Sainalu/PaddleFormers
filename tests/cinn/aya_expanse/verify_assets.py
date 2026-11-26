import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def check(path, entry):
    if (not path.is_file() or ("bytes" in entry and path.stat().st_size != entry["bytes"])
            or digest(path) != entry["sha256"]):
        raise ValueError(f"asset mismatch: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("resource", type=Path)
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).parent / "asset_manifest.json").read_text())
    if manifest["model"] != args.model:
        raise ValueError(f"unexpected asset manifest model: {manifest['model']}")
    roots = {"checkpoint": args.checkpoint, "resource": args.resource}
    for entry in manifest["files"]:
        path = roots[entry["root"]] / entry["path"]
        check(path, entry)
        if entry.get("manifest"):
            data = json.loads(path.read_text())
            for name, nested in data.get("files", {}).items():
                if isinstance(nested, dict) and "bytes" in nested and "sha256" in nested:
                    check(path.parent / name, nested)
            for batch in data.get("batches", []):
                check(path.parent / batch["file"], batch)
            source = data.get("source")
            if isinstance(source, dict) and source.get("file") and source.get("bytes"):
                source_path = path.parent.parent / source["file"]
                if source_path.is_file():
                    check(source_path, source)
    print(f"ASSETS_OK model={args.model} files={len(manifest['files'])}")


if __name__ == "__main__":
    main()
