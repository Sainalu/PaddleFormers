"""Freeze source fingerprints before formal quality and timing jobs."""

import argparse
from pathlib import Path

from common import sha256, source_state, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    patch_path = args.output.with_suffix(".patch")
    if args.output.exists() or patch_path.exists():
        parser.error("fresh source manifest and patch paths required")
    state, patch = source_state()
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    with patch_path.open("xb") as stream:
        stream.write(patch)
    write_json(args.output, {"source": state, "patch": {"path": patch_path.name, "sha256": sha256(patch_path)},
                            "policy": "verify this same source fingerprint before imports and after the run; "
                                      "refuse final results if any executable source changed"})
    print(f"SOURCE_FROZEN scripts={len(state['scripts'])} patch_sha256={state['patch_sha256']}", flush=True)


if __name__ == "__main__":
    main()
