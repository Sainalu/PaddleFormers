import ast
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent / "granite32"


def test_shell_entries_parse():
    for path in ROOT.glob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_python_entry_compiles():
    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))


def test_help_does_not_import_gpu_runtime():
    subprocess.run([sys.executable, str(ROOT / "run_model.py"), "--help"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_existing_output_is_rejected():
    output = ROOT / "asset_manifest.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "run_model.py"), "--mode", "dynamic", "--task", "infer",
         "--checkpoint", str(ROOT), "--batches", str(ROOT), "--output", str(output), "--device", "cpu"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    assert result.returncode != 0


def test_manifest_is_model_local():
    manifest = json.loads((ROOT / "asset_manifest.json").read_text())
    assert manifest["model"] == "granite32"
