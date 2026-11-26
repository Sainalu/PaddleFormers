import hashlib
import json
import os
import subprocess
import stat
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE = "1d7c2679f24d9d245422afa4d59014cdd9689fd6"


def shared_tools_root():
    for ancestor in (Path(os.environ.get("CINN_ROOT", "")), ROOT, *ROOT.parents):
        candidate = ancestor / "shared_tools"
        if (candidate / "gpu0.py").is_file():
            return candidate
    raise ValueError("shared GPU0 scheduling tool is missing")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def environment(cpu=True):
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "8"


def source_state():
    repository = Path(os.environ.get("PADDLEFORMERS_DIR", ROOT.parents[3]))
    def git(*arguments):
        return subprocess.run(["git", "-C", str(repository), *arguments], capture_output=True,
                              check=True, timeout=15).stdout

    commit = git("rev-parse", "HEAD").decode().strip()
    branch = git("branch", "--show-current").decode().strip()
    if commit != BASELINE:
        raise ValueError("official source must remain at the fixed baseline commit")
    untracked = git("ls-files", "--others", "--exclude-standard").decode().splitlines()
    runner_root = ROOT.parent.resolve()
    untracked_executable = []
    for name in untracked:
        path = (repository / name).resolve()
        if path.suffix in (".py", ".so") and not path.is_relative_to(runner_root):
            untracked_executable.append(name)
    if untracked_executable:
        raise ValueError("untracked executable model source is not covered by the frozen Git patch")
    patch = git("diff", "--binary", "HEAD")
    scripts = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(ROOT.iterdir())
               if path.is_file() and path.suffix in (".py", ".sh")}
    modes = {name: oct(stat.S_IMODE((ROOT / name).stat().st_mode)) for name in scripts}
    return {"baseline": commit, "branch": branch, "detached": not branch,
            "patch_sha256": hashlib.sha256(patch).hexdigest(),
            "scripts": scripts, "script_modes": modes}, patch
