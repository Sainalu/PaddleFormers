"""Run the real reduced ShieldGemma 2 with an explicitly verified source tree."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


PROJECT = Path(os.environ.get("PADDLEFORMERS_PROJECT", Path(__file__).resolve().parents[3])).resolve()
MODEL = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL))
from batches import INPUT_NAMES, sha256


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def source_state(source):
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(source), *arguments], timeout=30)
    commit = git("rev-parse", "HEAD").decode().strip()
    difference = git("diff", "--binary", "HEAD")
    lock_path = Path(__file__).with_name("source-lock.json")
    lock = json.loads(lock_path.read_text())
    patch = MODEL / "paddleformers-shieldgemma2-cinn.patch"
    expected = set(lock["changed_files"])
    changed = git("diff", "--name-only", "HEAD").decode().splitlines()
    runner_root = MODEL.parent.resolve()
    untracked = git("ls-files", "--others", "--exclude-standard").decode().splitlines()
    outside_runner = []
    for name in untracked:
        path = (source / name).resolve()
        if path.suffix in (".py", ".so") and not path.is_relative_to(runner_root):
            outside_runner.append(name)
    if (commit != lock["commit"] or set(changed) != expected or sha256(patch) != lock["patch_sha256"]
            or git("diff", "--cached") or outside_runner):
        raise ValueError("source tree differs from the declared Shield patch")
    branch = git("branch", "--show-current").decode().strip()
    return {"source_commit": commit, "branch": branch, "detached": not branch,
            "source_diff_sha256": hashlib.sha256(difference).hexdigest(),
            "changed_files": changed, "patch_sha256": sha256(patch), "source_lock_sha256": sha256(lock_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=os.environ.get("PADDLEFORMERS_DIR", PROJECT / ".worktrees/shieldgemma2-latest"))
    parser.add_argument("--checkpoint", type=Path, default=MODEL / "checkpoints/shieldgemma2-2text-2vision-bf16")
    parser.add_argument("--batches", type=Path, required=True)
    parser.add_argument("--task", choices=("train", "infer"), required=True)
    parser.add_argument("--mode", choices=("dynamic", "cinn"), required=True)
    parser.add_argument("--device", choices=("cpu", "gpu:0"), default="gpu:0")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--verify-gradients", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 0 <= args.warmup < args.steps:
        parser.error("fresh output and 0 <= warmup < steps required")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("a finite positive learning rate is required")
    if args.device == "gpu:0" and os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        parser.error("GPU runs require physical GPU0 via the shared scheduler")
    if args.device == "cpu" and (args.mode != "dynamic" or os.environ.get("CUDA_VISIBLE_DEVICES") != ""):
        parser.error("CPU runs require dynamic mode and CUDA_VISIBLE_DEVICES=''")
    source = args.source.resolve()
    before_source = source_state(source)
    for flag in ("FLAGS_prim_enable_dynamic", "FLAGS_prim_all", "FLAGS_use_cinn"):
        os.environ[flag] = "true" if args.mode == "cinn" else "false"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    sys.path.insert(0, str(source))
    import numpy as np
    import paddle
    import paddleformers
    from paddle.base import core
    from paddleformers.transformers import ShieldGemma2Config, ShieldGemma2ForImageClassification
    from paddleformers.transformers.model_utils import dtype_guard
    from classification_graph import ShieldGemma2InferGraph, ShieldGemma2TrainGraph
    from run_classification import checkpoint_assets, parameter_samples

    if Path(paddleformers.__file__).resolve().parent != source / "paddleformers":
        raise ValueError("imported PaddleFormers is not the verified source")
    print("VERIFIED_SOURCE " + json.dumps(before_source), flush=True)
    started = time.perf_counter()
    paddle.set_device(args.device)
    paddle.seed(args.seed)
    np.random.seed(args.seed)
    core.set_cublas_switch(False)
    core.set_cudnn_switch(False)
    checkpoint = args.checkpoint.resolve()
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
    if manifest["retained_layers"] != {"text": 2, "vision": 2} or manifest["tensor_count"] != 67:
        raise ValueError("the original text2/vision2 checkpoint is required")
    for name, entry in manifest["files"].items():
        path = checkpoint / name
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"checkpoint file differs: {name}")
    with dtype_guard("float32"):
        model, batches, assets = checkpoint_assets(paddle, ShieldGemma2Config, ShieldGemma2ForImageClassification,
                                                   checkpoint, args.batches)
    model.to(device=args.device)
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError("the original tied embedding/head must be retained")
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if parameter_dtypes != ["paddle.float32"]:
        raise ValueError("FP32 parameters required")
    config = model.config
    assets.update(checkpoint=os.path.relpath(checkpoint, PROJECT), strict_loading=True, compute_dtype="float32",
                  tf32=False, checkpoint_manifest_sha256=sha256(checkpoint / "checkpoint_manifest.json"),
                  hidden_size=config.text_config.hidden_size, vocab_size=config.text_config.vocab_size,
                  image_size=config.vision_config.image_size, image_tokens=config.mm_tokens_per_image,
                  sequence_length=batches[0]["input_ids"].shape[1], source_text_count=len(batches),
                  head_embedding_shared=True, safety_accuracy_claim=False)
    buffers = {name: {"dtype": str(buffer.dtype), "place": str(buffer.place)} for name, buffer in model.named_buffers()}
    if args.device == "gpu:0" and any("gpu:0" not in entry["place"] for entry in buffers.values()):
        raise ValueError("all model buffers must be on GPU0")
    training = args.task == "train"
    model.train() if training else model.eval()
    graph = ShieldGemma2TrainGraph(model) if training else ShieldGemma2InferGraph(model)
    if args.mode == "cinn":
        graph = paddle.jit.to_static(graph, full_graph=True)
    optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=args.learning_rate,
                                       weight_decay=0.0, multi_precision=False) if training else None
    names = INPUT_NAMES + (("labels",) if training else ())
    inputs = [tuple(paddle.to_tensor(batch[name]) for name in names) for batch in batches]
    initial_samples = parameter_samples(model.named_parameters()) if training else {}
    prefixes = {"vision": "model.vision_tower.", "projector": "model.multi_modal_projector.",
                "text": "model.language_model.layers.0."}
    witnesses, checked_steps, gradient_dtypes, absent = {}, [], set(), []
    gradient_count = 0
    records = []
    loop_start = time.perf_counter()
    with paddle.enable_grad() if training else paddle.no_grad():
        for step in range(args.steps):
            if args.device == "gpu:0":
                paddle.device.synchronize()
            step_start = time.perf_counter()
            outputs = graph(*inputs[step % len(inputs)])
            if training:
                outputs.backward()
                if args.verify_gradients or step == 0:
                    gradients = {name: parameter.grad for name, parameter in model.named_parameters()
                                 if parameter.grad is not None}
                    gradient_count = len(gradients)
                    if not gradients or not all(bool(paddle.isfinite(value).all()) for value in gradients.values()):
                        raise ValueError(f"missing or nonfinite gradients at step {step}")
                    gradient_dtypes.update(str(value.dtype) for value in gradients.values())
                    if gradient_dtypes != {"paddle.float32"}:
                        raise ValueError("gradients must remain FP32")
                    checked_steps.append(step)
                    if step == 0:
                        absent = [name for name, parameter in model.named_parameters() if parameter.grad is None]
                        for label, prefix in prefixes.items():
                            candidates = [(name, float(paddle.abs(gradient).sum())) for name, gradient in gradients.items()
                                          if name.startswith(prefix)]
                            selected = next((entry for entry in candidates if entry[1] > 0), None)
                            if selected is None:
                                raise ValueError(f"no real nonzero gradient through {label}")
                            witnesses[label] = {"parameter": selected[0], "first_gradient_abs_sum": selected[1]}
                optimizer.step()
                optimizer.clear_grad()
            if args.device == "gpu:0":
                paddle.device.synchronize()
            elapsed = time.perf_counter() - step_start
            record = {"step": step, "batch": step % len(inputs), "step_time_s": elapsed}
            if training:
                loss = float(outputs)
                if not math.isfinite(loss) or loss < 0:
                    raise ValueError("invalid training loss")
                record["loss"] = loss
                print(f"STEP_LOSS step={step} batch={record['batch']} loss={loss:.10f} step_time_s={elapsed:.8f}", flush=True)
            else:
                logits, probabilities = (output.numpy() for output in outputs)
                if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
                    raise ValueError("nonfinite classification output")
                record.update(logits=logits.tolist(), probabilities=probabilities.tolist(),
                              predictions=logits.argmax(-1).tolist())
                print("STEP_INFER " + json.dumps(record), flush=True)
            records.append(record)
    loop_wall = time.perf_counter() - loop_start
    final_audit, changes, states = {}, {}, []
    if training:
        samples = parameter_samples(model.named_parameters())
        for name, parameter in model.named_parameters():
            finite = bool(paddle.isfinite(parameter).all())
            if not finite:
                raise ValueError(f"nonfinite final parameter: {name}")
            changed = not np.array_equal(initial_samples[name], samples[name])
            final_audit[name] = {"finite": finite, "dtype": str(parameter.dtype), "sample_changed": changed}
        for label, witness in witnesses.items():
            name = witness["parameter"]
            changes[label] = float(np.max(np.abs(samples[name] - initial_samples[name])))
            if changes[label] <= 0:
                raise ValueError(f"positive LR did not update {label} witness")
        states = sorted({str(value.dtype) for value in optimizer.state_dict().values()
                         if isinstance(value, paddle.Tensor)})
        if states != ["paddle.float32"]:
            raise ValueError("optimizer tensor states must remain FP32")
    after_source = source_state(source)
    if before_source != after_source:
        raise ValueError("source changed during execution")
    environment = {"python": platform.python_version(), "paddle": paddle.__version__, "paddle_commit": paddle.version.commit,
                   "cuda": paddle.version.cuda(), "cudnn": paddle.version.cudnn(),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), **before_source,
                   "runner_sha256": sha256(Path(__file__)), "graph_sha256": sha256(MODEL / "scripts/classification_graph.py"),
                   "loader_sha256": sha256(MODEL / "scripts/run_classification.py")}
    steady = records[args.warmup:]
    mean = sum(row["step_time_s"] for row in steady) / len(steady)
    report = {"task": args.task, "mode": args.mode, "device": args.device, "dtype": "float32", "loss_dtype": "float32",
              "amp": False, "tf32": False, "assets": assets, "environment": environment, "steps": args.steps,
              "warmup_steps": args.warmup, "seed": args.seed, "learning_rate": args.learning_rate if training else None,
              "optimizer": "AdamW" if training else None, "weight_decay": 0.0, "world_size": 1,
              "compiled_boundary": "full image-policy classifier and binary CE; optimizer outside static graph",
              "gradient_checks_enabled": training and args.verify_gradients, "gradient_count": gradient_count,
              "gradient_dtypes": sorted(gradient_dtypes), "gradient_checked_steps": checked_steps,
              "first_step_absent_gradient_parameters": absent, "parameter_dtypes": parameter_dtypes,
              "optimizer_state_dtypes": states, "parameters_changed": bool(changes) if training else None,
              "updated_parameters_finite": all(entry["finite"] for entry in final_audit.values()) if training else None,
              "final_parameters": final_audit, "parameter_check_scope": "all finite; first64 per parameter for updates",
              "path_witnesses": witnesses, "watched_parameter_changes": changes, "buffer_audit": buffers,
              "code_verified_unchanged": True, "first_step_s": records[0]["step_time_s"],
              "first_step_s_includes_trace_compile": records[0]["step_time_s"], "steady_mean_step_s": mean,
              "steady_pairs_per_s": 1 / mean, "measured_step_sum_s": sum(row["step_time_s"] for row in records),
              "loop_wall_s": loop_wall, "process_work_s": time.perf_counter() - started,
              "gpu_peak_allocated_bytes": paddle.device.cuda.max_memory_allocated() if args.device == "gpu:0" else None,
              "gpu_peak_reserved_bytes": paddle.device.cuda.max_memory_reserved() if args.device == "gpu:0" else None,
              "performance_claim": False, "safety_accuracy_claim": False, "records": records}
    write_json(args.output, report)
    print("SUMMARY " + json.dumps({key: report[key] for key in ("task", "mode", "steps", "learning_rate", "parameters_changed", "process_work_s")}), flush=True)


if __name__ == "__main__":
    main()
