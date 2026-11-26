"""FP32 Aya Vision real-weight eager/CINN training or full multimodal forward."""

import argparse
import csv
import faulthandler
import hashlib
import json
import os
import platform
import sys
import subprocess
import time
from pathlib import Path

from common import BASELINE, ROOT, environment, sha256, source_state, write_json


def hardware_snapshot(device, paddle):
    if device == "cpu":
        return {"device": "cpu"}
    fields = ["name", "uuid", "pci.bus_id", "memory.total", "memory.free", "driver_version",
              "utilization.gpu", "temperature.gpu", "power.draw"]
    process = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=" + ",".join(fields),
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10, check=True)
    values = next(csv.reader([process.stdout.strip()]))
    snapshot = {key: value.strip() for key, value in zip(fields, values)}
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
        function = getattr(paddle.device.cuda, name, None)
        if function is not None:
            snapshot[name + "_bytes"] = int(function())
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batches", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("train", "infer"), required=True)
    parser.add_argument("--backend", choices=("eager", "cinn"), default="eager")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--gpu-scheduled", action="store_true")
    parser.add_argument("--split", choices=("mini", "train", "eval"))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--visual-audit", action="store_true")
    parser.add_argument("--frozen-source", type=Path)
    parser.add_argument("--check-gradients-every-step", action="store_true")
    parser.add_argument("--audit-parameters", action="store_true")
    parser.add_argument("--capture-logits", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.steps < 1 or args.warmup < 0:
        parser.error("new output directory and positive step count required")
    if args.mode == "train" and args.learning_rate <= 0:
        parser.error("training must use a positive learning rate")
    if args.steps >= 100 and args.frozen_source is None:
        parser.error("formal100-step runs require a frozen source manifest")
    if args.device == "gpu" and not args.gpu_scheduled:
        parser.error("parent scheduling required: --gpu-scheduled")
    if args.device == "cpu" and args.backend == "cinn":
        parser.error("CINN GPU run must be scheduled by the parent")
    environment(cpu=args.device == "cpu")
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    initial_source, unused_patch = source_state()
    if args.frozen_source is not None:
        frozen = json.loads(args.frozen_source.read_text())
        if frozen["source"] != initial_source:
            raise ValueError("current executable source differs from the frozen manifest")
        patch_path = args.frozen_source.parent / frozen["patch"]["path"]
        if sha256(patch_path) != frozen["patch"]["sha256"]:
            raise ValueError("frozen source patch hash mismatch")
    if args.device == "gpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        parser.error("parent must explicitly set CUDA_VISIBLE_DEVICES=0 for physical GPU0")
    for flag in ("FLAGS_use_cinn", "FLAGS_prim_all", "FLAGS_prim_enable_dynamic"):
        os.environ[flag] = "true" if args.backend == "cinn" else "false"
    os.environ["FLAGS_enable_cublas_tensor_op_math"] = "false"
    os.environ["FLAGS_enable_cudnn_tensor_op_math"] = "false"
    sys.path.insert(0, os.environ.get("PADDLEFORMERS_DIR", str(ROOT.parents[3])))
    import numpy as np
    import paddle
    from paddle.base import core
    from modeling import graph_class, load_model, loss_class

    paddle.set_device("cpu" if args.device == "cpu" else "gpu:0")
    paddle.set_default_dtype("float32")
    core.set_cublas_switch(False)
    np.random.seed(args.seed)
    paddle.seed(args.seed)
    hardware_start = hardware_snapshot(args.device, paddle)
    args.output.mkdir(parents=True)
    write_json(args.output / "source_start.json", initial_source)
    started = time.monotonic()
    model, loading = load_model(args.checkpoint)
    write_json(args.output / "loading.json", loading)
    training = args.mode == "train"
    model.train() if training else model.eval()
    graph = loss_class()(model) if training else graph_class()(model)
    if args.backend == "cinn":
        graph = paddle.jit.to_static(graph, full_graph=True)
    manifest_path = args.batches / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    split = args.split or ("train" if training else "eval")
    names = ("input_ids", "pixel_values", "attention_mask", "position_ids") + (("labels",) if training else ())
    inputs = []
    files = []
    for entry in manifest["batches"]:
        if entry["split"] != split:
            continue
        path = args.batches / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("fixed batch hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in names}
        image_token_count = int((arrays["input_ids"] == model.config.image_token_index).sum())
        if image_token_count != entry["projected_feature_count"] or image_token_count <= 0:
            raise ValueError("fixed batch does not execute the actual image path")
        inputs.append(tuple(paddle.to_tensor(arrays[name]) for name in names))
        files.append({"path": os.path.relpath(path, ROOT), "sha256": entry["sha256"]})
    if not inputs:
        raise ValueError("requested fixed split is empty")
    optimizer = paddle.optimizer.AdamW(learning_rate=args.learning_rate, parameters=model.parameters(),
                                      weight_decay=0.0, beta1=0.9, beta2=0.999,
                                      epsilon=1e-8) if training else None
    watched = {
        "vision": model.model.vision_tower.embeddings.patch_embedding.weight,
        "projector": model.model.multi_modal_projector.linear_1.weight,
        "text_sliding": model.model.language_model.layers[0].self_attn.q_proj.weight,
        "text_full": model.model.language_model.layers[3].self_attn.q_proj.weight}
    before = {name: parameter.numpy().copy() for name, parameter in watched.items()} if training else {}
    gradient_audit = {}
    gradient_check_steps = []
    gradient_dtypes = set()
    absent_gradient_names = []
    calls = {}
    handles = []
    if args.visual_audit and args.backend == "eager":
        layers = {"vision_embeddings": model.model.vision_tower.embeddings,
                  "vision_layer0": model.model.vision_tower.encoder.layers[0],
                  "vision_layer1": model.model.vision_tower.encoder.layers[1],
                  "projector": model.model.multi_modal_projector,
                  "text_sliding": model.model.language_model.layers[0],
                  "text_full": model.model.language_model.layers[3]}
        for name, layer in layers.items():
            def record_call(layer, layer_inputs, output, key=name):
                calls[key] = calls.get(key, 0) + 1
            handles.append(layer.register_forward_post_hook(record_call))
    def synchronize():
        if args.device == "gpu":
            paddle.device.cuda.synchronize()

    timings = []
    losses = []
    outputs = {}
    first_s = None
    total = args.steps + args.warmup
    with paddle.enable_grad() if training else paddle.no_grad():
        with (args.output / "steps.jsonl").open("x") as stream:
            for step in range(total):
                batch_index = step % len(inputs)
                synchronize()
                tick = time.monotonic()
                result = graph(*inputs[batch_index])
                if training:
                    result.backward()
                    if step == 0:
                        for name, parameter in watched.items():
                            gradient = parameter.grad
                            gradient_audit[name] = {
                                "present": gradient is not None,
                                "finite": gradient is not None and bool(paddle.isfinite(gradient).all()),
                                "abs_sum": float(paddle.abs(gradient).sum()) if gradient is not None else 0.0}
                        if not all(entry["present"] and entry["finite"] and entry["abs_sum"] > 0
                                   for entry in gradient_audit.values()):
                            raise ValueError(f"real visual/text path gradient missing: {gradient_audit}")
                    if step == 0 or args.check_gradients_every_step:
                        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                        if not gradients or not all(bool(paddle.isfinite(gradient).all()) for gradient in gradients):
                            raise ValueError("nonfinite model gradient")
                        gradient_dtypes.update(str(gradient.dtype) for gradient in gradients)
                        if gradient_dtypes != {"paddle.float32"}:
                            raise ValueError("FP32 gradients required")
                        gradient_check_steps.append(step)
                        if step == 0:
                            absent_gradient_names = [name for name, parameter in model.named_parameters()
                                                     if parameter.grad is None]
                    optimizer.step()
                    optimizer.clear_grad()
                synchronize()
                elapsed = time.monotonic() - tick
                if first_s is None:
                    first_s = elapsed
                if step >= args.warmup:
                    timings.append(elapsed)
                if training:
                    loss = float(result)
                    if not np.isfinite(loss):
                        raise ValueError("nonfinite training loss")
                    losses.append(loss)
                else:
                    logits, image_features = result
                    if not bool(paddle.isfinite(logits).all()) or not bool(paddle.isfinite(image_features).all()):
                        raise ValueError("nonfinite multimodal outputs")
                    predictions = paddle.argmax(logits, axis=-1).numpy()
                    if args.capture_logits and step >= args.warmup and batch_index not in outputs:
                        output_path = args.output / f"batch_{batch_index}.npz"
                        array_logits = logits.numpy()
                        np.savez(output_path, logits=array_logits, image_features=image_features.numpy(),
                                 predictions=array_logits.argmax(-1))
                        outputs[batch_index] = {"path": output_path.name, "sha256": sha256(output_path),
                                                "logits_shape": list(logits.shape),
                                                "image_features_shape": list(image_features.shape)}
                entry = {"step": step, "batch": batch_index, "warmup": step < args.warmup,
                         "seconds": elapsed, "loss": losses[-1] if training else None}
                if not training:
                    entry["inference_quality"] = {"logits_finite": True, "image_features_finite": True,
                                                  "predictions": predictions.tolist(),
                                                  "predictions_sha256": hashlib.sha256(predictions).hexdigest()}
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
                stream.flush()
                print("STEP " + json.dumps(entry), flush=True)
    for handle in handles:
        handle.remove()
    changes = {}
    final_parameters = {}
    if training:
        for name, parameter in watched.items():
            changes[name] = float(np.max(np.abs(parameter.numpy() - before[name])))
        if not all(change > 0 for change in changes.values()):
            raise ValueError(f"positive learning rate did not update all three model paths: {changes}")
        if args.audit_parameters:
            initial_hashes = {entry["target"]: entry["loaded_fp32_sha256"] for entry in loading["tensor_mapping"]}
            for name, parameter in model.named_parameters():
                array = parameter.numpy()
                if not np.isfinite(array).all():
                    raise ValueError(f"nonfinite final parameter: {name}")
                digest = hashlib.sha256(array).hexdigest()
                final_parameters[name] = {"sha256": digest, "changed": digest != initial_hashes[name],
                                          "finite": True, "dtype": str(array.dtype)}
    if args.visual_audit and args.backend == "eager" and len(calls) != 6:
        raise ValueError(f"incomplete real visual/text calls: {calls}")
    optimizer_dtype_report = {}
    if training:
        state = optimizer.state_dict()
        optimizer_dtype_report = {name: str(value.dtype) for name, value in state.items()
                                  if isinstance(value, paddle.Tensor)}
        float_state_dtypes = {dtype for dtype in optimizer_dtype_report.values()
                              if "float" in dtype or "bfloat" in dtype}
        if float_state_dtypes != {"paddle.float32"}:
            raise ValueError(f"AdamW floating states must be FP32: {float_state_dtypes}")
    final_source, unused_patch = source_state()
    if final_source != initial_source:
        raise ValueError("executable source changed during the run; formal result refused")
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if parameter_dtypes != ["paddle.float32"]:
        raise ValueError("parameter dtype changed during execution")
    report = {"mode": args.mode, "backend": args.backend, "device": args.device, "dtype": "float32",
              "amp": False, "tf32": False, "steps": args.steps, "warmup": args.warmup,
              "learning_rate": args.learning_rate if training else None, "seed": args.seed,
              "first_call_s": first_s, "steady_mean_s": float(np.mean(timings)),
              "steady_median_s": float(np.median(timings)), "step_seconds": timings,
              "losses": losses, "watched_gradients": gradient_audit, "watched_parameter_changes": changes,
              "gradient_audit_scope": {"checked_steps": gradient_check_steps,
                                       "scope": "all present parameter gradients on listed steps; four path witnesses on first step",
                                       "first_step_absent_gradient_parameters": absent_gradient_names,
                                       "gradient_dtypes": sorted(gradient_dtypes)},
              "dtype_audit": {"parameters": parameter_dtypes, "adamw_tensor_states": optimizer_dtype_report},
              "all_step_gradients_checked": training and args.check_gradients_every_step,
              "final_parameters": final_parameters, "source_unchanged_during_run": True,
              "source_start": {"path": "source_start.json", "sha256": sha256(args.output / "source_start.json")},
              "frozen_source": ({"path": os.path.relpath(args.frozen_source, ROOT), "sha256": sha256(args.frozen_source)}
                                if args.frozen_source is not None else None),
              "workload": {"batch_size": inputs[0][0].shape[0], "sequence_length": inputs[0][0].shape[1],
                           "pixel_values_shape": list(inputs[0][1].shape), "image_tokens_per_sample": image_token_count,
                           "vocab_size": model.config.text_config.vocab_size,
                           "text_layers": model.config.text_config.num_hidden_layers,
                           "vision_layers": model.config.vision_config.num_hidden_layers,
                           "attention_implementation": "eager", "cache": False},
              "visual_calls": calls, "outputs": outputs,
              "capture_logits": args.capture_logits,
              "loading_report": {"path": "loading.json", "sha256": sha256(args.output / "loading.json")},
              "step_report": {"path": "steps.jsonl", "sha256": sha256(args.output / "steps.jsonl")},
              "hyperparameters": {"seed": args.seed, "steps": args.steps, "warmup": args.warmup,
                                  "optimizer": "AdamW" if training else None,
                                  "learning_rate": args.learning_rate if training else None,
                                  "weight_decay": 0.0 if training else None,
                                  "beta1": 0.9 if training else None, "beta2": 0.999 if training else None,
                                  "epsilon": 1e-8 if training else None,
                                  "loss": "sum cross entropy / valid shifted assistant labels" if training else None},
              "timing_scope": "resident fixed inputs; synchronize before/after graph; training includes forward, backward, "
                              "AdamW step and clear_grad; finite loss/output checks, argmax, output capture and JSON outside; "
                              "enabled gradient checks occur inside training timing, first step lies in benchmark warmup",
              "hardware_start": hardware_start, "hardware_end": hardware_snapshot(args.device, paddle),
              "inputs": {"checkpoint_manifest": {"path": os.path.relpath(args.checkpoint / "checkpoint_manifest.json", ROOT),
                                                     "sha256": loading["checkpoint_manifest_sha256"]},
                         "batch_manifest": {"path": os.path.relpath(manifest_path, ROOT), "sha256": sha256(manifest_path)},
                         "batches": files},
              "process_work_s": time.monotonic() - started,
              "environment": {"python": platform.python_version(), "paddle": paddle.__version__,
                              "paddle_commit": paddle.version.commit, "cuda": paddle.version.cuda(),
                              "cudnn": paddle.version.cudnn(), "paddleformers_baseline": BASELINE,
                              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
              "scope": "original prefix-reduced Aya Vision, full image/text forward without KV cache; "
                       "not full original-model task quality or token decode performance"}
    write_json(args.output / "result.json", report)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
