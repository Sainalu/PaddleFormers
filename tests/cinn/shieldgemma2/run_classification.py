"""Run checkpoint-backed classification, or an explicitly synthetic diagnostic."""

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from batches import INPUT_NAMES, load_batches, sha256


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--diagnostic-random", action="store_true")
    parser.add_argument("--batches", type=Path)
    parser.add_argument("--task", choices=("train", "infer"), required=True)
    parser.add_argument("--mode", choices=("dynamic", "cinn"), default="dynamic")
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--verify-gradients", action="store_true")
    parser.add_argument("--save-checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if not 0 <= arguments.warmup_steps < arguments.steps:
        parser.error("require 0 <= warmup-steps < steps")
    if arguments.learning_rate < 0 or not math.isfinite(arguments.learning_rate):
        parser.error("learning-rate must be finite and nonnegative")
    if arguments.checkpoint and not arguments.batches:
        parser.error("checkpoint runs require --batches")
    if arguments.diagnostic_random and arguments.batches:
        parser.error("synthetic diagnostics cannot consume real batch assets")
    if arguments.output.exists():
        parser.error("output already exists; choose a new path to preserve prior evidence")
    if arguments.save_checkpoint and (arguments.task != "train" or not arguments.checkpoint):
        parser.error("saving a checkpoint requires checkpoint-backed training")
    if arguments.save_checkpoint and arguments.save_checkpoint.exists():
        parser.error("saved checkpoint directory already exists")
    if arguments.mode == "cinn" and not arguments.device.startswith("gpu"):
        parser.error("CINN verification requires a GPU; CPU runs are dynamic diagnostics only")
    if arguments.device == "cpu" and arguments.dtype != "float32":
        parser.error("CPU diagnostic supports float32 only")
    return arguments


def git_value(*arguments):
    repository = Path(os.environ.get("PADDLEFORMERS_DIR", Path(__file__).resolve().parents[3]))
    return subprocess.check_output(["git", "-C", str(repository), *arguments], text=True).strip()


def parameters_changed(parameters, snapshots):
    return any(not np.array_equal(value.numpy(), snapshots[name].numpy()) for name, value in parameters)


def parameter_samples(parameters):
    return {name: value.reshape([-1])[:64].numpy().copy() for name, value in parameters if not value.stop_gradient}


def save_trained_checkpoint(paddle, model, source, output, assets, arguments):
    output.mkdir(parents=True, exist_ok=False)
    paddle.save(model.state_dict(), str(output / "model_state.pdparams"))
    names = ("config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
             "special_tokens_map.json", "processor_config.json", "preprocessor_config.json",
             "chat_template.jinja", "chat_template.json", "added_tokens.json", "generation_config.json")
    for name in names:
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    manifest = {
        "scope": "trained reduced checkpoint for structure regression; not safety quality acceptance",
        "source_assets": assets, "steps": arguments.steps, "learning_rate": arguments.learning_rate,
        "dtype": arguments.dtype, "mode": arguments.mode, "seed": arguments.seed,
        "optimizer_state_saved": False, "exact_training_resume_supported": False,
        "model_state_sha256": sha256(output / "model_state.pdparams"),
    }
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"path": str(output.resolve()), **manifest}


def diagnostic_assets(paddle, config_class, model_class):
    config = config_class(
        text_config={
            "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "max_position_embeddings": 64, "rope_theta": 10000.0,
            "query_pre_attn_scalar": 8, "sliding_window": 16,
        },
        vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                       "num_attention_heads": 4, "image_size": 16, "patch_size": 4},
        image_token_index=5, mm_tokens_per_image=4, yes_token_index=3, no_token_index=7,
    )
    model = model_class(config)
    input_ids = np.asarray([[2, 5, 5, 5, 5, 8, 9, 10]], dtype=np.int64)
    batches = []
    for label in (0, 1):
        batches.append({
            "input_ids": input_ids.copy(),
            "pixel_values": np.random.standard_normal((1, 3, 16, 16)).astype(np.float32),
            "attention_mask": np.ones_like(input_ids),
            "position_ids": np.arange(8, dtype=np.int64)[None, :],
            "token_type_ids": (input_ids == 5).astype(np.int64),
            "labels": np.asarray([label], dtype=np.int64),
        })
    return model, batches, {"synthetic": True}


def checkpoint_assets(paddle, config_class, model_class, checkpoint, batches_directory):
    checkpoint = checkpoint.resolve()
    config = config_class.from_pretrained(str(checkpoint))
    weight_path = checkpoint / "model_state.pdparams"
    if weight_path.is_file():
        model = model_class(config)
        state = paddle.load(str(weight_path))
        expected = model.state_dict()
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        incompatible = [name for name in set(state) & set(expected)
                        if list(state[name].shape) != list(expected[name].shape)]
        if missing or unexpected or incompatible:
            raise ValueError(f"strict checkpoint mismatch: missing={missing}, unexpected={unexpected}, shapes={incompatible}")
        state = {name: value.astype(expected[name].dtype) for name, value in state.items()}
        model.set_state_dict(state)
        weight_hashes = {weight_path.name: sha256(weight_path)}
        checkpoint_hash = weight_hashes[weight_path.name]
    else:
        index_path = checkpoint / "model.safetensors.index.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            weight_names = sorted(set(index["weight_map"].values()))
        else:
            weight_names = ["model.safetensors"]
        weight_hashes = {}
        for name in weight_names:
            path = (checkpoint / name).resolve()
            if not path.is_relative_to(checkpoint):
                raise ValueError("weight shard must remain inside checkpoint directory")
            weight_hashes[name] = sha256(path)
        checkpoint_hash = hashlib.sha256(json.dumps(weight_hashes, sort_keys=True).encode()).hexdigest()
        model = model_class.from_pretrained(str(checkpoint), config=config, dtype=paddle.get_default_dtype())
    batches, manifest = load_batches(batches_directory)
    if manifest.get("checkpoint_config_sha256") != sha256(checkpoint / "config.json"):
        raise ValueError("batch processor checkpoint config does not match model checkpoint")
    for batch in batches:
        if batch["input_ids"].max() >= config.text_config.vocab_size:
            raise ValueError("input token outside checkpoint vocabulary")
        if (batch["input_ids"] == config.image_token_index).sum() != config.mm_tokens_per_image:
            raise ValueError("image token count does not match checkpoint")
    return model, batches, {"synthetic": False, "checkpoint": str(checkpoint),
                            "checkpoint_sha256": checkpoint_hash, "checkpoint_files_sha256": weight_hashes,
                            "structure_only": manifest.get("structure_only", False),
                            "text_layers": config.text_config.num_hidden_layers,
                            "vision_layers": config.vision_config.num_hidden_layers,
                            "batch_manifest_sha256": sha256(Path(batches_directory) / "manifest.json")}


def main():
    arguments = parse_arguments()
    use_cinn = arguments.mode == "cinn"
    for flag in ("FLAGS_prim_enable_dynamic", "FLAGS_prim_all", "FLAGS_use_cinn"):
        os.environ[flag] = "true" if use_cinn else "false"

    import paddle
    from paddleformers.transformers import ShieldGemma2Config, ShieldGemma2ForImageClassification
    from paddleformers.transformers.gemma3.modeling import _use_high_precision_cublas_for_fp32
    from paddleformers.transformers.model_utils import dtype_guard
    from classification_graph import ShieldGemma2InferGraph, ShieldGemma2TrainGraph

    started = time.perf_counter()
    paddle.set_device(arguments.device)
    paddle.seed(arguments.seed)
    np.random.seed(arguments.seed)
    with dtype_guard(arguments.dtype):
        if arguments.diagnostic_random:
            model, batches, assets = diagnostic_assets(paddle, ShieldGemma2Config, ShieldGemma2ForImageClassification)
        else:
            model, batches, assets = checkpoint_assets(
                paddle, ShieldGemma2Config, ShieldGemma2ForImageClassification,
                arguments.checkpoint, arguments.batches,
            )
    _use_high_precision_cublas_for_fp32(next(iter(model.parameters())))
    if arguments.diagnostic_random:
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.numpy().tobytes())
        assets["initial_parameters_sha256"] = digest.hexdigest()
        digest = hashlib.sha256()
        for batch in batches:
            for name in sorted(batch):
                digest.update(name.encode())
                digest.update(batch[name].tobytes())
        assets["input_batches_sha256"] = digest.hexdigest()
    training = arguments.task == "train"
    if training:
        model.train()
        graph = ShieldGemma2TrainGraph(model)
    else:
        model.eval()
        graph = ShieldGemma2InferGraph(model)
    if use_cinn:
        graph = paddle.jit.to_static(graph, full_graph=True)
    optimizer = None
    if training and arguments.learning_rate > 0:
        optimizer = paddle.optimizer.AdamW(
            parameters=model.parameters(), learning_rate=arguments.learning_rate,
            weight_decay=0.0, multi_precision=True,
        )
    names = INPUT_NAMES + (("labels",) if training else ())
    inputs = [tuple(paddle.to_tensor(batch[name]) for name in names) for batch in batches]
    verify = training and (arguments.verify_gradients or arguments.diagnostic_random)
    snapshots = {name: value.clone() for name, value in model.named_parameters()} if arguments.diagnostic_random else {}
    sample_snapshots = parameter_samples(model.named_parameters()) if verify and not arguments.diagnostic_random else {}
    records = []
    gradient_count = 0

    def synchronize():
        if arguments.device.startswith("gpu"):
            paddle.device.synchronize()

    context = paddle.enable_grad() if training else paddle.no_grad()
    with context:
        for step in range(arguments.steps):
            synchronize()
            step_started = time.perf_counter()
            outputs = graph(*inputs[step % len(inputs)])
            if training:
                outputs.backward()
                if verify:
                    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                    gradient_count = len(gradients)
                    if not gradients or not all(bool(paddle.isfinite(gradient).all()) for gradient in gradients):
                        raise RuntimeError(f"missing or nonfinite gradients at step {step}")
                if optimizer is not None:
                    optimizer.step()
                    optimizer.clear_grad()
                else:
                    model.clear_gradients()
            synchronize()
            elapsed = time.perf_counter() - step_started
            record = {"step": step, "step_time_s": elapsed}
            if training:
                loss = float(outputs)
                if not math.isfinite(loss):
                    raise RuntimeError(f"nonfinite loss at step {step}")
                record["loss"] = loss
                print(f"STEP_LOSS step={step} loss={loss:.10f} step_time_s={elapsed:.8f}", flush=True)
            else:
                logits, probabilities = outputs
                if not bool(paddle.isfinite(logits).all()) or not bool(paddle.isfinite(probabilities).all()):
                    raise RuntimeError(f"nonfinite classification output at step {step}")
                record.update(logits=logits.numpy().tolist(), probabilities=probabilities.numpy().tolist())
                print("STEP_INFER " + json.dumps(record), flush=True)
            records.append(record)
    changed = None
    if snapshots:
        changed = parameters_changed(model.named_parameters(), snapshots)
    elif sample_snapshots:
        current_samples = parameter_samples(model.named_parameters())
        changed = any(not np.array_equal(current_samples[name], sample) for name, sample in sample_snapshots.items())
    if changed is False and optimizer is not None:
        raise RuntimeError("optimizer completed without changing any checked parameter")
    steady = records[arguments.warmup_steps:]
    mean_step = sum(record["step_time_s"] for record in steady) / len(steady)
    saved = save_trained_checkpoint(paddle, model, arguments.checkpoint.resolve(), arguments.save_checkpoint,
                                   assets, arguments) if arguments.save_checkpoint else None
    summary = {
        "task": arguments.task, "mode": arguments.mode, "assets": assets,
        "formal_acceptance": False, "diagnostic_only": arguments.diagnostic_random,
        "performance_claim": False, "device": arguments.device,
        "environment": {"python": platform.python_version(), "paddle": paddle.__version__,
                        "paddle_commit": paddle.version.commit, "cuda_build": paddle.version.cuda(),
                        "cudnn_build": paddle.version.cudnn(), "source_commit": git_value("rev-parse", "HEAD"),
                        "source_diff_sha256": hashlib.sha256(git_value("diff", "HEAD").encode()).hexdigest(),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "classification_graph_sha256": sha256(Path(__file__).with_name("classification_graph.py"))},
        "dtype": arguments.dtype, "loss_dtype": "float32" if training else None,
        "seed": arguments.seed, "steps": arguments.steps, "warmup_steps": arguments.warmup_steps,
        "learning_rate": arguments.learning_rate if training else None,
        "optimizer": "AdamW" if optimizer is not None else None,
        "amp_autocast": False, "world_size": 1, "image_policy_pairs_per_batch": 1,
        "label_order": ["Yes/violates", "No/complies"], "gradient_count": gradient_count,
        "gradient_checks_enabled": verify, "parameters_changed": changed,
        "parameter_update_check": "all_parameters_exact" if snapshots else (
            "first_64_elements_per_trainable_parameter" if sample_snapshots else "disabled"),
        "saved_checkpoint": saved,
        "cinn_requested": use_cinn, "cinn_verified": False,
        "cinn_boundary": "full classification/loss wrapper" if use_cinn else None,
        "first_step_s": records[0]["step_time_s"], "steady_mean_step_s": mean_step,
        "steady_pairs_per_s": 1.0 / mean_step,
        "loop_total_s": sum(record["step_time_s"] for record in records),
        "process_total_s": time.perf_counter() - started, "records": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps({key: value for key, value in summary.items() if key != "records"}), flush=True)


if __name__ == "__main__":
    main()
