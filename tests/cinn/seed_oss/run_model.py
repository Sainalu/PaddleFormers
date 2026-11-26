"""Strict Seed-OSS FP32 dynamic/CINN entry used by the model test PR."""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from common import checked_file, load_batches, sha256, write_json


def load_model(checkpoint, training):
    import paddle
    from safetensors import safe_open
    from paddleformers.transformers import SeedOssConfig, SeedOssForCausalLM

    config = SeedOssConfig.from_pretrained(str(checkpoint))
    config._attn_implementation = "eager"
    config.use_cache = False
    for name in ("attention_dropout", "residual_dropout"):
        if hasattr(config, name):
            setattr(config, name, 0.0)
    config.use_fused_head_and_loss_fn = False
    config.use_filtered_label_loss = False
    config.attention_dropout = 0.0
    config.residual_dropout = 0.0
    config.use_fused_head_and_loss_fn = False
    config.use_filtered_label_loss = False
    model = SeedOssForCausalLM(config)
    expected = model.state_dict()
    transposed = tuple(f"{name}.weight" for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
    loaded = {}
    with safe_open(str(checkpoint / "model.safetensors"), framework="numpy") as source:
        for name in source.keys():
            values = np.asarray(source.get_tensor(name), dtype=np.float32)
            if name.endswith(transposed):
                values = values.T.copy()
            if name not in expected or list(values.shape) != list(expected[name].shape):
                raise ValueError(f"unexpected checkpoint tensor: {name}")
            loaded[name] = paddle.to_tensor(values)
    if "lm_head.weight" not in loaded and config.tie_word_embeddings:
        loaded["lm_head.weight"] = loaded["model.embed_tokens.weight"]
    if set(loaded) != set(expected):
        raise ValueError(f"strict tensor mismatch: missing={set(expected)-set(loaded)}, extra={set(loaded)-set(expected)}")
    model.set_state_dict(loaded)
    if {str(value.dtype) for value in model.parameters()} != {"paddle.float32"}:
        raise ValueError("FP32 parameters are required")
    model.train() if training else model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dynamic", "cinn"), required=True)
    parser.add_argument("--task", choices=("train", "infer"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu:0"), default="gpu:0")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.output.exists() or not 0 <= args.warmup < args.steps:
        parser.error("new output and 0 <= warmup < steps are required")
    if args.device == "gpu:0" and os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        parser.error("GPU runs require CUDA_VISIBLE_DEVICES=0")
    if args.device == "cpu" and (args.mode == "cinn" or os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""):
        parser.error("CPU runs only support dynamic mode with hidden CUDA")
    for flag in ("FLAGS_use_cinn", "FLAGS_prim_enable_dynamic", "FLAGS_prim_all"):
        os.environ[flag] = "true" if args.mode == "cinn" else "false"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    import paddle
    paddle.set_device(args.device)
    paddle.set_default_dtype("float32")
    paddle.seed(args.seed)
    np.random.seed(args.seed)
    checkpoint = args.checkpoint.resolve()
    manifest = json.loads(checked_file(checkpoint / "checkpoint_manifest.json").read_text())
    for name, entry in manifest["files"].items():
        checked_file(checkpoint / name, entry["sha256"])
    model = load_model(checkpoint, args.task == "train")
    batches, _ = load_batches(args.batches, checkpoint)

    class Graph(paddle.nn.Layer):
        def __init__(self, wrapped):
            super().__init__(); self.wrapped = wrapped
        def forward(self, input_ids, attention_mask, position_ids, labels=None):
            output = self.wrapped(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                  labels=labels, use_cache=False, return_dict=False)
            return output[0]

    training = args.task == "train"
    graph = Graph(model)
    if args.mode == "cinn":
        graph = paddle.jit.to_static(graph, full_graph=True)
    optimizer = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=args.learning_rate,
                                       weight_decay=0.0) if training else None
    inputs = []
    names = ("input_ids", "attention_mask", "position_ids", "labels") if training else ("input_ids", "attention_mask", "position_ids")
    for batch in batches:
        inputs.append(tuple(paddle.to_tensor(batch[name]) for name in names))
    records = []
    started = time.perf_counter()
    with paddle.enable_grad() if training else paddle.no_grad():
        for step in range(args.steps):
            if args.device == "gpu:0": paddle.device.synchronize()
            tick = time.perf_counter()
            result = graph(*inputs[step % len(inputs)])
            if training:
                result.backward(); optimizer.step(); optimizer.clear_grad()
                value = float(result)
                if not np.isfinite(value): raise ValueError("nonfinite loss")
            else:
                value = float(paddle.abs(result).max())
                if not bool(paddle.isfinite(result).all()): raise ValueError("nonfinite logits")
            if args.device == "gpu:0": paddle.device.synchronize()
            elapsed = time.perf_counter() - tick
            records.append({"step": step, "batch": step % len(inputs), "seconds": elapsed,
                            "loss": value if training else None, "max_abs_logit": value if not training else None})
            print(json.dumps(records[-1]), flush=True)
    write_json(args.output, {"model": "seed_oss", "task": args.task, "mode": args.mode, "device": args.device,
                             "dtype": "float32", "steps": args.steps, "warmup": args.warmup,
                             "checkpoint_sha256": sha256(checkpoint / "checkpoint_manifest.json"),
                             "batch_manifest_sha256": sha256(args.batches / "manifest.json"),
                             "source_commit": "1d7c2679f24d9d245422afa4d59014cdd9689fd6",
                             "environment": {"python": platform.python_version(), "paddle": paddle.__version__},
                             "elapsed_s": time.perf_counter() - started, "records": records})


if __name__ == "__main__":
    main()
