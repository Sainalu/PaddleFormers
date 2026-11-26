# Granite 3.2 CINN entry

This directory validates the real two-layer FP32 prefix checkpoint used by the CINN acceptance run. It does not include weights, NPZ batches, logs, or benchmark results.

Set `CHECKPOINT_DIR` to the external `granite-3.2-2b-instruct-2layer-bf16` directory and `CINN_RESOURCE_DIR` to the external CINN asset root containing `data/causal-128/{train,eval}`. Both checkpoint and batch manifests are checked before execution. The pinned PaddleFormers base is `1d7c2679f24d9d245422afa4d59014cdd9689fd6`; the PR contains the Granite source change in `paddleformers/transformers/granite/modeling.py` and its regression test.

```bash
export CHECKPOINT_DIR=/path/to/granite-3.2-2b-instruct-2layer-bf16
export CINN_RESOURCE_DIR=/path/to/granite-assets
RESULTS_DIR=/tmp/granite-dynamic bash tests/cinn/granite32/run_model.sh
RESULTS_DIR=/tmp/granite-cinn bash tests/cinn/granite32/run_model_cinn.sh
```

Use `TASK=train|infer`, `STEPS`, `WARMUP`, `LEARNING_RATE`, `TIMEOUT_S`, `DEVICE=cpu`, and `PYTHON_BIN` for bounded smoke runs. CPU supports dynamic mode only; GPU mode requires physical GPU0. Training includes forward, loss, backward, AdamW, and gradient clearing. Inference is fixed-sequence full-logit prefill and does not claim autoregressive decode performance.
