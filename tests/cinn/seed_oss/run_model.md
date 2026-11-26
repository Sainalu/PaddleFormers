# Seed-OSS CINN entry

This entry runs the verified real two-layer FP32 prefix checkpoint. Set `CHECKPOINT_DIR` to the external `Seed-OSS-36B-Instruct-prefix2` directory and `CINN_RESOURCE_DIR` to the external root containing `data/causal-128/{train,eval}`. Assets are checked against `asset_manifest.json`; no weights, NPZ files, logs, or results are committed.

The source baseline is `1d7c2679f24d9d245422afa4d59014cdd9689fd6`, with the Seed-OSS static-head/padding fix in the PR. Training and inference use the verified `run_v3` policy: FP32, eager attention, no backward blacklist, and task-specific parallel-matmul policy.

```bash
export CHECKPOINT_DIR=/path/to/Seed-OSS-36B-Instruct-prefix2
export CINN_RESOURCE_DIR=/path/to/seed-assets
RESULTS_DIR=/tmp/seed-dynamic bash tests/cinn/seed_oss/run_model.sh
RESULTS_DIR=/tmp/seed-cinn bash tests/cinn/seed_oss/run_model_cinn.sh
```

Set `TASK=train|infer`, `STEPS`, `WARMUP`, `LEARNING_RATE`, `TIMEOUT_S`, and `DEVICE=cpu` for bounded checks. CPU is dynamic-only; GPU requires physical GPU0. The benchmark covers fixed-sequence prefill and training updates, not autoregressive decode or full-model quality.
