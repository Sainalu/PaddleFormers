# Aya Expanse CINN entry

This entry runs the verified real two-layer Aya Expanse prefix in FP32. Set `CHECKPOINT_DIR` to the external `aya-expanse-8b-prefix2` directory and `CINN_RESOURCE_DIR` to the external asset root containing `data/causal-128-20261124-a/{train,eval}` and the pinned metadata directory. The manifest and original source config/index hashes are checked before model loading.

The source baseline is `1d7c2679f24d9d245422afa4d59014cdd9689fd6`; the PR contains only the Cohere static-head fix. Training uses the v1 code policy, while production inference used the v2 lock with parallel-matmul fusion disabled. This entry is a fixed-sequence prefill/optimizer regression and makes no decode or full-model claim.

```bash
export CHECKPOINT_DIR=/path/to/aya-expanse-8b-prefix2
export CINN_RESOURCE_DIR=/path/to/aya-expanse-assets
RESULTS_DIR=/tmp/aya-expanse-dynamic bash tests/cinn/aya_expanse/run_model.sh
RESULTS_DIR=/tmp/aya-expanse-cinn bash tests/cinn/aya_expanse/run_model_cinn.sh
```

Use `TASK=train|infer`, `STEPS`, `WARMUP`, `LEARNING_RATE`, `TIMEOUT_S`, and `DEVICE=cpu` for bounded runs. GPU runs require physical GPU0; CPU supports dynamic mode only. External weights and data are intentionally excluded from the PR.
