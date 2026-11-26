# ShieldGemma 2 CINN entry

This entry validates the real original text2/vision2 structural prefix in FP32. Set `CHECKPOINT_DIR` to the external checkpoint and `CINN_RESOURCE_DIR` to the external root containing `data/docling-structure-20261018/{train-batches,eval-batches}`. The asset manifest fixes all checkpoint and batch manifests; no safety accuracy claim is made and no weights/results are committed.

The source baseline is `1d7c2679f24d9d245422afa4d59014cdd9689fd6`, with the ShieldGemma2/Gemma3 CINN patch in this PR. GPU runs require physical GPU0; CPU is dynamic-only. Training uses the real image-policy classifier, binary loss, backward, AdamW and gradient clearing. Inference is fixed image-policy classification, not generation or KV-cache decode.

```bash
export PADDLEFORMERS_DIR=/path/to/PaddleFormers
export CHECKPOINT_DIR=/path/to/shieldgemma2-2text-2vision-bf16
export CINN_RESOURCE_DIR=/path/to/shieldgemma2-assets
RESULTS_DIR=/tmp/shield-dynamic bash tests/cinn/shieldgemma2/run_model.sh
RESULTS_DIR=/tmp/shield-cinn bash tests/cinn/shieldgemma2/run_model_cinn.sh
```

Set `TASK=train|infer`, `STEPS`, `WARMUP`, `LEARNING_RATE`, and `TIMEOUT_S` to bound a run. CPU is dynamic-only; GPU uses physical GPU0.
