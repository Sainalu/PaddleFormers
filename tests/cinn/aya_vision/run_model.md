# Aya Vision CINN entry

This entry uses the verified v4 runner for the real text4/vision2 prefix checkpoint, FP32, fixed 1024-token multimodal batches and full image/text forward. Set `CHECKPOINT_DIR` to the external checkpoint and `CINN_RESOURCE_DIR` to the external root containing `data/fixed_1024_20261124T0727Z` and `metadata/`. The manifest fixes all small metadata and batch identities; weights and images stay outside the PR.

The source baseline is `1d7c2679f24d9d245422afa4d59014cdd9689fd6`; this branch contains the Aya Vision v4 modeling, processor and SigLIP fixes. `run_model_cinn.sh` requires GPU0 and checks the real CINN compile marker. Set `TIMEOUT_S` to bound a run. No KV-cache decode or full-model quality claim is made.

```bash
export PYTHON_BIN=python3 CHECKPOINT_DIR=/path/to/aya-vision-8b-text4-vision2
export CINN_RESOURCE_DIR=/path/to/aya-vision-assets
RESULTS_DIR=/tmp/aya-vision-dynamic bash tests/cinn/aya_vision/run_model.sh
RESULTS_DIR=/tmp/aya-vision-cinn bash tests/cinn/aya_vision/run_model_cinn.sh
```
