#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi
CHECKPOINT_DIR=${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the external Aya Expanse checkpoint}
CINN_RESOURCE_DIR=${CINN_RESOURCE_DIR:?Set CINN_RESOURCE_DIR to the external Aya Expanse data root}
MODEL_NAME=aya_expanse
MODE=${MODE:-dynamic}; TASK=${TASK:-all}; DEVICE=${DEVICE:-gpu:0}
case "$MODE" in dynamic|cinn) ;; *) exit 2;; esac
case "$TASK" in train|infer|all) ;; *) exit 2;; esac
OUT=${RESULTS_DIR:-${OUTPUT_DIR:-$HERE/results/$MODE-$TASK}}
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 3; }
mkdir -p "$(dirname "$OUT")"
run_one() {
  local split=$1 task=$2
  local output="$OUT-$split.json" batch="$CINN_RESOURCE_DIR/data/causal-128-20261124-a/$split"
  "$PYTHON_BIN" "$HERE/verify_assets.py" "$MODEL_NAME" "$CHECKPOINT_DIR" "$CINN_RESOURCE_DIR"
  local args=(--mode "$MODE" --task "$task" --checkpoint "$CHECKPOINT_DIR" --batches "$batch" --output "$output" --device "$DEVICE" --steps "${STEPS:-20}" --warmup "${WARMUP:-2}" --learning-rate "${LEARNING_RATE:-1e-6}")
  if [[ "$DEVICE" == cpu ]]; then export CUDA_VISIBLE_DEVICES=; else export CUDA_VISIBLE_DEVICES=0; fi
  [[ ! -e "$output" && ! -e "${output%.json}.log" ]] || { echo "Refusing to overwrite $output or its log" >&2; exit 3; }
  timeout --foreground --signal=TERM --kill-after="${TIMEOUT_KILL_AFTER_S:-30}s" "${TIMEOUT_S:-7200}s" \
    "$PYTHON_BIN" -u "$HERE/run_model.py" "${args[@]}" 2>&1 | tee "${output%.json}.log"
  if [[ "$MODE" == cinn ]]; then grep -q 'Compiling subgraph with CINN backend' "${output%.json}.log"; fi
}
if [[ "$TASK" == all ]]; then run_one train train; run_one eval infer; elif [[ "$TASK" == infer ]]; then run_one eval infer; else run_one train train; fi
