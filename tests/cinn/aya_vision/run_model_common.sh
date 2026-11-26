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
CHECKPOINT_DIR=${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to the external Aya Vision checkpoint}
CINN_RESOURCE_DIR=${CINN_RESOURCE_DIR:?Set CINN_RESOURCE_DIR to the external Aya Vision asset root}
PADDLEFORMERS_DIR=${PADDLEFORMERS_DIR:-$(cd "$HERE/../../.." && pwd)}
MODE=${MODE:-dynamic}; TASK=${TASK:-all}; DEVICE=${DEVICE:-gpu}
case "$MODE" in dynamic|cinn) ;; *) exit 2;; esac
case "$TASK" in train|infer|all) ;; *) exit 2;; esac
case "$DEVICE" in cpu|gpu) ;; *) exit 2;; esac
OUT=${RESULTS_DIR:-${OUTPUT_DIR:-$HERE/results/$MODE-$TASK}}
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 3; }
mkdir -p "$(dirname "$OUT")"
if [[ "$DEVICE" == cpu && "$MODE" == cinn ]]; then echo 'CPU cannot validate CINN' >&2; exit 2; fi
run_one() {
  local split=$1 task=$2
  local output="$OUT-$split"
  [[ ! -e "$output" && ! -e "${output}.log" ]] || { echo "Refusing to overwrite $output or its log" >&2; exit 3; }
  "$PYTHON_BIN" "$HERE/verify_assets.py" aya_vision "$CHECKPOINT_DIR" "$CINN_RESOURCE_DIR"
  local args=(--checkpoint "$CHECKPOINT_DIR" --batches "$CINN_RESOURCE_DIR/data/fixed_1024_20261124T0727Z" --output "$output" --mode "$task" --backend "$MODE" --device "$DEVICE" --steps "${STEPS:-2}" --warmup "${WARMUP:-0}" --learning-rate "${LEARNING_RATE:-1e-5}")
  export PADDLEFORMERS_DIR CINN_ROOT="$CINN_RESOURCE_DIR" AYA_VISION_METADATA_DIR="$CINN_RESOURCE_DIR/metadata"
  if [[ "$DEVICE" == gpu ]]; then export CUDA_VISIBLE_DEVICES=0; args+=(--gpu-scheduled); else export CUDA_VISIBLE_DEVICES=; fi
  timeout --foreground --signal=TERM --kill-after="${TIMEOUT_KILL_AFTER_S:-30}s" "${TIMEOUT_S:-7200}s" \
    "$PYTHON_BIN" -u "$HERE/run.py" "${args[@]}" 2>&1 | tee "${output}.log"
  if [[ "$MODE" == cinn ]]; then grep -q 'Compiling subgraph with CINN backend' "${output}.log"; fi
}
if [[ "$TASK" == all ]]; then run_one train train; run_one eval infer; elif [[ "$TASK" == infer ]]; then run_one eval infer; else run_one train train; fi
