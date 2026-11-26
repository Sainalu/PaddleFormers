#!/usr/bin/env bash
set -euo pipefail
export MODE=dynamic TASK=${TASK:-all}
exec bash "$(dirname "$0")/run_model_common.sh"
