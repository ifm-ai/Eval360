#!/bin/bash

# Model-owned resources are supplied as explicit sbatch arguments by
# SlurmManager. This script deliberately contains no fixed resource request.

set -e


VENV_ACTIVATE="$venv_path"
if [ -d "$VENV_ACTIVATE" ]; then
    VENV_ACTIVATE="$VENV_ACTIVATE/bin/activate"
fi
if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "Missing venv activation script: $VENV_ACTIVATE" >&2
    exit 1
fi

source "$VENV_ACTIVATE"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${vllm_allow_long_max_model_len:-0}"
export VLLM_LOGGING_LEVEL="${vllm_logging_level:-WARNING}"

# Avoid shared compile-cache corruption across concurrent Slurm jobs. Keep the
# top-level path user-specific so stale directories from other users/images do
# not block cache creation.
CACHE_UID="${UID:-$(id -u)}"
CACHE_JOB_ID="${SLURM_JOB_ID:-$$}"
CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/eval360-vllm-${CACHE_UID}-${CACHE_JOB_ID}"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
mkdir -p "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

python - <<'EOF' &
import json, os, subprocess, base64

args = ["vllm", "serve", os.environ["model_path"]]
b64 = os.environ["vllm_args"]
args += json.loads(base64.b64decode(b64).decode())

print(f"About to run {args}")
subprocess.run(args)
EOF

sleep 86400
