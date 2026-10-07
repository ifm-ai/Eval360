#!/bin/bash

# Model-owned resources are supplied as explicit sbatch arguments by
# SlurmManager. This script deliberately contains no fixed resource request.

# Conda-mode variant: activates a conda environment by name.

set -e

# Initialize conda for non-interactive shells
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "conda not found — install miniconda/anaconda or use venv_path instead" >&2
    exit 1
fi

conda activate "$conda_env"

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
