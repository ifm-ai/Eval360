#!/bin/bash

# Model-owned resources are supplied as explicit sbatch arguments by
# SlurmManager. This script deliberately contains no fixed resource request.

# Container-mode variant: uses enroot/pyxis (--container-image on sbatch).
# No venv activation needed — VLLM is pre-installed in the container image.

set -e

export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${vllm_allow_long_max_model_len:-0}"
export VLLM_LOGGING_LEVEL="${vllm_logging_level:-WARNING}"
export PATH="/opt/venv/bin:${PATH}"

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

PYTHON_BIN="${bootstrap_python:-}"
is_usable_python() {
    local candidate="$1"
    [[ -n "${candidate}" && -x "${candidate}" ]] || return 1
    "${candidate}" -c 'import sys; raise SystemExit(0 if sys.version_info.major >= 3 else 1)' >/dev/null 2>&1
}

if [[ -n "${PYTHON_BIN}" ]]; then
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "Invalid bootstrap_python: not found or not executable: ${PYTHON_BIN}" >&2
        exit 127
    fi
    if ! is_usable_python "${PYTHON_BIN}"; then
        echo "Invalid bootstrap_python: not a usable Python interpreter: ${PYTHON_BIN}" >&2
        exit 127
    fi
else
    for python_candidate in "$(command -v python3 || true)" "$(command -v python || true)" "/opt/venv/bin/python"; do
        if is_usable_python "${python_candidate}"; then
            PYTHON_BIN="${python_candidate}"
            break
        fi
    done
    if [[ -z "${PYTHON_BIN}" ]]; then
        echo "No usable python interpreter found inside container" >&2
        exit 127
    fi
fi

VLLM_BIN_RESOLVED=""
if [[ -n "${VLLM_BIN:-}" ]]; then
    if [[ "${VLLM_BIN}" == */* ]]; then
        if [[ ! -x "${VLLM_BIN}" ]]; then
            echo "Invalid VLLM_BIN: not found or not executable: ${VLLM_BIN}" >&2
            exit 127
        fi
        VLLM_BIN_RESOLVED="${VLLM_BIN}"
    elif command -v -- "${VLLM_BIN}" >/dev/null 2>&1; then
        VLLM_BIN_RESOLVED="$(command -v -- "${VLLM_BIN}")"
    else
        echo "Invalid VLLM_BIN: command not found: ${VLLM_BIN}" >&2
        exit 127
    fi
elif [[ -x /opt/venv/bin/vllm ]]; then
    VLLM_BIN_RESOLVED="/opt/venv/bin/vllm"
elif command -v vllm >/dev/null 2>&1; then
    VLLM_BIN_RESOLVED="$(command -v vllm)"
else
    echo "No executable vllm binary found inside container" >&2
    exit 127
fi
export VLLM_BIN_RESOLVED

"${PYTHON_BIN}" - <<'EOF' &
import json, os, subprocess, base64

args = [os.environ["VLLM_BIN_RESOLVED"], "serve", os.environ["model_path"]]
b64 = os.environ["vllm_args"]
args += json.loads(base64.b64decode(b64).decode())

print(f"About to run {args}")
subprocess.run(args)
EOF

sleep 86400
