#!/bin/bash

# Model-owned resources are supplied as explicit sbatch arguments by
# SlurmManager. This script deliberately contains no fixed resource request.

# Container-mode variant: uses enroot/pyxis (--container-image on sbatch).
# No venv activation needed for VLLM — it is pre-installed in the container image.
# The benchmark runner venv is still created on the host filesystem.

set -e


# The EXIT trap below writes `.job_failed` into "${output_dir}", so that
# directory has to exist BEFORE the trap is installed. It used to be created
# only near the end of this script, after the health gate, which meant an early
# failure ran `touch` against a directory that did not exist yet: the touch
# failed silently and the job left NO sentinel at all. Nothing creates it
# beforehand either — `Scheduler.handle_imported_dataset_event` submits without
# it — so on a first run it genuinely is absent. The sentinels are the only
# signal separating "failed during setup" from "failed internally" from
# "preempted, reschedule", so losing them loses the reason the job failed.
mkdir -p "${output_dir}"

# Write .job_failed sentinel on any non-preempted failure so the scheduler
# can distinguish an internal crash from a preemption.
trap 'status=$?; if [ $status -ne 0 ] && [ ! -f "${output_dir}/.job_complete" ]; then touch "${output_dir}/.job_failed"; fi' EXIT

export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${vllm_allow_long_max_model_len:-0}"

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


# Run venv setup while VLLM deploys.
# Sentinel-guarded (idempotent, preemption-safe): if a previous run was
# preempted mid-install the sentinel won't exist, the broken venv is deleted,
# and setup runs cleanly from scratch.
VENV="${repo_root}/.eval360/envs/${runner_name}"
export VENV
SENTINEL="${VENV}/.setup_complete"

if [ ! -f "$SENTINEL" ]; then
    rm -rf "$VENV"
    PYTHON_BIN=""
    python_candidates=()
    if [ -n "${bootstrap_python:-}" ]; then
        python_candidates+=("$bootstrap_python")
    fi
    if command -v python3 >/dev/null 2>&1; then
        python_candidates+=("$(command -v python3)")
    fi
    if command -v python >/dev/null 2>&1; then
        python_candidates+=("$(command -v python)")
    fi
    for candidate in "${python_candidates[@]}"; do
        if [ -x "$candidate" ] && "$candidate" -c "import venv" >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
    if [ -z "$PYTHON_BIN" ]; then
        echo "No usable bootstrap Python interpreter found for venv creation" >&2
        exit 1
    fi
    echo "Using helper Python: $PYTHON_BIN"
    "$PYTHON_BIN" -m venv "$VENV"
    SETUP_SCRIPT=$(mktemp)
    echo "$setup_script_b64" | base64 -d > "$SETUP_SCRIPT"
    bash "$SETUP_SCRIPT"
    rm -f "$SETUP_SCRIPT"
    touch "$SENTINEL"
fi

# Wait for VLLM to be healthy on localhost:8000. Some container images do not
# include curl, but they do have Python because VLLM is launched above.
elapsed=0
until python - <<'EOF' > /dev/null 2>&1
import sys
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=3) as response:
        sys.exit(0 if 200 <= response.status < 300 else 1)
except Exception:
    sys.exit(1)
EOF
do
    if [ "$elapsed" -ge "$max_time_to_deploy" ]; then
        echo "VLLM did not become healthy within ${max_time_to_deploy}s" >&2
        exit 1
    fi
    sleep 10
    elapsed=$((elapsed + 10))
done

source "${VENV}/bin/activate"

mkdir -p "${output_dir}"
touch "${output_dir}/.setup_complete_job"

# Run the benchmark
BENCHMARK_SCRIPT=$(mktemp)
echo "$benchmark_script_b64" | base64 -d > "$BENCHMARK_SCRIPT"
bash "$BENCHMARK_SCRIPT"
rm -f "$BENCHMARK_SCRIPT"

touch "${output_dir}/.job_complete"
