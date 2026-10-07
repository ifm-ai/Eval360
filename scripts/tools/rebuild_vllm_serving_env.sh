#!/usr/bin/env bash

set -euo pipefail

# Create or update an Eval360 vLLM serving environment (Python 3.12) with conda.
#
# Usage:
#   bash scripts/tools/rebuild_vllm_serving_env.sh
#
# Custom location/name:
#   ENV_NAME=vllm-serving ENV_DIR="$HOME/vllm-serving" CONDA_BIN=conda \
#   bash scripts/tools/rebuild_vllm_serving_env.sh
#
# Notes:
# - If ENV_DIR already exists, this script reuses it and updates packages in
#   place. It does not delete the environment directory.
# - Install vLLM on a GPU node if that is required in your environment.
# - The script creates a small bin/activate wrapper inside the conda env so
#   Eval360 model configs can continue to point at <env>/bin/activate.
# - To install a pinned XLLM/vLLM GitHub revision, set:
#   XLLM_GITHUB_URL=https://github.com/ifm-ai/xllm.git XLLM_GITHUB_REF=<commit>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ENV_NAME="${ENV_NAME:-vllm-serving}"
ENV_DIR="${ENV_DIR:-${HOME}/${ENV_NAME}}"
CONDA_BIN="${CONDA_BIN:-conda}"
PYTHON_SPEC="${PYTHON_SPEC:-python=3.12}"
SKIP_CACHE_CLEANUP="${SKIP_CACHE_CLEANUP:-0}"
TORCH_WHL_INDEX="${TORCH_WHL_INDEX:-https://download.pytorch.org/whl/cu128}"
VLLM_PIP_SPEC="${VLLM_PIP_SPEC:-vllm}"
XLLM_GITHUB_URL="${XLLM_GITHUB_URL:-}"
XLLM_GITHUB_REF="${XLLM_GITHUB_REF:-}"

if [[ -n "${XLLM_GITHUB_URL}" || -n "${XLLM_GITHUB_REF}" ]]; then
  if [[ -z "${XLLM_GITHUB_URL}" || -z "${XLLM_GITHUB_REF}" ]]; then
    echo "XLLM_GITHUB_URL and XLLM_GITHUB_REF must be set together" >&2
    exit 1
  fi
  VLLM_PIP_SPEC="git+${XLLM_GITHUB_URL}@${XLLM_GITHUB_REF}"
fi

log_step() {
  echo
  echo "==> $*"
}

log_kv() {
  printf "    %-28s %s\n" "$1:" "$2"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

resolve_path() {
  local target="$1"
  local parent
  local base
  if [[ -d "$target" ]]; then
    (
      cd "$target"
      pwd -P
    )
    return
  fi
  parent="$(dirname "$target")"
  base="$(basename "$target")"
  (
    cd "$parent"
    printf "%s/%s\n" "$(pwd -P)" "$base"
  )
}

guard_env_dir() {
  local resolved_env_dir
  local resolved_home
  local resolved_repo_root
  local env_basename

  if [[ -z "${ENV_DIR//[[:space:]]/}" ]]; then
    echo "Refusing to use an empty ENV_DIR" >&2
    exit 1
  fi

  if ! resolved_env_dir="$(resolve_path "${ENV_DIR}")"; then
    echo "Refusing unsafe ENV_DIR because its parent does not exist: ${ENV_DIR}" >&2
    exit 1
  fi
  resolved_home="$(resolve_path "${HOME}")"
  resolved_repo_root="$(resolve_path "${REPO_ROOT}")"
  env_basename="$(basename "${resolved_env_dir}")"

  if [[ "${resolved_env_dir}" == "/" ]]; then
    echo "Refusing unsafe ENV_DIR: /" >&2
    exit 1
  fi
  if [[ "${resolved_env_dir}" == "${resolved_home}" ]]; then
    echo "Refusing unsafe ENV_DIR equal to HOME: ${ENV_DIR}" >&2
    exit 1
  fi
  if [[ "${resolved_env_dir}" == "${resolved_repo_root}" ]]; then
    echo "Refusing unsafe ENV_DIR equal to repository root: ${ENV_DIR}" >&2
    exit 1
  fi
  if [[ "${ALLOW_UNSAFE_ENV_DIR:-0}" != "1" && "${env_basename}" != *vllm* ]]; then
    echo "Refusing ENV_DIR without 'vllm' in its final path component: ${ENV_DIR}" >&2
    echo "Set ALLOW_UNSAFE_ENV_DIR=1 to override intentionally." >&2
    exit 1
  fi
}

require_command "${CONDA_BIN}"
CONDA_BASE="$("${CONDA_BIN}" info --base)"
guard_env_dir

log_step "Resolved vLLM serving environment configuration"
log_kv "repository root" "${REPO_ROOT}"
log_kv "environment name" "${ENV_NAME}"
log_kv "environment dir" "${ENV_DIR}"
log_kv "conda binary" "${CONDA_BIN}"
log_kv "conda base" "${CONDA_BASE}"
log_kv "python spec" "${PYTHON_SPEC}"
log_kv "torch wheel index" "${TORCH_WHL_INDEX}"
log_kv "vLLM pip spec" "${VLLM_PIP_SPEC}"
if [[ -n "${XLLM_GITHUB_URL}" ]]; then
  log_kv "XLLM GitHub URL" "${XLLM_GITHUB_URL}"
  log_kv "XLLM GitHub ref/hash" "${XLLM_GITHUB_REF}"
else
  log_kv "XLLM GitHub ref/hash" "not set; using vLLM pip spec above"
fi

if [[ -e "${ENV_DIR}" && ! -d "${ENV_DIR}" ]]; then
  echo "Refusing ENV_DIR because it exists and is not a directory: ${ENV_DIR}" >&2
  exit 1
fi

if [[ "${SKIP_CACHE_CLEANUP}" != "1" ]]; then
  log_step "Clearing stale JIT caches"
  log_kv "flashinfer cache" "${HOME}/.cache/flashinfer"
  log_kv "vLLM torch compile cache" "${HOME}/.cache/vllm/torch_compile_cache"
  rm -rf "${HOME}/.cache/flashinfer"
  rm -rf "${HOME}/.cache/vllm/torch_compile_cache"
else
  log_step "Skipping JIT cache cleanup"
  log_kv "reason" "SKIP_CACHE_CLEANUP=1"
fi

if [[ -d "${ENV_DIR}" ]]; then
  log_step "Reusing existing conda environment"
  log_kv "environment dir" "${ENV_DIR}"
else
  log_step "Creating conda environment"
  log_kv "environment dir" "${ENV_DIR}"
  log_kv "python spec" "${PYTHON_SPEC}"
  "${CONDA_BIN}" create -y -p "${ENV_DIR}" "${PYTHON_SPEC}"
fi

log_step "Starting environment patching"
log_kv "patch target" "${ENV_DIR}/bin/activate"
log_kv "patch action" "write Eval360-compatible conda activation wrapper"
log_kv "conda activation source" "${CONDA_BASE}/etc/profile.d/conda.sh"
log_kv "activated environment" "${ENV_DIR}"
mkdir -p -- "${ENV_DIR}/bin"
cat > "${ENV_DIR}/bin/activate" <<EOF
#!/usr/bin/env bash
# Auto-generated by scripts/tools/rebuild_vllm_serving_env.sh
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"
EOF
chmod +x "${ENV_DIR}/bin/activate"
log_step "Finished environment patching"
log_kv "patched file" "${ENV_DIR}/bin/activate"

# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

python -V
python -m pip install --upgrade pip setuptools wheel

log_step "Installing or updating vLLM/XLLM package"
log_kv "pip spec" "${VLLM_PIP_SPEC}"
log_kv "torch wheel index" "${TORCH_WHL_INDEX}"
if [[ -n "${XLLM_GITHUB_URL}" ]]; then
  log_kv "source" "${XLLM_GITHUB_URL}"
  log_kv "source ref/hash" "${XLLM_GITHUB_REF}"
fi
python -m pip install "${VLLM_PIP_SPEC}" --extra-index-url "${TORCH_WHL_INDEX}"
log_step "Finished installing or updating vLLM/XLLM package"
python - <<'PY'
from importlib import metadata

for package_name in ("vllm", "xllm"):
    try:
        dist = metadata.distribution(package_name)
    except metadata.PackageNotFoundError:
        continue
    print(f"Installed package: {package_name}")
    print(f"Installed version: {dist.version}")
    print(f"Installed location: {dist.locate_file('')}")
PY

echo
echo "Environment ready."
echo "Activate with:"
echo "  source \"${ENV_DIR}/bin/activate\""
