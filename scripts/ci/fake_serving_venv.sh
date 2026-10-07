#!/usr/bin/env bash
# Build the venv that `scheduler/slurm/sbatch_script.sh` will activate on the
# compute node, with a fake `vllm` on its PATH.
#
# The script under test does three things this has to satisfy, and it is worth
# being explicit about them because each is a place the real cluster has broken
# before:
#
#   1. `$venv_path` must be a directory containing bin/activate, OR the
#      activate script itself — `sbatch_script.sh` accepts both shapes, and
#      `ModelInstance.validate_venv_path` normalises between them. A real venv
#      is created here rather than a hand-made directory so that both forms are
#      genuinely exercised.
#   2. After `source .../activate`, a command named `vllm` must resolve. The
#      shim goes in the venv's own bin/ so it resolves the same way the real one
#      does — via PATH after activation, not via an absolute path the script
#      never uses.
#   3. The script execs `vllm serve <model_path> <decoded args...>`. The shim
#      passes "$@" through untouched so the decoded argv reaches the stub
#      exactly as sbatch_script.sh built it.
#
# Usage: fake_serving_venv.sh <venv-path>

set -euo pipefail

VENV_PATH="${1:?usage: fake_serving_venv.sh <venv-path>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FAKE_VLLM="${REPO_ROOT}/scripts/ci/fake_vllm_server.py"

if [ ! -f "$FAKE_VLLM" ]; then
    echo "FAIL: missing ${FAKE_VLLM}" >&2
    exit 1
fi

python3 -m venv "$VENV_PATH"

cat > "${VENV_PATH}/bin/vllm" <<SHIM
#!/usr/bin/env bash
# Stand-in for the real \`vllm\` console script. Deliberately a passthrough:
# anything this shim interpreted would be something the stub could not see.
exec python3 "${FAKE_VLLM}" "\$@"
SHIM
chmod +x "${VENV_PATH}/bin/vllm"

# Prove the shim resolves through activation rather than assuming it does — the
# failure mode otherwise is a job that dies on the compute node with the useless
# message "vllm: command not found", minutes after submission.
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"
resolved="$(command -v vllm || true)"
if [ "$resolved" != "${VENV_PATH}/bin/vllm" ]; then
    echo "FAIL: after activation 'vllm' resolved to '${resolved:-<nothing>}'," >&2
    echo "      expected '${VENV_PATH}/bin/vllm'" >&2
    exit 1
fi

echo "OK: serving venv at ${VENV_PATH}, fake vllm at ${resolved}"
