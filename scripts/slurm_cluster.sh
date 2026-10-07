#!/usr/bin/env bash
# eval360 slurm-cluster — a disposable Slurm cluster you can run anywhere.
#
# This is to Slurm what `kind` is to Kubernetes: a real control plane, in a
# container, on whatever machine you happen to be using. The point is that
# `scheduler/slurm_manager.py` can be pointed at something that actually runs
# sbatch, squeue, sinfo, scancel and sacct — instead of at
# `tests/fake_slurm.py`, which replaces the class, or at
# `tests/test_job_manager.py`, which patches `asyncio.create_subprocess_exec`
# and feeds the parsers strings written by the same person who wrote them.
#
# THE SAME IMAGE RUNS IN CI AND ON YOUR LAPTOP. That is the whole design, and
# it is why nothing here installs Slurm on the host. An earlier draft apt-got
# Slurm onto the GitHub runner and relied on the runner being thrown away
# afterwards — which meant it could never be torn down, never run twice on one
# machine, never run on a self-hosted runner, and never run locally at all. So
# the local and CI clusters would have had to be configured separately, and the
# one subsystem this effort exists to stop guessing about would have had two
# definitions to guess between.
#
#   up         build the image if needed, start the container, bring Slurm up
#   probe      drive the real SlurmManager against the cluster, by hand
#   test       run a pytest selection inside the cluster
#   shell      interactive shell in the container
#   exec       run an arbitrary command in the container
#   status     sinfo / squeue / node detail
#   logs       daemon logs (slurmctld, slurmd, slurmdbd) and job output
#   down       delete the container
#   rebuild    force a fresh image build, then up
#   image-tag  print the content-derived image tag (CI builds under this name)
#
# The working tree is bind-mounted READ-ONLY at /workspace and imported via
# PYTHONPATH, so edits take effect with no rebuild and nothing can be written
# back into the repo. Anything under test that needs to write gets an explicit
# tmpfs: /scratch, and /workspace/.eval360 for the runner venvs production
# insists on putting there.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${EVAL360_SLURM_CONTAINER:-eval360-slurm}"

# Proof that WE created a container, so `up` and `down` never delete someone
# else's. The name alone is not proof: `eval360-slurm` is a name anybody could
# have used, and removing it on the strength of the name is how a script
# destroys an unrelated container. A foreign name collision is a hard failure
# here, not a silent skip — carrying on would mean `up` starting a cluster that
# is not the one the user is about to be handed.
OWNER_LABEL_KEY="dev.eval360.slurm-cluster"
OWNER_LABEL_VALUE="IFM-AI/Eval360"

# Which user runs probe/test/shell/exec inside the container. NOT root: see the
# `docker run` block in cmd_up for why that matters given --privileged. `slurm`
# is chosen because it is the configured SlurmUser, which is what lets
# test_job_lifecycle.py's `scontrol update NodeName=... State=DRAIN` work
# without root — Slurm authorises on `uid == 0 || uid == SlurmUser ||
# AdminLevel >= Operator`. An unrelated unprivileged user would need an
# accounting record and an AdminLevel granted at bring-up; this needs nothing.
# Override only if you are debugging something that genuinely needs root.
RUNTIME_USER="${EVAL360_SLURM_RUNTIME_USER:-slurm}"

# Where anything under test is allowed to write. /workspace is mounted
# read-only, so a pytest --junitxml, a scratch file, or $HOME goes here.
CI_SCRATCH=/scratch
# A fixed hostname, because it ends up inside Slurm's NodeName and inside the
# URL the scheduler health-checks (http://{nodelist}:8000/health). Docker puts
# it in /etc/hosts, so it resolves from inside the container. A container-ID
# hostname would work too, but it would change every `up` and make logs from
# two sessions impossible to compare.
NODE_HOSTNAME="${EVAL360_SLURM_NODE:-eval360-node}"

# Where the cluster's scratch lives INSIDE the container. Deliberately not under
# /workspace: a venv and Slurm job logs have no business landing in the host
# working tree, and on macOS a bind-mounted venv is slow as well as wrong.
CI_SERVING_VENV=/tmp/serving-venv
CI_MODEL_PATH=/tmp/model
CI_OUTPUT_PATH=/tmp/output
CI_SLURM_LOG_DIR=/tmp/slurm-logs

_say()  { printf '\n\033[1;36m=== %s\033[0m\n' "$*"; }
_warn() { printf '\033[1;33m%s\033[0m\n' "$*" >&2; }
_die()  { printf '\033[1;31mFAIL: %s\033[0m\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
# The image tag is a hash of everything the image is built FROM.
#
# WHY, and it is not a nicety. `cmd_up` does `image_exists || build_image`. With
# a fixed tag that is a footgun with a delay fuse: edit the Dockerfile, run
# `up`, get the OLD image, and debug a cluster that does not contain your
# change. Worse in CI, where the image is built by a separate step and adopted
# here — a cached `:local` from an earlier commit is indistinguishable from a
# fresh one.
#
# Deriving the tag from the inputs makes "has anything changed?" a question with
# an answer instead of a judgement call: a different Dockerfile, a different
# dependency set, or a different lock is a different tag, and a different tag
# does not exist yet, so it gets built. `rebuild` stays as the way to re-run the
# build for an UNCHANGED tag — note that docker's own layer cache still applies,
# so it re-runs the recipe rather than fetching everything again.
#
# The hash covers the BUILD INPUTS ONLY. scripts/ci/*.sh and the project source
# are bind-mounted at run time and deliberately not baked in, so changing them
# must NOT invalidate the image — that is what makes `up` seconds rather than
# minutes.
IMAGE_INPUTS=(
    docker/slurm-ci/Dockerfile
    pyproject.toml
    uv.lock
)

_sha256_stdin() {
    # macOS has shasum, Linux has sha256sum, and a GitHub runner must compute
    # the same digits as a laptop or the CI pre-build lands on the wrong tag.
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum
    else
        shasum -a 256
    fi
}

image_tag() {
    local file
    for file in "${IMAGE_INPUTS[@]}"; do
        [ -f "$REPO_ROOT/$file" ] \
            || _die "image input missing: $file (cannot derive a reproducible tag)"
    done
    {
        # A scheme marker, so that changing WHAT is hashed forces new tags even
        # if the files themselves are untouched.
        printf 'eval360-slurm-ci/v1\n'
        for file in "${IMAGE_INPUTS[@]}"; do
            printf '### %s\n' "$file"
            cat "$REPO_ROOT/$file"
        done
    } | _sha256_stdin | cut -c1-16
}

# An explicit override still wins — it is how two people (or two agents) run
# side-by-side clusters without fighting over one tag.
IMAGE="${EVAL360_SLURM_IMAGE:-eval360-slurm-ci:$(image_tag)}"

require_docker() {
    command -v docker >/dev/null 2>&1 \
        || _die "docker not found. Install Docker Desktop (macOS) or docker.io (Linux)."
    docker info >/dev/null 2>&1 \
        || _die "docker is installed but not running. Start Docker and retry."
}

container_running() {
    [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo false)" = "true" ]
}

require_running() {
    container_running || _die "cluster is not running. Run: $0 up"
}

image_exists() {
    docker image inspect "$IMAGE" >/dev/null 2>&1
}

container_exists() {
    docker ps -aq -f "name=^${CONTAINER}$" | grep -q .
}

# Refuse to touch a container this script did not create. `docker inspect`
# prints `<no value>` for a missing label, which is exactly the answer wanted:
# anything that is not our label value is somebody else's container.
require_ours() {
    container_exists || return 0
    local owner
    owner="$(docker inspect \
        -f "{{index .Config.Labels \"$OWNER_LABEL_KEY\"}}" \
        "$CONTAINER" 2>/dev/null || true)"
    if [ "$owner" = "$OWNER_LABEL_VALUE" ]; then
        return 0
    fi
    _die "a container named '$CONTAINER' exists but was not created by this
      script (label $OWNER_LABEL_KEY=${owner:-<none>}, expected
      $OWNER_LABEL_VALUE). Refusing to remove it. Either delete it yourself
      or pick another name:  EVAL360_SLURM_CONTAINER=my-cluster $0 ...
      If you had a cluster before this check existed, this is what that
      looks like: it predates the label, so it reads as foreign. Remove
      it once with:  docker rm -f $CONTAINER"
}

# --------------------------------------------------------------------------
build_image() {
    _say "Building $IMAGE"
    # The build context is the repo root because the Dockerfile copies
    # pyproject.toml and uv.lock to pre-install dependencies into the image.
    docker build \
        -f "$REPO_ROOT/docker/slurm-ci/Dockerfile" \
        -t "$IMAGE" \
        "$REPO_ROOT"
}

cmd_up() {
    require_docker
    require_ours
    image_exists || build_image

    # `up` is idempotent by demolition: any previous container is removed rather
    # than reused. Reusing one would mean a cluster whose state depends on what
    # the last session did to it, which is the failure mode a disposable cluster
    # exists to avoid. `require_ours` above is what keeps "demolition" aimed at
    # our own container and nothing else.
    if container_exists; then
        _say "Removing previous container $CONTAINER"
        docker rm -f "$CONTAINER" >/dev/null
    fi

    # The mount point for the .eval360 tmpfs below has to exist on the host,
    # because runc mounts /workspace read-only first and then cannot create a
    # directory inside it. An EMPTY directory is invisible to git, and the
    # tmpfs keeps it empty — which is a strict improvement on today, where
    # imported-dataset jobs build real venvs in the checkout.
    mkdir -p "$REPO_ROOT/.eval360"

    _say "Starting container $CONTAINER"
    # --privileged is required, and it is the same bargain kind makes for its
    # node containers.
    #
    # WHY. slurmd initialises a cgroup context unconditionally on 23.11. With
    # `IgnoreSystemd=yes` it stops asking systemd for a scope over dbus, but it
    # still creates the scope DIRECTORY itself under
    # /sys/fs/cgroup/system.slice/, and Docker mounts /sys/fs/cgroup read-only
    # for an unprivileged container:
    #
    #   error: Could not create scope directory
    #          /sys/fs/cgroup/system.slice/<node>_slurmstepd.scope:
    #          No such file or directory
    #   error: Unable to initialize cgroup plugin
    #   error: slurmd initialization failed
    #
    # There is no narrower switch that makes cgroupfs writable — it comes with
    # privileged. The container is disposable, holds nothing but a test cluster,
    # and is built from a local image, which is the same reasoning that makes it
    # acceptable for a kind node.
    #
    # BUT PRIVILEGED APPLIES TO THE DAEMONS, NOT TO THE TESTS. --privileged is a
    # property of the container's root processes; an unprivileged process in the
    # same container holds no capabilities and cannot remount anything. So:
    #
    #   * /workspace is bind-mounted READ-ONLY, and
    #   * probe/test/shell/exec run as $RUNTIME_USER, not root (see cmd_test).
    #
    # Together those two are what make the read-only mount mean something. As
    # root in a privileged container, a test could simply `mount -o remount,rw`
    # it; as an unprivileged user it cannot, so a test — or a dependency a test
    # imports — genuinely cannot mutate the host checkout. Neither half is
    # sufficient alone, which is why both are here. Slurm jobs inherit the same
    # protection: slurmd runs each job AS THE SUBMITTING USER, so the job
    # scripts under test are unprivileged too.
    #
    # What this does NOT claim: the root daemons in this container still hold
    # privileged device and cgroup access, so a compromise of slurmctld/slurmd
    # itself is not contained by any of the above. See docs/SLURM_TEST_CLUSTER.md.
    #
    # --tmpfs /run because the pidfiles and the munge socket live there and the
    # image ships none of them.
    #
    # The two writable tmpfs mounts are the explicit escape hatches for code
    # that legitimately writes:
    #   /scratch              pytest --junitxml, $HOME, anything ad hoc
    #   /workspace/.eval360   NOT ours — `submit_imported_dataset_job` derives
    #                         repo_root from slurm_manager.py's location and
    #                         builds runner venvs under it, and repo_root is not
    #                         configurable. A tmpfs keeps those out of the
    #                         checkout instead of merely cleaning up after them.
    # mode=1777 because they are written by $RUNTIME_USER and by job steps, and
    # `exec` because docker's tmpfs default is noexec and .eval360 holds VENVS —
    # an imported-dataset job creates one and then runs its bin/python, which a
    # noexec mount turns into "Permission denied" from a path that plainly
    # exists and is +x.
    docker run -d \
        --name "$CONTAINER" \
        --hostname "$NODE_HOSTNAME" \
        --label "$OWNER_LABEL_KEY=$OWNER_LABEL_VALUE" \
        --label "dev.eval360.slurm-cluster.repo-root=$REPO_ROOT" \
        --privileged \
        -v "$REPO_ROOT:/workspace:ro" \
        -w /workspace \
        --tmpfs /run \
        --tmpfs "$CI_SCRATCH:rw,exec,mode=1777" \
        --tmpfs /workspace/.eval360:rw,exec,mode=1777 \
        "$IMAGE" >/dev/null

    _say "Bringing Slurm up inside the container"
    docker exec "$CONTAINER" bash /workspace/scripts/ci/slurm_up.sh

    _say "Preparing the fake serving venv and scratch directories"
    docker exec "$CONTAINER" bash /workspace/scripts/ci/fake_serving_venv.sh "$CI_SERVING_VENV"
    # Created by root, handed to $RUNTIME_USER: the tests and the job steps they
    # submit both run as that user now, and slurmd writes each job's stdout into
    # $CI_SLURM_LOG_DIR as the job's owner. Root-owned scratch would fail every
    # one of those with a permission error that says nothing about Slurm.
    docker exec "$CONTAINER" bash -c \
        "mkdir -p '$CI_MODEL_PATH' '$CI_OUTPUT_PATH' '$CI_SLURM_LOG_DIR' \
         && echo '{\"stub\": true}' > '$CI_MODEL_PATH/config.json' \
         && chown -R '$RUNTIME_USER' '$CI_MODEL_PATH' '$CI_OUTPUT_PATH' '$CI_SLURM_LOG_DIR'"

    _say "Ready"
    cat <<EOF
  cluster : $CONTAINER (node $NODE_HOSTNAME, image $IMAGE)
  runs as : $RUNTIME_USER (unprivileged; /workspace is read-only)
  next    : $0 probe     drive the real SlurmManager against it
            $0 shell     poke at it by hand
            $0 down      throw it away
EOF
}

# The `docker exec` flags every unprivileged entry point shares, as an array
# rather than a string because the values contain paths. HOME is /scratch
# because the `slurm` account's home is /nonexistent and anything that resolves
# ~ (pip, git, a stray cache) would fail on a read-only path.
RUNTIME_EXEC_ARGS=()
_set_runtime_exec_args() {
    RUNTIME_EXEC_ARGS=(
        -u "$RUNTIME_USER"
        -e HOME="$CI_SCRATCH"
        -e CI_SERVING_VENV="$CI_SERVING_VENV"
        -e CI_MODEL_PATH="$CI_MODEL_PATH"
        -e CI_OUTPUT_PATH="$CI_OUTPUT_PATH"
        -e CI_SLURM_LOG_DIR="$CI_SLURM_LOG_DIR"
        -e CI_SCRATCH="$CI_SCRATCH"
    )
}

cmd_probe() {
    require_docker
    require_running
    _set_runtime_exec_args
    docker exec "${RUNTIME_EXEC_ARGS[@]}" \
        -e DEPLOY_TIMEOUT_SECONDS="${DEPLOY_TIMEOUT_SECONDS:-180}" \
        "$CONTAINER" python3 /workspace/scripts/ci/slurm_probe.py
}

cmd_test() {
    require_docker
    require_running
    # No arguments means no selection, and a pytest with no selection would run
    # the entire suite inside a container built for cluster tests. Ask instead.
    [ "$#" -gt 0 ] || _die "usage: $0 test <pytest args...>   (e.g. $0 test tests/slurm -v)"
    _set_runtime_exec_args
    # `-p no:cacheprovider`: rootdir is the read-only /workspace, so pytest would
    # otherwise warn on every run that it cannot create .pytest_cache. Nothing
    # here uses --lf/--ff. PYTEST_ADDOPTS rather than an argument because the
    # caller's own args (including CI's `--override-ini=addopts=`) must stay
    # exactly what the caller wrote.
    docker exec "${RUNTIME_EXEC_ARGS[@]}" \
        -e EVAL360_IN_MEMORY_DB=true \
        -e PYTEST_ADDOPTS="-p no:cacheprovider ${PYTEST_ADDOPTS:-}" \
        "$CONTAINER" python3 -m pytest "$@"
}

cmd_shell() {
    require_docker
    require_running
    _set_runtime_exec_args
    docker exec -it "${RUNTIME_EXEC_ARGS[@]}" "$CONTAINER" bash
}

cmd_exec() {
    require_docker
    require_running
    [ "$#" -gt 0 ] || _die "usage: $0 exec <command...>"
    _set_runtime_exec_args
    docker exec "${RUNTIME_EXEC_ARGS[@]}" "$CONTAINER" "$@"
}

cmd_status() {
    require_docker
    require_running
    docker exec "$CONTAINER" bash -c '
        echo "--- sinfo ---";   sinfo -N -l
        echo "--- squeue ---";  squeue -o "%.10i %.40j %.9T %.10M %.20R"
        echo "--- node ---";    scontrol show node
    '
}

cmd_logs() {
    require_docker
    require_running
    docker exec "$CONTAINER" bash -c '
        for f in /var/log/slurm/slurmctld.log /var/log/slurm/slurmd.log \
                 /var/log/slurm/slurmdbd.log; do
            [ -f "$f" ] || continue
            echo "----- $f -----"; tail -n 100 "$f"
        done
        echo "----- slurm job output -----"
        for f in /tmp/slurm-logs/*.out; do
            [ -f "$f" ] || continue
            echo "--- $f ---"; cat "$f"
        done
        echo "----- sacct (raw, as get_job_accounting requests it) -----"
        sacct --noheader --parsable2 \
              --format=JobIDRaw,JobName%256,State,ExitCode,Reason || true
    '
}

cmd_down() {
    require_docker
    # Same ownership proof as `up`. `down` is the more dangerous of the two: it
    # is the one people run from a script, on a machine whose containers they
    # did not inventory.
    require_ours
    if container_exists; then
        docker rm -f "$CONTAINER" >/dev/null
        echo "Removed $CONTAINER"
    else
        echo "No container named $CONTAINER"
    fi
    # The mount point `up` had to create. Only if empty, and it is empty unless
    # somebody put something there while no cluster was running.
    rmdir "$REPO_ROOT/.eval360" 2>/dev/null || true
}

cmd_rebuild() {
    require_docker
    # Before the build, not after: a foreign name collision would stop `cmd_up`
    # anyway, and finding that out after several minutes of docker build is a
    # waste of the user's time.
    require_ours
    build_image
    cmd_up
}

cmd_image_tag() {
    # The one place the tag is defined, so CI can build under the name `cmd_up`
    # will look for. A second implementation on the CI side would be a second
    # thing to keep in step.
    echo "$IMAGE"
}

usage() {
    # Print the header comment block and stop at the first line that is not a
    # comment, rather than at a hard-coded line number — a fixed range silently
    # starts spilling source code into --help the moment the header grows.
    awk 'NR == 1 { next }
         /^#/    { sub(/^# ?/, ""); print; next }
         { exit }' "${BASH_SOURCE[0]}"
}

main() {
    local command="${1:-}"
    [ "$#" -gt 0 ] && shift || true
    case "$command" in
        up)      cmd_up "$@" ;;
        probe)   cmd_probe "$@" ;;
        test)    cmd_test "$@" ;;
        shell)   cmd_shell "$@" ;;
        exec)    cmd_exec "$@" ;;
        status)  cmd_status "$@" ;;
        logs)    cmd_logs "$@" ;;
        down)    cmd_down "$@" ;;
        rebuild) cmd_rebuild "$@" ;;
        image-tag) cmd_image_tag "$@" ;;
        ""|-h|--help|help) usage ;;
        *)       _warn "unknown command: $command"; usage; exit 2 ;;
    esac
}

main "$@"
