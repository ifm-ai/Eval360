# A disposable single-node Slurm cluster, for CI and for a laptop.
#
# This is the `kind` node image of this repo: the thing that makes a real Slurm
# control plane available without installing one on the machine you are sitting
# at. `scripts/ci/slurm_up.sh` configures and starts the daemons inside a
# container built from here; `scripts/slurm_cluster.sh` is the CLI that drives
# it. CI and local development run the SAME image, so a cluster that behaves one
# way on a runner and another way on a workstation is not a thing that can
# happen.
#
# WHAT IS BAKED IN HERE AND WHY. Everything slow and network-dependent: the
# Slurm packages, MariaDB, and a Python 3.14 interpreter with this project's
# pinned dependencies already installed. `slurm_cluster.sh up` should be
# seconds, and it should not fail because apt or PyPI is having a bad day.
#
# WHAT IS DELIBERATELY NOT BAKED IN. The project itself. The repo is
# bind-mounted at /workspace and reached through PYTHONPATH, so the code under
# test is whatever is checked out right now — no rebuild between edits, and
# nothing written back into the host repo. That is also why dependencies are
# installed from a throwaway copy of pyproject.toml/uv.lock below rather than
# from an editable install.
#
# EVERY EXTERNAL INPUT IS PINNED, and the image's TAG is a hash of the pinned
# inputs (`scripts/slurm_cluster.sh image-tag` computes it from this file plus
# pyproject.toml and uv.lock). Two properties follow, and both were missing
# before:
#
#   1. Editing this file changes the tag, so `cmd_up`'s `image_exists ||
#      build_image` rebuilds instead of silently reusing a stale image. The old
#      fixed `:local` tag meant a Dockerfile edit had NO effect until somebody
#      remembered `rebuild`.
#   2. A given tag denotes one set of bytes. Base image by digest, apt from a
#      frozen archive snapshot, a pinned uv, an exact CPython patch, and
#      dependencies resolved by `uv.lock` rather than live — so two builds of
#      the same tag, months apart, install the same things, and neither CI nor a
#      laptop depends on cache history or on what PyPI resolved today.
#
# Refreshing any pin (a newer snapshot, a newer uv) is a deliberate edit here,
# which is exactly the point.

# Pinned by digest, not by the mutable `24.04` tag. Refresh with:
#   docker buildx imagetools inspect ubuntu:24.04
FROM ubuntu:24.04@sha256:69cecf4bbf72d2d44a9eef1b71fb98c7fb973d78af11399deccef19beb008ad9

LABEL org.opencontainers.image.title="eval360-slurm-ci"
LABEL org.opencontainers.image.description="Single-node Slurm + slurmdbd for testing scheduler/slurm_manager.py"

# Every floating input, in one place.
#
# APT_SNAPSHOT — snapshot.ubuntu.com serves the archive AS IT WAS at a
#   timestamp, so `apt-get install slurm-wlm` resolves the same version forever
#   instead of whatever the pocket holds today. Explicit `=version` pins are the
#   obvious alternative and are worse: the Ubuntu archive keeps only the current
#   version of each package in a pocket, so a pinned version breaks the build
#   the first time a security update lands.
#   IT MUST NOT PREDATE THE BASE IMAGE DIGEST. The base already carries packages
#   from its own build date and an older snapshot cannot satisfy them:
#     libc6-dev : Depends: libc6 (= 2.39-0ubuntu8.8) but 2.39-0ubuntu8.9 is to
#     be installed ... E: Unable to correct problems, you have held broken packages
#   Bump the two pins together.
# UV_VERSION / PYTHON_VERSION — uv resolves and installs CPython, so both
#   decide what ends up in the venv.
# TEST_PINS — pytest and its closure. They are NOT project dependencies (they
#   live in the `test` extra, whose other members — datasets, evalplus,
#   transformers, pyarrow — are half a gigabyte this image has no use for), so
#   they cannot come from the default lock export. Versions are uv.lock's, and
#   the build FAILS below if they ever stop being uv.lock's.
ARG APT_SNAPSHOT=20260915T000000Z
ARG UV_VERSION=0.12.15
ARG PYTHON_VERSION=3.14.5
ARG TEST_PINS="pytest==9.1.1 pytest-asyncio==1.4.0 iniconfig==2.3.0 pluggy==1.6.0 pygments==2.20.0"

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_INSTALL_DIR=/opt/python

# CA certificates first, from the release pocket, because this is a
# chicken-and-egg: snapshot.ubuntu.com is HTTPS-only (plain HTTP 301s to HTTPS)
# and `ubuntu:24.04` ships no CA bundle at all, so apt cannot reach the snapshot
# until certificates are installed. This one package therefore still floats.
# Its INTEGRITY does not — apt verifies the archive signature either way — only
# its version does, and it is the last such package in the image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# slurm-wlm brings slurmctld/slurmd/sbatch/squeue/sinfo/scancel/sacct; slurmdbd
# and mariadb-server are what make `sacct` answer at all (see the accounting
# requirement in slurm_up.sh). util-linux supplies setpriv, used to drop to the
# munge and slurm users without a login shell. tini reaps the daemons the
# bring-up leaves running, so the container does not accumulate zombies over a
# long session.
RUN printf '%s\n' \
        'Types: deb' \
        "URIs: https://snapshot.ubuntu.com/ubuntu/${APT_SNAPSHOT}" \
        'Suites: noble noble-updates noble-security' \
        'Components: main restricted universe multiverse' \
        'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
        > /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
        slurm-wlm \
        slurmdbd \
        munge \
        mariadb-server \
        curl \
        git \
        procps \
        iproute2 \
        util-linux \
        tini \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 3.14, which this project requires (`requires-python = ">=3.14,<3.15"`)
# and Ubuntu 24.04 does not ship. uv is the least fragile way to get an exact
# patch: no third-party PPA, no source build. The installer URL carries
# ${UV_VERSION} — `https://astral.sh/uv/install.sh` without one is whatever
# astral released this morning.
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
        | env UV_INSTALL_DIR=/usr/local/bin sh \
    && uv --version \
    && uv python install "${PYTHON_VERSION}" \
    && ln -sf "$(uv python find "${PYTHON_VERSION}")" /usr/local/bin/python3.14 \
    && chmod -R a+rX /opt/python \
    && python3.14 --version

# Dependencies, baked into the image layer cache so `up` does not re-resolve
# them and does not fail because PyPI is having a bad day. Only the
# DEPENDENCIES are wanted: the `scheduler` package itself comes from the
# bind-mounted working tree at run time.
#
# INSTALLED FROM THE FROZEN LOCK, NOT RESOLVED HERE. `uv export --frozen` reads
# uv.lock and refuses to re-resolve, and `--no-emit-project` drops the `eval360`
# root so nothing tries to build this project from a source tree that is not in
# the build context. The export carries `--hash=sha256:...` for every
# requirement and `--require-hashes` enforces them, so a build either installs
# the exact artifacts the lock names or fails.
#
# Copying only pyproject.toml and uv.lock (not the source) is what keeps this
# layer valid across ordinary code edits — it rebuilds when the dependency set
# changes, and only then.
#
# THE PROJECT IS NOT INSTALLED BY NAME. An earlier version built a stand-in
# `eval360` package here and installed that, which failed in a way worth
# recording: there is an UNRELATED `eval360` on PyPI (0.10.6/0.10.7,
# Requires-Python <3.6), so when one dependency fetch came back empty pip
# backtracked looking for other "eval360" versions, found only those two, and
# reported the problem as
#
#   ERROR: Could not find a version that satisfies the requirement
#          aiohttp==3.13.2 (from eval360) (from versions: none)
#
# — a message that points at the wrong package and hides whether the real cause
# was a bad pin or a failed index fetch. A frozen export names no project at
# all, so there is nothing to backtrack on and any future failure names the
# package that actually failed.
COPY pyproject.toml uv.lock /tmp/lock/
RUN cd /tmp/lock \
    && uv export --frozen --no-dev --no-emit-project \
         --format requirements-txt -o /tmp/lock/requirements.txt \
    && echo '--- locked dependencies ---' \
    && grep -c '^[a-zA-Z]' /tmp/lock/requirements.txt \
    && uv venv --seed --python "${PYTHON_VERSION}" /opt/eval360-venv \
    && uv pip install --python /opt/eval360-venv --require-hashes \
         -r /tmp/lock/requirements.txt \
    && python3.14 -c "\
import tomllib, pathlib;\
lock = tomllib.loads(pathlib.Path('/tmp/lock/uv.lock').read_text());\
have = {p['name']: p['version'] for p in lock['package']};\
want = dict(s.split('==') for s in '${TEST_PINS}'.split());\
drift = {k: (v, have.get(k)) for k, v in want.items() if have.get(k) != v};\
assert not drift, f'TEST_PINS disagree with uv.lock, {{name: (pinned, locked)}}: {drift}'" \
    && uv pip install --python /opt/eval360-venv ${TEST_PINS} \
    && rm -rf /tmp/lock

# The venv on PATH, and the bind-mounted repo importable. Together these are why
# nothing needs installing at run time.
ENV PATH="/opt/eval360-venv/bin:${PATH}" \
    PYTHONPATH="/workspace" \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# tini as PID 1: slurm_up.sh daemonises munged, mariadbd, slurmdbd, slurmctld
# and slurmd, and something has to reap them.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sleep", "infinity"]
