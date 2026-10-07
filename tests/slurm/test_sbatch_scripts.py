"""The sbatch scripts themselves, executed on a real compute node.

What this tests:
    `scheduler/slurm/*.sh` — the shell that runs ON the node after sbatch
    accepts the job. Specifically: the base64 -> JSON -> argv reconstruction
    actually reaching the `vllm` process, `$venv_path` accepting both of the
    shapes `ModelInstance.validate_venv_path` can produce, the environment the
    scripts export around VLLM, the per-job compile-cache isolation, and the
    `$VENV/.setup_complete` idempotency sentinel, and the separate
    `.job_complete` / `.job_failed` / `.setup_complete_job` protocol in
    `output_dir` that
    `Scheduler.handle_imported_dataset_event` reads to decide whether an
    imported-dataset event failed, was preempted, or succeeded.

Why this exists:
    Nothing has ever executed these files. `tests/test_container_support.py`
    captures the argv and the `--export` string that sbatch would be handed and
    stops there, which cannot catch anything that happens after sbatch says
    "Submitted batch job N": a `$vllm_args` that decodes to the wrong argv, a
    venv shape the script's `-d`/`-f` branches do not handle, a cache directory
    that is shared between concurrent jobs, a sentinel the trap never writes.
    These scripts are the definition of code that either works on the cluster or
    does not, and a Python-level test cannot reach any of it.

    Where possible the job is submitted through the real `SlurmManager`, so the
    real `--export` string construction is exercised end to end and the script
    is fed exactly what production feeds it. Tests that need a branch production
    cannot currently reach (the scripts' own env defaults) or that must not
    write into the bind-mounted working tree (see WHY SOME IMPORTED-DATASET
    TESTS SUBMIT DIRECTLY, below) build the environment by hand, and say so at
    the call site.

How the assertions are made:
    A recording `vllm` shim (`_RECORDING_VLLM_SHIM`) is installed into a real
    venv and dumps its argv and its entire environment to a JSON file named
    after `$SLURM_JOB_ID`. That is the narrowest possible observation point: it
    sees exactly what `sbatch_script.sh` handed the process it believes is VLLM,
    and nothing between the script and it is stubbed.

WHY SOME IMPORTED-DATASET TESTS SUBMIT DIRECTLY:
    `submit_imported_dataset_job` hard-codes `repo_root` to the installed
    scheduler's parent directory, and `imported_dataset_script.sh` creates its
    runner venv at `${repo_root}/.eval360/envs/${runner_name}`. Inside the test
    cluster that path is the bind-mounted working tree, so a full-lifecycle run
    driven through `SlurmManager` would build (and later delete) a venv inside
    the developer's checkout. The full-lifecycle tests therefore submit the
    script directly with a hand-built environment pointed at a scratch
    `repo_root`; the early-failure tests, which exit before that code is
    reached, go through the real `SlurmManager` and so still cover the real
    `--export` construction for imported-dataset jobs.

Corner cases covered:
    Both `$venv_path` shapes (directory and `bin/activate` file); an argv
    containing spaces and braces surviving the base64/JSON round trip; the
    scripts' own `:-` defaults for `VLLM_ALLOW_LONG_MAX_MODEL_LEN` and
    `VLLM_LOGGING_LEVEL`, which `SlurmManager` always overrides and therefore
    nothing else can reach; two CONCURRENTLY RUNNING jobs getting disjoint
    compile-cache directories (the entire point of that block — a shared cache
    is corrupted by concurrent writers, and one job alone cannot show it);
    a `$venv_path` that does not exist AND one that exists but has no
    `bin/activate`, which take different branches to the same message; a
    benchmark that fails after setup succeeded; and a rerun finding the
    `.setup_complete` sentinel already there.

What is NOT covered, honestly:
    - `sbatch_script_container.sh` and `imported_dataset_script_container.sh`
      need pyxis/enroot, which vanilla Slurm does not ship and the test image
      does not have. `sbatch --container-image=...` is rejected outright, so
      there is no way to execute them here. They get `bash -n` and nothing more.
    - The conda variants cannot run to completion: conda is not installed in
      the image (verified: `command -v conda` is empty, `$HOME/miniconda3` and
      `$HOME/anaconda3` do not exist). What IS covered is the branch that
      matters when conda is missing — a clear failure and, for the imported
      variant, the `.job_failed` sentinel — rather than a job that hangs until
      its time limit. Everything after `conda activate` is untested.
    - VLLM itself; see `scripts/ci/fake_vllm_server.py`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from scheduler import slurm_manager

from .cluster_harness import (
    MODEL_PATH,
    SLURM_LOG_DIR,
    make_model,
    slurm_session,
    wait_for_running,
    wait_for_terminal,
    wait_until,
    reap_orphaned_stubs,
)

pytestmark = pytest.mark.cluster

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scheduler" / "slurm"

# Scratch that is deliberately NOT under the repo: the working tree is bind
# mounted into the container, and a venv has no business landing in it.
SCRATCH = Path("/tmp/eval360-sbatch-script-tests")

# The venv built by `scripts/ci/fake_serving_venv.sh` at cluster `up`. Its
# `vllm` binds 0.0.0.0:8000, which is what `imported_dataset_script.sh` health
# checks; the recording venv below deliberately does not bind anything.
SERVING_VENV = os.environ.get("CI_SERVING_VENV", "/tmp/serving-venv")

# Scripts this module executes, versus scripts it can only parse. The test
# `test_every_slurm_script_is_accounted_for` compares these against the
# directory listing so that adding a script forces a decision rather than
# silently landing outside CI.
EXECUTED_SCRIPTS = {
    "sbatch_script.sh",
    "imported_dataset_script.sh",
    # Conda variants: only the conda-is-absent branch is reachable here.
    "sbatch_script_conda.sh",
    "imported_dataset_script_conda.sh",
}
PARSE_ONLY_SCRIPTS = {
    "sbatch_script_container.sh": "needs pyxis/enroot; vanilla Slurm rejects --container-image",
    "imported_dataset_script_container.sh": "needs pyxis/enroot; vanilla Slurm rejects --container-image",
}

# A stand-in for the `vllm` console script that records what it was handed.
# Deliberately exits immediately: the enclosing `sbatch_script.sh` backgrounds
# this and then sleeps, so the job stays RUNNING either way, and not binding a
# port means several of these can run concurrently on the single test node.
# The write is atomic (write-then-rename) because the test polls for the file
# and would otherwise be able to read a half-written one.
_RECORDING_VLLM_SHIM = '''#!/usr/bin/env python3
"""Recording stand-in for `vllm`, installed into a venv by the test module."""
import json
import os
import pathlib
import sys

record_dir = pathlib.Path({record_dir!r})
record_dir.mkdir(parents=True, exist_ok=True)
name = os.environ.get("SLURM_JOB_ID") or f"nojob-{{os.getpid()}}"
payload = {{"argv": sys.argv[1:], "env": dict(os.environ)}}
tmp = record_dir / (name + ".json.tmp")
tmp.write_text(json.dumps(payload))
tmp.rename(record_dir / (name + ".json"))
'''


@dataclass(frozen=True)
class RecordingVenv:
    """A real venv whose `vllm` records argv and environment instead of serving."""

    path: Path
    records: Path

    @property
    def activate(self) -> Path:
        return self.path / "bin" / "activate"


# NOTE: this module used to carry its own autouse stub reaper. It is gone:
# conftest's `drained_cluster` reaps at setup AND teardown for every test in
# the suite, so a second copy could only ever disagree with it — and this one
# did, using the `pkill -f` form conftest documents as a trap.

@pytest.fixture(scope="module")
def scratch() -> Path:
    """A clean scratch tree per run, outside the bind-mounted repo."""
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    return SCRATCH


@pytest.fixture(scope="module")
def recording_venv(scratch: Path) -> RecordingVenv:
    """A real venv (not a hand-made directory) with a recording `vllm` on PATH.

    A real one matters: `sbatch_script.sh` sources `bin/activate` and then
    relies on both `python` and `vllm` resolving through PATH, exactly as they
    would on the cluster. A fake activate script would test a different thing.
    """
    venv = scratch / "recording-venv"
    records = scratch / "records"
    records.mkdir(parents=True, exist_ok=True)
    subprocess.run(["python3", "-m", "venv", str(venv)], check=True, capture_output=True)
    shim = venv / "bin" / "vllm"
    shim.write_text(_RECORDING_VLLM_SHIM.format(record_dir=str(records)))
    shim.chmod(0o755)
    return RecordingVenv(path=venv, records=records)


@pytest.fixture(scope="module")
def prepared_runner_repo(scratch: Path) -> tuple[Path, str]:
    """A `repo_root` whose runner venv already exists and is marked complete.

    `imported_dataset_script.sh` skips its whole setup block when
    `${VENV}/.setup_complete` is present. Pre-building that state is how the
    idempotency branch is reached — and it is the state every rerun of a real
    imported-dataset job finds.
    """
    runner = "ciprepared"
    repo_root = scratch / "prepared-repo"
    venv = repo_root / ".eval360" / "envs" / runner
    venv.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["python3", "-m", "venv", str(venv)], check=True, capture_output=True)
    (venv / ".setup_complete").touch()
    return repo_root, runner


# ----------------------------------------------------------------------------
# Helpers (local to this module by design: tests/slurm/cluster_harness.py is
# shared with other suites and must not grow script-specific knowledge).
# ----------------------------------------------------------------------------


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def _vllm_args_b64(args: list[str]) -> str:
    """Encode argv the way `SlurmManager` encodes it, for direct submissions."""
    return base64.b64encode(json.dumps(args).encode()).decode()


def _log_path(job_id: int) -> Path:
    return Path(SLURM_LOG_DIR) / f"slurm-{job_id}.out"


async def _wait_for_log(job_id: int, needle: str, *, timeout: float = 90) -> str:
    """Wait for `needle` to appear in a job's stdout, returning the whole log."""

    async def _ready():
        path = _log_path(job_id)
        if not path.exists():
            return None
        text = path.read_text(errors="replace")
        return text if needle in text else None

    return await wait_until(
        _ready, timeout=timeout, what=f"{needle!r} in the log of job {job_id}"
    )


async def _wait_for_records(venv: RecordingVenv, job_ids, *, timeout: float = 120) -> dict:
    """Wait until every given job's `vllm` invocation has been recorded."""

    async def _ready():
        found = {}
        for job_id in job_ids:
            path = venv.records / f"{job_id}.json"
            if not path.exists():
                return None
            found[job_id] = json.loads(path.read_text())
        return found

    return await wait_until(
        _ready,
        timeout=timeout,
        what=f"recorded vllm invocations for job(s) {sorted(job_ids)}",
    )


async def _wait_for_file(path: Path, *, timeout: float = 30) -> Path:
    async def _ready():
        return path if path.exists() else None

    return await wait_until(_ready, timeout=timeout, what=f"{path} to appear")


async def _submit_script(
    script: Path,
    exports: dict[str, str],
    *,
    job_name: str,
    time_limit: str = "0:10:00",
) -> int:
    """sbatch a script with a hand-built `--export`, returning the job ID.

    Used ONLY where driving `SlurmManager` cannot reach the branch under test.
    Every caller explains why at the call site.
    """
    export_arg = ",".join(f"{key}={value}" for key, value in exports.items())
    command = [
        "sbatch", "--parsable",
        f"--job-name={job_name}",
        "--partition=ci",
        "--nodes=1", "--ntasks=1",
        "--gres=gpu:1",
        "--cpus-per-task=1",
        f"--time={time_limit}",
        f"--output={SLURM_LOG_DIR}/slurm-%j.out",
        f"--export={export_arg}",
        str(script),
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"sbatch failed: {stderr.decode().strip()}")
    return int(stdout.decode().strip().split(";")[0])


async def _reset_port_8000(*, timeout: float = 60) -> None:
    """Reap leaked stub servers so every imported-dataset test starts the same.

    `imported_dataset_script.sh` gates on `http://localhost:8000/health`, and
    the stub serving venv binds that one port on the single test node.

    THE LEAK, measured on this cluster: a stub server started by a job OUTLIVES
    the job. After a job that had run to COMPLETED, `ss -ltnp` still showed
    `0.0.0.0:8000 users:(("python3",pid=235))` — the `fake_vllm_server.py` the
    script had backgrounded, with the batch script long gone. That is what
    `ProctrackType=proctrack/linuxproc` + `TaskPlugin=task/none` buys: the test
    cluster runs without cgroups (see docs/SLURM_TEST_CLUSTER.md, gotcha 2), so
    Slurm has no container to tear down and a detached descendant survives.

    Left alone that makes these tests order-dependent in the worst way: the
    second imported test's own stub finds the port taken and exits 0 (by
    design — see fake_vllm_server.py), while the leaked server from the
    PREVIOUS test answers the health check, so the job proceeds against a
    zombie. Killing the stub by name and then waiting for the port gives every
    test the same starting state. It is safe here because `fake_vllm_server.py`
    is only ever started by this suite's jobs, and the autouse `drained_cluster`
    fixture has already emptied the queue.
    """
    # ONE kill implementation, shared with conftest's `drained_cluster`.
    # Deliberately not `pkill -f fake_vllm_server.py`: `-f` matches the whole
    # command line, so any shell mentioning the script kills itself — see the
    # note on `reap_orphaned_stubs`. What this function adds over conftest's
    # reaping is the WAIT below: conftest runs between tests, and these tests
    # need the port verified free part-way through one.
    reap_orphaned_stubs()

    async def _free():
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # SO_REUSEADDR mirrors `ThreadingHTTPServer.allow_reuse_address`, so
        # this probe answers the question that actually matters — "could the
        # stub bind?" — rather than a stricter one. Without it the health
        # checks' TIME_WAIT sockets hold port 8000 for a further 60s after the
        # server itself is gone, and the probe reports a port in use that the
        # stub would have bound without complaint.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", 8000))
        except OSError:
            return None
        finally:
            probe.close()
        return True

    await wait_until(_free, timeout=timeout, what="port 8000 to be free on the node")


def _conda_model(name: str, conda_env: str = "eval360-ci-absent"):
    """A model whose deployment takes the conda branch of `update_allocation`.

    `ModelInstance` — unlike `ModelSpec` — has no mutual-exclusion validator for
    venv_path/conda_env/container_image, and `uses_conda` is simply
    `conda_env is not None`. Clearing venv_path keeps the instance honest about
    which branch it selects.
    """
    model = make_model(name)
    model.venv_path = None
    model.conda_env = conda_env
    return model


def _imported_exports(
    *,
    venv_path: str,
    repo_root: Path,
    runner_name: str,
    output_dir: Path,
    setup_script: str,
    benchmark_script: str,
    vllm_args: list[str] | None = None,
    max_time_to_deploy: int = 90,
) -> dict[str, str]:
    """The exact variable set `submit_imported_dataset_job` exports.

    Kept in one place so a drift between this and `slurm_manager.py` shows up as
    a failing script rather than as a test that quietly stopped matching
    production. The values differ from production only in `repo_root` (see the
    module docstring) and in being chosen to be observable.
    """
    return {
        "venv_path": venv_path,
        "model_path": MODEL_PATH,
        "vllm_args": _vllm_args_b64(vllm_args or []),
        "vllm_allow_long_max_model_len": "1",
        "bootstrap_python": shutil.which("python3") or "python3",
        "runner_name": runner_name,
        "repo_root": str(repo_root),
        "max_time_to_deploy": str(max_time_to_deploy),
        "setup_script_b64": _b64(setup_script),
        "benchmark_script_b64": _b64(benchmark_script),
        "output_dir": str(output_dir),
    }


# ----------------------------------------------------------------------------
# sbatch_script.sh — the serving script
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("venv_shape", ["directory", "activate-file"])
async def test_vllm_argv_round_trip_reaches_the_process(recording_venv, venv_shape):
    """`$vllm_args` decodes to the argv VLLM is actually exec'd with.

    Three layers have to agree for this to hold, and only the last of them has
    ever been executed by a test: `update_allocation` splits and quote-strips
    each configured flag and base64-encodes the JSON list; sbatch carries that
    string through `--export`; the script base64-decodes it, json-loads it, and
    prepends `["vllm", "serve", $model_path]`. The recording shim is handed the
    result, so what is asserted here is the final argv and not an intermediate.

    BOTH venv shapes are parametrised because `ModelInstance.validate_venv_path`
    only requires an absolute path — it does not normalise a directory to its
    activate script or the reverse — so a config may legitimately carry either,
    and `sbatch_script.sh` has a separate branch for each (`-d` appends
    `/bin/activate`, `-f` uses the path as-is). A regression in either branch is
    a job that dies on the node minutes after submission.

    The `--chat-template '{{ x }}'` argument is deliberately awkward: it carries
    a space, braces and quotes, which is what makes it evidence that the
    transport is base64/JSON and not naive string splitting.
    """
    venv_path = (
        str(recording_venv.path)
        if venv_shape == "directory"
        else str(recording_venv.activate)
    )
    async with slurm_session() as manager:
        model = make_model(
            f"ci-argv-{venv_shape}",
            venv_path=venv_path,
            vllm_cli_args=[
                "--max-model-len 256",
                "--tensor-parallel-size 1",
                "--enable-prefix-caching",
                "--chat-template '{{ x }}'",
            ],
        )
        await manager.update_allocation([(model, 1)], [])
        jobs = await wait_for_running(manager, 1)
        records = await _wait_for_records(recording_venv, [jobs[0]["job_id"]])
        record = records[jobs[0]["job_id"]]

        assert record["argv"] == [
            "serve",
            MODEL_PATH,
            "--max-model-len", "256",
            "--tensor-parallel-size", "1",
            "--enable-prefix-caching",
            "--chat-template", "{{ x }}",
            "--served-model-name", model.name,
        ]
        # The other half of the claim for this branch: the venv was activated,
        # which is why `vllm` resolved at all.
        assert record["env"]["VIRTUAL_ENV"] == str(recording_venv.path)


async def test_vllm_environment_carries_configured_values(recording_venv):
    """`VLLM_ALLOW_LONG_MAX_MODEL_LEN` and `VLLM_LOGGING_LEVEL` reach VLLM.

    Both are set with non-default values so that a script which hard-coded the
    defaults — or a `--export` that dropped the pair — fails here. `int(bool)`
    is what `update_allocation` sends, so the string the process sees is "0"
    or "1" and nothing else.
    """
    async with slurm_session() as manager:
        model = make_model("ci-env-exports", venv_path=str(recording_venv.path))
        model.allow_long_max_model_len = False
        model.vllm_logging_level = "DEBUG"

        await manager.update_allocation([(model, 1)], [])
        jobs = await wait_for_running(manager, 1)
        record = (await _wait_for_records(recording_venv, [jobs[0]["job_id"]]))[
            jobs[0]["job_id"]
        ]

        assert record["env"]["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] == "0"
        assert record["env"]["VLLM_LOGGING_LEVEL"] == "DEBUG"


async def test_script_supplies_its_own_env_defaults(recording_venv):
    """The `:-0` / `:-WARNING` fallbacks in the script are real.

    DIRECT SUBMISSION, and it has to be: `update_allocation` always exports both
    variables, so no configuration can reach this branch. It is nonetheless the
    script's documented contract, and the branch any other caller (a hand-run
    job, an older scheduler, a future container path) lands in. A test at the
    Python layer cannot see it at all.
    """
    job_id = await _submit_script(
        SCRIPT_DIR / "sbatch_script.sh",
        {
            "venv_path": str(recording_venv.path),
            "model_path": MODEL_PATH,
            "vllm_args": _vllm_args_b64([]),
        },
        job_name="ci-script-env-defaults",
    )
    record = (await _wait_for_records(recording_venv, [job_id]))[job_id]
    assert record["env"]["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] == "0"
    assert record["env"]["VLLM_LOGGING_LEVEL"] == "WARNING"


async def test_concurrent_jobs_get_disjoint_compile_caches(recording_venv):
    """Two jobs running AT THE SAME TIME get their own cache directories.

    This block exists in the script to stop concurrent jobs corrupting a shared
    torch/triton/vllm compile cache, so a single job proves nothing: the claim
    is about two, and the two must overlap in time. `wait_for_running(manager,
    2)` is what establishes the overlap — it only returns when Slurm reports
    both replicas RUNNING in the same poll.

    The directories are asserted to EXIST as well as to differ, because the
    script's `mkdir -p` is what makes the exported paths usable; exporting a
    path nothing created would leave VLLM to fall back to its default shared
    cache and silently reintroduce the corruption this guards against.
    """
    async with slurm_session() as manager:
        model = make_model("ci-cache-isolation", venv_path=str(recording_venv.path))
        await manager.update_allocation([(model, 2)], [])
        jobs = await wait_for_running(manager, 2)
        job_ids = [job["job_id"] for job in jobs]
        assert len(set(job_ids)) == 2

        records = await _wait_for_records(recording_venv, job_ids)

        roots: list[str] = []
        for job_id in job_ids:
            env = records[job_id]["env"]
            cache = env["VLLM_CACHE_ROOT"]
            inductor = env["TORCHINDUCTOR_CACHE_DIR"]
            triton = env["TRITON_CACHE_DIR"]

            # Job-specific by construction: SLURM_JOB_ID is in the path.
            assert str(job_id) in cache, f"{cache} is not job-specific"
            assert cache.endswith("/vllm")
            assert inductor.endswith("/torchinductor")
            assert triton.endswith("/triton")
            assert {Path(cache).parent, Path(inductor).parent, Path(triton).parent} == {
                Path(cache).parent
            }, "the three caches must share one per-job root"

            for directory in (cache, inductor, triton):
                assert Path(directory).is_dir(), f"{directory} was exported but not created"
            roots.append(str(Path(cache).parent))

        assert roots[0] != roots[1], (
            f"concurrent jobs shared a compile-cache root: {roots}"
        )


@pytest.mark.parametrize("venv_kind", ["missing-path", "directory-without-activate"])
async def test_missing_venv_fails_the_job_with_the_documented_message(
    scratch, venv_kind
):
    """A bad `$venv_path` fails fast and loudly instead of hanging.

    Both parametrisations reach the same `exit 1` down different branches: a
    path that does not exist fails the `-f` test directly, while a directory
    that exists but has no `bin/activate` first takes the `-d` branch, which is
    why the expected message differs by that suffix. Getting this wrong is not
    a crash — under `set -e` without the explicit check the script would carry
    on to `python -` and either die with something unrelated or sit in
    `sleep 86400`, holding a GPU for the job's full time limit while the
    scheduler waits for a health endpoint that will never answer.

    The bounded wait for a terminal accounting row is itself the anti-hang
    assertion: a job that hung would fail here on the timeout.
    """
    if venv_kind == "missing-path":
        venv_path = scratch / "no-such-venv"
        expected = f"Missing venv activation script: {venv_path}"
    else:
        venv_path = scratch / "venv-without-activate"
        venv_path.mkdir(exist_ok=True)
        expected = f"Missing venv activation script: {venv_path}/bin/activate"

    async with slurm_session() as manager:
        model = make_model(f"ci-badvenv-{venv_kind}", venv_path=str(venv_path))
        await manager.update_allocation([(model, 1)], [])

        submitted = manager.get_submitted_jobs()
        assert len(submitted) == 1
        job_id = submitted[0].job_id

        outcome = await wait_for_terminal(manager, job_id, timeout=120)
        assert outcome.state == "FAILED", f"expected FAILED, got {outcome.raw_state!r}"
        assert (outcome.exit_code, outcome.signal) == (1, 0)
        assert expected in await _wait_for_log(job_id, "Missing venv activation script")


# ----------------------------------------------------------------------------
# imported_dataset_script.sh — the sentinel protocol
# ----------------------------------------------------------------------------


async def test_imported_job_writes_complete_sentinel_and_runs_both_scripts(
    scratch, tmp_path
):
    """The success path end to end: setup, health gate, benchmark, sentinels.

    Direct submission with a scratch `repo_root` — see the module docstring for
    why this one cannot go through `SlurmManager`.

    What each assertion is for:
      - the marker files prove `setup_script_b64` and `benchmark_script_b64`
        survived base64 intact, which is why their content carries quotes, an
        unexpanded `$`, a newline and a non-ASCII character;
      - `$VENV` inside the setup script proves the script exports it, which the
        real BFCL setup script depends on to install into the right venv;
      - `$VIRTUAL_ENV` inside the benchmark script proves the runner venv was
        activated BEFORE the benchmark ran, not after;
      - `.setup_complete_job` is what tells the scheduler the job got past
        deployment, and it is the discriminator between "failed during setup"
        and "preempted" in `handle_imported_dataset_event`;
      - `.job_failed` must be ABSENT, which is the only check that the EXIT
        trap does not fire on a clean exit. A trap that fired unconditionally
        would mark every successful run as failed.
    """
    await _reset_port_8000()
    repo_root = scratch / "fresh-repo"
    runner = "cifresh"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    marker_dir = scratch / "markers-success"
    marker_dir.mkdir()

    # Single-quoted in the shell so the awkward characters survive to the file
    # rather than being expanded by the very shell that is meant to be
    # transporting them verbatim.
    setup_script = (
        f"""printf '%s\\n' 'setup "quoted" $NOT_EXPANDED — ünïcode' > {marker_dir}/setup\n"""
        f"""printf '%s\\n' "$VENV" >> {marker_dir}/setup\n"""
    )
    benchmark_script = (
        f"""printf '%s\\n' 'bench "quoted" $NOT_EXPANDED — ünïcode' > {marker_dir}/bench\n"""
        f"""printf '%s\\n' "$VIRTUAL_ENV" >> {marker_dir}/bench\n"""
    )

    job_id = await _submit_script(
        SCRIPT_DIR / "imported_dataset_script.sh",
        _imported_exports(
            venv_path=SERVING_VENV,
            repo_root=repo_root,
            runner_name=runner,
            output_dir=output_dir,
            setup_script=setup_script,
            benchmark_script=benchmark_script,
            vllm_args=["--max-model-len", "256", "--served-model-name", "ci-imported"],
        ),
        job_name="ci-script-imported-success",
    )

    async with slurm_session() as manager:
        outcome = await wait_for_terminal(manager, job_id, timeout=180)
    assert outcome.state == "COMPLETED", f"expected COMPLETED, got {outcome.raw_state!r}"
    assert (outcome.exit_code, outcome.signal) == (0, 0)

    runner_venv = repo_root / ".eval360" / "envs" / runner
    assert (runner_venv / ".setup_complete").exists(), "setup sentinel was not written"

    setup_lines = (marker_dir / "setup").read_text().splitlines()
    assert setup_lines[0] == 'setup "quoted" $NOT_EXPANDED — ünïcode'
    assert setup_lines[1] == str(runner_venv), "$VENV was not exported to the setup script"

    bench_lines = (marker_dir / "bench").read_text().splitlines()
    assert bench_lines[0] == 'bench "quoted" $NOT_EXPANDED — ünïcode'
    assert bench_lines[1] == str(runner_venv), (
        "the benchmark ran outside the runner venv"
    )

    assert (output_dir / ".setup_complete_job").exists()
    assert (output_dir / ".job_complete").exists()
    assert not (output_dir / ".job_failed").exists(), (
        "the EXIT trap wrote .job_failed for a successful job"
    )

    # The script's own copy of the base64 argv reconstruction, observed at the
    # stub serving venv rather than at the recording one.
    log = _log_path(job_id).read_text(errors="replace")
    assert (
        "[fake-vllm] argv as received : "
        f"['serve', '{MODEL_PATH}', '--max-model-len', '256', "
        "'--served-model-name', 'ci-imported']"
    ) in log


async def test_imported_job_failure_writes_failed_sentinel(
    prepared_runner_repo, tmp_path
):
    """A benchmark that exits non-zero produces `.job_failed` and no `.job_complete`.

    This is the discrimination `Scheduler.handle_imported_dataset_event` makes
    after the job leaves the queue: with `.setup_complete_job` present, the
    presence of `.job_failed` means "failed internally" and its ABSENCE means
    "preempted, reschedule it". If the trap did not fire, a crashed evaluation
    would be requeued forever; if it fired on preemption, a preempted one would
    be marked failed. Only executing the script can tell the two apart.

    Uses the prepared repo so the run skips venv creation — this test is about
    the trap, not about setup.
    """
    await _reset_port_8000()
    repo_root, runner = prepared_runner_repo
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    job_id = await _submit_script(
        SCRIPT_DIR / "imported_dataset_script.sh",
        _imported_exports(
            venv_path=SERVING_VENV,
            repo_root=repo_root,
            runner_name=runner,
            output_dir=output_dir,
            setup_script="true",
            benchmark_script="echo 'benchmark exploding on purpose' >&2\nexit 7\n",
        ),
        job_name="ci-script-imported-failure",
    )

    async with slurm_session() as manager:
        outcome = await wait_for_terminal(manager, job_id, timeout=180)
    assert outcome.state == "FAILED", f"expected FAILED, got {outcome.raw_state!r}"
    # The benchmark's own status, propagated by `set -e` and preserved by the
    # trap's `status=$?`. A trap that clobbered it would report 0 here.
    assert (outcome.exit_code, outcome.signal) == (7, 0)

    assert (output_dir / ".setup_complete_job").exists(), (
        "the job never got past deployment; this test would be measuring the "
        "wrong failure"
    )
    assert (output_dir / ".job_failed").exists(), "the EXIT trap did not fire"
    assert not (output_dir / ".job_complete").exists()


async def test_imported_setup_is_skipped_when_the_sentinel_exists(
    prepared_runner_repo, tmp_path, scratch
):
    """A rerun finding `.setup_complete` does not rebuild the venv.

    The sentinel is the script's idempotency guarantee, and it is not cosmetic:
    the setup block begins with `rm -rf "$VENV"`, so a sentinel check that
    failed would delete and reinstall the runner environment on every rerun —
    minutes of wasted GPU time per resumed job, and a window in which the venv
    does not exist at all.

    Proven two ways: a setup script that WOULD leave a marker if it ran, and a
    canary file inside the existing venv that `rm -rf "$VENV"` would take with
    it. The canary is the one that catches a sentinel check that runs setup
    anyway but happens to leave no marker.
    """
    await _reset_port_8000()
    repo_root, runner = prepared_runner_repo
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    marker = scratch / "setup-should-not-run"
    assert not marker.exists()
    canary = repo_root / ".eval360" / "envs" / runner / ".canary"
    canary.write_text("survived")

    job_id = await _submit_script(
        SCRIPT_DIR / "imported_dataset_script.sh",
        _imported_exports(
            venv_path=SERVING_VENV,
            repo_root=repo_root,
            runner_name=runner,
            output_dir=output_dir,
            setup_script=f"touch {marker}\n",
            benchmark_script="true",
        ),
        job_name="ci-script-imported-idempotent",
    )

    async with slurm_session() as manager:
        outcome = await wait_for_terminal(manager, job_id, timeout=180)
    assert outcome.state == "COMPLETED", f"expected COMPLETED, got {outcome.raw_state!r}"
    assert (output_dir / ".job_complete").exists()
    assert not marker.exists(), (
        "the setup script ran again despite .setup_complete being present"
    )
    assert canary.exists(), "the existing runner venv was deleted and rebuilt"


async def test_imported_job_missing_venv_writes_failed_sentinel(scratch, tmp_path):
    """An imported job with a bad `$venv_path` fails and leaves `.job_failed`.

    Submitted through the REAL `submit_imported_dataset_job`, so this is also
    the coverage for that method's `--export` string being something the script
    can consume — the script reads eleven variables out of it and an error in
    any one of them shows up here. It is safe to drive for real because the
    script exits at the venv check, before the block that would create a venv
    under the bind-mounted `repo_root`.

    `output_dir` exists, which is the case that matters: the trap's `touch`
    cannot create the directory itself. The absent-directory case is a known
    bug — docs/SLURM_TEST_CLUSTER.md, "Bugs and hazards this found" item 5 —
    pinned by test_imported_dataset_cluster.py::
    test_job_failed_sentinel_survives_an_absent_output_dir.
    """
    venv_path = scratch / "imported-no-such-venv"
    async with slurm_session() as manager:
        model = make_model("ci-imported-badvenv", venv_path=str(venv_path))
        job_id = await manager.submit_imported_dataset_job(
            event_uuid="00000000deadbeef",
            model_instance=model,
            runner_name="cibadvenv",
            setup_script="true",
            benchmark_script="true",
            output_dir=tmp_path,
        )

        outcome = await wait_for_terminal(manager, job_id, timeout=120)
        assert outcome.state == "FAILED", f"expected FAILED, got {outcome.raw_state!r}"
        assert (outcome.exit_code, outcome.signal) == (1, 0)
        assert f"Missing venv activation script: {venv_path}" in await _wait_for_log(
            job_id, "Missing venv activation script"
        )
        await _wait_for_file(tmp_path / ".job_failed")
        assert not (tmp_path / ".job_complete").exists()
        assert not (tmp_path / ".setup_complete_job").exists()


# ----------------------------------------------------------------------------
# The conda variants — only the conda-is-absent branch is reachable here
# ----------------------------------------------------------------------------


async def test_conda_serving_script_fails_clearly_when_conda_is_absent():
    """`sbatch_script_conda.sh` exits 1 with an actionable message, not a hang.

    This is the whole of what the test image can honestly cover for the conda
    variant, and it is the branch worth covering: everything after
    `conda activate` needs a conda that is not installed here. The failure being
    IMMEDIATE and NAMED is what stops a mis-configured conda job from occupying
    a node until its time limit while the scheduler health-checks an endpoint
    that will never exist.

    Driven through `update_allocation` so the conda `--export` (which carries
    `conda_env` and, notably, no `venv_path`) is the real one.
    """
    async with slurm_session() as manager:
        model = _conda_model("ci-conda-serving")
        assert model.uses_conda

        await manager.update_allocation([(model, 1)], [])
        submitted = manager.get_submitted_jobs()
        assert len(submitted) == 1
        job_id = submitted[0].job_id

        outcome = await wait_for_terminal(manager, job_id, timeout=120)
        assert outcome.state == "FAILED", f"expected FAILED, got {outcome.raw_state!r}"
        assert (outcome.exit_code, outcome.signal) == (1, 0)
        assert "conda not found" in await _wait_for_log(job_id, "conda not found")


async def test_conda_imported_script_fails_and_writes_failed_sentinel(tmp_path):
    """The conda imported variant fails clearly AND still writes `.job_failed`.

    The sentinel matters more here than the message: the conda script carries
    its own copy of the EXIT trap, and a copy is exactly the kind of thing that
    drifts from the original. Without it, a conda-mode imported job that died
    before deployment would look to the scheduler like a preemption and be
    rescheduled indefinitely.
    """
    async with slurm_session() as manager:
        model = _conda_model("ci-conda-imported")
        job_id = await manager.submit_imported_dataset_job(
            event_uuid="00000000cafebabe",
            model_instance=model,
            runner_name="ciconda",
            setup_script="true",
            benchmark_script="true",
            output_dir=tmp_path,
        )

        outcome = await wait_for_terminal(manager, job_id, timeout=120)
        assert outcome.state == "FAILED", f"expected FAILED, got {outcome.raw_state!r}"
        assert (outcome.exit_code, outcome.signal) == (1, 0)
        assert "conda not found" in await _wait_for_log(job_id, "conda not found")
        await _wait_for_file(tmp_path / ".job_failed")
        assert not (tmp_path / ".job_complete").exists()


# ----------------------------------------------------------------------------
# Static coverage of every script, including the ones that cannot run here
# ----------------------------------------------------------------------------


def _scripts_on_disk() -> list[str]:
    return sorted(path.name for path in SCRIPT_DIR.glob("*.sh"))


@pytest.mark.parametrize("script_name", _scripts_on_disk())
def test_every_slurm_script_parses(script_name):
    """`bash -n` every script, derived from the directory rather than a list.

    For the container variants this is the ONLY execution-adjacent check that
    exists anywhere — they need pyxis, so nothing in this repo can run them —
    and a syntax error in one of them is otherwise discovered by a job dying on
    a production cluster. It is a low bar; it is also strictly more than zero.
    """
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT_DIR / script_name)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"{script_name} is not valid bash:\n{result.stderr}"


def test_every_slurm_script_is_accounted_for():
    """Adding a script must force a decision about testing it.

    The lists are compared against the directory listing, so a seventh script
    appearing with no coverage fails here instead of quietly joining the set of
    files nothing has ever run.
    """
    on_disk = set(_scripts_on_disk())
    declared = EXECUTED_SCRIPTS | set(PARSE_ONLY_SCRIPTS)
    assert on_disk == declared, (
        "scheduler/slurm/ changed. Scripts with no coverage decision: "
        f"{sorted(on_disk - declared)}; declared but absent: "
        f"{sorted(declared - on_disk)}. Either exercise the new script in this "
        "module or add it to PARSE_ONLY_SCRIPTS with the reason it cannot run."
    )
    # The scripts SlurmManager can actually select must all be in one bucket.
    for name in ("sbatch_script.sh", "imported_dataset_script.sh"):
        assert name in EXECUTED_SCRIPTS


def test_slurm_manager_submits_the_scripts_this_module_covers():
    """The files under test are the files production submits, by name.

    A test module that ran copies, or that kept covering a script nobody
    submits any more, would be green while the real scheduler did something
    else. `SlurmManager` builds every script path as
    `Path(slurm_manager.__file__).parent / "slurm/<name>"`, so both halves —
    the directory and each filename — are pinned here.
    """
    manager_dir = Path(slurm_manager.__file__).resolve().parent
    assert manager_dir / "slurm" == SCRIPT_DIR

    source = Path(slurm_manager.__file__).read_text()
    for name in sorted(EXECUTED_SCRIPTS | set(PARSE_ONLY_SCRIPTS)):
        assert f"slurm/{name}" in source, (
            f"{name} exists in scheduler/slurm/ and is covered here, but "
            "slurm_manager.py never submits it"
        )
