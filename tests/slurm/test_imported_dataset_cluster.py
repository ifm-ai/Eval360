"""The imported-dataset job path, run against a real Slurm cluster.

What this tests:
    `SlurmManager.submit_imported_dataset_job` and the three polling helpers the
    scheduler drives it with — `get_job_node`, `wait_for_vllm_health` and
    `wait_for_job_completion` — plus the runner contract those helpers exist to
    serve: `scheduler/slurm/imported_dataset_script.sh` really creating the
    runner venv, really running `build_setup_script`, really running
    `build_benchmark_script` against a server on localhost:8000, and really
    writing the `.setup_complete_job` / `.job_complete` / `.job_failed`
    sentinels that `Scheduler.handle_imported_dataset_event` branches on.

Why this exists:
    All four of these methods are stubbed wholesale in `tests/fake_slurm.py`.
    `FakeSlurmManager.get_job_node` reads an attribute off an in-memory object;
    `wait_for_vllm_health` consults a set of "healthy" serving keys;
    `wait_for_job_completion` checks a string field. None of them run squeue,
    none of them open a socket, and none of them can notice that the real
    implementations parse `squeue --format=%N` output or that the real health
    check is an HTTP GET. `submit_imported_dataset_job` is faked even harder —
    the fake never builds an sbatch argv at all, so `imported_dataset_script.sh`
    has never been executed by any test, and neither has the `--export=` payload
    that carries the two base64 scripts, `bootstrap_python` and `repo_root` to
    the compute node.

    So every existing assertion about imported datasets is an assertion about
    the fake. This module is the other half.

Corner cases covered:
    A job that is PENDING when `get_job_node` first looks (its `%N` is measured
    here, not assumed) and only later gets a node; the returned node being a
    real resolvable host that the health URL can be built from; health returning
    True against a server that is actually listening and False — bounded, not
    hanging — against a node where nothing answers on :8000; completion
    detection for a job that exits on its own AND for one that is cancelled,
    distinguished by what sacct says afterwards; a full lifecycle whose setup
    and benchmark scripts both really run; a benchmark that exits non-zero
    producing `.job_failed` after `.setup_complete_job`; and a setup that exits
    non-zero producing `.job_failed` with NO `.setup_complete_job` — which is
    the only thing that tells the scheduler "failed during setup" apart from
    "failed internally".

Not covered, and why:
    The real runner is BFCL, whose `build_setup_script` is
    `pip install bfcl-eval` — a package index the test container cannot reach.
    `CIImportedDatasetRunner` below stands in for it: a genuine
    `ImportedDatasetRunnerBase` whose scripts the compute node really executes,
    substituted at the same narrow point `fake_vllm_server.py` substitutes for
    VLLM. What it cannot exercise is anything specific to BFCL's own output
    format, which is parser logic and belongs in a pure-Python test anyway.

    Container and conda variants (`imported_dataset_script_container.sh`,
    `imported_dataset_script_conda.sh`) are out of reach for the reasons
    docs/SLURM_TEST_CLUSTER.md gives: pyxis/enroot is not part of vanilla Slurm,
    and there is no conda in the image.

Cleanup, all three kinds:
    `tests/slurm/conftest.py` already cancels every queued job after each test
    and waits for the queue to drain, so nothing here needs to repeat that. Two
    things it cannot know about are handled locally: the runner venv, which the
    job writes into the working tree (see `runner_name`), and stub servers that
    outlive their job on a cgroup-less cluster (see
    conftest's `drained_cluster` fixture).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from scheduler.grader.base import Score
from scheduler.imported_dataset import registry as runner_registry
from scheduler.imported_dataset.base import (
    ImportedDatasetResultError,
    ImportedDatasetRunnerBase,
)
from scheduler.slurm_manager import SlurmJobVanished
from scheduler.task import ImportedDatasetConfig, ImportedDatasetTask

from .cluster_harness import (
    CLUSTER_GPUS,
    make_model,
    scancel,
    slurm_session,
    submit_raw,
    wait_for_running,
    wait_for_terminal,
    wait_until,
)

pytestmark = pytest.mark.cluster

# `submit_imported_dataset_job` computes `repo_root` as the parent of the
# `scheduler` package and exports it to the job, which uses it to place the
# runner venv at <repo_root>/.eval360/envs/<runner_name>. It is not
# configurable, so the tests have to know the same path in order to clean up.
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_ENV_ROOT = REPO_ROOT / ".eval360" / "envs"

# Every wait here is bounded well below the module's ~2.5 minute budget. These
# are ceilings that turn a hang into a named failure, not expected durations:
# the cluster normally starts a job in about two seconds.
NODE_TIMEOUT = 90.0
COMPLETION_TIMEOUT = 180.0
# `wait_for_vllm_health` and `get_job_node` default to a 10s poll interval,
# which is right for a real VLLM load and far too slow for a suite that gates
# every PR. Both signatures take an override; the scheduler does not pass one.
FAST_POLL = 0.5


# ---------------------------------------------------------------------------
# A runner whose scripts actually execute
# ---------------------------------------------------------------------------

# `.replace` rather than `.format`/f-strings on purpose: these are bash scripts
# full of `${...}`, and every brace would otherwise need doubling — which is a
# reliable way to ship a script that is subtly not the one you read.
_OUTPUT_DIR = "__OUTPUT_DIR__"


def _record(filename: str) -> str:
    """A bash fragment that dumps the venv-visible environment to a file."""
    return (
        "{\n"
        '  echo "venv=${VENV:-}"\n'
        '  echo "virtual_env=${VIRTUAL_ENV:-}"\n'
        '  echo "which_python=$(command -v python || true)"\n'
        f'}} > "__OUTPUT_DIR__/{filename}"\n'
    )


_PREAMBLE = 'set -eu\nmkdir -p "__OUTPUT_DIR__"\n'

# base.py promises build_setup_script that $VENV exists and is NOT yet
# activated. Recording $VIRTUAL_ENV and the resolved `python` at this moment is
# what lets the test check the second half of that promise, which is otherwise
# invisible: an sbatch script that activated the runner venv too early would
# still produce a job that passed.
_GOOD_SETUP = (
    _PREAMBLE
    + _record("setup_record.txt")
    + '"$VENV/bin/python" -c \'import sys, pathlib; '
      'pathlib.Path("__OUTPUT_DIR__/setup_prefix.txt").write_text(sys.prefix)\'\n'
)

# base.py promises build_benchmark_script the opposite: the venv IS activated,
# and VLLM is reachable on localhost:8000. Both are recorded from inside the
# real job rather than inferred.
_GOOD_BENCHMARK = (
    _PREAMBLE
    + _record("benchmark_record.txt")
    + 'python -c \'import sys, pathlib; '
      'pathlib.Path("__OUTPUT_DIR__/benchmark_prefix.txt").write_text(sys.prefix)\'\n'
      'curl -sf http://localhost:8000/v1/models > "__OUTPUT_DIR__/models.json"\n'
      'printf \'%s\' \'{"accuracy": 0.5, "total": 4}\' > "__OUTPUT_DIR__/results.json"\n'
)

_FAILING_BENCHMARK = (
    _PREAMBLE
    + _record("benchmark_record.txt")
    + 'echo "benchmark refusing to run" >&2\nexit 3\n'
)

# Creates output_dir before failing so the EXIT trap in
# imported_dataset_script.sh has somewhere to write `.job_failed`: the trap runs
# `touch "${output_dir}/.job_failed"` and the script's own `mkdir -p` is further
# down, after the setup wait.
_FAILING_SETUP = (
    _PREAMBLE
    + _record("setup_record.txt")
    + 'echo "setup refusing to install" >&2\nexit 7\n'
)

# Deliberately WITHOUT _PREAMBLE, so output_dir does not exist when the job
# fails. This is the shape production actually produces: nothing creates
# output_dir before submission (`Scheduler.handle_imported_dataset_event` does
# not), and the script's own `mkdir -p` sits far below the EXIT trap.
_FAILING_SETUP_NO_MKDIR = (
    'set -eu\necho "setup refusing to install" >&2\nexit 7\n'
)


class CIImportedDatasetRunner(ImportedDatasetRunnerBase):
    """A real `ImportedDatasetRunnerBase` whose scripts run on the Slurm node.

    Not a mock of a runner: the scripts it returns are the scripts the compute
    node executes, and `parse_results` reads the files those scripts wrote. It
    stands in for BFCL only because BFCL's setup script is
    `pip install bfcl-eval`, which needs the network and a package index that
    the test container has neither of. Everything structural about a runner —
    the three-method shape, `$VENV` semantics, localhost:8000, the output
    contract — is genuine.

    The constructor arguments all default so that `Runner()` works, because that
    is how `Scheduler.handle_imported_dataset_event` builds one:
    `get_runner(runner_name)()`.
    """

    def __init__(
        self,
        setup_body: str = _GOOD_SETUP,
        benchmark_body: str = _GOOD_BENCHMARK,
    ) -> None:
        self._setup_body = setup_body
        self._benchmark_body = benchmark_body
        # `build_setup_script` is handed only `repo_root`, so a runner that
        # wants to write somewhere else has to have been told where. Set by
        # `_scripts_for` before either script is built; kept out of __init__ so
        # that `Runner()` — the registry's calling convention — still works.
        self.output_dir: Path | None = None
        # Recorded so a test can assert the value the scheduler passed through.
        self.seen_repo_root: Path | None = None

    def build_setup_script(self, repo_root: Path) -> str:
        self.seen_repo_root = repo_root
        return self._setup_body.replace(_OUTPUT_DIR, str(self.output_dir))

    def build_benchmark_script(self, model_instance, task, output_dir) -> str:
        return self._benchmark_body.replace(_OUTPUT_DIR, str(output_dir))

    def parse_results(self, output_dir: Path, task, model_instance) -> list[Score]:
        results = Path(output_dir) / "results.json"
        if not results.exists():
            raise ImportedDatasetResultError(
                f"no results.json under {output_dir} — the benchmark did not finish"
            )
        payload = json.loads(results.read_text())
        return [Score(name=key, value=value) for key, value in sorted(payload.items())]


def _make_task() -> ImportedDatasetTask:
    """A real task object, because that is what the scheduler passes.

    A runner that read `task.imported_dataset.args` would otherwise be handed
    something that merely looks like a task.
    """
    return ImportedDatasetTask(
        uuid=str(uuid.uuid4()),
        dataset_name="ci-imported",
        semantic_version="1.0.0",
        imported_dataset=ImportedDatasetConfig(name="ci-imported", commit="HEAD"),
    )


def _scripts_for(
    runner: CIImportedDatasetRunner, model, task, output_dir: Path
) -> tuple[str, str]:
    """Build the pair of scripts exactly as the scheduler does."""
    runner.output_dir = output_dir
    return (
        runner.build_setup_script(REPO_ROOT),
        runner.build_benchmark_script(model, task, output_dir),
    )


def _read_record(path: Path) -> dict[str, str]:
    """Parse the `key=value` file the scripts above write."""
    record: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            record[key] = value
    return record


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _port_is_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Whether anything accepts a TCP connection on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _queue_is_empty() -> bool:
    result = subprocess.run(
        ["squeue", "--noheader", "--me", "--format", "%i"],
        capture_output=True, text=True, check=False,
    )
    return not result.stdout.strip()


# NOTE: the orphaned-stub reaper and serving-port guard that used to live
# here are now in tests/slurm/conftest.py's `drained_cluster` fixture, which
# runs at setup AND teardown for every test in the suite. Three modules had
# grown their own copy; consolidating removes the chance of them disagreeing.

@pytest.fixture
def runner_name():
    """Hand out runner names and delete the venvs they cause to be created.

    WHY THIS IS NEEDED AT ALL. `imported_dataset_script.sh` builds the runner's
    venv at `<repo_root>/.eval360/envs/<runner_name>`, and `repo_root` is not a
    configurable input — `submit_imported_dataset_job` derives it from the
    location of `scheduler/slurm_manager.py`. In this container that directory
    is the bind-mounted working tree, so an imported-dataset job writes into the
    developer's checkout. Without this teardown the suite would leave a venv
    behind in the repo on every run.

    WHY THE NAMES ARE UNIQUE. The venv is sentinel-guarded
    (`$VENV/.setup_complete`), so a fixed name would make the second run of a
    test skip setup entirely and quietly stop testing it. A fresh name per test
    costs about 1.4s of `python -m venv` and keeps each run honest.
    """
    created: list[str] = []

    def _make(stem: str) -> str:
        name = f"ci-{stem}-{uuid.uuid4().hex[:6]}"
        created.append(name)
        return name

    yield _make

    for name in created:
        shutil.rmtree(RUNNER_ENV_ROOT / name, ignore_errors=True)
    # Only remove the shared parents if they are now empty — another test
    # run, or a developer, may legitimately have one of their own.
    for directory in (RUNNER_ENV_ROOT, RUNNER_ENV_ROOT.parent):
        try:
            directory.rmdir()
        except OSError:
            pass


@pytest.fixture
def registered_runner(runner_name):
    """Register a runner in the real registry and unregister it afterwards.

    The scheduler reaches a runner only through `get_runner(name)()`, so a test
    that instantiates the class directly skips the lookup the production path
    depends on. The registry is process-global and raises on duplicate names, so
    the entry has to be removed again or it would leak into any other test
    module that inspects `list_runners()`.
    """
    name = runner_name("registered")
    runner_registry.register(name)(CIImportedDatasetRunner)
    try:
        yield name
    finally:
        runner_registry._REGISTRY.pop(name, None)


# ---------------------------------------------------------------------------
# Small squeue helper — deliberately local, not added to cluster_harness
# ---------------------------------------------------------------------------


async def _squeue_format(job_id: int, fmt: str) -> str:
    """Return one squeue field for one job, as raw text."""
    proc = await asyncio.create_subprocess_exec(
        "squeue", "--jobs", str(job_id), f"--format={fmt}", "--noheader",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    return stdout.decode().strip()


async def _cluster_node() -> str:
    """The name of the partition's single node.

    Asked of sinfo rather than assumed from the container hostname, because the
    name that matters is the one Slurm will put in `%N` and therefore the one
    the scheduler will build a health URL from.
    """
    proc = await asyncio.create_subprocess_exec(
        "sinfo", "-h", "-p", "ci", "-o", "%N",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    node = stdout.decode().strip()
    if not node:
        raise RuntimeError(f"sinfo named no node in partition ci: {stderr.decode()!r}")
    return node


async def _submit_imported_job(
    manager,
    *,
    runner: CIImportedDatasetRunner,
    name: str,
    model,
    task,
    output_dir: Path,
    event_uuid: str,
) -> int:
    setup_script, benchmark_script = _scripts_for(runner, model, task, output_dir)
    return await manager.submit_imported_dataset_job(
        event_uuid, model, name, setup_script, benchmark_script, output_dir
    )


async def _wait_for_sentinel(output_dir: Path, name: str, *, timeout: float = 30.0):
    """Wait for a sentinel file to appear.

    Needed because `wait_for_job_completion` watches the QUEUE, and Slurm drops
    a job from squeue around the moment the batch script's last `touch` lands.
    Polling the file for a few seconds afterwards is the difference between
    testing the sentinel contract and testing a race.
    """

    async def _present():
        return (output_dir / name).exists() or None

    return await wait_until(_present, timeout=timeout, what=f"the {name} sentinel")


# ---------------------------------------------------------------------------
# get_job_node
# ---------------------------------------------------------------------------


async def test_get_job_node_blocks_while_pending_and_returns_the_node_on_start():
    """A PENDING job yields no node; the same call returns one once it starts.

    MEASURED, not assumed: for a PENDING job `squeue --format=%N` prints an
    EMPTY string on this cluster. It is `%R` that gives `(Resources)`. That
    matters because `get_job_node` skips values beginning with `(` — a guard
    which, with the `%N` the method actually requests, is unreachable. The
    behaviour the scheduler depends on therefore rests entirely on the empty
    string, which is what this asserts.

    Asserting that the call has NOT returned while the job is pending is the
    point of the test. A `get_job_node` that returned `''` for a pending job
    would still satisfy "eventually returns a node" if you only awaited it, and
    the scheduler would go on to health-check `http://:8000/health`.
    """
    # Hold every GPU so the job under test cannot be scheduled.
    blocker = await submit_raw(
        "ci-imported-blocker", "sleep 120", gres=f"gpu:{CLUSTER_GPUS}"
    )
    try:
        async with slurm_session() as manager:
            model = make_model("ci-node-poll")
            await manager.update_allocation([(model, 1)], [])

            async def _pending_row():
                jobs = await manager.get_all_jobs()
                rows = [job for job in jobs.values() if job["state"] == "PENDING"]
                return rows[0] if rows else None

            row = await wait_until(_pending_row, timeout=60, what="a PENDING job row")
            job_id = row["job_id"]

            assert await _squeue_format(job_id, "%N") == "", (
                "a pending job's %N is expected to be empty on this cluster; if "
                "this now prints a reason string, get_job_node's '(' guard has "
                "become load-bearing"
            )
            reason = await _squeue_format(job_id, "%R")
            assert reason.startswith("("), (
                f"the parenthesised reason ({reason!r}) lives in %R, which "
                "get_job_node does not request"
            )

            poll = asyncio.create_task(manager.get_job_node(job_id, poll_interval=FAST_POLL))
            await asyncio.sleep(2)
            if poll.done():
                pytest.fail(
                    "get_job_node returned "
                    f"{poll.result()!r} for a job with no node allocated yet"
                )

            # Free the GPUs; the pending job now starts.
            await scancel(blocker)
            node = await asyncio.wait_for(poll, timeout=NODE_TIMEOUT)

            assert node
            assert not node.startswith("(")
            running = await wait_for_running(manager, 1)
            assert node == running[0]["nodelist"]
    finally:
        await scancel(blocker)


async def test_get_job_node_raises_instead_of_hanging_when_the_job_is_gone():
    """A job that leaves the queue must end the poll, not continue it forever.

    THE BUG THIS PINS. `get_job_node` loops until `%N` is non-empty and never
    inspects `returncode`, so a job that disappears before it is ever seen
    RUNNING polls forever. `handle_imported_dataset_event` awaits it with no
    timeout, so that hangs the event — no error, no progress, nothing in the
    log.

    Both ways a job can be absent are covered here, because they look different
    to squeue and only one of them is an error exit. Measured on this cluster:

        recently ended  -> rc=0, ZERO rows          (indistinguishable from
                                                     pending after .strip())
        purged          -> rc=1, "Invalid job id specified"

    A pending job, by contrast, is rc=0 with ONE row whose `%N` is empty — so
    "no output" cannot be the signal to keep waiting; "a row exists but has no
    node yet" is.

    `asyncio.wait_for` is what turns the bug into a test failure rather than a
    hung suite; before the fix this fails on the timeout, after it the call
    raises promptly.

    THE TYPE IS PART OF THE CONTRACT, not decoration.
    `Scheduler.handle_imported_dataset_event` catches `SlurmJobVanished` and
    nothing wider, so that a real fault still propagates. If real Slurm produced
    an absence this method reported as a plain `RuntimeError`, that catch would
    miss it and one preempted job would again take down the whole event loop —
    a failure the mocked suite cannot see, because there the exception is raised
    by the same fake that the assertion trusts.
    """
    async with slurm_session() as manager:
        # A job that ends on its own, so nothing about the test cancels it.
        job_id = await submit_raw("eval360-vanish", "true")

        async def _left_the_queue():
            return not await _squeue_has_row(job_id) or None

        await wait_until(
            _left_the_queue, timeout=60, what="the job to leave the queue"
        )

        with pytest.raises(SlurmJobVanished, match="no longer queued|not found|Invalid"):
            await asyncio.wait_for(
                manager.get_job_node(job_id, poll_interval=FAST_POLL), timeout=20
            )

        # And the purged form, which exits non-zero rather than returning
        # nothing. A job ID slurmctld has never issued is the reliable way to
        # reach it without waiting out Slurm's retention.
        with pytest.raises(SlurmJobVanished, match="no longer queued|not found|Invalid"):
            await asyncio.wait_for(
                manager.get_job_node(999_999, poll_interval=FAST_POLL), timeout=20
            )


async def test_cancelling_a_vanished_job_does_not_raise():
    """`handle_imported_dataset_event`'s `finally` scancels a job that is gone.

    WHY THIS IS LOAD-BEARING FOR THE VANISH FIX. Handling the disappearance at
    the event boundary is pointless if the cleanup that follows it throws: the
    `finally` runs `cancel_job(job_id)` whenever `.job_complete` is absent,
    which after a vanish means cancelling a job Slurm no longer has. And
    `cancel_job` re-raises on a non-zero `scancel` once terminal-result capture
    is on — so if scancel reported an unknown job id as an error, the exception
    would escape the boundary anyway and reopen exactly the blast radius the fix
    closes.

    MEASURED on this cluster: `scancel` exits 0 both for a job that has already
    ended and for a job id that was never issued. That is the fact the fix rests
    on, and it is a property of Slurm rather than of this repository, so it is
    asserted here rather than assumed in a comment.
    """
    # `slurm_session` already calls `begin_terminal_result_capture()`, which is
    # what arms the re-raise in `cancel_job` — so this runs under the strictest
    # setting production has.
    async with slurm_session() as manager:
        job_id = await submit_raw("eval360-vanish-cancel", "true")

        async def _left_the_queue():
            return not await _squeue_has_row(job_id) or None

        await wait_until(
            _left_the_queue, timeout=60, what="the job to leave the queue"
        )

        # Recently ended, and never issued at all — the two absent forms
        # get_job_node distinguishes.
        await manager.cancel_job(job_id, cancellation_intent="scheduler_release")
        await manager.cancel_job(999_999, cancellation_intent="scheduler_release")


async def _squeue_has_row(job_id: int) -> bool:
    """True while squeue still reports a row for this job."""
    process = await asyncio.create_subprocess_exec(
        "squeue", "--jobs", str(job_id), "--noheader", "--format=%i",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()
    return process.returncode == 0 and bool(stdout.decode().strip())


# ---------------------------------------------------------------------------
# wait_for_vllm_health
# ---------------------------------------------------------------------------


async def test_wait_for_vllm_health_is_true_for_the_node_get_job_node_returned():
    """The scheduler's two-step — find the node, then health-check it — end to end.

    Composing the two calls is the thing worth testing: `get_job_node` returns a
    bare node NAME and `wait_for_vllm_health` interpolates it into
    `http://{node}:8000/health`, so the contract between them is that the name
    resolves. `socket.gethostbyname` is asserted explicitly because a name that
    did not resolve would fail as a health TIMEOUT — indistinguishable, from the
    scheduler's logs, from a model that failed to load.

    Conftest's `drained_cluster` fixture makes the True non-vacuous: nothing was
    answering on :8000 when this test started, so the healthy endpoint is the
    one this test's own job put there.
    """
    async with slurm_session() as manager:
        model = make_model("ci-health-true")
        await manager.update_allocation([(model, 1)], [])
        jobs = await wait_for_running(manager, 1)

        node = await asyncio.wait_for(
            manager.get_job_node(jobs[0]["job_id"], poll_interval=FAST_POLL),
            timeout=NODE_TIMEOUT,
        )
        assert node == jobs[0]["nodelist"]
        socket.gethostbyname(node)  # raises socket.gaierror if it does not resolve

        assert await manager.wait_for_vllm_health(
            node, max_time_to_deploy=90, poll_interval=FAST_POLL
        ) is True


async def test_wait_for_vllm_health_returns_false_instead_of_hanging():
    """Nothing on :8000 means False, returned promptly and without raising.

    The replica is made unreachable by serving a different port, so the Slurm
    job is genuinely RUNNING and only the endpoint is absent — the same shape as
    a VLLM that dies during model load, which is the case this deadline exists
    for.

    Two failure modes are ruled out. Returning False is asserted rather than
    "not True", because `wait_for_vllm_health` swallows every exception from the
    HTTP call and a bug that let one escape would surface as an error here, not
    a False. And the elapsed time is bounded, because the method counts only its
    own sleeps towards `max_time_to_deploy` — connection errors and the 3s
    aiohttp timeout are free — so a node that BLACKHOLED packets rather than
    refusing them would overshoot the deadline by a factor of four. The bound is
    loose enough not to be a stopwatch and tight enough to fail a hang.

    That :8000 really is unanswered is a precondition, not an assumption: see
    conftest's `drained_cluster` fixture, which this test is a reason for.
    """
    async with slurm_session() as manager:
        model = make_model("ci-health-false", vllm_cli_args=["--port 9001"])
        await manager.update_allocation([(model, 1)], [])
        jobs = await wait_for_running(manager, 1)
        node = jobs[0]["nodelist"]

        started = time.monotonic()
        healthy = await asyncio.wait_for(
            manager.wait_for_vllm_health(
                node, max_time_to_deploy=6, poll_interval=1.0
            ),
            timeout=60,
        )
        elapsed = time.monotonic() - started

        assert healthy is False
        assert elapsed < 30, (
            f"a 6s deploy deadline took {elapsed:.1f}s to give up; the sleep-only "
            "elapsed accounting has stopped bounding the wait"
        )


# ---------------------------------------------------------------------------
# wait_for_job_completion
# ---------------------------------------------------------------------------


async def test_wait_for_job_completion_returns_when_the_job_finishes_on_its_own():
    """It waits for the job to leave the queue, and not a moment before.

    The "not a moment before" half is the real assertion. `wait_for_job_completion`
    returns when squeue prints nothing OR when squeue exits non-zero, and those
    two conditions are indistinguishable from a transient squeue failure — a
    version that returned immediately would look identical to the scheduler,
    which would then read the sentinels of a job that had not written them yet.

    sacct is consulted afterwards to prove the job COMPLETED rather than being
    reaped, which is what separates this test from the cancellation one below.
    """
    async with slurm_session() as manager:
        job_id = await submit_raw("ci-imported-finishes", "sleep 8")

        waiter = asyncio.create_task(
            manager.wait_for_job_completion(job_id, poll_interval=FAST_POLL)
        )
        await asyncio.sleep(3)
        assert not waiter.done(), "returned while the job was still running"

        await asyncio.wait_for(waiter, timeout=COMPLETION_TIMEOUT)

        assert await _squeue_format(job_id, "%T") == ""
        outcome = await wait_for_terminal(manager, job_id)
        assert outcome.state == "COMPLETED", (
            f"expected a self-completed job, sacct says {outcome.raw_state!r}"
        )


async def test_wait_for_job_completion_returns_when_the_job_is_cancelled():
    """A cancelled job also leaves the queue, and the wait must end.

    This is the preemption path. If `wait_for_job_completion` only ever returned
    for clean completions, a preempted imported-dataset job would leave the
    scheduler awaiting a job that no longer exists — forever, since the method
    has no deadline of its own.
    """
    async with slurm_session() as manager:
        job_id = await submit_raw("ci-imported-cancelled", "sleep 300")

        waiter = asyncio.create_task(
            manager.wait_for_job_completion(job_id, poll_interval=FAST_POLL)
        )
        await asyncio.sleep(2)
        assert not waiter.done(), "returned before anything had happened to the job"

        await scancel(job_id)
        await asyncio.wait_for(waiter, timeout=COMPLETION_TIMEOUT)

        outcome = await wait_for_terminal(manager, job_id)
        assert outcome.state == "CANCELLED", (
            f"expected a cancelled job, sacct says {outcome.raw_state!r}"
        )


# ---------------------------------------------------------------------------
# The full imported-dataset lifecycle
# ---------------------------------------------------------------------------


async def test_a_full_imported_dataset_job_runs_setup_then_benchmark(
    tmp_path, registered_runner
):
    """Submit through the real path and check everything the job was asked to do.

    This is the test `tests/fake_slurm.py` cannot approximate at all: the fake
    returns a job ID and never builds an sbatch command, so nothing has ever
    executed `imported_dataset_script.sh`, decoded the two base64 scripts,
    created the runner venv from `bootstrap_python`, or written a sentinel.

    The assertions follow the script's own order so a failure says which stage
    broke:

    1. the job name Slurm accepted and the evidence ledger entry,
    2. `$VENV` landing where base.py documents it, un-activated during setup,
    3. that same venv ACTIVE during the benchmark,
    4. VLLM reachable from the benchmark at localhost:8000 — the address the
       runner contract tells benchmark authors to use, which is a different
       claim from the scheduler reaching it at `http://{node}:8000`,
    5. `.setup_complete_job` then `.job_complete`, and no `.job_failed`,
    6. `parse_results` turning the benchmark's output into Scores.

    The runner is fetched through `get_runner`, as the scheduler does, rather
    than constructed directly.
    """
    async with slurm_session() as manager:
        runner = runner_registry.get_runner(registered_runner)()
        assert isinstance(runner, ImportedDatasetRunnerBase)

        model = make_model("ci-imported-model", max_time_to_deploy=120)
        task = _make_task()
        output_dir = tmp_path / "ci-imported_output"
        event_uuid = str(uuid.uuid4())

        job_id = await _submit_imported_job(
            manager, runner=runner, name=registered_runner, model=model,
            task=task, output_dir=output_dir, event_uuid=event_uuid,
        )

        # (1) Identity. The name is built from the runner and model names and is
        # the only handle `cancel_all_owned_jobs` has on an imported job, so it
        # has to survive Slurm's own name handling intact.
        job_name = await _squeue_format(job_id, "%j")
        assert job_name.startswith(f"eval360id-{manager.instance_id}-"), (
            f"{job_name!r} does not carry this scheduler's instance id, so "
            "cancel_all_owned_jobs would not recognise it as ours"
        )
        assert job_name.endswith(f"-{event_uuid[:8]}")
        # Asserting on the value squeue GIVES BACK, rather than on the format
        # string, is what makes this a round trip: a name Slurm truncated would
        # lose the event-uuid suffix above.
        assert "ci-registered" in job_name and "ci-imported-model" in job_name
        ledger = {job.job_id: job for job in manager.get_submitted_jobs()}
        assert ledger[job_id].kind == "imported_dataset"
        assert ledger[job_id].event_uuid == event_uuid
        assert ledger[job_id].runner_name == registered_runner
        # `repo_root` is passed to the runner AND, separately, to the job via
        # the environment. They must be the same directory or the venv the
        # setup script populates is not the venv the benchmark activates.
        assert runner.seen_repo_root == REPO_ROOT

        node = await asyncio.wait_for(
            manager.get_job_node(job_id, poll_interval=FAST_POLL), timeout=NODE_TIMEOUT
        )
        assert await manager.wait_for_vllm_health(
            node, max_time_to_deploy=120, poll_interval=FAST_POLL
        ) is True

        await asyncio.wait_for(
            manager.wait_for_job_completion(job_id, poll_interval=FAST_POLL),
            timeout=COMPLETION_TIMEOUT,
        )

        # (5) Sentinels, in the order Scheduler.handle_imported_dataset_event
        # reads them.
        await _wait_for_sentinel(output_dir, ".job_complete")
        assert (output_dir / ".setup_complete_job").exists()
        assert not (output_dir / ".job_failed").exists()

        # (2) Setup ran, with $VENV where base.py says it is and not activated.
        expected_venv = RUNNER_ENV_ROOT / registered_runner
        setup = _read_record(output_dir / "setup_record.txt")
        assert setup["venv"] == str(expected_venv)
        assert (output_dir / "setup_prefix.txt").read_text() == str(expected_venv)
        assert setup["virtual_env"] != str(expected_venv), (
            "the runner venv was already active during setup; base.py promises "
            "it is created but not activated"
        )
        assert setup["which_python"] != f"{expected_venv}/bin/python"

        # (3) Benchmark ran with that venv activated.
        benchmark = _read_record(output_dir / "benchmark_record.txt")
        assert benchmark["virtual_env"] == str(expected_venv)
        assert benchmark["which_python"] == f"{expected_venv}/bin/python"
        assert (output_dir / "benchmark_prefix.txt").read_text() == str(expected_venv)

        # (4) VLLM was reachable from inside the job at the documented address.
        served = json.loads((output_dir / "models.json").read_text())
        assert [entry["id"] for entry in served["data"]] == [model.name], (
            "--served-model-name did not survive the base64 round trip into the "
            "imported-dataset sbatch script"
        )

        # (6) The runner's own half of the contract.
        scores = runner.parse_results(output_dir, task, model)
        assert scores == [Score(name="accuracy", value=0.5), Score(name="total", value=4)]


async def test_a_failing_benchmark_writes_job_failed_after_setup_completed(
    tmp_path, runner_name
):
    """A benchmark that exits non-zero is "failed internally", not "failed setup".

    The scheduler tells those two apart purely by which sentinels exist:
    `.setup_complete_job` present and `.job_complete` absent means the benchmark
    was reached and broke; `.setup_complete_job` absent means it never got that
    far. Both are written by a `trap ... EXIT` in `imported_dataset_script.sh`
    that no unit test can run, and getting them the wrong way round would send a
    broken benchmark down the "retry after preemption" path forever.

    `parse_results` is asserted to raise, because the scheduler never calls it
    in this state — so a runner that returned empty Scores instead of raising
    would look like a model that scored zero.
    """
    name = runner_name("benchfail")
    async with slurm_session() as manager:
        runner = CIImportedDatasetRunner(benchmark_body=_FAILING_BENCHMARK)
        model = make_model("ci-benchfail-model", max_time_to_deploy=120)
        task = _make_task()
        output_dir = tmp_path / "ci-benchfail_output"

        job_id = await _submit_imported_job(
            manager, runner=runner, name=name, model=model, task=task,
            output_dir=output_dir, event_uuid=str(uuid.uuid4()),
        )
        await asyncio.wait_for(
            manager.wait_for_job_completion(job_id, poll_interval=FAST_POLL),
            timeout=COMPLETION_TIMEOUT,
        )

        await _wait_for_sentinel(output_dir, ".job_failed")
        assert (output_dir / ".setup_complete_job").exists(), (
            "the benchmark ran, so setup must be recorded as complete — without "
            "this the scheduler would report a setup failure"
        )
        assert not (output_dir / ".job_complete").exists()
        # The benchmark really started before failing, rather than the job dying
        # somewhere earlier and happening to leave the same sentinels.
        assert (output_dir / "benchmark_record.txt").exists()

        with pytest.raises(ImportedDatasetResultError):
            runner.parse_results(output_dir, task, model)


async def test_a_failing_setup_leaves_no_setup_complete_job_sentinel(
    tmp_path, runner_name
):
    """Setup failure is distinguishable from benchmark failure, as the scheduler assumes.

    The mirror of the test above, and the reason both are needed: the sentinels
    are the ONLY signal, so a script change that wrote `.setup_complete_job`
    before waiting on the setup subshell would still leave a green benchmark
    test while silently reclassifying every failed install as a benchmark crash.

    Note the ordering inside the script: it waits for VLLM to be healthy BEFORE
    it waits on the setup subshell, so a setup failure is not noticed until
    after the server is up. That is why this test costs a full deploy despite
    testing an early failure.
    """
    name = runner_name("setupfail")
    async with slurm_session() as manager:
        runner = CIImportedDatasetRunner(setup_body=_FAILING_SETUP)
        model = make_model("ci-setupfail-model", max_time_to_deploy=120)
        task = _make_task()
        output_dir = tmp_path / "ci-setupfail_output"

        job_id = await _submit_imported_job(
            manager, runner=runner, name=name, model=model, task=task,
            output_dir=output_dir, event_uuid=str(uuid.uuid4()),
        )
        await asyncio.wait_for(
            manager.wait_for_job_completion(job_id, poll_interval=FAST_POLL),
            timeout=COMPLETION_TIMEOUT,
        )

        await _wait_for_sentinel(output_dir, ".job_failed")
        assert not (output_dir / ".setup_complete_job").exists()
        assert not (output_dir / ".job_complete").exists()
        # The setup script did run — this is a failure inside it, not a job that
        # never reached setup at all.
        assert _read_record(output_dir / "setup_record.txt")["venv"] == str(
            RUNNER_ENV_ROOT / name
        )
        # The sentinel is the guard, not the venv: a half-installed venv must be
        # left WITHOUT `.setup_complete`, so the next attempt rebuilds it from
        # scratch rather than activating a broken one.
        assert not (RUNNER_ENV_ROOT / name / ".setup_complete").exists()


async def test_job_failed_sentinel_survives_an_absent_output_dir(
    tmp_path, runner_name
):
    """An early failure must still leave `.job_failed`, not nothing at all.

    THE BUG THIS PINS. `imported_dataset_script.sh` installs its EXIT trap at
    the top, but only creates `output_dir` near the bottom, after the health
    gate. So the trap's `touch "${output_dir}/.job_failed"` writes into a
    directory that does not exist yet and fails silently — an early failure
    leaves NO sentinel at all.

    Nothing creates that directory beforehand:
    `Scheduler.handle_imported_dataset_event` submits without it, so on a first
    run the directory genuinely is absent. The suite's other sentinel tests all
    create it inside their setup script (`_PREAMBLE`), which is precisely the
    condition that hides this.

    Why it matters even though the event still fails today: the scheduler
    distinguishes "failed during setup" from "failed internally" from
    "preempted, reschedule" using these sentinels. With none written, the
    outcome is inferred from the absence of `.setup_complete_job` — the right
    answer for the wrong reason, one refactor away from being wrong, and it logs
    a misleading message in the meantime.
    """
    name = runner_name("trapdir")
    async with slurm_session() as manager:
        runner = CIImportedDatasetRunner(setup_body=_FAILING_SETUP_NO_MKDIR)
        model = make_model("ci-trapdir-model", max_time_to_deploy=120)
        task = _make_task()
        output_dir = tmp_path / "never-created-by-anyone"
        assert not output_dir.exists(), "the test must not create it either"

        job_id = await _submit_imported_job(
            manager, runner=runner, name=name, model=model, task=task,
            output_dir=output_dir, event_uuid=str(uuid.uuid4()),
        )
        await asyncio.wait_for(
            manager.wait_for_job_completion(job_id, poll_interval=FAST_POLL),
            timeout=COMPLETION_TIMEOUT,
        )

        await _wait_for_sentinel(output_dir, ".job_failed")
        assert not (output_dir / ".job_complete").exists()
