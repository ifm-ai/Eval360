"""Shared helpers for driving the Slurm test cluster.

Every test in this package talks to a real slurmctld, so the awkward parts are
resource contention and cleanup rather than mocking. This module centralises
both.

WHY EACH TEST GETS ITS OWN `SlurmManager`. `SlurmManager._deployment_lock` is an
`asyncio.Lock`, and an asyncio primitive binds to the first event loop that uses
it. pytest-asyncio gives each async test a fresh loop, so a manager shared
across tests would carry a lock bound to a dead loop and fail on the second
`update_allocation`. Managers are cheap; loops are not shareable. Creating one
per test also gives each test its own `instance_id`, which is exactly the
mechanism `SlurmManager` uses to ignore other schedulers' jobs — so tests cannot
see, cancel, or be confused by each other's work.

WHY CLEANUP WAITS. `scancel` returns long before the job leaves the queue. A
teardown that only calls `cancel_all_owned_jobs` would let the next test start
while the previous test's replicas still hold GPUs, and that test would then sit
PENDING and fail on a timeout that has nothing to do with what it was testing.
`slurm_session` therefore waits for the queue to drain before yielding control
back.

WHY EVERY CLEANUP PRIMITIVE HERE FAILS CLOSED. The reaper below SIGKILLs
processes, and its entire safety argument is "the Slurm queue is empty,
therefore nothing legitimately owns the serving port". A `squeue` that merely
FAILED — controller restarting, munge socket gone, binary hung — produces empty
stdout, which is indistinguishable from an empty queue if you only read stdout.
The argument then collapses silently and the reaper shoots a live job's server,
which surfaces later as a baffling failure in an unrelated test.

So nothing in this module infers "empty" from "no output". `queued_job_ids`
raises `ClusterControlError` unless `squeue` actually succeeded, every
subprocess call is bounded by a timeout so a hung binary cannot stall a suite,
`cancel_all_user_jobs` checks that `scancel` was accepted, and
`reap_orphaned_stubs` re-confirms the drain itself rather than trusting its
caller to have done so. The failure mode we accept is a loud teardown error; the
one we refuse is a quiet kill.
"""

from __future__ import annotations

import asyncio
import getpass
import os
import re
import signal
import subprocess
import time
from contextlib import asynccontextmanager

from scheduler.event import DeploymentInfo
from scheduler.model import ModelInstance, ModelType, ServingSlurmResources
from scheduler.slurm_manager import SlurmManager

SERVING_VENV = os.environ.get("CI_SERVING_VENV", "/tmp/serving-venv")
MODEL_PATH = os.environ.get("CI_MODEL_PATH", "/tmp/model")
OUTPUT_PATH = os.environ.get("CI_OUTPUT_PATH", "/tmp/output")
SLURM_LOG_DIR = os.environ.get("CI_SLURM_LOG_DIR", ".")

# The cluster advertises gpu:4 (scripts/ci/slurm_up.sh), so a one-GPU request
# leaves room for up to four concurrent replicas.
CLUSTER_GPUS = int(os.environ.get("FAKE_GPUS", "4"))

# Long enough to absorb a slow CI runner, short enough that a genuine hang fails
# the job rather than the 30-minute workflow timeout.
DEPLOY_TIMEOUT_SECONDS = int(os.environ.get("DEPLOY_TIMEOUT_SECONDS", "180"))
DRAIN_TIMEOUT_SECONDS = int(os.environ.get("DRAIN_TIMEOUT_SECONDS", "90"))

# A single Slurm client call answers in milliseconds on a healthy cluster. This
# bound is not a performance target — it is the difference between "the
# controller is wedged, fail the teardown" and a suite that hangs until the
# workflow timeout kills it with no indication of where it stopped.
CONTROL_COMMAND_TIMEOUT_SECONDS = float(
    os.environ.get("SLURM_CONTROL_TIMEOUT_SECONDS", "30")
)

# Resolved through PATH under normal use. They are module attributes rather than
# literals so `test_cluster_teardown_guard.py` can point them at a stub binary
# that really fails or really hangs, and watch the guards refuse to reap. A
# guard nobody has watched fail is not evidence that it can.
SQUEUE_BINARY = "squeue"
SCANCEL_BINARY = "scancel"


class ClusterControlError(RuntimeError):
    """A cleanup command did not answer, so the cluster's state is UNKNOWN.

    Deliberately distinct from "the cluster is busy". Callers may not downgrade
    this to "nothing was queued": the whole point is that an unanswered `squeue`
    tells you nothing, and acting on nothing is how the reaper kills a live
    job's server.
    """


def _run_control_command(
    command: list[str],
    *,
    timeout: float | None = None,
    ok_returncodes: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess:
    """Run a cleanup command, or raise `ClusterControlError` saying why not.

    Three failures are folded into one exception because the caller's response
    to all three is identical — refuse to act — and only the message differs:
    the binary is missing, it exceeded `timeout`, or it exited outside
    `ok_returncodes`. stderr is carried through because "squeue failed" without
    `slurm_load_jobs error: Unable to contact slurm controller` sends whoever
    reads the CI log looking in the wrong place.

    `timeout=None` means the module default, read HERE rather than bound as a
    parameter default: a default argument is evaluated once at import, which
    would make the bound unpatchable and the test that proves it bounds anything
    impossible to write.
    """
    if timeout is None:
        timeout = CONTROL_COMMAND_TIMEOUT_SECONDS
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=timeout
        )
    except FileNotFoundError as error:
        raise ClusterControlError(f"{command[0]!r} is not on PATH: {error}") from error
    except subprocess.TimeoutExpired as error:
        raise ClusterControlError(
            f"{command[0]!r} did not answer within {timeout}s: {command}"
        ) from error
    if result.returncode not in ok_returncodes:
        raise ClusterControlError(
            f"{command[0]!r} exited {result.returncode} (expected one of "
            f"{ok_returncodes}): {command}\nstderr: {result.stderr.strip()!r}"
        )
    return result


def queued_job_ids() -> list[str]:
    """Job IDs this user owns, or raise if `squeue` could not be asked.

    The empty list means the QUEUE is empty. It never means the query failed —
    that is `ClusterControlError`. Keeping those two apart is the entire fix for
    the teardown that used to SIGKILL processes on the strength of a `squeue`
    that had not run.
    """
    result = _run_control_command(
        [SQUEUE_BINARY, "--noheader", "--me", "--format", "%i"]
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def cancel_all_user_jobs() -> None:
    """`scancel` everything this user owns, and check the request was accepted.

    Safe here in a way it would never be on a real cluster: the container holds
    nothing but the test cluster and is thrown away afterwards. Checking the
    exit status matters because a rejected `scancel` followed by an unchecked
    drain looks exactly like a cluster that simply took a while to empty — until
    the drain times out for a reason nobody can see.

    `scancel` returns when the CONTROLLER accepts the request, not when the jobs
    are gone, so every caller must still `drain_queue` afterwards.
    """
    _run_control_command([SCANCEL_BINARY, "--user", getpass.getuser()])


def drain_queue(timeout: float = 60) -> None:
    """Block until the queue is CONFIRMED empty; raise if it cannot be.

    A transient inspection failure is retried until the deadline — controllers
    do blip — but the deadline never resolves in favour of "drained". Either
    `squeue` said "no jobs" or this raises.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            queued = queued_job_ids()
        except ClusterControlError as error:
            if time.monotonic() >= deadline:
                raise ClusterControlError(
                    f"could not establish that the queue drained within "
                    f"{timeout}s, so it must be assumed FULL: {error}"
                ) from error
            time.sleep(0.5)
            continue
        if not queued:
            return
        if time.monotonic() >= deadline:
            raise ClusterControlError(
                f"cluster did not drain within {timeout}s; jobs still queued: "
                f"{queued}. Later tests would fail for unrelated reasons, and "
                "freeing the serving port now could shoot a live job's server."
            )
        time.sleep(0.5)


def make_model(
    name: str = "ci-stub-model",
    *,
    gpus_per_node: int = 1,
    cpus_per_task: int = 1,
    memory_gb: int | None = 1,
    time_limit: str = "0:20:00",
    max_time_to_deploy: int = DEPLOY_TIMEOUT_SECONDS,
    vllm_cli_args: list[str] | None = None,
    venv_path: str | None = None,
) -> ModelInstance:
    """A ModelInstance the CI node can actually schedule.

    Defaults request one GPU because `_serving_resource_args` always emits
    `--gres=gpu:N` and `gpus_per_node` is a StrictPositiveInt — no configuration
    can ask for zero, which is why the cluster has to advertise fake GPUs at all.
    """
    return ModelInstance(
        name=name,
        path=MODEL_PATH,
        venv_path=venv_path or SERVING_VENV,
        max_simultaneous_requests=4,
        max_time_to_deploy=max_time_to_deploy,
        vllm_cli_args=vllm_cli_args if vllm_cli_args is not None else ["--max-model-len 256"],
        openai_kwargs={"temperature": 0.0},
        parser_type="noop",
        model_type=ModelType.CHAT,
        owner="eval360-ci",
        output_path=OUTPUT_PATH,
        serving_slurm_resources=ServingSlurmResources(
            gpus_per_node=gpus_per_node,
            cpus_per_task=cpus_per_task,
            memory_gb=memory_gb,
            time_limit=time_limit,
        ),
    )


def deployment_for(*models: ModelInstance) -> dict[str, DeploymentInfo]:
    """The `desired_models_dict` shape `get_model_state` expects."""
    return {
        model.name: DeploymentInfo(
            model=model, priority=0, generation_events=set(), grader_events=set()
        )
        for model in models
    }


@asynccontextmanager
async def slurm_session(*, instance_id: str | None = None, poll_interval: float = 1.0):
    """A SlurmManager whose jobs are guaranteed gone when the block exits."""
    manager = SlurmManager(
        log_dir=SLURM_LOG_DIR, partition="ci", instance_id=instance_id,
        poll_interval=poll_interval,
    )
    # Enables the submission ledger. Without it `update_allocation` omits
    # `--parsable` and skips `_parse_sbatch_job_id` entirely — both guarded at
    # the call site — so `get_submitted_jobs()` is always empty, which would
    # make several assertions vacuously true.
    manager.begin_terminal_result_capture()
    try:
        yield manager
    finally:
        try:
            await manager.cancel_all_owned_jobs()
            await drain(manager)
        except Exception as error:  # noqa: BLE001
            # Teardown must not mask the failure that brought us here, but a
            # silent failure to clean up would break every later test with an
            # unrelated PENDING timeout, so say so.
            print(f"WARNING: cluster teardown for {manager.instance_id} failed: {error!r}")


async def drain(manager: SlurmManager, timeout: float = DRAIN_TIMEOUT_SECONDS) -> None:
    """Wait until this manager owns no queued jobs.

    `scancel` is asynchronous: it returns as soon as the controller accepts the
    request, not when the job is gone. Without this wait the next test starts
    while these replicas still hold GPUs.
    """
    await wait_until(
        lambda: _no_jobs(manager),
        timeout=timeout,
        what=f"jobs owned by {manager.instance_id} to leave the queue",
    )


async def _no_jobs(manager: SlurmManager) -> bool:
    return not await manager.get_all_jobs()


async def wait_until(
    condition,
    *,
    timeout: float,
    interval: float = 0.5,
    what: str = "condition",
):
    """Poll an async predicate until it is truthy, or fail with context.

    Returns the truthy value, so callers can wait and capture in one step.
    `TimeoutError` carries `what` because "timed out" on its own tells you
    nothing about which of a test's several waits gave up.
    """
    deadline = time.monotonic() + timeout
    while True:
        result = await condition()
        if result:
            return result
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out after {timeout}s waiting for {what}")
        await asyncio.sleep(interval)


async def running_jobs(manager: SlurmManager) -> list[dict]:
    """Every job this manager owns that Slurm reports as RUNNING."""
    jobs = await manager.get_all_jobs()
    return [job for job in jobs.values() if job["state"] == "RUNNING"]


async def wait_for_running(manager: SlurmManager, count: int, *, timeout: float = DEPLOY_TIMEOUT_SECONDS):
    """Wait until exactly `count` of this manager's jobs are RUNNING."""

    async def _ready():
        jobs = await running_jobs(manager)
        return jobs if len(jobs) == count else None

    return await wait_until(
        _ready, timeout=timeout, what=f"{count} RUNNING job(s) for {manager.instance_id}"
    )


async def wait_for_live(manager: SlurmManager, deployment: dict, *, timeout: float = DEPLOY_TIMEOUT_SECONDS):
    """Wait until `get_model_state` reports at least one live replica."""

    async def _ready():
        pending, deploying, live, dead, replicas = await manager.get_model_state(deployment)
        return live or None

    return await wait_until(_ready, timeout=timeout, what="a live replica")


async def submit_raw(
    job_name: str,
    script: str = "true",
    *,
    gres: str = "gpu:1",
    extra_args: tuple[str, ...] = (),
) -> int:
    """Submit a job with sbatch directly, bypassing SlurmManager.

    Used for what SlurmManager cannot express: jobs with names it does NOT own
    (to prove they are ignored), jobs with chosen exit statuses (to exercise
    sacct's ExitCode parsing without waiting for a real crash), and jobs sized
    via `gres` to fill the cluster so a scheduler replica is forced PENDING.
    `extra_args` appends raw sbatch flags — note it lands AFTER this function's
    own `--time=0:10:00`, so a later `--time` wins.
    """
    command = [
        "sbatch", "--parsable",
        f"--job-name={job_name}",
        "--partition=ci",
        "--nodes=1", "--ntasks=1",
        f"--gres={gres}",
        "--cpus-per-task=1",
        "--time=0:10:00",
        f"--output={SLURM_LOG_DIR}/slurm-%j.out",
        *extra_args,
        f"--wrap={script}",
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


async def scancel(job_id: int) -> None:
    """Cancel a job without going through SlurmManager."""
    process = await asyncio.create_subprocess_exec(
        "scancel", str(job_id),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await process.communicate()


async def wait_for_terminal(manager: SlurmManager, job_id: int, *, timeout: float = DRAIN_TIMEOUT_SECONDS):
    """Wait for a settled accounting row: terminal state AND a real name.

    Being terminal is not sufficient. slurmdbd populates a row's state and its
    name independently, so for up to ~1.5s after a scancel sacct serves a row
    that is terminal and still carries the placeholder `JobName='allocation'`.
    Tests that assert on the name must wait for the row to settle; see
    docs/SLURM_TEST_CLUSTER.md.
    """

    async def _ready():
        outcomes = await manager.get_job_accounting([job_id])
        outcome = outcomes.get(job_id)
        if outcome is None or not outcome.is_terminal:
            return None
        if outcome.job_name == "allocation":
            return None
        return outcome

    return await wait_until(
        _ready, timeout=timeout, what=f"a settled terminal sacct row for job {job_id}"
    )


# -------------------------------------------------------------------------
# Orphaned stub servers
# -------------------------------------------------------------------------

# The stub server's script name, used to find orphans. Kept as a constant so the
# marker appears exactly once.
STUB_MARKER = "fake_vllm_server.py"


def reap_orphaned_stubs() -> None:
    """Kill stub servers that outlived the Slurm job that started them.

    WHY THIS IS NECESSARY, and why it is a correctness issue rather than tidiness.

    `sbatch_script.sh` backgrounds the server and the job's lifetime is the batch
    script's. With `ProctrackType=proctrack/linuxproc` and `TaskPlugin=task/none`
    (see scripts/ci/slurm_up.sh — the cgroup plugins cannot be used in this
    container) Slurm has no cgroup to tear down, so what happens to the child
    depends on HOW the job ended:

        cancelled  -> the process group is signalled, the stub dies      (verified)
        COMPLETED  -> the script exits, the stub is orphaned and SURVIVES (verified)

    Measured directly: after a job reached COMPLETED,
    `ss -ltnp` still showed `0.0.0.0:8000` held by that job's python process.

    That is a FALSE-GREEN generator, not a leak of memory. The scheduler
    health-checks a hard-coded `http://{nodelist}:8000/health`, and this is a
    single-node cluster, so an orphan from an earlier test answers for a later
    test's replica — `get_model_state` would report a model live that is not
    serving, and a test asserting "the model became live" would pass without the
    code under test having worked.

    NOTE THE PID EXCLUSIONS. `pkill -f fake_vllm_server.py` is the obvious
    implementation and it is a trap: `-f` matches the whole command line, so any
    shell whose command line mentions the script — including the one running the
    pkill — matches itself and dies. That cost two debugging cycles while
    investigating this very bug. Matching PIDs explicitly and skipping our own
    process makes the failure mode impossible.

    ONLY REAPS WITH A CONFIRMED-EMPTY QUEUE. That is what makes the kill safe:
    if no job is queued, nothing legitimately owns the serving port, so anything
    still holding it is an orphan by definition.

    The confirmation happens HERE, not in the caller, and that is the point.
    Callers do drain first, but "my caller drained a moment ago" is a promise,
    while `queued_job_ids()` raising on a failed `squeue` is a fact. If the
    queue cannot be inspected, or is inspected and is not empty, this raises
    `ClusterControlError` and kills NOTHING — a teardown that errors loudly is
    recoverable, a teardown that SIGKILLs a running job's server is a mystery
    failure three tests later. Re-checking also closes the window between the
    caller's drain and the kill.

    Two nets, because they catch different things. Matching the process finds a
    stub regardless of which port it bound (a replica told to serve :9001 still
    holds a GPU's worth of confusion); checking the listener finds anything
    squatting :8000 even if it is not one of ours. Kills are announced so a leak
    shows up in CI output instead of being silently smoothed over.
    """
    queued = queued_job_ids()  # raises rather than guessing — see the module docstring
    if queued:
        raise ClusterControlError(
            f"refusing to reap serving processes: {len(queued)} job(s) are still "
            f"queued ({queued}). Anything holding the serving port may belong to "
            "one of them, and killing it would break a live test rather than a "
            "leaked one."
        )

    victims: dict[int, str] = {}

    # rc 1 is pgrep's "no processes matched", which is the normal healthy case.
    # Anything above that is a real failure, and swallowing it would leave
    # orphans alive to answer a later test's health check — a false GREEN, which
    # is the other half of this function's job.
    found = _run_control_command(
        ["pgrep", "-f", STUB_MARKER], ok_returncodes=(0, 1)
    )
    for token in found.stdout.split():
        try:
            victims[int(token)] = f"matches {STUB_MARKER}"
        except ValueError:
            continue
    for pid in serving_port_listeners():
        victims.setdefault(pid, f"listening on :{SERVING_PORT}")

    ours = {os.getpid(), os.getppid()}
    for pid, why in victims.items():
        if pid in ours:
            continue
        print(
            f"NOTE: ORPHANED SERVING PROCESS: pid {pid} ({why}) survived with an "
            "empty Slurm queue, so it outlived the job that spawned it. Killing "
            "it; otherwise a later test would be served by a stranger."
        )
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    if not victims:
        return
    deadline = time.monotonic() + 15
    while True:
        holders = serving_port_listeners()
        if not holders:
            return
        if time.monotonic() >= deadline:
            raise ClusterControlError(
                f"port {SERVING_PORT} is still held by {holders} after SIGKILL; "
                "a serving job under test could not bind it."
            )
        time.sleep(0.25)


SERVING_PORT = 8000
_LISTENER_PID_RE = re.compile(r"pid=(\d+)")


def serving_port_listeners() -> list[int]:
    """PIDs listening on the port the scheduler health-checks.

    Raises rather than returning `[]` when `ss` cannot answer. The empty list is
    read by callers as "the port is free" — both as a kill list and as the
    post-SIGKILL proof that the reap worked — so a silently failed `ss` would
    certify a port as free while a stub still held it, and the next test would
    be health-checked by a stranger.
    """
    result = _run_control_command(["ss", "-lptnH", f"sport = :{SERVING_PORT}"])
    return [int(pid) for pid in _LISTENER_PID_RE.findall(result.stdout)]
