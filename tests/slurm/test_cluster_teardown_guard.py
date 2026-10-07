"""The teardown reaper's safety argument, put under test.

What this tests:
    `cluster_harness.reap_orphaned_stubs` SIGKILLs processes. It is allowed to
    do that because of exactly one claim — the Slurm queue is empty, therefore
    nothing legitimately owns the serving port. This module attacks that claim
    from both sides: a `squeue` that cannot answer, and a `squeue` that answers
    with a job still queued. In both cases the reaper must kill NOTHING.

Why this exists:
    The failure it guards against is silent and remote. `squeue` writes its
    errors to stderr and leaves stdout empty, so a teardown that reads only
    stdout cannot tell "no jobs" from "the controller did not answer". It then
    SIGKILLs whatever holds :8000 — which, if a job really was live, is that
    job's serving process. Nothing fails at that moment; a later test's health
    check simply gets no answer, and the suite reports a failure in code that
    had nothing to do with it. That bug class is only catchable by making the
    query fail on purpose.

    A guard nobody has watched fail is not evidence that it can, so every test
    here starts from a REAL process that the reaper genuinely wants to kill —
    one whose command line matches `STUB_MARKER` — and asserts on whether it is
    still breathing afterwards. `test_the_reaper_still_kills_a_real_orphan`
    exists to stop the whole module from passing vacuously: a guard that refused
    everything would satisfy every other assertion here.

Corner cases covered:
    A failing `squeue` (non-zero exit, empty stdout — the exact shape that used
    to read as "drained"); a `squeue` that hangs, which must hit a bounded
    timeout rather than stalling the suite; a genuinely non-empty queue holding
    a REAL Slurm job; a rejected `scancel`; and the happy path, where a
    confirmed-empty queue does let the reap through.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

from . import cluster_harness
from .cluster_harness import (
    STUB_MARKER,
    ClusterControlError,
    cancel_all_user_jobs,
    drain_queue,
    queued_job_ids,
    reap_orphaned_stubs,
    scancel,
    submit_raw,
    wait_until,
)

pytestmark = pytest.mark.cluster


# A stand-in orphan. It must be a process the reaper would REALLY pick up,
# otherwise "it survived" proves nothing — so it is named `fake_vllm_server.py`
# (the marker `pgrep -f` matches) and each test asserts it is a candidate before
# asserting it was spared. It sleeps instead of serving: binding :8000 for real
# would race the cluster's own stubs, and the port net is not what is under test
# here — the queue gate is.
_SACRIFICIAL_STUB = """\
import time
time.sleep(600)
"""


@pytest.fixture
def sacrificial_stub(tmp_path: Path):
    """A live process the reaper is entitled to kill, cleaned up either way.

    Yields the `subprocess.Popen`. The fixture kills it on the way out because
    half these tests deliberately prevent the reaper from doing so, and a
    ten-minute sleeper surviving a test run is exactly the leak this suite
    exists to stop.
    """
    script = tmp_path / STUB_MARKER
    script.write_text(_SACRIFICIAL_STUB)
    process = subprocess.Popen(["python3", str(script)])
    try:
        # Not a formality. Everything below reads "still alive" as evidence, and
        # that only means something once the reaper can actually see it.
        deadline = time.monotonic() + 10
        while process.pid not in _marked_pids():
            assert time.monotonic() < deadline, (
                f"the stand-in orphan (pid {process.pid}) never appeared in "
                f"`pgrep -f {STUB_MARKER}`, so nothing below would be testing "
                "the reaper's behaviour towards a real candidate"
            )
            time.sleep(0.1)
        yield process
    finally:
        try:
            os.kill(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def _marked_pids() -> list[int]:
    """PIDs the reaper's own `pgrep` net would match."""
    found = subprocess.run(
        ["pgrep", "-f", STUB_MARKER], capture_output=True, text=True, check=False
    )
    return [int(token) for token in found.stdout.split()]


def _is_alive(process: subprocess.Popen) -> bool:
    return process.poll() is None


def _stub_binary(tmp_path: Path, name: str, body: str) -> str:
    """Write an executable stand-in for a Slurm client and return its path.

    A real binary that really fails, rather than a patched `subprocess.run`.
    The bug being guarded against lives in how the return code and stderr of an
    actual process are (not) read, so faking the process away would fake away
    the bug.
    """
    path = tmp_path / name
    path.write_text(body)
    path.chmod(0o755)
    return str(path)


BROKEN_SQUEUE = """\
#!/bin/sh
# The real shape of the failure: the diagnosis goes to stderr and stdout is
# EMPTY — byte-identical to a drained queue if you only read stdout.
echo "slurm_load_jobs error: Unable to contact slurm controller" >&2
exit 1
"""

# `exec` so the timeout's SIGKILL lands on the sleep itself rather than on a
# shell that leaves the sleep orphaned for the next hour.
HANGING_SQUEUE = """\
#!/bin/sh
exec sleep 3600
"""

BROKEN_SCANCEL = """\
#!/bin/sh
echo "scancel: error: Kill job error on job id 1: Access/permission denied" >&2
exit 1
"""


def test_queued_job_ids_refuses_to_read_a_failed_squeue_as_an_empty_queue(
    monkeypatch, tmp_path: Path
):
    """The distinction the whole fix rests on: unknown is not empty."""
    monkeypatch.setattr(
        cluster_harness,
        "SQUEUE_BINARY",
        _stub_binary(tmp_path, "squeue", BROKEN_SQUEUE),
    )
    with pytest.raises(ClusterControlError) as raised:
        queued_job_ids()
    # The controller's own words must survive into the failure, or whoever reads
    # the CI log has to reproduce the outage to learn what broke.
    assert "Unable to contact slurm controller" in str(raised.value)


def test_queue_inspection_is_bounded_rather_than_waiting_forever(
    monkeypatch, tmp_path: Path
):
    """A wedged client must fail the teardown, not stall the whole suite.

    Without a timeout the failure mode is a suite that stops producing output
    until the CI job's own timeout kills it — which reports no line number and
    no cause. Two seconds here only to keep the test quick; the production bound
    is `CONTROL_COMMAND_TIMEOUT_SECONDS`.
    """
    monkeypatch.setattr(
        cluster_harness,
        "SQUEUE_BINARY",
        _stub_binary(tmp_path, "squeue", HANGING_SQUEUE),
    )
    monkeypatch.setattr(cluster_harness, "CONTROL_COMMAND_TIMEOUT_SECONDS", 2.0)

    started = time.monotonic()
    with pytest.raises(ClusterControlError, match="did not answer within"):
        queued_job_ids()
    assert time.monotonic() - started < 30, "the bound did not bound anything"


def test_the_reaper_kills_nothing_when_the_queue_cannot_be_inspected(
    monkeypatch, tmp_path: Path, sacrificial_stub
):
    """The finding, reproduced: a failed `squeue` must not license a SIGKILL.

    Before this guard the reaper saw empty stdout, concluded the cluster was
    idle and killed every process matching the stub marker. Here the stand-in
    orphan is a confirmed `pgrep` match — the fixture proves that before
    yielding — so its survival is the whole result: the reaper looked at a
    candidate it would normally kill and declined, because it could not
    establish that the queue was empty.
    """
    monkeypatch.setattr(
        cluster_harness,
        "SQUEUE_BINARY",
        _stub_binary(tmp_path, "squeue", BROKEN_SQUEUE),
    )
    assert sacrificial_stub.pid in _marked_pids()

    # The raise is CAUGHT rather than asserted with `pytest.raises`, so that the
    # survival check below runs either way. Against the unguarded reaper the
    # useful diagnosis is "it killed the process", not "it failed to raise" —
    # `pytest.raises` would stop at the latter and never reach the former.
    raised: ClusterControlError | None = None
    try:
        reap_orphaned_stubs()
    except ClusterControlError as error:
        raised = error

    assert _is_alive(sacrificial_stub), (
        "the reaper SIGKILLed a process on the strength of a `squeue` that "
        "FAILED. On the cluster that process is a live job's serving stub, and "
        "the test that notices is some unrelated one, minutes later."
    )
    assert raised is not None, (
        "the reaper spared the process but returned quietly; a teardown that "
        "cannot see the queue must say so, or the next run inherits an orphan "
        "nobody knows about"
    )


def test_the_reaper_kills_nothing_while_a_real_job_is_still_queued(
    sacrificial_stub,
):
    """The other half of the gate: a confirmed-NON-empty queue also refuses.

    Uses a real `sbatch` job rather than a stubbed `squeue`, because this is the
    case the guard exists for in production: the drain has not finished, the job
    holding :8000 is legitimate, and killing it breaks a running test instead of
    cleaning up after a dead one.
    """
    job_id = _submit_sleeper()
    try:
        assert queued_job_ids(), "the job never reached the queue; nothing is gated"

        # Caught rather than `pytest.raises`d, for the same reason as above: the
        # assertion that matters is about the process, not about the exception.
        raised: ClusterControlError | None = None
        try:
            reap_orphaned_stubs()
        except ClusterControlError as error:
            raised = error

        assert _is_alive(sacrificial_stub), (
            "the reaper killed a serving process while a job was still queued — "
            "the exact process that job may have started"
        )
        assert raised is not None and "still queued" in str(raised), (
            f"expected a refusal naming the queued job; got {raised!r}"
        )
    finally:
        _cancel_sleeper(job_id)


def test_the_reaper_still_kills_a_real_orphan_once_the_queue_is_confirmed_empty(
    sacrificial_stub,
):
    """Proof the gate is a gate and not a wall.

    Every other test here passes if `reap_orphaned_stubs` simply refuses
    everything, which would be a worse bug than the one being fixed: orphaned
    stubs answering later tests' health checks is a false-GREEN generator. So
    one test drives the happy path and insists the kill still happens.
    """
    drain_queue()
    assert queued_job_ids() == []

    reap_orphaned_stubs()

    # SIGKILL is delivered synchronously but reaping the zombie is not, so poll
    # rather than asserting on the instant after the call.
    deadline = time.monotonic() + 10
    while _is_alive(sacrificial_stub) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _is_alive(sacrificial_stub), (
        "an orphaned stub survived a reap with a confirmed-empty queue; the "
        "guard has become a blanket refusal and orphans will answer later "
        "tests' health checks"
    )


def test_drain_queue_reports_an_unanswerable_queue_instead_of_declaring_success(
    monkeypatch, tmp_path: Path
):
    """`drain_queue` may only resolve in favour of "empty" on evidence.

    Retrying a blip is fine — controllers do blip — but the deadline has to
    expire into a failure, never into a shrug. A `drain_queue` that returned
    quietly here would hand a false all-clear to the reaper immediately after.
    """
    monkeypatch.setattr(
        cluster_harness,
        "SQUEUE_BINARY",
        _stub_binary(tmp_path, "squeue", BROKEN_SQUEUE),
    )
    with pytest.raises(ClusterControlError, match="must be assumed FULL"):
        drain_queue(timeout=2)


def test_a_rejected_scancel_is_reported_rather_than_assumed_to_have_worked(
    monkeypatch, tmp_path: Path
):
    """An unchecked `scancel` turns a permissions error into a drain timeout.

    Same class of bug as the `squeue` one: the command's verdict was thrown
    away, so the failure surfaced later and somewhere else — as "the cluster did
    not drain", which points at the cluster rather than at the cancel that was
    refused.
    """
    monkeypatch.setattr(
        cluster_harness,
        "SCANCEL_BINARY",
        _stub_binary(tmp_path, "scancel", BROKEN_SCANCEL),
    )
    with pytest.raises(ClusterControlError, match="permission denied"):
        cancel_all_user_jobs()


# --------------------------------------------------------------------------
# Real-job helpers. Sync tests, so the async harness calls are run on their own
# loop; `asyncio.run` is safe here because nothing in this module shares an
# asyncio primitive across calls (see cluster_harness on `_deployment_lock`).
# --------------------------------------------------------------------------


def _submit_sleeper() -> int:
    async def _go() -> int:
        job_id = await submit_raw("teardown-guard-blocker", script="sleep 120")
        await wait_until(
            lambda: _queued(job_id),
            timeout=60,
            what=f"job {job_id} to appear in the queue",
        )
        return job_id

    return asyncio.run(_go())


async def _queued(job_id: int) -> bool:
    return str(job_id) in queued_job_ids()


def _cancel_sleeper(job_id: int) -> None:
    async def _go() -> None:
        await scancel(job_id)

    asyncio.run(_go())
    # The autouse `drained_cluster` teardown would fail on a queue that is still
    # occupied, and it would be right to — so wait here rather than leaving the
    # next fixture to discover it.
    drain_queue(timeout=60)
