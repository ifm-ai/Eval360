"""Job lifecycle states, and the terminal-evidence ledger, against a real Slurm.

What this tests:
    Two things that only a real scheduler can produce.

    (A) The Slurm job states and time formats `slurm_manager.py` classifies:
        `SlurmJobOutcome.state` / `.exit_code` / `.signal` for a job that ran
        out of wall clock (TIMEOUT), a job killed by a signal (`0:9`), a job
        cancelled by an administrator (`CANCELLED by 0`), `_normalize_sacct_state`
        on the decorated and truncated forms sacct really emits, `_to_seconds`
        on a real squeue `%M` field, and `get_available_nodes` when the only
        node is drained or down.

    (B) The terminal-evidence ledger — `begin_terminal_result_capture`,
        `get_submitted_jobs`, `record_role_completion`,
        `release_submitted_model_serving_jobs` and
        `wait_for_terminal_job_outcomes` — driven with real sbatch children
        rather than fabricated `SubmittedSlurmJob` values.

Why this exists:
    The mocked suites cannot reach any of this. `tests/fake_slurm.py` replaces
    `SlurmManager` outright, and `tests/test_job_manager.py` feeds the parsers
    canned squeue strings; neither can produce a `TIMEOUT`, a `0:9` ExitCode, a
    `CANCELLED by 0` decoration or a drained node, because those are decisions
    slurmctld makes, not strings a test author types. `tests/slurm/
    test_slurm_accounting.py` covers sacct's row *shape* but only ever sees the
    states COMPLETED and FAILED, and only ever sees signal 0 — `0:0`, `1:0`
    and `3:0`.

    The ledger is the newest and most safety-critical code in the file: it is
    what `Scheduler._reconcile_terminal_jobs` reads to decide whether an
    `evaluate-now` invocation may publish a terminal result at all. A wrong
    ledger does not degrade a run, it fails it after all the compute is spent —
    or, worse, publishes evidence for children that were never reconciled. Up
    to now nothing has exercised it against children Slurm actually accounted
    for.

Corner cases covered:
    A job that exceeds `--time` (TIMEOUT, `0:0` — terminal, not clean, and NOT
    a failure the exit code can reveal); a job killed by SIGKILL (FAILED,
    `0:9`, the first non-zero *signal* any test has seen); the two decorated
    state forms sacct emits (`CANCELLED by 0` in `--parsable2`, `CANCELLED+`
    when a field width truncates) and their normalisation; `_to_seconds` on the
    `M:SS` form squeue produces for a young job; a drained node and a down node
    seen through `get_available_nodes`; a `wait_for_terminal_job_outcomes` over
    four children that reach terminal states more than a minute apart; the
    ledger surviving the same child being observed repeatedly by the squeue
    poll loop; the changed-identity guard; role binding to an active serving
    child and its failure modes; and the release path for a model-serving
    child, including the `cancellation_intent` that `_reconcile_terminal_jobs`
    requires before it will accept a CANCELLED child as successful.

Cost note (read before adding more of these):
    TIMEOUT is expensive. Slurm's `--time` granularity is one minute and
    slurmctld only sweeps job time limits periodically, so the cheapest
    possible TIMEOUT measured on this cluster is ~71s wall clock — there is no
    configuration that makes it faster. It is therefore paid ONCE, by the
    module-scoped `terminal_fleet` fixture, which submits four children at the
    same time and lets the other three finish while the slow one burns its
    minute. That single wait doubles as the staggered-terminal evidence for
    `wait_for_terminal_job_outcomes`. Do not add a second job that has to time
    out.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from dataclasses import dataclass, field

import pytest

from scheduler.slurm_manager import SlurmManager, SlurmJobOutcome, SubmittedSlurmJob

from .cluster_harness import (
    drain,
    make_model,
    scancel,
    slurm_session,
    submit_raw,
    wait_for_terminal,
    wait_until,
)

pytestmark = pytest.mark.cluster

# Generous relative to the ~71s a TIMEOUT actually takes here, so a slow CI
# runner does not turn a working cluster into a red build, but far below the
# workflow timeout so a genuine hang still fails as a hang.
FLEET_TIMEOUT_SECONDS = 240

# How long a ledger test will wait for its freshly submitted child to become
# visible to squeue. Submission is fast; this only absorbs controller latency.
QUEUE_VISIBLE_TIMEOUT_SECONDS = 60


# ---------------------------------------------------------------------------
# (A) Four real children, four different terminal stories, one wait.
# ---------------------------------------------------------------------------


@dataclass
class TerminalFleet:
    """What four concurrently submitted children did, and when.

    Shaped like `conftest.ClusterEvidence`: collection stores its failure
    instead of raising, because a raise inside a module-scoped fixture errors
    every test in the module with one traceback and hides which stage broke.
    """

    job_ids: dict[str, int] = field(default_factory=dict)
    outcomes: dict[int, SlurmJobOutcome] = field(default_factory=dict)
    settled: dict[int, SlurmJobOutcome] = field(default_factory=dict)
    first_terminal_at: dict[int, float] = field(default_factory=dict)
    last_nonterminal_at: dict[int, float] = field(default_factory=dict)
    wait_returned_at: float = 0.0
    failure: BaseException | None = None
    notes: list[str] = field(default_factory=list)

    def outcome(self, label: str) -> SlurmJobOutcome:
        """Return one child's terminal outcome, failing THIS test if collection broke."""
        if self.failure is not None:
            pytest.fail(
                "terminal_fleet collection did not complete: "
                f"{type(self.failure).__name__}: {self.failure}"
            )
        return self.outcomes[self.job_ids[label]]

    def settled_outcome(self, label: str) -> SlurmJobOutcome:
        """Same, but from the re-read that waited for the job name to settle."""
        self.outcome(label)
        return self.settled[self.job_ids[label]]


async def _collect_fleet(fleet: TerminalFleet) -> None:
    # poll_interval drives `wait_for_terminal_job_outcomes`' own loop. 2s keeps
    # the sacct call rate sane over a 70s wait without blurring the staggering
    # this fixture is trying to observe.
    async with slurm_session(poll_interval=2.0) as manager:
        # `exit 0` / `kill -9 $$` are the two endings a batch script can choose
        # for itself. `sleep 300` is cancelled externally below, and the fourth
        # is identical but submitted with a one-minute limit so slurmctld ends
        # it. `--time` is repeated after submit_raw's own `--time=0:10:00`;
        # sbatch takes the last occurrence, verified as TimeLimit=00:01:00 via
        # `scontrol show job`.
        plan = {
            "completed": ("exit 0", ()),
            "signalled": ("kill -9 $$", ()),
            "cancelled": ("sleep 300", ()),
            "timed_out": ("sleep 300", ("--time=0:01",)),
        }
        try:
            for label, (script, extra) in plan.items():
                fleet.job_ids[label] = await submit_raw(
                    f"eval360-life-{label}", script, extra_args=extra
                )
            job_ids = sorted(fleet.job_ids.values())

            # sacct has no row at all for ~1s after submission, and
            # `wait_for_terminal_job_outcomes` raises rather than retrying on a
            # missing row (docs/SLURM_TEST_CLUSTER.md, "Known hazard"). Waiting
            # for the rows to exist first keeps this fixture measuring the
            # terminal wait rather than that documented fragility.
            await wait_until(
                lambda: _rows_for_all(manager, job_ids),
                timeout=60,
                what="an sacct row for all four children",
            )

            # Cancel only once the job is actually RUNNING: cancelling a PENDING
            # job also yields CANCELLED, but then nothing has proved the child
            # was ever dispatched, and the state would be reachable without the
            # node having done anything.
            cancelled_id = fleet.job_ids["cancelled"]
            await wait_until(
                lambda: _in_state(manager, cancelled_id, "RUNNING"),
                timeout=60,
                what=f"job {cancelled_id} to reach RUNNING",
            )
            await scancel(cancelled_id)

            started = time.monotonic()
            observer = asyncio.create_task(
                _observe_first_terminal(manager, job_ids, fleet, started)
            )
            try:
                fleet.outcomes = await asyncio.wait_for(
                    manager.wait_for_terminal_job_outcomes(job_ids),
                    timeout=FLEET_TIMEOUT_SECONDS,
                )
            finally:
                observer.cancel()
                try:
                    await observer
                except asyncio.CancelledError:
                    pass
            fleet.wait_returned_at = time.monotonic() - started

            # Re-read through the harness helper, which waits out the window in
            # which a row is terminal while still carrying the placeholder name
            # `allocation`. Only the name assertions use this; the state
            # assertions use what the production wait actually returned.
            for label, job_id in fleet.job_ids.items():
                if fleet.outcomes[job_id].job_name == "allocation":
                    fleet.notes.append(
                        f"TERMINAL PLACEHOLDER OBSERVED for {label} ({job_id})"
                    )
                fleet.settled[job_id] = await wait_for_terminal(
                    manager, job_id, timeout=60
                )
        except BaseException as error:  # noqa: BLE001
            fleet.failure = error
            for job_id in fleet.job_ids.values():
                await scancel(job_id)


async def _rows_for_all(manager: SlurmManager, job_ids: list[int]):
    outcomes = await manager.get_job_accounting(job_ids)
    return outcomes if len(outcomes) == len(job_ids) else None


async def _in_state(manager: SlurmManager, job_id: int, state: str):
    outcome = (await manager.get_job_accounting([job_id])).get(job_id)
    return outcome if outcome is not None and outcome.state == state else None


async def _observe_first_terminal(
    manager: SlurmManager,
    job_ids: list[int],
    fleet: TerminalFleet,
    started: float,
) -> None:
    """Watch the children independently of the wait being tested.

    Records, for each child, when it was FIRST seen terminal and when it was
    LAST seen non-terminal. Both directions are needed. The fast children are
    already terminal before this starts, so they only ever yield a first-terminal
    time; the slow one is usually still running when the wait returns and the
    observer is cancelled, so it only reliably yields a last-non-terminal time.
    Together they bracket the wait without depending on the observer winning a
    race against it — which it lost on the first run of this test.
    """
    while True:
        outcomes = await manager.get_job_accounting(job_ids)
        now = time.monotonic() - started
        for job_id, outcome in outcomes.items():
            if outcome.is_terminal:
                fleet.first_terminal_at.setdefault(job_id, now)
            else:
                fleet.last_nonterminal_at[job_id] = now
        await asyncio.sleep(0.5)


@pytest.fixture(scope="module")
def terminal_fleet() -> TerminalFleet:
    """Submit four children with four different endings and wait them all out."""
    fleet = TerminalFleet()
    asyncio.run(_collect_fleet(fleet))
    for note in fleet.notes:
        print(f"NOTE: {note}")
    return fleet


def test_a_job_that_exceeds_its_time_limit_is_TIMEOUT(terminal_fleet):
    """Wall-clock exhaustion is its own state, and the exit code cannot see it.

    `TIMEOUT` is in `_TERMINAL_SLURM_STATES` but not COMPLETED, so it is
    terminal and not a clean completion — and critically its ExitCode is `0:0`,
    identical to a successful job's. Any code that judged a child by its exit
    code alone would accept a job Slurm killed for running too long. That is
    exactly why `_reconcile_terminal_jobs` tests `is_clean_completion`, which
    requires `state == "COMPLETED"` as well.
    """
    outcome = terminal_fleet.outcome("timed_out")

    assert outcome.state == "TIMEOUT", f"expected TIMEOUT, got {outcome!r}"
    assert outcome.raw_state == "TIMEOUT"
    assert outcome.exit_code == 0
    assert outcome.signal == 0
    assert outcome.is_terminal
    assert not outcome.is_clean_completion, (
        "a timed-out job has ExitCode 0:0 and must still not count as a clean "
        "completion"
    )


def test_a_job_killed_by_a_signal_reports_the_signal_number(terminal_fleet):
    """The `M` half of sacct's `N:M` ExitCode is parsed, not assumed zero.

    Every existing test sees `0:0` or `3:0`, so `signal` has only ever been
    observed as 0 — a parser that hard-coded it would pass all of them. Here the
    batch script SIGKILLs itself, and sacct reports `0:9`: exit code zero,
    signal nine. The state is FAILED rather than CANCELLED, because the job
    ended by itself rather than being cancelled.

    Note the asymmetry this pins down: `scancel --signal=KILL --full` does NOT
    produce this. Measured on this cluster, that route leaves the root row
    `CANCELLED by 0` with ExitCode `0:0` and puts the signal only on the
    `.batch` step row, which `get_job_accounting` deliberately skips. A
    self-inflicted signal is the only way the root row carries one.
    """
    outcome = terminal_fleet.outcome("signalled")

    assert outcome.state == "FAILED", f"expected FAILED, got {outcome!r}"
    assert outcome.signal == 9, f"expected signal 9, got {outcome!r}"
    assert outcome.exit_code == 0, (
        "SIGKILL sets the signal half of ExitCode, not the exit-code half"
    )
    assert outcome.is_terminal
    assert not outcome.is_clean_completion


def test_a_cancelled_job_keeps_its_decoration_as_evidence(terminal_fleet):
    """`CANCELLED by 0` normalises to `CANCELLED` and the raw form is retained.

    Both halves matter. `state` has to be the bare word or it would never match
    `_TERMINAL_SLURM_STATES` and the terminal wait would spin forever;
    `raw_state` has to keep the decoration, because that is the only record of
    WHO cancelled the job, and `terminal_result.py` re-derives `state` from
    `raw_state` when it verifies a published result.
    """
    outcome = terminal_fleet.outcome("cancelled")

    assert outcome.raw_state.startswith("CANCELLED by "), (
        f"expected an administrator-decorated cancellation, got {outcome!r}"
    )
    assert outcome.state == "CANCELLED"
    assert outcome.is_terminal
    assert not outcome.is_clean_completion
    # The exact re-derivation `_verify_yaml_scheduler_result` performs on the
    # published record. If these ever disagree, a correct run fails to publish.
    assert outcome.raw_state.split(maxsplit=1)[0].removesuffix("+") == outcome.state


def test_normalization_strips_the_truncation_marker_sacct_really_emits(
    terminal_fleet,
):
    """The trailing `+` is a real sacct form — from a width-limited field.

    `_normalize_sacct_state` strips a trailing `+`, and it would be easy to
    assume that is defensive code for a form nothing produces. It is not.
    Asked for `State` WITHOUT `--parsable2`, sacct applies the default 10-column
    width and returns this same cancellation as `CANCELLED+`. Production always
    passes `--parsable2`, which ignores widths — so the `+` never reaches
    `get_job_accounting` today — but the handling is correct for output sacct
    genuinely produces, and this asserts it against the real string rather than
    one typed here.
    """
    job_id = terminal_fleet.job_ids["cancelled"]
    terminal_fleet.outcome("cancelled")  # fail early if collection broke

    truncated = asyncio.run(_sacct_state_with_default_widths(job_id))

    assert truncated.endswith("+"), (
        "expected sacct's default field width to truncate 'CANCELLED by 0'; "
        f"got {truncated!r}. If sacct stopped truncating, this test is no "
        "longer evidence for the '+' branch and should be deleted, not relaxed."
    )
    assert SlurmManager._normalize_sacct_state(truncated) == "CANCELLED"


async def _sacct_state_with_default_widths(job_id: int) -> str:
    """The root row's State as sacct formats it WITHOUT `--parsable2`."""
    process = await asyncio.create_subprocess_exec(
        "sacct", "--noheader", "--jobs", str(job_id), "--format=State",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"sacct failed: {stderr.decode().strip()}")
    # The first line is the root row; later lines are `.batch` steps.
    return stdout.decode().splitlines()[0].strip()


def test_clean_completion_is_narrower_than_terminal(terminal_fleet):
    """All four children are terminal; exactly one is a clean completion.

    This is the distinction `_reconcile_terminal_jobs` hangs its decision on —
    a terminal child is merely finished, and only a clean completion (or an
    explicitly released cancellation) lets a terminal result publish. Asserting
    it across four states Slurm produced, rather than one, is the point.
    """
    labels = ("completed", "signalled", "cancelled", "timed_out")
    outcomes = {label: terminal_fleet.outcome(label) for label in labels}

    assert all(outcome.is_terminal for outcome in outcomes.values())
    clean = {label for label, o in outcomes.items() if o.is_clean_completion}
    assert clean == {"completed"}, f"unexpected clean completions: {clean}"

    assert outcomes["completed"].state == "COMPLETED"
    assert (outcomes["completed"].exit_code, outcomes["completed"].signal) == (0, 0)
    # Four children, four distinct terminal states — the evidence that this
    # fixture is exercising classification rather than one state four times.
    assert {o.state for o in outcomes.values()} == {
        "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
    }


def test_wait_for_terminal_outcomes_waits_for_the_slowest_child(terminal_fleet):
    """One wait, four children, terminal states more than a minute apart.

    `wait_for_terminal_job_outcomes` loops until EVERY requested child is
    terminal. A version that returned on the first terminal row, or that
    stopped polling children it had already seen finish, would still return a
    plausible-looking mapping — so the assertion is on TIMING: the wait must
    not have returned anywhere near the moment the first child finished.

    The timings come from a concurrent observer that queries sacct itself, so
    they are independent of anything the production wait reports. The bound
    used is the last moment the slow child was seen STILL non-terminal rather
    than the moment it went terminal: the observer is cancelled as soon as the
    wait returns, so it cannot be relied on to witness that final transition —
    it did not, on the first run of this test.
    """
    terminal_fleet.outcome("completed")  # fail early if collection broke
    first_terminal = terminal_fleet.first_terminal_at
    last_alive = terminal_fleet.last_nonterminal_at
    assert set(first_terminal) | set(last_alive) == set(
        terminal_fleet.job_ids.values()
    ), f"observer never saw some child at all: {first_terminal} / {last_alive}"

    fast = terminal_fleet.job_ids["completed"]
    slow = terminal_fleet.job_ids["timed_out"]
    assert fast in first_terminal, "the fast child was never seen terminal"
    assert slow in last_alive, "the slow child was never seen running"

    assert last_alive[slow] - first_terminal[fast] > 30, (
        "the children did not actually finish at different times, so this "
        f"proves nothing about the wait: {first_terminal} / {last_alive}"
    )
    assert terminal_fleet.wait_returned_at >= last_alive[slow], (
        "wait_for_terminal_job_outcomes returned while the slowest child was "
        f"still non-terminal (returned at {terminal_fleet.wait_returned_at:.1f}s, "
        f"slow child still running at {last_alive[slow]:.1f}s)"
    )


def test_terminal_rows_settle_on_the_submitted_job_name(terminal_fleet):
    """Every child's accounting row ends up named what was submitted.

    `Scheduler._reconcile_terminal_jobs` compares `outcome.job_name` against the
    ledgered name and refuses to publish on a mismatch, so this is a hard
    requirement rather than cosmetic. See the module docstring of
    `cluster_harness.wait_for_terminal` for why the settled re-read is used
    here: a row can be terminal while still carrying sacct's `allocation`
    placeholder.
    """
    for label, job_id in terminal_fleet.job_ids.items():
        outcome = terminal_fleet.settled_outcome(label)
        assert outcome.job_name == f"eval360-life-{label}"
        assert outcome.job_id_raw == str(job_id)
        assert outcome.job_name != "batch", "a .batch step row was parsed as the job"


# ---------------------------------------------------------------------------
# (A, continued) squeue's elapsed field, and node availability.
# ---------------------------------------------------------------------------


async def test_elapsed_time_is_parsed_from_the_field_squeue_actually_prints():
    """`_to_seconds` against a real `%M`, which for a young job is `M:SS`.

    WHAT THIS CAN AND CANNOT PRODUCE. `_to_seconds` accepts three shapes —
    `MM:SS`, `HH:MM:SS` and `DD-HH:MM:SS` — and squeue chooses between them by
    how long the job has been running. Only the first is reachable in a gating
    test: the second needs a child that has run for an hour and the third needs
    one that has run for a day, and there is no way to make squeue report an
    elapsed time a job has not accumulated. Those two branches are therefore
    NOT covered here, and this docstring is the honest record of that rather
    than a fabricated fixture pretending otherwise.

    The assertion goes through `get_all_jobs`, not `_to_seconds` directly, so
    it also pins the five-way `%j|%i|%T|%M|%N` split that feeds it.
    """
    async with slurm_session() as manager:
        model = make_model(name="ci-elapsed-model")
        await manager.update_allocation([(model, 1)], [])

        job = await wait_until(
            lambda: _one_running_job(manager),
            timeout=QUEUE_VISIBLE_TIMEOUT_SECONDS,
            what="a RUNNING replica",
        )
        raw = await _squeue_elapsed_field(job["job_id"])

        assert re.fullmatch(r"\d+:\d{2}", raw), (
            f"expected squeue's MM:SS elapsed form for a young job, got {raw!r}"
        )
        assert manager._to_seconds(raw) == job["elapsed_time"]
        assert job["elapsed_time"] == int(raw.split(":")[0]) * 60 + int(
            raw.split(":")[1]
        )


async def _one_running_job(manager: SlurmManager):
    jobs = await manager.get_all_jobs()
    running = [job for job in jobs.values() if job["state"] == "RUNNING"]
    return running[0] if running else None


async def _squeue_elapsed_field(job_id: int) -> str:
    process = await asyncio.create_subprocess_exec(
        "squeue", "--noheader", "--jobs", str(job_id), "--format=%M",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"squeue failed: {stderr.decode().strip()}")
    return stdout.decode().strip()


async def test_a_drained_or_down_node_is_not_reported_as_available():
    """`get_available_nodes` counts only `idle`, which is what capacity means.

    The scheduler uses this number to decide how many replicas it may ask for.
    A drained node still exists, still appears in sinfo, and still reports a
    node count of 1 — it simply will not accept new work. If `get_available_nodes`
    counted it, the scheduler would submit jobs that sit PENDING forever and
    read that as deployment being slow.

    Both administrative states are covered because sinfo spells them
    differently (`drained` vs `down`) and the function matches the state string
    exactly. The node is restored in a `finally`, without which every later
    test in the session would fail for an unrelated reason.
    """
    async with slurm_session() as manager:
        node = await _only_node()
        assert await manager.get_available_nodes() == 1, (
            "the cluster was not idle before this test drained it"
        )
        for state, expected_sinfo in (("DRAIN", "drained"), ("DOWN", "down")):
            try:
                await _set_node_state(node, state, reason="eval360-lifecycle-test")
                await wait_until(
                    lambda expected=expected_sinfo: _sinfo_state_is(expected),
                    timeout=30,
                    what=f"node {node} to report {expected_sinfo}",
                )
                assert await manager.get_available_nodes() == 0, (
                    f"a {expected_sinfo} node was counted as available capacity"
                )
            finally:
                await _set_node_state(node, "RESUME")
                await wait_until(
                    lambda: _sinfo_state_is("idle"),
                    timeout=60,
                    what=f"node {node} to return to idle",
                )
        assert await manager.get_available_nodes() == 1


async def _only_node() -> str:
    process = await asyncio.create_subprocess_exec(
        "sinfo", "-h", "-N", "-o", "%N", "-p", "ci",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"sinfo failed: {stderr.decode().strip()}")
    names = [line.strip() for line in stdout.decode().splitlines() if line.strip()]
    assert len(names) == 1, f"this suite assumes a single-node cluster, saw {names}"
    return names[0]


async def _set_node_state(node: str, state: str, *, reason: str | None = None) -> None:
    command = ["scontrol", "update", f"NodeName={node}", f"State={state}"]
    if reason is not None:
        # Slurm rejects DRAIN/DOWN without a reason, and accepts RESUME only
        # without one.
        command.append(f"Reason={reason}")
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"scontrol update failed: {stderr.decode().strip()}")


async def _sinfo_state_is(expected: str):
    process = await asyncio.create_subprocess_exec(
        "sinfo", "-h", "-p", "ci", "-o", "%T|%D",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()
    states = [
        line.split("|")[0] for line in stdout.decode().splitlines() if "|" in line
    ]
    return expected if expected in states else None


# ---------------------------------------------------------------------------
# (B) The terminal-evidence ledger.
# ---------------------------------------------------------------------------


async def test_submitted_children_are_ledgered_in_job_id_order():
    """`get_submitted_jobs` returns the real children, ordered, fully attributed.

    `_reconcile_terminal_jobs` iterates this tuple and builds one published
    record per entry, so every field asserted here ends up in a terminal-result
    file. Two replicas rather than one, because a single entry cannot
    distinguish "ordered by job ID" from "whatever order it happened to be
    inserted in", and because `update_allocation`'s replica indices are part of
    the name it ledgers.
    """
    async with slurm_session() as manager:
        model = make_model(name="ci-ledger-model")
        await manager.update_allocation([(model, 2)], [])

        jobs = manager.get_submitted_jobs()
        assert len(jobs) == 2, f"expected two ledgered replicas, got {jobs!r}"
        assert [job.job_id for job in jobs] == sorted(job.job_id for job in jobs)
        for replica_index, job in enumerate(jobs):
            assert job.kind == "model_serving"
            assert job.submission_origin == "submitted"
            assert job.model_name == "ci-ledger-model"
            assert job.serving_key == model.serving_key
            assert job.cancellation_intent is None
            assert job.completed_event_roles == ()
            assert job.job_name == manager._to_job_name(
                "ci-ledger-model", model.serving_key, replica_index
            )
        # The ledgered IDs must be the IDs Slurm knows about, not something
        # `_parse_sbatch_job_id` invented from `sbatch --parsable` stdout.
        queued = await wait_until(
            lambda: _queued_jobs(manager, 2),
            timeout=QUEUE_VISIBLE_TIMEOUT_SECONDS,
            what="both replicas to become visible to squeue",
        )
        assert {job.job_id for job in jobs} == {
            entry["job_id"] for entry in queued.values()
        }


async def _queued_jobs(manager: SlurmManager, count: int):
    jobs = await manager.get_all_jobs()
    return jobs if len(jobs) == count else None


async def test_the_squeue_poll_loop_cannot_downgrade_a_submitted_child():
    """Re-observing a child merges into its entry instead of replacing it.

    This is not a hypothetical: `get_all_jobs` calls `_remember_job` for every
    active job it sees, with `submission_origin="adopted"`, and the scheduler
    calls `get_all_jobs` on every poll tick. So a child submitted by THIS
    scheduler is re-offered to the ledger every few seconds for its whole life.
    If that overwrote the entry, every terminal result would claim its own
    children were adopted from somewhere else, and any role completions already
    bound to them would be lost.
    """
    async with slurm_session() as manager:
        model = make_model(name="ci-reobserved-model")
        await manager.update_allocation([(model, 1)], [])
        (original,) = manager.get_submitted_jobs()

        # The re-observation only happens once squeue can see the job, so wait
        # for that rather than asserting against three no-op polls.
        await wait_until(
            lambda: _any_queued_job(manager),
            timeout=QUEUE_VISIBLE_TIMEOUT_SECONDS,
            what="the serving child to become visible to squeue",
        )
        for _ in range(3):
            await manager.get_all_jobs()
            await asyncio.sleep(0.5)

        jobs = manager.get_submitted_jobs()
        assert len(jobs) == 1, f"re-observation duplicated the child: {jobs!r}"
        assert jobs[0] == original, (
            "a squeue re-observation changed the ledgered child: "
            f"{original!r} -> {jobs[0]!r}"
        )
        assert jobs[0].submission_origin == "submitted"


async def test_a_job_id_that_changes_identity_is_rejected():
    """The ledger refuses to let one job ID mean two different children.

    WHY THIS IS REACHED THROUGH `_remember_job` RATHER THAN A PUBLIC CALL.
    slurmctld does not reuse a job ID while it is running, so within one
    scheduler invocation there is no sequence of real submissions that makes
    this happen — which is precisely why the guard exists: it protects against
    a bug elsewhere (a ledger keyed wrongly, an ID parsed from the wrong
    stdout), not against Slurm. The job ID and name here are a REAL submitted
    child's, so the guard is tested against genuine ledger contents; only the
    conflicting second observation is constructed.

    Both discriminators are checked. `kind` matters because
    `_reconcile_terminal_jobs` rejects any child that is not `model_serving`,
    so a silently-changed kind would turn a publishable run into a crash at the
    very end.
    """
    async with slurm_session() as manager:
        model = make_model(name="ci-identity-model")
        await manager.update_allocation([(model, 1)], [])
        (job,) = manager.get_submitted_jobs()

        with pytest.raises(RuntimeError, match="changed identity"):
            manager._remember_job(
                SubmittedSlurmJob(
                    job_id=job.job_id,
                    job_name=job.job_name + "-renamed",
                    kind=job.kind,
                    submission_origin="adopted",
                    model_name=job.model_name,
                    serving_key=job.serving_key,
                )
            )
        with pytest.raises(RuntimeError, match="changed identity"):
            manager._remember_job(
                SubmittedSlurmJob(
                    job_id=job.job_id,
                    job_name=job.job_name,
                    kind="imported_dataset",
                    submission_origin="submitted",
                    model_name=job.model_name,
                    serving_key=job.serving_key,
                )
            )
        assert manager.get_submitted_jobs() == (job,), (
            "a rejected observation must leave the ledger untouched"
        )


async def test_capture_cannot_be_restarted_over_a_populated_ledger():
    """`begin_terminal_result_capture` is once per invocation, and says so.

    It clears `_submitted_jobs`, so a second call part-way through a run would
    silently discard the children already submitted and every role completion
    bound to them — and the resulting terminal result would look complete.
    """
    async with slurm_session() as manager:
        model = make_model(name="ci-recapture-model")
        await manager.update_allocation([(model, 1)], [])
        before = manager.get_submitted_jobs()
        assert before, "nothing was ledgered, so this would pass vacuously"

        with pytest.raises(RuntimeError, match="already active"):
            manager.begin_terminal_result_capture()

        assert manager.get_submitted_jobs() == before


async def test_role_completion_binds_events_to_the_active_serving_child():
    """A successful event role attaches to whichever child is serving it.

    `record_role_completion` resolves the child by SERVING KEY through a live
    squeue query — it does not take a job ID — so this only works if the job
    name round-tripped through Slurm still parses back to the same serving key.
    That is the whole chain `_JOB_NAME_RE` exists for, and it is what
    `_reconcile_terminal_jobs` later re-checks when it validates that a child's
    serving key matches the model of the event it claims to have served.

    Both roles are recorded for the same child because one serving replica
    legitimately serves generation and LLM-as-judge grading for the same event,
    and the ledger has to accumulate rather than replace.
    """
    async with slurm_session() as manager:
        model = make_model(name="ci-roles-model")
        await manager.update_allocation([(model, 1)], [])
        await wait_until(
            lambda: _any_queued_job(manager),
            timeout=QUEUE_VISIBLE_TIMEOUT_SECONDS,
            what="the serving child to become visible to squeue",
        )
        event_uuid = str(uuid.uuid4())

        await manager.record_role_completion(model.serving_key, event_uuid, "generation")
        (job,) = manager.get_submitted_jobs()
        assert job.completed_event_roles == ((event_uuid, "generation"),)

        await manager.record_role_completion(model.serving_key, event_uuid, "grading")
        (job,) = manager.get_submitted_jobs()
        assert job.completed_event_roles == (
            (event_uuid, "generation"),
            (event_uuid, "grading"),
        )

        # Recording the same role twice is what a resumed run does; it must be
        # idempotent rather than duplicating the completion, because
        # `terminal_result.py` rejects a duplicated (event, role) link.
        await manager.record_role_completion(model.serving_key, event_uuid, "grading")
        (job,) = manager.get_submitted_jobs()
        assert len(job.completed_event_roles) == 2


async def test_role_completion_refuses_unknown_roles_and_absent_children():
    """The two ways binding a role can be wrong both raise, loudly.

    An unsupported role would sail through the ledger and only be caught at
    publish time by `_reconcile_terminal_jobs`, after the compute is spent. A
    missing serving child means the caller believes an event was served by a
    job that is not running — recording that would fabricate evidence, so it
    must fail rather than no-op.
    """
    async with slurm_session() as manager:
        model = make_model(name="ci-badroles-model")
        await manager.update_allocation([(model, 1)], [])
        await wait_until(
            lambda: _any_queued_job(manager),
            timeout=QUEUE_VISIBLE_TIMEOUT_SECONDS,
            what="the serving child to become visible to squeue",
        )

        with pytest.raises(ValueError, match="unsupported serving role"):
            await manager.record_role_completion(
                model.serving_key, str(uuid.uuid4()), "aggregation"
            )

        # A well-formed serving key that belongs to no running job: the shape
        # `_JOB_NAME_RE` would accept, so the lookup fails for the right reason.
        with pytest.raises(RuntimeError, match="no active serving child"):
            await manager.record_role_completion(
                "0123456789ab", str(uuid.uuid4()), "generation"
            )

        (job,) = manager.get_submitted_jobs()
        assert job.completed_event_roles == ()


async def _any_queued_job(manager: SlurmManager):
    jobs = await manager.get_all_jobs()
    return jobs or None


async def test_release_cancels_the_serving_child_and_records_the_intent():
    """The release path, end to end, including the field that legitimises it.

    `_reconcile_terminal_jobs` accepts a CANCELLED child as SUCCESSFUL only
    when `cancellation_intent == "scheduler_release"` — that pair is the
    difference between "the scheduler shut its own replica down after the work
    was done" and "something killed our job". So the intent is not bookkeeping;
    it is the evidence. This asserts the intent, and then that the child really
    did reach CANCELLED, and finally that the exact conjunction
    `_reconcile_terminal_jobs` evaluates holds.

    The second release call is part of the test because the scheduler can
    release more than once (per-model teardown and then invocation teardown);
    it must not re-cancel a job or disturb the ledger.
    """
    async with slurm_session() as manager:
        model = make_model(name="ci-release-model")
        await manager.update_allocation([(model, 1)], [])
        await wait_until(
            lambda: _any_queued_job(manager),
            timeout=QUEUE_VISIBLE_TIMEOUT_SECONDS,
            what="the serving child to become visible to squeue",
        )
        event_uuid = str(uuid.uuid4())
        await manager.record_role_completion(model.serving_key, event_uuid, "generation")
        (submitted,) = manager.get_submitted_jobs()

        await manager.release_submitted_model_serving_jobs()

        (released,) = manager.get_submitted_jobs()
        assert released.cancellation_intent == "scheduler_release"
        assert released.job_id == submitted.job_id
        assert released.completed_event_roles == ((event_uuid, "generation"),)

        outcome = await wait_for_terminal(manager, released.job_id)
        assert outcome.state == "CANCELLED"
        assert outcome.job_name == released.job_name

        # The literal predicate from Scheduler._reconcile_terminal_jobs.
        expected_release = (
            released.kind == "model_serving"
            and released.cancellation_intent == "scheduler_release"
            and outcome.state == "CANCELLED"
            and bool(released.completed_event_roles)
        )
        assert expected_release, (
            "a released serving child would be reported as an unsuccessful "
            f"terminal child: {released!r} / {outcome!r}"
        )

        await manager.release_submitted_model_serving_jobs()
        assert manager.get_submitted_jobs() == (released,)


async def test_release_leaves_a_child_that_already_finished_alone():
    """Release consults sacct first and does not cancel a terminal child.

    A serving replica can end before the scheduler releases it — it crashed, it
    hit its time limit, an operator cancelled it. `release_submitted_model_serving_jobs`
    queries accounting precisely so it does not scancel those, and the visible
    consequence is that their `cancellation_intent` stays unset. That matters
    downstream: `_reconcile_terminal_jobs` would otherwise be handed a child
    labelled as a deliberate release when in fact nobody released it, and would
    accept a dead replica as a successful one.

    The child here is cancelled WITHOUT an intent, which is exactly what
    `cancel_job` does when the caller has no release to declare.
    """
    async with slurm_session() as manager:
        model = make_model(name="ci-already-done-model")
        await manager.update_allocation([(model, 1)], [])
        await wait_until(
            lambda: _any_queued_job(manager),
            timeout=QUEUE_VISIBLE_TIMEOUT_SECONDS,
            what="the serving child to become visible to squeue",
        )
        (job,) = manager.get_submitted_jobs()

        await manager.cancel_job(job.job_id)
        outcome = await wait_for_terminal(manager, job.job_id)
        assert outcome.is_terminal

        await manager.release_submitted_model_serving_jobs()

        (after,) = manager.get_submitted_jobs()
        assert after.cancellation_intent is None, (
            "release invented a scheduler_release intent for a child it never "
            "cancelled"
        )
        assert after == job
        # And nothing was resurrected or re-queued by the release attempt.
        await drain(manager, timeout=30)
