"""sacct parsing, against rows slurmdbd actually wrote.

What this tests:
    `get_job_accounting` turning `sacct --noheader --parsable2
    --format=JobIDRaw,JobName%256,State,ExitCode,Reason` into `SlurmJobOutcome`,
    and `is_clean_completion` / `is_terminal` classifying the result.

Why this exists:
    This is the newest load-bearing parser in `slurm_manager.py` and the one
    with the most ways to be quietly wrong. It hand-splits on `|`, pops a
    trailing empty field, regex-matches ExitCode as `N:M`, normalises decorated
    states like `CANCELLED by 0`, and skips non-numeric JobIDRaw values. Every
    one of those is a guess about sacct's output format, and every one was
    previously checked only against strings written next to the code that
    parses them.

    It is also the parser whose failure is most expensive: terminal results
    refuse to publish when it disagrees, so a format surprise does not degrade
    the run, it fails it at the very end.

Corner cases covered:
    A job that succeeds and one that exits non-zero, so ExitCode is exercised
    with something other than `0:0`; the `.batch` step rows sacct returns
    alongside every root row, which must not be mistaken for jobs; several job
    IDs in one query; and an ID that does not exist, which must come back empty
    rather than raising.
"""

from __future__ import annotations

import pytest

from .cluster_harness import (
    make_model,
    slurm_session,
    submit_raw,
    wait_for_terminal,
    wait_until,
)

pytestmark = pytest.mark.cluster


async def test_successful_job_reports_a_clean_completion():
    """Exit 0 means COMPLETED, `0:0`, and `is_clean_completion`."""
    async with slurm_session() as manager:
        job_id = await submit_raw("eval360-acct-ok", "exit 0")
        outcome = await wait_for_terminal(manager, job_id)

        assert outcome.state == "COMPLETED"
        assert outcome.exit_code == 0
        assert outcome.signal == 0
        assert outcome.is_terminal
        assert outcome.is_clean_completion


async def test_failed_job_reports_its_exit_code():
    """A non-zero exit is parsed out of the `N:M` ExitCode field.

    `0:0` is the only ExitCode most tests ever see, and it is the one value
    that cannot distinguish a working parser from one that returns zeros.
    """
    async with slurm_session() as manager:
        job_id = await submit_raw("eval360-acct-fail", "exit 3")
        outcome = await wait_for_terminal(manager, job_id)

        assert outcome.state == "FAILED"
        assert outcome.exit_code == 3, f"expected exit 3, got {outcome!r}"
        assert outcome.signal == 0
        assert outcome.is_terminal
        assert not outcome.is_clean_completion


async def test_step_rows_are_not_mistaken_for_jobs():
    """sacct returns `N` and `N.batch`; only the root row is an outcome.

    `get_job_accounting` relies on `JobIDRaw` being non-decimal for step rows
    (`5.batch`) and skips them. If that filter broke, the step row would
    overwrite the root row — and since the step's JobName is the literal string
    `batch`, every terminal-result name check would fail.
    """
    async with slurm_session() as manager:
        job_id = await submit_raw("eval360-acct-steps", "exit 0")
        outcome = await wait_for_terminal(manager, job_id)

        outcomes = await manager.get_job_accounting([job_id])
        assert list(outcomes) == [job_id]
        assert outcome.job_name == "eval360-acct-steps"
        assert outcome.job_name != "batch", "a .batch step row was parsed as the job"
        assert outcome.job_id_raw == str(job_id)


async def test_several_jobs_are_returned_from_one_query():
    """One sacct call covers every requested ID and keys them correctly."""
    async with slurm_session() as manager:
        first = await submit_raw("eval360-acct-multi-a", "exit 0")
        second = await submit_raw("eval360-acct-multi-b", "exit 1")

        await wait_for_terminal(manager, first)
        await wait_for_terminal(manager, second)

        outcomes = await manager.get_job_accounting([first, second])
        assert set(outcomes) == {first, second}
        assert outcomes[first].job_name == "eval360-acct-multi-a"
        assert outcomes[second].job_name == "eval360-acct-multi-b"
        assert outcomes[first].exit_code == 0
        assert outcomes[second].exit_code == 1


async def test_duplicate_ids_are_collapsed_not_rejected():
    """`get_job_accounting` de-duplicates its input before querying.

    It raises "duplicate root sacct row" if it ever sees the same job twice in
    one response, so the de-duplication of the REQUEST is what keeps a caller
    passing `[id, id]` from tripping an error about sacct.
    """
    async with slurm_session() as manager:
        job_id = await submit_raw("eval360-acct-dupe", "exit 0")
        await wait_for_terminal(manager, job_id)

        outcomes = await manager.get_job_accounting([job_id, job_id, job_id])
        assert list(outcomes) == [job_id]


async def test_an_unknown_job_id_returns_no_row():
    """A job that never existed yields an empty mapping, not an exception.

    This is what `wait_for_terminal_job_outcomes` turns into its "sacct
    returned no root accounting row" error, so the empty result has to be
    reachable rather than masked by a parse failure.
    """
    async with slurm_session() as manager:
        outcomes = await manager.get_job_accounting([999_999])
        assert outcomes == {}


async def test_empty_request_does_not_call_sacct():
    async with slurm_session() as manager:
        assert await manager.get_job_accounting([]) == {}


async def test_wait_for_terminal_outcomes_raises_for_a_job_that_never_existed():
    """The missing-row path is an error, not a silent pass.

    Terminal evidence is only worth anything if an absent row fails loudly;
    this asserts the raise rather than trusting the docstring.
    """
    async with slurm_session() as manager:
        with pytest.raises(RuntimeError, match="no root accounting row"):
            await manager.wait_for_terminal_job_outcomes([999_998])


async def test_a_running_job_is_not_terminal():
    """`is_terminal` distinguishes an in-flight job from a finished one.

    `wait_for_terminal_job_outcomes` loops on exactly this predicate, so a
    RUNNING job being classified terminal would end the wait early and report
    an unfinished job as an outcome.
    """
    async with slurm_session() as manager:
        job_id = await submit_raw("eval360-acct-running", "sleep 120")

        async def _running_row():
            outcomes = await manager.get_job_accounting([job_id])
            outcome = outcomes.get(job_id)
            return outcome if outcome and outcome.state == "RUNNING" else None

        outcome = await wait_until(
            _running_row, timeout=60, what="a RUNNING accounting row"
        )
        assert not outcome.is_terminal
        assert not outcome.is_clean_completion

        from .cluster_harness import scancel

        await scancel(job_id)


# ---------------------------------------------------------------------------
# Known-bug coverage: sacct lag is not tolerated
# ---------------------------------------------------------------------------


async def test_terminal_outcomes_tolerate_sacct_lag():
    """`wait_for_terminal_job_outcomes` must survive a freshly submitted child.

    THE BUG THIS PINS, in two symptoms with one cause: slurmdbd publishes a
    job's accounting row in stages, and the method tolerates neither stage.

      1. For roughly the first second after submission there is NO row at all.
         The method raises "sacct returned no root accounting row" on its very
         first query, with no retry.
      2. The row then appears carrying a PLACEHOLDER name, `allocation`, and —
         critically — it can already be TERMINAL. The method returns as soon as
         `is_terminal` is true, so the caller gets a row whose name is not the
         job's. `Scheduler._reconcile_terminal_jobs` compares that name against
         the submitted child and raises "sacct job name ... does not match the
         submitted child identity".

    Net effect in production: a short-lived child can fail an otherwise
    successful evaluation's terminal result for a reason that has nothing to do
    with the job. Measured at up to ~1.5s after a scancel, reproduced 3/3, and
    hit from three separate call sites while building this suite.

    The cancel is deliberately immediate — no wait — because that is what puts
    the query inside the window. The suite's other tests wait the window out
    (see `cluster_harness.wait_for_terminal`); this one must not, because the
    window is the subject.
    """
    async with slurm_session() as manager:
        model = make_model()
        await manager.update_allocation([(model, 1)], [])
        submitted = manager.get_submitted_jobs()
        assert len(submitted) == 1
        child = submitted[0]

        # Straight into the lag window.
        await manager.cancel_all_owned_jobs()

        outcomes = await manager.wait_for_terminal_job_outcomes([child.job_id])
        outcome = outcomes[child.job_id]

        assert outcome.is_terminal
        assert outcome.job_name == child.job_name, (
            f"terminal outcome carries {outcome.job_name!r} instead of the "
            f"submitted name {child.job_name!r}; Scheduler would raise "
            "'sacct job name does not match the submitted child identity'"
        )
