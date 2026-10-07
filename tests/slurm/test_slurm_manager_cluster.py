"""SlurmManager against a real slurmd, with nothing patched.

What this tests:
    Every place `scheduler/slurm_manager.py` parses output from a Slurm CLI —
    sbatch, squeue, sinfo and sacct — meeting output that Slurm actually
    produced, plus the full submit-to-live path through the real
    `scheduler/slurm/sbatch_script.sh`.

Why this exists:
    These parsers had no cluster coverage at all. `tests/test_job_manager.py`
    patches `asyncio.create_subprocess_exec` and feeds them strings written by
    the same person who wrote the parsers; `tests/fake_slurm.py` replaces
    `SlurmManager` outright with an in-memory reimplementation that duplicates
    its logic, so both copies can be wrong in the same way and agree. Neither
    can catch a format assumption that Slurm does not share, and neither
    executes the sbatch scripts at all.

    Concretely, these are the assumptions under test: `--format %j|%i|%T|%M|%N`
    splits into exactly five fields; a Slurm elapsed field parses as MM:SS /
    HH:MM:SS / DD-HH:MM:SS; a job name survives a round trip through Slurm and
    still matches `_JOB_NAME_RE`; `sbatch --parsable` prints a bare job ID;
    `sacct --parsable2` emits exactly five columns with NO trailing delimiter
    (measured: four `|` per row, for both the root row and its `.batch` step —
    the six-field branch in `get_job_accounting` is defensive, and is what
    `--parsable` would produce); and a node name from `%N` resolves as a
    hostname for the health check.

Corner cases covered:
    The job is submitted through `update_allocation`, the production entry
    point, so the argv and the base64 `vllm_args` round trip are the real ones.
    The health check has to resolve the node name Slurm chose rather than
    localhost. The terminal-accounting assertions run after a real `scancel`,
    which is the state production actually reconciles against. The sacct
    placeholder window (see conftest) is recorded but deliberately not
    asserted, because nothing reads that row before the job is terminal.

What this cannot cover, and why:
    Container jobs (`--container-image`, `sbatch_script_container.sh`) need
    pyxis/enroot, which is not part of vanilla Slurm — those stay argv-only
    tested in `tests/test_container_support.py`. cgroup accounting is disabled
    in the test cluster because a container cannot give slurmd a systemd scope.
    VLLM itself is replaced by `scripts/ci/fake_vllm_server.py`; everything
    between the scheduler and it is real.

Run with:
    scripts/slurm_cluster.sh up
    scripts/slurm_cluster.sh test tests/slurm -v
"""

from __future__ import annotations

import pytest

from scheduler.slurm_manager import SlurmJobOutcome

pytestmark = pytest.mark.cluster


# ---------------------------------------------------------------------------
# sinfo
# ---------------------------------------------------------------------------


def test_get_available_nodes_parses_real_sinfo(cluster_evidence):
    """`sinfo -h -p ci -o %T|%D` yields an idle-node count."""
    idle_nodes = cluster_evidence.get("sinfo")
    assert isinstance(idle_nodes, int)
    # The suite cannot run at all without somewhere to run; asserting >= 1 keeps
    # a cluster whose node never reached `idle` from looking like a pass.
    assert idle_nodes >= 1, "no idle node — the cluster did not come up"


# ---------------------------------------------------------------------------
# sbatch
# ---------------------------------------------------------------------------


def test_update_allocation_submits_and_parses_the_job_id(cluster_evidence):
    """`_parse_sbatch_job_id` handles what `sbatch --parsable` really prints."""
    submitted = cluster_evidence.get("sbatch")
    assert len(submitted) == 1, f"expected one ledgered child, got {submitted}"
    job = submitted[0]
    assert job.job_id > 0
    assert job.kind == "model_serving"
    assert job.model_name == "ci-stub-model"
    assert job.serving_key == cluster_evidence.serving_key


def test_serving_resource_args_are_accepted_by_real_sbatch(cluster_evidence):
    """The resource request Slurm accepted is the one the model asked for.

    `--gres=gpu:N` is always emitted and cannot be zero, so this is also the
    assertion that the cluster's GPU advertisement works at all.
    """
    args = cluster_evidence.resource_args
    assert "--nodes=1" in args
    assert "--ntasks=1" in args
    assert "--gres=gpu:1" in args
    assert "--cpus-per-task=1" in args
    assert "--time=0:20:00" in args
    assert "--mem=1G" in args
    # Reaching a ledgered submission means sbatch did not reject any of them.
    assert cluster_evidence.get("sbatch")


# ---------------------------------------------------------------------------
# squeue
# ---------------------------------------------------------------------------


def test_get_all_jobs_parses_a_real_squeue_row(cluster_evidence):
    """The five-field `%j|%i|%T|%M|%N` split survives real output."""
    record = cluster_evidence.get("squeue")
    assert record["state"] == "RUNNING"
    assert isinstance(record["job_id"], int)
    assert record["job_id"] > 0


def test_job_name_round_trips_through_slurm(cluster_evidence):
    """`_to_job_name` → Slurm → `_JOB_NAME_RE` recovers the identity.

    The name carries the instance ID, serving key and replica index, and is the
    only thing tying a queue entry back to a model. A name Slurm truncated or
    rewrote would silently orphan every job this scheduler owns.
    """
    record = cluster_evidence.get("squeue")
    assert record["serving_key"] == cluster_evidence.serving_key
    assert record["replica_index"] == 0
    assert record["model_name"] == "ci-stub-model"
    assert cluster_evidence.instance_id in record["name"]


def test_elapsed_time_parses_from_a_real_slurm_duration(cluster_evidence):
    """`_to_seconds` handles the elapsed field Slurm emits for `%M`."""
    record = cluster_evidence.get("squeue")
    assert isinstance(record["elapsed_time"], int)
    assert record["elapsed_time"] >= 0


def test_nodelist_is_a_resolvable_node_not_a_pending_reason(cluster_evidence):
    """`%N` gives a node name, not a bracketed reason like `(Resources)`.

    `get_model_state` builds `http://{nodelist}:8000/health` from this field
    without validating it, so a pending-reason string here would become a
    health-check URL that can never resolve.
    """
    record = cluster_evidence.get("squeue")
    assert record["nodelist"]
    assert not record["nodelist"].startswith("(")


# ---------------------------------------------------------------------------
# The full path: sbatch script → fake vllm → health discovery
# ---------------------------------------------------------------------------


def test_model_reaches_live_through_the_real_sbatch_script(cluster_evidence):
    """End to end: Slurm ran `sbatch_script.sh` and the scheduler found it.

    Reaching `live` proves the whole chain: the script activated `$venv_path`,
    base64-decoded `$vllm_args` into an argv, exec'd `vllm serve`, and the
    scheduler health-checked the node name Slurm assigned. Any broken link
    leaves the model in `deploying` instead.
    """
    live = cluster_evidence.get("health")
    assert live, "model never became live — see the Slurm job output"
    names = [name for name, _ in live]
    urls = [url for _, url in live]
    assert names == ["ci-stub-model"]
    assert urls == [f"http://{cluster_evidence.get('squeue')['nodelist']}:8000"]


# ---------------------------------------------------------------------------
# sacct
# ---------------------------------------------------------------------------


def test_get_job_accounting_returns_a_parsed_root_row(cluster_evidence):
    """sacct answers, and `--parsable2` splits into the five expected columns.

    This needs slurmdbd and a real accounting database, which is why the test
    cluster runs both. Without accounting, sacct returns nothing and
    `wait_for_terminal_job_outcomes` raises rather than degrading.
    """
    outcomes = cluster_evidence.get("sacct_active")
    job_id = cluster_evidence.get("squeue")["job_id"]
    assert job_id in outcomes
    outcome = outcomes[job_id]
    assert isinstance(outcome, SlurmJobOutcome)
    assert outcome.job_id_raw == str(job_id)
    assert outcome.exit_code == 0
    assert outcome.signal == 0


def test_cancelled_job_reaches_a_terminal_accounting_state(cluster_evidence):
    """`wait_for_terminal_job_outcomes` resolves after a real scancel."""
    outcome = cluster_evidence.get("sacct_terminal")
    assert outcome.is_terminal
    assert outcome.state == "CANCELLED"
    # `_normalize_sacct_state` has to strip the decoration Slurm appends —
    # the raw state is "CANCELLED by <uid>", which is not a state name.
    assert outcome.raw_state.startswith("CANCELLED")
    assert outcome.state in outcome.raw_state


def test_terminal_row_converges_on_the_submitted_job_name(cluster_evidence):
    """A settled sacct row names the job that was submitted.

    This is the contract `Scheduler` depends on: it raises "sacct job name ...
    does not match the submitted child identity" on a mismatch.

    NOTE the word "converges". Being terminal is NOT sufficient — slurmdbd
    populates state and name independently, so a row can read
    `allocation|CANCELLED by 0` for up to ~1.5s after a scancel. The fixture
    waits that out before this assertion; production does not, which is a real
    hazard for short-lived children and is recorded in
    docs/SLURM_TEST_CLUSTER.md. The assertion is deliberately NOT weakened to
    accept the placeholder: the name genuinely does converge, and that is what
    is worth guaranteeing.
    """
    outcome = cluster_evidence.get("sacct_terminal")
    assert outcome.job_name == cluster_evidence.get("squeue")["name"]


def test_ledgered_child_matches_the_accounting_row(cluster_evidence):
    """Submission ledger and accounting agree on job ID and name.

    This pairing is how `Scheduler` attributes an sacct outcome back to the
    child it submitted; if the two ever disagreed, terminal results would be
    attributed to the wrong job rather than failing loudly.
    """
    ledger = cluster_evidence.get("ledger")
    outcome = cluster_evidence.get("sacct_terminal")
    assert outcome.job_id in ledger
    assert ledger[outcome.job_id].job_name == outcome.job_name
