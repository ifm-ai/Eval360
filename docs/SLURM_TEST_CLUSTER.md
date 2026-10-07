# The Slurm test cluster

A real single-node Slurm cluster in a container, so `scheduler/slurm_manager.py`
can be tested against actual `sbatch`/`squeue`/`sinfo`/`sacct` instead of against
mocks. This is to Slurm what `kind` is to Kubernetes.

```bash
scripts/slurm_cluster.sh up                      # ~20s warm, ~6min first build
scripts/slurm_cluster.sh test tests/slurm -v     # the suite
scripts/slurm_cluster.sh probe                   # same path, by hand, verbose
scripts/slurm_cluster.sh down
```

`up` · `probe` · `test` · `shell` · `exec` · `status` · `logs` · `down` · `rebuild` · `image-tag`.

## Why it exists

The Slurm boundary had no cluster coverage. Three layers of substitute stood in
for it, none of which can catch a format assumption Slurm does not share — plus
one layer nothing touched at all:

| Substitute | What it does | What it cannot catch |
|---|---|---|
| `tests/fake_slurm.py` | Replaces `SlurmManager` wholesale, reimplementing `get_unneeded_models` and state classification | Both copies can be wrong the same way and agree |
| `tests/test_job_manager.py` | Patches `asyncio.create_subprocess_exec`, feeds canned `squeue` strings | The fixtures were written by whoever wrote the parser |
| `tests/test_container_support.py` | Captures the `sbatch` argv | Never checks real `sbatch` accepts it |
| `scheduler/slurm/*.sh` | — | Never executed by anything |

The suite in `tests/slurm/` puts each parser in front of output Slurm actually
produced: the five-field `%j|%i|%T|%M|%N` split, `_to_seconds` on a real elapsed
field, `_JOB_NAME_RE` on a name that round-tripped through Slurm,
`_parse_sbatch_job_id` on real `sbatch --parsable` stdout, and
`get_job_accounting` on real `sacct --parsable2` rows.

## What the suite covers

106 tests, ~8 minutes, ten modules:

| Module | Covers |
|---|---|
| `test_slurm_manager_cluster.py` | One job's full lifecycle: sinfo, sbatch, squeue parsing, health discovery, sacct |
| `test_slurm_allocation.py` | Replica indices, poll-loop idempotency, scale-up, targeted cancellation, `get_unneeded_models` |
| `test_slurm_isolation.py` | The multi-tenancy promise: two instances sharing a queue must not see or cancel each other's jobs |
| `test_slurm_accounting.py` | sacct parsing: exit codes, `.batch` step-row filtering, multi-job queries, missing rows |
| `test_slurm_failure_modes.py` | PENDING rows, a replica that never serves becoming `dead`, external cancellation, sbatch rejection |
| `test_sbatch_scripts.py` | `scheduler/slurm/*.sh` actually executing: argv round trip, venv shapes, cache isolation, sentinels |
| `test_job_lifecycle.py` | TIMEOUT, signal exit codes, node drain, and the terminal-evidence ledger |
| `test_imported_dataset_cluster.py` | `submit_imported_dataset_job` and the three polling helpers, end to end |
| `test_scheduler_e2e.py` | The real `Scheduler` driving a real job to real generations/grades/scores |
| `test_cluster_teardown_guard.py` | That the harness FAILS CLOSED — a broken `squeue` must stop the reaper, not license it to kill |

## Bugs and hazards this found

All verified against the cluster; none fixed here, because changing production
behaviour does not belong in a test change.

1. **`get_job_node` never terminates for a job that has left the queue.** It
   loops while stdout is empty and ignores `returncode` entirely, and
   `handle_imported_dataset_event` awaits it with no timeout. Both exits are
   reachable: a purged job gives `rc=1` with `slurm_load_jobs error: Invalid job
   id specified`, and a recently-ended job gives **`rc=0` with empty stdout**.
   Either way a job that ends before it is first observed RUNNING hangs that
   event forever. `wait_for_job_completion`, by contrast, does check
   `returncode` — so the two helpers disagree.
2. **sacct populates state and name independently** — see the section below.
3. **`TIMEOUT` carries ExitCode `0:0`**, byte-identical to success.
   `is_clean_completion` is safe only because it also requires
   `state == "COMPLETED"`; anything judging a child by exit code alone would
   accept a job Slurm killed for overrunning.
4. **`scancel --signal` never reaches the root row** — the signal lands on the
   skipped `.batch` step row, so `SlurmJobOutcome.signal` is always 0 for jobs
   we cancel.
5. **The `.job_failed` EXIT trap fires before the script's `mkdir -p`.** An
   imported-dataset job that fails early leaves no sentinel at all, because
   `touch "${output_dir}/.job_failed"` cannot write to a directory that does not
   exist yet. Benign today — the scheduler checks `.setup_complete_job` first —
   but latent if that branch order changes.

## Design

Nothing is installed on the host — not on a GitHub runner, not on your laptop.
An earlier draft `apt-get install`ed Slurm onto the runner and leaned on the
runner being disposable; that buys infrastructure that cannot be torn down,
cannot run twice on one machine, breaks on a self-hosted runner, and cannot run
locally at all — which would force a *separate* local cluster definition that
then drifts from the CI one, in exactly the subsystem this exists to stop
guessing about. CI and local share one image and one script.

Nothing is installed at run time either, and the image is a hard prerequisite
rather than a cache. `scripts/ci/slurm_up.sh` checks for the commands it needs
and fails naming the missing apt packages and `slurm_cluster.sh rebuild`; it
used to `apt-get` them instead, which bought nothing. That fallback could not
produce a usable cluster — on a bare `ubuntu:24.04` the daemons come up and then
`fake_serving_venv.sh` dies on `python3: command not found`, because the
interpreter, the pinned dependencies, pytest and `PYTHONPATH` are image-supplied
too — and it silently decoupled the Slurm version from the version-specific
config the script writes. Measured: bare `ubuntu:24.04` gives slurm-wlm 23.11.4
and an idle node, bare `ubuntu:26.04` gives 25.11.2 and a dead slurmd
(`Unable to initialize cgroup plugin`, per gotcha 2), five minutes in.

The working tree is bind-mounted **read-only** at `/workspace` and imported via
`PYTHONPATH`, so edits need no rebuild and nothing under test can write back into
the repo.

### The image tag is a hash of the image's inputs

`scripts/slurm_cluster.sh image-tag` hashes `docker/slurm-ci/Dockerfile`,
`pyproject.toml` and `uv.lock` and names the image
`eval360-slurm-ci:<hash>`. `cmd_up` does `image_exists || build_image`, so with
a fixed tag a Dockerfile edit had **no effect** until somebody remembered
`rebuild`; with a content-derived tag a changed input is a tag that does not
exist yet, and `up` rebuilds. CI asks the script for the tag rather than
recomputing it, so the pre-built image and the one `up` looks for cannot drift.
Verified that macOS `shasum` and Linux `sha256sum` produce the same tag.

Inside the image every external input is pinned: the base by digest, apt against
a `snapshot.ubuntu.com` timestamp, `uv` and the CPython patch by version, and
dependencies installed from `uv.lock` via `uv export --frozen` with
`--require-hashes`. **Refreshing a pin is a deliberate edit**, and the snapshot
must not predate the base digest or apt reports held broken packages. Two
residues, both deliberate: `ca-certificates` still comes from the live pocket
(the snapshot is HTTPS-only and `ubuntu:24.04` ships no CA bundle, so apt cannot
reach it otherwise), and pytest's closure is pinned by `ARG TEST_PINS` rather
than exported from the lock — the lock only knows it via the `test` extra, whose
other members (datasets, evalplus, transformers, pyarrow) are half a gigabyte
this image has no use for. The build asserts those pins still match `uv.lock`,
so they cannot drift silently.

### Privilege, and what is and is not isolated

**The container needs `--privileged`** — the same bargain kind makes for its node
containers. slurmd initialises a cgroup context unconditionally and must create a
scope directory under `/sys/fs/cgroup/system.slice/`, which Docker mounts
read-only otherwise. There is no narrower switch.

`--privileged` is a property of the container's **root** processes, though, so
the test side is kept out of it:

- `/workspace` is bind-mounted **read-only**.
- `probe`, `test`, `shell` and `exec` run as `slurm`, not root.

Neither half works alone. Root in a privileged container can simply
`mount -o remount,rw /workspace`; an unprivileged process holds no capabilities
and cannot. Together they mean a test — or a dependency a test imports — cannot
mutate the host checkout. Slurm jobs inherit it: slurmd runs each job as the
submitting user, so the job scripts under test are unprivileged too.

`slurm` specifically, because it is the configured `SlurmUser`, and Slurm
authorises administrative calls on `uid == 0 || uid == SlurmUser || AdminLevel >=
Operator`. That is what lets `test_job_lifecycle.py` drain and resume the node
without root. Any other unprivileged account would need an accounting record and
an `AdminLevel` granted at bring-up. Set `EVAL360_SLURM_RUNTIME_USER=root` if you
are debugging something that genuinely needs it.

Two tmpfs mounts are the explicit writable scratch, both `exec` because docker's
tmpfs default is `noexec` and venvs live in one of them:

| Mount | For |
|---|---|
| `/scratch` | `$HOME`, `pytest --junitxml`, anything ad hoc. CI writes the JUnit report here and `cat`s it out. |
| `/workspace/.eval360` | Not ours: `submit_imported_dataset_job` derives `repo_root` from `slurm_manager.py`'s location and builds runner venvs under it, and `repo_root` is not configurable. A tmpfs keeps those out of the checkout rather than cleaning up after them. `up` creates the empty host directory because runc cannot create a mount point inside a read-only bind; git does not track empty directories, and `down` removes it. |

**What this does not claim.** The root daemons still hold privileged device and
cgroup access, so a compromise of slurmctld/slurmd itself is contained by none of
the above, and the `slurm` account can write the control plane's own state
directories. Moving pytest into a *separate* unprivileged container would not
change that: the workload under test **is** Slurm jobs, and slurmd executes those
inside the privileged container by construction. Full separation would move the
pytest process without moving the thing that actually runs repository code.

### Ownership

`up` and `down` remove a container only after proving this script created it —
a `dev.eval360.slurm-cluster=IFM-AI/Eval360` label. A same-named container
without it is a hard failure, not a silent skip, so an unrelated container that
happens to be called `eval360-slurm` is never destroyed. Use
`EVAL360_SLURM_CONTAINER` to pick another name.

## Gotchas, all of which cost a debug cycle

Verified on Ubuntu 24.04 / slurm-wlm 23.11.4.

1. **`sbatch --version` before `slurm.conf` exists is fatal.** Since 23.11 a
   client with no config falls back to configless mode, does a DNS SRV lookup for
   the controller and exits fatal. Report tool versions *after* writing config.

2. **slurmd always initialises cgroups**, even with `proctrack/linuxproc` and
   `task/none`. On a cgroup/v2 host it asks systemd for a scope over dbus and
   dies. Needs `IgnoreSystemd=yes` in `cgroup.conf`. `CgroupPlugin=disabled` is a
   24.05 addition — on 23.11 it is read as a plugin *name* and is equally fatal.

3. **A file-less GPU GRES is silently dropped.** `Name=gpu Count=N` with no
   `File=` logs one line — `Ignoring file-less GPU gpu:(null)` — registers the
   node with `gpu:0`, and leaves it `INVALID_REG`, presenting as a node stuck
   `inval` with every job PENDING forever. Count-only works for generic GRES but
   not for `gpu`, so the cluster creates fake device files. This matters because
   `_serving_resource_args` always emits `--gres=gpu:N` and `gpus_per_node` is a
   `StrictPositiveInt`, so no config can request zero GPUs. Note also that
   `INVAL → RESUME` is an illegal transition, so retrying `scontrol update
   State=RESUME` only spams the log.

4. **sacct lags squeue.** See below — this one is not just a CI concern.

5. **Node capacity is declared, never measured.** `scripts/ci/slurm_up.sh` sets
   `FAKE_CPUS=16`, `FAKE_MEM_MB=32768` and `FAKE_GPUS=4` as fixed values.
   **Do not make these host-dependent.** They were derived from `nproc` and
   `/proc/meminfo`, and that made the cluster a different size on every machine:
   a GitHub runner container reports `CPUs=2` where a laptop reports many more, so
   tests allocating three 1-CPU replicas passed locally and hung in CI waiting
   180s each for a third job that could never start. That is the
   local-versus-CI divergence this cluster exists to eliminate, arriving through
   the back door. `SlurmdParameters=config_overrides` — already required for the
   fake GPUs — lets the node advertise capacity the host does not have, and
   nothing enforces the numbers because cgroups are disabled, so `--mem` and
   `--cpus-per-task` are accounting only.

   The topology is spelled out (`CoresPerSocket=$FAKE_CPUS ThreadsPerCore=1`)
   rather than inferred, because `SelectTypeParameters=CR_Core` allocates whole
   **cores**: a host presenting 2 CPUs as 1 core × 2 threads would fit one job
   per core regardless of what `CPUs` claimed.

   Note also that **Slurm has no fractional CPU request** — verified against the
   cluster, `--cpus-per-task=0.5` is rejected with `Invalid numeric value`, as is
   `0`. One whole CPU is the minimum allocatable unit, so "just ask for less" is
   not available as an alternative. `OverSubscribe` is the real oversubscription
   lever, but it would remain host-dependent (2 × N on a runner, more on a laptop) and would change
   scheduling semantics away from the real cluster, which is why declaring
   capacity is preferred: it keeps allocation semantics identical to production
   and changes only the node's advertised size.

## Known hazard: sacct populates state and name independently

Measured on this cluster, reproduced 3/3:

```
$ sbatch --parsable --job-name=fast-1 ... ; scancel $JOBID
6|allocation|CANCELLED by 0      # t+0.0s  — terminal, but placeholder name
6|fast-1|CANCELLED by 0          # t+0.5s  — settled
```

After submission there is ~1s with no accounting row at all, then a placeholder
row (`JobName='allocation'`, `State='PENDING'`), and — critically — **a row can
be terminal while the name is still the placeholder**, for up to ~1.5s after a
`scancel`.

Two consequences for production code:

- `SlurmManager.wait_for_terminal_job_outcomes` returns as soon as `is_terminal`
  is true. `Scheduler` then compares `outcome.job_name` against the submitted
  name and raises `sacct job name ... does not match the submitted child
  identity`. **A short-lived child can therefore fail a terminal result for a
  reason unrelated to the job.** This was found by the cluster suite, which
  cancels its job about a second after submitting it.
- `wait_for_terminal_job_outcomes` also raises immediately on a missing row with
  no retry, so it is fragile for a job queried within ~1s of submission.

Neither is fixed here — changing terminal-evidence semantics does not belong in
a CI change. `tests/slurm/conftest.py` waits for the row to settle before
asserting the name (and logs a `TERMINAL PLACEHOLDER OBSERVED` note whenever it
sees the window) so the suite is deterministic without being weakened to accept
`allocation`. `cluster_harness.wait_for_terminal` does the same for tests that
assert on a name — **use it rather than polling for a row's mere existence.**

This hazard has now been hit three separate times from three call sites: the
original discovery on a cancelled serving job, an imported-dataset test locally,
and the same test again on CI where the runner's timing differs. It is not a
curiosity of one code path, and that is the argument for fixing
`wait_for_terminal_job_outcomes` rather than only documenting it.

## What this cannot test

- **Container jobs** (`--container-image`, `sbatch_script_container.sh`) need
  pyxis/enroot, which is not part of vanilla Slurm. Those stay argv-only tested.
- **cgroup accounting and enforcement**, disabled per gotcha 2.
- **VLLM itself.** `scripts/ci/fake_vllm_server.py` stands in for it — the
  narrowest possible substitution, since everything between the scheduler and it
  is real.
- **Multi-node behaviour.** One node, four fake GPUs.

## CI

The `slurm-cluster` job in `.github/workflows/tests.yml` runs exactly the same
commands and is in the `needs` list of the `Tests: all` summary job, so a broken
Slurm boundary blocks the merge. It pre-builds the image for the layer cache,
under the tag `slurm_cluster.sh image-tag` prints, and `up` then adopts it. The
JUnit report is written to `/scratch` and `cat`-ed out, because `/workspace` is
read-only.

`tests/slurm/conftest.py` skips the suite when `sbatch` is absent, which is right
on a laptop but means pytest would exit 0 having tested nothing. The job
therefore enforces an execution floor with
`scripts/ci/assert_pytest_executed.py`, which reads the JUnit XML and fails when
fewer than 90 tests actually executed. Verified in both directions — with Slurm
hidden, pytest exits 0 with all 97 skipped and the floor exits 1 — and the script's
own `--self-test` runs in CI on every PR, because a gate nobody has watched go
red is not a tested gate. **Raise the floor when the suite grows.**
