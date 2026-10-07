# ImportedDataset Runners

This sub-package lets Eval360 evaluate models using an external benchmark's **own code** rather than a custom grader. The benchmark runs directly on the same Slurm node as the VLLM server, with VLLM in the background and the benchmark CLI in the foreground.

## How it works

For an `imported_dataset` task, the scheduler:

1. Spawns a VLLM Slurm job for the model (same as a normal generation job)
2. The sbatch script creates the benchmark's venv under `.eval360/envs/<name>/`, using a sentinel file (`.setup_complete`) to detect and recover from partial installs caused by node preemption
3. Executes the script returned by `build_setup_script` (the benchmark-specific install commands) before the benchmark venv is activated
4. Starts VLLM in the background, waits for it to be healthy on `localhost:8000`, then executes the script returned by `build_benchmark_script`
5. After the job finishes, calls `parse_results` to read the benchmark's output and write `scores.yaml`

The setup and benchmark scripts are bash strings executed inside the Slurm job. VLLM runs in the background; if the job exits the VLLM process is killed automatically by Slurm.

The imported-dataset sbatch templates use the same per-job compile-cache layout
as normal VLLM jobs. `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`,
`TRITON_CACHE_DIR`, and `CUDA_CACHE_PATH` are placed under
`${SLURM_TMPDIR:-/tmp}/eval360-vllm-${UID}-${SLURM_JOB_ID}` so concurrent jobs
and stale cache directories from other users do not interfere with setup or
serving.

## Dataset YAML

Instead of `data_path` and `grader`, use `imported_dataset`:

```yaml
uuid: "my-benchmark-eval"
dataset_name: "my_benchmark"
semantic_version: "1.0.0"
imported_dataset:
  name: my-benchmark      # must match a registered runner
  commit: abcdef12345     # benchmark repo commit to use
  args:
    version: 4.0
    judge_model: gpt-4o   # benchmark-specific config
```

`data_path`, `grader`, and `num_generations` must **not** be set for imported-dataset tasks.

## Adding a runner

Subclass `ImportedDatasetRunnerBase`, implement the three methods, and decorate with `@register`. Place the file in this directory — it will be auto-discovered at startup.

```python
# scheduler/imported_dataset/my_benchmark.py
from pathlib import Path

from .base import ImportedDatasetRunnerBase, ImportedDatasetResultError
from .registry import register
from ..grader.base import Score
from ..model import ModelInstance
from ..task import ImportedDatasetTask


@register("my-benchmark")
class MyBenchmarkRunner(ImportedDatasetRunnerBase):

    def build_setup_script(self, repo_root: Path) -> str:
        # $VENV is set by the sbatch script. The venv is not activated yet —
        # use "$VENV/bin/python -m pip" explicitly.
        return '"$VENV/bin/python" -m pip install my-benchmark-package'

    def build_benchmark_script(
        self,
        model_instance: ModelInstance,
        task: ImportedDatasetTask,
        output_dir: Path,
    ) -> str:
        # VLLM is already serving on localhost:8000.
        # model_instance.name = VLLM served name (for API calls)
        # model_instance.path = HF model ID / local path (for tokenizer loading)
        # The benchmark venv is activated before this runs — use "$VENV/bin/python" etc.
        return f"""\
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"

"$VENV/bin/python" -m my_benchmark.run \\
    --model {model_instance.name} \\
    --output-dir "{output_dir}"
"""

    def parse_results(
        self,
        output_dir: Path,
        task: ImportedDatasetTask,
        model_instance: ModelInstance,
    ) -> list[Score]:
        import json
        results_file = output_dir / "results.json"
        if not results_file.exists():
            raise ImportedDatasetResultError(
                f"Expected results file not found: {results_file}"
            )
        data = json.loads(results_file.read_text())
        return [Score(name="my_benchmark_accuracy", value=data["accuracy"])]
```

## Method reference

### `build_setup_script(repo_root: Path) -> str`

Returns a bash snippet containing **only the benchmark-specific install commands** (e.g. `pip install ...`). The sbatch script owns everything else:

- Creates the venv at `<repo_root>/.eval360/envs/<name>/`
- Guards against partial installs using a sentinel file (`.setup_complete`) — if a node is preempted mid-install, the broken venv is deleted and reinstalled cleanly on the next run
- Sets `$VENV` to the venv path, but does **not** activate it before `build_setup_script` runs

Use `"$VENV/bin/python" -m pip` rather than bare `pip`. The `.eval360/` directory is gitignored and is per-user, per-cluster.

> **Note:** The sentinel means the setup script only runs once. If you update `build_setup_script` (e.g. to add a new dependency), delete the existing venv on the cluster to force reinstallation:
> ```bash
> rm -rf <repo_root>/.eval360/envs/<runner_name>
> ```

### `build_benchmark_script(model_instance, task, output_dir) -> str`

Returns a bash snippet that runs the benchmark end-to-end. At the time this script runs:

- VLLM is already serving on `localhost:8000`
- `$VENV` points to the benchmark venv after activation
- `output_dir` is the directory where results should be written

**`model_instance.name` vs `model_instance.path`**: `.name` is the VLLM served model name (the `model` parameter for OpenAI API calls). `.path` is the HF model ID or local path (for tokenizer loading, config loading, etc.). Most benchmarks that call the OpenAI-compatible API need `.name`; benchmarks that load the tokenizer themselves need `.path`.

**OpenAI-compatible env vars**: set these in your script if the benchmark reads them:

| Variable | Purpose |
|---|---|
| `REMOTE_OPENAI_BASE_URL` | Base URL (e.g. `http://localhost:8000/v1`) — some benchmarks read this automatically |
| `REMOTE_OPENAI_API_KEY` | API key (`"EMPTY"` for VLLM) |
| `REMOTE_OPENAI_TOKENIZER_PATH` | If the benchmark loads a tokenizer separately, point this at `model_instance.path` |

**Inline Python**: if the benchmark has no CLI and must be driven via Python API, embed a Python script using a heredoc:

```python
return f"""\
export REMOTE_OPENAI_BASE_URL="http://localhost:8000/v1"

"$VENV/bin/python" - <<'EOF'
from my_benchmark import run
run(model={model_instance.name!r}, output_dir={str(output_dir)!r})
EOF
"""
```

The script must exit non-zero on failure so the scheduler marks the event as failed.

Use `task.imported_dataset.args` to access benchmark-specific configuration from the dataset YAML.

### `parse_results(output_dir: Path, task, model_instance) -> list[Score]`

Reads the benchmark's output files from `output_dir` and returns a list of `Score` objects. Raise `ImportedDatasetResultError` if the results are absent or incomplete — the scheduler will mark the event as failed.

Score names should be prefixed with the benchmark name (e.g. `my_benchmark_accuracy`) so they are identifiable in the output YAML.

## Debugging

Output files land under `<output_path>/<model_name>/`. The scheduler writes `scores.yaml` there after `parse_results` succeeds. The exact subdirectory structure inside is benchmark-specific.

To re-run a benchmark without redoing setup, delete only the result/score files — leave the `.eval360/envs/` venv intact.

To force a fresh install:
```bash
rm -rf <repo_root>/.eval360/envs/<runner_name>
```

## Plugin runners

Runners can also live in an external package:

```toml
# pyproject.toml of your plugin package
[project.entry-points."eval360.imported_datasets"]
my_benchmark = "my_package.my_runner_module"
```

```bash
python -m pip install -e "$HOME/src/my-runner-package"
```

Verify: `python -c "import scheduler.imported_dataset; from scheduler.imported_dataset import list_runners; print(list_runners())"`
