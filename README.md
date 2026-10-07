# Eval360

LLM evaluation framework that orchestrates VLLM inference jobs on Slurm clusters, grades model outputs, and reports accuracy metrics.

## Table of contents

- [Architecture overview](#architecture-overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [FAQ](FAQ.md)
- [Adding a model](#adding-a-model)
- [Running selected pairs with eval configs](#running-selected-pairs-with-eval-configs)
- [Adding a dataset](#adding-a-dataset)
- [Adding a custom grader](#adding-a-custom-grader)
- [Adding an imported dataset runner](scheduler/imported_dataset/README.md)
- [BFCL web search evaluation](docs/BFCL_WEB_SEARCH.md)
- [Debugging score discrepancies](docs/DEBUGGING_SCORE_DISCREPANCIES.md)

---

## Architecture overview

The system has two main components:

| Component | Purpose | Python |
|---|---|---|
| **`data_layer/`** | Converts raw datasets into evaluation-ready JSONL files | ≥ 3.10 |
| **`scheduler/`** | Launches VLLM Slurm jobs, streams generations, grades results, writes output | 3.14 |

The scheduler also requires a separate **VLLM serving environment** for each model — either a Python 3.12 venv or a container image (enroot/pyxis). Each model config specifies exactly one of `venv_path` or `container_image` and may bind its model-serving Slurm resources.

---

## Installation

### 0. Bootstrap your tooling (if needed)

- Set the paths you will use throughout the setup:
```bash
export EVAL360_ROOT="$HOME/src/Eval360"
```

- Install `uv` if it is not already on `PATH`:
```bash
python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
```
- If using `conda`, initialize it first:
```bash
source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc
```

### 1. Create a Python 3.14 environment for the scheduler

```bash
# uv (recommended)
uv venv --python 3.14 "$HOME/.venvs/eval360-scheduler"
source "$HOME/.venvs/eval360-scheduler/bin/activate"
python -m ensurepip --upgrade
python -m pip install -e "$EVAL360_ROOT"

# or conda
conda create -n eval360-scheduler python=3.14
conda activate eval360-scheduler
python -m pip install -e "$EVAL360_ROOT"

# or venv
python3.14 -m venv "$HOME/.venvs/eval360-scheduler"
source "$HOME/.venvs/eval360-scheduler/bin/activate"
python -m ensurepip --upgrade
python -m pip install -e "$EVAL360_ROOT"
```

> Installing on a login node may cause issues — run this on a worker node.

### 2. Install

```bash
python -m pip install -e "$EVAL360_ROOT"
```

For install troubleshooting, including `daytona-sdk` issues, see [FAQ.md](FAQ.md).

### 3. Verify

```bash
eval360 --help
eval360 evaluate-now --help
```

---

## Quick start

### Step 1 — Create a VLLM serving environment

This environment is separate from the scheduler environment and is used to serve models on GPU nodes. You have two options:

#### Option A — Python venv or conda (default)

Fastest path:

```bash
bash scripts/tools/rebuild_vllm_serving_env.sh
```

This script creates or reuses the conda environment and writes a
scheduler-compatible `<env>/bin/activate` wrapper for use in model configs. By
default, it uses `$HOME/vllm-serving`. If that directory already exists, the
script keeps it in place and updates packages instead of deleting it. The helper
logs the resolved vLLM/XLLM package source, the optional GitHub ref/hash, and the
activation-wrapper patch target before and after patching.

To install from a pinned XLLM/vLLM GitHub revision instead of the default PyPI
`vllm` package, pass both source fields:

```bash
XLLM_GITHUB_URL=https://github.com/ifm-ai/xllm.git \
XLLM_GITHUB_REF=<commit-or-tag> \
bash scripts/tools/rebuild_vllm_serving_env.sh
```

For a non-GitHub pip spec, set `VLLM_PIP_SPEC` directly. For example:

```bash
VLLM_PIP_SPEC="vllm==0.10.0" bash scripts/tools/rebuild_vllm_serving_env.sh
```

Manual equivalent with `uv`:

```bash
python3.12 -m ensurepip --upgrade   # if pip is missing for Python 3.12
python3.12 -m pip install --user uv  # if uv is missing
uv venv --seed "$HOME/vllm-serving" --python 3.12
source "$HOME/vllm-serving/bin/activate"
python3.12 -m pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128   # run on a GPU node
```

Manual equivalent with conda:

```bash
conda create -y -p "$HOME/vllm-serving" python=3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/vllm-serving"
python -m pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128   # run on a GPU node
```

Test: `vllm serve IFM/K2-Think-V2 --tensor-parallel-size 8`

Use either the env root or its `bin/activate` file in the model config. Both venv and conda environments work with `venv_path`.

#### Option B — Container image (enroot/pyxis)

If your Slurm cluster has [pyxis](https://github.com/NVIDIA/pyxis) installed, you can use a container image instead of a venv. This avoids per-node venv setup and ensures a consistent environment.

The `container_image` field accepts:
- **Docker registry URIs** — pulled and converted automatically by pyxis (e.g. `nvcr.io/nvidia/pytorch:24.01-py3`)
- **Local `.sqsh` files** — pre-built enroot images on shared storage (e.g. `/<SHARED_STORAGE>/images/vllm-serving.sqsh`)

To build a local `.sqsh` image:
```bash
# Import from Docker Hub / NGC
enroot import docker://nvcr.io/nvidia/pytorch:24.01-py3
# Creates nvcr.io+nvidia+pytorch+24.01-py3.sqsh

# Or build from Dockerfile and convert
docker build -t vllm-serving .
enroot import dockerd://vllm-serving
```

### Step 2 — Create a model config

Copy [`examples/example_remote_model.yaml`](examples/example_remote_model.yaml) (venv) or [`examples/example_container_model.yaml`](examples/example_container_model.yaml) (container) and update:

```yaml
remote_model:
  path: IFM/K2-Think-V2         # HuggingFace repo or local path
  revision: null                 # branch/tag/commit, null = latest main
output_path: "~/src/Eval360/output"
owner: your.name

# Serving environment — exactly one of venv_path, conda_env, or container_image required:

# Option A: venv
venv_path: "~/.venvs/vllm-serving"   # or "~/.venvs/vllm-serving/bin/activate"

# Option B: conda
# conda_env: "vllm-serving"           # conda environment name

# Option C: container (enroot/pyxis)
# container_image: "nvcr.io/nvidia/pytorch:24.01-py3"  # registry URI or local .sqsh path
# container_mounts:                                     # optional bind mounts
#   - "/<SHARED_STORAGE>/models:/models:ro"
#   - "/<SHARED_STORAGE>/output:/output:rw"

# One serving replica always uses one task on one node. These fields control
# its GPU, CPU, memory, and wall-time request.
serving_slurm_resources:
  gpus_per_node: <GPUS_PER_NODE>   # set to the config's --tensor-parallel-size
  cpus_per_task: <CPUS_PER_TASK>
  memory_gb: <MEMORY_GB>
  time_limit: <TIME_LIMIT>
```

### Step 3 — Create a dataset config

Copy [`data_zoo/bbh.yaml`](data_zoo/bbh.yaml) or any config from `data_zoo/` as a starting point. See [Adding a dataset](#adding-a-dataset) for supported formats.

```yaml
uuid: "bbh_3shot"
dataset_name: "bbh"
data_path: "hf://<HF_ORG>/<EVAL_SOURCES_REPO>/bbh/*.jsonl@main"
grader:
  type: exact_match
average_over: [1]
pass_at: [1]
semantic_version: "1.0.0"
num_generations: 6511
```

### Step 4 — Run

> **Do not run on the login node.**

```bash
eval360 \
  --max-generation-jobs 1 \
  --max-grading-parallelism 20 \
  --log-dir logs \
  evaluate-now \
  --model-paths "$EVAL360_ROOT/examples/example_remote_model.yaml" \
  --data-paths "$EVAL360_ROOT/data_zoo/bbh.yaml"
```

If grading fails or you need recovery steps for stale outputs, see [FAQ.md](FAQ.md).

### Monitoring

```bash
# Slurm jobs
watch -n 1 "squeue --user $USER -o '%.18i %.9P %.100j %.8u %.2t %.10M %.6D %R'"

# Output file growth
watch -n 1 "wc -l <output_path>/*"

# Scheduler logs
tail -f logs/scheduler.log
```

Output files written to `output_path/<model-name>/`:
- `<dataset>_generations.jsonl` — raw generations
- `<dataset>_grades.jsonl` — per-sample grading results
- `<dataset>_scores.jsonl` — aggregate accuracy scores
- `<dataset>_run_metadata.yaml` — run-level request settings such as cache salt mode

Example run metadata:
```yaml
model: my-model
dataset: my_dataset
cache_salt:
  enabled: true
  mode: unique        # disabled | static | unique
  source: cli         # cli | model_config | null
  provider_field: extra_body.cache_salt
  value: <per-request unique>
```

For static salts, the raw salt is not written to metadata; the file stores
`value: <redacted>` and a short `value_sha256_12` fingerprint instead.

---

## CLI reference

```
eval360 [global options] <command> [command options]

Global options:
  --max-generation-jobs N       Max concurrent Slurm VLLM jobs. In evaluate-now,
                                omission derives it from unique serving keys;
                                required for long-running-scheduler.
  --max-grading-parallelism N   Max concurrent grading tasks. In evaluate-now,
                                omission allows every pair in the finite parsed request;
                                required for long-running-scheduler.
  --log-dir DIR                 Directory for scheduler.log and slurm output (default: .)
  --hf-cache-dir DIR            Override HF_HUB_CACHE for dataset downloads
  --debug                       Capture raw templated prompts and generation metadata
  --external-total-deadline-seconds SECONDS
                                Override external-model total deadlines; must exceed 7200

Commands:
  evaluate-now               Run one-shot evaluation and exit
    --model-paths FILE [...]   Candidate model config YAMLs
    --data-paths FILE [...]    Candidate dataset config YAMLs
    --eval-paths FILE [...]    Optional eval config YAMLs; when supplied, run only selected
                               tagged model/data pairs from the candidate pools
                               instead of the full cartesian product
    --ignore-grader-errors     Log errors instead of raising
    --force                    Delete existing generations/grades/scores/run metadata and regenerate from scratch
    --no-slurm                 Skip Slurm entirely; only external_model configs are allowed
    --salt-cache               Send a unique per-request cache salt for cache-bypass validation (requires --force)
    --slurm-partition NAME     Slurm partition to submit jobs to (default: the cluster's default partition)
    --logprobs                 Collect per-token log probabilities
    --evaluation-request-path FILE
                               Run an evaluator-owned canonical request instead of YAML inputs
    --terminal-result-path FILE
                               Exclusively publish write-last success evidence. With normal
                               YAML inputs this requires --eval-paths; with
                               --evaluation-request-path it publishes the request-native result

  long-running-scheduler     Watch directories and evaluate new configs as they appear
    --model-registration-path DIR
    --data-registration-path DIR
    --logprobs                 Collect per-token log probabilities
```

---

## Adding a model

There are three model source types. Exactly one must be specified per config.

In direct `--model-paths` x `--data-paths` mode, every model config must include
`owner` and `output_path`. In eval-config mode (`--eval-paths`), those fields may
be omitted from model configs because the eval config supplies `owner` and each
group's `output_root`.

### Option A — Remote model (HuggingFace, served via Slurm VLLM)

```yaml
remote_model:
  base_name: my-model           # short name used in output filenames
  path: org/model-name          # HuggingFace repo or local path to HF-format weights
  revision: null                # branch, tag, or commit hash (null = latest main)

# Use local_model instead of remote_model to watch a directory for new checkpoints:
# local_model:
#   path_glob: "/checkpoints/my-model-*.bin"
#   enqueue_existing: true

# Serving environment — exactly one required:
venv_path: "~/.venvs/vllm-serving"   # or "~/.venvs/vllm-serving/bin/activate"
# conda_env: "vllm-serving"           # conda environment name
# container_image: "nvcr.io/nvidia/pytorch:24.01-py3"  # or "/path/to/image.sqsh"
# container_mounts:                                     # optional, container_image only
#   - "/<SHARED_STORAGE>/models:/models:ro"

serving_slurm_resources:
  gpus_per_node: 8   # equals --tensor-parallel-size below
  cpus_per_task: <CPUS_PER_TASK>
  memory_gb: <MEMORY_GB>
  time_limit: <TIME_LIMIT>

output_path: "~/src/Eval360/output"
owner: your.name

vllm_cli_args:
  - "--tensor-parallel-size"
  - "8"

openai_kwargs:
  temperature: 0.0
  max_tokens: 1024
  stop: ["</s>", "\n\n"]
  seed: 1234

# Optional cache debugging. Omit by default for normal reproducibility runs.
# Configure only through this typed field. Raw `openai_kwargs.cache_salt`,
# `openai_settings.cache_salt`, and `extra_body.cache_salt` are rejected.
# `static` partitions cache entries deterministically; `unique` forces
# per-request cache misses.
cache_salt:
  mode: static   # disabled | static | unique
  salt: debug-partition-a

```

### Option B — Local model (checkpoint directory watcher, served via Slurm VLLM)

```yaml
local_model:
  path_glob: "/checkpoints/my-model-*.bin"
  enqueue_existing: true        # process already-present checkpoints at startup

venv_path: "~/.venvs/vllm-serving"
output_path: "~/src/Eval360/output"
owner: your.name
serving_slurm_resources:
  gpus_per_node: 8
vllm_cli_args: ["--tensor-parallel-size", "8"]
```

### Option C — External model (pre-existing OpenAI-compatible endpoint, no Slurm)

Use this for OpenAI, Azure OpenAI, Anthropic (via a proxy), or a self-hosted VLLM instance you're already running. No Slurm job is submitted — the URL is used directly.

```yaml
external_model:
  base_name: gpt-4o             # short name used in output filenames
  base_url: https://api.openai.com/v1   # full base URL including /v1
  api_key_env: OPENAI_API_KEY   # env var name (not the key itself); default: OPENAI_API_KEY
  # Optional token-bucket rate limiter. When present, this must be a strict
  # positive integer; omit it for no client-side RPM limit.
  requests_per_minute: 500
  retry_policy:
    max_attempts: 4
    request_timeout_seconds: 120
    total_deadline_seconds: 300
    initial_backoff_seconds: 1
    max_backoff_seconds: 30
    jitter: full                  # full | equal | decorrelated

model_type: instruct            # base or instruct
parser_type: noop
max_simultaneous_requests: 20   # max concurrent in-flight requests
openai_kwargs:
  temperature: 0.0
  max_tokens: 2048
output_path: "~/src/Eval360/output"
owner: your.name
```

External request budgets are capped at 7,200 seconds, and both backoff values
are capped at 60 seconds. Model YAML total deadlines are capped at 7,200
seconds and may be explicitly overridden for every external candidate model
in a run with
`--external-total-deadline-seconds`; the override must be finite and greater
than 7,200 seconds. Strict RPM and concurrency input
validation applies to external models without changing the established
local/remote model input contract. Retry jitter defaults to `full`; `equal`
and `decorrelated` are also supported. See
[`docs/EXTERNAL_RETRY_POLICY.md`](docs/EXTERNAL_RETRY_POLICY.md) for defaults,
limits, persistence behavior, and validation rationale.

Run with `--no-slurm` (and omit `--max-generation-jobs`):

```bash
export OPENAI_API_KEY=sk-...

eval360 \
  --max-grading-parallelism 20 \
  evaluate-now \
  --no-slurm \
  --model-paths my_external_model.yaml \
  --data-paths data_zoo/bbh.yaml
```

See [`examples/example_external_model.yaml`](examples/example_external_model.yaml) for a complete template.

### Custom output parsing

If the model wraps its answer in special tokens (e.g. `<think>...</think>`), implement a parser:

```python
# scheduler/grader/my_parsers.py
from .parser_registry import register_parser

@register_parser("think_tag")
def _think_tag_parser(generation: str) -> str | None:
    import re
    match = re.search(r"</think>(.*)$", generation, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    extracted = match.group(1).strip()
    return extracted if extracted else None
```

Then set `parser_type: think_tag` in the model config. Return `None` if the generation cannot be parsed — this signals a failed parse to the grader, which decides how to handle it (e.g. mark as incorrect, skip, or make a best-effort attempt).

---

## Running selected pairs with eval configs

By default, `evaluate-now` runs the cartesian product of all `--model-paths` and
`--data-paths`. Use `--eval-paths` when you want explicit model/data pair
selection and per-group runtime overrides.

With `--eval-paths`, `--model-paths` and `--data-paths` are candidate pools; eval
groups select matching configs by tag.

Eval configs match model configs by `tag` and dataset configs by `tag`:

```yaml
version: 1
owner: your.name
groups:
  - name: reasoning-aime
    model_tag: reasoning
    data_tag: aime
    parser_type: boxed
    output_root: /<OUTPUT_ROOT>/results
    vllm_cli_args:
      - --tensor-parallel-size 8
    openai_overrides:
      temperature: 0.0
      max_tokens: 32768
```

A group that changes `--tensor-parallel-size` needs a model config whose `gpus_per_node` matches.

Run it with the same model and data config inputs, plus one or more eval configs:

```bash
eval360 \
  --max-generation-jobs 1 \
  --max-grading-parallelism 20 \
  evaluate-now \
  --model-paths model_zoo/*.yaml \
  --data-paths data_zoo/*.yaml \
  --eval-paths evals/reasoning.yaml
```

For each matched pair, Eval360 creates an isolated internal model/task variant.
The variant name includes a stable `-eval-<token>` suffix so output files and
database rows do not collide, but outbound API requests still use the original
base model name. VLLM is launched with that original name plus any sibling eval
variant names as served aliases, so shared deployments continue to work.

Important details:
- `owner` comes from the eval config.
- `output_root` must be a non-empty absolute path; `~` is expanded, and relative
  paths are rejected.
- Output files are written under
  `<group.output_root>/<base-model-name>/<pair-token>/`.
- `parser_type`, `vllm_cli_args`, `openai_overrides`, and optional `grader` are
  applied per group.
- `--served-model-name` must not be set manually in `vllm_cli_args`; Eval360
  injects the correct aliases.
- Imported-dataset tasks are not supported in eval-config mode.

### Terminal evidence for normal eval-config YAML

The normal eval-config invocation can opt into output-only terminal evidence by
adding `--terminal-result-path`:

```bash
eval360 evaluate-now \
  --model-paths model_zoo/*.yaml \
  --data-paths data_zoo/*.yaml \
  --eval-paths evals/reasoning.yaml \
  --terminal-result-path /absolute/evidence/terminal-result.json
```

The model, data, and eval YAMLs remain the authoritative behavior definitions;
this route does not require or synthesize an input manifest. Before parsing,
Eval360 binds the exact installed runner entrypoint and the ordered model, data,
and eval YAML files. For every resolved pair, the result records its source
YAML/group association, resolved model and task definition digests, ordered
dataset files, required output files, and controller generation, grading, and
aggregation roles. Shared model-serving Slurm children retain their exact event
role associations and authoritative terminal root-job outcomes from `sacct`.

The result is published exclusively and write-last only after every selected
event, required output, and child outcome proves complete success. Failure,
interruption, changed inputs, missing outputs, or an existing result path leaves
no new success file and never replaces the existing file. This output-only
schema requires `--eval-paths` and supports standard generation/grading tasks;
it does not support `imported_dataset` tasks.

Runner identity comes from the natural `sys.argv[0]` of the installed `eval360`
entrypoint. Callers must invoke that entrypoint normally and must not wrap the
process in a way that rewrites `argv[0]` to a different executable.

### Request-native machine-readable terminal result

For an evidence-producing invocation, supply `--evaluation-request-path` and
`--terminal-result-path` together instead of `--model-paths`, `--data-paths`,
or `--eval-paths`. Eval360 strictly parses canonical
`eval360.evaluate-now-request/1.0`, `eval360.runner-definition/1.0`,
`eval360.suite-catalog/1.0`, and `eval360.suite/1.0` JSON objects. The runner
definition owns source repository/commit/tree, the actual entrypoint payload,
and request/result contracts. The catalog selects one exact suite manifest;
the suite owns model compatibility, serving-runtime requirement, ordered
nonempty task definitions, exact dataset payloads, settings, and totals.
Dynamic release, serving, dataset, and output paths remain request bindings.
Eval360 derives its in-memory model/tasks directly from this closure and does
not generate or parse YAML on this route.
Here, canonical JSON means one duplicate-free JSON object encoded as UTF-8,
with keys sorted, compact `,`/`:` separators, no non-finite numbers, and
exactly one trailing newline. The terminal result uses the same encoding.

The terminal result is a success artifact, not an attempt log. It is created
exclusively as the final controller action only after every selected event is
successful, every required output is a stable regular file, and every
applicable Slurm child has an authoritative terminal `sacct` root-job record.
Expected cancellation of a VLLM serving job is successful only when Eval360
recorded both its completed event/role association and its own
`scheduler_release` intent. Missing
jobs, missing outputs, non-terminal or unsuccessful child outcomes, changed
input manifests/configs, interruption, and controller failure leave the result
path absent. An existing result is never replaced.

The result preserves the exact request and owner-manifest bindings, runner
source and actual entrypoint identity, selected suite and ordered task closure,
controller generation/grading/aggregation units, child job IDs and completed
event-role associations, raw and normalized scheduler state, exit
code/signal/reason, and required output file sizes and digests. Existing output
reuse is allowed only when its persisted request/task binding matches. This
contract does not treat disappearance from `squeue` or a zero controller exit
alone as proof of evaluation success. Schema 1.0 supports standard generation
tasks with non-LLM graders; a later owner schema is required before selecting
an imported-dataset task or LLM-as-judge runtime.

---

## Adding a dataset

### Dataset config YAML

```yaml
uuid: "unique-id"
dataset_name: "my_dataset"
semantic_version: "1.0.0"
data_path: "~/src/Eval360/data/*.jsonl"   # local path/glob or hf:// URI (see below)
grader:
  type: multiple_choice
average_over: [1]
pass_at: [1]
meta:
  split: test
```

### Data path formats

`data_path` accepts a local path, a glob, or a Hugging Face dataset URI:

```yaml
# Local file
data_path: "/data/mmlu.jsonl"

# Local glob (multiple files)
data_path: "/data/mmlu/*.jsonl"

# HuggingFace dataset file
data_path: "hf://<HF_ORG>/<EVAL_SOURCES_REPO>/gpqa-diamond/gpqa_diamond.jsonl@main"

# HuggingFace dataset glob
data_path: "hf://<HF_ORG>/<EVAL_SOURCES_REPO>/bbh/*.jsonl@main"
```

HF URI format: `hf://<org>/<repo>/<path>[@<revision>]`
- `revision` is optional and defaults to `main`
- For glob URIs, all matching files are downloaded before evaluation

**HF authentication:** private or gated repos require a token. Set `HF_TOKEN` or run `hf auth login` once. The `--hf-cache-dir` CLI flag overrides the default HF cache location.

### JSONL record format

Each record must have:

| Field | Type | Description |
|---|---|---|
| `row` | int | Line number |
| `completion_input` | str | Prompt for base/completion models |
| `chat_input` | list | OpenAI-format messages (`[{"role": ..., "content": ...}]`) |
| `ground_truth` | any | Expected answer (grader-specific) |

To create JSONL from standard benchmarks, see [`data_layer/README.md`](data_layer/README.md).

---

## Adding a custom grader

A grader subclasses `AccuracyGraderBase`, implements `grade_sample`, and registers itself with `@register`. It is auto-discovered at startup if placed in `scheduler/grader/`.

### Option A — add a file to this repo

```python
# scheduler/grader/my_grader.py
import copy
from typing import Any
from .base import AccuracyGraderBase
from .registry import register

@register("my-grader")
class MyGrader(AccuracyGraderBase):
    async def grade_sample(self, sample: Any, *_):
        result = copy.deepcopy(sample)
        result["correct"] = [1 if g == sample["ground_truth"] else 0
                              for g in sample["parsed_generations"]]
        return result
```

### Option B — install as a plugin package

Create a standalone Python package (see the [Packaging Python Projects tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/) for an introduction to Python packaging) and declare it as an eval360 plugin:

```toml
# pyproject.toml
[project.entry-points."eval360.graders"]
my_grader = "my_grader_package.my_grader"
```

```bash
python -m pip install -e "$EVAL360_ROOT"
python -m pip install -e "$HOME/src/my-grader-package"
```

Verify: `python -c "import scheduler.grader; from scheduler.grader import list_graders; print(list_graders())"`

### Rules for `grade_sample`

- `sample` contains all input JSONL fields plus `parsed_generations` (list of strings, already processed by the model's `parser_type`). Grade from `parsed_generations`, not `generations`.
- Return a dict with a `correct` field: a list of `0`/`1` values, one per generation.
- Must be stateless — may be called on partial datasets during resume.
- `await` freely for HTTP or file I/O.
- Raise for unrecoverable errors; the scheduler records them in the output file.
- **Never execute LLM-generated code from `sample["generations"]` locally.**

---

Result snapshots: [LLM360/eval360-snapshots](https://github.com/LLM360/eval360-snapshots)
