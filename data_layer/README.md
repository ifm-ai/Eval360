# Eval360 Data Layer

## Overview

The Eval360 data layer standardizes how raw sources are transformed into evaluation-ready datasets.

**Inputs**

- Raw corpora stored on local disk or hosted on Hugging Face.

**Outputs**

- JSONL files where each record represents a single evaluation item grouped by document: 
  - `row`: Integer identifier mirroring the lm-eval `doc_id`.
  - `completion_input`: Prompt string formatted for completion-style APIs.
  - `chat_input`: Ordered chat transcript (list of `{role, content}` messages) ready for chat APIs.
  - `ground_truth`: Optional ground truth answer used for grading.

## Dependencies
- Run all commands referenced in this README from the `Eval360/` project root (the directory containing `data_layer/`).
- If you want to create data files from scratch without relying on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/03c44adc0586f88bb343a74da1a1c602103536dd) (see the [DIY workflow](#diy)), you can skip these setup steps.
- Python 3.10+.
- Install the shared Python requirements:
  ```bash
  pip install -r data_layer/requirements.txt
  ```
- Pin the compatible lm-evaluation-harness version (without auto-installing its extras):
  ```bash
  pip install --no-deps "git+https://github.com/EleutherAI/lm-evaluation-harness.git@03c44adc0586f88bb343a74da1a1c602103536dd"
  ```
- Some lm-eval benchmarks require additional extras. Locate the relevant optional dependency group under [`[project.optional-dependencies]` in `pyproject.toml`](https://github.com/EleutherAI/lm-evaluation-harness/blob/03c44adc0586f88bb343a74da1a1c602103536dd/pyproject.toml#L59) and install the listed packages manually. For example, to support the `ruler` benchmark, install the packages on [line 79 of `pyproject.toml`](https://github.com/EleutherAI/lm-evaluation-harness/blob/03c44adc0586f88bb343a74da1a1c602103536dd/pyproject.toml#L79): `pip install nltk wonderwords scipy`.

## Data Preparation Modes

Eval360 supports three progressively flexible ways to prepare datasets:

- Built-in tasks
- Task YAML
- DIY

### Built-in Tasks
Use the CLI helper `data_layer/persist_builtin_dataset.py` to persist one or more lm-evaluation-harness tasks directly into Eval360-compatible JSONL datasets.

- Provide task names exactly as they appear in lm-eval (comma separated for multiple tasks).
- Choose an output directory under this project; each task produces a `<task>.jsonl` file.
- (Optional) Supply an include path if you maintain custom task definitions for lm-eval.

Refer to the lm-evaluation-harness [task table](https://github.com/EleutherAI/lm-evaluation-harness/blob/03c44adc0586f88bb343a74da1a1c602103536dd/lm_eval/tasks/README.md) for the list of supported built-in task names.

#### Key command-line flags

- `--tasks` (required): Comma-separated lm-eval task names to export.
- `--output-dir` (required): Destination directory where `<task>.jsonl` dumps are written.
- `--include-path`: Add directories that register custom lm-eval tasks before building requests.
- `--preview`: Print the first `N` grouped records to stdout for quick inspection.
- `--overwrite`: Permit replacing existing JSONL outputs when re-running the command.
- `--limit`: Cap the number of documents materialized per task when sampling broadly.
- `--samples`: Persist only the explicit document indices provided (e.g., `0 3 9`).
- `--system-instruction`: Inject a system prompt into chat-style task templates.
- `--fewshot-as-multiturn`: Render few-shot examples as alternating chat turns instead of a monologue.
- `--num-fewshot`: Override the task-configured few-shot count for prompt construction.
- `--verbosity`: Adjust TaskManager logging level for noisier debugging output.

The helper invokes lm-eval to build requests, aggregates all requests that belong to the same lm-eval document, and writes the grouped records using the field layout above. The snippet below shows a direct CLI invocation.

```bash
python data_layer/persist_builtin_dataset.py \
  --tasks mmlu \
  --output-dir examples/data/mmlu \
  --verbosity INFO \
  --preview 3 \
  --system-instruction "You are a helpful assistant." \
  --overwrite
```

One of the sample outputs from this flow is available at `examples/data/mmlu_philosophy.jsonl` for quick inspection.

### Task YAML
Author a task configuration YAML following the lm-evaluation-harness guide [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/03c44adc0586f88bb343a74da1a1c602103536dd/docs/new_task_guide.md).

After registering the YAML with lm-eval, you can reuse the same CLI flow described above to materialize JSONL datasets that honour the same `row`/`completion_input`/`chat_input` schema.

#### Example: Custom MMLU Variant
The snippet below illustrates how to expose a chain-of-thought flavoured version of the MMLU philosophy subset.

- Create `examples/data/mmlu_philosophy_cot.yaml` with:
  ```yaml
  task: mmlu_philosophy_cot
  task_alias: philosophy_cot
  dataset_path: cais/mmlu
  dataset_name: philosophy
  test_split: test
  fewshot_split: dev
  fewshot_config:
    sampler: first_n
  output_type: multiple_choice
  doc_to_text: |
    {{question.strip()}}
    A. {{choices[0]}}
    B. {{choices[1]}}
    C. {{choices[2]}}
    D. {{choices[3]}}
    Let's think step by step.
    Answer:
  doc_to_choice: ["A", "B", "C", "D"]
  doc_to_target: answer
  metric_list:
    - metric: acc
      aggregation: mean
      higher_is_better: true
  metadata:
    version: 0.0
  ```
  This YAML reuses the standard MMLU settings but tweaks the prompt to encourage chain-of-thought reasoning while keeping the multiple-choice grading interface intact.

- Materialize the dataset with the CLI:
  ```bash
  python data_layer/persist_builtin_dataset.py \
    --tasks mmlu_philosophy_cot \
    --output-dir examples/data/mmlu_philosophy_cot \
    --include-path examples/data \
    --preview 2 \
    --overwrite
  ```
  The `--include-path` flag points lm-eval to the directory that hosts your custom YAML so it can register the new task name before building requests.

### DIY
If you need full control, generate the records yourself while adhering to the output schema described earlier.

```python
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from scheduler.choice_scoring_schema import choice_scoring_fields

OUTPUT_PATH = Path("examples/data/mmlu_diy.jsonl")
DATASET_ID = "cais/mmlu"
SUBJECTS = ["philosophy"]
SPLIT = "test"


def build_prompt(subject: str, question: str, choices: list[str]) -> str:
    subject_label = subject.replace("_", " ")
    options = "\n".join(f"{label}. {text}" for label, text in zip("ABCD", choices))
    instruction = f"The following are multiple choice questions (with answers) about {subject_label}."
    prompt = f"{question.strip()}\n{options}\nAnswer:"
    return instruction, prompt


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
    row_id = 0
    for subject in SUBJECTS:
        dataset = load_dataset(DATASET_ID, subject, split=SPLIT)
        for doc in dataset:
            instruction, prompt = build_prompt(subject, doc["question"], doc["choices"])

            record = {
                "row": row_id,
                "completion_input": instruction + "\n\n" + prompt,
                "chat_input": [{"role": "system", "content": instruction}, {"role": "user", "content": prompt}],
                "ground_truth": "ABCD"[int(doc["answer"])],
                **choice_scoring_fields(len(doc["choices"])),
            }

            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")

            row_id += 1
```

This script produces JSONL rows like the abbreviated examples below (see the full sample at `examples/data/mmlu_diy.jsonl`):

```json
{"row": 0, "completion_input": "The following are multiple choice questions (with answers) about philosophy.\n\nAesthetics deals with objects that are_____.\nA. essential to our existence\nB. unimportant to most people\nC. not essential to our existence\nD. rarely viewed\nAnswer:", "chat_input": [{"role": "system", "content": "The following are multiple choice questions (with answers) about philosophy."}, {"role": "user", "content": "Aesthetics deals with objects that are_____.\nA. essential to our existence\nB. unimportant to most people\nC. not essential to our existence\nD. rarely viewed\nAnswer:"}], "ground_truth": "C"}
{"row": 1, "completion_input": "The following are multiple choice questions (with answers) about philosophy.\n\nFor Socrates, an unexamined life is a tragedy because it results in grievous harm to _____.\nA. the state\nB. the justice system\nC. the body\nD. the soul\nAnswer:", "chat_input": [{"role": "system", "content": "The following are multiple choice questions (with answers) about philosophy."}, {"role": "user", "content": "For Socrates, an unexamined life is a tragedy because it results in grievous harm to _____.\nA. the state\nB. the justice system\nC. the body\nD. the soul\nAnswer:"}], "ground_truth": "D"}
{"row": 2, "completion_input": "The following are multiple choice questions (with answers) about philosophy.\n\nAccording to Kant, nothing can be called “good” without qualification except _____.\nA. right action\nB. good consequences\nC. happiness\nD. a good will\nAnswer:", "chat_input": [{"role": "system", "content": "The following are multiple choice questions (with answers) about philosophy."}, {"role": "user", "content": "According to Kant, nothing can be called “good” without qualification except _____.\nA. right action\nB. good consequences\nC. happiness\nD. a good will\nAnswer:"}], "ground_truth": "D"}
```

For base-model NLL scoring, rows become choice-scoring-capable with `scoring_mode: "choice_scoring"` and `scoring_completions`, but the dataset YAML must still opt in with `grader: {type: choice_scoring}` or the `multiple_choice_nll` alias. Letter-choice datasets can use `choice_scoring_fields(n_choices)`, which adds A-Z completions and matching display labels. Imported lm-eval datasets that score full choice text should instead set `scoring_completions` to the `doc_to_choice` strings. Datasets that need a distinct full prompt for each choice can set `scoring_prompt_prefixes`; otherwise `completion_input` or `scoring_prompt_prefix` is reused for every choice.

## Create Dataset Configuration and Launch
Once your dataset is in place, run the dataset configuration script with task- and model-specific arguments to trigger an evaluation job.

The resulting YAML file includes the following top-level fields:
- `uuid`: Unique identifier for this configuration artifact.
- `grader`: Indicates the grading strategy, for example `type: multiple_choice`, `type: choice_scoring`, or `type: free_form`.
- `average_over`: List of generation counts used for averaging metrics.
- `pass_at`: List of pass@ settings evaluated during scoring.
- `meta`: Optional bookkeeping fields such as `split`, `priority`, or `fewshot`.
- `dataset_name`: Identifier passed through to the evaluator for task selection.
- `data_path`: Absolute path to a JSONL file or a directory containing JSONL files.
- `semantic_version`: Version string that tracks changes to the launch configuration.
- `num_generations`: Total number of generations (auto-inferred from the dataset unless overridden).

```bash
python data_layer/create_dataset_config.py \
  --output examples/data/mmlu_philosophy_config.yaml \
  --grader-type multiple_choice \
  --pass-at 1 5 \
  --avg-at 1 3 \
  --dataset-name mmlu_philosophy \
  --data-path examples/data/mmlu_philosophy.jsonl \
  --semantic-version 1.0.0 \
  --meta '{"split": "test", "priority": "high", "fewshot": 0}'
```

Refer to `examples/data/mmlu_philosophy_config.yaml` as a sample output when adapting the command for your own dataset.

Ensure the `--output` path resides in a directory monitored by the scheduler so the evaluation run is picked up automatically.
