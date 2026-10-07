# Data Zoo

This directory contains dataset configuration YAML files used by Eval360.

## ARC-Challenge Documentation

ARC-Challenge ships with two canonical Eval360 dataset configs:

- `data_zoo/arc_challenge.yaml`
- `data_zoo/arc_challenge_25shot.yaml`

The canonical data sources are on Hugging Face via:

- `hf://<HF_ORG>/<EVAL_SOURCES_REPO>/arc-challenge/arc_challenge.jsonl@main`
- `hf://<HF_ORG>/<EVAL_SOURCES_REPO>/arc-challenge/arc_challenge_25shot.jsonl@main`

These are Eval360-native generative multiple-choice datasets. They use the
standard `multiple_choice` grader and report the framework's usual
accuracy/pass@1 metrics. They do not reproduce lm-eval's choice-scoring
`acc_norm` setup.

### Rebuild ARC-Challenge JSONL locally

```bash
python data_layer/build_arc_challenge.py \
  --output-jsonl ~/data/eval360/arc-challenge/arc_challenge.jsonl \
  --overwrite \
  --config-output ~/data/eval360/arc-challenge/arc_challenge.yaml \
  --config-data-path ~/data/eval360/arc-challenge/arc_challenge.jsonl \
  --dataset-name arc_challenge \
  --dataset-uuid arc_challenge \
  --fewshot-count 0
```

```bash
python data_layer/build_arc_challenge.py \
  --output-jsonl ~/data/eval360/arc-challenge/arc_challenge_25shot.jsonl \
  --overwrite \
  --config-output ~/data/eval360/arc-challenge/arc_challenge_25shot.yaml \
  --config-data-path ~/data/eval360/arc-challenge/arc_challenge_25shot.jsonl \
  --dataset-name arc_challenge_25shot \
  --dataset-uuid arc_challenge_25shot \
  --fewshot-count 25 \
  --fewshot-seed 0
```

### ARC-Challenge troubleshooting

If an ARC run crashes during grading with:

```text
IndexError: string index out of range
```

from `scheduler/grader/multiple_choice.py`, first check whether
`<output_path>/<model-name>/arc_challenge_generations.jsonl` already existed.
Eval360 resumes from existing generations by default, so this error often means
the file still contains a stale empty generation from an older run.

Use `--force` on `evaluate-now` to clear the prior ARC outputs and regenerate:

```bash
eval360 \
  --max-generation-jobs 1 \
  --max-grading-parallelism 20 \
  --log-dir ~/data/eval360/logs/arc-challenge-9b \
  evaluate-now \
  --force \
  --model-paths model_zoo/qwen3.5-9b_arc.yaml \
  --data-paths <DATASET_ROOT>/arc-challenge/arc_challenge.yaml
```

## MATH-500

Two dataset configs for evaluating on the [MATH-500](https://arxiv.org/abs/2206.14858) benchmark.

### `math500.yaml` — 4-shot CoT (minerva_math format)
- Grader: `math_verify_llm_as_judge` with `gptoss-120b` as LLM judge
- Data: `hf://<HF_ORG>/<EVAL_SOURCES_REPO>/math500/math500.jsonl`
- The 4-shot preamble in `data_layer/build_math500.py` matches the lm-evaluation-harness `minerva_math` task exactly: each example ends with `\boxed{answer}` and `"Final Answer: The final answer is $X$. I hope it is correct."`
- Pair with `model_zoo/k2-v2-base.yaml`

To rebuild the JSONL:
```bash
python data_layer/build_math500.py \
  --output-jsonl data_zoo/math500/math500.jsonl \
  [--overwrite]
```

### `math500_reasoning.yaml` — 0-shot `<think>/<answer>` CoT
- Grader: `math_verify_llm_as_judge` with `gptoss-120b` as LLM judge
- Data: `hf://<HF_ORG>/<EVAL_SOURCES_REPO>/math500/math500_reasoning.jsonl`
- System prompt instructs the model to produce `<think>...</think><answer>\boxed{...}</answer>`
- The prompt does **not** include a `\boxed{}` example to avoid the model quoting it back and triggering the `</answer>` stop sequence prematurely
- Pair with `model_zoo/k2-v2-base-reasoning.yaml`

To rebuild the JSONL:
```bash
python data_layer/build_math500_reasoning.py \
  --output-jsonl data_zoo/math500_reasoning/math500_reasoning.jsonl \
  [--overwrite]
```

---

## GPQA-Diamond Documentation

GPQA-Diamond uses one canonical dataset config:
`data_zoo/gpqa-diamond/gpqa_diamond.yaml`.

The canonical data source is on Hugging Face via:
`hf://<HF_ORG>/<EVAL_SOURCES_REPO>/gpqa-diamond/*.jsonl@main`

Thinking vs non-thinking is an eval/runtime distinction only. The dataset content and dataset config are shared across both modes, and the difference lives in model configs and parser/inference settings.

### Prerequisites

1. Request access to GPQA from Hugging Face (`Idavidrein/gpqa`) and download `gpqa_diamond.csv` from the gated dataset export.
2. Keep GPQA raw files outside this repository.
3. Ensure the scheduler env is installed and your serving venv is available.

### Rebuild GPQA JSONL locally (optional)

Use the builder if you want a local JSONL artifact from the gated GPQA CSV.

```bash
python data_layer/build_gpqa_diamond.py \
  --input-csv /abs/path/to/gpqa_diamond.csv \
  --output-jsonl /abs/path/to/gpqa_diamond.jsonl \
  --seed 0 \
  --overwrite \
  --config-output /abs/path/to/gpqa_diamond.yaml \
  --config-data-path /abs/path/to/gpqa_diamond.jsonl \
  --dataset-name gpqa_diamond \
  --dataset-uuid gpqa_diamond
```

### Run with Qwen3.5-0.8B

Use:
- `model_zoo/qwen3.5-0.8b_gpqa.yaml` for GPQA.

Example run:

```bash
eval360 --max-generation-jobs 1 --max-grading-parallelism 20 \
  evaluate-now \
  --model-path /abs/path/to/qwen3.5-0.8b_gpqa.yaml \
  --data-paths /abs/path/to/gpqa_diamond.yaml
```
