# Model Zoo

This directory stores model configuration YAML files.

## K2 Horizon 7B base full suite

`k2_horizon_7b_base_full.yaml`, the eight configs under
`data_zoo/k2_horizon_7b_base_full/`, and
`group_zoo/k2_horizon_7b_base_full.yaml` form one normal Eval360 eval-config
closure. The model template retains the reviewed K2 Horizon base-model serving
settings. The data configs retain the canonical task/grader/sampling semantics
and identify every file in dataset snapshot
`2f926def94ae9516d311380678121cca173afd46` by source path, byte size, and
SHA-256.

The quoted `/__EVAL360_*__` strings are literal absolute-path
placeholders; Eval360 does not expand them. A caller must materialize copies with absolute
paths for the converted HF release, serving-venv activation file, local
dataset-snapshot root, and output root. The resulting files run through the
ordinary interface:

```bash
eval360 evaluate-now \
  --model-paths /absolute/materialized/model.yaml \
  --data-paths /absolute/materialized/data/*.yaml \
  --eval-paths /absolute/materialized/eval.yaml \
  --terminal-result-path /absolute/evidence/terminal-result.json
```

`--terminal-result-path` is optional output-only evidence for this normal YAML
route; it does not replace these YAMLs with an input manifest. The result binds
the natural installed `eval360` executable from `sys.argv[0]`, the exact ordered
model/data/eval configs, every resolved dataset and output file, controller
generation/grading/aggregation roles, and authoritative terminal `sacct`
outcomes for the shared serving children. It is published exclusively and
write-last only after all eight selected pairs succeed. Invoke the installed
entrypoint normally without rewriting `argv[0]`. This schema requires the eval
config route and does not support `imported_dataset` tasks.

The model config binds its serving replica to one node, one task and one GPU,
matching its tensor-parallel size of 1, without an exclusive node request;
CPU, memory and wall time are left to the site. The eval config deliberately selects a parser for each dataset: `boxed` for
AIME, `passthrough` for BBH, `mc_answer` for GPQA-Diamond, `gsm8k_base` for
GSM8K, `passthrough` for IFEval and MBPP, and `the_answer_is` for MMLU-Pro and
MMLU zero-shot. All eight pairs retain one serving key and therefore share the
model deployment. When the operator omits the one-shot concurrency flags,
Eval360 derives one generation job from that unique serving key and permits all
eight finite pairs to grade; explicit operator values still override those
one-shot semantics.

## K2-V2

Two configs for evaluating the K2-V2 base model on MATH-500:

### `k2-v2-base.yaml` — 4-shot CoT
- `parser_type: boxed` — extracts last `\boxed{}` from generation
- `max_tokens: 8192`, `temperature: 0.0`
- Stop sequences: `</s>`, `\nProblem:` (matches the 4-shot preamble format)
- `repetition_penalty: 1.05` — breaks greedy decoding loops on long sequences
- Pair with `data_zoo/math500.yaml`

### `k2-v2-base-reasoning.yaml` — 0-shot `<think>/<answer>`
- `parser_type: boxed` — extracts last `\boxed{}` from the `<answer>` block
- `max_tokens: 32768`, `temperature: 1.0`, no repetition penalty
- Stop sequences: `</s>`, `</answer>`
- `name_modifier: reasoning` — output goes to `k2_base_results/k2-v2-reasoning/`
- Pair with `data_zoo/math500_reasoning.yaml`
