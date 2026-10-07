# Choice Scoring Architecture Plan

## Summary

Choice scoring should be a row-level contract shared by dataset builders, the request layer, and the grader. A dataset row becomes choice-scoring-capable by providing:

```json
{
  "scoring_mode": "choice_scoring",
  "scoring_completions": ["A", "B", "C", "D"],
  "scoring_completion_labels": ["A", "B", "C", "D"]
}
```

The request layer appends each `scoring_completions` entry to the prompt, requests echo logprobs from a base/completions endpoint, and stores the raw payload. The grader then derives NLL metrics from those raw payloads and resolves correctness through `ground_truth_index`, `ground_truth`, `scoring_completion_labels`, and `scoring_completions`.

This keeps the old `multiple_choice` generative flow available for chat and reasoning models while giving base model multiple-choice evaluation a deterministic NLL path that does not depend on answer parsing.

## Why This Is Needed

Multiple-choice rows historically had only prompt text plus `ground_truth`. That worked for generative grading because the model generated an answer and the parser tried to recover a final letter. The NLL flow needs a different operation: score every candidate completion directly. Without structured candidate fields, the request layer has to infer choices from prompt formatting or duplicate per-dataset logic.

The divergence happened because the dataset rows described what to ask the model, but not what the valid scoring surface was. The new contract makes the scoring surface explicit:

```json
{
  "completion_input": "Question...\nA. alpha\nB. beta\nAnswer:",
  "ground_truth": "B",
  "scoring_mode": "choice_scoring",
  "scoring_completions": ["A", "B"],
  "scoring_completion_labels": ["A", "B"]
}
```

The same row still works with the existing generative `multiple_choice` grader. The structured fields are inert unless the dataset YAML selects `grader: {type: choice_scoring}` or the `multiple_choice_nll` alias; when it does, the runtime uses the structured fields instead of parsed generations.

If the generative prompt asks for chain-of-thought or reasoning tags, the row must provide a separate `scoring_prompt_prefix` that ends at the direct answer surface. Choice scoring appends each candidate completion to that prefix; it must not append answer letters after "Let's think step by step" or inside a `<reasoning>` block.

## Current Divergence Examples

### Generative multiple-choice rows

Builders such as ARC-Challenge emitted:

```json
{
  "completion_input": "...A. option one\nB. option two\nAnswer:",
  "chat_input": [{"role": "user", "content": "..."}],
  "ground_truth": "B"
}
```

The only machine-readable target was `ground_truth`. The answer choices were embedded in prompt prose.

### Choice-scoring runtime rows

The NLL flow introduced runtime behavior that needs:

```json
{
  "scoring_mode": "choice_scoring",
  "scoring_completions": ["A", "B"],
  "scoring_completion_labels": ["A", "B"]
}
```

The initial implementation derived or duplicated pieces near the OpenAI request path and grader. That made each builder and each consumer responsible for remembering the same implicit shape.

### Unified flow

The dataset builder now owns the available choices once, using `choice_scoring_fields(n_choices)` or an explicit equivalent when the choices are not letter labels. The request and grading layers consume the same helpers from `scheduler.choice_scoring_schema`.

## Design

### Row Contract

Required for `choice_scoring` rows:

- `scoring_mode`: must be `"choice_scoring"`.
- `scoring_completions`: completions to append and score, in choice order.
- `completion_input`, `scoring_prompt_prefix`, or `scoring_prompt_prefixes`: prefix used for full prompt scoring.
- `ground_truth` or `ground_truth_index`: correct answer.

Optional:

- `scoring_prompt_prefixes`: per-choice full-prompt prefixes for rows where each choice needs a different prompt context, such as Winogrande-style pronoun substitution.
- `scoring_completion_labels`: output labels used in grade records. Defaults to `scoring_completions`.
- `scoring_completion_prefix`: prompt prefix used for completion-only normalization. Defaults to `"Answer:"`.
- `scoring_completion_n_tokens`: precomputed positive integral token counts. Exact-integer floats and numeric strings representing exact integers are accepted for artifact compatibility and normalized to integers; booleans and fractions are rejected before an endpoint call. If absent, the grader uses `text_offset` from raw logprobs.
- `scoring_completion_n_chars`: character counts for character-normalized metrics. Defaults to string lengths.

### Ownership Boundaries

- Dataset builders own choice cardinality and target semantics.
- `scheduler.choice_scoring_schema` owns validation, label generation, prompt construction, and ground-truth resolution.
- `scheduler.openai_interface` owns API calls and raw logprob serialization.
- `scheduler.grader.choice_scoring` owns metric derivation from raw payloads.

This prevents prompt parsing from becoming a hidden API between builders and runtime code.

### Runtime Constraints

Choice scoring currently requires:

- base/completions model type, not chat-only endpoints.
- `average_over=[1]` and `pass_at=[1]`.
- logprobs with echo support and usable `text_offset` values when token counts are not precomputed.
- the backend must return logprobs for the echoed prompt suffix, not only the
  token generated after the prompt. OpenAI-compatible Dynamo/SGLang deployments
  that reject `max_tokens=0` and return `text_offset: []` with a single
  generated-token logprob are not sufficient for choice NLL scoring until they
  expose prompt logprobs or a native scoring endpoint through the configured API.
- requests use `echo=true` with at most one generated token for provider
  compatibility. `max_tokens=1` is only an upper bound, so the validator does
  not assume that a tail exists. When offsets are unavailable, it derives a
  zero-or-one tail from each returned choice text and requires the sum to match
  the response's strict integer `usage.completion_tokens`. Missing or
  contradictory usage is rejected before scoring. The serialized payload
  records each proven exclusion so the grader applies the same boundary after
  restart; unmarked historical artifacts retain their previous interpretation.
- dataset YAML opt-in via `grader: {type: choice_scoring}`.
- malformed MessagePack failures from batched vLLM `/v1/completions` requests are retried or narrowed to one request per choice. Each fallback response is still validated inside the retry boundary; an unusable single-choice suffix is retried and then reported with structured failure evidence.
- every selected completion-suffix token needs a finite numeric logprob. Live endpoint validation requires a non-boolean numeric value even when the logprob is nested in a dict or object; persisted artifacts remain tolerant of numeric strings for backward compatibility. Optional offsets must align with the logprob list, be non-negative ordered integers, and identify at least one suffix token; without offsets, rows must provide positive integral `scoring_completion_n_tokens`. Choice metadata and SDK payloads are normalized in JSON mode and must pass strict JSON serialization before either phase commits.

Rows can include choice-scoring fields before all dataset YAMLs are flipped. Those fields do not change runtime behavior until a dataset YAML explicitly selects the choice-scoring grader, which keeps adoption incremental.

## Current Eval Modifications

### GPQA-Diamond

Not migrated in this PR.

Reason: GPQA-Diamond base-model choice scoring showed high run-to-run variance in review discussion. The canonical GPQA-Diamond builder and YAML stay on the existing `multiple_choice` path until that variance is characterized separately.

### ARC-Challenge

Modified builder: `data_layer/build_arc_challenge.py`

Change: add `choice_scoring_fields(len(question.choices))` for variable choice counts.

Benefit: ARC rows with two through five options expose the exact valid label set. The request layer no longer has to infer whether `E` exists by parsing prompt text.

### ArabicMMLU

Modified exporter: `data_zoo/scripts/export_arabicmmlu.py`

Change: add `choice_scoring_fields(len(choices))` after validating the answer key is within the available labels.

Benefit: subject-specific files carry the same structured scoring fields, so downstream scoring does not need ArabicMMLU-specific materialization logic.

### MMLU-Pro

Modified builder: `data_prep_zoo/prepare_mmlu_pro.py`

Change: add `choice_scoring_fields(len(doc["options"]))` plus a direct-answer `scoring_prompt_prefix`.

Benefit: MMLU-Pro can expose up to ten options without hard-coding A-D. Base model scoring uses the deterministic NLL configs against `Answer: A/B/...`, while the original `completion_input` remains a chain-of-thought generative prompt.

Base-model configs:

- `data_zoo/mmlu_pro_choice_scoring.yaml`
- `data_zoo/mmlu_pro_small_choice_scoring.yaml`
- `data_zoo/mmlu_pro_5shot_choice_scoring.yaml`
- `data_zoo/mmlu_pro_5shot_small_choice_scoring.yaml`

### MMLU-Pro Reasoning

Modified builder: `data_prep_zoo/prepare_mmlu_pro_reasoning.py`

Change: do not emit choice-scoring fields.

Reason: the reasoning prompt intentionally ends inside `<reasoning>`, so appending a bare choice letter would score the wrong surface. This variant remains on `multiple_choice` generated-answer grading unless a separate direct-answer scoring surface is designed and regenerated.

### MMLU-Pro 5-shot

Modified builder: `data_prep_zoo/prepare_mmlu_pro_5shot.py`

Change: add structured fields to the few-shot rows plus a direct-answer few-shot `scoring_prompt_prefix`. The scoring prefix keeps exemplar answers as bare letters and ends the test question with `Answer:`.

Benefit: few-shot generation can keep chain-of-thought exemplars, while choice scoring evaluates a clean direct-answer surface without duplicating option-label logic.

### MMLU-Redux

Modified builder: `data_layer/persist_mmlu_redux.py`

Change: add `choice_scoring_fields(len(choices))` after resolving original or corrected answer labels.

Benefit: the corrected-answer source of truth is preserved while the scoring choices remain machine-readable.

### lm-eval Builtins

Modified builder: `data_layer/persist_builtin_dataset.py`

Change: when `doc_to_choice` is available, set:

```json
{
  "scoring_mode": "choice_scoring",
  "scoring_completions": ["choice text 1", "choice text 2"],
  "scoring_completion_labels": ["choice text 1", "choice text 2"]
}
```

Benefit: lm-eval style tasks often score completion text rather than letter labels. This keeps the generic importer faithful to `doc_to_choice` instead of assuming A/B/C/D labels.

## POLA, DRY, and KISS

POLA:

- Rows that are compatible with choice scoring say so explicitly with `scoring_mode`, while the dataset YAML still owns the runtime opt-in.
- `choice_scoring_fields(n_choices)` emits A-Z labels in predictable order.
- Consumers fail early with validation errors when structured fields are missing or inconsistent.

DRY:

- Choice label creation, prompt construction, row detection, and ground-truth resolution live in one module.
- Builders call one helper instead of open-coding A-D, A-E, or A-J fields.
- Request and grader code use the same prompt builder, so full-prompt and completion-only prompts cannot drift.

KISS:

- The contract is flat JSONL, not a nested schema that would require a migration layer.
- Existing generative `multiple_choice` flows remain unchanged.
- Adoption is an opt-in dataset/config change, so every benchmark does not need to migrate at once.

## Adoption Surface

To adopt this format for an existing multiple-choice dataset:

1. Add `scoring_mode`, `scoring_completions`, and optionally `scoring_completion_labels` to each row. This only makes rows compatible; it does not opt the dataset into choice-scoring execution.
2. Ensure `ground_truth` matches either a scoring label, a scoring completion, or add `ground_truth_index`.
3. Use dataset YAML `grader: {type: choice_scoring}` only for base/completions model evaluations that meet the runtime constraints.
4. Keep `multiple_choice` for chat and reasoning evaluations that depend on generated answer parsing.

This PR updates current choice-oriented builders/exporters so most adoption work becomes a config decision plus dataset regeneration.

Configs that select `grader: {type: choice_scoring}` require regenerated JSONL rows containing the structured fields described above. Existing hosted datasets without those fields must be regenerated or pinned to a revision that includes them before the choice-scoring YAMLs are used.

## Review Path

Use a GitHub issue as the architecture review thread. The issue should confirm:

- whether `choice_scoring` should remain opt-in per dataset YAML.
- whether A-Z labels are sufficient for all planned letter-choice datasets.
- whether generic lm-eval imported tasks should score `doc_to_choice` text, as implemented here, or map to explicit labels when available.
- whether future dataset builders should treat `choice_scoring_fields()` as the canonical helper for letter-choice rows.
