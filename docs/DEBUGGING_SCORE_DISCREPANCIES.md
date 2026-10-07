# Debugging Dataset Score Discrepancies

This guide covers the most common reasons your Eval360 scores might not match public or pre-existing scores for the same model and dataset, and how to diagnose each one.

---

## 1. Prompt format

**Base vs. instruct model type**

`model_type: base` sends `completion_input` (a raw string) to the completions endpoint.
`model_type: instruct` sends `chat_input` (an OpenAI-format message list) to the chat completions endpoint, where VLLM applies the model's chat template.

If the reference evaluation used a chat-formatted prompt for a base model (or vice versa), your scores will diverge. Check the reference implementation to confirm which endpoint and format it used.

**Few-shot examples**

Some public benchmarks prepend few-shot examples in the prompt. Verify that your `completion_input` / `chat_input` in the JSONL includes (or intentionally excludes) the same examples as the reference.

---

## 2. Chat template and reasoning effort

For instruct models (`model_type: instruct`), VLLM applies the model's chat template when converting `chat_input` into a raw prompt. Two things can go wrong here:

**Wrong or missing `chat_template_kwargs`**

Some models accept extra template kwargs that control formatting, special tokens, or reasoning budget. These are passed via `openai_kwargs`:

```yaml
openai_kwargs:
  extra_body:
    chat_template_kwargs:
      reasoning_effort: high
```

**Known bug:** `reasoning_effort` (and other `chat_template_kwargs`) may not be applied correctly in some VLLM versions. If you suspect this, verify the actual prompt VLLM received by checking the VLLM server logs directly.

**`model_type` mismatch**

Setting `model_type: base` for an instruct model silently sends a raw string to the completions endpoint, bypassing the chat template entirely.

---

## 3. Generation parameters

Check `openai_kwargs` in your model YAML against the reference evaluation:

```yaml
openai_kwargs:
  temperature: 0.0
  max_tokens: 1024
  stop: ["</s>", "Q", "\n\n"]
  seed: 1234
```

Key parameters to verify:
- **`temperature`** — public benchmarks vary; some use greedy decoding (`temperature: 0.0`), others use sampling. Don't assume.
- **`max_tokens`** — if too low, generations are truncated before the model outputs its answer. This silently hurts accuracy.
- **`stop`** — stop token sequences cause the model to halt early. A missing or wrong stop list can produce over-generation (e.g. the model answers multiple questions in one generation) or under-generation.
- **`seed`** — controls sampling reproducibility. If the reference used a fixed seed, set the same one to get comparable outputs.
- **`top_p`, `top_k`** — less common but can matter for sampling-based evals.
- **`pass_at` / `average_over`** — if the reference reports pass@1 with a single greedy generation, make sure you're not averaging over multiple samples.

---

## 4. Parsing

The parser (`parser_type` in the model config) transforms raw generations into the string the grader sees. Inspect `*_grades.jsonl` to compare `generations` (raw) vs `parsed_generations` (after parsing):

```bash
# spot-check parsing for a few rows
python3 -c "
import json
with open('output/my-model/dataset_grades.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 5: break
        row = json.loads(line)
        print('RAW:   ', row['generations'])
        print('PARSED:', row['parsed_generations'])
        print()
"
```

Things to check:
- A `null` / `None` in `parsed_generations` means the parser returned nothing for that sample — it is **often** marked incorrect regardless of the raw generation. If you see a high null rate, your parser is not matching the model's output format.
- **Beginning vs. end of generation** — some parsers look for the answer at the start of the response, others at the end (e.g. after a `</think>` tag). Make sure yours matches where the model actually puts its answer.
- **Case sensitivity and whitespace** — verify the parser strips the same trailing whitespace and handles the same case variants as the reference.

---

## 5. Grader logic

Check which grader is active:

```bash
python3 -c "
import json
with open('output/my-model/dataset_grades.jsonl') as f:
    row = json.loads(f.readline())
    print(row.get('grader_type', 'not recorded'))
"
```

Then compare the grader's behaviour against the reference:
- Does the reference use exact match, or normalized match (lowercased, punctuation stripped)?
- For multiple-choice benchmarks: does the reference match on the letter (`A`), the full option text, or both?
- Does the reference handle ties or abstentions differently?

---

## 6. Using logprobs to diagnose borderline cases

When `--logprobs` is enabled, each generation record in `*_generations.jsonl` includes a `logprobs` field containing per-token log probabilities. This is useful for diagnosing cases where the model is "almost correct":

```bash
python3 -c "
import json
with open('output/my-model/dataset_generations.jsonl') as f:
    row = json.loads(f.readline())
    for tok in (row.get('logprobs') or [[]])[0][:10]:
        print(tok)
"
```

Use cases:
- **Check if the correct answer was the second-most-likely token** — if so, a prompt or parsing tweak may recover it.
- **Verify the model is not truncating** — if the sequence ends with a high-probability non-EOS token, `max_tokens` may be too low.
- **Compare answer token probability distributions** across model versions or prompt formats.

To enable logprobs for a run:

```bash
eval360 --max-generation-jobs 1 --max-grading-parallelism 20 \
  evaluate-now --logprobs \
  --model-paths model.yaml \
  --data-paths dataset.yaml
```

---

## 7. Re-running grading without re-running generation

If you want to iterate on the grader or parser without re-generating (which requires a VLLM job), delete the grades and scores files while leaving the generations file intact:

```bash
OUTPUT=output/my-model/my_dataset
rm ${OUTPUT}_grades.jsonl ${OUTPUT}_scores.yaml
```

Then re-run `evaluate-now` with the same config. The scheduler will read the existing `*_generations.jsonl` directly and skip VLLM entirely, then re-grade from scratch.

This is the fastest way to test grader or parser changes — no GPU allocation needed.

---

## 8. Model version

Make sure `revision` in your model config is pinned to the same commit as the reference evaluation:

```yaml
remote_model:
  path: IFM/K2-Think-V2
  revision: abc1234    # pin to the exact commit used in the reference
```

Leaving `revision: null` uses the latest commit on the default branch, which may differ from what the reference evaluated.
