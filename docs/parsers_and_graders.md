# Parsers and Graders

## Overview

**Parsers** are configured on the **model** YAML (`parser_type`). They transform a raw generation string into the text that the grader sees — stripping reasoning tags, extracting a boxed answer, pulling out a code block, etc.

**Graders** are configured on the **dataset** YAML (`grader.type`). They take the parsed generation and score it against the ground truth.

The two are independent: any parser can be combined with any grader (as long as the output format matches what the grader expects).

---

## Parsers

Parsers are registered in `scheduler/grader/base_parsers.py` and `scheduler/grader/parser_registry.py`.

| Name(s) | What it does | Typical use case |
|---|---|---|
| `passthrough` / `noop` / `identity` | Returns the generation unchanged. | Base models or graders that do their own parsing internally. |
| `mc_answer` / `multiple_choice_answer` / `think_tag_mc` | Strips `</think>` prefix, then extracts a single A–E letter from the last line, first line, explicit `<answer>` tag, `\boxed{}`, or "answer: X" pattern. Falls back to full text. Normalises digits 1–5 to A–E. | Instruct/thinking models on multiple-choice benchmarks (MMLU, GPQA, ARC). |
| `the_answer_is` | Finds the first occurrence of `"the answer is"` followed by a letter A–E (with optional brackets/punctuation). | Base models that end answers with "the answer is X". |
| `the_answer_is_last` | Same as `the_answer_is` but takes the **last** occurrence. | Base models that loop or re-evaluate, repeating "the answer is" multiple times before settling. |
| `so_the_answer_is` | Finds the first occurrence of `"the (correct) answer is"` and returns everything after it (up to the next character). | Older base model format. |
| `so_the_answer_is_last` | Same as `so_the_answer_is` but takes the **last** occurrence. | Base models that loop. |
| `answer_tag` | Extracts text inside `<answer>…</answer>` tags. | Reasoning models prompted to wrap their final answer in `<answer>` tags. |
| `think_suffix` / `think_tag` | Splits on `</think*>` and returns the tail. Returns `None` if there is no text after the tag. | Instruct/thinking models where the answer follows `</think>`. |
| `boxed` | Extracts the content of the **last** `\boxed{…}` or `\fbox{…}` (brace-balanced). | Math benchmarks where the model writes the answer in LaTeX boxed notation. |
| `gsm8k_base` | Strips `### `, `**Answer:**`, `Answer:`, `boxed{`, `The answer is` prefixes/sections and returns the remainder. | GSM8K-style base models. |
| `code` | Extracts the **last** ` ```python ` fenced code block; prefers the last block containing a `def`/`class` to skip trailing example usage. | Coding benchmarks (HumanEval) on instruct models. |
| `code_completion` / `humaneval` | Like `code`, but also strips `</think>` prefixes (for thinking models). Falls back to extracting the last fenced block or the last paragraph if the `<think>` tag is unclosed. | HumanEval on thinking/instruct models. |
| `code-think` | Strips `</think>` prefix, then extracts the last ` ```python ` or ` ```python3 ` fenced block. | Thinking models on coding benchmarks where code must be inside a fence. |

---

## Graders

Graders are registered in `scheduler/grader/`.

| Name(s) | File | What it grades | Notes |
|---|---|---|---|
| `multiple_choice` / `multiple-choice` | `multiple_choice.py` | Exact match of extracted letter (A–E) against ground truth. | Parser must emit a single letter. |
| `choice_scoring` / `multiple_choice_nll` | `choice_scoring.py` | Base-model negative log-likelihood over structured answer choices. | Requires dataset YAML opt-in plus rows with `scoring_mode: choice_scoring` and `scoring_completions`; no parser is used for correctness. |
| `exact-match` / `exact_match` | `match.py` | Case-insensitive exact string match of parsed generation vs ground truth. | Used by BBH. |
| `math-verify` / `math_verify` | `math.py` | Symbolic math equivalence via `math_verify` library. | Used by AIME, GSM8K. Handles LaTeX, fractions, decimals. |
| `math_verify_llm_as_judge` / `math-verify-llm-as-judge` | `math_verify_llm_as_judge.py` | Tries `math-verify` first; if it can't decide, falls back to an LLM-as-judge call. | Used by MATH-500. |
| `sympy-llm-as-judge` / `sympy_llm_as_judge` | `sympy_llm_as_judge.py` | Tries SymPy symbolic equivalence; falls back to LLM-as-judge. | Alternative to `math_verify_llm_as_judge` using SymPy. |
| `llm_as_judge` / `boxed-llm-as-judge` | `llm_as_judge.py` | Sends generation + ground truth to an LLM judge and returns its verdict. | Configurable model via `llm_as_judge` block in dataset YAML. |
| `llm_as_judge_lcr` | `llm_as_judge.py` | LLM-as-judge variant tuned for LCR (long-context reasoning). | Used by LCR dataset. |
| `humaneval` / `human_eval` / `human-eval` | `humaneval.py` | Runs HumanEval test cases locally against the generated code. | |
| `humaneval-plus-daytona` | `humaneval_plus_daytona.py` | Runs HumanEval+ test cases in isolated Daytona sandboxes. | Requires Daytona credentials. |
| `mbpp-daytona` | `mbpp_daytona.py` | Runs MBPP+ test cases in isolated Daytona sandboxes. | Requires Daytona credentials. |
| `lcbv6-daytona` | `lcbv6_daytona.py` | Runs LiveCodeBench v6 test cases in isolated Daytona sandboxes. | Requires Daytona credentials. |
| `ifeval` / `if_eval` | `ifeval.py` | Evaluates instruction-following using the IFEval rubric (keyword/format/length constraints). | Downloads NLTK's English `punkt_tab` with `nltk.download` on first use. |
| `ruler` | `ruler.py` | Evaluates RULER long-context retrieval tasks (NIAH, QA, etc.). | |
| `aptbench` | `aptbench.py` | Grader for AptBench spatial/logical reasoning tasks. | |
| `countdown` / `cd` | `countdown.py` | Grader for Countdown number puzzles. | |
| `knights-and-knaves` / `kk` | `knights_and_knaves.py` | Grader for Knights-and-Knaves logic puzzles. | |
| `order-puzzle` / `order` | `order_puzzle.py` | Grader for ordering/sequencing puzzles. | |
| `sum-puzzle` / `sum` | `sum_puzzle.py` | Grader for sum/arithmetic puzzles. | |

---

## Dataset → Grader + Parser mapping

`parser_type` is set on the **model** YAML, not the dataset. The table below shows which parsers are typically used with each dataset based on model configs in `model_zoo/` and `examples/`.

| Dataset YAML | Grader | Typical `parser_type` on model | Notes |
|---|---|---|---|
| `data_zoo/mmlu.yaml` | `multiple_choice` | `mc_answer`, `the_answer_is`, `the_answer_is_last` | Instruct: `mc_answer`; base: `the_answer_is` |
| `data_zoo/mmlu_pro.yaml` | `multiple_choice` | `mc_answer`, `the_answer_is`, `the_answer_is_last` | Same family as MMLU |
| `data_zoo/mmlu_pro_5shot.yaml` | `multiple_choice` | `the_answer_is`, `the_answer_is_last` | 5-shot prompting |
| `data_zoo/mmlu_pro_choice_scoring.yaml` | `choice_scoring` | `noop` / `passthrough` | Base/completions NLL path for MMLU-Pro |
| `data_zoo/mmlu_pro_5shot_choice_scoring.yaml` | `choice_scoring` | `noop` / `passthrough` | Base/completions NLL path for 5-shot MMLU-Pro |
| `data_zoo/mmlu_pro_reasoning.yaml` | `multiple_choice` | `mc_answer` | Reasoning model variant |
| `data_zoo/mmlu_redux_config.yaml` | `multiple_choice` | `mc_answer`, `the_answer_is` | |
| `data_zoo/mmlu_redux_2.0_config.yaml` | `multiple_choice` | `mc_answer`, `the_answer_is` | |
| `data_zoo/gpqa-diamond/gpqa_diamond.yaml` | `multiple_choice` | `mc_answer` | |
| `data_zoo/arc_challenge.yaml` | `multiple_choice` | `mc_answer` | |
| `data_zoo/arc_challenge_25shot.yaml` | `multiple_choice` | `mc_answer` | 25-shot ARC |
| Choice-scoring-capable rows in MMLU, MMLU-Pro, MMLU-Redux, ARC-Challenge, ArabicMMLU, and compatible imported lm-eval tasks | `choice_scoring` only when a dataset YAML opts in | `noop` / `passthrough` | Base/completions NLL path. Dedicated MMLU-Pro choice-scoring YAMLs are provided; other datasets need their own opt-in YAMLs after rows are regenerated with `scoring_completions`. GPQA-Diamond intentionally remains on `multiple_choice`. |
| `data_zoo/bbh.yaml` | `exact_match` | `so_the_answer_is`, `answer_tag`, `passthrough` | |
| `data_zoo/gsm8k.yaml` | `math-verify` | `boxed`, `gsm8k_base`, `passthrough` | |
| `data_zoo/aime-2024.yaml` | `math-verify` | `boxed` | |
| `data_zoo/aime-2025.yaml` | `math-verify` | `boxed` | |
| `data_zoo/aime-2026.yaml` | `math-verify` | `boxed` | |
| `data_zoo/math500.yaml` | `math_verify_llm_as_judge` | `noop` / `passthrough` | LLM judge uses the raw generation |
| `data_zoo/math500_reasoning.yaml` | `math_verify_llm_as_judge` | `noop` / `passthrough` | Reasoning model variant |
| `data_zoo/humaneval.yaml` | `humaneval` | `code_completion`, `code` | |
| `data_zoo/humaneval_plus.yaml` | `humaneval-plus-daytona` | `code_completion`, `code` | |
| `data_zoo/mbpp_plus.yaml` | `mbpp-daytona` | `code_completion`, `code` | |
| `data_zoo/lcbv6.yaml` | `lcbv6-daytona` | `code-think`, `code` | |
| `data_zoo/lcr.yaml` | `llm_as_judge_lcr` | `think_tag` (set in dataset YAML on the judge model) | LCR uses LLM-as-judge; `parser_type` is on the judge model spec inside the dataset YAML |
| `data_zoo/ifeval.yaml` | `ifeval` | `passthrough` / `noop` | IFEval grader inspects raw text and uses local tokenization helpers |
| `data_zoo/ruler/ruler_*.yaml` | `ruler` | `passthrough` / `noop` | Various context lengths (4k–128k) |
| `data_zoo/aptbench.yaml` | `aptbench` | `passthrough` | |
| `data_zoo/cd_config.yaml` | `countdown` / `cd` | `answer_tag`, `passthrough` | |
| `data_zoo/kk_config.yaml` | `knights-and-knaves` / `kk` | `passthrough` | |
| `data_zoo/order_config.yaml` | `order-puzzle` / `order` | `passthrough` | |
| `data_zoo/sum_config.yaml` | `sum-puzzle` / `sum` | `passthrough` | |
