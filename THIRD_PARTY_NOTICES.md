# Third-party notices

Eval360 is Copyright IFM and is licensed under the Apache License, Version 2.0
(see `LICENSE`). It includes the third-party code and data listed below, each
under its own license. Files that were changed from the upstream version say so
in their own header or docstring.

## Code

| Path in this repository | Upstream | License | Copyright |
|---|---|---|---|
| `scheduler/grader/bbh_lib/filters.py` (extraction filters, one function changed) | [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | MIT | Copyright (c) 2020 EleutherAI |
| `data_layer/build_math500.py` (`_FEWSHOT_PREAMBLE`, the 4-shot Minerva prompt from `lm_eval/tasks/minerva_math/utils.py`; its problems come from [hendrycks/math](https://github.com/hendrycks/math)), `data_layer/build_gpqa_diamond.py` (`_preprocess`, adapted from `lm_eval/tasks/gpqa/zeroshot/utils.py`) | [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | MIT | Copyright (c) 2020 EleutherAI; Copyright (c) 2021 Dan Hendrycks (MATH problems) |
| `scheduler/grader/mbpp_local.py` (`_extract_code`, a reimplementation inspired by `extract_code_blocks` in `lm_eval/tasks/mbpp/utils.py`) | [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | MIT | Copyright (c) 2020 EleutherAI |
| `scheduler/grader/humaneval.py` (the sandboxed-execution section, `_TimeoutException` through `check_correctness`, adapted from `human_eval/execution.py`) | [openai/human-eval](https://github.com/openai/human-eval) | MIT | Copyright (c) OpenAI (https://openai.com) |
| `scheduler/grader/sympy_llm_as_judge.py` (Sections 1a and 1b: MATH normalization and sympy grading utilities) | [openai/prm800k](https://github.com/openai/prm800k) (`grading/math_normalize.py`, `grading/grader.py`), itself largely derived from [hendrycks/math](https://github.com/hendrycks/math) | MIT | Copyright (c) 2023 OpenAI; Copyright (c) 2021 Dan Hendrycks |
| `scheduler/grader/math_verify_llm_as_judge.py`, `scheduler/grader/sympy_llm_as_judge.py` (`EQUALITY_TEMPLATE`) | [openai/simple-evals](https://github.com/openai/simple-evals) (`common.py`) | MIT | Copyright (c) 2024 OpenAI |
| `scheduler/grader/llm_as_judge.py` (the opening sentences of `GRADER_TEMPLATE`) | [langchain-ai/langchain](https://github.com/langchain-ai/langchain) (QA eval prompt, `evaluation/qa/eval_prompt.py`) | MIT | Copyright (c) LangChain, Inc. |
| `scheduler/grader/countdown.py` (`validate_equation`, `evaluate_equation` and the tolerance check in `score_equation`, modified) | [Jiayi-Pan/TinyZero](https://github.com/Jiayi-Pan/TinyZero) (`verl/utils/reward_score/countdown.py`) | Apache-2.0 | NOTICE: Copyright 2023-2024 Bytedance Ltd. and/or its affiliates |
| `scheduler/grader/knights_and_knaves.py` (`parse_solution_text_format`; `parse_model_answer`, modified) | [Unakar/Logic-RL](https://github.com/Unakar/Logic-RL) (`verl/utils/reward_score/kk.py`) | Apache-2.0 | NOTICE: Copyright 2023-2024 Bytedance Ltd. and/or its affiliates |
| `scheduler/imported_dataset/bfcl.py` (in the generated benchmark script: `_query_FC`, `_add_vllm_reasoning_content_if_available` and `_serper_search_engine_query`, modified) | [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla) `berkeley-function-call-leaderboard` at commit `f7cf7359b7ac615a0b294831c5ba2bc95ee4a000` (`bfcl_eval/model_handler/api_inference/openai_completion.py`, `bfcl_eval/eval_checker/multi_turn_eval/func_source_code/web_search.py`) | Apache-2.0 | No copyright line or NOTICE file upstream |
| `scheduler/grader/aptbench.py` (answer extraction functions, modified) | [TencentYoutuResearch/APTBench](https://github.com/TencentYoutuResearch/APTBench) (`code/predict.py`) | Apache-2.0 | Copyright (C) 2025 Tencent |
| `scheduler/grader/ifeval_lib/*.py` (from commit `5b09c22d73a9d35eb6c5d2a99b95677a45053466`; every file's package import is relative, and `instructions.py` and `instructions_util.py` are further modified) | [google-research/google-research](https://github.com/google-research/google-research/tree/master/instruction_following_eval) `instruction_following_eval` | Apache-2.0 | Copyright 2026 The Google Research Authors (per the file headers) |
| `scheduler/grader/hle.py`, `scheduler/grader/hle_aa.py` (judge prompt, lightly normalized) | [centerforaisafety/hle](https://github.com/centerforaisafety/hle) (`hle_eval/run_judge_results.py`) | MIT | Copyright (c) 2025 centerforaisafety |
| `scheduler/grader/lcr_aa.py` (`OFFICIAL_JUDGE_TEMPLATE`) | [ArtificialAnalysis/AA-LCR](https://huggingface.co/datasets/ArtificialAnalysis/AA-LCR) (dataset card at revision `bdae010bbce259820c0e34c1d7cce210d966fb75`) | Apache-2.0 | No copyright line stated upstream (license from the dataset card) |

## Data

| Path in this repository | Upstream | License | Notes |
|---|---|---|---|
| `data_zoo/mbpp_plus/mbpp_plus.jsonl`; individual items used as fixtures in `tests/test_mbpp_daytona.py` and `tests/test_persist_mbpp_plus.py` | MBPP+ v0.2.0 from [evalplus/evalplus](https://github.com/evalplus/evalplus), derived from [MBPP](https://github.com/google-research/google-research/tree/master/mbpp) by Google Research | Apache-2.0 (MBPP+); [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) (MBPP) | Modified: reformatted into the Eval360 JSONL record format. |
| `examples/data/mmlu_diy.jsonl`, `examples/data/mmlu_philosophy.jsonl`; three example rows in `data_layer/README.md` | MMLU philosophy test split from [hendrycks/test](https://github.com/hendrycks/test) | MIT | Copyright (c) 2020 Dan Hendrycks. Modified: reformatted into the Eval360 JSONL record format. |

## NOTICE files of Apache-2.0 upstreams

Jiayi-Pan/TinyZero and Unakar/Logic-RL each ship a `Notice.txt` that reads, in
full:

```
Copyright 2023-2024 Bytedance Ltd. and/or its affiliates
```

The Apache License, Version 2.0 is in `LICENSE` and at
<https://www.apache.org/licenses/LICENSE-2.0>.

## MIT License

The following text applies to every item above marked MIT, with that item's
copyright line in place of the placeholder:

```
MIT License

Copyright (c) <year> <copyright holders>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Creative Commons Attribution 4.0

MBPP is licensed under CC-BY-4.0: <https://creativecommons.org/licenses/by/4.0/>.
