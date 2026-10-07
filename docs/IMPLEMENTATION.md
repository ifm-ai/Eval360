# Eval360 Implementation Reference

A quick-reference guide to the scheduler's internals for contributors.

---

## High-Level Architecture

```
Model YAML + Dataset YAML
        │
        ▼
   Scheduler (scheduler.py)
   ├── FSManager        — watches dirs for new YAML configs (long-running mode)
   ├── JobManager       — submits/polls Slurm jobs, allocates VLLM nodes
   ├── EventManager     — tracks (model, dataset) pairs; produces EventInstances
   ├── GenerationManager — async iterators that stream raw generations
   └── GradingManager   — async iterators that stream grades/scores
        │
        ▼
  handle_event()  →  generations JSONL
        │
        ▼
  handle_generation()  →  grades JSONL + scores JSONL
```

---

## Key Data Structures

### Generation Record (written to `*_generations.jsonl`)
Each record is a deep copy of the input dataset row with extra fields added:
```json
{
  "row": 0,
  "completion_input": "...",
  "chat_input": [{"role": "user", "content": "..."}],
  "ground_truth": "A",
  "generations": ["output1", "output2"],
  "finish_reasons": ["stop", "length"],
  "logprobs": [{...}, {...}],
  "reasoning": ["step-by-step thinking..."],
  "tool_calls": [[{"type": "function", "function": {"name": "search", "arguments": "{}"}}]],
  "raw_text_input": "<|start|>system<|message|>..."
}
```
- `finish_reasons` is a list parallel to `generations`, one entry per choice (`"stop"`, `"length"`, etc.). Present whenever the API returns a non-null finish_reason.
- `logprobs` is present only when requested (always in `evaluate-now`, config-driven in long-running)
- `logprobs` is a list parallel to `generations`, each entry is `.model_dump()` of the OpenAI SDK logprobs object
- Choice-scoring rows do not add `generations`; instead they add `choice_scoring_full_logprobs`, `choice_scoring_completion_logprobs`, and `choice_scoring_metadata`. These are raw echo-logprob payloads for each structured `scoring_completions` entry, with request fallback metadata recorded when vLLM rejects a batched choice prompt.
- `reasoning` is present when the model returns reasoning content via the API response (e.g. o-series, VLLM thinking models). Extracted from `choice.message.model_extra["reasoning_content"]` or `["reasoning"]`. List parallel to `generations`.
- `tool_calls` is present when the model returns tool calls via the API response. Each entry is a list of tool call dicts (from `choice.message.tool_calls`). List parallel to `generations`.
- `raw_text_input` is present only in `--debug` mode. For BASE models: the constructed prompt string. For CHAT models: the fully-templated prompt retrieved via VLLM `/tokenize` + `/detokenize` endpoints.

### Grade Record (written to `*_grades.jsonl`)
The generation record plus grader-added fields:
```json
{
  "...all generation fields...",
  "parsed_generations": ["A", null],
  "parsed_reasoning": ["step-by-step thinking...", null],
  "parsed_answer": ["A", null],
  "parsed_tool_calls": [["search()"], null],
  "correct": [1, 0],
  "picked": ["A", null]
}
```
- `parsed_reasoning`, `parsed_answer`, `parsed_tool_calls` are extracted by `_extract_text_metadata()` in `grader/base.py` from the raw generation text. They detect `<think*>...</think*>` blocks (including `<think_fast>`, `<think_faster>`, etc.) and `<tool_call>...</tool_call>` blocks. These are always populated when the content is present in the generation text, regardless of parser type. They supplement (not replace) the API-level `reasoning` and `tool_calls` fields.

For Daytona-based graders (e.g. LCBv6), additional fields:
```json
{
  "...all generation fields...",
  "parsed_generations": ["code...", null],
  "correct": [1, 0],
  "evaluation_details": ["pass", "wrong_answer"],
  "sandbox_results": [[{...per_test_case_details...}], null]
}
```
- `evaluation_details`: per-generation string reason (`"pass"`, `"wrong_answer"`, `"timeout"`, `"no_output"`)
- `sandbox_results`: per-generation list of per-test-case dicts enriched with `"expected"` field; `null` when details unavailable; omitted entirely if all generations returned `null` details

### Error Records in generation/grade files
When `make_request` fails unrecoverably, an `ExceptionWrapper` is yielded instead of a normal dict. These are written to the JSONL as plain dicts with `"exception"` and `"trace"` fields (and `"generations": []` or similar partial state). **Error records count as incorrect: the grader sets `correct=[0]` and includes them in the accuracy denominator.** This ensures a dead server that corrupts 10% of generations is reflected as ≤90% accuracy rather than being silently excluded.

- `dataset_iterator` (with `read_only=False`) now yields error records rather than truncating the file at them. This prevents the file from being silently wiped on the next run.
- Error records are distinguished by `"exception" in sample` in `async_grade_all_samples` / `async_grade_all_samples_nonblocking`.

### Score Record (written to `*_scores.jsonl`)
```json
{"accuracy (avg over 1)": 0.75, "accuracy (pass@1)": 0.75, "bootstrap_std (avg over 1)": 0.02}
```

---

## External Models (`external_model` + `--no-slurm`)

External models point to a pre-existing OpenAI-compatible endpoint — no Slurm job is submitted.

**Registration flow (differs from VLLM models):**
`register_model_spec` detects `model_spec.external_model is not None`, creates a `ModelInstance` with `is_external=True`, and **immediately pre-populates** `LOCKED_CONNECTIONS[serving_key]` with the endpoint URL. This means `make_request` can acquire a connection slot right away without waiting for `handle_job_update` to fire (which is the normal VLLM path).

**`--no-slurm` flag:**
When passed, `Scheduler.__init__` sets `self.slurm_manager = None` and `self._no_slurm = True`. In `run_evaluate_now`:
- The Slurm source task is not created (no squeue polling)
- `remote_model`/`local_model` configs are rejected with a clear error
- `cancel_all_owned_jobs` and `handle_job_update(None)` are skipped in the finally block

**`get_desired_allocation` skips external models:** External models have `is_external=True`. The allocation loop skips them so no Slurm node is ever requested for them.

**Client creation:** For external models, `_get_client` passes `base_url=url` directly (user-supplied, already includes `/v1`), and `api_key=model.api_key` (resolved from the env var named by `api_key_env` at registration time). For VLLM models, `base_url=f"{url}/v1"` is appended and `api_key="fake key"`.

**Rate limiting and the shared request kernel:**
If `requests_per_minute` is set on an external model, `RateLimiter` is shared
through the module-level `RATE_LIMITERS` registry. The key is an opaque quota
identity derived from the canonical endpoint and a SHA-256 credential
fingerprint; neither the logical model name nor the raw credential is retained
as quota identity. The bucket starts full and refills at `rpm / 60` permits per
second. A waiter parks on an `asyncio.Condition`, so an unused-permit refund can
wake it immediately without waiting behind the refill interval.

At this stack layer, the existing `make_request` caller remains on its legacy
path and acquires the rate permit before the connection-pool lease. The new
`ExternalRequestRunner` defines the replacement order—pool lease, client setup,
provider permit, then HTTP attempt—and cancellation-safe cleanup. Dependent
stack layers wire ordinary generation, choice scoring, and judges to that
kernel; do not infer that defining the kernel alone reroutes those callers.

**VLLM-specific features skipped for external models:**
- `_get_templated_prompt` returns `None` immediately (no `/tokenize` endpoint)
- `_poll_vllm_metrics` task is replaced with `asyncio.sleep(0)` (no `/metrics` endpoint)
- Dead-URL eviction logic in `make_request` is not triggered (external URL is permanent)

**`serving_key` for external models:** `ModelInstance.serving_key` uses `None` for `vllm_cli_args` and `venv_path` when they are absent. This ensures the hash is distinct from any VLLM model, even one with the same name.

---

## Generation Pipeline (`openai_interface.py`)

`OpenAIConnection` manages all API calls for one (model, event) pair.

**Key parameters:**
- `openai_kwargs` = merged `task.openai_settings` + `model.openai_kwargs`
- `force_logprobs=True` (set by `evaluate-now`) injects `logprobs=True` (chat) or `logprobs=5` (base) if not already set
- `cache_salt` = typed model config serialized into `extra_body.cache_salt` only when enabled
- `max_simultaneous_requests` — per-model semaphore enforced via `asyncio.Condition`

**Cache salt support:** `ModelSpec.cache_salt` / `ModelInstance.cache_salt` supports `disabled`, `static`, and `unique` modes. Disabled is the default and sends no cache salt. Static mode sends the configured salt for deterministic cache partitioning. Unique mode generates a fresh per-outbound-request salt, which is useful for cache-bypass validation and stale-cache debugging. Cache salt is configured only through the typed model field; raw `openai_kwargs.cache_salt`, task `openai_settings.cache_salt`, and `extra_body.cache_salt` are rejected. Generation and LLM-as-judge code paths both serialize cache salt through `scheduler.cache_salt.request_kwargs_with_cache_salt`. `evaluate-now --salt-cache` requires `--force`, overrides generation and LLM-as-judge model config to unique mode, and writes the effective setting to `<dataset>_run_metadata.yaml`. Resuming from existing generations is allowed only when the existing run metadata has matching cache-salt settings; otherwise rerun with `--force`.

**Lazy deployment (`_ensure_ready`):**
`OpenAIConnection` does NOT connect to VLLM at construction time. The URL is resolved and the HTTP client created lazily inside `launch_requests.consumer()`, immediately before the first real API request is dispatched. If the input stream is exhausted before any API request is needed (i.e. all generations are already in the output file and `Sentinel.COMPLETED` is the first element), `_ensure_ready` is never called and no deployment wait occurs. This is the core mechanism that makes `evaluate-now` fast when resuming a completed run.

`_ensure_ready` calls `job_manager.wait_for_hostname(model)` to get the VLLM URL, then polls `/health` until the endpoint is live. If the job dies before VLLM becomes healthy it raises `EndpointNeverReadyError`. Any exception (including `RuntimeError` from `wait_for_hostname`) is caught by `consumer()`, stored in `consumer_error`, put in the result queue to unblock the main loop, and re-raised after the loop exits.

For external models `_ensure_ready` skips `job_manager.wait_for_hostname` — the URL is already in `LOCKED_CONNECTIONS` from registration time, so `pool.acquire()` returns immediately.

**Multi-replica support:** Multiple VLLM replicas can be deployed for the same model. `LOCKED_CONNECTIONS` (keyed by `serving_key`) maintains a per-URL slot pool. `make_request` acquires a slot from whichever URL has capacity, enabling load balancing across replicas. Dead replicas are evicted via `remove_url`. The scheduler's `get_desired_allocation` distributes available nodes as extra replicas to active generation deployments (Step 3).

**Connection semaphore (`LOCKED_CONNECTIONS`):**
A module-level dict maps `ModelInstance.serving_key` to one
`ModelConnectionPool`. Each pool tracks available slots independently for every
canonical live URL and uses an `asyncio.Condition` to wake blocked callers when
capacity or a new replica becomes available. The pool is shared across all
`OpenAIConnection` instances for that serving identity; conflicting live
capacity registrations are rejected instead of inheriting the first value.

**`make_request(i, elem, num_generations)`** — inner coroutine:
- Deep-copies the input row into `result`
- Loops making batched API calls (up to `n=4` at once) until `num_generations` reached
- Accumulates `result["generations"]` and `result["logprobs"]` across calls
- On HTTP error: retries 3×, then wraps in `ExceptionWrapper`
- On any other exception: wraps immediately in `ExceptionWrapper` (no retry)
- **Null content (`content=None`):** Reasoning models (e.g. deepseek_r1 parser) may return `content=None` with all output in `reasoning_content`. If `finish_reason` is `"stop"` or `"length"`, `content=None` is treated as `""` (empty generation). If `finish_reason` is anything else, a `RuntimeError` is raised immediately (no retry) and the record is written as an error record.
- **"Input too long" (`BadRequestError` with context-length message):** Does NOT produce an `ExceptionWrapper`. Instead produces a normal result dict with `result["eval360_input_too_long"] = True` and `result["generations"] = [""] * num_generations`. This flows into the grader which short-circuits to `correct = [0] * N` without calling `grade_fn`. This ensures the sample is counted as wrong. Note: if `max_tokens` approaches the model's total context length, every request will fail with this error — set `max_tokens < context_length - max_prompt_tokens`.

**Model types:**
- `ModelType.BASE` → `client.completions.create(prompt=elem["completion_input"], ...)`
- `ModelType.CHAT` → `client.chat.completions.create(messages=elem["chat_input"], ...)`
- Dataset YAMLs that select `grader: {type: choice_scoring}` or `multiple_choice_nll` require `ModelType.BASE`, `average_over=[1]`, and `pass_at=[1]`. The request layer sends batched echo-logprob completion requests for `scoring_completions` instead of sampling generated text, applies `prompt_prefix_instructions` to full prompts only, and falls back to one request per choice when a batched request hits vLLM's malformed MessagePack failure. Rows that contain choice-scoring fields but run under another grader still use the normal generative request path.

**Live choice-scoring integration test:** The real-endpoint echo-logprob path is
covered by `tests/integration/test_choice_scoring_vllm.py`. It skips by default.
To run it against an OpenAI-compatible base/completions endpoint, start a server
that supports `echo=True` logprobs and run:

```bash
EVAL360_CHOICE_SCORING_BASE_URL=http://host:8000/v1 \
EVAL360_CHOICE_SCORING_MODEL=served-model-name \
python -m pytest -o addopts='' tests/integration/test_choice_scoring_vllm.py -q
```

Set `EVAL360_CHOICE_SCORING_API_KEY` if the endpoint requires a real key. The
test exercises `OpenAIConnection.launch_requests()` with a structured
`choice_scoring` row, verifies the raw `choice_scoring_full_logprobs`,
`choice_scoring_completion_logprobs`, and `choice_scoring_metadata` payload, then
feeds that row through `ChoiceScoring.run()` and checks the grade and score
shape.

**Logprobs serialization:** OpenAI SDK returns Pydantic models. Call `.model_dump()` on each `choice.logprobs` before storing to make it JSON-serializable. Logprobs are only written to the record if `response.choices[0].logprobs is not None`.

---

## Grading Pipeline (`grader/`)

### Parser Registry (`grader/parser_registry.py`)
- `@register_parser("name")` registers a `Callable[[str], str | None]`
- Selected by `model_instance.parser_type` (from model YAML)
- Applied element-wise to `sample["generations"]` → `sample["parsed_generations"]`
- Built-in parsers in `base_parsers.py`: `passthrough`, `think_tag`, `think_suffix`, `mc_answer`, `boxed`, `code_completion`, `code`, `code-think`, `answer_tag`, `the_answer_is`, `the_answer_is_last`, `so_the_answer_is`, `so_the_answer_is_last`, `gsm8k_base`
- `so_the_answer_is` / `so_the_answer_is_last`: extract text after first/last occurrence of "the (correct) answer is"; `_last` variant is better for base models that loop and re-evaluate before settling on a final answer
- `multiple_choice` grader extracts the first alphabetic character via `re.search(r"[A-Za-z]", sampled)`. If no letter is found and the string is non-empty, uses `sampled.strip()[0]`. If the generation is empty (e.g. truncated by `max_tokens`), `picked=None` and the sample is counted as incorrect.

### Grader Registry (`grader/`)
- `@register("name")` registers a grader class
- Selected by `task.grader.type` (from dataset YAML)
- Auto-discovered at import via `pkgutil.iter_modules`
- External graders via `eval360.graders` entry point

### Choice Scoring
`choice_scoring` / `multiple_choice_nll` grades base-model multiple-choice rows by deriving negative log-likelihood from raw logprobs. Dataset YAML selects this path with `grader.type`; rows must then provide `scoring_mode: choice_scoring` and `scoring_completions`. Rows may also provide `scoring_prompt_prefixes` when each choice needs a distinct full prompt. The shared schema helpers in `scheduler/choice_scoring_schema.py` validate rows, build full and completion-only prompts, and resolve the ground-truth index. See `docs/CHOICE_SCORING_ARCHITECTURE.md` for the row contract and adoption plan.

### `AccuracyGraderBase` (`grader/base.py`)
Key method: `async_grade_all_samples(start, grade_fn)`:
1. Skips first `start` samples (already graded on resume)
2. Passes through `ExceptionWrapper` without grading
3. Short-circuits samples with `eval360_input_too_long=True` → sets `correct=[0]*N`, yields directly without calling `grade_fn`
4. For each new sample: adds `parsed_generations` via parser, calls `grade_fn(sample)`
5. Returns `Grade(element=result)` or `ExceptionWrapper` on error

Graders must implement `grade_sample(sample)` returning a dict with `"correct": list[int]`.

**Concurrent grading (`async_grade_all_samples_nonblocking`):** For LLM-as-judge graders where each sample makes network calls, sequential grading is slow. `async_grade_all_samples_nonblocking` starts all sample gradings as concurrent asyncio tasks and yields results in original order (via an index buffer) for deterministic checkpointing. Subclasses opt in by overriding `_grading_generator`:
```python
def _grading_generator(self, start: int):
    return self.async_grade_all_samples_nonblocking(start=start, grade_fn=self.grade_sample)
```
The default `_grading_generator` uses the sequential `async_grade_all_samples`. `run()` always calls `self._grading_generator(start=count)` rather than `async_grade_all_samples` directly.

**IMPORTANT — `parse_generations` requirement:** Both `async_grade_all_samples` and `async_grade_all_samples_nonblocking` call `self.parse_generations(sample)` before invoking `grade_fn`. This is the hook that populates `parsed_generations`, `parsed_reasoning`, `parsed_answer`, and `parsed_tool_calls` on every sample. If a new grading pathway is added (i.e. a new method that iterates over samples and calls `grade_fn` directly), it **must** call `sample["parsed_generations"] = self.parse_generations(sample)` before calling `grade_fn`, otherwise these fields will be absent from the grade record and metadata extraction will be skipped entirely.

### `MathVerifyLLMasJudge` grader (`grader/math_verify_llm_as_judge.py`)
Hybrid grader for math problems. Uses concurrent grading via `_grading_generator` override.

**Grading flow per generation:**
1. **`math_verify` pass:** Tries `parse_answer_with_verify(generation)` + `compare_answers(pred, gt)` for each ground truth. Returns score=1 immediately on match.
2. **LLM fallback:** If math_verify finds no match, calls an LLM judge. Two candidates are tried in order:
   - `generation` (the parser-extracted expression, e.g. from boxed parser)
   - `raw_generation` (the full model output, if it differs from `generation`)

   The raw generation fallback handles cases where the parser extracts the wrong expression (e.g. a boxed example from the system prompt) but the correct answer is in the prose.

**LLM judge prompts:**
- Short expression (< 200 chars): `EQUALITY_TEMPLATE` — asks "are these two expressions equivalent?"
- Long prose (≥ 200 chars): `SOLUTION_GRADING_TEMPLATE` — asks "did the student arrive at the correct answer?", using only the last 800 chars of the solution
- Strips `</think>` tokens from judge response before checking for "yes"/"no"
- Retries up to 3× on failure

**Diagnostics:** Each grade record includes `grading_diagnostics` with per-generation method/result breakdown and aggregate counts (`math_verify_match_count`, `llm_called_count`, etc.).

### Daytona Graders (`grader/daytona_base.py`)

`DaytonaGraderBase` runs code in ephemeral Daytona sandboxes. It extends `GraderBase` directly (not `AccuracyGraderBase`) to own the full grading pipeline.

**Sandbox execution flow:**
1. `grade_sample(sample)` launches one `asyncio.Task` per generation via `_run_in_sandbox`
2. `_run_in_sandbox` guards: empty/whitespace `parsed_response` → `("no_output", None)` immediately
3. Script is uploaded via `sandbox.fs.upload_file(harness.encode(), "/tmp/harness.py")` and run with `sandbox.process.exec("python /tmp/harness.py", timeout=...)` — NOT `code_run`, which has command-length/escaping limits
4. Unknown `DaytonaError` retries up to `_MAX_RETRIES=3` times before giving up
5. Result unpacked as `(grade, details)` tuple; `grade_sample` stores `evaluation_details` and optionally `sandbox_results`

**Override hooks** (for subclasses):
- `_build_sandbox_script(parsed_response, test_cases) -> str`: builds the harness script uploaded to the sandbox
- `_grade_sandbox_result(response, test_cases) -> tuple[str, list | None]`: compares sandbox output locally and returns `(grade, details)`

**Sandbox naming:** `{user}-eval360-{model}-{dataset}-row-{row}-gen-{gen_index}-{run_id}` — `run_id` is always last (8-char hex)

**`run()` aggregate:** accumulates `Counter[str]` of `evaluation_details` and writes breakdown scores (e.g., `"pass_count"`, `"wrong_answer_count"`, `"timeout_count"`) alongside the standard accuracy metrics.

### HumanEval+ Daytona Grader (`grader/humaneval_plus_daytona.py`)

Subclasses `DaytonaGraderBase` for EvalPlus HumanEval+ (164 problems).

**Dataset format:** `ground_truth` is a JSON string `{"test": ..., "entry_point": ..., "prompt": ...}`.
- `test` — the full self-contained test block (defines `check(candidate)`, `assertion()`, `is_floats()`, and imports numpy; contains the combined HumanEval base + EvalPlus plus test cases; does **not** call `check()` itself)
- `entry_point` — the function name the model must implement
- `prompt` — the function signature + docstring (same as `completion_input`)

**Harness construction:**
```
{prompt}
{generated_code}

{test_code}           <- defines check(), assertion(), is_floats(); imports numpy

[assertion override]  <- wraps assertion() to record {"passed": bool, ...} per call
check({entry_point})
print(json.dumps(_results))
sys.exit(0 if all passed else 1)
```
- `prompt` is always prepended: for completion models `prompt + body = complete function`; for chat models that generate a full function, the docstring-only stub defined by prompt is immediately overridden by the full definition.
- `check({entry_point})` is explicitly appended — the EvalPlus `test` field does not include this call.
- `assertion()` is overridden after the test code defines it. When `check()` runs, every assertion call is intercepted: success appends `{"passed": True}`, failure appends `{"passed": False, "actual": ..., "expected": ...}`. This gives per-assertion granularity without re-implementing the float comparison logic.
- `numpy` comes from the sandbox image — no pip install in the harness script.

**Sandbox image:** `ganler/evalplus:v0.2.1` (~153 MB compressed) — official EvalPlus image on Docker Hub with Python 3.11-slim + numpy pre-installed. Uses `CreateSandboxFromImageParams` (not `CreateSandboxFromSnapshotParams`). The v0.2.1 tag is pinned over `latest` (~1.07 GB) which pulls heavy ML deps (transformers, etc.) that are not needed.

**`_grade_sandbox_result`:** Parses the JSON array from stdout. All-passed → `"pass"`. Any failed → `"wrong_answer"`. Empty output or JSON parse error (function crashed before print) → `"wrong_answer"` with `{"parse_error": True, ...}` detail.

**Data preparation:** `data_layer/build_humaneval_plus.py` loads `evalplus/humanevalplus` from HuggingFace and writes `completion_input=prompt`, `chat_input=[system, user]`, and `ground_truth=json.dumps({"test", "entry_point", "prompt"})`. The dataset has exactly 164 problems (test split only).

### LCBv6 Daytona Grader (`grader/lcbv6_daytona.py`)

Subclasses `DaytonaGraderBase` for LCBv6 code-generation problems. Two problem types:

**`testtype: "stdin"`** (rows 0–111): solution reads from stdin, one subprocess per test case.
- `_build_stdin_harness`: launches subprocesses with `input=test["input"]`, returns JSON array of `{"stdout", "exit_code"}` or `{"timeout": True}`

**`testtype: "functional"`** (rows 112–174): LeetCode-style `class Solution`.
- `_build_functional_harness`: `exec(solution, namespace)`, calls `Solution().method_name(*[json.loads(line) for line in input.splitlines()])`, returns JSON array of `{"stdout": json.dumps(actual), "exit_code": 0}` or `{"error": ..., "exit_code": -1}`
- `method_name` is stored in each test case dict (set by `convert_lcbv6.py` via regex on `starter_code`)

**`_grade_sandbox_result`**: parses the JSON array, compares outputs locally:
- Timeout entry → `"timeout"`
- Non-zero exit code → `"wrong_answer"`
- Mismatch → `"wrong_answer"` (stdin: line-by-line with float coercion; functional: `json.loads` equality)
- JSON parse failure → `[{"parse_error": True, "raw_output": ..., "exit_code": ..., "stderr": ...}]`
- Count mismatch → `[{"count_mismatch": True, "got": N, "expected_count": M}]`
- Each result element is enriched with `"expected"` field from the test case

**Data preparation:** `convert_lcbv6.py` (repo root, not in `scheduler/`) converts raw `lcbv6.jsonl`:
- Decompresses `private_test_cases` (base64 → zlib → pickle → JSON)
- For functional problems: extracts `method_name` from `starter_code` via `r"def (\w+)\(self"`, adds to each test case dict, includes starter code in prompt
- Output format: `{"row", "completion_input", "chat_input", "ground_truth": json.dumps(test_cases)}`

---

## Scheduler Event Loop (`scheduler/scheduler.py`)

### `handle_event(event_instance)` — Generation phase
1. Calls `event_manager.add_desired_model` immediately so Slurm job submission begins concurrently
2. Reads existing generations from JSONL (for resumption)
3. Creates `OpenAIConnection` (no network activity yet — URL is resolved lazily)
4. Streams API responses via `launch_requests`; `_ensure_ready` is called before the first real request, or not at all if the input stream is already exhausted
5. Writes each generation to `*_generations.jsonl`, yields to `GradingManager`

### `handle_generation(event_instance, generation_aiter)` — Grading phase
1. Creates grader instance, passes it the generation iterator
2. Reads existing grades from JSONL (for resumption)
3. Streams new grades, writes to `*_grades.jsonl`
4. After all grades: writes aggregate scores to `*_scores.jsonl`

### `run_evaluate_now` restrictions
- Direct model/data mode accepts `remote_model` and `external_model` configs. `local_model` configs require long-running mode unless they are expanded through eval-config mode.
- `--no-slurm` accepts only `external_model` configs.
- Eval-config mode (`--eval-paths`) accepts `remote_model`, `local_model`, and `external_model` configs, but rejects `imported_dataset` tasks.
- Accepts `remote_model` configs for Slurm-backed runs and `external_model` configs with `--no-slurm`. Local checkpoint-watcher models require the long-running scheduler.
- Sets `_force_logprobs = True` so all generations include logprobs.
- Exits when `count_completed_events() >= total_events` (counts both phase=2 success and phase=-1 failure).

### `handle_event` for `ImportedDatasetEventInstance`
Delegates to `handle_imported_dataset_event`:
1. Checks sentinel files for resumption/failure *before* submitting a new job:
   - `.job_complete` → skip to `parse_results` immediately
   - `.job_failed` → mark event failed (no retry)
2. Submits Slurm job (VLLM + benchmark script on same node)
3. Venv setup runs in a background subshell **concurrently** with VLLM deployment to save wall time
4. Polls for job completion via sentinels written by the bash script:
   - `.setup_complete_job` — venv/deps are ready (written by the background subshell)
   - `.job_complete` — benchmark finished successfully
   - `.job_failed` — benchmark crashed (written by `trap` on non-zero exit)
   - neither after job dies → preempted, re-enqueue
5. Calls `runner.parse_results()` to extract scores

**Venv caching:** Each imported dataset runner gets a venv at `.eval360/envs/<runner_name>/` (relative to `repo_root`). The venv is only created once; the `.setup_complete_job` sentinel prevents re-running setup. **If the runner's `build_setup_script` changes (e.g., new dependencies added), the old venv must be deleted manually** — the sentinel will not detect this.

**`output_dir`** for imported datasets is:
```
pathlib.Path(event_instance.path_to_scores).parent / f"{task.dataset_name}_output"
```
This is passed to the Slurm job as `$output_dir` and set as `BFCL_PROJECT_ROOT` (or equivalent). The benchmark's native output files land here.

### Node allocation
```python
total_occupied = len(pending_jobs) + len(deploying_jobs) + len(live_jobs)  # one per replica
available_nodes = max_generation_jobs - total_occupied - _active_imported_dataset_jobs
```
Both standard VLLM replicas and imported-dataset jobs count against the budget.

`get_desired_allocation` groups desired models by `serving_key` (so sibling models sharing a VLLM deployment are deduplicated). It distributes `available_nodes` in three steps:
1. Give each new generation deployment 1 node (capped at `available_nodes`)
2. Give grader deployments 1 node each from remaining slots
3. Distribute leftover nodes as extra replicas across all generation deployments (new + existing)

Already-running deployments (`created_sks`) don't consume a new node slot for their first replica but are still eligible for extra replicas in step 3. `update_allocation` enforces idempotency: it only launches replicas above the current count.

### Slurm launch details

**Served model aliases:** `update_allocation` owns `--served-model-name`.
Config files must not include that flag manually. For a normal direct model it
passes the model instance name. For eval-config variants it passes the
API-facing base name first, followed by all sibling eval variant names that
share the same `serving_key`. The aliases are emitted under a single
`--served-model-name` flag because vLLM accepts multiple names after one flag.
Every eval variant sends requests through that base name, so the first serving
job remains valid when sibling events become desired incrementally.

**Per-job compile caches:** All standard and imported-dataset sbatch templates
set a user/job-specific cache root:

```bash
CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/eval360-vllm-${UID}-${SLURM_JOB_ID}"
```

They export `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`,
`TRITON_CACHE_DIR`, and `CUDA_CACHE_PATH` underneath that root. This isolates
concurrent jobs and avoids stale shared `/tmp/eval360-vllm` directories owned by
other users or previous container images.

---

## Config System

### Model YAML → `ModelSpec` / `ModelInstance` (`model.py`)
```yaml
remote_model:
  base_name: my-model
  path: org/my-model
  revision: null          # optional HF revision
model_type: instruct      # instruct | base
parser_type: mc_answer    # parser to apply to generations
max_simultaneous_requests: 60
max_time_to_deploy: 900
vllm_cli_args:
  - --tensor-parallel-size 8
  - --gpu-memory-utilization 0.95
serving_slurm_resources:
  gpus_per_node: 8        # equals --tensor-parallel-size above
  cpus_per_task: <CPUS_PER_TASK>
  memory_gb: <MEMORY_GB>
  time_limit: <TIME_LIMIT>
openai_kwargs:
  temperature: 0.0
  logprobs: true          # optional; always set in evaluate-now
owner: user.name
output_path: "~/src/Eval360/output"

# Serving environment — exactly one required:
venv_path: "~/.venvs/vllm-serving/bin/activate"
# OR
# conda_env: "vllm-serving"                            # conda environment name
# OR
# container_image: "nvcr.io/nvidia/pytorch:24.01-py3"  # registry URI or local .sqsh
# container_mounts:                                     # optional bind mounts
#   - "/<SHARED_STORAGE>/models:/models:ro"
```

**Serving environment modes:** Exactly one of `venv_path`, `conda_env`, or `container_image` must be set. Each mode uses a dedicated sbatch script:
- **`venv_path`** — sources `$venv_path/bin/activate` (accepts both venv and the path to `bin/activate` directly)
- **`conda_env`** — initializes conda and runs `conda activate $conda_env`
- **`container_image`** — sbatch commands include pyxis/enroot flags (`--container-image`, `--container-writable`, `--container-mounts`) and skip venv/conda activation entirely. Accepts Docker registry URIs (e.g. `nvcr.io/...`) or local `.sqsh` files (e.g. `/path/to/image.sqsh`). `container_mounts` is optional and only valid with `container_image`.

**Serving Slurm resources:** each replica always uses one task on one node.
`serving_slurm_resources` binds GPUs per node, CPUs per task, optional memory in
GiB, and wall time. `SlurmManager` passes those as explicit `sbatch` arguments;
the launch templates contain no fixed resource or exclusive-node directives.

**`serving_key`** includes `venv_path`, `conda_env`, `container_image`, and
`serving_slurm_resources` so different runtime environments or resource
requests produce different deployment keys even for the same model weights.

**Runtime fields:** Direct model registration requires `owner`, `output_path`,
and `parser_type`. Eval-config mode may omit them from the model YAML because
`EvalConfigParser._build_resolved_pair()` fills `owner` from the eval config,
sets `output_path` to
`os.path.join(group.output_root, base_model.name, pair_token)` (the per-pair
token keeps two selected tasks that share a `dataset_name` from colliding on the
same output files), and sets `parser_type` from the group. `group.output_root`
must be a non-empty absolute path; `~` is expanded and relative paths are
rejected.
`ModelParser.parse_yaml(..., eval_mode=True)` is used only for this eval-driven
expansion: it injects placeholder values for any of these fields that are absent
so `ModelSpec` validation passes, and `_build_resolved_pair()` overwrites all
three per resolved pair so the placeholders never reach generation or output.
Direct registration (`parse_yaml` without `eval_mode`) and
`DatabaseManager.register_model_family()` still raise a clear
`ValueError`/`ValidationError` when the fields are missing.

**`ModelInstance.name`, `api_model_name`, and `path`** — critical distinction:
- `model_instance.name` = internal scheduler/output identity. In direct mode it is usually the `base_name` from YAML; in eval-config mode it is suffixed with `-eval-<token>` to keep pairs isolated.
- `model_instance.api_model_name` = API-facing alias used by `OpenAIConnection` when it must differ from `name`. External models set it to `external_model.base_name`; eval-config variants set it to the base model name. If it is `None`, requests use `model_instance.name`.
- `model_instance.path` = HF model ID or local filesystem path passed to `vllm serve`. Not suitable as an API model name (may contain `/` or not exist as a filesystem path).
- Use `api_model_name or name` for outbound API requests, and use `path` only when the serving process or benchmark needs the underlying model weights/tokenizer location.

### Eval Config YAML (`--eval-paths`)
```yaml
version: 1
owner: user.name
groups:
  - name: reasoning-aime
    model_tag: reasoning
    data_tag: aime
    parser_type: boxed
    output_root: /<OUTPUT_ROOT>/results
    grader:
      type: math-verify
    vllm_cli_args:
      - --tensor-parallel-size 8
    openai_overrides:
      temperature: 0.0
      max_tokens: 32768
```

A group that changes `--tensor-parallel-size` needs a model config whose `gpus_per_node` matches.

Eval-config mode changes registration from "cartesian product" to explicit
tag-selected groups:
- `--model-paths` and `--data-paths` are candidate pools; eval groups select
  matching configs by tag.
- Model configs are matched when `model_spec.tag == group.model_tag`.
- Dataset configs are matched when `task.tag == group.data_tag`.
- Each `(model config path, dataset config path)` pair may be produced by only
  one eval group; overlaps are rejected.
- Each concrete pair gets a stable 12-character token derived from eval config
  path, group name, model path, and task path.
- The model variant gets `name=<base-name>-eval-<token>`,
  `api_model_name=<base-name>`, eval-level `owner`, output path under
  `group.output_root`, group parser, and group vLLM args if provided.
- The task variant gets a matching private tag, merged request settings
  (`task.openai_settings` -> `model.openai_kwargs` -> `group.openai_overrides`),
  and the group grader when one is provided.
- `--served-model-name` is still prohibited in group `vllm_cli_args`; the Slurm
  manager injects aliases for all sibling variants that share the deployment.

### Dataset YAML → `AsyncGenerationTask` or `ImportedDatasetTask` (`task.py`)
```yaml
# Standard task
uuid: my-eval-v1
dataset_name: mmlu
semantic_version: "1.0.0"
grader:
  type: multiple_choice
average_over: [1]
pass_at: [1]
data_path: "~/src/Eval360/data/*.jsonl"

# Imported dataset task
uuid: bfcl-non-live
dataset_name: bfcl
semantic_version: "1.0.0"
imported_dataset:
  name: bfcl              # matches a registered runner
  commit: abc123
  args:
    test_category: non_live
```

---

## Output File Layout
```
<output_path>/<model-name>/
├── <dataset>_generations.jsonl   # raw API outputs
├── <dataset>_grades.jsonl        # per-sample grading results
├── <dataset>_scores.jsonl        # aggregate accuracy metrics
└── <dataset>_run_metadata.yaml   # effective run-level request metadata

# For imported dataset tasks:
<output_path>/<model-name>/
├── <dataset>_output/             # benchmark's native output dir
│   ├── .setup_complete_job       # venv+deps installed
│   ├── .job_complete             # benchmark ran to completion
│   └── .job_failed               # internal crash (no retry)
└── <dataset>_scores.jsonl        # scores parsed from benchmark output
```

`<dataset>_run_metadata.yaml` records the effective cache-salt setting used for
generation so resumption cannot silently mix salted and unsalted outputs:

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

Static cache salts are redacted in metadata and include `value_sha256_12` for
comparison without writing the raw salt value.

---

## Resumption

Every pipeline stage is resumption-safe:
- **Generations**: existing JSONL is read first; only missing rows are generated
- **Grades**: existing grades JSONL is counted; grading starts at `start=count`
- **Imported datasets**: if `.job_complete` exists, skip straight to `parse_results`
- **Cache salt metadata**: existing generations can resume only when the stored
  cache-salt metadata matches the current model/CLI setting. Missing, malformed,
  or mismatched metadata requires `--force` when cache salt is enabled.

The DB tracks event phase (0=generating, 1=grading, 2=complete, -1=failed).

`count_completed_events()` counts phase=2 OR phase=-1 (both success and failure count as "done" for the purpose of `evaluate-now` exit). `count_successful_events()` counts only phase=2.

When normal model/data/eval YAML inputs include `--terminal-result-path`, the
CLI creates a `YamlTerminalInvocation` before parsing and passes it through
`run_evaluate_now(..., yaml_terminal_invocation=...)`. This is an output-only
evidence mode, not a second input contract: the existing YAML files remain the
authoritative definitions and no request, suite, catalog, or other input
manifest is generated. Schema 1.0 requires `--eval-paths` and supports only
standard generation/grading tasks, not `imported_dataset` tasks.

`YamlTerminalInvocation` binds the natural installed runner executable from
`sys.argv[0]` plus the exact ordered model, data, and eval YAML files. Callers
must therefore invoke the installed entrypoint normally; a wrapper must not
rewrite `argv[0]` to name different bytes. After eval-group resolution, each
selected event binds its model/data/eval source files and group, resolved model
and task definition digests, ordered concrete dataset files, and ordered
generation/grade/score/run-metadata outputs. The scheduler records controller
generation, grading, and aggregation units and reconciles every shared
model-serving Slurm child with its exact event/role completions and terminal
root-job `sacct` outcome.

`publish_yaml_terminal_result()` verifies the bound inputs, outputs, event/job
links, and scheduler outcomes again. It uses the same exclusive, fsynced,
write-last publication primitive as the request-native result and writes only
after complete success. Failure, interruption, incomplete or unsuccessful
child accounting, changed files, or an existing target produces no new success
artifact and never replaces the target. Omitting `--terminal-result-path`
retains the ordinary YAML behavior without this evidence capture.

When `--evaluation-request-path` and `--terminal-result-path` are supplied,
`evaluate-now` uses the request-native evidence route. The two flags are
all-or-none and mutually exclusive with legacy YAML inputs. The strict
canonical request selects strict runner-definition, suite-catalog, and suite
owner manifests. It binds the exact actual runner entrypoint, release and
serving manifests and payloads, ordered task/dataset closure, execution paths,
and output root. In-memory `ModelSpec` and task objects are derived from that
closure without generated YAML. Existing outputs also carry request and task
closure digests, so a stale resume fails unless `--force` is explicit.

`scheduler/terminal_result.py` rechecks every bound input and required output
at final publication and publishes canonical JSON through an exclusive hard
link only after all evidence is complete. The temporary inode is fully written
and fsynced before the terminal name becomes visible; the parent directory is
then fsynced, and a failed post-link fsync removes the terminal name.

`SlurmManager` retains every submitted or adopted root job ID for the scheduler
instance. Evidence reconciliation queries `sacct` for
`JobIDRaw,JobName%256,State,ExitCode,Reason`, ignores job-step rows, and waits
without an elapsed-time deadline while a present root record is non-terminal.
A missing root record fails reconciliation. `COMPLETED` is successful only at
exit `0:0`; a `CANCELLED` model-serving child is accepted only with Eval360's
recorded `scheduler_release` intent and an exact successful event/role
completion recorded before release. Ambiguous `sbatch --parsable` output and
failed cancellation propagate on this route. Other terminal states,
unassociated jobs, missing required units, or changed bindings prevent
terminal-result publication. Ordinary YAML CLI behavior is unchanged when the
request/result flags are omitted.

**Forcing a re-run:** Pass `--force` to `evaluate-now` to clear existing state:
- Regular tasks: deletes `*_generations.jsonl`, `*_grades.jsonl`, `*_scores.jsonl`, and `*_run_metadata.yaml`
- Imported dataset tasks: removes `.job_complete`, `.job_failed`, `.setup_complete_job` sentinels (does **not** delete the benchmark output itself)

---

## Adding a Grader
1. Create `scheduler/grader/my_grader.py`
2. Subclass `AccuracyGraderBase`, decorate with `@register("my_grader")`
3. Implement `async grade_sample(self, sample)` — must set `result["correct"]`
4. Set `grader: {type: my_grader}` in dataset YAML

## Adding a Parser
1. Add to any file in `scheduler/grader/` (or `base_parsers.py`)
2. Decorate with `@register_parser("my_parser")`
3. Signature: `(generation: str) -> str | None`
4. Set `parser_type: my_parser` in model YAML

## Adding an Imported Dataset Runner
See `scheduler/imported_dataset/README.md`.

For BFCL web-search usage, including Serper setup, dataset/model parameters,
and launch commands, see `docs/BFCL_WEB_SEARCH.md`.

**BFCL-specific notes** (`imported_dataset/bfcl.py`):
- Uses `model_instance.name` (not `.path`) as the BFCL model name — this must match `--served-model-name` used by VLLM.
- Registers the model in `MODEL_CONFIG_MAPPING` using a K2 Horizon vLLM handler defined inline in the script (adapted from BFCL's `OpenAICompletionsHandler`, Apache-2.0; see `THIRD_PARTY_NOTICES.md`) plus a small Eval360 subclass **in-process** (not via subprocess) so the registration persists across the generate → evaluate steps. The PyPI `bfcl-eval` package used in the job venv has no handler for this model, so none is imported from the package at runtime. The entire benchmark script runs as a heredoc piped to `"$VENV/bin/python" -`.
- `bfcl_eval` imports `qwen_agent` at module level which requires `soundfile` — included in `build_setup_script`.
- `bfcl_eval` uses **Typer** not Click: call `cli([...])` directly, not `cli.main(...)`. Typer raises `SystemExit(0)` on success; catch it in the generate step so evaluate still runs.
- Do not pass `--local-model-path`: `bfcl_eval` validates it as a filesystem path, which fails for HF model IDs.
- Score files are written to `$BFCL_PROJECT_ROOT/score/<model_name>/BFCL_v4_*_score.json`.
