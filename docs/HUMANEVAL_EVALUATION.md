# HumanEval Evaluation

## Dataset

HumanEval is a benchmark of 164 hand-written Python programming problems from OpenAI's "Evaluating Large Language Models Trained on Code" paper. Each problem contains:

| Field | Description |
|---|---|
| `task_id` | Unique ID (e.g. `HumanEval/0`) |
| `prompt` | Function signature + docstring (the model's input) |
| `entry_point` | Name of the function to call during testing |
| `canonical_solution` | Reference implementation (function body) |
| `test` | A `check(candidate)` function with assert-based test cases |

## Evaluation Protocol

1. **Generate**: The model receives the `prompt` (incomplete function) and must produce the function body (the "completion").
2. **Assemble**: The check program is constructed as: `prompt + completion + test + "check(entry_point)"`.
3. **Execute**: The assembled program runs in a sandboxed subprocess with a timeout (default 10s). Safety guards disable dangerous OS functions (`os.kill`, `subprocess.Popen`, `shutil.rmtree`, etc.).
4. **Verdict**: If execution completes without error, the sample **passes**. Otherwise it **fails** (syntax error, assertion failure, timeout, or runtime exception).

## pass@k Metric

HumanEval uses the **pass@k** metric: the probability that at least one of `k` sampled completions passes all tests.

Given `n` total samples per problem with `c` correct:

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

This is an unbiased estimator (no need to cherry-pick). Standard values: pass@1, pass@10, pass@100.

## Eval360 Integration

- **Dataset format**: Each JSONL record stores the prompt in `completion_input` and the test suite + entry point in `ground_truth` (as a dict).
- **Grader**: `scheduler/grader/humaneval.py` executes each generation against the test suite using the sandboxed `check_correctness` function from the `human_eval` package.
- **Parser**: The `code_completion` parser extracts code from markdown fences when chat/instruct models wrap their output.
- **Metrics**: The base `AccuracyGraderBase` computes pass@k using the same unbiased estimator.
