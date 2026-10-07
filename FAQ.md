# FAQ

## Why does scheduler startup fail with `ModuleNotFoundError: No module named 'daytona_sdk'`?

The scheduler installs its Python dependencies from `pyproject.toml`, including
`daytona-sdk`. At startup it imports grader modules, so a missing `daytona-sdk`
dependency can fail immediately before any evaluation runs.

If your environment was created before that dependency was added, refresh it:

```bash
python3.14 -m pip install --upgrade -e /path/to/Eval360
```

You can verify the package is installed with:

```bash
python3.14 -m pip show daytona-sdk
```

## Why does `evaluate-now` fail with `IndexError: string index out of range` during grading?

If the error comes from `scheduler/grader/multiple_choice.py`, the usual cause
is a stale `*_generations.jsonl` file that still contains an older empty
generation from a previous run.

Re-run with `--force` to clear cached generations, grades, and scores before
regenerating:

```bash
eval360 \
  --max-generation-jobs 1 \
  --max-grading-parallelism 20 \
  --log-dir logs \
  evaluate-now \
  --force \
  --model-paths <model-config.yaml> \
  --data-paths <data-config.yaml>
```
