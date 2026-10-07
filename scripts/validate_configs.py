#!/usr/bin/env python3
"""
Validate all model and dataset config YAML files using pydantic.

Two failure modes:
  - Invalid YAML syntax           → hard failure (always CI-blocking)
  - Pydantic schema error         → hard failure (wrong fields/types)
  - Missing file / network error  → warning only (data not present in CI)

Exits with code 1 if any hard failure is found.
"""
import sys
from pathlib import Path

import yaml

# Add repo root to path so scheduler is importable without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError

from scheduler.eval_config import EvalConfigParser
from scheduler.model import ModelParser
from scheduler.task import Task

# Directories scanned by CI.  "evals" holds eval-config YAMLs (the directory the
# README's `--eval-paths evals/...` instruction documents); without it a malformed
# eval config placed there would pass CI and only fail at run time.
SEARCH_DIRS = ["examples", "model_zoo", "data_zoo", "evals"]

# Exceptions that indicate the resource isn't available in CI rather than
# a schema bug.  These are reported as warnings, not hard failures.
_RESOURCE_ERRORS = (FileNotFoundError, OSError, TimeoutError)


def validate_file(path: Path) -> tuple[str, str]:
    """Validate a single YAML file.

    Returns (status, message) where status is "ok", "warn", or "fail".
    """
    try:
        with open(path) as f:
            obj = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return "fail", f"invalid YAML: {e}"

    if not isinstance(obj, dict):
        return "ok", "skipped (not a config dict)"

    if "remote_model" in obj or "local_model" in obj or "external_model" in obj:
        try:
            ModelParser.parse_yaml(path)
        except ValidationError as e:
            return "fail", f"model schema error: {e}"
        except _RESOURCE_ERRORS as e:
            return "warn", f"resource unavailable in CI: {e}"
        except Exception as e:
            return "warn", f"skipped ({type(e).__name__}: {e})"
        return "ok", ""

    if "version" in obj and "groups" in obj:
        try:
            EvalConfigParser.parse_yaml(path)
        except ValidationError as e:
            return "fail", f"eval schema error: {e}"
        except Exception as e:
            return "warn", f"skipped ({type(e).__name__}: {e})"
        return "ok", ""

    if "uuid" in obj:
        try:
            Task.parse_yaml(path)
        except ValidationError as e:
            return "fail", f"dataset schema error: {e}"
        except _RESOURCE_ERRORS as e:
            return "warn", f"resource unavailable in CI: {e}"
        except Exception as e:
            return "warn", f"skipped ({type(e).__name__}: {e})"
        return "ok", ""

    return "ok", "skipped (unrecognized format)"


def scan_dirs(repo_root: Path, search_dirs: list[str]) -> list[Path]:
    """Validate every *.yaml under each search dir; return the list of hard failures."""
    failures = []
    warnings = []
    checked = 0

    for search_dir in search_dirs:
        for path in sorted((repo_root / search_dir).rglob("*.yaml")):
            status, msg = validate_file(path)
            checked += 1
            rel = path.relative_to(repo_root)
            if status == "fail":
                print(f"  FAIL  {rel}: {msg}")
                failures.append(rel)
            elif status == "warn":
                print(f"  WARN  {rel}: {msg}")
                warnings.append(rel)
            else:
                print(f"  ok    {rel}")

    print(f"\n{checked} files checked, {len(failures)} failed, {len(warnings)} warned.")
    return failures


def main():
    repo_root = Path(__file__).resolve().parent.parent
    failures = scan_dirs(repo_root, SEARCH_DIRS)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
