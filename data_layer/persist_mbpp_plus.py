#!/usr/bin/env python3
"""Convert MBPP+ (EvalPlus v0.2.0) into Eval360-ready JSONL.

Loads the MBPP+ dataset from evalplus, runs canonical solutions to compute
expected outputs, and generates assert strings that embed those expected
values directly.  The resulting JSONL is drop-in compatible with the existing
MBPPDaytonaGrader (``mbpp-daytona``).

Assert generation follows five categories matching evalplus's comparison
semantics exactly:

1. **SET_EQ** (8 tasks)    — ``assert set(fn(*args)) == set(expected)``
2. **NOT_NONE** (3 tasks)  — ``assert (fn(*args) is not None) == bool``
3. **surface_Area** (1)    — accept canonical *or* oracle output
4. **digit_distance_nums** (1) — accept canonical *or* oracle output
5. **Default** (365 tasks) — exact match or ``abs()``-tolerance for floats

Float auto-promotion: when ``atol == 0`` but the canonical output is a float,
``atol`` is promoted to ``1e-6`` (matching evalplus's ``is_floats`` check).
The tolerance formula ``abs(out - exp) <= atol + 1e-07 * abs(exp)`` replicates
``np.allclose(out, exp, rtol=1e-07, atol=atol)`` for scalars.

Usage::

    python data_layer/persist_mbpp_plus.py output.jsonl
    python data_layer/persist_mbpp_plus.py output.jsonl \\
        --prompt-suffix "Wrap your code in a python code block."
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import signal
from collections import Counter

from evalplus.data import get_mbpp_plus
from evalplus.eval._special_oracle import (
    MBPP_OUTPUT_NOT_NONE_TASKS,
    MBPP_OUTPUT_SET_EQ_TASKS,
    _digit_distance_nums,
    _surface_Area,
)

_CANONICAL_TIMEOUT_SECS = 10


# ------------------------------------------------------------------
# Canonical-solution execution
# ------------------------------------------------------------------

class CanonicalTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise CanonicalTimeout("Canonical solution timed out")


def run_canonical(
    canonical_solution: str,
    entry_point: str,
    inputs: list,
) -> list:
    """Execute *canonical_solution* and return expected outputs for every input.

    Uses ``signal.alarm`` to enforce a per-call timeout so that a
    pathological canonical solution does not hang the pipeline.

    Note: Python's ``exec`` is used here on *trusted* canonical solutions
    from the EvalPlus dataset — not on untrusted user input.
    """
    ns: dict = {}
    exec(canonical_solution, ns)  # noqa: S102 — trusted canonical code
    fn = ns[entry_point]

    outputs: list = []
    for inp in inputs:
        # Deep-copy to protect against in-place mutation (e.g. heap_sort
        # calls heapify + heappop on the list argument).
        inp_copy = copy.deepcopy(inp)
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(_CANONICAL_TIMEOUT_SECS)
        try:
            outputs.append(fn(*inp_copy))
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    return outputs


# ------------------------------------------------------------------
# Assert generation
# ------------------------------------------------------------------

def safe_repr(value) -> str:
    """Return a ``repr``-like string that is valid Python without extra imports.

    Handles two edge cases that ``repr()`` gets wrong:

    * ``float('inf')`` / ``float('nan')`` — ``repr()`` produces bare ``inf``
      / ``nan`` which are not Python literals.
    * ``collections.Counter`` — ``repr()`` produces ``Counter({...})`` which
      needs an import.  We emit the equivalent ``dict`` literal instead,
      relying on ``Counter.__eq__(dict)`` being True.
    """
    if isinstance(value, float):
        if math.isinf(value):
            return "float('-inf')" if value < 0 else "float('inf')"
        if math.isnan(value):
            return "float('nan')"
        return repr(value)
    if isinstance(value, Counter):
        return repr(dict(value))
    if isinstance(value, (list, tuple)):
        inner = ", ".join(safe_repr(x) for x in value)
        if isinstance(value, tuple):
            return f"({inner},)" if len(value) == 1 else f"({inner})"
        return f"[{inner}]"
    if isinstance(value, dict):
        items = ", ".join(
            f"{safe_repr(k)}: {safe_repr(v)}" for k, v in value.items()
        )
        return "{" + items + "}"
    if isinstance(value, set):
        if not value:
            return "set()"
        inner = ", ".join(safe_repr(x) for x in sorted(value, key=repr))
        return "{" + inner + "}"
    return repr(value)


def is_float_output(value) -> bool:
    """Check if *value* is a float or a homogeneous list/tuple of floats.

    Mirrors evalplus's ``is_floats`` (minus the numpy path, which never
    appears in canonical MBPP+ outputs).
    """
    if isinstance(value, float):
        return True
    if isinstance(value, (list, tuple)) and value:
        return all(isinstance(i, float) for i in value)
    return False


def generate_assert(
    entry_point: str,
    inp: list,
    expected,
    atol: float,
) -> str:
    """Return an assert string for one *(input, expected-output)* pair.

    The returned string may span multiple lines (e.g. for list-of-float
    element-wise checks).
    """
    args_repr = ", ".join(safe_repr(a) for a in inp)
    call = f"{entry_point}({args_repr})"

    # ---- special oracles (checked first, matching evalplus order) ----

    if entry_point in MBPP_OUTPUT_SET_EQ_TASKS:
        sorted_exp = sorted(expected, key=repr)
        return f"assert set({call}) == set({safe_repr(sorted_exp)})"

    if entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
        expected_bool = expected is not None
        return f"assert ({call} is not None) == {expected_bool}"

    if entry_point == "surface_Area":
        oracle_out = _surface_Area(*inp)
        if oracle_out == expected:
            return f"assert {call} == {safe_repr(expected)}"
        return (
            f"_r = {call}\n"
            f"assert _r == {safe_repr(expected)} or _r == {safe_repr(oracle_out)}"
        )

    if entry_point == "digit_distance_nums":
        oracle_out = _digit_distance_nums(*inp)
        if oracle_out == expected:
            return f"assert {call} == {safe_repr(expected)}"
        return (
            f"_r = {call}\n"
            f"assert _r == {safe_repr(expected)} or _r == {safe_repr(oracle_out)}"
        )

    # ---- float tolerance (auto-promote atol for float outputs) ----

    effective_atol = atol
    if effective_atol == 0 and is_float_output(expected):
        effective_atol = 1e-6

    if effective_atol != 0:
        # inf/nan can't use tolerance arithmetic (inf - inf = nan).
        # Use exact match for inf, math.isnan for nan (matching evalplus
        # where inf == inf passes via exact_match before np.allclose).
        if isinstance(expected, float) and (math.isinf(expected) or math.isnan(expected)):
            if math.isnan(expected):
                return f"import math\nassert math.isnan({call})"
            return f"assert {call} == {safe_repr(expected)}"
        if isinstance(expected, (list, tuple)):
            lines = [f"_r = {call}"]
            lines.append(f"assert len(_r) == {len(expected)}")
            for j, exp_elem in enumerate(expected):
                tol = f"{effective_atol} + 1e-07 * abs({safe_repr(exp_elem)})"
                lines.append(
                    f"assert abs(_r[{j}] - {safe_repr(exp_elem)}) <= {tol}"
                )
            return "\n".join(lines)
        return (
            f"assert abs({call} - {safe_repr(expected)}) "
            f"<= {effective_atol} + 1e-07 * abs({safe_repr(expected)})"
        )

    # ---- default exact match ----
    return f"assert {call} == {safe_repr(expected)}"


# Skip individual test cases whose assert repr exceeds this size.
# Some tasks (e.g. combinations_colors) produce combinatorially huge outputs
# that would make the ground_truth impractically large.  The remaining tests
# (100+ per task) still provide strong signal.
_MAX_ASSERT_LEN = 10_000


def build_ground_truth(
    entry_point: str,
    canonical_solution: str,
    inputs: list,
    atol: float,
) -> str:
    """Build the full ground-truth string (all asserts joined by newlines)."""
    expected_outputs = run_canonical(canonical_solution, entry_point, inputs)
    asserts: list[str] = []
    for inp, exp in zip(inputs, expected_outputs):
        line = generate_assert(entry_point, inp, exp, atol)
        if len(line) <= _MAX_ASSERT_LEN:
            asserts.append(line)
    return "\n".join(asserts)


# ------------------------------------------------------------------
# Prompt building
# ------------------------------------------------------------------

def extract_description(prompt: str) -> str:
    """Extract the plain-English description from an MBPP+ docstring prompt.

    The prompt looks like::

        \"\"\"
        Write a function to ...
        assert fn(...) == ...
        \"\"\"

    We strip the triple-quote markers and return everything before the
    first ``assert`` line.
    """
    text = prompt.strip()
    if text.startswith('"""'):
        text = text[3:]
    if text.endswith('"""'):
        text = text[:-3]
    parts = re.split(r"\nassert\s", text, maxsplit=1)
    return parts[0].strip()


def build_completion_prompt(
    prompt: str,
    entry_point: str,
) -> tuple[str, str]:
    """Return ``(completion_input, completion_prefix)`` for base models.

    ``completion_input`` is the full text sent to the model; it ends with
    an incomplete ``def`` stub.  ``completion_prefix`` is the stub itself,
    which the grader prepends to the model's raw completion to form
    runnable code.
    """
    prefix = f"def {entry_point}("
    completion_input = f"{prompt}\n{prefix}"
    return completion_input, prefix


def strip_docstring_markers(prompt: str) -> str:
    """Strip triple-quote markers from an MBPP+ prompt, returning the content."""
    text = prompt.strip()
    if text.startswith('"""'):
        text = text[3:]
    if text.endswith('"""'):
        text = text[:-3]
    return text.strip()


def build_chat_prompt(
    raw_prompt: str,
    entry_point: str,
    prompt_suffix: str | None,
) -> str:
    """Build an instruction-style prompt for instruct/chat models.

    Includes the full problem description **and** example assertions from
    the MBPP+ prompt.  EvalPlus passes these examples to instruct models;
    they provide critical type/signature information that significantly
    affects accuracy.
    """
    content = strip_docstring_markers(raw_prompt)
    prompt = f"{content}\nThe function should be named `{entry_point}`."
    if prompt_suffix:
        prompt = f"{prompt}\n\n{prompt_suffix}"
    return prompt


# ------------------------------------------------------------------
# Main conversion
# ------------------------------------------------------------------

def convert(output_path: str, prompt_suffix: str | None = None) -> None:
    data = get_mbpp_plus()
    sorted_tasks = sorted(
        data.items(), key=lambda x: int(x[0].split("/")[-1])
    )

    count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for row_idx, (task_id, task) in enumerate(sorted_tasks):
            entry_point = task["entry_point"]
            prompt = task["prompt"]
            canonical = task["canonical_solution"]
            atol = task["atol"]
            base = task["base_input"] if isinstance(task["base_input"], list) else []
            plus = task["plus_input"] if isinstance(task["plus_input"], list) else []
            all_inputs = base + plus

            description = extract_description(prompt)
            completion_input, completion_prefix = build_completion_prompt(
                prompt, entry_point,
            )
            user_content = build_chat_prompt(
                prompt, entry_point, prompt_suffix,
            )
            ground_truth = build_ground_truth(
                entry_point, canonical, all_inputs, atol,
            )

            record = {
                "row": row_idx,
                "completion_input": completion_input,
                "completion_prefix": completion_prefix,
                "chat_input": [{"role": "user", "content": user_content}],
                "ground_truth": ground_truth,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} records to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MBPP+ (EvalPlus v0.2.0) to Eval360 JSONL.",
    )
    parser.add_argument(
        "output_jsonl", help="Path to write the output JSONL file.",
    )
    parser.add_argument(
        "--prompt-suffix",
        default=None,
        help="Extra instructions appended to the user prompt in chat_input.",
    )
    args = parser.parse_args()
    convert(args.output_jsonl, args.prompt_suffix)


if __name__ == "__main__":
    main()
