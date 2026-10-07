#!/usr/bin/env python3
"""Build the canonical Eval360 math500 JSONL from the HuggingFace source.

completion_input is formatted as a 4-shot Minerva-style prompt so base models
produce \boxed{} answers without any prompt_prefix_instructions in the model
spec. chat_input is left unchanged for instruct model use.

Usage:
    python data_layer/build_math500.py \
        --output-jsonl data_zoo/math500/math500.jsonl \
        [--overwrite]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

HF_REPO = "<HF_ORG>/<EVAL_SOURCES_REPO>"
HF_FILENAME = "math500/math500.jsonl"

# 4-shot Minerva/MATH CoT preamble matching the lm-evaluation-harness minerva_math task.
# Each solution ends with \boxed{} and the "Final Answer: ... I hope it is correct." trailer,
# which teaches the model the expected answer format and provides a natural stopping point.
# Copied from lm-evaluation-harness lm_eval/tasks/minerva_math/utils.py (MIT, Copyright (c)
# 2020 EleutherAI); the problems are from hendrycks/math (MIT). See THIRD_PARTY_NOTICES.md.
_FEWSHOT_PREAMBLE = """\
Problem: Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.
Solution: The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.

Problem: If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$
Solution: We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$
Final Answer: The final answer is $24$. I hope it is correct.

Problem: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?
Solution: If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$:
\\begin{align*}
30n&=480\\\\
\\Rightarrow\\qquad n&=480/30=\\boxed{16}
\\end{align*}
Final Answer: The final answer is $16$. I hope it is correct.

Problem: If the system of equations $6x-4y=a$ and $6y-9x=b$ has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.
Solution: If we multiply the first equation by $-\\frac{3}{2}$, we obtain

$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have

$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$
Final Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.

"""


def build_math500_records(source_path: Path) -> list[dict[str, Any]]:
    records = []
    with source_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            user_message = next(
                (m["content"] for m in row["chat_input"] if m["role"] == "user"),
                row["completion_input"].removeprefix("You are a helpful assistant.\n\n"),
            )

            completion_input = _FEWSHOT_PREAMBLE + f"Problem: {user_message}\nSolution:"

            records.append({
                "row": row["row"],
                "completion_input": completion_input,
                "chat_input": row["chat_input"],
                "ground_truth": row["ground_truth"],
            })
    return records


def write_jsonl(records: list[dict[str, Any]], output_path: Path, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace it.")
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Eval360 math500 JSONL with 4-shot preamble")
    parser.add_argument("--output-jsonl", type=Path, required=True,
                        help="Path to write the rebuilt JSONL")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Downloading {HF_FILENAME} from {HF_REPO}...")
    source_path = Path(hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILENAME,
        repo_type="dataset",
    ))
    records = build_math500_records(source_path)
    write_jsonl(records, args.output_jsonl, args.overwrite)
    print(f"Wrote {len(records)} records to {args.output_jsonl}")

    first = records[0]
    print(f"\nSample row 0 completion_input:\n{first['completion_input']}")


if __name__ == "__main__":
    main()
