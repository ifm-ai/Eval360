#!/usr/bin/env python3
"""Build the math500_reasoning JSONL for 0-shot CoT evaluation.

completion_input is formatted as a 0-shot reasoning prompt that instructs the
model to produce <think>...</think><answer>\\boxed{...}</answer> output.
chat_input is left unchanged for instruct model use.

Usage:
    python data_layer/build_math500_reasoning.py \
        --output-jsonl data_zoo/math500_reasoning/math500_reasoning.jsonl \
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

# 0-shot reasoning prompt matching the K2-V2 paper evaluation setup.
# The model is expected to produce:
#   <think> reasoning </think><answer> \boxed{answer} </answer>
_SYSTEM_PROMPT = (
    "The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then "
    "provides the user with the final answer. The reasoning process and answer are "
    "enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think>"
    "<answer> answer here </answer>. "
    "In the answer, box your final answer using \\boxed{} notation. "
    "Now the user asks you to solve a math reasoning problem."
)


def build_math500_reasoning_records(source_path: Path) -> list[dict[str, Any]]:
    records = []
    with source_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            user_message = next(
                (m["content"] for m in row["chat_input"] if m["role"] == "user"),
                row["completion_input"].removeprefix("You are a helpful assistant.\n\n"),
            )

            completion_input = f"{_SYSTEM_PROMPT}\n\nUser:{user_message}\nAssistant: <think>"

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
    parser = argparse.ArgumentParser(description="Build math500_reasoning JSONL with 0-shot CoT prompt")
    parser.add_argument("--output-jsonl", type=Path, required=True,
                        help="Path to write the JSONL")
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
    records = build_math500_reasoning_records(source_path)
    write_jsonl(records, args.output_jsonl, args.overwrite)
    print(f"Wrote {len(records)} records to {args.output_jsonl}")

    first = records[0]
    print(f"\nSample row 0 completion_input:\n{first['completion_input']}")


if __name__ == "__main__":
    main()
