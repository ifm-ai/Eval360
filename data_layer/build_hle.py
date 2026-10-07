#!/usr/bin/env python3
"""Build Eval360-compatible JSONL from the HLE dataset (text-only questions).

Filters out image-based questions (image field is non-empty) and writes one
JSONL record per text-only question. answer_type is stored in each record so
the hle grader can dispatch to the correct scoring logic at grade time.

Usage:
    python data_layer/build_hle.py \
        --output-jsonl data_zoo/hle/hle.jsonl \
        [--overwrite]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_MC_SYSTEM = (
    "You are a helpful assistant.\n\n"
    "The following are multiple choice questions (with answers). "
    "Choose the correct answer from the options."
)

_EXACT_SYSTEM = "You are a helpful assistant."

_EXACT_INSTRUCTION = "Provide only your final answer."


def _build_record(row_index: int, row: dict[str, Any]) -> dict[str, Any]:
    question = row["question"].strip()
    answer = row["answer"]
    answer_type = row["answer_type"]

    if answer_type == "multipleChoice":
        user_prompt = f"{question}\nAnswer:"
        completion_input = f"{_MC_SYSTEM}\n\n{user_prompt}"
        chat_input = [
            {"role": "system", "content": _MC_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
    else:
        completion_input = f"{_EXACT_SYSTEM}\n\n{_EXACT_INSTRUCTION}\n\n{question}"
        chat_input = [
            {"role": "system", "content": f"{_EXACT_SYSTEM}\n\n{_EXACT_INSTRUCTION}"},
            {"role": "user", "content": question},
        ]

    return {
        "row": row_index,
        "completion_input": completion_input,
        "chat_input": chat_input,
        "ground_truth": answer,
        "answer_type": answer_type,
        "category": row.get("category", ""),
        "raw_subject": row.get("raw_subject", ""),
        "id": row.get("id", ""),
    }


def build_hle_records(split: str = "test") -> list[dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset("cais/hle", split=split)
    table = ds.data.table

    records = []
    row_index = 0
    for i in range(len(table)):
        image = table.column("image")[i].as_py()
        if image:
            continue
        row = {col: table.column(col)[i].as_py() for col in table.schema.names if col != "image"}
        records.append(_build_record(row_index, row))
        row_index += 1

    return records


def write_jsonl(records: list[dict[str, Any]], output_path: Path, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace it.")
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Eval360 HLE JSONL (text-only questions)")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Path to write JSONL")
    parser.add_argument("--split", default="test", help="Dataset split (default: test)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading HLE dataset...")
    records = build_hle_records(split=args.split)

    mc = sum(1 for r in records if r["answer_type"] == "multipleChoice")
    exact = sum(1 for r in records if r["answer_type"] == "exactMatch")
    print(f"Text-only questions: {len(records)} (multipleChoice: {mc}, exactMatch: {exact})")

    write_jsonl(records, args.output_jsonl, args.overwrite)
    print(f"Wrote {len(records)} records to {args.output_jsonl}")

    print("\nSample multipleChoice record:")
    for r in records:
        if r["answer_type"] == "multipleChoice":
            print(f"  completion_input: {r['completion_input'][:200]}")
            print(f"  ground_truth: {r['ground_truth']}")
            break

    print("\nSample exactMatch record:")
    for r in records:
        if r["answer_type"] == "exactMatch":
            print(f"  completion_input: {r['completion_input'][:200]}")
            print(f"  ground_truth: {r['ground_truth']}")
            break


if __name__ == "__main__":
    main()
