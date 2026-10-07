#!/usr/bin/env python3
"""Build the canonical Eval360 GPQA-Diamond JSONL from the official GPQA CSV."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import uuid
from pathlib import Path
from typing import Any

import yaml


SYSTEM_PROMPT = (
    "You are a helpful assistant.\n\n"
    "The following are multiple choice questions (with answers). "
    "Choose the correct answer from the options.\n\n"
)


def _preprocess(text: str | None) -> str:
    # Adapted from lm-evaluation-harness lm_eval/tasks/gpqa/zeroshot/utils.py::preprocess
    # (MIT, Copyright (c) 2020 EleutherAI). See THIRD_PARTY_NOTICES.md.
    if text is None:
        return ""
    value = text.strip()
    value = value.replace(" [title]", ". ")
    value = re.sub(r"\[.*?\]", "", value)
    value = value.replace("  ", " ")
    return value


def _question_prompt(question: str, choices: list[str]) -> str:
    return (
        f"{question}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        "Answer:"
    )


def build_gpqa_record(row: dict[str, str], row_index: int, seed: int = 0) -> dict[str, Any]:
    question = _preprocess(row["Question"])
    correct = _preprocess(row["Correct Answer"])
    incorrect = [
        _preprocess(row["Incorrect Answer 1"]),
        _preprocess(row["Incorrect Answer 2"]),
        _preprocess(row["Incorrect Answer 3"]),
    ]

    choices_with_correct = [(text, False) for text in incorrect] + [(correct, True)]
    rng = random.Random(seed + row_index)
    rng.shuffle(choices_with_correct)

    shuffled_choices = [text for text, _ in choices_with_correct]
    answer_idx = next(i for i, (_, is_correct) in enumerate(choices_with_correct) if is_correct)
    answer_letter = chr(ord("A") + answer_idx)

    user_prompt = _question_prompt(question=question, choices=shuffled_choices)
    completion_input = f"{SYSTEM_PROMPT}{user_prompt}"

    return {
        "row": row_index,
        "ground_truth": answer_letter,
        "completion_input": completion_input,
        "chat_input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }


def build_gpqa_dataset(input_csv: Path, seed: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "Question",
            "Correct Answer",
            "Incorrect Answer 1",
            "Incorrect Answer 2",
            "Incorrect Answer 3",
        }
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            missing_display = ", ".join(sorted(missing))
            raise ValueError(f"Missing required GPQA columns: {missing_display}")

        for idx, row in enumerate(reader):
            records.append(build_gpqa_record(row=row, row_index=idx, seed=seed))

    return records


def write_jsonl(records: list[dict[str, Any]], output_jsonl: Path, overwrite: bool) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if output_jsonl.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_jsonl}. Pass --overwrite to replace it.")

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def write_dataset_config(
    output_path: Path,
    dataset_name: str,
    data_path: str,
    num_generations: int,
    semantic_version: str,
    average_over: list[int],
    pass_at: list[int],
    meta: dict[str, Any],
    dataset_uuid: str | None,
) -> None:
    payload = {
        "uuid": dataset_uuid or str(uuid.uuid4()),
        "grader": {"type": "multiple_choice"},
        "average_over": average_over,
        "pass_at": pass_at,
        "dataset_name": dataset_name,
        "data_path": data_path,
        "semantic_version": semantic_version,
        "num_generations": num_generations,
        "meta": meta,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _parse_meta(value: str | None) -> dict[str, Any]:
    default_meta = {"split": "test", "priority": "high", "fewshot": 0}
    if value is None:
        return default_meta

    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--meta-json must decode to a JSON object")

    merged = dict(default_meta)
    merged.update(parsed)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Eval360 GPQA-Diamond JSONL from GPQA CSV")
    parser.add_argument("--input-csv", type=Path, required=True, help="Path to gpqa_diamond.csv")
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Path to write Eval360 JSONL")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic shuffle seed offset")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSONL output")

    parser.add_argument("--config-output", type=Path, default=None, help="Optional dataset YAML output path")
    parser.add_argument("--dataset-name", default=None, help="Dataset name for generated YAML")
    parser.add_argument("--dataset-uuid", default=None, help="UUID for generated YAML")
    parser.add_argument("--semantic-version", default="1.0.0", help="Semantic version for generated YAML")
    parser.add_argument(
        "--config-data-path",
        default=None,
        help="data_path value to write in YAML (defaults to absolute output-jsonl path)",
    )
    parser.add_argument("--average-over", nargs="+", type=int, default=[1], help="average_over values")
    parser.add_argument("--pass-at", nargs="+", type=int, default=[1], help="pass_at values")
    parser.add_argument("--meta-json", default=None, help="Optional JSON object to merge into meta")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_gpqa_dataset(input_csv=args.input_csv, seed=args.seed)
    write_jsonl(records=records, output_jsonl=args.output_jsonl, overwrite=args.overwrite)

    if args.config_output is not None:
        dataset_name = args.dataset_name or "gpqa_diamond"
        config_data_path = args.config_data_path or str(args.output_jsonl.expanduser().resolve())
        meta = _parse_meta(args.meta_json)
        write_dataset_config(
            output_path=args.config_output,
            dataset_name=dataset_name,
            data_path=config_data_path,
            num_generations=len(records),
            semantic_version=args.semantic_version,
            average_over=args.average_over,
            pass_at=args.pass_at,
            meta=meta,
            dataset_uuid=args.dataset_uuid,
        )


if __name__ == "__main__":
    main()
