#!/usr/bin/env python3
"""Build canonical Eval360 ARC-Challenge JSONL datasets."""

from __future__ import annotations

import argparse
import json
import random
import string
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from scheduler.choice_scoring_schema import choice_scoring_fields


DATASET_ID = "allenai/ai2_arc"
SUBSET = "ARC-Challenge"
TEST_SPLIT = "test"
TRAIN_SPLIT = "train"
LETTER_LABELS = string.ascii_uppercase
MAX_CHOICES = 5

SYSTEM_PROMPT = (
    "You are a helpful assistant.\n\n"
    "The following are multiple choice science questions. "
    "Choose the correct answer from the options and answer with the correct letter.\n\n"
)


class ArcBuildError(ValueError):
    """Raised when the ARC source rows cannot be converted safely."""


@dataclass(frozen=True)
class ArcQuestion:
    question: str
    choices: tuple[str, ...]
    answer_letter: str


def _expand_path(path: Path) -> Path:
    return path.expanduser()


def normalize_source_label(label: Any) -> str:
    value = str(label).strip().upper()
    if value in LETTER_LABELS[:MAX_CHOICES]:
        return value
    if value.isdigit():
        index = int(value) - 1
        if 0 <= index < MAX_CHOICES:
            return LETTER_LABELS[index]
    raise ArcBuildError(f"Unsupported ARC choice label: {label!r}")


def _string_list(values: Any, *, field_name: str) -> list[str]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ArcBuildError(f"Expected {field_name} to be a sequence of strings")

    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text:
            raise ArcBuildError(f"Empty string found in {field_name}")
        result.append(text)
    return result


def parse_arc_row(row: dict[str, Any]) -> ArcQuestion:
    question = str(row.get("question", "")).strip()
    if not question:
        raise ArcBuildError("ARC row is missing a non-empty question")

    choices_obj = row.get("choices")
    if not isinstance(choices_obj, dict):
        raise ArcBuildError("ARC row is missing a valid choices object")

    labels = _string_list(choices_obj.get("label"), field_name="choices.label")
    texts = _string_list(choices_obj.get("text"), field_name="choices.text")
    if len(labels) != len(texts):
        raise ArcBuildError("ARC row has mismatched choice labels/text lengths")
    if len(labels) < 2 or len(labels) > MAX_CHOICES:
        raise ArcBuildError(
            f"ARC row has unsupported number of choices: {len(labels)}"
        )

    normalized_labels = [normalize_source_label(label) for label in labels]
    if len(set(normalized_labels)) != len(normalized_labels):
        raise ArcBuildError("ARC row contains duplicate choice labels")

    normalized_answer = normalize_source_label(row.get("answerKey"))
    if normalized_answer not in normalized_labels:
        raise ArcBuildError("ARC row answerKey does not match any choice label")

    answer_index = normalized_labels.index(normalized_answer)
    answer_letter = LETTER_LABELS[answer_index]
    return ArcQuestion(
        question=question,
        choices=tuple(texts),
        answer_letter=answer_letter,
    )


def format_options(choices: Sequence[str]) -> str:
    return "\n".join(
        f"{LETTER_LABELS[index]}. {choice}" for index, choice in enumerate(choices)
    )


def format_question_block(question: ArcQuestion, *, answer: str | None = None) -> str:
    answer_suffix = f" {answer}" if answer is not None else ""
    return (
        f"Question:\n{question.question}\n"
        f"Choices:\n{format_options(question.choices)}\n"
        f"Answer:{answer_suffix}"
    )


def select_fewshot_examples(
    train_rows: Iterable[dict[str, Any]],
    *,
    fewshot_count: int,
    fewshot_seed: int,
) -> list[ArcQuestion]:
    if fewshot_count == 0:
        return []

    train_questions = [parse_arc_row(row) for row in train_rows]
    if fewshot_count > len(train_questions):
        raise ArcBuildError(
            f"Requested {fewshot_count} few-shot exemplars, but only "
            f"{len(train_questions)} train rows are available"
        )

    rng = random.Random(fewshot_seed)
    indices = rng.sample(range(len(train_questions)), fewshot_count)
    return [train_questions[index] for index in indices]


def build_arc_record(
    row: dict[str, Any],
    *,
    row_index: int,
    fewshot_examples: Sequence[ArcQuestion] | None = None,
) -> dict[str, Any]:
    question = parse_arc_row(row)
    blocks = [
        format_question_block(example, answer=example.answer_letter)
        for example in (fewshot_examples or [])
    ]
    blocks.append(format_question_block(question))
    user_prompt = "\n\n".join(blocks)

    return {
        "row": row_index,
        "ground_truth": question.answer_letter,
        "completion_input": SYSTEM_PROMPT + user_prompt,
        "chat_input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        **choice_scoring_fields(len(question.choices)),
    }


def build_arc_dataset(
    test_rows: Iterable[dict[str, Any]],
    *,
    fewshot_examples: Sequence[ArcQuestion] | None = None,
) -> list[dict[str, Any]]:
    return [
        build_arc_record(row, row_index=index, fewshot_examples=fewshot_examples)
        for index, row in enumerate(test_rows)
    ]


def write_jsonl(records: Sequence[dict[str, Any]], output_jsonl: Path, overwrite: bool) -> None:
    output_jsonl = _expand_path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if output_jsonl.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists: {output_jsonl}. Pass --overwrite to replace it."
        )

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _default_dataset_name(fewshot_count: int) -> str:
    return "arc_challenge" if fewshot_count == 0 else f"arc_challenge_{fewshot_count}shot"


def _parse_meta(value: str | None, *, fewshot_count: int) -> dict[str, Any]:
    default_meta = {
        "split": TEST_SPLIT,
        "priority": "high",
        "fewshot": fewshot_count,
        "subset": SUBSET,
    }
    if value is None:
        return default_meta

    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--meta-json must decode to a JSON object")

    merged = dict(default_meta)
    merged.update(parsed)
    return merged


def write_dataset_config(
    output_path: Path,
    *,
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
        "uuid": dataset_uuid or dataset_name or str(uuid.uuid4()),
        "grader": {"type": "multiple_choice"},
        "average_over": average_over,
        "pass_at": pass_at,
        "dataset_name": dataset_name,
        "data_path": data_path,
        "semantic_version": semantic_version,
        "num_generations": num_generations,
        "meta": meta,
    }

    output_path = _expand_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _load_arc_split(*, split: str, revision: str | None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in real usage only
        raise ModuleNotFoundError(
            "The 'datasets' package is required to build ARC-Challenge datasets. "
            "Install it with: pip install -r data_layer/requirements.txt"
        ) from exc

    dataset = load_dataset(DATASET_ID, SUBSET, split=split, revision=revision)
    return [dict(row) for row in dataset]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical Eval360 ARC-Challenge JSONL datasets"
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Path to write the Eval360 JSONL dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing JSONL output",
    )
    parser.add_argument(
        "--config-output",
        type=Path,
        default=None,
        help="Optional dataset YAML output path",
    )
    parser.add_argument(
        "--config-data-path",
        default=None,
        help="data_path value to write in YAML (defaults to absolute output JSONL path)",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset name for generated YAML",
    )
    parser.add_argument(
        "--dataset-uuid",
        default=None,
        help="UUID for generated YAML",
    )
    parser.add_argument(
        "--semantic-version",
        default="1.0.0",
        help="Semantic version for generated YAML",
    )
    parser.add_argument(
        "--average-over",
        nargs="+",
        type=int,
        default=[1],
        help="average_over values for generated YAML",
    )
    parser.add_argument(
        "--pass-at",
        nargs="+",
        type=int,
        default=[1],
        help="pass_at values for generated YAML",
    )
    parser.add_argument(
        "--meta-json",
        default=None,
        help="Optional JSON object to merge into the default metadata",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional source revision for the Hugging Face dataset",
    )
    parser.add_argument(
        "--fewshot-count",
        type=int,
        choices=(0, 25),
        default=0,
        help="How many train exemplars to prepend to each test prompt",
    )
    parser.add_argument(
        "--fewshot-seed",
        type=int,
        default=0,
        help="Seed used to sample deterministic train exemplars",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_jsonl = _expand_path(args.output_jsonl)
    config_output = _expand_path(args.config_output) if args.config_output is not None else None
    test_rows = _load_arc_split(split=TEST_SPLIT, revision=args.revision)
    fewshot_examples: list[ArcQuestion] = []
    if args.fewshot_count > 0:
        fewshot_examples = select_fewshot_examples(
            _load_arc_split(split=TRAIN_SPLIT, revision=args.revision),
            fewshot_count=args.fewshot_count,
            fewshot_seed=args.fewshot_seed,
        )
    records = build_arc_dataset(test_rows, fewshot_examples=fewshot_examples)
    write_jsonl(records=records, output_jsonl=output_jsonl, overwrite=args.overwrite)

    if config_output is not None:
        dataset_name = args.dataset_name or _default_dataset_name(args.fewshot_count)
        config_data_path = args.config_data_path or str(output_jsonl.resolve())
        write_dataset_config(
            output_path=config_output,
            dataset_name=dataset_name,
            data_path=config_data_path,
            num_generations=len(records),
            semantic_version=args.semantic_version,
            average_over=args.average_over,
            pass_at=args.pass_at,
            meta=_parse_meta(args.meta_json, fewshot_count=args.fewshot_count),
            dataset_uuid=args.dataset_uuid or dataset_name,
        )


if __name__ == "__main__":
    main()
