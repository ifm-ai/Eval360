from __future__ import annotations

import json
from pathlib import Path

from scheduler.choice_scoring_schema import (
    choice_scoring_fields,
    choice_scoring_labels,
    validate_choice_scoring_row,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "eval_datasets"
DATASET_ID = "TIGER-Lab/MMLU-Pro"


def format_category(doc: dict) -> str:
    return str(doc["category"]).replace("_", " ")


def format_options(doc: dict) -> str:
    labels = choice_scoring_labels(len(doc["options"]))
    return "\n".join(
        f"{label}. {text}" for label, text in zip(labels, doc["options"])
    )


def format_generation_instruction(doc: dict) -> str:
    category = format_category(doc)
    return (
        f"The following are multiple choice questions (with answers) about {category}. "
        f'Think step by step and then finish your answer with "the answer is (X)" '
        f"where X is the correct letter choice."
    )


def format_generation_prompt(doc: dict) -> str:
    return (
        f"Question:\n{str(doc['question']).strip()}\nOptions:\n{format_options(doc)}\n"
        "Answer: Let's think step by step."
    )


def format_scoring_prompt_prefix(doc: dict) -> str:
    category = format_category(doc)
    return (
        f"The following are multiple choice questions (with answers) about {category}. "
        "Answer with only the correct letter.\n"
        f"Question:\n{str(doc['question']).strip()}\nOptions:\n{format_options(doc)}\n"
        "Answer:"
    )


def build_record(doc: dict, row_id: int) -> dict:
    instruction = format_generation_instruction(doc)
    question_block = (
        f"Question:\n{str(doc['question']).strip()}\nOptions:\n{format_options(doc)}"
    )
    record = {
        "row": row_id,
        "completion_input": instruction + "\n" + format_generation_prompt(doc),
        "chat_input": [{"role": "user", "content": instruction + "\n" + question_block}],
        "ground_truth": doc["answer"],
        "scoring_prompt_prefix": format_scoring_prompt_prefix(doc),
        **choice_scoring_fields(len(doc["options"])),
    }
    validate_choice_scoring_row(record, phase="build")
    return record


def write_records(ds, full_path: Path, small_path: Path, small_limit: int = 100) -> int:
    row_count = 0
    with full_path.open("w", encoding="utf-8") as f_full, small_path.open(
        "w", encoding="utf-8"
    ) as f_small:
        for row_id, doc in enumerate(ds):
            record = build_record(doc, row_id)
            line = json.dumps(record, ensure_ascii=False)
            f_full.write(line + "\n")
            if row_id < small_limit:
                f_small.write(line + "\n")
            row_count = row_id + 1
    return row_count


def main() -> None:
    from datasets import load_dataset

    full_path = OUTPUT_DIR / "mmlu_pro.jsonl"
    small_path = OUTPUT_DIR / "mmlu_pro_small.jsonl"
    ds = load_dataset(DATASET_ID, split="test")
    row_count = write_records(ds, full_path, small_path)

    print(f"Wrote {row_count} rows to {full_path}")
    print(f"Wrote {min(row_count, 100)} rows to {small_path}")


if __name__ == "__main__":
    main()
