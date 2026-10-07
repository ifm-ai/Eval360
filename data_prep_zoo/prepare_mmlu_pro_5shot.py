from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from scheduler.choice_scoring_schema import (
    choice_scoring_fields,
    choice_scoring_labels,
    validate_choice_scoring_row,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "eval_datasets"
DATASET_ID = "TIGER-Lab/MMLU-Pro"


def build_exemplars_by_category(val_ds) -> dict[str, list[dict]]:
    exemplars_by_category: dict[str, list[dict]] = defaultdict(list)
    for doc in val_ds:
        exemplars_by_category[doc["category"]].append(doc)
    return exemplars_by_category


def format_options(doc: dict) -> str:
    labels = choice_scoring_labels(len(doc["options"]))
    return "\n".join(
        f"{label}. {text}" for label, text in zip(labels, doc["options"])
    )


def format_generation_instruction(doc: dict) -> str:
    category = str(doc["category"]).replace("_", " ")
    return (
        f"The following are multiple choice questions (with answers) about {category}. "
        f'Think step by step and then finish your answer with "the answer is (X)" '
        f"where X is the correct letter choice."
    )


def format_shot(doc: dict) -> str:
    cot = doc["cot_content"]
    # Strip leading "A: " prefix from cot_content
    if cot.startswith("A: "):
        cot = cot[3:]
    return (
        f"Question:\n{doc['question'].strip()}\nOptions:\n{format_options(doc)}\n"
        f"Answer: {cot}"
    )


def format_test_question(doc: dict) -> str:
    return (
        f"Question:\n{doc['question'].strip()}\nOptions:\n{format_options(doc)}\n"
        f"Answer: Let's think step by step."
    )


def format_scoring_shot(doc: dict) -> str:
    return (
        f"Question:\n{str(doc['question']).strip()}\nOptions:\n{format_options(doc)}\n"
        f"Answer: {doc['answer']}"
    )


def format_scoring_test_question(doc: dict) -> str:
    return (
        f"Question:\n{str(doc['question']).strip()}\nOptions:\n{format_options(doc)}\n"
        "Answer:"
    )


def format_scoring_prompt_prefix(doc: dict, shots: list[dict]) -> str:
    category = str(doc["category"]).replace("_", " ")
    instruction = (
        f"The following are multiple choice questions (with answers) about {category}. "
        "Answer with only the correct letter."
    )
    prompt_body = "\n\n".join(
        [format_scoring_shot(shot) for shot in shots]
        + [format_scoring_test_question(doc)]
    )
    return instruction + "\n" + prompt_body


def build_record(doc: dict, row_id: int, shots: list[dict]) -> dict:
    instruction = format_generation_instruction(doc)
    shot_blocks = [format_shot(shot) for shot in shots]
    prompt_body = "\n\n".join(shot_blocks + [format_test_question(doc)])

    record = {
        "row": row_id,
        "completion_input": instruction + "\n" + prompt_body,
        "ground_truth": doc["answer"],
        "scoring_prompt_prefix": format_scoring_prompt_prefix(doc, shots),
        **choice_scoring_fields(len(doc["options"])),
    }
    validate_choice_scoring_row(record, phase="build")
    return record


def write_records(
    test_ds,
    exemplars_by_category: dict[str, list[dict]],
    full_path: Path,
    small_path: Path,
    small_limit: int = 100,
) -> int:
    row_count = 0
    with full_path.open("w", encoding="utf-8") as f_full, small_path.open(
        "w", encoding="utf-8"
    ) as f_small:
        for row_id, doc in enumerate(test_ds):
            shots = exemplars_by_category[doc["category"]]
            record = build_record(doc, row_id, shots)
            line = json.dumps(record, ensure_ascii=False)
            f_full.write(line + "\n")
            if row_id < small_limit:
                f_small.write(line + "\n")
            row_count = row_id + 1
    return row_count


def main() -> None:
    from datasets import load_dataset

    full_path = OUTPUT_DIR / "mmlu_pro_5shot.jsonl"
    small_path = OUTPUT_DIR / "mmlu_pro_5shot_small.jsonl"
    ds = load_dataset(DATASET_ID)
    row_count = write_records(
        ds["test"],
        build_exemplars_by_category(ds["validation"]),
        full_path,
        small_path,
    )

    print(f"Wrote {row_count} rows to {full_path}")
    print(f"Wrote {min(row_count, 100)} rows to {small_path}")


if __name__ == "__main__":
    main()
