from __future__ import annotations

import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "eval_datasets"
DATASET_ID = "TIGER-Lab/MMLU-Pro"
LABELS = "ABCDEFGHIJ"

PREFIX = (
    "You are a helpful assistant. To answer the user's question, you first think about the reasoning process and then provide the user with the answer. "
    "The reasoning process and answer are enclosed within <reasoning> </reasoning> and <answer> </answer> tags, respectively, i.e., <reasoning> reasoning process here </reasoning><answer> answer here </answer>. "
    "Provide a single letter (A, B, C, ...) as the answer, for example, <answer> C </answer>. Now the user asks you to answer a multiple choice question.\n\nUser: {quiz}\nAssistant:\n<reasoning>"
)


def format_options(doc: dict) -> str:
    return "\n".join(
        f"{LABELS[i]}. {text}" for i, text in enumerate(doc["options"])
    )


def build_record(doc: dict, row_id: int) -> dict:
    category = str(doc["category"]).replace("_", " ")
    question_block = (
        f"The following are multiple choice questions about {category}.\n\n"
        f"{str(doc['question']).strip()}\n{format_options(doc)}"
    )

    completion_input = PREFIX.format(quiz=question_block)
    chat_input = [
        {
            "role": "user",
            "content": (
                f"{question_block}\nProvide a single letter (A, B, C, ...) as the "
                "answer enclosed within the <answer> </answer> tags. For example, "
                "<answer> C </answer>."
            ),
        },
    ]

    return {
        "row": row_id,
        "completion_input": completion_input,
        "chat_input": chat_input,
        "ground_truth": doc["answer"],
    }


def write_records(ds, full_path: Path, small_path: Path, small_limit: int = 100) -> int:
    row_count = 0
    with full_path.open("w", encoding="utf-8") as f_full, small_path.open(
        "w", encoding="utf-8"
    ) as f_small:
        for row_id, doc in enumerate(ds):
            line = json.dumps(build_record(doc, row_id), ensure_ascii=False)
            f_full.write(line + "\n")
            if row_id < small_limit:
                f_small.write(line + "\n")
            row_count = row_id + 1
    return row_count


def main() -> None:
    from datasets import load_dataset

    full_path = OUTPUT_DIR / "mmlu_pro_reasoning.jsonl"
    small_path = OUTPUT_DIR / "mmlu_pro_reasoning_small.jsonl"
    ds = load_dataset(DATASET_ID, split="test")
    row_count = write_records(ds, full_path, small_path)

    print(f"Wrote {row_count} rows to {full_path}")
    print(f"Wrote {min(row_count, 100)} rows to {small_path}")


if __name__ == "__main__":
    main()
