#!/usr/bin/env python3
"""Convert a HuggingFace MMLU-Redux 2.0 dataset (Arrow format) to Eval360 JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset, load_from_disk

from scheduler.choice_scoring_schema import choice_scoring_fields

ANSWER_LETTERS = ["A", "B", "C", "D"]
ANSWER_INSTRUCTION = "Please respond with the correct letter (A, B, C or D) without any additional comments, only the correct letter:"

SUBJECT_DISPLAY = {
    slug: slug.replace("_", " ").title()
    for slug in [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology",
        "high_school_statistics", "high_school_us_history",
        "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies",
        "machine_learning", "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
        "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology",
        "us_foreign_policy", "virology", "world_religions",
    ]
}


def load_fewshot_examples(subjects: list[str], num_fewshot: int) -> dict[str, list[dict]]:
    """Load few-shot examples from the MMLU dev split on HuggingFace."""
    if num_fewshot <= 0:
        return {}
    fewshot = {}
    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split="dev")
        fewshot[subject] = [ds[i] for i in range(min(num_fewshot, len(ds)))]
    return fewshot


def format_fewshot_example(question: str, choices: list[str], answer_idx: int) -> str:
    """Format a single few-shot example as text: question + choices + answer letter."""
    lines = [question]
    for i, choice in enumerate(choices):
        lines.append(f"{ANSWER_LETTERS[i]}. {choice}")
    lines.append(f"{ANSWER_INSTRUCTION} {ANSWER_LETTERS[answer_idx]}")
    return "\n".join(lines)


def format_prompt(subject_display: str, question: str, choices: list[str],
                  fewshot_examples: list[dict] | None = None) -> str:
    """Build the MMLU-style text prompt with optional few-shot examples."""
    lines = [f"The following are multiple choice questions (with answers) about {subject_display.lower()}.", ""]
    if fewshot_examples:
        for ex in fewshot_examples:
            lines.append(format_fewshot_example(ex["question"], ex["choices"], ex["answer"]))
            lines.append("")
    lines.append(question)
    for i, choice in enumerate(choices):
        lines.append(f"{ANSWER_LETTERS[i]}. {choice}")
    lines.append(ANSWER_INSTRUCTION)
    return "\n".join(lines)


def convert_row(subject: str, row_index: int, row: dict, use_corrected: bool,
                fewshot_examples: list[dict] | None = None) -> dict:
    """Convert a single HuggingFace row to Eval360 JSONL format."""
    display = SUBJECT_DISPLAY.get(subject, subject.replace("_", " ").title())
    question = row["question"]
    choices = row["choices"]

    prompt_body = question + "\n" + "\n".join(
        f"{ANSWER_LETTERS[i]}. {c}" for i, c in enumerate(choices)
    ) + "\n" + ANSWER_INSTRUCTION

    completion_input = format_prompt(display, question, choices, fewshot_examples)

    # Build chat_input with few-shot examples as multi-turn messages
    system_msg = f"The following are multiple choice questions (with answers) about {display.lower()}."
    chat_input = [{"role": "system", "content": system_msg}]
    if fewshot_examples:
        for ex in fewshot_examples:
            ex_body = ex["question"] + "\n" + "\n".join(
                f"{ANSWER_LETTERS[i]}. {c}" for i, c in enumerate(ex["choices"])
            ) + "\n" + ANSWER_INSTRUCTION
            chat_input.append({"role": "user", "content": ex_body})
            chat_input.append({"role": "assistant", "content": ANSWER_LETTERS[ex["answer"]]})
    chat_input.append({"role": "user", "content": prompt_body})

    if use_corrected and row.get("correct_answer") is not None:
        raw = row["correct_answer"].strip()
        if raw in ANSWER_LETTERS:
            ground_truth = raw
        elif raw.isdigit() and int(raw) < len(ANSWER_LETTERS):
            ground_truth = ANSWER_LETTERS[int(raw)]
        else:
            ground_truth = ANSWER_LETTERS[row["answer"]]
    else:
        ground_truth = ANSWER_LETTERS[row["answer"]]

    return {
        "row": row_index,
        "completion_input": completion_input,
        "chat_input": chat_input,
        "ground_truth": ground_truth,
        **choice_scoring_fields(len(choices)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MMLU-Redux 2.0 Arrow dataset to Eval360 JSONL.")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Path to the downloaded MMLU-Redux 2.0 dataset directory (with Arrow files).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Destination directory for JSONL files.")
    parser.add_argument("--combined", action="store_true",
                        help="Write a single combined JSONL file instead of per-subject files.")
    parser.add_argument("--use-corrected", action="store_true",
                        help="Use the 'correct_answer' field (Redux corrections) instead of the original 'answer'.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing JSONL files.")
    parser.add_argument("--num-fewshot", type=int, default=0,
                        help="Number of few-shot examples to prepend from the MMLU dev split (default: 0).")
    parser.add_argument("--preview", type=int, default=0,
                        help="Print the first N records for inspection.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_from_disk(str(args.input_dir))
    subjects = sorted(ds.keys())

    fewshot_map = load_fewshot_examples(subjects, args.num_fewshot)
    if fewshot_map:
        print(f"Loaded {args.num_fewshot}-shot examples for {len(fewshot_map)} subjects from cais/mmlu dev split")

    global_row = 0
    preview_remaining = args.preview

    if args.combined:
        combined_path = args.output_dir / "mmlu_redux.jsonl"
        if combined_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite: {combined_path}. Pass --overwrite.")

    per_subject_handles = {}

    try:
        combined_handle = None
        if args.combined:
            combined_handle = open(combined_path, "w", encoding="utf-8")

        for subject in subjects:
            split = ds[subject]
            if not args.combined:
                out_path = args.output_dir / f"{subject}.jsonl"
                if out_path.exists() and not args.overwrite:
                    raise FileExistsError(f"Refusing to overwrite: {out_path}. Pass --overwrite.")
                per_subject_handles[subject] = open(out_path, "w", encoding="utf-8")

            for local_idx in range(len(split)):
                row = split[local_idx]
                record = convert_row(subject, global_row, row, args.use_corrected,
                                     fewshot_examples=fewshot_map.get(subject))

                line = json.dumps(record, ensure_ascii=False) + "\n"

                if args.combined:
                    combined_handle.write(line)
                else:
                    per_subject_handles[subject].write(line)

                if preview_remaining > 0:
                    print(json.dumps(record, indent=2, ensure_ascii=False))
                    preview_remaining -= 1

                global_row += 1

            if not args.combined:
                per_subject_handles[subject].close()
                print(f"  {subject}: {len(split)} rows -> {args.output_dir / f'{subject}.jsonl'}")

    finally:
        if combined_handle:
            combined_handle.close()
        for h in per_subject_handles.values():
            if not h.closed:
                h.close()

    print(f"\nTotal: {global_row} rows across {len(subjects)} subjects")
    if args.combined:
        print(f"Output: {combined_path}")
    else:
        print(f"Output: {args.output_dir}/ ({len(subjects)} files)")


if __name__ == "__main__":
    main()
