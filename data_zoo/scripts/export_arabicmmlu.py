#!/usr/bin/env python3
"""Export ArabicMMLU from Hugging Face datasets into Eval360 JSONL records."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import DownloadConfig, DownloadMode, get_dataset_config_names, load_dataset

from scheduler.choice_scoring_schema import choice_scoring_fields


LABELS = ["A", "B", "C", "D", "E"]
OPTION_KEYS = ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"]
DEFAULT_DATASET_ID = "MBZUAI/ArabicMMLU"
MULTI_CONFIG_CACHE_ERROR = r"configurations in the cache:\s*(.*?)\nPlease specify"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ArabicMMLU Hugging Face dataset splits to Eval360 JSONL files.",
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=f"Hugging Face dataset id (default: {DEFAULT_DATASET_ID}).",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Optional list (or comma-separated values) of dataset config names to export.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to export (default: test).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local HF cache files (no network requests).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where exported JSONL files are written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing JSONL files in output-dir.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Print the first N records from each exported config.",
    )
    # Kept for backward compatibility with older invocations.
    parser.add_argument("--cache-root", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_config_values(raw_values: Optional[List[str]]) -> Optional[List[str]]:
    if raw_values is None:
        return None

    values: List[str] = []
    for value in raw_values:
        for item in value.split(","):
            normalized = item.strip()
            if normalized:
                values.append(normalized)
    return sorted(set(values)) if values else None


def sanitize_task_name(config_name: str) -> str:
    normalized = config_name.lower()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return f"arabicmmlu_{normalized}"


def clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text if text else None


def build_prompt(doc: Dict[str, Any], config_name: str) -> Tuple[str, List[str]]:
    question = clean_text(doc.get("Question")) or ""
    context = clean_text(doc.get("Context"))
    subject = clean_text(doc.get("Subject"))
    level = clean_text(doc.get("Level"))
    if subject and level:
        topic = f"{subject} ({level})"
    else:
        topic = subject or config_name

    choices: List[str] = []
    for option_key in OPTION_KEYS:
        option_text = clean_text(doc.get(option_key))
        if option_text is not None:
            choices.append(option_text)

    option_lines = [f"{label}. {choice}" for label, choice in zip(LABELS, choices)]
    header = f"The following are multiple choice questions (with answers) about {topic}."
    body_parts: List[str] = [question]
    if context:
        body_parts.insert(0, f"Context: {context}")

    body = "\n".join([part for part in body_parts if part])
    prompt = f"{header}\n\n{body}\n" + "\n".join(option_lines) + "\nAnswer:"
    return prompt, choices


def parse_cached_config_names_from_exception(message: str) -> List[str]:
    match = re.search(MULTI_CONFIG_CACHE_ERROR, message, flags=re.DOTALL)
    if not match:
        return []
    payload = match.group(1).strip()
    if not payload:
        return []
    return [value.strip() for value in payload.split(",") if value.strip()]


def discover_config_names(
    dataset_id: str,
    download_config: DownloadConfig,
) -> List[str]:
    config_names: List[str] = []
    try:
        config_names = list(
            get_dataset_config_names(
                dataset_id,
                download_config=download_config,
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
            )
        )
    except Exception:
        config_names = []

    # Some cached builders return only "default" even when per-subset configs exist.
    if config_names and not (len(config_names) == 1 and config_names[0] == "default"):
        return sorted(config_names)

    try:
        load_dataset(
            dataset_id,
            split="test",
            download_config=download_config,
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
        )
    except ValueError as exc:
        inferred = parse_cached_config_names_from_exception(str(exc))
        if inferred:
            return sorted(inferred)
    except Exception:
        pass

    if config_names:
        return sorted(config_names)

    raise RuntimeError(
        "Could not auto-discover ArabicMMLU configs. Pass explicit --configs values."
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(args.output_dir.glob("*.jsonl"))
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output dir already has JSONL files. Re-run with --overwrite: {args.output_dir}"
        )
    for existing_file in existing:
        existing_file.unlink()

    download_config = DownloadConfig(
        local_files_only=args.local_files_only,
        cache_dir=args.cache_root,
    )
    config_names = parse_config_values(args.configs)
    if config_names is None:
        config_names = discover_config_names(args.dataset_id, download_config)

    exported_tasks = 0
    total_rows = 0
    for config_name in config_names:
        dataset = load_dataset(
            args.dataset_id,
            name=config_name,
            split=args.split,
            download_config=download_config,
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
        )

        task_name = sanitize_task_name(config_name)
        output_path = args.output_dir / f"{task_name}.jsonl"

        with output_path.open("w", encoding="utf-8") as handle:
            for row_idx, doc in enumerate(dataset):
                prompt, choices = build_prompt(doc, config_name)
                answer_key = clean_text(doc.get("Answer Key"))
                if answer_key is None:
                    raise ValueError(f"{task_name} row {row_idx}: missing Answer Key")
                answer_key = answer_key.upper()
                if answer_key not in LABELS:
                    raise ValueError(
                        f"{task_name} row {row_idx}: invalid Answer Key {answer_key!r}"
                    )
                if LABELS.index(answer_key) >= len(choices):
                    raise ValueError(
                        f"{task_name} row {row_idx}: answer {answer_key} exceeds {len(choices)} choices"
                    )

                record = {
                    "row": row_idx,
                    "completion_input": prompt,
                    "chat_input": [{"role": "user", "content": prompt}],
                    "ground_truth": answer_key,
                    **choice_scoring_fields(len(choices)),
                }
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
                if args.preview > 0 and row_idx < args.preview:
                    print(json.dumps(record, ensure_ascii=False))

        exported_tasks += 1
        total_rows += len(dataset)
        print(f"Exported {task_name}: {len(dataset)} rows -> {output_path}")

    if exported_tasks == 0:
        raise RuntimeError("No configs exported. Check dataset id/split/config selections.")

    print(f"Completed export: {exported_tasks} tasks, {total_rows} total rows")


if __name__ == "__main__":
    main()
