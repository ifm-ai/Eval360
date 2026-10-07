#!/usr/bin/env python3
"""Utility script for generating Eval360 dataset configuration YAML files."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_json(input_value: str) -> Dict[str, Any]:
    """Parse a JSON object string into a dictionary."""
    try:
        parsed = json.loads(input_value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object (dictionary).")
    return parsed


def parse_int_list(values: Iterable[str]) -> List[int]:
    """Return the provided values as integers, stripping whitespace."""
    parsed: List[int] = []
    for value in values:
        value_stripped = value.strip()
        try:
            parsed.append(int(value_stripped))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Expected an integer, received '{value_stripped}'."
            ) from exc
    return parsed


def format_scalar(value: Any) -> str:
    """Format scalar values for YAML output using JSON for compatibility."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value)


def format_key(key: Any) -> str:
    """Format dictionary keys for YAML output."""
    if isinstance(key, str) and key and all(c.isalnum() or c in {"_", "-"} for c in key):
        return key
    return json.dumps(key)


def render_yaml(data: Any, indent: int = 0) -> str:
    """Render a Python data structure as a YAML string without external dependencies."""
    indent_str = "  " * indent

    if isinstance(data, dict):
        lines: List[str] = []
        for key, value in data.items():
            key_repr = format_key(key)
            if isinstance(value, (dict, list)):
                lines.append(f"{indent_str}{key_repr}:")
                lines.append(render_yaml(value, indent + 1))
            else:
                value_repr = format_scalar(value)
                lines.append(f"{indent_str}{key_repr}: {value_repr}")
        return "\n".join(lines)

    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{indent_str}-")
                lines.append(render_yaml(item, indent + 1))
            else:
                item_repr = format_scalar(item)
                lines.append(f"{indent_str}- {item_repr}")
        return "\n".join(lines)

    return f"{indent_str}{format_scalar(data)}"


def _jsonl_files_from_path(data_path: Path) -> List[Path]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")
    if data_path.is_file():
        return [data_path]
    if data_path.is_dir():
        candidates = sorted(p for p in data_path.rglob("*.jsonl") if p.is_file())
        if not candidates:
            raise ValueError(f"No JSONL files found under directory: {data_path}")
        return candidates
    raise ValueError(f"Unsupported dataset path type: {data_path}")


def analyze_dataset(data_path: Path) -> int:
    """Count dataset records."""
    jsonl_files = _jsonl_files_from_path(data_path)

    count = 0
    for jsonl_file in jsonl_files:
        with jsonl_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                count += 1
    return count


def build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """Construct the payload that will be serialized to YAML."""
    data_path = Path(args.data_path).expanduser().resolve()

    datapoint_count_inferred = analyze_dataset(data_path)

    datapoint_count = (
        args.datapoint_count if args.datapoint_count is not None else datapoint_count_inferred
    )

    configured_data_path = str(data_path)
    if data_path.is_dir():
        configured_data_path = f"{configured_data_path}/*.jsonl"

    payload: Dict[str, Any] = {
        "uuid": str(uuid.uuid4()),
        "grader": {"type": args.grader_type},
        "average_over": parse_int_list(args.avg_at),
        "pass_at": parse_int_list(args.pass_at),
        "dataset_name": args.dataset_name,
        "data_path": configured_data_path,
        "semantic_version": args.semantic_version,
        "num_generations": datapoint_count,
    }
    if args.meta:
        payload["meta"] = args.meta
    return payload


def write_yaml_file(output_path: Path, payload: Dict[str, Any]) -> None:
    """Write the payload to the specified output path in YAML format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_content = render_yaml(payload)
    output_path.write_text(f"{yaml_content}\n", encoding="ascii")


def parse_args() -> argparse.Namespace:
    """Configure and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a dataset configuration YAML file for Eval360 evaluations.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination path for the YAML output file.",
    )
    parser.add_argument(
        "--pass-at",
        nargs="+",
        required=True,
        help="One or more integer pass@ configuration values (e.g. 1 4 16).",
    )
    parser.add_argument(
        "--avg-at",
        nargs="+",
        required=True,
        help="One or more integer average-over configuration values (e.g. 1 2 16).",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset name to pass through to evaluation.",
    )
    parser.add_argument(
        "--grader-type",
        required=True,
        choices=("multiple_choice", "choice_scoring", "free_form"),
        help="Grader type to use for evaluation (e.g. multiple_choice, choice_scoring, free_form).",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Absolute path to a JSONL file or a directory containing JSONL files.",
    )
    parser.add_argument(
        "--semantic-version",
        required=True,
        help="Semantic version of the configuration.",
    )
    parser.add_argument(
        "--datapoint-count",
        type=int,
        default=None,
        help="Override the inferred number of generations (defaults to counting JSONL records).",
    )
    parser.add_argument(
        "--meta",
        type=parse_json,
        default=None,
        help="Optional JSON object with extra metadata (e.g. split, priority, fewshot).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(args)
    write_yaml_file(args.output, payload)


if __name__ == "__main__":
    main()
