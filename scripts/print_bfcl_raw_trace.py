#!/usr/bin/env python3
"""
Print BFCL raw inference logs as readable text.

Accepts a glob pattern (shell-expanded) of turn files, sample dirs, or a raw_inference root.

Usage:
    # Specific turn files (shell glob)
    python scripts/print_bfcl_raw_trace.py path/to/multi_turn_base_0/*

    # Specific sample dir
    python scripts/print_bfcl_raw_trace.py path/to/multi_turn/multi_turn_base_0

    # All samples in a category
    python scripts/print_bfcl_raw_trace.py path/to/raw_inference/multi_turn/*

    # List available categories and samples
    python scripts/print_bfcl_raw_trace.py path/to/raw_inference

    # Control truncation
    python scripts/print_bfcl_raw_trace.py path/to/multi_turn_base_0/* --max-len 2000
"""
import argparse
import json
import sys
from pathlib import Path


def fmt_content(content, max_len=None):
    if content is None:
        return "(none)"
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                t = c.get("type", "")
                if t == "text":
                    parts.append(c.get("text", ""))
                else:
                    parts.append(json.dumps(c))
            else:
                parts.append(str(c))
        text = "\n".join(parts)
    else:
        text = str(content)
    if max_len and len(text) > max_len:
        text = text[:max_len] + f"...[truncated, total {len(text)} chars]"
    return text


def fmt_tool_calls(tool_calls, indent="  "):
    lines = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            lines.append(f"{indent}- id: {tc.get('id', '')}")
            lines.append(f"{indent}  name: {fn.get('name', '')}")
            try:
                args = json.loads(fn.get("arguments", "{}"))
                lines.append(f"{indent}  args: {json.dumps(args)}")
            except Exception:
                lines.append(f"{indent}  args: {fn.get('arguments', '')}")
    return "\n".join(lines)


def print_step(step_num, entry, content_max_len=1000):
    messages = entry["messages"]
    response = entry["response"]
    tools = entry.get("tools") or []

    print(f"  --- Step {step_num} ---")
    if tools:
        print(f"    TOOLS ({len(tools)}):")
        for t in tools:
            for line in json.dumps(t, indent=2).splitlines():
                print(f"      {line}")
        print()
    print("    MESSAGES SENT:")
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content")
        tool_calls = m.get("tool_calls", [])
        tool_call_id = m.get("tool_call_id")
        name = m.get("name")

        if tool_call_id:
            print(f"      [{role}] tool_result id={tool_call_id}" + (f" name={name}" if name else ""))
            print(f"        {fmt_content(content)}")
        elif tool_calls:
            print(f"      [{role}] tool_calls:")
            print(fmt_tool_calls(tool_calls, indent="        "))
        else:
            print(f"      [{role}]")
            for line in fmt_content(content, max_len=content_max_len).splitlines():
                print(f"        {line}")

    print()
    print("    RESPONSE:")
    for ch in response.get("choices", []):
        msg = ch.get("message", {})
        role = msg.get("role", "?")
        content = msg.get("content") or ""
        tool_calls = msg.get("tool_calls", [])
        if content:
            print(f"      [{role}] content:")
            for line in fmt_content(content, max_len=content_max_len).splitlines():
                print(f"        {line}")
        if tool_calls:
            print(f"      [{role}] tool_calls:")
            print(fmt_tool_calls(tool_calls, indent="        "))
    print()


def print_turn_file(turn_file: Path, content_max_len=1000):
    turn_num = int(turn_file.stem.split("_")[1])
    print(f"=== TURN {turn_num} ===")
    with open(turn_file) as f:
        for step_num, line in enumerate(f):
            if line.strip():
                print_step(step_num, json.loads(line), content_max_len=content_max_len)


def print_sample_dir(sample_dir: Path, content_max_len=1000):
    turn_files = sorted(sample_dir.glob("turn_*.jsonl"), key=lambda p: int(p.stem.split("_")[1]))
    if not turn_files:
        print("  (no turn files found)")
        return
    for turn_file in turn_files:
        print_turn_file(turn_file, content_max_len=content_max_len)


def list_contents(raw_dir: Path):
    categories = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())
    if not categories:
        print("No categories found.")
        return
    print(f"Categories in {raw_dir}:")
    for cat in categories:
        samples = sorted((raw_dir / cat).iterdir())
        print(f"  {cat}/  ({len(samples)} samples)")
        for s in samples[:5]:
            turns = sorted(s.glob("turn_*.jsonl"))
            print(f"    {s.name}/  ({len(turns)} turns)")
        if len(samples) > 5:
            print(f"    ... and {len(samples) - 5} more")


def main():
    parser = argparse.ArgumentParser(description="Print BFCL raw inference logs as readable text.")
    parser.add_argument("paths", nargs="+", type=Path, help="Turn files, sample dirs, or raw_inference root")
    parser.add_argument("--max-len", type=int, default=None, help="Max content length before truncation (default: no truncation)")
    args = parser.parse_args()

    paths = args.paths

    # Single directory: could be raw_inference root, category dir, or sample dir
    if len(paths) == 1 and paths[0].is_dir():
        d = paths[0]
        children = list(d.iterdir())
        # Sample dir: contains turn_*.jsonl files
        if any(c.name.startswith("turn_") and c.suffix == ".jsonl" for c in children):
            print(f"SAMPLE: {d.name}")
            print_sample_dir(d, content_max_len=args.max_len)
        # Category dir: contains sample subdirs
        elif all(c.is_dir() for c in children if not c.name.startswith(".")):
            for sample_dir in sorted(children, key=lambda p: p.name):
                if sample_dir.is_dir():
                    print(f"\n{'=' * 60}")
                    print(f"SAMPLE: {sample_dir.name}")
                    print(f"{'=' * 60}")
                    print_sample_dir(sample_dir, content_max_len=args.max_len)
        else:
            list_contents(d)
        return

    # Multiple paths: sort turn files first, then sample dirs
    turn_files = sorted([p for p in paths if p.is_file() and p.suffix == ".jsonl"],
                        key=lambda p: int(p.stem.split("_")[1]))
    sample_dirs = sorted([p for p in paths if p.is_dir()], key=lambda p: p.name)

    if turn_files:
        for tf in turn_files:
            print_turn_file(tf, content_max_len=args.max_len)

    for sample_dir in sample_dirs:
        print(f"\n{'=' * 60}")
        print(f"SAMPLE: {sample_dir.name}")
        print(f"{'=' * 60}")
        print_sample_dir(sample_dir, content_max_len=args.max_len)


if __name__ == "__main__":
    main()
