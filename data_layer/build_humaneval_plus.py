"""
Convert the EvalPlus HumanEval+ dataset to Eval360-compatible JSONL.

Each record in the output has:
  row              — integer row index
  completion_input — the raw HumanEval prompt (function signature + docstring),
                     suitable for completion-style models
  chat_input       — chat transcript asking the model to complete the function
  ground_truth     — JSON string with "test" (test block), "entry_point"
                     (function name), and "prompt" (same as completion_input)

Usage:
    python data_layer/build_humaneval_plus.py \\
        --output-path data_zoo/humaneval_plus/humaneval_plus.jsonl
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

_SYSTEM_PROMPT = "You are an expert Python programmer."
_USER_TEMPLATE = (
    "Complete the implementation for the following Python function:\n\n"
    "```python\n{prompt}\n```\n\n"
    "Wrap your complete implementation in a python code block "
    "(i.e. ```python\nyour code here```)."
)


def main():
    parser = argparse.ArgumentParser(description="Build HumanEval+ JSONL for Eval360")
    parser.add_argument(
        "--output-path", required=True, help="Destination .jsonl file path"
    )
    parser.add_argument(
        "--split", default="test", help="Dataset split to use (default: test)"
    )
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("evalplus/humanevalplus", split=args.split)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row_id, doc in enumerate(dataset):
            prompt = doc["prompt"]
            ground_truth = json.dumps({
                "test": doc["test"],
                "entry_point": doc["entry_point"],
                "prompt": prompt,
            })
            record = {
                "row": row_id,
                "completion_input": prompt,
                "chat_input": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _USER_TEMPLATE.format(prompt=prompt)},
                ],
                "ground_truth": ground_truth,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} records to {output_path}")


if __name__ == "__main__":
    main()
