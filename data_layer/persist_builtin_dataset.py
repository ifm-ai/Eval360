#!/usr/bin/env python3
"""Persist lm-eval built-in tasks as Eval360-ready datasets."""

from __future__ import annotations

import argparse
import json
import operator
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Any, List, NamedTuple

from lm_eval.tasks import TaskManager, get_task_dict  # type: ignore
from lm_eval.evaluator_utils import get_task_list  # type: ignore

from scheduler.choice_scoring_schema import CHOICE_SCORING_MODE


class JsonChatStr(NamedTuple):
    prompt: str

    def encode(self, encoding):
        return self.prompt.encode(encoding)


try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable, **_kwargs):
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persist lm-eval tasks to Eval360 JSONL datasets.",
    )
    parser.add_argument(
        "--tasks",
        required=True,
        help="Comma separated list of lm-eval task names to persist.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for JSONL dataset files.",
    )
    parser.add_argument(
        "--include-path",
        default=None,
        help="Optional path with additional lm-eval task definitions.",
    )
    parser.add_argument(
        "--verbosity",
        default="INFO",
        help="Logging level passed to TaskManager.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Print the first N records for inspection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing JSONL files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to process per task (defaults to no limit).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=None,
        help="Explicit document indices to sample (defaults to all documents).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank of the current worker when sharding build requests.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Total number of workers when sharding build requests.",
    )
    parser.add_argument(
        "--cache-requests",
        action="store_true",
        default=False,
        help="Cache built requests to disk so subsequent runs can reuse them.",
    )
    parser.add_argument(
        "--rewrite-requests-cache",
        action="store_true",
        default=False,
        help="Rewrite cached requests even if they already exist.",
    )
    parser.add_argument(
        "--system-instruction",
        default=None,
        help="Optional system instruction string applied to chat-style prompts.",
    )
    parser.add_argument(
        "--fewshot-as-multiturn",
        action="store_true",
        default=False,
        help="Format few-shot examples as multi-turn chat messages.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="",
        help="Tokenizer identifier passed to lm-eval when building requests.",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="Override the task's configured number of fewshot examples.",
    )
    return parser.parse_args()


def ensure_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {path}. Pass --overwrite to allow."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_doc_to_choice(task, doc):
    if not hasattr(task, "doc_to_choice"):
        return None
    if getattr(task.config, "output_type", None) != "multiple_choice":
        return None
    try:
        return task.doc_to_choice(doc)
    except (NotImplementedError, TypeError):
        return None


def resolve_choice_target(
    target: Any,
    choices: List[Any],
    *,
    context: Optional[str] = None,
) -> str:
    choice_texts = [str(choice) for choice in choices]

    def fail() -> None:
        details = f" target={target!r}, choices={choice_texts!r}"
        if context:
            details = f" {context};" + details
        raise ValueError("Unable to resolve multiple-choice target:" + details)

    if isinstance(target, str):
        stripped = target.strip()
        if stripped in choice_texts:
            return stripped
        if stripped and stripped.lstrip("+-").isdigit():
            index = int(stripped)
            if 0 <= index < len(choice_texts):
                return choice_texts[index]
        fail()

    if not isinstance(target, bool):
        try:
            index = operator.index(target)
        except TypeError:
            index = None
        if index is not None:
            if 0 <= index < len(choice_texts):
                return choice_texts[index]
            fail()

    for choice, choice_text in zip(choices, choice_texts):
        if target == choice:
            return choice_text

    fail()


def group_instances_by_doc(instances: List) -> Dict[int, List]:
    grouped: Dict[int, List] = defaultdict(list)
    for inst in instances:
        grouped[inst.doc_id].append(inst)
    return grouped


def parse_chat_messages(instance) -> Optional[List[Dict[str, str]]]:
    prompt = instance.args[0] if instance.args else None
    if isinstance(prompt, JsonChatStr):
        try:
            return json.loads(prompt.prompt)
        except json.JSONDecodeError:
            return None
    if isinstance(prompt, str):
        try:
            return json.loads(prompt)
        except json.JSONDecodeError:
            return None
    return None


def build_task_requests(
    task,
    build_kwargs: Dict[str, Any],
    *,
    system_instruction: Optional[str],
    apply_chat_template: bool,
    chat_template_callable: Optional[Any],
    tokenizer_name: str,
) -> List:
    task.build_all_requests(
        limit=build_kwargs.get("limit"),
        samples=build_kwargs.get("samples"),
        rank=build_kwargs.get("rank", 0),
        world_size=build_kwargs.get("world_size", 1),
        cache_requests=build_kwargs.get("cache_requests", False),
        rewrite_requests_cache=build_kwargs.get("rewrite_requests_cache", False),
        system_instruction=system_instruction,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=build_kwargs.get("fewshot_as_multiturn", False),
        chat_template=chat_template_callable,
        tokenizer_name=tokenizer_name if apply_chat_template else "",
    )
    return list(task.instances)


def instances_to_record(
    task,
    doc_id: int,
    completion_instances: List,
    system_instruction: Optional[str],
    chat_messages: Optional[List[Dict[str, str]]],
) -> dict:
    doc = completion_instances[0].doc
    target = task.doc_to_target(doc)
    record: Dict[str, Any] = {
        "row": doc_id,
        "ground_truth": target,
    }

    choices = safe_doc_to_choice(task, doc)
    if choices is not None:
        record["ground_truth"] = resolve_choice_target(
            target,
            choices,
            context=f"doc_id={doc_id}",
        )
        record["scoring_mode"] = CHOICE_SCORING_MODE
        record["scoring_completions"] = [str(choice) for choice in choices]
        record["scoring_completion_labels"] = [str(choice) for choice in choices]

    primary_ctx = ""
    if completion_instances:
        args = completion_instances[0].args
        if len(args) > 0 and args[0] is not None:
            primary_ctx = args[0]

    if primary_ctx is None:
        primary_ctx = ""

    if not isinstance(primary_ctx, str):
        primary_ctx = str(primary_ctx)

    record["completion_input"] = primary_ctx
    if chat_messages is None:
        chat_messages = []
        if system_instruction:
            chat_messages.append({"role": "system", "content": system_instruction})
        if primary_ctx:
            chat_messages.append({"role": "user", "content": primary_ctx})
    record["chat_input"] = chat_messages

    return record


def persist_task_dataset(
    task_output,
    output_dir: Path,
    overwrite: bool,
    preview: int,
    build_kwargs: Dict[str, Any],
) -> int:
    task = task_output.task
    if task is None:
        return preview

    dataset_name = task_output.task_name.replace("/", "_")
    output_path = output_dir / f"{dataset_name}.jsonl"
    ensure_can_write(output_path, overwrite)

    num_fewshot_override = build_kwargs.get("num_fewshot")
    if num_fewshot_override is not None:
        task.set_config("num_fewshot", num_fewshot_override)
        task.config.metadata = task.config.metadata or {}
        task.config.metadata["num_fewshot"] = num_fewshot_override

    completion_instances = build_task_requests(
        task,
        build_kwargs,
        system_instruction=build_kwargs.get("system_instruction"),
        apply_chat_template=False,
        chat_template_callable=None,
        tokenizer_name="",
    )
    completion_by_doc = group_instances_by_doc(completion_instances)

    def render_chat_history(chat_history, add_generation_prompt=True):
        return JsonChatStr(
            json.dumps(
                chat_history,
                ensure_ascii=False,
            )
        )

    chat_instances = build_task_requests(
        task,
        build_kwargs,
        system_instruction=build_kwargs.get("system_instruction"),
        apply_chat_template=True,
        chat_template_callable=render_chat_history,
        tokenizer_name=build_kwargs.get("tokenizer_name", ""),
    )
    chat_by_doc = group_instances_by_doc(chat_instances)

    chat_messages_map: Dict[int, List[Dict[str, str]]] = {}
    for doc_id, instances in chat_by_doc.items():
        if instances:
            messages = parse_chat_messages(instances[0])
            if messages is not None:
                chat_messages_map[doc_id] = messages

    with output_path.open("w", encoding="utf-8") as handle:
        for doc_id in tqdm(sorted(completion_by_doc.keys()), desc=dataset_name):
            record = instances_to_record(
                task,
                doc_id,
                completion_by_doc[doc_id],
                build_kwargs.get("system_instruction"),
                chat_messages_map.get(doc_id),
            )
            if preview > 0:
                print(record)
                preview -= 1
            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")

    return preview


def main(args: argparse.Namespace) -> None:
    task_names = [name.strip() for name in args.tasks.split(",") if name.strip()]
    if not task_names:
        raise ValueError("No valid task names provided.")

    task_manager = TaskManager(args.verbosity, include_path=args.include_path)
    task_dict = get_task_dict(task_names, task_manager)

    remaining_preview = args.preview
    build_kwargs = {
        "limit": args.limit,
        "samples": args.samples,
        "rank": args.rank,
        "world_size": args.world_size,
        "cache_requests": args.cache_requests,
        "rewrite_requests_cache": args.rewrite_requests_cache,
        "system_instruction": args.system_instruction,
        "fewshot_as_multiturn": args.fewshot_as_multiturn,
        "tokenizer_name": args.tokenizer_name,
        "num_fewshot": args.num_fewshot,
    }
    for task_output in get_task_list(task_dict):
        remaining_preview = persist_task_dataset(
            task_output,
            args.output_dir,
            args.overwrite,
            remaining_preview,
            build_kwargs,
        )


if __name__ == "__main__":
    main(parse_args())
