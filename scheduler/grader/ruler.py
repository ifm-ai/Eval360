import asyncio
import copy
import re
from collections import defaultdict
from typing import Any, Iterable

from ..metrics import get_accuracy, get_bootstrap_accuracy_std, mean_pass_at_k
from ..utils import Sentinel, ExceptionWrapper
from .base import AccuracyGraderBase, Grade, Score
from .registry import register


_NIAH_SINGLE = {
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
}
_NIAH_MULTI = {
    "niah_multivalue",
    "niah_multiquery",
}
_QA_TASKS = {
    "ruler_qa_squad",
    "ruler_qa_hotpot",
}
_SUPPORTED_TASKS = (
    _NIAH_SINGLE
    | _NIAH_MULTI
    | _QA_TASKS
    | {"ruler_fwe", "ruler_cwe", "ruler_vt"}
)

# NVIDIA RULER per-subtask generation-token budgets. Official RULER scores
# only the first N output tokens of each subtask; matching on the full
# generation lets stray text (other needles, reasoning) produce false
# positives and inflate the score. Truncation is applied only when the task
# YAML sets `meta.truncate_tokenizer` (a HF tokenizer path); otherwise the
# full generation is graded, exactly as before.
# Per-subtask truncation budgets. Tasks NOT in this dict are graded without
# truncation (the QA subtasks ruler_qa_squad/hotpot fall in that bucket --
# their answers are free-form spans and capping the model's generation can
# cut off the correct phrase).
_MAX_GEN_TOKS = {
    "niah_single_1": 128, "niah_single_2": 128, "niah_single_3": 128,
    "niah_multikey_1": 128, "niah_multikey_2": 128, "niah_multikey_3": 128,
    "niah_multiquery": 128, "niah_multivalue": 128,
    "ruler_cwe": 120, "ruler_fwe": 50, "ruler_vt": 30,
}


def _truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    if not text:
        return text
    ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _match_single_value(haystack: str, expected: str) -> bool:
    pattern = rf"(?<!\w){re.escape(expected.lower())}(?!\w)"
    return re.search(pattern, haystack.lower()) is not None


def check_ruler(task: str, generations: list[str], expected: list[str]):
    expected = list(expected)
    correct = []
    for generation in generations:
        generation = generation or ""
        if task in _NIAH_SINGLE:
            correct.append(_match_single_value(generation, expected[0]))
        elif task in _NIAH_MULTI:
            hay = generation.lower()
            correct.extend(value.lower() in hay for value in expected)
        elif task in _QA_TASKS:
            hay = generation.lower()
            correct.append(any(value.lower() in hay for value in expected))
        elif task == "ruler_fwe":
            hay = generation.lower()
            correct.extend(value.lower() in hay for value in expected)
        elif task == "ruler_cwe":
            hay = generation.lower()
            correct.extend(value.lower() in hay for value in expected)
        elif task == "ruler_vt":
            hay = generation.upper()
            correct.extend(value.upper() in hay for value in expected)
        else:
            raise ValueError(f"Unknown ruler_task '{task}'")

    return {
        "correct": correct,
        "accuracy": get_accuracy(correct) if correct else float("nan"),
        "bootstrap_std": get_bootstrap_accuracy_std(correct) if len(correct) > 1 else 0.0,
    }


@register("ruler")
class RulerGrader(AccuracyGraderBase):
    _tokenizer = None

    async def initialize(self):
        """Optionally load the tokenizer used for NVIDIA-style truncation.

        Enabled by setting `meta.truncate_tokenizer` (a HF tokenizer path) in
        the ruler task YAML. Absent -> no truncation, full generation graded.
        """
        self._tokenizer = None
        meta = (getattr(self, "task", None) and self.task.meta) or {}
        tok_path = meta.get("truncate_tokenizer")
        if tok_path:
            from transformers import AutoTokenizer

            self._tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained, tok_path, trust_remote_code=True
            )

    async def grade_sample(self, sample: Any, *_):
        if sample == Sentinel.COMPLETED:
            return sample
        assert isinstance(sample, dict), "sample must be a dict"
        assert "completion_input" in sample, "sample must have a 'completion_input' key"
        assert "parsed_generations" in sample, "sample must have a 'parsed_generations' key"
        assert "ground_truth" in sample, "sample must have a 'ground_truth' key"
        assert "ruler_task" in sample, "sample must have a 'ruler_task' key"
        assert sample["ruler_task"] in _SUPPORTED_TASKS, (
            f"Unsupported ruler_task '{sample['ruler_task']}'. "
            f"Supported tasks: {sorted(_SUPPORTED_TASKS)}"
        )

        generations = []
        for index, generation in enumerate(sample["parsed_generations"]):
            if generation is None:
                generations.append(sample["generations"][index])
            else:
                generations.append(generation)

        # NVIDIA-style per-subtask truncation (only when a tokenizer is set).
        if self._tokenizer is not None:
            budget = _MAX_GEN_TOKS.get(sample["ruler_task"])
            if budget is not None:
                generations = [
                    _truncate_to_tokens(self._tokenizer, g, budget)
                    for g in generations
                ]

        new_fields = check_ruler(
            task=sample["ruler_task"],
            generations=generations,
            expected=list(sample["ground_truth"]) if isinstance(sample["ground_truth"], list) else [sample["ground_truth"]],
        )

        result = copy.deepcopy(sample)
        result.update(new_fields)
        return result

    async def run(self, existing, average_over, pass_at):
        """Aggregate per-row results AND emit per-subtask accuracy.

        Same accuracy / avg-over / pass-at logic as AccuracyGraderBase.run,
        plus one extra ``Score`` per ruler subtask so the resulting
        ``*_scores.yaml`` carries the per-subtask breakdown natively (no need
        to re-aggregate from grades.jsonl downstream).
        """
        if not pass_at:
            pass_at = [1]
        if not average_over:
            average_over = [1]
        all_correct: list = []
        per_subtask: dict = defaultdict(lambda: [0, 0])  # subtask -> [correct, total]
        completed = False
        graded_rows: set = set()

        def _track(result: dict) -> None:
            flags = result.get("correct") or []
            if not flags:
                return
            sub = result.get("ruler_task")
            if sub:
                s = per_subtask[sub]
                s[1] += len(flags)
                s[0] += sum(1 for x in flags if x)

        async for result in existing:
            if isinstance(result, dict):
                if "correct" in result:
                    all_correct.append(result["correct"])
                    _track(result)
                if "row" in result:
                    graded_rows.add(result["row"])
        async for result in self._grading_generator(skip_rows=graded_rows):
            if result == Sentinel.COMPLETED:
                completed = True
                break
            elif isinstance(result, ExceptionWrapper):
                yield result
            elif isinstance(result, dict) and "correct" in result:
                all_correct.append(result["correct"])
                _track(result)
                yield Grade(element=result)
        if not completed:
            return

        if not all_correct or not any(all_correct):
            for n in average_over:
                yield Score(name=f"accuracy (avg over {n})", value=float("nan"))
                yield Score(name=f"bootstrap_std (avg over {n})", value=float("nan"))
            for n in pass_at:
                yield Score(name=f"accuracy (pass@{n})", value=float("nan"))
                yield Score(name=f"bootstrap_std (pass@{n})", value=float("nan"))
        else:
            for n in average_over:
                yield Score(
                    name=f"accuracy (avg over {n})",
                    value=get_accuracy([s[:n] for s in all_correct]),
                )
                if n > 1:
                    yield Score(
                        name=f"bootstrap_std (avg over {n})",
                        value=get_bootstrap_accuracy_std([s[:n] for s in all_correct]),
                    )
            for n in pass_at:
                per = [(len(s), sum(s)) for s in all_correct]
                yield Score(name=f"accuracy (pass@{n})", value=mean_pass_at_k(per, k=n))

        # Per-subtask breakdown -- one Score per ruler subtask encountered.
        for sub in sorted(per_subtask):
            c, n = per_subtask[sub]
            yield Score(
                name=f"accuracy/{sub}",
                value=(c / n) if n else float("nan"),
            )
        yield Sentinel.COMPLETED
