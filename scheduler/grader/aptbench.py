"""Grader for APTBench (Agent Process-level Testing Benchmark).

10 subtasks kept (per correlation analysis):
  - APTBench-SWE: EnvSetup (plan, error), IssueFix (locate, fix_patch, test_patch)
  - APTBench-DR:  Closed-ended (plan_en/zh, summ_ans_en/zh), Open-ended (openend_plan_en)

Metrics reported at four levels:
  1. Overall accuracy
  2. Per-domain (SWE / DR)
  3. Per-category (EnvSetup / IssueFix / Closed-ended / Open-ended)
  4. Per-subtask (10 individual tasks)

Two subtasks additionally report ROUGE scores (supplementary):
  deepresearch/summ_ans_{en,zh}

The answer extraction functions are ported, with modifications, from
TencentYoutuResearch/APTBench code/predict.py (Apache-2.0, Copyright (C) 2025
Tencent). See THIRD_PARTY_NOTICES.md.
"""
import copy
import re
import logging
from collections import defaultdict
from typing import Any, AsyncIterator, Dict

from ..utils import Sentinel, ExceptionWrapper
from ..metrics import get_accuracy, get_bootstrap_accuracy_std, mean_pass_at_k
from .base import AccuracyGraderBase, Grade, Score
from .registry import register

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Answer extraction functions (ported from APTBench code/predict.py)
# ---------------------------------------------------------------------------

def extract_answer(response: str) -> str | None:
    """Extract single letter answer via regex."""
    response = response.replace("*", "")
    match = re.search(r"\b([A-Za-z])[\)\n]", response)
    return match.group(1) if match else None


def extract_answer_summ_ans(response: str) -> str | None:
    if "]" in response:
        return response.split("]")[0]
    return None


EXTRACTORS: dict[str, Any] = {
    "extract_answer": extract_answer,
    "extract_answer_summ_ans": extract_answer_summ_ans,
}

# ---------------------------------------------------------------------------
# Hierarchy mapping: subtask -> (domain, category)
# ---------------------------------------------------------------------------

ROUGE_TASKS = frozenset({
    "deepresearch/summ_ans_en",
    "deepresearch/summ_ans_zh",
})


def _classify(difficulty: str) -> tuple[str, str]:
    """Map a difficulty string (domain/subtask) to (domain_label, category_label).

    Paper category mapping (APTBench Section 2.3):
      SWE / EnvSetup   : env_setup/{plan, error}            — Planning
      SWE / IssueFix   : issue_fix/{locate, fix_patch, test_patch} — Planning + Action
      DR  / Closed-ended: deepresearch/{plan_en, plan_zh}    — Planning
                          deepresearch/{summ_ans_en, summ_ans_zh} — Action
      DR  / Open-ended  : deepresearch/openend_plan_en       — Planning
    """
    if difficulty.startswith("env_setup/"):
        return ("SWE", "EnvSetup")
    if difficulty.startswith("issue_fix/"):
        return ("SWE", "IssueFix")
    if difficulty.startswith("deepresearch/openend_"):
        return ("DR", "Open-ended")
    if difficulty.startswith("deepresearch/"):
        return ("DR", "Closed-ended")
    return ("unknown", "unknown")


# ---------------------------------------------------------------------------
# Optional ROUGE support
# ---------------------------------------------------------------------------

try:
    from rouge import Rouge as _Rouge
    _rouge = _Rouge()
    _HAS_ROUGE = True
except ImportError:
    _HAS_ROUGE = False


def _compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute average ROUGE-1 F-score. Returns empty dict if rouge unavailable."""
    if not _HAS_ROUGE:
        logger.warning("rouge package not installed – ROUGE scores will be skipped")
        return {}
    if not predictions or not references:
        return {}
    # Filter out empty pairs
    pairs = [(p, r) for p, r in zip(predictions, references)
             if p and p.strip() and r and r.strip()]
    if not pairs:
        return {}
    preds, refs = zip(*pairs)
    try:
        scores = _rouge.get_scores(list(preds), list(refs), avg=True)
        return {
            "rouge-1": scores["rouge-1"]["f"] * 100,
        }
    except Exception as e:
        logger.warning(f"ROUGE computation failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

@register("aptbench")
class APTBench(AccuracyGraderBase):

    async def grade_sample(self, sample: Any, *_):
        if sample == Sentinel.COMPLETED:
            return sample

        assert isinstance(sample, dict)
        assert "parsed_generations" in sample
        assert "ground_truth" in sample

        extractor_name = sample.get("extractor", "extract_answer")
        extractor_fn = EXTRACTORS.get(extractor_name, extract_answer)
        judge_type = sample.get("judge_type", "exact")
        ground_truth = sample["ground_truth"]

        correct = []
        picked = []
        for gen in sample["parsed_generations"]:
            if gen is None:
                correct.append(0)
                picked.append(None)
                continue
            pred = extractor_fn(gen)
            picked.append(pred)
            if pred is None:
                correct.append(0)
            else:
                gt_str = ground_truth if isinstance(ground_truth, str) else str(ground_truth)
                correct.append(1 if pred == gt_str else 0)

        result = copy.deepcopy(sample)
        result["correct"] = correct
        result["picked"] = picked
        return result

    async def run(self, existing: AsyncIterator[Dict[str, Any]],
                  average_over: list[int], pass_at: list[int]):
        if not pass_at:
            pass_at = [1]
        if not average_over:
            average_over = [1]

        all_correct: list[list[int]] = []
        all_difficulty: list[str] = []
        all_picked: list[list[str | None]] = []
        all_gt: list[Any] = []
        completed = False
        count = 0

        # Read existing grades
        async for result in existing:
            if "correct" in result:
                all_correct.append(result["correct"])
                all_difficulty.append(result.get("difficulty", "unknown"))
                all_picked.append(result.get("picked", []))
                all_gt.append(result.get("ground_truth", ""))
            count += 1

        # Grade new samples
        async for result in self.async_grade_all_samples(start=count, grade_fn=self.grade_sample):
            if result == Sentinel.COMPLETED:
                completed = True
                break
            elif isinstance(result, ExceptionWrapper):
                yield result
            elif isinstance(result, dict) and "correct" in result:
                all_correct.append(result["correct"])
                all_difficulty.append(result.get("difficulty", "unknown"))
                all_picked.append(result.get("picked", []))
                all_gt.append(result.get("ground_truth", ""))
                yield Grade(element=result)

        if not completed:
            return

        # --- Yield scores at multiple levels ---

        def _yield_accuracy(label: str, correct_lists: list[list[int]]):
            """Helper generator for accuracy + pass@k scores."""
            scores = []
            if not correct_lists or not any(correct_lists):
                for n in average_over:
                    scores.append(Score(name=f"{label} accuracy (avg over {n})", value=float("nan")))
                for n in pass_at:
                    scores.append(Score(name=f"{label} accuracy (pass@{n})", value=float("nan")))
            else:
                for n in average_over:
                    scores.append(Score(name=f"{label} accuracy (avg over {n})",
                                        value=get_accuracy([s[:n] for s in correct_lists])))
                for n in pass_at:
                    per_problem = [(len(s), sum(s)) for s in correct_lists]
                    scores.append(Score(name=f"{label} accuracy (pass@{n})",
                                        value=mean_pass_at_k(per_problem, k=n)))
            return scores

        # 1. Overall
        for s in _yield_accuracy("overall", all_correct):
            yield s

        # 2. Per-domain (SWE / DR)
        by_domain: dict[str, list[list[int]]] = defaultdict(list)
        for corr, diff in zip(all_correct, all_difficulty):
            domain_label, _ = _classify(diff)
            by_domain[domain_label].append(corr)
        for domain_label in sorted(by_domain.keys()):
            for s in _yield_accuracy(domain_label, by_domain[domain_label]):
                yield s

        # 3. Per-category (EnvSetup / IssueFix / Closed-ended / Open-ended)
        by_category: dict[str, list[list[int]]] = defaultdict(list)
        for corr, diff in zip(all_correct, all_difficulty):
            _, cat_label = _classify(diff)
            by_category[cat_label].append(corr)
        for cat_label in sorted(by_category.keys()):
            for s in _yield_accuracy(cat_label, by_category[cat_label]):
                yield s

        # 4. Per-subtask
        by_subtask: dict[str, list[list[int]]] = defaultdict(list)
        for corr, diff in zip(all_correct, all_difficulty):
            by_subtask[diff].append(corr)
        for subtask in sorted(by_subtask.keys()):
            for s in _yield_accuracy(subtask, by_subtask[subtask]):
                yield s

        # 5. ROUGE supplementary scores for applicable tasks
        if _HAS_ROUGE:
            rouge_data: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
            for diff, pk, gt in zip(all_difficulty, all_picked, all_gt):
                if diff in ROUGE_TASKS:
                    preds_list, refs_list = rouge_data[diff]
                    pred_str = pk[0] if pk else ""
                    gt_str = gt if isinstance(gt, str) else str(gt)
                    preds_list.append(pred_str or "")
                    refs_list.append(gt_str)

            for subtask in sorted(rouge_data.keys()):
                preds_list, refs_list = rouge_data[subtask]
                rouge_scores = _compute_rouge(preds_list, refs_list)
                for metric, value in rouge_scores.items():
                    yield Score(name=f"{subtask} {metric}", value=value)

        yield Sentinel.COMPLETED
