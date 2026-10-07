import copy
from collections import defaultdict
from statistics import mean
import traceback
from typing import Any

from ..choice_scoring_schema import (
    build_choice_scoring_prompts,
    choice_scoring_suffix_start_chars,
    resolve_ground_truth_index,
    resolve_scoring_completion_token_counts,
    resolve_scoring_completion_labels,
    resolve_scoring_completion_prefix,
    resolve_scoring_completions,
    resolve_scoring_prompt_prefixes,
    sum_choice_scoring_suffix_logprobs_with_count,
)
from ..utils import Sentinel, ExceptionWrapper
from .base import GraderBase, Grade, Score
from .registry import register


@register("choice_scoring", "multiple_choice_nll")
class ChoiceScoring(GraderBase):
    @staticmethod
    def _argmin(values: list[float]) -> tuple[int, float]:
        if not values:
            raise ValueError("cannot take argmin of an empty list")
        best_idx = 0
        best_val = float(values[0])
        for idx, value in enumerate(values[1:], start=1):
            value = float(value)
            if value < best_val:
                best_idx = idx
                best_val = value
        return best_idx, best_val

    @staticmethod
    def _mean_or_nan(values: list[float]) -> float:
        return float(mean(values)) if values else float("nan")

    @staticmethod
    def _sum_suffix_token_logprobs_with_count(
        choice_logprobs,
        suffix_n_tokens: int | None,
        *,
        suffix_start_chars: int | None = None,
        prompt_text: str | None = None,
    ) -> tuple[float, int]:
        return sum_choice_scoring_suffix_logprobs_with_count(
            choice_logprobs,
            suffix_n_tokens,
            suffix_start_chars=suffix_start_chars,
            prompt_text=prompt_text,
        )

    @staticmethod
    def _sum_suffix_token_logprobs(
        choice_logprobs,
        suffix_n_tokens: int,
        *,
        suffix_start_chars: int | None = None,
        prompt_text: str | None = None,
    ) -> float:
        total, _ = ChoiceScoring._sum_suffix_token_logprobs_with_count(
            choice_logprobs,
            suffix_n_tokens,
            suffix_start_chars=suffix_start_chars,
            prompt_text=prompt_text,
        )
        return total

    @staticmethod
    def _suffix_start_chars(
        prompt_text: str,
        completion: Any,
        fallback_prefix_chars: int,
    ) -> int:
        return choice_scoring_suffix_start_chars(
            prompt_text,
            completion,
            fallback_prefix_chars,
        )

    @staticmethod
    def _choice_prompts(sample: dict[str, Any]) -> tuple[list[str], list[str]]:
        metadata = sample.get("choice_scoring_metadata") or {}
        full_prompts = metadata.get("full_prompts")
        completion_prompts = metadata.get("completion_prompts")
        if full_prompts is not None and completion_prompts is not None:
            return full_prompts, completion_prompts

        return build_choice_scoring_prompts(sample)

    @staticmethod
    def _choice_token_counts(
        sample: dict[str, Any],
        n_choices: int,
        *,
        required: bool,
    ) -> list[int] | None:
        return resolve_scoring_completion_token_counts(
            sample,
            n_choices,
            required=required,
        )

    @staticmethod
    def _choice_char_counts(sample: dict[str, Any], completions: list[Any]) -> list[float]:
        values = sample.get("scoring_completion_n_chars")
        if values is None:
            counts = [float(len(str(completion))) for completion in completions]
        else:
            if len(values) != len(completions):
                raise ValueError(
                    "scoring_completions/scoring_completion_n_chars length mismatch: "
                    f"{len(completions)} vs {len(values)}"
                )
            counts = [float(value) for value in values]
        if any(value <= 0 for value in counts):
            raise ValueError(f"scoring_completion_n_chars must be positive: {counts}")
        return counts

    @staticmethod
    def _resolve_ground_truth_index(
        sample: dict[str, Any],
        labels: list[Any],
        completions: list[Any],
    ) -> int:
        return resolve_ground_truth_index(sample, labels=labels, completions=completions)

    @staticmethod
    def _extract_choice_nlls(
        sample: dict[str, Any],
    ) -> tuple[list[float], list[float], list[int]]:
        if "choice_nll" in sample and "choice_nll_completion" in sample:
            choice_nll = [float(value) for value in sample["choice_nll"]]
            token_counts = ChoiceScoring._choice_token_counts(
                sample,
                len(choice_nll),
                required=True,
            )
            return (
                choice_nll,
                [float(value) for value in sample["choice_nll_completion"]],
                token_counts,
            )

        full_logprobs = sample.get("choice_scoring_full_logprobs")
        completion_logprobs = sample.get("choice_scoring_completion_logprobs")
        if full_logprobs is None or completion_logprobs is None:
            raise ValueError(
                "choice_scoring sample is missing raw logprob payloads and precomputed "
                "choice_nll fields"
            )

        completions = resolve_scoring_completions(sample)
        completion_token_counts = ChoiceScoring._choice_token_counts(
            sample,
            len(completions),
            required=False,
        )

        full_prompts, completion_prompts = ChoiceScoring._choice_prompts(sample)
        prompt_prefixes = resolve_scoring_prompt_prefixes(sample, len(completions))
        completion_prefix = resolve_scoring_completion_prefix(sample)

        if len(full_logprobs) != len(completions):
            raise ValueError(
                "choice_scoring full-logprobs length mismatch: "
                f"{len(full_logprobs)} vs {len(completions)}"
            )
        if len(completion_logprobs) != len(completions):
            raise ValueError(
                "choice_scoring completion-logprobs length mismatch: "
                f"{len(completion_logprobs)} vs {len(completions)}"
            )

        choice_nll = []
        choice_nll_completion = []
        inferred_token_counts = []
        for idx, completion in enumerate(completions):
            suffix_n_tokens = (
                completion_token_counts[idx]
                if completion_token_counts is not None
                else None
            )
            full_sum, full_token_count = ChoiceScoring._sum_suffix_token_logprobs_with_count(
                full_logprobs[idx],
                suffix_n_tokens,
                suffix_start_chars=ChoiceScoring._suffix_start_chars(
                    full_prompts[idx],
                    completion,
                    len(prompt_prefixes[idx]),
                ),
                prompt_text=full_prompts[idx],
            )
            completion_sum, _ = ChoiceScoring._sum_suffix_token_logprobs_with_count(
                completion_logprobs[idx],
                suffix_n_tokens,
                suffix_start_chars=ChoiceScoring._suffix_start_chars(
                    completion_prompts[idx],
                    completion,
                    len(completion_prefix),
                ),
                prompt_text=completion_prompts[idx],
            )
            choice_nll.append(-full_sum)
            choice_nll_completion.append(-completion_sum)
            inferred_token_counts.append(suffix_n_tokens or full_token_count)
        return choice_nll, choice_nll_completion, inferred_token_counts

    @staticmethod
    def _compute_choice_metrics(sample: dict[str, Any]) -> dict[str, Any]:
        completions = resolve_scoring_completions(sample)
        choice_nll, choice_nll_completion, choice_n_tokens = (
            ChoiceScoring._extract_choice_nlls(sample)
        )
        if (
            len(choice_nll) != len(completions)
            or len(choice_nll_completion) != len(completions)
            or len(choice_n_tokens) != len(completions)
        ):
            raise ValueError(
                "choice-scoring metric length mismatch: "
                f"{len(choice_nll)=} {len(choice_nll_completion)=} "
                f"{len(choice_n_tokens)=} {len(completions)=}"
            )
        choice_n_chars = ChoiceScoring._choice_char_counts(sample, completions)
        labels = resolve_scoring_completion_labels(sample)
        label_idx = ChoiceScoring._resolve_ground_truth_index(sample, labels, completions)

        nll_idx, nll_val = ChoiceScoring._argmin(choice_nll)
        nll_char_values = [nll / n_chars for nll, n_chars in zip(choice_nll, choice_n_chars)]
        nll_char_idx, nll_char_val = ChoiceScoring._argmin(nll_char_values)
        nll_token_values = [nll / n_tokens for nll, n_tokens in zip(choice_nll, choice_n_tokens)]
        nll_token_idx, nll_token_val = ChoiceScoring._argmin(nll_token_values)
        nll_compl_values = [
            nll - nll_completion
            for nll, nll_completion in zip(choice_nll, choice_nll_completion)
        ]
        nll_compl_idx, nll_compl_val = ChoiceScoring._argmin(nll_compl_values)

        result = copy.deepcopy(sample)
        result.update(
            {
                "choice_nll": choice_nll,
                "choice_nll_completion": choice_nll_completion,
                "ground_truth_index": label_idx,
                "scoring_completion_n_tokens": choice_n_tokens,
                "scoring_completion_n_chars": choice_n_chars,
                "picked": [labels[nll_idx]],
                "picked_char": [labels[nll_char_idx]],
                "picked_token": [labels[nll_token_idx]],
                "picked_compl": [labels[nll_compl_idx]],
                "correct": [nll_idx == label_idx],
                "correct_char": [nll_char_idx == label_idx],
                "correct_token": [nll_token_idx == label_idx],
                "correct_compl": [nll_compl_idx == label_idx],
                "acc": 100.0 * float(nll_idx == label_idx),
                "acc_char": 100.0 * float(nll_char_idx == label_idx),
                "acc_token": 100.0 * float(nll_token_idx == label_idx),
                "acc_compl": 100.0 * float(nll_compl_idx == label_idx),
                "nll": nll_val,
                "nll_char": nll_char_val,
                "nll_token": nll_token_val,
                "nll_compl": nll_compl_val,
            }
        )
        return result

    async def grade_sample(self, sample: Any):
        raise NotImplementedError("ChoiceScoring grades inline in run()")

    async def run(self, existing, average_over, pass_at):
        del average_over, pass_at

        completed = False
        graded_rows: set[int] = set()
        metric_names = (
            "acc",
            "acc_char",
            "acc_token",
            "acc_compl",
            "nll",
            "nll_char",
            "nll_token",
            "nll_compl",
        )
        metric_sums = {metric: 0.0 for metric in metric_names}
        metric_counts = {metric: 0 for metric in metric_names}
        grouped_sums: dict[str, dict[str, float]] = defaultdict(
            lambda: {metric: 0.0 for metric in metric_names}
        )
        grouped_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {metric: 0 for metric in metric_names}
        )

        def record_metrics(sample: dict[str, Any]) -> None:
            group = sample.get("choice_group")
            group_key = str(group) if group is not None else None
            for metric in metric_names:
                if metric not in sample:
                    continue
                value = float(sample[metric])
                metric_sums[metric] += value
                metric_counts[metric] += 1
                if group_key is not None:
                    grouped_sums[group_key][metric] += value
                    grouped_counts[group_key][metric] += 1

        def recorded_mean(metric: str) -> float:
            count = metric_counts[metric]
            return metric_sums[metric] / count if count else float("nan")

        def grouped_mean(group_key: str, metric: str) -> float:
            count = grouped_counts[group_key][metric]
            return (
                grouped_sums[group_key][metric] / count
                if count
                else float("nan")
            )

        async for result in existing:
            if isinstance(result, dict) and "row" in result:
                graded_rows.add(result["row"])
            if isinstance(result, dict) and "acc" in result and "nll" in result:
                record_metrics(result)

        async for sample in self.samples_generator:
            if sample == Sentinel.COMPLETED:
                completed = True
                break
            if isinstance(sample, ExceptionWrapper):
                yield sample
                continue
            if sample.get("row") in graded_rows:
                continue
            try:
                if sample.get("eval360_input_too_long"):
                    scored = copy.deepcopy(sample)
                    scored.update(
                        {
                            "correct": [False],
                            "correct_char": [False],
                            "correct_token": [False],
                            "correct_compl": [False],
                            "acc": 0.0,
                            "acc_char": 0.0,
                            "acc_token": 0.0,
                            "acc_compl": 0.0,
                            "nll": float("inf"),
                            "nll_char": float("inf"),
                            "nll_token": float("inf"),
                            "nll_compl": float("inf"),
                        }
                    )
                else:
                    scored = self._compute_choice_metrics(sample)
                record_metrics(scored)
                yield Grade(element=scored)
            except Exception as exc:
                tb = "".join(traceback.TracebackException.from_exception(exc).format())
                yield ExceptionWrapper(exception=exc, trace=tb, instance=sample)

        if not completed:
            return

        if grouped_sums:
            for metric in metric_names:
                yield Score(
                    name=f"macro_avg/{metric}",
                    value=self._mean_or_nan(
                        [grouped_mean(group_key, metric) for group_key in grouped_sums]
                    ),
                )
                yield Score(
                    name=f"micro_avg/{metric}",
                    value=recorded_mean(metric),
                )
        else:
            for metric in metric_names:
                yield Score(
                    name=metric,
                    value=recorded_mean(metric),
                )
        yield Sentinel.COMPLETED
