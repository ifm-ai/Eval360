from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


CHOICE_SCORING_MODE = "choice_scoring"
CHOICE_SCORING_GRADER_TYPES = frozenset({"choice_scoring", "multiple_choice_nll"})
DEFAULT_SCORING_COMPLETION_PREFIX = "Answer:"
GENERATED_TAIL_N_TOKENS_FIELD = "eval360_generated_tail_n_tokens"
_CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def is_choice_scoring_grader_type(grader_type: Any) -> bool:
    return str(grader_type).strip() in CHOICE_SCORING_GRADER_TYPES


def choice_scoring_labels(n_choices: int) -> list[str]:
    if n_choices <= 0:
        raise ValueError(f"choice_scoring requires at least one choice, got {n_choices}")
    if n_choices > len(_CHOICE_LABELS):
        raise ValueError(
            f"choice_scoring supports at most {len(_CHOICE_LABELS)} labels, got {n_choices}"
        )
    return list(_CHOICE_LABELS[:n_choices])


def choice_scoring_fields(n_choices: int) -> dict[str, Any]:
    labels = choice_scoring_labels(n_choices)
    return {
        "scoring_mode": CHOICE_SCORING_MODE,
        "scoring_completions": list(labels),
        "scoring_completion_labels": list(labels),
    }


def resolve_scoring_completions(sample: dict[str, Any]) -> list[Any]:
    if "scoring_completions" not in sample:
        raise ValueError("choice_scoring sample requires scoring_completions")

    completions = sample["scoring_completions"]
    if isinstance(completions, (str, bytes)) or not isinstance(completions, Sequence):
        raise ValueError("scoring_completions must be a list of choices")

    resolved = list(completions)
    if not resolved:
        raise ValueError("scoring_completions must not be empty")
    return resolved


def resolve_scoring_completion_labels(sample: dict[str, Any]) -> list[Any]:
    completions = resolve_scoring_completions(sample)
    labels = (
        sample["scoring_completion_labels"]
        if "scoring_completion_labels" in sample
        else completions
    )
    if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
        raise ValueError("scoring_completion_labels must be a list when provided")

    resolved = list(labels)
    if len(resolved) != len(completions):
        raise ValueError(
            "scoring_completions/scoring_completion_labels length mismatch: "
            f"{len(completions)} vs {len(resolved)}"
        )
    return resolved


def resolve_scoring_prompt_prefix(sample: dict[str, Any]) -> str:
    if "scoring_prompt_prefix" in sample:
        return str(sample["scoring_prompt_prefix"]).rstrip()
    if "completion_input" in sample:
        return str(sample["completion_input"]).rstrip()
    raise ValueError(
        "choice_scoring sample requires scoring_prompt_prefix or completion_input"
    )


def resolve_scoring_prompt_prefixes(
    sample: dict[str, Any],
    n_choices: int | None = None,
) -> list[str]:
    completions = (
        resolve_scoring_completions(sample) if n_choices is None else [None] * n_choices
    )
    if "scoring_prompt_prefixes" not in sample:
        return [resolve_scoring_prompt_prefix(sample)] * len(completions)

    prefixes = sample["scoring_prompt_prefixes"]
    if isinstance(prefixes, (str, bytes)) or not isinstance(prefixes, Sequence):
        raise ValueError("scoring_prompt_prefixes must be a list when provided")

    resolved = [str(prefix).rstrip() for prefix in prefixes]
    if len(resolved) != len(completions):
        raise ValueError(
            "scoring_completions/scoring_prompt_prefixes length mismatch: "
            f"{len(completions)} vs {len(resolved)}"
        )
    return resolved


def resolve_scoring_completion_prefix(sample: dict[str, Any]) -> str:
    return str(
        sample.get("scoring_completion_prefix", DEFAULT_SCORING_COMPLETION_PREFIX)
    ).rstrip()


def resolve_scoring_completion_token_counts(
    sample: dict[str, Any],
    n_choices: int,
    *,
    required: bool,
) -> list[int] | None:
    """Resolve optional per-choice token counts used without text offsets."""
    values = sample.get("scoring_completion_n_tokens")
    if values is None:
        if required:
            raise ValueError(
                "choice_scoring sample needs scoring_completion_n_tokens when "
                "raw logprob offsets are unavailable"
            )
        return None
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or len(values) != n_choices
    ):
        raise ValueError(
            "scoring_completions/scoring_completion_n_tokens length mismatch: "
            f"{n_choices} vs "
            f"{len(values) if isinstance(values, Sequence) else 'invalid'}"
        )

    counts = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(
                "scoring_completion_n_tokens must contain positive integral "
                f"non-boolean values: {values!r}"
            )
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "scoring_completion_n_tokens must contain positive integral "
                f"values: {values!r}"
            ) from error
        if (
            not math.isfinite(numeric_value)
            or numeric_value <= 0
            or not numeric_value.is_integer()
        ):
            raise ValueError(
                "scoring_completion_n_tokens must contain positive integral "
                f"values: {values!r}"
            )
        counts.append(int(numeric_value))
    return counts


def choice_scoring_suffix_start_chars(
    prompt_text: str,
    completion: Any,
    fallback_prefix_chars: int,
) -> int:
    """Locate the completion suffix inside one rendered scoring prompt."""
    completion_text = str(completion)
    spaced_completion = f" {completion_text}"
    if prompt_text.endswith(spaced_completion):
        return len(prompt_text) - len(spaced_completion)
    if prompt_text.endswith(completion_text):
        return len(prompt_text) - len(completion_text)
    return fallback_prefix_chars


def _extract_token_logprobs(choice_logprobs: Any) -> Any:
    if choice_logprobs is None:
        return None
    if isinstance(choice_logprobs, dict):
        return choice_logprobs.get("token_logprobs")
    return getattr(choice_logprobs, "token_logprobs", None)


def _extract_text_offsets(choice_logprobs: Any) -> Any:
    if choice_logprobs is None:
        return None
    if isinstance(choice_logprobs, dict):
        return choice_logprobs.get("text_offset")
    return getattr(choice_logprobs, "text_offset", None)


def _resolve_generated_tail_n_tokens(
    choice_logprobs: Any,
    generated_tail_n_tokens: int | None,
) -> int:
    if generated_tail_n_tokens is None:
        if isinstance(choice_logprobs, dict):
            generated_tail_n_tokens = choice_logprobs.get(
                GENERATED_TAIL_N_TOKENS_FIELD,
                0,
            )
        else:
            attributes = getattr(choice_logprobs, "__dict__", {})
            generated_tail_n_tokens = (
                attributes.get(GENERATED_TAIL_N_TOKENS_FIELD, 0)
                if isinstance(attributes, dict)
                else 0
            )
    if (
        isinstance(generated_tail_n_tokens, bool)
        or not isinstance(generated_tail_n_tokens, int)
        or generated_tail_n_tokens < 0
    ):
        raise ValueError(
            "generated choice-scoring tail must be a non-negative integer"
        )
    return generated_tail_n_tokens


def _finite_token_logprob(
    token_logprob: Any,
    *,
    strict: bool,
) -> float | None:
    if token_logprob is None or isinstance(token_logprob, bool):
        return None
    if isinstance(token_logprob, dict):
        value = token_logprob.get("logprob")
    elif isinstance(token_logprob, (int, float)):
        value = token_logprob
    else:
        value = getattr(token_logprob, "logprob", None)
    if value is None or isinstance(value, bool):
        return None
    if strict and not isinstance(value, (int, float)):
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(resolved):
        raise ValueError("non-finite token logprob in completion suffix")
    return resolved


def sum_choice_scoring_suffix_logprobs_with_count(
    choice_logprobs: Any,
    suffix_n_tokens: int | None,
    *,
    suffix_start_chars: int | None = None,
    prompt_text: str | None = None,
    strict: bool = False,
    generated_tail_n_tokens: int | None = None,
) -> tuple[float, int]:
    """Validate and sum the usable completion-suffix token logprobs."""
    token_logprobs = _extract_token_logprobs(choice_logprobs)
    if not isinstance(token_logprobs, list):
        raise ValueError(
            "token_logprobs missing or invalid in choice-scoring response"
        )
    if suffix_n_tokens is not None and suffix_n_tokens <= 0:
        raise ValueError(
            f"suffix_n_tokens must be positive, got {suffix_n_tokens}"
        )

    text_offsets = _extract_text_offsets(choice_logprobs)
    # Some OpenAI-compatible servers signal unavailable offsets as an empty
    # list.  Treat that the same as an omitted field so the explicit token-count
    # fallback can validate a proven generated tail.
    if text_offsets == []:
        text_offsets = None
    if text_offsets is not None:
        if (
            not isinstance(text_offsets, list)
            or len(text_offsets) != len(token_logprobs)
        ):
            raise ValueError(
                "choice-scoring text_offset/token_logprobs length mismatch"
            )
        if any(
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            for offset in text_offsets
        ):
            raise ValueError(
                "choice-scoring text_offset values must be non-negative integers"
            )
        if any(
            current < previous
            for previous, current in zip(
                text_offsets,
                text_offsets[1:],
            )
        ):
            raise ValueError(
                "choice-scoring text_offset values must be ordered"
            )

    if (
        suffix_start_chars is not None
        and prompt_text is not None
        and text_offsets is not None
    ):
        prompt_end = len(prompt_text)
        total = 0.0
        matched = 0
        for idx, (offset, token_logprob) in enumerate(
            zip(text_offsets, token_logprobs)
        ):
            if offset >= prompt_end:
                continue
            end = (
                text_offsets[idx + 1]
                if idx + 1 < len(text_offsets)
                else prompt_end
            )
            end = min(end, prompt_end)
            if end <= suffix_start_chars:
                continue
            value = _finite_token_logprob(
                token_logprob,
                strict=strict,
            )
            if value is None:
                raise ValueError(
                    "missing token logprob or non-numeric token logprob "
                    "in completion suffix"
                )
            total += value
            matched += 1
        if matched == 0:
            raise ValueError(
                "choice-scoring response did not include any "
                "completion-suffix tokens"
            )
        return total, matched

    if suffix_n_tokens is None:
        raise ValueError(
            "choice-scoring response needs text_offset when "
            "scoring_completion_n_tokens is omitted"
        )
    tail_n_tokens = _resolve_generated_tail_n_tokens(
        choice_logprobs,
        generated_tail_n_tokens,
    )
    prompt_token_end = len(token_logprobs) - tail_n_tokens
    if prompt_token_end < suffix_n_tokens:
        raise ValueError(
            "choice-scoring response shorter than the requested completion "
            "suffix after excluding generated tokens: "
            f"{len(token_logprobs)=} {tail_n_tokens=} "
            f"{suffix_n_tokens=}"
        )

    total = 0.0
    suffix_start = prompt_token_end - suffix_n_tokens
    for token_logprob in token_logprobs[
        suffix_start:prompt_token_end
    ]:
        value = _finite_token_logprob(
            token_logprob,
            strict=strict,
        )
        if value is None:
            raise ValueError(
                "missing token logprob or non-numeric token logprob "
                "in completion suffix"
            )
        total += value
    return total, suffix_n_tokens


def build_choice_scoring_prompts(sample: dict[str, Any]) -> tuple[list[str], list[str]]:
    completions = resolve_scoring_completions(sample)
    prompt_prefixes = resolve_scoring_prompt_prefixes(sample, len(completions))
    completion_prefix = resolve_scoring_completion_prefix(sample)
    return (
        [
            f"{prompt_prefix} {completion}"
            for prompt_prefix, completion in zip(prompt_prefixes, completions)
        ],
        [f"{completion_prefix} {completion}" for completion in completions],
    )


def is_choice_scoring_row(sample: Any) -> bool:
    return (
        isinstance(sample, dict)
        and sample.get("scoring_mode") == CHOICE_SCORING_MODE
        and "scoring_completions" in sample
        and (
            "scoring_prompt_prefixes" in sample
            or "scoring_prompt_prefix" in sample
            or "completion_input" in sample
        )
    )


def validate_choice_scoring_row(
    sample: dict[str, Any],
    *,
    phase: str = "request",
) -> None:
    if not isinstance(sample, dict):
        raise ValueError(f"choice_scoring {phase} sample must be a dict")
    if sample.get("scoring_mode") != CHOICE_SCORING_MODE:
        raise ValueError(
            f"choice_scoring {phase} sample requires scoring_mode={CHOICE_SCORING_MODE!r}"
        )
    resolve_scoring_completions(sample)
    resolve_scoring_completion_labels(sample)
    resolve_scoring_prompt_prefixes(sample)
    resolve_ground_truth_index(sample)


def resolve_ground_truth_index(
    sample: dict[str, Any],
    labels: list[Any] | None = None,
    completions: list[Any] | None = None,
) -> int:
    completions = (
        completions if completions is not None else resolve_scoring_completions(sample)
    )
    labels = (
        labels if labels is not None else resolve_scoring_completion_labels(sample)
    )

    if "ground_truth_index" in sample:
        label_idx = int(sample["ground_truth_index"])
    else:
        if "ground_truth" not in sample:
            raise ValueError(
                "choice_scoring sample requires ground_truth_index or ground_truth"
            )
        ground_truth = sample["ground_truth"]
        if isinstance(ground_truth, int) and not isinstance(ground_truth, bool):
            label_idx = ground_truth
        else:
            candidates = (
                list(ground_truth)
                if isinstance(ground_truth, (list, tuple, set))
                else [ground_truth]
            )
            label_idx = -1
            for candidate in candidates:
                candidate_text = str(candidate).strip()
                for idx, label in enumerate(labels):
                    if str(label).strip() == candidate_text:
                        label_idx = idx
                        break
                if label_idx >= 0:
                    break
                for idx, completion in enumerate(completions):
                    if str(completion).strip() == candidate_text:
                        label_idx = idx
                        break
                if label_idx >= 0:
                    break
            if label_idx < 0:
                raise ValueError(
                    "ground_truth does not match any scoring_completion_labels or "
                    f"scoring_completions: {ground_truth!r}"
                )

    if label_idx < 0 or label_idx >= len(labels):
        raise ValueError(
            f"ground_truth_index out of range: {label_idx} for {len(labels)} choices"
        )
    return label_idx
