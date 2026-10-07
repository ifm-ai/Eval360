"""HLE grader for the frozen Artificial Analysis (AA) evaluation contract."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import random
import re
import time
from typing import Any

from ..cache_salt import request_kwargs_with_cache_salt
from ..metrics import get_accuracy
from ..utils import Sentinel
from .base import AccuracyGraderBase
from .registry import register

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MAX_ATTEMPTS = 6
DEFAULT_JUDGE_MAX_RESPONSE_CHARS = 120_000
DEFAULT_JUDGE_TIMEOUT_SECONDS = 600.0

HLE_REASONING_CLOSE_TAGS = (
    "</ifm|think>",
    "</ifm|think_fast>",
    "</ifm|think_faster>",
    "</think>",
    "</think_fast>",
    "</think_faster>",
)

HLE_JUDGE_FINAL_CHANNEL_MARKERS = (
    "assistantfinal",
    "<|channel|>final<|message|>",
    "<|channel|>final",
)

# Adapted from the official HLE judge prompt (centerforaisafety/hle,
# hle_eval/run_judge_results.py; MIT). Lightly normalized: upstream's "0|\%|" and
# "100|\%|" are written "0%" and "100%", and one doubled blank line is single.
# See THIRD_PARTY_NOTICES.md.
JUDGE_PROMPT = """\
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available."""


def extract_hle_aa_mc_answer(generation: str | None) -> str | None:
    """Extract the last explicit choice from the visible final response."""
    if not generation:
        return None
    text = str(generation).strip()

    closing_positions = [(text.rfind(tag), tag) for tag in HLE_REASONING_CLOSE_TAGS]
    position, tag = max(closing_positions)
    if position >= 0:
        text = text[position + len(tag):].strip()
        if not text:
            return None

    patterns = (
        r"\\boxed\s*\{\s*([A-Z])\s*\}",
        r"\b(?:the\s+)?(?:final\s+|correct\s+)?answer[*_`]*\s*(?:is|:|-)"
        r"[*_`]*\s*(?:option\s+|choice\s+)?[*_`]*[\(\[]?([A-Z])\b",
        r"\b(?:the\s+)?(?:final\s+|correct\s+)?(?:answer\s+)?(?:option|choice)"
        r"[*_`]*\s*(?:(?:is|:|-)[*_`]*\s*)?[*_`]*[\(\[]?([A-Z])\b",
        r"\b(?:I\s+)?(?:choose|select)\s+(?:option\s+|choice\s+)?"
        r"[*_`]*[\(\[]?([A-Z])\b",
    )
    candidates: list[tuple[int, str]] = []
    for pattern in patterns:
        candidates.extend(
            (match.start(), match.group(1).upper())
            for match in re.finditer(pattern, text, flags=re.IGNORECASE)
        )
    if candidates:
        return max(candidates, key=lambda candidate: candidate[0])[1]

    leading_choice = re.match(
        r"^\s*[*_`]*[\(\[]?([A-Z])[\)\]]?[*_`]*"
        r"(?:[.:\-\u2013\u2014][*_`]*|\s)",
        text,
        flags=re.IGNORECASE,
    )
    if leading_choice:
        return leading_choice.group(1).upper()

    line_choices = {
        match.group(1).upper()
        for match in re.finditer(
            r"(?:^|\n)\s*[*_`]*[\(\[]?([A-Z])[\)\].:][*_`]*(?:\s|$)",
            text,
            flags=re.IGNORECASE,
        )
    }
    if len(line_choices) == 1:
        return line_choices.pop()

    final_line_choice = re.search(
        r"(?:^|\n)\s*[*_`]*[\(\[]?([A-Z])[\)\].:][*_`]*[^\n]*\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if final_line_choice:
        return final_line_choice.group(1).upper()

    standalone = re.search(
        r"(?:^|\n)\s*[*_`]*[\(\[]?([A-Z])[\)\].]?[*_`]*\s*$",
        text,
        flags=re.IGNORECASE,
    )
    return standalone.group(1).upper() if standalone else None


def prepare_hle_aa_judge_response(
    response: str | None,
    max_chars: int = DEFAULT_JUDGE_MAX_RESPONSE_CHARS,
) -> tuple[str, dict[str, Any]]:
    """Select visible final text, bounding only the selected judge input."""
    original = str(response or "")
    positions = [(original.rfind(tag), tag) for tag in HLE_REASONING_CLOSE_TAGS]
    position, close_tag = max(positions)
    if position >= 0:
        selected = original[position + len(close_tag):].strip()
        policy = "visible_final_after_reasoning_close_tag"
    else:
        selected = original.strip()
        close_tag = None
        policy = "no_close_tag_bounded_tail"

    omitted_chars = 0
    if max_chars > 0 and len(selected) > max_chars:
        omitted_chars = len(selected) - max_chars
        selected = selected[-max_chars:]
        if close_tag is not None:
            policy = "visible_final_after_reasoning_close_tag_bounded"

    empty_visible_final = close_tag is not None and not selected
    submitted = selected or "No answer"
    return submitted, {
        "policy": policy,
        "reasoning_close_tag": close_tag,
        "original_response_chars": len(original),
        "submitted_response_chars": len(submitted),
        "omitted_chars": omitted_chars,
        "empty_visible_final": empty_visible_final,
        "max_response_chars": max_chars,
    }


def _extract_judge_final_channel(response: str) -> tuple[str, str]:
    marker_positions = [
        (response.rfind(marker), marker) for marker in HLE_JUDGE_FINAL_CHANNEL_MARKERS
    ]
    position, marker = max(marker_positions)
    if position >= 0:
        return response[position + len(marker):].strip(), f"after_{marker}"

    think_position = response.rfind("</think>")
    if think_position >= 0:
        return response[think_position + len("</think>"):].strip(), "after_judge_think"
    return response.strip(), "plain_final_message"


def _json_correct_candidates(text: str) -> list[tuple[int, bool]]:
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, bool]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        value = str(parsed.get("correct", "")).strip().lower()
        if value in {"yes", "no"}:
            candidates.append((index, value == "yes"))
    return candidates


def parse_hle_aa_judge_verdict(response: str | None) -> tuple[bool | None, dict[str, Any]]:
    """Parse the last verdict from the judge's visible final channel."""
    if not response:
        return None, {
            "final_channel_policy": "empty_response",
            "parse_policy": "unparseable",
        }

    final_text, channel_policy = _extract_judge_final_channel(str(response))
    candidates: list[tuple[int, bool, str]] = [
        (position, value, "last_final_channel_json_correct")
        for position, value in _json_correct_candidates(final_text)
    ]
    line_pattern = re.compile(
        r"(?im)^\s*[*_`-]*correct\s*:?[*_`]*\s*:?[*_`]*\s*(yes|no)\b"
    )
    candidates.extend(
        (match.start(), match.group(1).lower() == "yes", "last_final_channel_correct_line")
        for match in line_pattern.finditer(final_text)
    )
    bracketed_pattern = re.compile(
        r"(?im)^\s*\[correct_answer\]\s*:\s*(yes|no)\s*[*_`]*\s*$"
    )
    candidates.extend(
        (
            match.start(),
            match.group(1).lower() == "yes",
            "last_final_channel_bracketed_correct_answer_boolean",
        )
        for match in bracketed_pattern.finditer(final_text)
    )
    if candidates:
        _, correct, parse_policy = max(candidates, key=lambda item: item[0])
        return correct, {
            "final_channel_policy": channel_policy,
            "parse_policy": parse_policy,
            "final_channel_chars": len(final_text),
        }

    return None, {
        "final_channel_policy": channel_policy,
        "parse_policy": "unparseable",
        "final_channel_chars": len(final_text),
    }


class HLEAAJudgeFailure(RuntimeError):
    """Raised rather than silently counting an unavailable judge as incorrect."""

    def __init__(self, audit: dict[str, Any]):
        self.audit = audit
        self.error_code = "hle_aa_judge_failure"
        self.attempts = audit["attempt_count"]
        self.elapsed_seconds = audit["elapsed_seconds"]
        self.retriable = True
        self.http_status = audit.get("last_http_status")
        super().__init__(
            f"HLE-AA judge failed after {self.attempts} attempts: {audit['last_error']}"
        )


@register("hle_aa")
class HLEAAGrader(AccuracyGraderBase):
    """Dispatch HLE-AA rows to local MC scoring or the gpt-oss judge."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "hle_aa"
        self.model = self.task.grader.llm_as_judge
        if self.model is None:
            raise ValueError("HLE-AA exactMatch scoring requires grader.llm_as_judge")
        self.openai_connection = self.request_openai_connection(self.model)

        contract = (self.task.meta or {}).get("judge_contract") or {}
        self.max_attempts = max(
            1, int(contract.get("max_attempts", DEFAULT_JUDGE_MAX_ATTEMPTS))
        )
        self.max_response_chars = int(
            contract.get("max_response_chars", DEFAULT_JUDGE_MAX_RESPONSE_CHARS)
        )
        self.timeout_seconds = float(
            contract.get("timeout_seconds", DEFAULT_JUDGE_TIMEOUT_SECONDS)
        )
        concurrency = max(1, int(self.model.max_simultaneous_requests or 1))
        self._judge_semaphore = asyncio.Semaphore(concurrency)

    def _grading_generator(self, start: int = 0, skip_rows: set[int] | None = None):
        # Keep consuming generation results while judge requests wait for a live
        # deployment. A bounded batch here creates pipeline backpressure: once
        # the batch fills, generation cannot reach its completion hook and free
        # a Slurm slot for the judge. Judge HTTP concurrency remains bounded by
        # self._judge_semaphore in _judge_exact_generation.
        return self.async_grade_all_samples_nonblocking(
            start=start,
            grade_fn=self.grade_sample,
            skip_rows=skip_rows,
        )

    @staticmethod
    def _sample_field(sample: dict[str, Any], name: str) -> Any:
        value = sample.get(name)
        if value is not None:
            return value
        metadata = sample.get("metadata")
        return metadata.get(name) if isinstance(metadata, dict) else None

    async def _judge_exact_generation(
        self,
        *,
        question: str,
        generation: str | None,
        correct_answer: str,
    ) -> tuple[bool, dict[str, Any]]:
        judge_input, extraction = prepare_hle_aa_judge_response(
            generation,
            max_chars=self.max_response_chars,
        )
        messages = [{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                question=question,
                response=judge_input,
                correct_answer=correct_answer,
            ),
        }]

        attempts: list[dict[str, Any]] = []
        transport_errors: list[str] = []
        last_error = "unknown judge failure"
        last_http_status = None
        started = time.monotonic()

        for attempt in range(self.max_attempts):
            attempt_record: dict[str, Any] = {"attempt": attempt + 1}
            try:
                async with self._judge_semaphore:
                    async with asyncio.timeout(self.timeout_seconds):
                        # Include endpoint acquisition in the timeout. Otherwise
                        # an unavailable judge can block forever before the
                        # request timeout even begins.
                        client = await self.openai_connection.get_client()
                        request_kwargs = request_kwargs_with_cache_salt(
                            self.model.openai_kwargs,
                            self.model.cache_salt,
                        )
                        completion = await client.chat.completions.create(
                            model=self.model.api_model_name or self.model.name,
                            messages=messages,
                            **request_kwargs,
                        )
                judge_response = completion.choices[0].message.content
                verdict, parse_details = parse_hle_aa_judge_verdict(judge_response)
                attempt_record.update({
                    "transport_ok": True,
                    "judge_response": judge_response,
                    **parse_details,
                })
                attempts.append(attempt_record)
                if verdict is not None:
                    return verdict, {
                        "method": "hle_aa_llm_judge",
                        "extraction": extraction,
                        "attempt_count": attempt + 1,
                        "attempts": attempts,
                        "transport_errors": transport_errors,
                        "terminal_failure": False,
                        "elapsed_seconds": time.monotonic() - started,
                        **parse_details,
                    }
                last_error = "empty or unparseable judge response"
                attempt_record["error"] = last_error
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                last_http_status = getattr(exc, "status_code", None)
                transport_errors.append(last_error)
                attempt_record.update({
                    "transport_ok": False,
                    "http_status": last_http_status,
                    "error": last_error,
                })
                attempts.append(attempt_record)

            if attempt + 1 < self.max_attempts:
                delay = min(4.0, 0.5 * (2 ** attempt)) + random.random() * 0.25
                logger.warning(
                    "HLE-AA judge attempt %d/%d failed (%s); retrying in %.2fs",
                    attempt + 1,
                    self.max_attempts,
                    last_error,
                    delay,
                )
                await asyncio.sleep(delay)

        audit = {
            "method": "hle_aa_llm_judge_error",
            "extraction": extraction,
            "attempt_count": self.max_attempts,
            "attempts": attempts,
            "transport_errors": transport_errors,
            "terminal_failure": True,
            "last_error": last_error,
            "last_http_status": last_http_status,
            "elapsed_seconds": time.monotonic() - started,
        }
        raise HLEAAJudgeFailure(audit)

    async def grade_sample(self, sample: Any, *_):
        if sample == Sentinel.COMPLETED:
            return sample
        if not isinstance(sample, dict):
            raise TypeError("HLE-AA sample must be a dictionary")

        generations = sample.get("generations")
        if not isinstance(generations, list) or not generations:
            raise ValueError("HLE-AA sample must contain a non-empty generations list")
        if "ground_truth" not in sample:
            raise ValueError("HLE-AA sample is missing ground_truth")

        answer_type = self._sample_field(sample, "answer_type")
        result = copy.deepcopy(sample)

        if answer_type == "multipleChoice":
            expected = str(sample["ground_truth"]).strip().upper()
            picked = [extract_hle_aa_mc_answer(generation) for generation in generations]
            correct = [choice == expected for choice in picked]
            result.update({
                "picked": picked,
                "correct": correct,
                "accuracy": get_accuracy(correct),
                "hle_aa_score_audits": [
                    {
                        "method": "hle_aa_multiple_choice_visible_final",
                        "picked": choice,
                        "terminal_failure": False,
                    }
                    for choice in picked
                ],
            })
            return result

        if answer_type == "exactMatch":
            question = self._sample_field(sample, "question")
            if not question:
                raise ValueError("HLE-AA exactMatch sample is missing the raw question")
            judged = await asyncio.gather(*(
                self._judge_exact_generation(
                    question=str(question),
                    generation=generation,
                    correct_answer=str(sample["ground_truth"]),
                )
                for generation in generations
            ))
            correct = [verdict for verdict, _ in judged]
            result.update({
                "correct": correct,
                "accuracy": get_accuracy(correct),
                "hle_aa_score_audits": [audit for _, audit in judged],
            })
            return result

        raise ValueError(f"Unknown HLE-AA answer_type: {answer_type!r}")
