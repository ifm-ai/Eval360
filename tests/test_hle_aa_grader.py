import asyncio
from types import SimpleNamespace

import pytest

from scheduler.cache_salt import CacheSaltConfig
from scheduler.grader.hle_aa import (
    HLEAAGrader,
    HLEAAJudgeFailure,
    extract_hle_aa_mc_answer,
    parse_hle_aa_judge_verdict,
    prepare_hle_aa_judge_response,
)
from scheduler.utils import Sentinel


@pytest.mark.parametrize(
    ("generation", "expected"),
    [
        ("Explanation: final.\nAnswer: C", "C"),
        (r"Reasoning. \boxed{B}", "B"),
        ("Answer: B, reconsidered.\nAnswer: A", "A"),
        ("D. The fourth option is correct.", "D"),
        ("Reasoning without a final marker", None),
        ("analysis Answer: A\n</think>\nExplanation: final.\nAnswer: D", "D"),
        ("analysis Answer: A\n</ifm|think>", None),
    ],
)
def test_extract_hle_aa_mc_answer(generation, expected):
    assert extract_hle_aa_mc_answer(generation) == expected


def test_prepare_judge_response_uses_visible_final():
    submitted, audit = prepare_hle_aa_judge_response(
        "hidden reasoning\n</ifm|think>\nExact Answer: Paris"
    )
    assert submitted == "Exact Answer: Paris"
    assert audit["policy"] == "visible_final_after_reasoning_close_tag"
    assert audit["reasoning_close_tag"] == "</ifm|think>"
    assert audit["omitted_chars"] == 0


def test_prepare_judge_response_bounds_selected_tail():
    submitted, audit = prepare_hle_aa_judge_response("abcdefgh", max_chars=4)
    assert submitted == "efgh"
    assert audit["policy"] == "no_close_tag_bounded_tail"
    assert audit["omitted_chars"] == 4


def test_prepare_judge_response_empty_visible_final_is_no_answer():
    submitted, audit = prepare_hle_aa_judge_response("reasoning</think>")
    assert submitted == "No answer"
    assert audit["empty_visible_final"] is True


@pytest.mark.parametrize(
    ("response", "expected", "policy"),
    [
        ('{"correct": "yes"}', True, "last_final_channel_json_correct"),
        ("correct: no", False, "last_final_channel_correct_line"),
        ("</think>\ncorrect: yes", True, "last_final_channel_correct_line"),
        (
            "correct: no\nassistantfinal\ncorrect: yes",
            True,
            "last_final_channel_correct_line",
        ),
        (
            "correct: yes\ncorrect: no",
            False,
            "last_final_channel_correct_line",
        ),
        (
            "[correct_answer]: yes",
            True,
            "last_final_channel_bracketed_correct_answer_boolean",
        ),
    ],
)
def test_parse_hle_aa_judge_verdict(response, expected, policy):
    verdict, details = parse_hle_aa_judge_verdict(response)
    assert verdict is expected
    assert details["parse_policy"] == policy


def test_parse_hle_aa_judge_verdict_rejects_unparseable():
    verdict, details = parse_hle_aa_judge_verdict("No boolean verdict here")
    assert verdict is None
    assert details["parse_policy"] == "unparseable"


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))]
        )


class FakeConnection:
    def __init__(self, completions):
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

    async def get_client(self):
        return self.client


def make_judge(outcomes, *, max_attempts=2):
    grader = HLEAAGrader.__new__(HLEAAGrader)
    completions = FakeCompletions(outcomes)
    grader.model = SimpleNamespace(
        openai_kwargs={"temperature": 0.0, "max_tokens": 16384},
        cache_salt=CacheSaltConfig(),
        api_model_name=None,
        name="gptoss-20b-judge",
    )
    grader.openai_connection = FakeConnection(completions)
    grader.max_attempts = max_attempts
    grader.max_response_chars = 120_000
    grader.timeout_seconds = 1
    grader._judge_semaphore = asyncio.Semaphore(1)
    return grader, completions


def test_exact_judge_retries_unparseable_response(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("scheduler.grader.hle_aa.asyncio.sleep", no_sleep)
    grader, completions = make_judge(
        ["missing verdict", "assistantfinal\ncorrect: yes"]
    )
    verdict, audit = asyncio.run(grader._judge_exact_generation(
        question="What is the capital of France?",
        generation="</think>\nExact Answer: Paris",
        correct_answer="Paris",
    ))
    assert verdict is True
    assert audit["attempt_count"] == 2
    assert audit["terminal_failure"] is False
    assert len(completions.calls) == 2
    assert "[response]: Exact Answer: Paris" in completions.calls[0]["messages"][0]["content"]


def test_exact_judge_terminal_failure_raises(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("scheduler.grader.hle_aa.asyncio.sleep", no_sleep)
    grader, _ = make_judge([RuntimeError("offline")], max_attempts=1)
    with pytest.raises(HLEAAJudgeFailure) as exc_info:
        asyncio.run(grader._judge_exact_generation(
            question="Question",
            generation="Exact Answer: answer",
            correct_answer="answer",
        ))
    assert exc_info.value.attempts == 1
    assert exc_info.value.audit["terminal_failure"] is True


def test_exact_judge_timeout_includes_endpoint_acquisition():
    class BlockingConnection:
        async def get_client(self):
            await asyncio.Event().wait()

    grader, _ = make_judge([], max_attempts=1)
    grader.openai_connection = BlockingConnection()
    grader.timeout_seconds = 0.01

    with pytest.raises(HLEAAJudgeFailure) as exc_info:
        asyncio.run(grader._judge_exact_generation(
            question="Question",
            generation="Exact Answer: answer",
            correct_answer="answer",
        ))

    assert exc_info.value.audit["attempt_count"] == 1
    assert exc_info.value.audit["attempts"][0]["transport_ok"] is False
    assert exc_info.value.audit["last_error"].startswith("TimeoutError:")


def test_grading_generator_drains_upstream_while_judge_is_blocked():
    async def run_test():
        judge_release = asyncio.Event()
        source_consumed: list[object] = []

        async def samples():
            for row in range(3):
                sample = {"row": row, "generations": [f"answer {row}"]}
                source_consumed.append(row)
                yield sample
            source_consumed.append(Sentinel.COMPLETED)
            yield Sentinel.COMPLETED

        grader = HLEAAGrader.__new__(HLEAAGrader)
        grader.samples_generator = samples()
        grader._initialized = True
        grader._initialize_lock = asyncio.Lock()

        async def parse_generations_async(sample):
            return sample["generations"]

        async def blocked_grade(sample, *_):
            await judge_release.wait()
            result = dict(sample)
            result["correct"] = [True]
            return result

        grader.parse_generations_async = parse_generations_async
        grader.grade_sample = blocked_grade

        results = []

        async def consume_results():
            async for result in grader._grading_generator():
                results.append(result)

        consumer = asyncio.create_task(consume_results())
        for _ in range(20):
            if source_consumed and source_consumed[-1] == Sentinel.COMPLETED:
                break
            await asyncio.sleep(0)

        assert source_consumed == [0, 1, 2, Sentinel.COMPLETED]
        assert not consumer.done()

        judge_release.set()
        await consumer
        assert [result["row"] for result in results[:-1]] == [0, 1, 2]
        assert results[-1] == Sentinel.COMPLETED

    asyncio.run(run_test())


def test_multiple_choice_grade_sample_uses_metadata_fallback():
    grader = HLEAAGrader.__new__(HLEAAGrader)
    sample = {
        "row": 0,
        "ground_truth": "D",
        "metadata": {"answer_type": "multipleChoice"},
        "generations": ["analysis Answer: A\n</think>\nAnswer: D"],
    }
    result = asyncio.run(grader.grade_sample(sample))
    assert result["picked"] == ["D"]
    assert result["correct"] == [True]
