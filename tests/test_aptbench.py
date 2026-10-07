"""Tests for the APTBench grader helpers and aggregation flow."""

from __future__ import annotations

import math
from typing import Any, AsyncIterator

import pytest

from scheduler.grader.aptbench import APTBench
from scheduler.grader import aptbench as aptbench_module
from scheduler.grader.base import Grade, Score
from scheduler.utils import ExceptionWrapper, Sentinel


class _MockEvent:
    """Minimal event object that selects the passthrough parser."""

    parser_type = "passthrough"


class APTBenchTestBase:
    """Shared async grader setup for APTBench tests.

    What: centralizes fake sample streams and run collection helpers.
    Executes: the APTBench constructor and `run` stream interface with local data.
    Why: APTBench grading consumes async streams, so the fakes preserve scheduler-style iteration without scheduler dependencies.
    """

    @staticmethod
    async def async_iter(*items: Any) -> AsyncIterator[Any]:
        """Yield items as an async iterator for grader run tests.

        What: adapts in-memory samples into the async stream expected by graders.
        Executes: the same async iteration protocol used by scheduler pipelines.
        Why: avoids live scheduler dependencies while preserving streaming behavior.
        """
        for item in items:
            yield item

    @classmethod
    def make_grader(cls, *samples: Any) -> APTBench:
        """Create an APTBench grader with a fake async samples generator.

        What: builds the grader with only the event parser state it needs.
        Executes: APTBench initialization against an async sample stream.
        Why: isolates grading logic from scheduler, job-manager, and event-manager setup.
        """
        return APTBench(
            samples_generator=cls.async_iter(*samples),
            event_manager=None,
            job_manager=None,
            event=_MockEvent(),
            task=None,
        )

    @classmethod
    async def collect_run(
        cls,
        grader: APTBench,
        *existing: dict[str, Any],
        average_over: list[int] | None = None,
        pass_at: list[int] | None = None,
    ) -> list[Any]:
        """Collect all objects yielded by APTBench.run().

        What: materializes the async run stream for assertions.
        Executes: `APTBench.run` with optional existing records and score settings.
        Why: lets tests verify emitted grades, scores, and sentinels deterministically.
        """
        results = []
        async for item in grader.run(
            existing=cls.async_iter(*existing),
            average_over=average_over or [],
            pass_at=pass_at or [],
        ):
            results.append(item)
        return results


class TestAnswerExtraction(APTBenchTestBase):
    """What: groups tests for APTBench answer extractor helpers.
    Executes: `extract_answer` and `extract_answer_summ_ans` with representative model text.
    Why: these helpers define the picked answer stored by APTBench.grade_sample().
    """

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("A)\nThe answer is clear.", "A"),
            ("**b)** selected after reasoning", "b"),
            ("No single-letter marker here", None),
            ("First line says z\nthen more text", "z"),
        ],
    )
    def test_extract_answer_uses_letter_marker(self, response: str, expected: str | None) -> None:
        """What: verifies single-letter answers are extracted only when followed by ')' or newline.
        Executes: `extract_answer()` against starred, lowercase, missing-marker, and first-line inputs.
        Why: covers the parser contract that turns raw generations into exact-match choices.
        """
        assert aptbench_module.extract_answer(response) == expected

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ("summary answer] trailing rationale", "summary answer"),
            ("] empty prefix is allowed", ""),
            ("missing bracket", None),
        ],
    )
    def test_extract_answer_summ_ans_takes_prefix_before_bracket(
        self,
        response: str,
        expected: str | None,
    ) -> None:
        """What: verifies summary-answer extraction returns text before the first closing bracket.
        Executes: `extract_answer_summ_ans()` for present, empty-prefix, and absent bracket cases.
        Why: covers the DeepResearch summary extractor without needing ROUGE or grader state.
        """
        assert aptbench_module.extract_answer_summ_ans(response) == expected


class TestClassificationAndRouge(APTBenchTestBase):
    """What: groups tests for APTBench difficulty classification and optional ROUGE handling.
    Executes: `_classify()` and `_compute_rouge()` with mocked optional ROUGE dependencies.
    Why: hierarchy labels and supplementary scores are core to APTBench score aggregation.
    """

    @pytest.mark.parametrize(
        ("difficulty", "expected"),
        [
            ("env_setup/plan", ("SWE", "EnvSetup")),
            ("issue_fix/test_patch", ("SWE", "IssueFix")),
            ("deepresearch/openend_plan_en", ("DR", "Open-ended")),
            ("deepresearch/plan_en", ("DR", "Closed-ended")),
            ("mystery/task", ("unknown", "unknown")),
        ],
    )
    def test_classify_maps_subtasks_to_hierarchy(
        self,
        difficulty: str,
        expected: tuple[str, str],
    ) -> None:
        """What: verifies known subtasks map to their documented domain and category labels.
        Executes: `_classify()` across SWE, DeepResearch, open-ended, and unknown difficulties.
        Why: protects the score buckets emitted by APTBench.run().
        """
        assert aptbench_module._classify(difficulty) == expected

    def test_compute_rouge_skips_when_dependency_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """What: verifies ROUGE returns an empty metric dict when the optional package is unavailable.
        Executes: `_compute_rouge()` with `_HAS_ROUGE` patched false.
        Why: summary-answer grading must remain usable on installations that omit the optional rouge package.
        """
        monkeypatch.setattr(aptbench_module, "_HAS_ROUGE", False)
        assert aptbench_module._compute_rouge(["prediction"], ["reference"]) == {}

    def test_compute_rouge_filters_empty_pairs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """What: verifies ROUGE averages only non-empty prediction/reference pairs.
        Executes: `_compute_rouge()` through a fake Rouge.get_scores implementation.
        Why: covers the empty-pair filtering branch while asserting the scorer call shape.
        """
        calls: list[tuple[list[str], list[str], bool]] = []

        class FakeRouge:
            """Small fake that records the pairs sent to Rouge.get_scores()."""

            def get_scores(self, predictions: list[str], references: list[str], avg: bool) -> dict[str, Any]:
                """Return a deterministic score for non-empty pairs."""
                calls.append((predictions, references, avg))
                return {"rouge-1": {"f": 0.25}}

        monkeypatch.setattr(aptbench_module, "_HAS_ROUGE", True)
        monkeypatch.setattr(aptbench_module, "_rouge", FakeRouge(), raising=False)

        assert aptbench_module._compute_rouge(["", "kept"], ["ignored", "ref"]) == {"rouge-1": 25.0}
        assert calls == [(["kept"], ["ref"], True)]

    def test_compute_rouge_returns_empty_on_scorer_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """What: verifies ROUGE scorer exceptions are swallowed so grading can continue.
        Executes: `_compute_rouge()` with a fake scorer that raises RuntimeError.
        Why: covers the optional dependency failure path without failing the grader run.
        """

        class FailingRouge:
            """Fake scorer that raises like a fragile optional dependency."""

            def get_scores(self, predictions: list[str], references: list[str], avg: bool) -> dict[str, Any]:
                """Raise an error to exercise the warning path."""
                raise RuntimeError("rouge failed")

        monkeypatch.setattr(aptbench_module, "_HAS_ROUGE", True)
        monkeypatch.setattr(aptbench_module, "_rouge", FailingRouge(), raising=False)

        assert aptbench_module._compute_rouge(["prediction"], ["reference"]) == {}


class TestGradeSample(APTBenchTestBase):
    """What: groups tests for APTBench.grade_sample().
    Executes: `APTBench.grade_sample()` with sentinels, extractor selection, and copied samples.
    Why: grade_sample is the public unit that converts parsed generations into correctness arrays.
    """

    @pytest.mark.asyncio
    async def test_sentinel_is_returned_unchanged(self) -> None:
        """What: verifies sentinel.COMPLETED passes through without grading.
        Executes: `APTBench.grade_sample()` with `Sentinel.COMPLETED`.
        Why: preserves the stream-termination contract used by inherited run loops.
        """
        grader = self.make_grader()
        assert await grader.grade_sample(Sentinel.COMPLETED) == Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_default_extractor_marks_correct_and_missing_answers(self) -> None:
        """What: verifies default extraction stores picked answers and per-generation correctness.
        Executes: `APTBench.grade_sample()` through the default single-letter extractor.
        Why: covers correct, incorrect, None, and unparsable generations without mutating input samples.
        """
        grader = self.make_grader()
        sample = {
            "parsed_generations": ["A)\n", "B)\n", None, "no answer"],
            "ground_truth": "A",
        }

        result = await grader.grade_sample(sample)

        assert result["picked"] == ["A", "B", None, None]
        assert result["correct"] == [1, 0, 0, 0]
        assert "correct" not in sample

    @pytest.mark.asyncio
    async def test_unknown_extractor_falls_back_to_default(self) -> None:
        """What: verifies an unknown extractor name falls back to the single-letter extractor.
        Executes: `APTBench.grade_sample()` with an unregistered `extractor` field.
        Why: protects compatibility with malformed or future config values.
        """
        grader = self.make_grader()
        sample = {
            "parsed_generations": ["A)\n"],
            "ground_truth": "A",
            "extractor": "not_registered",
        }

        result = await grader.grade_sample(sample)

        assert result["picked"] == ["A"]
        assert result["correct"] == [1]

    @pytest.mark.asyncio
    async def test_summary_extractor_uses_configured_function(self) -> None:
        """What: verifies the summary extractor path compares the bracket prefix against ground truth.
        Executes: `APTBench.grade_sample()` using `extract_answer_summ_ans`.
        Why: covers the configured extractor route used by DeepResearch summary-answer tasks.
        """
        grader = self.make_grader()
        sample = {
            "parsed_generations": ["summary] rationale", "other] rationale"],
            "ground_truth": "summary",
            "extractor": "extract_answer_summ_ans",
        }

        result = await grader.grade_sample(sample)

        assert result["picked"] == ["summary", "other"]
        assert result["correct"] == [1, 0]


class TestRunAggregation(APTBenchTestBase):
    """What: groups tests for APTBench.run() defaults, scores, and error passthrough.
    Executes: `APTBench.run()` over mocked async sample streams.
    Why: run aggregation emits the user-visible hierarchy scores and completion sentinel.
    """

    @pytest.mark.asyncio
    async def test_run_defaults_emit_hierarchy_scores_and_completed(self) -> None:
        """What: verifies empty average/pass lists default to avg@1/pass@1 and emit hierarchy scores.
        Executes: `APTBench.run()` with SWE and DeepResearch samples plus `Sentinel.COMPLETED`.
        Why: covers the default score settings and hierarchy buckets that downstream reports consume.
        """
        samples = [
            {
                "row": 0,
                "generations": ["A)\n"],
                "ground_truth": "A",
                "difficulty": "env_setup/plan",
            },
            {
                "row": 1,
                "generations": ["B)\n"],
                "ground_truth": "A",
                "difficulty": "deepresearch/openend_plan_en",
            },
            Sentinel.COMPLETED,
        ]
        grader = self.make_grader(*samples)

        results = await self.collect_run(grader)

        grades = [item for item in results if isinstance(item, Grade)]
        scores = {item.name: item.value for item in results if isinstance(item, Score)}
        assert [grade.element["correct"] for grade in grades] == [[1], [0]]
        assert scores["overall accuracy (avg over 1)"] == pytest.approx(0.5)
        assert scores["overall accuracy (pass@1)"] == pytest.approx(0.5)
        assert scores["SWE accuracy (avg over 1)"] == pytest.approx(1.0)
        assert scores["DR accuracy (avg over 1)"] == pytest.approx(0.0)
        assert scores["EnvSetup accuracy (avg over 1)"] == pytest.approx(1.0)
        assert scores["Open-ended accuracy (avg over 1)"] == pytest.approx(0.0)
        assert results[-1] == Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_run_yields_rouge_scores_for_summary_tasks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """What: verifies ROUGE tasks emit supplementary scores after accuracy aggregation.
        Executes: `APTBench.run()` with `_compute_rouge` patched to a deterministic score.
        Why: covers summary-task metric emission without importing the optional rouge package.
        """
        monkeypatch.setattr(aptbench_module, "_HAS_ROUGE", True)
        monkeypatch.setattr(aptbench_module, "_compute_rouge", lambda predictions, references: {"rouge-1": 88.0})
        sample = {
            "row": 0,
            "generations": ["summary] rationale"],
            "ground_truth": "summary",
            "difficulty": "deepresearch/summ_ans_en",
            "extractor": "extract_answer_summ_ans",
        }
        grader = self.make_grader(sample, Sentinel.COMPLETED)

        results = await self.collect_run(grader, average_over=[1], pass_at=[1])

        scores = {item.name: item.value for item in results if isinstance(item, Score)}
        assert scores["deepresearch/summ_ans_en rouge-1"] == pytest.approx(88.0)

    @pytest.mark.asyncio
    async def test_run_yields_exception_wrappers_and_nan_scores(self) -> None:
        """What: verifies exceptionWrapper inputs are yielded and all-empty grading produces NaN scores.
        Executes: `APTBench.run()` with an upstream `ExceptionWrapper` and completion sentinel.
        Why: covers the scheduler error passthrough path and empty-score NaN contract.
        """
        wrapper = ExceptionWrapper(exception=RuntimeError("boom"), trace="trace", instance={"row": 0})
        grader = self.make_grader(wrapper, Sentinel.COMPLETED)

        results = await self.collect_run(grader)

        scores = [item for item in results if isinstance(item, Score)]
        assert wrapper in results
        assert math.isnan(scores[0].value)
        assert results[-1] == Sentinel.COMPLETED
