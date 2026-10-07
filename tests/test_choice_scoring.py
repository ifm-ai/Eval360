import math
from unittest.mock import MagicMock

import pytest

import scheduler.choice_scoring_schema as choice_scoring_schema
from scheduler.grader.base import Grade, Score
from scheduler.grader.choice_scoring import ChoiceScoring
from scheduler.choice_scoring_schema import (
    build_choice_scoring_prompts,
    choice_scoring_fields,
    is_choice_scoring_row,
    resolve_ground_truth_index,
    resolve_scoring_completion_labels,
    validate_choice_scoring_row,
)
from scheduler.utils import ExceptionWrapper, Sentinel


class MockEvent:
    parser_type = "noop"


async def make_samples_generator(*samples):
    for sample in samples:
        yield sample
    yield Sentinel.COMPLETED


async def make_incomplete_samples_generator(*samples):
    for sample in samples:
        yield sample


async def empty_existing():
    if False:
        yield


async def collect_run_items(grader, average_over=None, pass_at=None):
    items = []
    async for item in grader.run(
        existing=empty_existing(),
        average_over=average_over or [1],
        pass_at=pass_at or [1],
    ):
        items.append(item)
    return items


def make_grader(*samples):
    return ChoiceScoring(
        samples_generator=make_samples_generator(*samples),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )


def test_choice_scoring_schema_builds_lean_prompt_contract():
    sample = {
        "row": 0,
        "completion_input": "Question: pick one\nAnswer:",
        **choice_scoring_fields(2),
        "ground_truth": "B",
    }

    assert is_choice_scoring_row(sample)
    validate_choice_scoring_row(sample)
    assert build_choice_scoring_prompts(sample) == (
        ["Question: pick one\nAnswer: A", "Question: pick one\nAnswer: B"],
        ["Answer: A", "Answer: B"],
    )
    assert resolve_scoring_completion_labels(sample) == ["A", "B"]
    assert resolve_ground_truth_index(sample) == 1


@pytest.mark.parametrize("value", [True, 1.9, 0.5])
def test_choice_scoring_token_counts_reject_non_integral_values(value):
    sample = {"scoring_completion_n_tokens": [value]}
    resolver = getattr(
        choice_scoring_schema,
        "resolve_scoring_completion_token_counts",
        None,
    )
    assert callable(resolver)

    with pytest.raises(ValueError, match="positive integral"):
        resolver(
            sample,
            1,
            required=True,
        )


@pytest.mark.parametrize("value", [2, 2.0, "2", "2.0"])
def test_choice_scoring_token_counts_normalize_compatible_integers(value):
    sample = {"scoring_completion_n_tokens": [value]}
    resolver = getattr(
        choice_scoring_schema,
        "resolve_scoring_completion_token_counts",
        None,
    )
    assert callable(resolver)

    result = resolver(
        sample,
        1,
        required=True,
    )

    assert result == [2]
    assert type(result[0]) is int


def test_choice_scoring_fields_returns_independent_label_lists():
    fields = choice_scoring_fields(2)

    fields["scoring_completions"].append("C")

    assert fields["scoring_completions"] == ["A", "B", "C"]
    assert fields["scoring_completion_labels"] == ["A", "B"]


def test_choice_scoring_schema_builds_per_choice_prompt_prefixes():
    sample = {
        "row": 0,
        "scoring_mode": "choice_scoring",
        "scoring_prompt_prefixes": [
            "The trophy does not fit in the suitcase because the",
            "The trophy does not fit in the suitcase because the",
        ],
        "scoring_completions": ["trophy is too large", "suitcase is too large"],
        "scoring_completion_labels": ["A", "B"],
        "ground_truth": "A",
    }

    assert is_choice_scoring_row(sample)
    validate_choice_scoring_row(sample)
    assert build_choice_scoring_prompts(sample) == (
        [
            "The trophy does not fit in the suitcase because the trophy is too large",
            "The trophy does not fit in the suitcase because the suitcase is too large",
        ],
        [
            "Answer: trophy is too large",
            "Answer: suitcase is too large",
        ],
    )


def test_choice_scoring_schema_rejects_prompt_prefix_length_mismatch():
    sample = {
        "row": 0,
        "scoring_mode": "choice_scoring",
        "scoring_prompt_prefixes": ["Question A"],
        "scoring_completions": ["A", "B"],
        "ground_truth": "A",
    }

    with pytest.raises(ValueError, match="scoring_prompt_prefixes length mismatch"):
        validate_choice_scoring_row(sample)


def test_choice_scoring_schema_rejects_missing_structured_choices():
    sample = {
        "row": 0,
        "scoring_mode": "choice_scoring",
        "completion_input": "Question:",
        "ground_truth": "A",
    }

    assert not is_choice_scoring_row(sample)
    with pytest.raises(ValueError, match="scoring_completions"):
        validate_choice_scoring_row(sample)


def test_choice_scoring_schema_rejects_unmatched_ground_truth():
    sample = {
        "row": 0,
        "completion_input": "Question: pick one\nAnswer:",
        **choice_scoring_fields(2),
        "ground_truth": "C",
    }

    with pytest.raises(ValueError, match="ground_truth does not match"):
        validate_choice_scoring_row(sample)


def test_choice_scoring_schema_rejects_out_of_range_ground_truth_index():
    sample = {
        "row": 0,
        "completion_input": "Question: pick one\nAnswer:",
        **choice_scoring_fields(2),
        "ground_truth_index": 2,
    }

    with pytest.raises(ValueError, match="ground_truth_index out of range"):
        validate_choice_scoring_row(sample)


def test_choice_scoring_schema_rejects_empty_explicit_labels():
    sample = {
        "row": 0,
        "completion_input": "Question: pick one\nAnswer:",
        **choice_scoring_fields(2),
        "scoring_completion_labels": [],
        "ground_truth": "A",
    }

    with pytest.raises(ValueError, match="length mismatch"):
        validate_choice_scoring_row(sample)


def make_grader_with_generator(samples_generator):
    return ChoiceScoring(
        samples_generator=samples_generator,
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )


async def collect_run_items_with_existing(grader, existing):
    items = []
    async for item in grader.run(
        existing=existing,
        average_over=[1],
        pass_at=[1],
    ):
        items.append(item)
    return items


def make_precomputed_sample(row=0):
    return {
        "row": row,
        "scoring_completions": ["A", "B"],
        "scoring_completion_n_tokens": [1, 1],
        "scoring_completion_n_chars": [1, 1],
        "ground_truth_index": 0,
        "choice_nll": [0.1, 0.9],
        "choice_nll_completion": [0.0, 0.0],
    }

def test_sum_suffix_token_logprobs_from_attr():
    logprobs = MagicMock()
    logprobs.token_logprobs = [None, -1.5, -0.25]
    logprobs.text_offset = None
    assert ChoiceScoring._sum_suffix_token_logprobs(logprobs, 2) == pytest.approx(-1.75)


def test_sum_suffix_token_logprobs_from_objects():
    tok0 = MagicMock()
    tok0.logprob = None
    tok1 = MagicMock()
    tok1.logprob = -0.5
    tok2 = MagicMock()
    tok2.logprob = -0.125
    logprobs = MagicMock()
    logprobs.token_logprobs = [tok0, tok1, tok2]
    logprobs.text_offset = None
    assert ChoiceScoring._sum_suffix_token_logprobs(logprobs, 2) == pytest.approx(-0.625)


def test_sum_suffix_token_logprobs_prefers_text_offsets():
    logprobs = {
        "text_offset": [0, 6, 7],
        "token_logprobs": [None, -0.5, -0.125],
    }
    assert ChoiceScoring._sum_suffix_token_logprobs(
        logprobs,
        3,
        suffix_start_chars=7,
        prompt_text="Answer: pesticides",
    ) == pytest.approx(-0.125)


def test_sum_suffix_token_logprobs_ignores_generated_token_after_echoed_prompt():
    prompt = "Answer: A"
    logprobs = {
        "text_offset": [0, len("Answer:"), len(prompt)],
        "token_logprobs": [None, -0.25, -99.0],
    }

    total, count = ChoiceScoring._sum_suffix_token_logprobs_with_count(
        logprobs,
        None,
        suffix_start_chars=len("Answer:"),
        prompt_text=prompt,
    )

    assert total == pytest.approx(-0.25)
    assert count == 1


def test_sum_suffix_token_logprobs_rejects_generated_only_backend_payload():
    logprobs = {
        "text_offset": [],
        "token_logprobs": [-1.4108354],
        "tokens": ["."],
    }

    with pytest.raises(ValueError, match="needs text_offset"):
        ChoiceScoring._sum_suffix_token_logprobs_with_count(
            logprobs,
            None,
            suffix_start_chars=len("Answer:"),
            prompt_text="Answer: A",
        )


def test_compute_choice_metrics_from_raw_logprobs():
    sample = {
        "row": 0,
        "scoring_mode": "choice_scoring",
        "scoring_prompt_prefix": "Question: test\nAnswer:",
        "scoring_completion_prefix": "Answer:",
        "scoring_completions": ["A", "B"],
        "scoring_completion_n_tokens": [1, 1],
        "scoring_completion_n_chars": [1, 1],
        "scoring_completion_labels": ["A", "B"],
        "ground_truth_index": 0,
        "choice_scoring_full_logprobs": [
            {"token_logprobs": [None, -0.3]},
            {"token_logprobs": [None, -0.7]},
        ],
        "choice_scoring_completion_logprobs": [
            {"token_logprobs": [None, -0.2]},
            {"token_logprobs": [None, -0.4]},
        ],
    }

    result = ChoiceScoring._compute_choice_metrics(sample)

    assert result["choice_nll"] == pytest.approx([0.3, 0.7])
    assert result["choice_nll_completion"] == pytest.approx([0.2, 0.4])
    assert result["picked"] == ["A"]
    assert result["picked_compl"] == ["A"]
    assert result["acc"] == pytest.approx(100.0)


def test_compute_choice_metrics_uses_text_offsets_when_token_counts_drift():
    sample = {
        "row": 0,
        "scoring_mode": "choice_scoring",
        "scoring_prompt_prefix": "Question:",
        "scoring_completion_prefix": "Answer:",
        "scoring_completions": ["pesticides"],
        "scoring_completion_n_tokens": [3],
        "scoring_completion_n_chars": [10],
        "scoring_completion_labels": ["A"],
        "ground_truth_index": 0,
        "choice_scoring_full_logprobs": [
            {
                "text_offset": [0, 8, 9],
                "token_logprobs": [None, -0.3, -0.7],
            }
        ],
        "choice_scoring_completion_logprobs": [
            {
                "text_offset": [0, 6, 7],
                "token_logprobs": [None, -0.2, -0.4],
            }
        ],
    }

    result = ChoiceScoring._compute_choice_metrics(sample)

    assert result["choice_nll"] == pytest.approx([0.7])
    assert result["choice_nll_completion"] == pytest.approx([0.4])


def test_compute_choice_metrics_derives_optional_scoring_fields():
    sample = {
        "row": 0,
        "scoring_mode": "choice_scoring",
        "completion_input": "Question:",
        "scoring_completions": ["air friction", "gravity"],
        "scoring_completion_labels": ["A", "B"],
        "ground_truth": "B",
        "choice_scoring_full_logprobs": [
            {
                "text_offset": [0, 9],
                "token_logprobs": [None, -0.8],
            },
            {
                "text_offset": [0, 9],
                "token_logprobs": [None, -0.2],
            },
        ],
        "choice_scoring_completion_logprobs": [
            {
                "text_offset": [0, 7],
                "token_logprobs": [None, -0.1],
            },
            {
                "text_offset": [0, 7],
                "token_logprobs": [None, -0.1],
            },
        ],
    }

    result = ChoiceScoring._compute_choice_metrics(sample)

    assert result["choice_nll"] == pytest.approx([0.8, 0.2])
    assert result["scoring_completion_n_tokens"] == pytest.approx([1.0, 1.0])
    assert result["scoring_completion_n_chars"] == pytest.approx([12.0, 7.0])
    assert result["ground_truth_index"] == 1
    assert result["picked"] == ["B"]
    assert result["correct"] == [True]


def test_compute_choice_metrics_accepts_per_choice_prompt_prefixes():
    sample = {
        "row": 0,
        "scoring_mode": "choice_scoring",
        "scoring_prompt_prefixes": ["It was the", "It was the"],
        "scoring_completions": ["trophy", "suitcase"],
        "scoring_completion_labels": ["A", "B"],
        "ground_truth": "A",
        "choice_scoring_full_logprobs": [
            {"text_offset": [0, 10], "token_logprobs": [None, -0.1]},
            {"text_offset": [0, 10], "token_logprobs": [None, -0.9]},
        ],
        "choice_scoring_completion_logprobs": [
            {"text_offset": [0, 7], "token_logprobs": [None, -0.2]},
            {"text_offset": [0, 7], "token_logprobs": [None, -0.8]},
        ],
    }

    result = ChoiceScoring._compute_choice_metrics(sample)

    assert result["choice_nll"] == pytest.approx([0.1, 0.9])
    assert result["ground_truth_index"] == 0
    assert result["picked"] == ["A"]


@pytest.mark.asyncio
async def test_run_emits_grouped_scores():
    sample1 = {
        "row": 0,
        "choice_group": "g1",
        "scoring_completions": ["A", "B"],
        "scoring_completion_n_tokens": [1, 1],
        "scoring_completion_n_chars": [1, 1],
        "scoring_completion_labels": ["A", "B"],
        "ground_truth_index": 0,
        "choice_nll": [0.1, 0.9],
        "choice_nll_completion": [0.0, 0.0],
    }
    sample2 = {
        "row": 1,
        "choice_group": "g1",
        "scoring_completions": ["A", "B"],
        "scoring_completion_n_tokens": [1, 1],
        "scoring_completion_n_chars": [1, 1],
        "scoring_completion_labels": ["A", "B"],
        "ground_truth_index": 0,
        "choice_nll": [0.9, 0.1],
        "choice_nll_completion": [0.0, 0.0],
    }
    sample3 = {
        "row": 2,
        "choice_group": "g2",
        "scoring_completions": ["A", "B"],
        "scoring_completion_n_tokens": [1, 1],
        "scoring_completion_n_chars": [1, 1],
        "scoring_completion_labels": ["A", "B"],
        "ground_truth_index": 0,
        "choice_nll": [0.1, 0.9],
        "choice_nll_completion": [0.0, 0.0],
    }

    items = await collect_run_items(make_grader(sample1, sample2, sample3))
    grades = [item.element for item in items if isinstance(item, Grade)]
    scores = {
        item.name: item.value
        for item in items
        if isinstance(item, Score)
    }

    assert len(grades) == 3
    assert scores["macro_avg/acc"] == pytest.approx(75.0)
    assert scores["micro_avg/acc"] == pytest.approx(66.66666666666667)


@pytest.mark.asyncio
async def test_run_streams_existing_choice_scoring_grades_without_regrading_rows():
    async def existing():
        yield {
            "row": 0,
            "choice_group": "g1",
            "acc": 100.0,
            "acc_char": 100.0,
            "acc_token": 100.0,
            "acc_compl": 100.0,
            "nll": 0.1,
            "nll_char": 0.1,
            "nll_token": 0.1,
            "nll_compl": 0.1,
            "choice_scoring_full_logprobs": ["x" * 100_000],
            "choice_scoring_completion_logprobs": ["x" * 100_000],
        }

    skipped = {
        "row": 0,
        "choice_group": "g1",
        "scoring_completions": ["A", "B"],
        "scoring_completion_n_tokens": [1, 1],
        "scoring_completion_n_chars": [1, 1],
        "scoring_completion_labels": ["A", "B"],
        "ground_truth_index": 0,
        "choice_nll": [0.9, 0.1],
        "choice_nll_completion": [0.0, 0.0],
    }
    new = dict(skipped, row=1)

    items = []
    async for item in make_grader(skipped, new).run(
        existing=existing(),
        average_over=[1],
        pass_at=[1],
    ):
        items.append(item)

    grades = [item.element for item in items if isinstance(item, Grade)]
    scores = {item.name: item.value for item in items if isinstance(item, Score)}

    assert [grade["row"] for grade in grades] == [1]
    assert scores["macro_avg/acc"] == pytest.approx(50.0)
    assert scores["micro_avg/acc"] == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_run_handles_eval360_input_too_long():
    sample = {
        "row": 0,
        "eval360_input_too_long": True,
        "scoring_completions": ["A", "B"],
        "scoring_completion_n_tokens": [1, 1],
        "scoring_completion_n_chars": [1, 1],
        "ground_truth_index": 0,
    }

    items = await collect_run_items(make_grader(sample))
    grades = [item.element for item in items if isinstance(item, Grade)]
    scores = {item.name: item.value for item in items if isinstance(item, Score)}

    assert grades[0]["acc"] == pytest.approx(0.0)
    assert grades[0]["nll"] == float("inf")
    assert scores["acc"] == pytest.approx(0.0)
    assert scores["nll"] == float("inf")


@pytest.mark.asyncio
async def test_run_wraps_malformed_raw_payload_as_exception():
    sample = {
        "row": 0,
        "scoring_mode": "choice_scoring",
        "scoring_prompt_prefix": "Question:",
        "scoring_completion_prefix": "Answer:",
        "scoring_completions": ["A"],
        "scoring_completion_n_tokens": [1],
        "scoring_completion_n_chars": [1],
        "ground_truth_index": 0,
        "choice_scoring_full_logprobs": [{"token_logprobs": [None, -0.3]}],
    }

    items = await collect_run_items(make_grader(sample))
    wrappers = [item for item in items if isinstance(item, ExceptionWrapper)]
    scores = [item for item in items if isinstance(item, Score)]

    assert len(wrappers) == 1
    assert "missing raw logprob payloads" in wrappers[0].trace
    assert scores
    assert all(math.isnan(item.value) for item in scores)


class TestChoiceScoringScalarHelpers:
    """What: groups coverage for small scalar helper functions.
    Executes: ChoiceScoring's numeric reducers before larger metric aggregation tests.
    Why: these reducers decide the winning choice and no-data aggregate values used by every emitted metric.
    """

    def test_argmin_casts_values_and_returns_lowest_index(self):
        """What: verifies _argmin casts comparable values and reports the lowest value.
        Executes: ChoiceScoring._argmin() with stringified numeric scores and a lowest middle element.
        Why: covers core score-selection behavior for rows whose serialized metrics arrive as strings.
        """
        idx, value = ChoiceScoring._argmin(["3.5", "1.25", "2.0"])

        assert idx == 1
        assert value == pytest.approx(1.25)

    def test_argmin_rejects_empty_values(self):
        """What: verifies _argmin raises a useful error for empty input.
        Executes: ChoiceScoring._argmin() with no candidate values.
        Why: exercises the defensive guard that prevents empty metric lists from silently picking index 0.
        """
        with pytest.raises(ValueError, match="empty list"):
            ChoiceScoring._argmin([])

    def test_mean_or_nan_returns_mean_or_nan(self):
        """What: verifies _mean_or_nan returns an average for values and NaN for no values.
        Executes: ChoiceScoring._mean_or_nan() for populated and empty iterables.
        Why: covers the aggregate-score helper's normal mean path and its no-data sentinel value.
        """
        assert ChoiceScoring._mean_or_nan([1.0, 2.0, 6.0]) == pytest.approx(3.0)
        assert math.isnan(ChoiceScoring._mean_or_nan([]))


class TestChoiceScoringLogprobErrors:
    """What: groups coverage for suffix logprob success and error paths.
    Executes: suffix-token logprob extraction before full choice NLL metrics are computed.
    Why: isolates payload-shape handling from the larger run() stream tests.
    """

    def test_sum_suffix_token_logprobs_accepts_dict_token_values(self):
        """What: verifies dict token logprobs are cast before suffix totals are computed.
        Executes: ChoiceScoring._sum_suffix_token_logprobs() on OpenAI-style dict token records.
        Why: covers core compatibility with serialized logprob payloads that store values under "logprob".
        """
        logprobs = {"token_logprobs": [{"logprob": "-0.4"}, {"logprob": "-0.1"}]}

        assert ChoiceScoring._sum_suffix_token_logprobs(logprobs, 2) == pytest.approx(-0.5)

    @pytest.mark.parametrize(
        "choice_logprobs,suffix_n_tokens,kwargs,match",
        [
            (None, 1, {}, "token_logprobs missing"),
            ({"token_logprobs": [-0.1]}, 0, {}, "suffix_n_tokens must be positive"),
            ({"token_logprobs": [-0.1]}, 2, {}, "shorter than the requested"),
            ({"token_logprobs": [None]}, 1, {}, "missing token logprob"),
            (
                {"text_offset": [0], "token_logprobs": [None]},
                1,
                {"suffix_start_chars": 0, "prompt_text": "X"},
                "missing token logprob",
            ),
            (
                {"text_offset": [0], "token_logprobs": [-0.1]},
                1,
                {"suffix_start_chars": 2, "prompt_text": "X"},
                "did not include any completion-suffix tokens",
            ),
        ],
    )
    def test_sum_suffix_token_logprobs_rejects_bad_suffix_payloads(
        self,
        choice_logprobs,
        suffix_n_tokens,
        kwargs,
        match,
    ):
        """What: verifies malformed suffix payloads raise targeted ValueError messages.
        Executes: ChoiceScoring._sum_suffix_token_logprobs() across missing, short, null, and offset-mismatched payloads.
        Why: exercises the validation branches that turn bad raw logprob data into actionable grader errors.
        """
        with pytest.raises(ValueError, match=match):
            ChoiceScoring._sum_suffix_token_logprobs(
                choice_logprobs,
                suffix_n_tokens,
                **kwargs,
            )


class TestChoiceScoringNllExtraction:
    """What: groups coverage for prompt selection and NLL extraction paths.
    Executes: prompt metadata selection and NLL extraction before async grading emits scores.
    Why: choice rows can arrive as metadata-expanded or precomputed data, and both forms must normalize before scoring.
    """

    def test_choice_prompts_prefer_metadata_when_complete(self):
        """What: verifies metadata prompts take precedence over synthetic fallback prompts.
        Executes: ChoiceScoring._choice_prompts() when choice_scoring_metadata supplies both prompt lists.
        Why: covers the metadata-precedence branch used by pre-expanded datasets that should bypass synthetic prompt construction.
        """
        sample = {
            "choice_scoring_metadata": {
                "full_prompts": ["metadata full"],
                "completion_prompts": ["metadata completion"],
            },
            "scoring_completions": ["fallback"],
            "scoring_prompt_prefix": "Synthetic:",
            "scoring_completion_prefix": "Answer:",
        }

        full_prompts, completion_prompts = ChoiceScoring._choice_prompts(sample)

        assert full_prompts == ["metadata full"]
        assert completion_prompts == ["metadata completion"]

    def test_precomputed_nll_values_are_cast_before_metrics(self):
        """What: verifies precomputed NLL strings are cast to floats during metric computation.
        Executes: ChoiceScoring._compute_choice_metrics() with string NLLs, token counts, and ground-truth index.
        Why: covers core precomputed-score ingestion where JSONL values may be serialized as strings.
        """
        sample = {
            "row": 0,
            "scoring_completions": ["A", "B"],
            "scoring_completion_n_tokens": ["2", "1"],
            "scoring_completion_n_chars": ["4", "2"],
            "ground_truth_index": "1",
            "choice_nll": ["2.0", "0.5"],
            "choice_nll_completion": ["0.25", "0.125"],
        }

        result = ChoiceScoring._compute_choice_metrics(sample)

        assert result["choice_nll"] == pytest.approx([2.0, 0.5])
        assert result["choice_nll_completion"] == pytest.approx([0.25, 0.125])
        assert result["picked"] == ["B"]
        assert result["acc"] == pytest.approx(100.0)

    @pytest.mark.parametrize(
        "sample_update,match",
        [
            (
                {"choice_scoring_full_logprobs": [{"token_logprobs": [-0.1]}]},
                "full-logprobs length mismatch",
            ),
            (
                {"choice_scoring_completion_logprobs": [{"token_logprobs": [-0.1]}]},
                "completion-logprobs length mismatch",
            ),
        ],
    )
    def test_raw_logprob_length_mismatches_raise_value_error(self, sample_update, match):
        """What: verifies raw logprob payloads must align with completion token counts.
        Executes: ChoiceScoring._extract_choice_nlls() with one full or completion logprob list shortened.
        Why: exercises the length-mismatch guards that prevent pairing a choice with the wrong token logprobs.
        """
        sample = {
            "scoring_prompt_prefix": "Question:",
            "scoring_completion_prefix": "Answer:",
            "scoring_completions": ["A", "B"],
            "scoring_completion_n_tokens": [1, 1],
            "choice_scoring_full_logprobs": [
                {"token_logprobs": [-0.1]},
                {"token_logprobs": [-0.2]},
            ],
            "choice_scoring_completion_logprobs": [
                {"token_logprobs": [-0.1]},
                {"token_logprobs": [-0.2]},
            ],
        }
        sample.update(sample_update)

        with pytest.raises(ValueError, match=match):
            ChoiceScoring._extract_choice_nlls(sample)


class TestChoiceScoringRunEdgeCases:
    """What: groups coverage for run() stream and existing-row edge cases.
    Executes: ChoiceScoring.run() and grade_sample() paths that sit around metric computation.
    Why: stream completion, resume skips, and upstream failures determine whether aggregate scores are trustworthy.
    """

    @pytest.mark.asyncio
    async def test_run_skips_rows_already_seen_in_existing_results(self):
        """What: verifies rows present in existing results are not graded a second time.
        Executes: ChoiceScoring.run() with an existing result stream containing the same row id as the sample.
        Why: exercises resume/idempotency behavior so reruns aggregate prior scores without duplicate Grade output.
        """

        async def existing():
            yield {
                "row": 0,
                "choice_group": "existing",
                "acc": 25.0,
                "acc_char": 25.0,
                "acc_token": 25.0,
                "acc_compl": 25.0,
                "nll": 2.0,
                "nll_char": 2.0,
                "nll_token": 2.0,
                "nll_compl": 2.0,
            }

        items = await collect_run_items_with_existing(
            make_grader(make_precomputed_sample(row=0)),
            existing(),
        )
        grades = [item for item in items if isinstance(item, Grade)]
        scores = {item.name: item.value for item in items if isinstance(item, Score)}

        assert grades == []
        assert scores["macro_avg/acc"] == pytest.approx(25.0)
        assert scores["micro_avg/acc"] == pytest.approx(25.0)
        assert items[-1] == Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_run_yields_exception_wrapper_samples_unchanged(self):
        """What: verifies exceptionWrapper samples pass through without metric computation.
        Executes: ChoiceScoring.run() when the sample stream yields an upstream ExceptionWrapper.
        Why: covers the pass-through error contract that preserves upstream failures instead of masking them with scoring work.
        """
        wrapper = ExceptionWrapper(
            exception=RuntimeError("upstream failed"),
            trace="upstream trace",
            instance={"row": 3},
        )

        items = await collect_run_items(make_grader(wrapper))

        assert items[0] is wrapper
        assert all(not isinstance(item, Grade) for item in items)
        assert items[-1] == Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_run_returns_without_scores_for_incomplete_stream(self):
        """What: verifies an input stream without Sentinel.COMPLETED emits grades but no scores.
        Executes: ChoiceScoring.run() against a generator that stops before yielding Sentinel.COMPLETED.
        Why: exercises the incomplete-stream corner case where per-row grades exist but final aggregate scores must not be emitted.
        """
        grader = make_grader_with_generator(
            make_incomplete_samples_generator(make_precomputed_sample(row=1))
        )

        items = await collect_run_items(grader)

        assert [item for item in items if isinstance(item, Grade)]
        assert [item for item in items if isinstance(item, Score)] == []
        assert Sentinel.COMPLETED not in items

    @pytest.mark.asyncio
    async def test_grade_sample_declares_inline_only(self):
        """What: verifies grade_sample documents that choice scoring happens in run().
        Executes: ChoiceScoring.grade_sample() directly instead of through the async run() stream.
        Why: covers the explicit inline-only contract so callers do not accidentally bypass run()'s grouped metric handling.
        """
        with pytest.raises(NotImplementedError, match="grades inline"):
            await make_grader().grade_sample({})
