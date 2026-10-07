"""Tests for the IFEval grader and evaluation helper library."""

import copy
import json
import math

import pytest

from scheduler.grader.base import Grade, Score
from scheduler.grader.ifeval import IFEval
from scheduler.grader.ifeval_lib.evaluation_lib import (
    InputExample,
    OutputExample,
    print_report,
    read_prompt_list,
    read_prompt_to_response_dict,
    test_instruction_following_loose as eval_loose,
    test_instruction_following_strict as eval_strict,
    write_outputs,
)
from scheduler.grader.ifeval_lib.instructions_registry import (
    INSTRUCTION_CONFLICTS,
    conflict_make,
)
from scheduler.grader.ifeval_lib import instructions_util
from scheduler.utils import ExceptionWrapper, Sentinel


SAMPLE_PASS = {
    "row": 0,
    "ground_truth": {
        "instruction_id_list": [
            "punctuation:no_comma",
            "length_constraints:number_words",
        ],
        "kwargs": [{}, {"relation": "at least", "num_words": 5}],
    },
    "completion_input": "Write a short sentence without commas.",
    "chat_input": [
        {"role": "user", "content": "Write a short sentence without commas."},
    ],
    "generations": [
        "The quick brown fox jumps over the lazy dog today.",
    ],
    "parsed_generations": [None],
}

SAMPLE_FAIL_STRICT = {
    "row": 1,
    "ground_truth": {
        "instruction_id_list": [
            "punctuation:no_comma",
            "length_constraints:number_words",
        ],
        "kwargs": [{}, {"relation": "at least", "num_words": 5}],
    },
    "completion_input": "Write a short sentence without commas.",
    "chat_input": [
        {"role": "user", "content": "Write a short sentence without commas."},
    ],
    "generations": [
        "The quick, brown fox jumps over the lazy dog today.",
    ],
    "parsed_generations": [None],
}

SAMPLE_FAIL_WORD_COUNT = {
    "row": 2,
    "ground_truth": {
        "instruction_id_list": ["length_constraints:number_words"],
        "kwargs": [{"relation": "at least", "num_words": 100}],
    },
    "completion_input": "Write a long essay.",
    "chat_input": [
        {"role": "user", "content": "Write a long essay."},
    ],
    "generations": [
        "This is short.",
    ],
    "parsed_generations": [None],
}


class MockEvent:
    """Minimal event object that selects the passthrough parser."""

    parser_type = "noop"


async def make_samples_generator(*samples):
    for sample in samples:
        yield copy.deepcopy(sample)
    yield Sentinel.COMPLETED


async def make_incomplete_samples_generator(*samples):
    for sample in samples:
        yield copy.deepcopy(sample)


async def existing_results(*results):
    for result in results:
        yield copy.deepcopy(result)


def make_grader(*samples):
    return IFEval(
        samples_generator=make_samples_generator(*samples),
        event_manager=None,
        job_manager=None,
        event=MockEvent(),
    )


def make_input(sample):
    gt = sample["ground_truth"]
    return InputExample(
        key=sample["row"],
        instruction_id_list=gt["instruction_id_list"],
        prompt=sample["completion_input"],
        kwargs=gt["kwargs"],
    )


async def collect_run_items(grader, existing=()):
    items = []
    async for item in grader.run(
        existing=existing_results(*existing),
        average_over=[1],
        pass_at=[1],
    ):
        items.append(item)
    return items


class TestIFEvalGrader:
    """What: verifies IFEval sample grading and async run aggregation.
    Executes: `IFEval.grade_sample()` and `IFEval.run()` with local sample streams.
    Why: covers the scheduler-facing contract for grades, scores, errors, and sentinels.
    """

    @pytest.mark.asyncio
    async def test_grade_sample_scores_real_ifeval_result(self):
        """What: verifies grade_sample adds strict, loose, and correct scores to a pass.
        Executes: `IFEval.grade_sample()` with `SAMPLE_PASS` through both evaluators.
        Why: covers the core path that converts one benchmark row into a grade payload.
        """
        sample = copy.deepcopy(SAMPLE_PASS)
        grader = make_grader()

        result = await grader.grade_sample(sample)

        assert result is not sample
        assert result["strict_prompt_follow"] is True
        assert result["strict_inst_follow"] == [True, True]
        assert result["loose_prompt_follow"] is True
        assert result["loose_inst_follow"] == [True, True]
        assert result["correct"] == [1]

    @pytest.mark.asyncio
    async def test_grade_sample_prefers_parsed_generation_when_available(self):
        """What: verifies grade_sample uses parsed text instead of the raw generation.
        Executes: the parsed-generation selection inside `IFEval.grade_sample()`.
        Why: ensures parser output, not raw model text, drives IFEval scoring when present.
        """
        sample = copy.deepcopy(SAMPLE_FAIL_STRICT)
        sample["parsed_generations"] = [
            "The quick brown fox jumps over the lazy dog today."
        ]
        grader = make_grader()

        result = await grader.grade_sample(sample)

        assert result["strict_prompt_follow"] is True
        assert result["correct"] == [1]

    @pytest.mark.asyncio
    async def test_grade_sample_returns_completed_sentinel(self):
        """What: verifies grade_sample passes the completed sentinel through unchanged.
        Executes: the sentinel guard at the top of `IFEval.grade_sample()`.
        Why: the stream terminator is a control signal, not an IFEval row, and must bypass instruction decoding.
        """
        grader = make_grader()

        result = await grader.grade_sample(Sentinel.COMPLETED)

        assert result is Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_run_emits_grades_scores_and_completion(self):
        """What: verifies run grades new samples and emits aggregate IFEval scores.
        Executes: `IFEval.run()` over two generated samples and final `Score` records.
        Why: covers the main scheduler output path for per-row grades and aggregate metrics.
        """
        grader = make_grader(SAMPLE_PASS, SAMPLE_FAIL_STRICT)

        items = await collect_run_items(grader)

        grades = [item.element for item in items if isinstance(item, Grade)]
        scores = {item.name: item.value for item in items if isinstance(item, Score)}
        assert [grade["correct"] for grade in grades] == [[1], [0]]
        assert scores["strict_prompt_accuracy"] == pytest.approx(0.5)
        assert scores["strict_instruction_accuracy"] == pytest.approx(0.75)
        assert scores["loose_prompt_accuracy"] == pytest.approx(0.5)
        assert scores["loose_instruction_accuracy"] == pytest.approx(0.75)
        assert items[-1] is Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_run_aggregates_existing_rows_before_new_samples(self):
        """What: verifies run skips already graded leading rows but includes their scores.
        Executes: `IFEval.run()` preloading `existing` rows before grading new samples.
        Why: covers resume behavior where prior rows still contribute to aggregate metrics.
        """
        existing = {
            **copy.deepcopy(SAMPLE_PASS),
            "strict_prompt_follow": True,
            "strict_inst_follow": [True, True],
            "loose_prompt_follow": True,
            "loose_inst_follow": [True, True],
            "correct": [1],
        }
        grader = make_grader(SAMPLE_PASS, SAMPLE_FAIL_STRICT)

        items = await collect_run_items(grader, existing=(existing,))

        grades = [item.element for item in items if isinstance(item, Grade)]
        scores = {item.name: item.value for item in items if isinstance(item, Score)}
        assert len(grades) == 1
        assert grades[0]["row"] == SAMPLE_FAIL_STRICT["row"]
        assert scores["strict_prompt_accuracy"] == pytest.approx(0.5)
        assert scores["strict_instruction_accuracy"] == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_run_emits_nan_scores_without_any_graded_samples(self):
        """What: verifies run emits nan aggregate scores when no graded rows exist.
        Executes: `IFEval.run()` after an empty stream reaches `Sentinel.COMPLETED`.
        Why: documents the no-data metric contract without forcing zero or crashing.
        """
        grader = make_grader()

        items = await collect_run_items(grader)

        scores = [item for item in items if isinstance(item, Score)]
        assert {score.name for score in scores} == {
            "strict_prompt_accuracy",
            "strict_instruction_accuracy",
            "loose_prompt_accuracy",
            "loose_instruction_accuracy",
        }
        assert all(math.isnan(score.value) for score in scores)
        assert items[-1] is Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_run_forwards_exception_wrappers_from_grading_stream(self):
        """What: verifies run yields wrapped sample errors and still completes scoring.
        Executes: the `ExceptionWrapper` branch in `IFEval.run()`.
        Why: ensures grading errors propagate while later completion still emits metrics.
        """
        wrapped = ExceptionWrapper(
            exception=ValueError("bad sample"),
            trace="traceback",
            instance={"row": 99},
        )
        grader = make_grader(wrapped)

        items = await collect_run_items(grader)

        wrappers = [item for item in items if isinstance(item, ExceptionWrapper)]
        scores = [item for item in items if isinstance(item, Score)]
        assert len(wrappers) == 1
        assert wrappers[0].trace == "traceback"
        assert str(wrappers[0].exception) == "bad sample"
        assert wrappers[0].instance == {"row": 99}
        assert all(math.isnan(score.value) for score in scores)
        assert items[-1] is Sentinel.COMPLETED

    @pytest.mark.asyncio
    async def test_run_returns_without_scores_when_stream_does_not_complete(self):
        """What: verifies run withholds aggregate scores when the sample stream lacks sentinel.
        Executes: the early-return path in `IFEval.run()` before completion scoring.
        Why: prevents partial sample streams from publishing final aggregate metrics.
        """
        grader = IFEval(
            samples_generator=make_incomplete_samples_generator(SAMPLE_PASS),
            event_manager=None,
            job_manager=None,
            event=MockEvent(),
        )

        items = await collect_run_items(grader)

        assert [type(item) for item in items] == [Grade]
        assert items[0].element["correct"] == [1]

    @pytest.mark.asyncio
    async def test_grade_sample_empty_response_fails_all_instructions(self):
        """Empty responses fail every IFEval instruction."""
        sample = copy.deepcopy(SAMPLE_PASS)
        sample["generations"] = [""]
        sample["parsed_generations"] = [None]
        grader = make_grader()

        result = await grader.grade_sample(sample)

        assert result["strict_prompt_follow"] is False
        assert result["strict_inst_follow"] == [False, False]
        assert result["loose_inst_follow"] == [False, False]
        assert result["correct"] == [0]


class TestIFEvalEvaluationHelpers:
    """What: verifies strict, loose, JSONL, and reporting helper behavior.
    Executes: evaluator helpers, JSONL readers and writer, and `print_report()`.
    Why: covers the standalone IFEval library contracts used around grader scoring.
    """

    def test_strict_all_pass(self):
        """Response follows all instructions strictly."""
        inp = make_input(SAMPLE_PASS)
        prompt_to_response = {inp.prompt: SAMPLE_PASS["generations"][0]}

        out = eval_strict(inp, prompt_to_response)

        assert out.follow_all_instructions is True
        assert out.follow_instruction_list == [True, True]

    def test_strict_comma_fail(self):
        """Response contains a comma, so strict no-comma validation fails."""
        inp = make_input(SAMPLE_FAIL_STRICT)
        prompt_to_response = {inp.prompt: SAMPLE_FAIL_STRICT["generations"][0]}

        out = eval_strict(inp, prompt_to_response)

        assert out.follow_all_instructions is False
        assert out.follow_instruction_list == [False, True]

    def test_strict_word_count_fail(self):
        """Response is too short for the strict word-count requirement."""
        inp = make_input(SAMPLE_FAIL_WORD_COUNT)
        prompt_to_response = {
            inp.prompt: SAMPLE_FAIL_WORD_COUNT["generations"][0]
        }

        out = eval_strict(inp, prompt_to_response)

        assert out.follow_all_instructions is False
        assert out.follow_instruction_list == [False]

    def test_loose_is_at_least_as_lenient_as_strict(self):
        """Loose evaluation passes whenever strict evaluation passes."""
        inp = make_input(SAMPLE_PASS)
        prompt_to_response = {inp.prompt: SAMPLE_PASS["generations"][0]}

        strict_out = eval_strict(inp, prompt_to_response)
        loose_out = eval_loose(inp, prompt_to_response)

        assert strict_out.follow_all_instructions is True
        assert loose_out.follow_all_instructions is True

    def test_loose_comma_fail(self):
        """Loose evaluation still fails when no generated variant removes the comma."""
        inp = make_input(SAMPLE_FAIL_STRICT)
        prompt_to_response = {inp.prompt: SAMPLE_FAIL_STRICT["generations"][0]}

        out = eval_loose(inp, prompt_to_response)

        assert out.follow_all_instructions is False
        assert out.follow_instruction_list == [False, True]

    @pytest.mark.parametrize(
        ("instruction_id", "kwargs", "response"),
        [
            (
                "punctuation:no_comma",
                {},
                "This removed line has, a comma\nThis clean line passes",
            ),
            (
                "punctuation:no_comma",
                {},
                "This clean line passes\nThis removed line has, a comma",
            ),
            (
                "punctuation:no_comma",
                {},
                "This removed line has, a comma\nThis clean line passes\n"
                "This other removed line has, a comma",
            ),
            ("startend:quotation", {}, '*"quoted answer"*'),
            ("startend:quotation", {}, 'noise\n*"quoted answer"*'),
            ("startend:quotation", {}, '*"quoted answer"*\nnoise'),
            ("startend:quotation", {}, 'noise\n*"quoted answer"*\nnoise'),
        ],
    )
    def test_loose_variants_can_pass_when_strict_fails(
        self, instruction_id, kwargs, response
    ):
        """What: verifies loose evaluation tries line-trimmed and asterisk-stripped variants.
        Executes: `test_instruction_following_loose()` over wrapper-removal variants.
        Why: covers compatibility mode for benchmark answers wrapped by extra formatting.
        """
        inp = InputExample(
            key=10,
            instruction_id_list=[instruction_id],
            prompt="Follow the instruction.",
            kwargs=[kwargs],
        )
        prompt_to_response = {inp.prompt: response}

        strict_out = eval_strict(inp, prompt_to_response)
        loose_out = eval_loose(inp, prompt_to_response)

        assert strict_out.follow_instruction_list == [False]
        assert loose_out.follow_instruction_list == [True]

    def test_jsonl_helpers_round_trip_prompts_outputs_and_responses(self, tmp_path):
        """What: verifies JSONL helpers read prompts, responses, and output rows.
        Executes: `read_prompt_list()`, `read_prompt_to_response_dict()`, and `write_outputs()`.
        Why: covers the file format boundary used by standalone IFEval evaluation tools.
        """
        prompts_path = tmp_path / "prompts.jsonl"
        responses_path = tmp_path / "responses.jsonl"
        outputs_path = tmp_path / "outputs.jsonl"
        prompts_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "key": 1,
                            "instruction_id_list": ["punctuation:no_comma"],
                            "prompt": "Prompt one",
                            "kwargs": [{}],
                        }
                    ),
                    json.dumps(
                        {
                            "key": 2,
                            "instruction_id_list": ["startend:quotation"],
                            "prompt": "Prompt two",
                            "kwargs": [{}],
                        }
                    ),
                ]
            )
            + "\n"
        )
        responses_path.write_text(
            json.dumps({"prompt": "Prompt one", "response": "No comma"}) + "\n"
        )
        outputs = [
            OutputExample(
                instruction_id_list=["punctuation:no_comma"],
                prompt="Prompt one",
                response="No comma",
                follow_all_instructions=True,
                follow_instruction_list=[True],
            )
        ]

        prompts = read_prompt_list(prompts_path)
        prompt_to_response = read_prompt_to_response_dict(responses_path)
        write_outputs(outputs_path, outputs)

        assert [prompt.key for prompt in prompts] == [1, 2]
        assert prompts[1].instruction_id_list == ["startend:quotation"]
        assert prompt_to_response == {"Prompt one": "No comma"}
        assert json.loads(outputs_path.read_text().strip()) == {
            "follow_all_instructions": True,
            "follow_instruction_list": [True],
            "instruction_id_list": ["punctuation:no_comma"],
            "prompt": "Prompt one",
            "response": "No comma",
        }

    def test_write_outputs_rejects_empty_output_list(self, tmp_path):
        """What: verifies write_outputs rejects empty output lists before writing JSONL.
        Executes: the non-empty assertion in `write_outputs()`.
        Why: preserves the helper contract that an output JSONL file must contain rows.
        """
        with pytest.raises(AssertionError):
            write_outputs(tmp_path / "empty.jsonl", [])

    def test_print_report_outputs_prompt_instruction_and_tier_scores(self, capsys):
        """What: verifies print_report emits prompt, instruction, tier-0, and tier-1 scores.
        Executes: `print_report()` aggregation over prompt, instruction, and tier buckets.
        Why: covers the human-readable report used to inspect IFEval helper output.
        """
        outputs = [
            OutputExample(
                instruction_id_list=[
                    "punctuation:no_comma",
                    "length_constraints:number_words",
                ],
                prompt="Prompt one",
                response="No comma and enough words",
                follow_all_instructions=True,
                follow_instruction_list=[True, True],
            ),
            OutputExample(
                instruction_id_list=["punctuation:no_comma"],
                prompt="Prompt two",
                response="Has, comma",
                follow_all_instructions=False,
                follow_instruction_list=[False],
            ),
        ]

        print_report(outputs)

        report = capsys.readouterr().out
        assert "prompt-level: 0.5" in report
        assert "instruction-level: 0.6666666666666666" in report
        assert "length_constraints 1.0" in report
        assert "punctuation 0.5" in report
        assert "length_constraints:number_words 1.0" in report
        assert "punctuation:no_comma 0.5" in report


class TestIFEvalInstructionsUtil:
    """What: verifies helper utilities used by IFEval instructions.
    Executes: sentence splitting, word counting, tokenizer loading, and keyword sampling.
    Why: covers shared helper paths that multiple instruction classes delegate to.
    """

    def test_split_into_sentences_preserves_abbreviations_and_decimals(self):
        """What: verifies split_into_sentences avoids splitting common abbreviations.
        Executes: `instructions_util.split_into_sentences()` abbreviation rewrites.
        Why: covers sentence parsing around initials and decimals used by length checks.
        """
        sentences = instructions_util.split_into_sentences(
            "Dr. Smith paid 3.14 dollars. This worked!"
        )

        assert sentences == ["Dr. Smith paid 3.14 dollars.", "This worked!"]

    def test_count_words_uses_word_tokens(self):
        """What: verifies count_words tokenizes punctuation-separated word fragments.
        Executes: `instructions_util.count_words()` with the regex word tokenizer.
        Why: covers word-count behavior for punctuation and contractions in responses.
        """
        assert instructions_util.count_words("Hello, runner_1! Can't stop.") == 5

    @staticmethod
    def _punkt_tab_dir(tmp_path):
        """Create an English punkt_tab directory with the four parameter files."""
        language_root = tmp_path / "tokenizers" / "punkt_tab" / "english"
        language_root.mkdir(parents=True)
        for name in (
            "collocations.tab",
            "sent_starters.txt",
            "abbrev_types.txt",
            "ortho_context.tab",
        ):
            (language_root / name).write_text("")
        return language_root

    def test_missing_punkt_is_downloaded_once_then_loaded(self, monkeypatch, tmp_path):
        """What: verifies a missing punkt_tab is fetched with nltk.download exactly once.
        Executes: `_get_sentence_tokenizer()` with `nltk.data.find` failing once and a
        recording `nltk.download`, then loading from the found directory.
        Why: the English Punkt data is no longer vendored and must be fetched at runtime.
        """
        language_root = self._punkt_tab_dir(tmp_path)
        find_calls = []
        downloads = []

        def fake_find(resource_name, *args, **kwargs):
            find_calls.append(resource_name)
            if len(find_calls) == 1:
                raise LookupError(resource_name)
            return instructions_util.nltk.data.FileSystemPathPointer(str(language_root))

        def fake_download(*args, **kwargs):
            downloads.append((args, kwargs))
            return True

        monkeypatch.setattr(instructions_util.nltk.data, "find", fake_find)
        monkeypatch.setattr(instructions_util.nltk, "download", fake_download)
        instructions_util._get_sentence_tokenizer.cache_clear()
        try:
            tokenizer = instructions_util._get_sentence_tokenizer()
            assert instructions_util._get_sentence_tokenizer() is tokenizer
        finally:
            instructions_util._get_sentence_tokenizer.cache_clear()

        assert isinstance(tokenizer, instructions_util.nltk.tokenize.PunktSentenceTokenizer)
        assert find_calls == ["tokenizers/punkt_tab/english/"] * 2
        assert downloads == [(("punkt_tab",), {"quiet": True, "raise_on_error": True})]

    def test_real_punkt_supports_sentence_and_word_paths(self):
        """What: verifies both IFEval Punkt paths work with NLTK's English data.
        Executes: sentence and word tokenization with the real punkt_tab parameters
        (downloaded on first use if absent).
        Why: covers real Punkt abbreviation handling, not just the loading contract.
        """
        instructions_util._get_sentence_tokenizer.cache_clear()

        assert (
            instructions_util.count_sentences("Dr. Smith arrived. NASA launched!")
            == 2
        )
        assert instructions_util.tokenize_words("NASA launched.") == [
            "NASA",
            "launched",
            ".",
        ]

    def test_present_punkt_is_not_downloaded(self, monkeypatch, tmp_path):
        """What: verifies an already installed punkt_tab is used without a download.
        Executes: `_get_sentence_tokenizer()` with `nltk.data.find` succeeding.
        Why: evaluation must not touch the network when NLTK data is present.
        """
        language_root = self._punkt_tab_dir(tmp_path)
        downloads = []

        monkeypatch.setattr(
            instructions_util.nltk.data,
            "find",
            lambda *args, **kwargs: instructions_util.nltk.data.FileSystemPathPointer(
                str(language_root)
            ),
        )
        monkeypatch.setattr(
            instructions_util.nltk, "download", lambda *args, **kwargs: downloads.append(args)
        )
        instructions_util._get_sentence_tokenizer.cache_clear()
        try:
            tokenizer = instructions_util._get_sentence_tokenizer()
        finally:
            instructions_util._get_sentence_tokenizer.cache_clear()

        assert isinstance(tokenizer, instructions_util.nltk.tokenize.PunktSentenceTokenizer)
        assert downloads == []

    def test_generate_keywords_samples_from_word_list(self, monkeypatch):
        """What: verifies generate_keywords samples the requested number of words.
        Executes: `instructions_util.generate_keywords()` through `random.sample()`.
        Why: covers benchmark keyword generation while making random selection deterministic.
        """
        calls = []

        def fake_sample(candidates, k):
            calls.append((candidates, k))
            return ["alpha", "beta"]

        monkeypatch.setattr(instructions_util.random, "sample", fake_sample)

        keywords = instructions_util.generate_keywords(2)

        assert keywords == ["alpha", "beta"]
        assert calls == [(instructions_util.WORD_LIST, 2)]


class TestIFEvalInstructionConflicts:
    """What: verifies instruction conflict graphs become symmetric.
    Executes: `conflict_make()` with synthetic and registered conflict maps.
    Why: covers exclusion logic that prevents incompatible IFEval instructions from pairing.
    """

    def test_conflict_make_adds_self_and_reverse_edges(self):
        """What: verifies conflict_make makes a small conflict graph symmetric.
        Executes: `conflict_make()` on a minimal mutable conflict dictionary.
        Why: covers self-edge insertion and reverse-edge propagation in one small graph.
        """
        conflicts = {"a": {"b"}, "b": set(), "c": set()}

        result = conflict_make(conflicts)

        assert result is conflicts
        assert result["a"] == {"a", "b"}
        assert result["b"] == {"a", "b"}
        assert result["c"] == {"c"}

    def test_registered_conflicts_become_symmetric_for_known_ids(self):
        """What: verifies the registered IFEval conflict map can be symmetrized safely.
        Executes: `conflict_make()` over a copied `INSTRUCTION_CONFLICTS` registry.
        Why: guards known cross-instruction exclusions such as JSON format versus no comma.
        """
        conflicts = {
            key: set(value)
            for key, value in INSTRUCTION_CONFLICTS.items()
        }

        result = conflict_make(conflicts)

        assert "detectable_format:json_format" in result["punctuation:no_comma"]
        assert "punctuation:no_comma" in result["detectable_format:json_format"]
        assert "language:response_language" in result["language:response_language"]
