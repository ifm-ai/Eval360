"""Focused tests for IFEval instruction classes."""

from dataclasses import dataclass
from typing import Any

import pytest

from scheduler.grader.ifeval_lib import instructions


@dataclass(frozen=True)
class InstructionCase:
    """Representative passing and failing responses for one instruction."""

    instruction_id: str
    instruction_cls: type[instructions.Instruction]
    kwargs: dict[str, Any]
    passing_response: str
    failing_response: str


FOCUSED_INSTRUCTION_CASES = [
    pytest.param(
        InstructionCase(
            instruction_id="keywords:existence",
            instruction_cls=instructions.KeywordChecker,
            kwargs={"keywords": ["alpha", "beta"]},
            passing_response="Alpha and beta are both present.",
            failing_response="Only alpha is present.",
        ),
        id="keyword-existence",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="keywords:frequency",
            instruction_cls=instructions.KeywordFrequencyChecker,
            kwargs={"keyword": "echo", "frequency": 2, "relation": "at least"},
            passing_response="echo then ECHO",
            failing_response="echo once",
        ),
        id="keyword-frequency",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="keywords:forbidden_words",
            instruction_cls=instructions.ForbiddenWords,
            kwargs={"forbidden_words": ["zeta"]},
            passing_response="A clean response.",
            failing_response="This mentions zeta.",
        ),
        id="forbidden-words",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="length_constraints:number_words",
            instruction_cls=instructions.NumberOfWords,
            kwargs={"num_words": 3, "relation": "at least"},
            passing_response="one two three",
            failing_response="one two",
        ),
        id="number-words",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="detectable_content:number_placeholders",
            instruction_cls=instructions.PlaceholderChecker,
            kwargs={"num_placeholders": 2},
            passing_response="Hello [name], meet at [place].",
            failing_response="Hello [name].",
        ),
        id="placeholders",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="detectable_format:number_bullet_lists",
            instruction_cls=instructions.BulletListChecker,
            kwargs={"num_bullets": 2},
            passing_response="* one\n- two",
            failing_response="* one",
        ),
        id="bullets",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="detectable_format:constrained_response",
            instruction_cls=instructions.ConstrainedResponseChecker,
            kwargs={},
            passing_response="My answer is yes.",
            failing_response="Yes.",
        ),
        id="constrained-response",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="multi-turn:constrained_start",
            instruction_cls=instructions.ConstrainedStartChecker,
            kwargs={"starter": "I think"},
            passing_response="I think this works.",
            failing_response="Maybe I think this works.",
        ),
        id="constrained-start",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="detectable_format:number_highlighted_sections",
            instruction_cls=instructions.HighlightSectionChecker,
            kwargs={"num_highlights": 2},
            passing_response="This has *one* and *two* highlights.",
            failing_response="This has *one* highlight.",
        ),
        id="highlights",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="detectable_format:multiple_sections",
            instruction_cls=instructions.SectionChecker,
            kwargs={"section_spliter": "Section", "num_sections": 2},
            passing_response="Section 1\nAlpha\nSection 2\nBeta",
            failing_response="Section 1\nAlpha",
        ),
        id="sections",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="length_constraints:number_paragraphs",
            instruction_cls=instructions.ParagraphChecker,
            kwargs={"num_paragraphs": 2},
            passing_response="First paragraph***Second paragraph",
            failing_response="First paragraph***",
        ),
        id="paragraphs",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="detectable_content:postscript",
            instruction_cls=instructions.PostscriptChecker,
            kwargs={"postscript_marker": "P.S."},
            passing_response="Body text.\nP.S. extra note",
            failing_response="Body text only.",
        ),
        id="postscript",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="detectable_format:json_format",
            instruction_cls=instructions.JsonFormat,
            kwargs={},
            passing_response='```json\n{"answer": true}\n```',
            failing_response="{answer: true",
        ),
        id="json-format",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="length_constraints:nth_paragraph_first_word",
            instruction_cls=instructions.ParagraphFirstWordCheck,
            kwargs={
                "num_paragraphs": 2,
                "nth_paragraph": 2,
                "first_word": "banana",
            },
            passing_response="Apple starts here.\n\nBanana starts there.",
            failing_response="Apple starts here.\n\nOrange starts there.",
        ),
        id="paragraph-first-word",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="keywords:key_sentences",
            instruction_cls=instructions.KeySentenceChecker,
            kwargs={"key_sentences": ["Keep this sentence."], "num_sentences": 1},
            passing_response="Keep this sentence.",
            failing_response="Use a different sentence.",
        ),
        id="key-sentence",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="combination:two_responses",
            instruction_cls=instructions.TwoResponsesChecker,
            kwargs={},
            passing_response="First answer******Second answer",
            failing_response="Same answer******Same answer",
        ),
        id="two-responses",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="combination:repeat_prompt",
            instruction_cls=instructions.RepeatPromptThenAnswer,
            kwargs={"prompt_to_repeat": "Say hi"},
            passing_response="Say hi\nHello!",
            failing_response="Hello!",
        ),
        id="repeat-prompt",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="startend:end_checker",
            instruction_cls=instructions.EndChecker,
            kwargs={"end_phrase": "done"},
            passing_response="All done",
            failing_response="done then more",
        ),
        id="end-checker",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="detectable_format:title",
            instruction_cls=instructions.TitleChecker,
            kwargs={},
            passing_response="<<A clear title>>\nBody",
            failing_response="<A clear title>",
        ),
        id="title",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="keywords:letter_frequency",
            instruction_cls=instructions.LetterFrequencyChecker,
            kwargs={"letter": "x", "let_frequency": 2, "let_relation": "at least"},
            passing_response="xylophone x",
            failing_response="xylophone",
        ),
        id="letter-frequency",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="punctuation:no_comma",
            instruction_cls=instructions.CommaChecker,
            kwargs={},
            passing_response="No comma here",
            failing_response="Comma, here",
        ),
        id="no-comma",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="startend:quotation",
            instruction_cls=instructions.QuotationChecker,
            kwargs={},
            passing_response='"quoted"',
            failing_response="quoted",
        ),
        id="quotation",
    ),
    pytest.param(
        InstructionCase(
            instruction_id="detectable_content:rephrase_paragraph",
            instruction_cls=instructions.RephraseParagraph,
            kwargs={"original_paragraph": "alpha beta gamma", "low": 1, "high": 2},
            passing_response="alpha delta",
            failing_response="alpha beta gamma",
        ),
        id="rephrase-paragraph",
    ),
]


class TestInstructionClassMatrix:
    """What: groups representative instruction class behavior without exhaustive brittleness.
    Executes: instruction construction, description building, argument keys, and checks.
    Why: covers broad IFEval instruction behavior through stable pass/fail examples.
    """

    @pytest.mark.parametrize("case", FOCUSED_INSTRUCTION_CASES)
    def test_instruction_accepts_passing_and_rejects_failing_response(self, case):
        """What: verifies each representative instruction validates pass and fail examples.
        Executes: `build_description()`, `get_instruction_args_keys()`, and `check_following()`.
        Why: covers the common instruction lifecycle used by strict and loose evaluators.
        """
        instruction = case.instruction_cls(case.instruction_id)

        description = instruction.build_description(**case.kwargs)

        assert description
        assert isinstance(instruction.get_instruction_args_keys(), list)
        assert instruction.check_following(case.passing_response) is True
        assert instruction.check_following(case.failing_response) is False


class TestLanguageAndTokenizerBackedInstructions:
    """What: groups stable coverage for instructions that depend on language or tokenizers.
    Executes: language checkers, case checkers, sentence counts, and capital-word checks.
    Why: covers instruction corners that need deterministic langdetect and NLTK substitutes.
    """

    def test_response_language_checker_uses_detected_language(self, monkeypatch):
        """What: verifies responseLanguageChecker compares langdetect output with the target.
        Executes: `ResponseLanguageChecker.check_following()` with patched detection.
        Why: covers the target-language gate for multilingual IFEval prompts.
        """
        detected = "en"

        def fake_detect(_value):
            return detected

        monkeypatch.setattr(instructions.langdetect, "detect", fake_detect)
        instruction = instructions.ResponseLanguageChecker(
            "language:response_language"
        )
        instruction.build_description(language="en")

        assert instruction.check_following("Hello there") is True

    def test_response_language_checker_treats_detection_errors_as_followed(
        self, monkeypatch
    ):
        """What: verifies responseLanguageChecker keeps upstream behavior for detection errors.
        Executes: the `LangDetectException` branch in `check_following()`.
        Why: preserves the loose upstream behavior for undetectable short responses.
        """

        def fake_detect(_value):
            raise instructions.langdetect.LangDetectException(
                code=0,
                message="no features",
            )

        monkeypatch.setattr(instructions.langdetect, "detect", fake_detect)
        instruction = instructions.ResponseLanguageChecker(
            "language:response_language"
        )
        instruction.build_description(language="en")

        assert instruction.check_following("") is True

    def test_case_language_checkers_require_expected_case_and_language(
        self, monkeypatch
    ):
        """What: verifies english case checkers combine case checks with language detection.
        Executes: capital and lowercase English checkers with patched language detection.
        Why: covers the instruction corner where casing only counts for English responses.
        """
        monkeypatch.setattr(instructions.langdetect, "detect", lambda _value: "en")
        uppercase = instructions.CapitalLettersEnglishChecker(
            "change_case:english_capital"
        )
        lowercase = instructions.LowercaseLettersEnglishChecker(
            "change_case:english_lowercase"
        )
        uppercase.build_description()
        lowercase.build_description()

        assert uppercase.check_following("LOUD ENGLISH WORDS") is True
        assert uppercase.check_following("Loud English Words") is False
        assert lowercase.check_following("quiet english words") is True
        assert lowercase.check_following("Quiet English Words") is False

    def test_number_of_sentences_uses_instruction_util_counter(self, monkeypatch):
        """What: verifies numberOfSentences delegates sentence counting to instructions_util.
        Executes: `NumberOfSentences.check_following()` via patched `count_sentences()`.
        Why: covers relation evaluation without depending on the real tokenizer.
        """
        monkeypatch.setattr(
            instructions.instructions_util,
            "count_sentences",
            lambda value: 2 if "two" in value else 1,
        )
        instruction = instructions.NumberOfSentences(
            "length_constraints:number_sentences"
        )
        instruction.build_description(num_sentences=2, relation="at least")

        assert instruction.check_following("two sentence marker") is True
        assert instruction.check_following("one sentence marker") is False

    def test_capital_word_frequency_uses_punkt_word_tokenizer(self):
        """What: verifies capitalWordFrequencyChecker uses NLTK's English Punkt data.
        Executes: `CapitalWordFrequencyChecker.check_following()` without mocks.
        Why: covers the real uppercase word-frequency scoring path.
        """
        instruction = instructions.CapitalWordFrequencyChecker(
            "change_case:capital_word_frequency"
        )
        instruction.build_description(
            capital_frequency=2,
            capital_relation="at least",
        )

        assert instruction.check_following("NASA CPU ok") is True
        assert instruction.check_following("NASA cpu ok") is False


class TestInstructionValidationBranches:
    """What: groups coverage for validation and exception branches in instructions.
    Executes: relation validation, rephrase marker checks, and repeat-prompt validation.
    Why: covers malformed instruction configurations before they reach benchmark scoring.
    """

    @pytest.mark.parametrize(
        ("instruction_cls", "kwargs"),
        [
            (
                instructions.NumberOfWords,
                {"num_words": 1, "relation": "exactly"},
            ),
            (
                instructions.KeywordFrequencyChecker,
                {"keyword": "echo", "frequency": 1, "relation": "exactly"},
            ),
            (
                instructions.LetterFrequencyChecker,
                {"letter": "e", "let_frequency": 1, "let_relation": "exactly"},
            ),
            (
                instructions.CapitalWordFrequencyChecker,
                {"capital_frequency": 1, "capital_relation": "exactly"},
            ),
        ],
    )
    def test_comparison_instructions_reject_unknown_relations(
        self, instruction_cls, kwargs
    ):
        """What: verifies comparison-based instructions reject unsupported relation names.
        Executes: `build_description()` for relation-based instruction classes.
        Why: rejects ambiguous comparison operators across word, keyword, letter, and case checks.
        """
        instruction = instruction_cls("test:instruction")

        with pytest.raises(ValueError, match="supported relation"):
            instruction.build_description(**kwargs)

    def test_rephrase_checker_requires_change_markers_when_building(self):
        """What: verifies rephraseChecker rejects original messages without change markers.
        Executes: `RephraseChecker.build_description()` marker validation.
        Why: prevents undefined rephrase targets from entering IFEval instruction checks.
        """
        instruction = instructions.RephraseChecker("detectable_format:rephrase")

        with pytest.raises(ValueError, match="does not contain changes"):
            instruction.build_description(original_message="No marked change")

    def test_rephrase_checker_requires_change_markers_when_checking(self):
        """What: verifies rephraseChecker rejects checked values without change markers.
        Executes: `RephraseChecker.check_following()` marker validation.
        Why: surfaces malformed candidate responses instead of scoring them as followed.
        """
        instruction = instructions.RephraseChecker("detectable_format:rephrase")
        instruction.build_description(original_message="Keep *this* text")

        with pytest.raises(ValueError, match="does not contain"):
            instruction.check_following("Keep that text")

    def test_repeat_prompt_requires_prompt_to_repeat(self):
        """What: verifies repeatPromptThenAnswer rejects missing prompt configuration.
        Executes: `RepeatPromptThenAnswer.build_description()` without `prompt_to_repeat`.
        Why: the instruction cannot validate a repeated prompt unless the expected prompt is configured.
        """
        instruction = instructions.RepeatPromptThenAnswer(
            "combination:repeat_prompt"
        )

        with pytest.raises(ValueError, match="prompt_to_repeat must be set"):
            instruction.build_description()
