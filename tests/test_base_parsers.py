import scheduler.grader.base_parsers  # ensure parsers are registered
from scheduler.grader.parser_registry import get_parser


class ParserTestBase:
    """Shared parser helper for parser test classes.

    What: centralizes parser invocation used by the parser tests.
    Executes: `get_parser(...).parse_generations(...)` for one generation.
    Why: every parser assertion should pass through the registry so aliases and registration stay covered.
    """

    @staticmethod
    def parse(parser_name: str, text):
        """Return the parser output for one generation.

        What: parses one raw generation through the named parser.
        Executes: the parser registry and parser callable selected by `parser_name`.
        Why: avoids duplicating registry lookup while preserving behavioral assertions.
        """
        parser = get_parser(parser_name)
        return parser.parse_generations([text])[0]


# ---------------------------------------------------------------------------
# so_the_answer_is
# ---------------------------------------------------------------------------

class TestSoTheAnswerIs(ParserTestBase):
    def test_match_returns_content(self):
        # "the answer is 42." — greedy .* with (?=.) lookahead stops before last char
        result = self.parse("so_the_answer_is", "the answer is 42.")
        assert result == "42"

    def test_correct_answer_is(self):
        assert self.parse("so_the_answer_is", "the correct answer is (B).") == "(B)"

    def test_case_insensitive(self):
        result = self.parse("so_the_answer_is", "The Answer Is yes!")
        assert result is not None

    def test_no_match_returns_none(self):
        assert self.parse("so_the_answer_is", "nothing relevant here") is None

    def test_no_trailing_char_returns_none(self):
        # "(?=.)" requires at least one char after the captured group's last char;
        # if the string ends right after "the answer is X" with no trailing chars
        # the regex will still match — capturing everything except the final char.
        # A single-char answer with no trailing chars cannot satisfy both groups:
        result = self.parse("so_the_answer_is", "The answer is X")
        # "X" has nothing after it; .* matches "" and (?=.) needs "X"; returns ""
        assert result == ""


# ---------------------------------------------------------------------------
# so_the_answer_is_last
# ---------------------------------------------------------------------------

class TestSoTheAnswerIsLast(ParserTestBase):
    def test_single_occurrence(self):
        assert self.parse("so_the_answer_is_last", "The answer is (B).") == "(B)"

    def test_takes_last_same_line(self):
        # Two occurrences on the same line — must return content after the last one
        assert self.parse("so_the_answer_is_last", "the answer is $1,000. The answer is (B).") == "(B)"

    def test_takes_last_with_newline(self):
        assert self.parse("so_the_answer_is_last", "the answer is (A).\nthe answer is (C).") == "(C)"

    def test_looping_generation(self):
        gen = "the answer is (A). Let me reconsider. the answer is (C)."
        assert self.parse("so_the_answer_is_last", gen) == "(C)"

    def test_no_match_returns_none(self):
        assert self.parse("so_the_answer_is_last", "nothing relevant here") is None

    def test_correct_answer_is(self):
        assert self.parse("so_the_answer_is_last", "the correct answer is (B).") == "(B)"

    def test_correct_answer_is_last(self):
        assert self.parse("so_the_answer_is_last", "the answer is (A). the correct answer is (C).") == "(C)"

    def test_case_insensitive(self):
        assert self.parse("so_the_answer_is_last", "THE ANSWER IS yes.") == "yes"

    def test_word_answer(self):
        assert self.parse("so_the_answer_is_last", "So the answer is yes.") == "yes"


# ---------------------------------------------------------------------------
# passthrough / noop / identity
# ---------------------------------------------------------------------------

class TestPassthrough(ParserTestBase):
    def test_returns_input_unchanged(self):
        assert self.parse("passthrough", "hello world") == "hello world"

    def test_noop_alias(self):
        assert self.parse("noop", "abc") == "abc"

    def test_identity_alias(self):
        assert self.parse("identity", "abc") == "abc"


# ---------------------------------------------------------------------------
# answer_tag
# ---------------------------------------------------------------------------

class TestAnswerTag(ParserTestBase):
    def test_extracts_content(self):
        assert self.parse("answer_tag", "<answer>42</answer>") == "42"

    def test_case_insensitive(self):
        assert self.parse("answer_tag", "<ANSWER>yes</ANSWER>") == "yes"

    def test_strips_whitespace(self):
        assert self.parse("answer_tag", "<answer>  hello  </answer>") == "hello"

    def test_multiline_content(self):
        assert self.parse("answer_tag", "<answer>line1\nline2</answer>") == "line1\nline2"

    def test_empty_tag_returns_none(self):
        assert self.parse("answer_tag", "<answer></answer>") is None

    def test_whitespace_only_tag_returns_none(self):
        assert self.parse("answer_tag", "<answer>   </answer>") is None

    def test_no_tag_returns_none(self):
        assert self.parse("answer_tag", "no tags here") is None


# ---------------------------------------------------------------------------
# think_tag
# ---------------------------------------------------------------------------

class TestThinkTag(ParserTestBase):
    """think_tag is now an alias for think_suffix — same behaviour."""

    def test_extracts_after_think(self):
        assert self.parse("think_tag", "<think>reasoning</think>answer") == "answer"

    def test_strips_whitespace(self):
        assert self.parse("think_tag", "</think>  result  ") == "result"

    def test_empty_after_think_returns_none(self):
        assert self.parse("think_tag", "</think>") is None

    def test_whitespace_only_after_think_returns_none(self):
        assert self.parse("think_tag", "</think>   ") is None

    def test_none_input_returns_none(self):
        assert self.parse("think_tag", None) is None

    def test_no_think_tag_returns_full_text(self):
        # think_suffix (and now think_tag) fall back to the full text when no
        # </think> tag is present, rather than returning None.
        assert self.parse("think_tag", "no think tag here") == "no think tag here"

    def test_case_insensitive(self):
        assert self.parse("think_tag", "</THINK>answer") == "answer"

    def test_think_fast_variant_stripped(self):
        assert self.parse("think_tag", "</think_fast>answer") == "answer"

    def test_think_faster_variant_stripped(self):
        assert self.parse("think_tag", "</think_faster>answer") == "answer"

    def test_keeps_last_tail_on_multiple_tags(self):
        assert self.parse("think_tag", "</think>first</think>last") == "last"

    def test_is_alias_of_think_suffix(self):
        from scheduler.grader.parser_registry import get_parser
        assert get_parser("think_tag")._processor is get_parser("think_suffix")._processor


# ---------------------------------------------------------------------------
# think_suffix
# ---------------------------------------------------------------------------

class TestThinkSuffix(ParserTestBase):
    def test_none_input_returns_none(self):
        assert self.parse("think_suffix", None) is None

    def test_no_think_tag_returns_full_text(self):
        assert self.parse("think_suffix", "plain text") == "plain text"

    def test_extracts_after_think_tag(self):
        assert self.parse("think_suffix", "<think>hidden</think>answer") == "answer"

    def test_empty_tail_returns_none(self):
        assert self.parse("think_suffix", "prefix</think>") is None

    def test_no_output_cap(self):
        tail = "x" * 1500
        result = self.parse("think_suffix", f"<think>r</think>{tail}")
        assert result == "x" * 1500

    def test_think_fast_variant_stripped(self):
        assert self.parse("think_suffix", "</think_fast>answer") == "answer"

    def test_think_faster_variant_stripped(self):
        assert self.parse("think_suffix", "</think_faster>answer") == "answer"

    def test_keeps_last_tail_on_multiple_tags(self):
        result = self.parse("think_suffix", "</think>first</think>last")
        assert result == "last"


# ---------------------------------------------------------------------------
# the_answer_is
# ---------------------------------------------------------------------------

class TestTheAnswerIs(ParserTestBase):
    def test_extracts_letter(self):
        assert self.parse("the_answer_is", "the answer is A") == "A"

    def test_case_insensitive_letter(self):
        assert self.parse("the_answer_is", "The answer is b") == "b"

    def test_with_parentheses(self):
        assert self.parse("the_answer_is", "the answer is (C)") == "C"

    def test_with_surrounding_newlines(self):
        assert self.parse("the_answer_is", "\nthe answer is D\n") == "D"

    def test_no_match_returns_none(self):
        assert self.parse("the_answer_is", "no relevant text") is None

    def test_returns_first_match(self):
        # re.search returns first match
        result = self.parse("the_answer_is", "the answer is A, then the answer is B")
        assert result == "A"

    def test_matches_with_chat_before_and_after(self):
        gen = "Thanks for waiting. After reviewing the options, the answer is (C). Let me know if you want the reasoning."
        assert self.parse("the_answer_is", gen) == "C"


# ---------------------------------------------------------------------------
# the_answer_is_last
# ---------------------------------------------------------------------------

class TestTheAnswerIsLast(ParserTestBase):
    def test_extracts_last_letter(self):
        result = self.parse("the_answer_is_last", "the answer is A, then the answer is B")
        assert result == "B"

    def test_single_match(self):
        assert self.parse("the_answer_is_last", "the answer is C") == "C"

    def test_single_match_with_surrounding_newlines(self):
        assert self.parse("the_answer_is_last", "\nthe answer is D\n") == "D"

    def test_no_match_returns_none(self):
        assert self.parse("the_answer_is_last", "nothing here") is None

    def test_matches_last_with_chat_before_and_after(self):
        gen = "I first guessed A. After checking again, the answer is (B). Happy to explain more."
        assert self.parse("the_answer_is_last", gen) == "B"


# ---------------------------------------------------------------------------
# boxed
# ---------------------------------------------------------------------------

class TestBoxed(ParserTestBase):
    def test_none_input_returns_none(self):
        assert self.parse("boxed", None) is None

    def test_no_boxed_returns_none(self):
        assert self.parse("boxed", "no boxed here") is None

    def test_simple_boxed(self):
        assert self.parse("boxed", r"\boxed{42}") == "42"

    def test_fbox(self):
        assert self.parse("boxed", r"\fbox{hello}") == "hello"

    def test_nested_braces(self):
        assert self.parse("boxed", r"\boxed{x + {y}}") == "x + {y}"

    def test_last_of_multiple(self):
        assert self.parse("boxed", r"\boxed{first} \boxed{second}") == "second"

    def test_strips_whitespace(self):
        assert self.parse("boxed", r"\boxed{ 42 }") == "42"

    def test_unmatched_brace_returns_full_generation(self):
        gen = r"\boxed{unclosed"
        assert self.parse("boxed", gen) == gen

    def test_case_insensitive(self):
        assert self.parse("boxed", r"\BOXED{42}") == "42"


# ---------------------------------------------------------------------------
# gsm8k_base
# ---------------------------------------------------------------------------

class TestGsm8kBase(ParserTestBase):
    def test_hash_separator(self):
        assert self.parse("gsm8k_base", "work\n### 42") == "42"

    def test_answer_colon(self):
        assert self.parse("gsm8k_base", "Answer: 42") == "42"

    def test_bold_answer_colon(self):
        assert self.parse("gsm8k_base", "step **Answer: 42") == "42"

    def test_bold_answer(self):
        result = self.parse("gsm8k_base", "step **Answer** 42")
        assert "42" in result

    def test_bold_final_answer(self):
        assert self.parse("gsm8k_base", "work **Final Answer:** 42") == "42"

    def test_boxed(self):
        # splits on "boxed", takes last part
        assert self.parse("gsm8k_base", r"boxed{42}") == "{42}"

    def test_plain_text_unchanged(self):
        assert self.parse("gsm8k_base", "42") == "42"

    def test_empty_string_returns_none(self):
        assert self.parse("gsm8k_base", "") is None

    def test_the_answer_is_splits(self):
        # "The answer is 42" should return the stripped answer text.
        result = self.parse("gsm8k_base", "The answer is 42")
        assert result == "42"

    def test_the_answer_is_strips_whitespace(self):
        result = self.parse("gsm8k_base", "work\nThe answer is 42 \n")
        assert result == "42"


# ---------------------------------------------------------------------------
# code_completion / humaneval
# ---------------------------------------------------------------------------

class TestCodeCompletion(ParserTestBase):
    def test_none_input_returns_none(self):
        assert self.parse("code_completion", None) is None

    def test_plain_code_returned_unchanged(self):
        code = "def foo():\n    return 1\n"
        assert self.parse("code_completion", code) == code

    def test_strips_closed_think_block(self):
        text = "<think>reasoning here</think>\ndef foo(): pass"
        result = self.parse("code_completion", text)
        assert "think" not in result.lower()
        assert "def foo" in result

    def test_unclosed_think_with_fence(self):
        text = "<think>\nsome thinking\n```python\ndef foo(): pass\n```"
        result = self.parse("code_completion", text)
        assert "def foo" in result

    def test_unclosed_think_without_fence_takes_last_paragraph(self):
        text = "<think>\nparagraph one\n\ndef answer(): pass"
        result = self.parse("code_completion", text)
        assert "def answer" in result

    def test_extracts_from_markdown_fence(self):
        text = "Some prose\n```python\ndef bar(): return 2\n```\nmore prose"
        result = self.parse("code_completion", text)
        assert "def bar" in result

    def test_empty_code_returns_none(self):
        assert self.parse("code_completion", "   ") is None

    def test_humaneval_alias(self):
        assert self.parse("humaneval", "def f(): pass") == "def f(): pass"


# ---------------------------------------------------------------------------
# _extract_mc_letter (via mc_answer)
# ---------------------------------------------------------------------------

class TestExtractMcLetter(ParserTestBase):
    def test_no_letter_returns_none(self):
        # text with no A-D pattern at all → _extract_mc_letter returns None
        assert self.parse("mc_answer", "xyz 123") is None

    def test_answer_tag_pattern(self):
        assert self.parse("mc_answer", "<answer>C</answer>") == "C"

    def test_final_answer_tag(self):
        assert self.parse("mc_answer", "<final answer>B</final answer>") == "B"

    def test_boxed_pattern(self):
        assert self.parse("mc_answer", r"\boxed{A}") == "A"

    def test_option_word(self):
        assert self.parse("mc_answer", "option B is correct") == "B"

    def test_parenthesized(self):
        assert self.parse("mc_answer", "(D)") == "D"

    def test_bare_letter_on_own_line(self):
        assert self.parse("mc_answer", "\nA\n") == "A"

    def test_bare_letter_in_sentence(self):
        # last resort: bare \b([A-D])\b
        assert self.parse("mc_answer", "The choice is B.") == "B"

    def test_supports_e(self):
        assert self.parse("mc_answer", "Answer: E") == "E"

    def test_maps_numeric_choice_to_letter(self):
        assert self.parse("mc_answer", "Answer: 5") == "E"

    def test_markdown_answer_marker_beats_later_option_mentions(self):
        text = "Answer: **A**\nOption D sounds plausible, but it is incorrect."
        assert self.parse("mc_answer", text) == "A"

    def test_leading_standalone_answer_beats_later_rationale_mentions(self):
        text = "B\nOption C is tempting, but option D is wrong."
        assert self.parse("mc_answer", text) == "B"

    def test_long_markdown_conclusion_with_trailing_reference(self):
        reasoning = "\n".join(f"Step {i}: detailed reasoning." for i in range(140))
        text = "\n".join(
            [
                reasoning,
                "The calculated value matches option **D (42)**.",
                "---",
                "**Reference**",
                "Standard textbook treatment of the relevant formulas.",
            ]
        )
        assert self.parse("mc_answer", text) == "D"

    def test_only_option_phrase(self):
        text = "Only option **D (42 units)** is larger than this requirement."
        assert self.parse("mc_answer", text) == "D"


# ---------------------------------------------------------------------------
# mc_answer (lines 161, 165, 176)
# ---------------------------------------------------------------------------

class TestMcAnswer(ParserTestBase):
    def test_none_returns_none(self):
        assert self.parse("mc_answer", None) is None

    def test_empty_string_returns_none(self):
        assert self.parse("mc_answer", "") is None

    def test_whitespace_only_returns_none(self):
        assert self.parse("mc_answer", "   ") is None

    def test_plain_letter(self):
        assert self.parse("mc_answer", "A") == "A"

    def test_parenthesized_letter(self):
        assert self.parse("mc_answer", "(c)") == "C"

    def test_answer_prefix(self):
        assert self.parse("mc_answer", "Final. Answer: option d") == "D"

    def test_boxed(self):
        assert self.parse("mc_answer", r"\boxed{b}") == "B"

    def test_boxed_numeric_choice(self):
        assert self.parse("mc_answer", r"\boxed{4}") == "D"

    def test_answer_tag(self):
        assert self.parse("mc_answer", "<answer>c</answer>") == "C"

    def test_think_then_answer(self):
        assert self.parse("mc_answer", "<think>reasoning</think>\nAnswer: B") == "B"

    def test_think_tag_mc_alias(self):
        assert self.parse("think_tag_mc", "<think>hidden</think>\nThe answer is (A).") == "A"

    def test_multiple_choice_answer_alias(self):
        assert self.parse("multiple_choice_answer", "The answer is B") == "B"

    def test_fallback_to_full_text_when_post_think_empty(self):
        # After </think> there is no letter, but full text has one
        text = "Answer is C</think>"
        result = self.parse("mc_answer", text)
        assert result == "C"

    def test_final_markdown_answer_after_reasoning_is_respected(self):
        text = "<think>Option C seems tempting.</think>\nAnswer: **B**"
        assert self.parse("mc_answer", text) == "B"


# ---------------------------------------------------------------------------
# code-think
# ---------------------------------------------------------------------------

class TestCodeThink(ParserTestBase):
    """What: groups coverage for the code-think parser used by thinking code models.
    Executes: the registered code-think parser and its think-tag stripping plus Python fence extraction.
    Why: protects code-task parsing for models that wrap final code in hidden reasoning blocks.
    """

    def test_none_input_returns_none(self):
        """What: verifies none generations should pass through as None.
        Executes: the code-think parser's None guard through Parser.parse_generations().
        Why: preserves the parser contract for missing generations produced by failed or skipped model calls.
        """
        assert self.parse("code-think", None) is None

    def test_missing_python_fence_returns_none(self):
        """What: verifies non-fenced prose should not be treated as executable code.
        Executes: the code-think parser branch where no ```python fence is found.
        Why: avoids submitting plain reasoning or prose as code to downstream code graders.
        """
        assert self.parse("code-think", "def solve():\n    return 1") is None

    def test_strips_closed_think_block_before_extracting_code(self):
        """What: verifies a closed thinking block should be removed before code extraction.
        Executes: the code-think parser's </think> trimming and Python fence extraction.
        Why: hidden reasoning must not be submitted to code graders, while the final solution block must remain intact.
        """
        text = "<think>hidden reasoning</think>\n```python\ndef solve():\n    return 1\n```"
        assert self.parse("code-think", text) == "def solve():\n    return 1\n"

    def test_uses_last_python_fence(self):
        """What: verifies when multiple python fences exist, the final code fence is the answer.
        Executes: the code-think parser's regex collection and last-match selection.
        Why: covers model outputs that draft code first and put the corrected solution in the final fence.
        """
        text = "\n".join(
            [
                "```python",
                "def draft():",
                "    return 0",
                "```",
                "```python3",
                "def solve():",
                "    return 1",
                "```",
            ]
        )
        assert self.parse("code-think", text) == "def solve():\n    return 1\n"


# ---------------------------------------------------------------------------
# code
# ---------------------------------------------------------------------------

class TestCodeBlockParser(ParserTestBase):
    """What: groups coverage for markdown code block selection and example stripping.
    Executes: the registered code parser, _best_code_block(), and _strip_example_code().
    Why: protects code-grader inputs by selecting solution code while dropping demonstration calls.
    """

    def test_none_input_returns_none(self):
        """What: verifies none generations should pass through as None.
        Executes: the code parser's None guard through Parser.parse_generations().
        Why: preserves the parser contract for absent generations without raising in grading.
        """
        assert self.parse("code", None) is None

    def test_missing_fence_returns_none(self):
        """What: verifies the code parser requires a fenced python block.
        Executes: the code parser branch where no ```python or ```python3 block is found.
        Why: avoids sending unfenced text to code execution tasks that expect an extracted solution.
        """
        assert self.parse("code", "def solve():\n    return 1") is None

    def test_prefers_last_block_with_definition(self):
        """What: verifies example-only trailing blocks should not beat the final definition block.
        Executes: _best_code_block() scanning backward for a def or class before stripping examples.
        Why: covers model responses that append runnable examples after the actual solution block.
        """
        text = "\n".join(
            [
                "```python",
                "print('example')",
                "```",
                "```python",
                "def solve():",
                "    return 42",
                "```",
                "```python",
                "print(solve())",
                "```",
            ]
        )
        assert self.parse("code", text) == "def solve():\n    return 42"

    def test_strips_trailing_example_usage(self):
        """What: verifies top-level sample calls after a solution definition should be removed.
        Executes: _strip_example_code() on a fenced block containing imports, a function, and print usage.
        Why: prevents generated examples from running in the harness before the grader's own tests.
        """
        text = "\n".join(
            [
                "```python",
                "import math",
                "",
                "def solve():",
                "    return int(math.sqrt(4))",
                "",
                "print(solve())",
                "```",
            ]
        )
        assert self.parse("code", text) == "import math\n\ndef solve():\n    return int(math.sqrt(4))"

    def test_falls_back_to_last_block_when_no_definition_exists(self):
        """What: verifies when no block has a def or class, the parser keeps the last block.
        Executes: _best_code_block() fallback selection when no fenced block defines a function or class.
        Why: preserves useful code answers for prompts that do not require a named definition.
        """
        text = "```python\nprint('first')\n```\n```python\nprint('last')\n```"
        assert self.parse("code", text) == "print('last')\n"

    def test_keeps_imports_and_classes(self):
        """What: verifies top-level imports and class definitions belong to the submitted code.
        Executes: _strip_example_code() on a class-based solution with a dependency import.
        Why: ensures cleanup removes only example invocation code, not definitions needed by the grader.
        """
        text = "\n".join(
            [
                "```python",
                "from collections import deque",
                "",
                "class Solver:",
                "    def solve(self):",
                "        return deque([1])",
                "",
                "Solver().solve()",
                "```",
            ]
        )
        expected = (
            "from collections import deque\n\n"
            "class Solver:\n"
            "    def solve(self):\n"
            "        return deque([1])"
        )
        assert self.parse("code", text) == expected


# ---------------------------------------------------------------------------
# longbench-v2 / the-correct-answer-is
# ---------------------------------------------------------------------------

class TestLongbenchV2(ParserTestBase):
    """What: groups coverage for LongBench multiple-choice answer extraction.
    Executes: the longbench-v2 parser and its the-correct-answer-is alias.
    Why: protects answer normalization for LongBench-style multiple-choice grading.
    """

    def test_none_input_returns_none(self):
        """What: verifies none generations should pass through as None.
        Executes: the longbench-v2 parser's None guard through Parser.parse_generations().
        Why: preserves graceful handling for missing model generations.
        """
        assert self.parse("longbench-v2", None) is None

    def test_correct_answer_phrase_returns_uppercase_choice(self):
        """What: verifies the explicit correct-answer phrase should be preferred.
        Executes: the longbench-v2 primary regex for "the correct answer is" responses.
        Why: covers the canonical LongBench answer format and normalizes lowercase choices.
        """
        assert self.parse("longbench-v2", "the correct answer is (b).") == "B"

    def test_parenthesized_choice_fallback(self):
        """What: verifies a standalone parenthesized choice is accepted as a fallback.
        Executes: the longbench-v2 fallback regex that extracts choices from "(A)"-style text.
        Why: covers terse model answers that omit the full "correct answer" phrase.
        """
        assert self.parse("longbench-v2", "After review, choose (d).") == "D"

    def test_alias_uses_same_parser(self):
        """What: verifies the historical alias should expose the same extraction behavior.
        Executes: parser registry lookup for the-correct-answer-is and the shared longbench-v2 processor.
        Why: preserves backward compatibility for configs that still name the older parser alias.
        """
        assert self.parse("the-correct-answer-is", "The correct answer is C") == "C"

    def test_no_match_returns_none(self):
        """What: verifies generations without a choice should return None.
        Executes: the longbench-v2 no-match branch after both answer patterns fail.
        Why: avoids inventing a multiple-choice label when the model did not state one.
        """
        assert self.parse("longbench-v2", "No option is stated here.") is None
