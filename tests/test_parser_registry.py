"""Tests for the parser registry, including duplicate registration detection."""
import pytest
from scheduler.grader.parser_registry import Parser, _PARSER_REGISTRY, register_parser, get_parser


@pytest.fixture(autouse=True)
def isolated_registry():
    """Snapshot and restore the registry around each test so registrations don't leak."""
    snapshot = dict(_PARSER_REGISTRY)
    yield
    _PARSER_REGISTRY.clear()
    _PARSER_REGISTRY.update(snapshot)


class TestRegisterParser:
    def test_registered_parser_is_retrievable(self):
        @register_parser("test-unique-parser")
        def _p(g):
            return g.upper()

        parser = get_parser("test-unique-parser")
        assert parser.parse_generations(["hello"]) == ["HELLO"]

    def test_duplicate_registration_raises(self):
        @register_parser("test-dup-parser")
        def _p1(g):
            return g

        with pytest.raises(ValueError, match="Duplicate parser registration.*test-dup-parser"):
            @register_parser("test-dup-parser")
            def _p2(g):
                return g.upper()

    def test_duplicate_does_not_overwrite_original(self):
        """The original registration must survive a failed duplicate attempt."""
        @register_parser("test-no-overwrite-parser")
        def _original(g):
            return "original"

        with pytest.raises(ValueError):
            @register_parser("test-no-overwrite-parser")
            def _replacement(g):
                return "replacement"

        assert get_parser("test-no-overwrite-parser").parse_generations(["x"]) == ["original"]

    def test_multi_name_registration_raises_on_any_duplicate(self):
        """register_parser accepts multiple names; raise if any one is already taken."""
        @register_parser("test-multi-a")
        def _p1(g):
            return g

        with pytest.raises(ValueError, match="test-multi-a"):
            @register_parser("test-multi-b", "test-multi-a")
            def _p2(g):
                return g

    def test_distinct_names_do_not_conflict(self):
        @register_parser("test-name-x")
        def _px(g):
            return "x"

        @register_parser("test-name-y")
        def _py(g):
            return "y"

        assert get_parser("test-name-x").parse_generations([""])[0] == "x"
        assert get_parser("test-name-y").parse_generations([""])[0] == "y"


class TestParserMetadataExtraction:
    """What: groups tests for reasoning and tool-call metadata extraction.
    Executes: Parser.extract_reasoning_and_tools() against think and tool_call tag variants.
    Why: protects the metadata fields merged into parsed grader records.
    """

    def test_empty_generation_has_no_metadata(self):
        """What: verifies empty or falsey generations should produce no parsed metadata.
        Executes: Parser.extract_reasoning_and_tools() through its early falsey-generation return.
        Why: prevents blank model outputs from creating misleading parsed_* fields.
        """
        parser = Parser(lambda generation: generation)
        assert parser.extract_reasoning_and_tools("") == {}

    def test_closed_think_block_extracts_reasoning_and_answer(self):
        """What: verifies a closed think tag should split reasoning from the visible answer.
        Executes: Parser.extract_reasoning_and_tools() through the closed <think>...</think> branch.
        Why: downstream graders need the visible answer without hidden reasoning mixed into it.
        """
        parser = Parser(lambda generation: generation)
        result = parser.extract_reasoning_and_tools("<think> hidden </think> answer ")
        assert result == {"reasoning": "hidden", "answer": "answer"}

    def test_multiple_think_blocks_are_flagged(self):
        """What: verifies multiple closed think blocks should preserve the first and set a flag.
        Executes: Parser.extract_reasoning_and_tools() with repeated closed think blocks.
        Why: records malformed multi-block reasoning without losing the first parsed reasoning span.
        """
        parser = Parser(lambda generation: generation)
        result = parser.extract_reasoning_and_tools(
            "<think>first</think> visible <think>second</think>"
        )
        assert result["reasoning"] == "first"
        assert result["answer"] == "visible <think>second</think>"
        assert result["multiple_think_blocks"] is True

    def test_orphan_closing_think_tag_uses_prefix_as_reasoning(self):
        """What: verifies a generation that starts mid-thought should treat text before close as reasoning.
        Executes: Parser.extract_reasoning_and_tools() through the orphan </think*> closing-tag branch.
        Why: covers prompt-prefix workflows where the opening think tag was outside the generated text.
        """
        parser = Parser(lambda generation: generation)
        result = parser.extract_reasoning_and_tools("partial reasoning</think_fast> final")
        assert result == {"reasoning": "partial reasoning", "answer": "final"}

    def test_unclosed_think_tag_marks_answer_unknown(self):
        """What: verifies an unclosed think tag should mark the answer as unavailable.
        Executes: Parser.extract_reasoning_and_tools() through the unclosed <think*> branch.
        Why: preserves truncated reasoning while making the missing final answer explicit.
        """
        parser = Parser(lambda generation: generation)
        result = parser.extract_reasoning_and_tools("<think_faster>still reasoning")
        assert result == {
            "reasoning": "still reasoning",
            "answer": None,
            "unclosed_think_tag": True,
        }

    def test_closed_tool_calls_are_collected_and_removed_from_answer(self):
        """What: verifies closed tool calls should be stripped when deriving the answer field.
        Executes: Parser.extract_reasoning_and_tools() through closed <tool_call> extraction.
        Why: separates tool payload metadata from the visible answer text.
        """
        parser = Parser(lambda generation: generation)
        result = parser.extract_reasoning_and_tools(
            "answer text <tool_call>{\"name\": \"calc\"}</tool_call>"
        )
        assert result == {
            "tool_calls": ['{"name": "calc"}'],
            "answer": "answer text",
        }

    def test_unclosed_tool_call_is_flagged(self):
        """What: verifies an unterminated tool call should preserve the partial payload and flag it.
        Executes: Parser.extract_reasoning_and_tools() through the unclosed <tool_call> branch.
        Why: marks truncated tool payloads so callers can distinguish them from clean answers.
        """
        parser = Parser(lambda generation: generation)
        result = parser.extract_reasoning_and_tools("<tool_call>{\"name\": \"calc\"}")
        assert result == {
            "tool_calls": ['{"name": "calc"}'],
            "unclosed_tool_call": True,
            "answer": None,
        }

    def test_think_answer_takes_precedence_over_tool_answer(self):
        """What: verifies tool extraction should not overwrite the answer already found after thinking.
        Executes: Parser.extract_reasoning_and_tools() with both a closed think block and a tool call.
        Why: preserves the post-think answer exactly when later tool metadata is also present.
        """
        parser = Parser(lambda generation: generation)
        result = parser.extract_reasoning_and_tools(
            "<think>plan</think> final <tool_call>ignored</tool_call>"
        )
        assert result["reasoning"] == "plan"
        assert result["answer"] == "final <tool_call>ignored</tool_call>"
        assert result["tool_calls"] == ["ignored"]
