"""
Tests for debug-mode raw text input capture and generation metadata
extraction (reasoning, tool_calls, answer).

Covers:
  - GraderBase.extract_reasoning_and_tools() in scheduler/grader/base.py
  - GraderBase.parse_generations() metadata population
  - OpenAIConnection debug flag: raw_text_input for BASE and CHAT models
  - OpenAIConnection reasoning/tool_calls extraction from API response
  - CLI --debug flag propagation
"""

import re
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from scheduler.grader.parser_registry import Parser as _Parser


def _extract_reasoning_and_tools(generation):
    """Test shim: call the public Parser method without a registered parser."""
    return _Parser.extract_reasoning_and_tools(None, generation)


# ---------------------------------------------------------------------------
# _extract_reasoning_and_tools — reasoning extraction
# ---------------------------------------------------------------------------

class TestExtractReasoningAndToolsReasoning:

    def test_standard_think_tags(self):
        gen = "<think>step 1, step 2</think>The answer is A"
        meta = _extract_reasoning_and_tools(gen)
        assert meta["reasoning"] == "step 1, step 2"
        assert meta["answer"] == "The answer is A"

    def test_think_fast_variant(self):
        gen = "<think_fast>quick reasoning</think_fast>B"
        meta = _extract_reasoning_and_tools(gen)
        assert meta["reasoning"] == "quick reasoning"
        assert meta["answer"] == "B"

    def test_think_faster_variant(self):
        gen = "<think_faster>speedy</think_faster>C"
        meta = _extract_reasoning_and_tools(gen)
        assert meta["reasoning"] == "speedy"
        assert meta["answer"] == "C"

    def test_case_insensitive(self):
        gen = "<THINK>upper case reasoning</THINK>answer here"
        meta = _extract_reasoning_and_tools(gen)
        assert meta["reasoning"] == "upper case reasoning"
        assert meta["answer"] == "answer here"

    def test_multiline_reasoning(self):
        gen = "<think>\nLine 1\nLine 2\n</think>\nFinal answer"
        meta = _extract_reasoning_and_tools(gen)
        assert "Line 1" in meta["reasoning"]
        assert "Line 2" in meta["reasoning"]
        assert meta["answer"] == "Final answer"

    def test_no_think_tags_no_reasoning_field(self):
        gen = "Just a plain answer with no thinking"
        meta = _extract_reasoning_and_tools(gen)
        assert "reasoning" not in meta
        assert "answer" not in meta

    def test_empty_think_block(self):
        gen = "<think></think>The answer"
        meta = _extract_reasoning_and_tools(gen)
        assert meta["reasoning"] == ""
        assert meta["answer"] == "The answer"

    def test_think_with_only_whitespace_answer(self):
        gen = "<think>reasoning</think>   "
        meta = _extract_reasoning_and_tools(gen)
        assert meta["reasoning"] == "reasoning"
        assert meta["answer"] is None

    def test_empty_generation(self):
        assert _extract_reasoning_and_tools("") == {}

    def test_none_generation(self):
        assert _extract_reasoning_and_tools(None) == {}

    def test_unclosed_think_tag_returns_remaining_as_reasoning(self):
        gen = "<think>This model ran out of tokens mid-thought"
        meta = _extract_reasoning_and_tools(gen)
        assert meta["reasoning"] == "This model ran out of tokens mid-thought"
        assert meta["answer"] is None
        assert meta["unclosed_think_tag"] is True

    def test_unclosed_think_variant(self):
        gen = "<think_fast>partial reasoning here"
        meta = _extract_reasoning_and_tools(gen)
        assert "partial reasoning here" in meta["reasoning"]
        assert meta["unclosed_think_tag"] is True

    def test_multiple_think_blocks_sets_warning(self):
        gen = "<think>first</think>middle<think>second</think>end"
        meta = _extract_reasoning_and_tools(gen)
        assert meta["multiple_think_blocks"] is True
        # Only first block captured
        assert meta["reasoning"] == "first"
        assert meta["answer"] == "middle<think>second</think>end"

    def test_single_think_block_no_multiple_warning(self):
        gen = "<think>only one</think>answer"
        meta = _extract_reasoning_and_tools(gen)
        assert "multiple_think_blocks" not in meta

    def test_orphaned_closing_think_tag(self):
        # Opening <think_fast> was in the prompt prefix; generation starts mid-block
        gen = "step 1, step 2\n</think_fast>\nThe answer is 42"
        meta = _extract_reasoning_and_tools(gen)
        assert meta["reasoning"] == "step 1, step 2"
        assert meta["answer"] == "The answer is 42"
        assert "unclosed_think_tag" not in meta

    def test_orphaned_closing_think_tag_no_answer(self):
        gen = "reasoning only\n</think>"
        meta = _extract_reasoning_and_tools(gen)
        assert meta["reasoning"] == "reasoning only"
        assert meta["answer"] is None

    def test_orphaned_closing_think_basic_variant(self):
        gen = "some reasoning</think>answer"
        meta = _extract_reasoning_and_tools(gen)
        assert meta["reasoning"] == "some reasoning"
        assert meta["answer"] == "answer"


# ---------------------------------------------------------------------------
# _extract_reasoning_and_tools — tool_call extraction
# ---------------------------------------------------------------------------

class TestExtractReasoningAndToolsToolCalls:

    def test_single_tool_call(self):
        gen = 'I need to search. <tool_call>{"name": "search", "args": {"q": "test"}}</tool_call> Done.'
        meta = _extract_reasoning_and_tools(gen)
        assert len(meta["tool_calls"]) == 1
        assert '"search"' in meta["tool_calls"][0]
        assert meta["answer"] is not None

    def test_multiple_tool_calls(self):
        gen = '<tool_call>call1</tool_call>middle<tool_call>call2</tool_call>end'
        meta = _extract_reasoning_and_tools(gen)
        assert len(meta["tool_calls"]) == 2
        assert meta["tool_calls"][0] == "call1"
        assert meta["tool_calls"][1] == "call2"

    def test_no_tool_calls_no_field(self):
        gen = "Plain text without any tool calls"
        meta = _extract_reasoning_and_tools(gen)
        assert "tool_calls" not in meta

    def test_tool_call_case_insensitive(self):
        gen = "<TOOL_CALL>func()</TOOL_CALL>result"
        meta = _extract_reasoning_and_tools(gen)
        assert len(meta["tool_calls"]) == 1

    def test_tool_call_answer_strips_tags(self):
        gen = "Before <tool_call>fn()</tool_call> After"
        meta = _extract_reasoning_and_tools(gen)
        assert "tool_call" not in meta["answer"].lower()
        assert "Before" in meta["answer"]
        assert "After" in meta["answer"]

    def test_unclosed_tool_call_captured(self):
        gen = "I will call <tool_call>search(query truncated"
        meta = _extract_reasoning_and_tools(gen)
        assert len(meta["tool_calls"]) == 1
        assert "search(query truncated" in meta["tool_calls"][0]
        assert meta["unclosed_tool_call"] is True

    def test_unclosed_tool_call_answer_is_none(self):
        gen = "<tool_call>fn(truncated"
        meta = _extract_reasoning_and_tools(gen)
        assert meta["answer"] is None
        assert meta["unclosed_tool_call"] is True


# ---------------------------------------------------------------------------
# _extract_reasoning_and_tools — combined reasoning + tool_calls
# ---------------------------------------------------------------------------

class TestExtractReasoningAndToolsCombined:

    def test_think_and_tool_call(self):
        gen = "<think>Let me think</think>I'll use a tool <tool_call>search()</tool_call> The answer is X"
        meta = _extract_reasoning_and_tools(gen)
        assert meta["reasoning"] == "Let me think"
        assert len(meta["tool_calls"]) == 1
        # answer comes from think tag (after </think>), not from tool stripping
        assert "I'll use a tool" in meta["answer"]

    def test_tool_call_only_sets_answer(self):
        gen = "Let me search <tool_call>search(q)</tool_call> Found it: 42"
        meta = _extract_reasoning_and_tools(gen)
        assert "reasoning" not in meta
        assert meta["tool_calls"] == ["search(q)"]
        assert "42" in meta["answer"]
        assert "tool_call" not in meta["answer"].lower()


# ---------------------------------------------------------------------------
# GraderBase.parse_generations metadata population
# ---------------------------------------------------------------------------

class TestParseGenerationsMetadata:

    def _make_grader(self, parser_type="noop"):
        """Create a minimal GraderBase subclass instance for testing."""
        from scheduler.grader.base import GraderBase

        class StubGrader(GraderBase):
            async def grade_sample(self, sample):
                return sample
            async def run(self):
                pass

        event = MagicMock()
        event.parser_type = parser_type
        return StubGrader(
            samples_generator=None,
            event_manager=MagicMock(),
            job_manager=MagicMock(),
            event=event,
            task=None,
        )

    def test_think_tags_populate_parsed_reasoning(self):
        grader = self._make_grader("noop")
        sample = {"generations": ["<think>reason</think>answer"]}
        grader.parse_generations(sample)
        assert sample["parsed_reasoning"] == ["reason"]
        assert sample["parsed_answer"] == ["answer"]

    def test_no_metadata_when_plain_text(self):
        grader = self._make_grader("noop")
        sample = {"generations": ["just a plain answer"]}
        grader.parse_generations(sample)
        assert "parsed_reasoning" not in sample
        assert "parsed_answer" not in sample
        assert "parsed_tool_calls" not in sample

    def test_mixed_generations_parallel_lists(self):
        """Generations with and without think tags produce parallel lists."""
        grader = self._make_grader("noop")
        sample = {
            "generations": [
                "<think>r1</think>a1",
                "plain text",
                "<think>r3</think>a3",
            ],
        }
        grader.parse_generations(sample)
        # All three generations have an entry — None for the plain one
        assert sample["parsed_reasoning"] == ["r1", None, "r3"]
        assert sample["parsed_answer"] == ["a1", None, "a3"]
        assert len(sample["parsed_reasoning"]) == 3

    def test_empty_generations_no_fields_added(self):
        grader = self._make_grader("noop")
        sample = {"generations": []}
        grader.parse_generations(sample)
        assert "parsed_reasoning" not in sample
        assert "parsed_tool_calls" not in sample

    def test_tool_calls_populate_parsed_tool_calls(self):
        grader = self._make_grader("noop")
        sample = {"generations": ["Let me <tool_call>search()</tool_call> check"]}
        grader.parse_generations(sample)
        assert sample["parsed_tool_calls"] == [["search()"]]
        assert "parsed_answer" in sample

    def test_api_level_reasoning_untouched(self):
        """API-level 'reasoning' field is separate from text-level 'parsed_reasoning'."""
        grader = self._make_grader("noop")
        sample = {
            "generations": ["<think>text-level</think>answer"],
            "reasoning": ["api-level-reasoning"],
        }
        grader.parse_generations(sample)
        assert sample["reasoning"] == ["api-level-reasoning"]
        assert sample["parsed_reasoning"] == ["text-level"]

    def test_parser_still_returns_correct_parsed_generations(self):
        grader = self._make_grader("think_tag")
        sample = {"generations": ["<think>thinking</think>The answer is B"]}
        parsed = grader.parse_generations(sample)
        assert parsed == ["The answer is B"]

    def test_unclosed_think_sets_flag_in_parallel_list(self):
        grader = self._make_grader("noop")
        sample = {
            "generations": [
                "<think>complete</think>done",
                "<think>truncated",
            ],
        }
        grader.parse_generations(sample)
        assert sample["parsed_unclosed_think_tag"] == [None, True]
        assert sample["parsed_reasoning"] == ["complete", "truncated"]


# ---------------------------------------------------------------------------
# OpenAIConnection debug mode — raw_text_input
# ---------------------------------------------------------------------------

class TestOpenAIConnectionDebugFlag:

    def test_debug_false_by_default(self):
        from scheduler.openai_interface import OpenAIConnection, LOCKED_CONNECTIONS
        from scheduler.task import AsyncGenerationTask
        model = MagicMock()
        model.name = "test-model-debug-default"
        model.model_type = MagicMock()
        model.max_simultaneous_requests = 4
        model.openai_kwargs = {}
        model.prompt_prefix_instructions = None

        task = MagicMock(spec=AsyncGenerationTask)
        task.openai_settings = None
        task.average_over = [1]
        task.pass_at = [1]
        task.dataset_name = "test-dataset"

        conn = OpenAIConnection(
            model=model,
            task=task,
            event_instance=MagicMock(),
            job_manager=MagicMock(),
            progress_manager=MagicMock(),
            new_field_name="generations",
        )
        assert conn._debug is False

    def test_debug_true_when_set(self):
        from scheduler.openai_interface import OpenAIConnection, LOCKED_CONNECTIONS
        from scheduler.task import AsyncGenerationTask
        model = MagicMock()
        model.name = "test-model-debug-true"
        model.model_type = MagicMock()
        model.max_simultaneous_requests = 4
        model.openai_kwargs = {}
        model.prompt_prefix_instructions = None

        task = MagicMock(spec=AsyncGenerationTask)
        task.openai_settings = None
        task.average_over = [1]
        task.pass_at = [1]
        task.dataset_name = "test-dataset"

        conn = OpenAIConnection(
            model=model,
            task=task,
            event_instance=MagicMock(),
            job_manager=MagicMock(),
            progress_manager=MagicMock(),
            new_field_name="generations",
            debug=True,
        )
        assert conn._debug is True


# ---------------------------------------------------------------------------
# OpenAIConnection — reasoning/tool_calls from API response
# ---------------------------------------------------------------------------

class TestOpenAIConnectionReasoningExtraction:

    def test_reasoning_from_model_extra(self):
        """Verify reasoning is extracted from choice.message.model_extra."""
        from scheduler.openai_interface import OpenAIConnection

        # Build a mock choice with reasoning in model_extra
        choice = MagicMock()
        choice.message.content = "The answer is 4"
        choice.message.reasoning_content = None  # not a standard field
        choice.message.model_extra = {"reasoning_content": "2+2=4 because addition"}
        choice.message.tool_calls = []
        choice.logprobs = None

        # Simulate what the code does for a single choice
        result = {"generations": []}
        result["generations"].append(choice.message.content)

        reasoning = getattr(choice.message, "reasoning_content", None)
        if not reasoning:
            extra = getattr(choice.message, "model_extra", None) or {}
            reasoning = extra.get("reasoning_content") or extra.get("reasoning")
        if reasoning:
            result.setdefault("reasoning", []).append(reasoning)

        assert result["reasoning"] == ["2+2=4 because addition"]

    def test_reasoning_from_reasoning_key(self):
        """VLLM may return 'reasoning' instead of 'reasoning_content'."""
        choice = MagicMock()
        choice.message.content = "answer"
        choice.message.reasoning_content = None
        choice.message.model_extra = {"reasoning": "my reasoning"}

        reasoning = getattr(choice.message, "reasoning_content", None)
        if not reasoning:
            extra = getattr(choice.message, "model_extra", None) or {}
            reasoning = extra.get("reasoning_content") or extra.get("reasoning")

        assert reasoning == "my reasoning"

    def test_no_reasoning_when_absent(self):
        choice = MagicMock()
        choice.message.content = "answer"
        choice.message.reasoning_content = None
        choice.message.model_extra = {}

        reasoning = getattr(choice.message, "reasoning_content", None)
        if not reasoning:
            extra = getattr(choice.message, "model_extra", None) or {}
            reasoning = extra.get("reasoning_content") or extra.get("reasoning")

        assert reasoning is None

    def test_tool_calls_extracted(self):
        """Verify tool_calls are extracted from choice.message.tool_calls."""
        tc = MagicMock()
        tc.model_dump.return_value = {"type": "function", "function": {"name": "search", "arguments": "{}"}}

        choice = MagicMock()
        choice.message.tool_calls = [tc]

        result = {"tool_calls": []}
        tool_calls = choice.message.tool_calls
        if tool_calls:
            result["tool_calls"].append([t.model_dump() for t in tool_calls])

        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0][0]["function"]["name"] == "search"

    def test_empty_tool_calls_not_added(self):
        choice = MagicMock()
        choice.message.tool_calls = []

        result = {}
        tool_calls = choice.message.tool_calls
        if tool_calls:
            result.setdefault("tool_calls", []).append([t.model_dump() for t in tool_calls])

        assert "tool_calls" not in result


# ---------------------------------------------------------------------------
# CLI --debug flag
# ---------------------------------------------------------------------------

class TestCLIDebugFlag:

    def test_debug_before_subcommand(self):
        """--debug before the subcommand is accepted."""
        import sys
        with patch.object(sys, 'argv', [
            'eval360',
            '--max-generation-jobs', '1',
            '--max-grading-parallelism', '1',
            '--debug',
            'evaluate-now',
            '--model-paths', 'foo.yaml',
            '--data-paths', 'bar.yaml',
        ]):
            with patch('scheduler.cli.run_evaluation') as mock_run:
                from scheduler.cli import main
                main()
                _, kwargs = mock_run.call_args
                assert kwargs["debug"] is True

    def test_debug_after_subcommand(self):
        """--debug after the subcommand is accepted."""
        import sys
        with patch.object(sys, 'argv', [
            'eval360',
            '--max-generation-jobs', '1',
            '--max-grading-parallelism', '1',
            'evaluate-now',
            '--model-paths', 'foo.yaml',
            '--data-paths', 'bar.yaml',
            '--debug',
        ]):
            with patch('scheduler.cli.run_evaluation') as mock_run:
                from scheduler.cli import main
                main()
                _, kwargs = mock_run.call_args
                assert kwargs["debug"] is True

    def test_max_generation_jobs_after_subcommand(self):
        """--max-generation-jobs after the subcommand is accepted."""
        import sys
        with patch.object(sys, 'argv', [
            'eval360',
            'evaluate-now',
            '--model-paths', 'foo.yaml',
            '--data-paths', 'bar.yaml',
            '--max-generation-jobs', '4',
            '--max-grading-parallelism', '20',
        ]):
            with patch('scheduler.cli.run_evaluation') as mock_run:
                from scheduler.cli import main
                main()
                _, kwargs = mock_run.call_args
                assert kwargs["max_generation_jobs"] == 4
                assert kwargs["max_grading_parallelism"] == 20

    def test_global_flags_after_long_running_subcommand(self):
        """Global flags after long-running-scheduler subcommand are accepted."""
        import sys
        with patch.object(sys, 'argv', [
            'eval360',
            'long-running-scheduler',
            '--model-registration-path', '/models',
            '--data-registration-path', '/data',
            '--max-generation-jobs', '4',
            '--max-grading-parallelism', '20',
            '--debug',
        ]):
            with patch('scheduler.cli.run_scheduler') as mock_run:
                from scheduler.cli import main
                main()
                _, kwargs = mock_run.call_args
                assert kwargs["debug"] is True
                assert kwargs["max_generation_jobs"] == 4

    def test_debug_default_false(self):
        import sys
        with patch.object(sys, 'argv', [
            'eval360',
            '--max-generation-jobs', '1',
            '--max-grading-parallelism', '1',
            'evaluate-now',
            '--model-paths', 'foo.yaml',
            '--data-paths', 'bar.yaml',
        ]):
            with patch('scheduler.cli.run_evaluation') as mock_run:
                from scheduler.cli import main
                main()
                _, kwargs = mock_run.call_args
                assert kwargs["debug"] is False

    def test_salt_cache_after_subcommand(self):
        import sys
        with patch.object(sys, 'argv', [
            'eval360',
            '--max-generation-jobs', '1',
            '--max-grading-parallelism', '1',
            'evaluate-now',
            '--model-paths', 'foo.yaml',
            '--data-paths', 'bar.yaml',
            '--force',
            '--salt-cache',
        ]):
            with patch('scheduler.cli.run_evaluation') as mock_run:
                from scheduler.cli import main
                main()
                _, kwargs = mock_run.call_args
                assert kwargs["salt_cache"] is True

    def test_salt_cache_requires_force(self):
        import sys
        with patch.object(sys, 'argv', [
            'eval360',
            '--max-generation-jobs', '1',
            '--max-grading-parallelism', '1',
            'evaluate-now',
            '--model-paths', 'foo.yaml',
            '--data-paths', 'bar.yaml',
            '--salt-cache',
        ]):
            from scheduler.cli import main
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code != 0

    def test_salt_cache_default_false(self):
        import sys
        with patch.object(sys, 'argv', [
            'eval360',
            '--max-generation-jobs', '1',
            '--max-grading-parallelism', '1',
            'evaluate-now',
            '--model-paths', 'foo.yaml',
            '--data-paths', 'bar.yaml',
        ]):
            with patch('scheduler.cli.run_evaluation') as mock_run:
                from scheduler.cli import main
                main()
                _, kwargs = mock_run.call_args
                assert kwargs["salt_cache"] is False

    def test_abbreviated_salt_flag_rejected(self):
        import sys
        with patch.object(sys, 'argv', [
            'eval360',
            '--max-generation-jobs', '1',
            '--max-grading-parallelism', '1',
            'evaluate-now',
            '--model-paths', 'foo.yaml',
            '--data-paths', 'bar.yaml',
            '--salt',
        ]):
            from scheduler.cli import main
            with pytest.raises(SystemExit) as excinfo:
                main()
            assert excinfo.value.code != 0

    def test_omitted_parallelism_is_forwarded_for_evaluate_now(self):
        """One-shot concurrency is derived after the request is parsed."""
        import sys
        with patch.object(sys, 'argv', [
            'eval360',
            'evaluate-now',
            '--model-paths', 'foo.yaml',
            '--data-paths', 'bar.yaml',
        ]):
            with patch('scheduler.cli.run_evaluation') as mock_run:
                from scheduler.cli import main
                main()
                _, kwargs = mock_run.call_args
                assert kwargs["max_generation_jobs"] is None
                assert kwargs["max_grading_parallelism"] is None


# ---------------------------------------------------------------------------
# OpenAIConnection._get_templated_prompt
# ---------------------------------------------------------------------------

class TestGetTemplatedPrompt:

    @pytest.mark.asyncio
    async def test_returns_prompt_from_detokenize(self):
        from scheduler.openai_interface import OpenAIConnection
        import json

        model = MagicMock()
        model.name = "test-model-template"
        model.model_type = MagicMock()
        model.max_simultaneous_requests = 4
        model.openai_kwargs = {}
        model.prompt_prefix_instructions = None
        model.is_external = False

        task = MagicMock()
        task.openai_settings = None
        task.average_over = [1]
        task.pass_at = [1]

        conn = OpenAIConnection(
            model=model, task=task, event_instance=MagicMock(),
            job_manager=MagicMock(), progress_manager=MagicMock(),
            new_field_name="generations", debug=True,
        )
        conn._url = "http://fake:8000"

        tokenize_resp = MagicMock()
        tokenize_resp.status = 200
        tokenize_resp.json = AsyncMock(return_value={"tokens": [1, 2, 3], "count": 3})
        tokenize_resp.__aenter__ = AsyncMock(return_value=tokenize_resp)
        tokenize_resp.__aexit__ = AsyncMock(return_value=False)

        detokenize_resp = MagicMock()
        detokenize_resp.status = 200
        detokenize_resp.json = AsyncMock(return_value={"prompt": "<|start|>user<|message|>Hello<|end|>"})
        detokenize_resp.__aenter__ = AsyncMock(return_value=detokenize_resp)
        detokenize_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        call_count = 0

        def mock_post(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tokenize_resp
            return detokenize_resp

        mock_session.post = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("scheduler.openai_interface.aiohttp.ClientSession", return_value=mock_session):
            result = await conn._get_templated_prompt("http://fake:8000", [{"role": "user", "content": "Hello"}])

        assert result == "<|start|>user<|message|>Hello<|end|>"

    @pytest.mark.asyncio
    async def test_chat_template_kwargs_forwarded_to_tokenize(self):
        """chat_template_kwargs from extra_body must be sent to /tokenize."""
        from scheduler.openai_interface import OpenAIConnection

        model = MagicMock()
        model.name = "test-model-ctk"
        model.model_type = MagicMock()
        model.max_simultaneous_requests = 4
        model.openai_kwargs = {"extra_body": {"chat_template_kwargs": {"reasoning_effort": "high"}}}
        model.prompt_prefix_instructions = None
        model.is_external = False

        task = MagicMock()
        task.openai_settings = None

        conn = OpenAIConnection(
            model=model, task=task, event_instance=MagicMock(),
            job_manager=MagicMock(), progress_manager=MagicMock(),
            new_field_name="generations", debug=True,
        )
        conn._url = "http://fake:8000"

        captured_tokenize_body = {}

        tokenize_resp = MagicMock()
        tokenize_resp.status = 200
        tokenize_resp.json = AsyncMock(return_value={"tokens": [1, 2, 3]})
        tokenize_resp.__aenter__ = AsyncMock(return_value=tokenize_resp)
        tokenize_resp.__aexit__ = AsyncMock(return_value=False)

        detokenize_resp = MagicMock()
        detokenize_resp.status = 200
        detokenize_resp.json = AsyncMock(return_value={"prompt": "<think>\n"})
        detokenize_resp.__aenter__ = AsyncMock(return_value=detokenize_resp)
        detokenize_resp.__aexit__ = AsyncMock(return_value=False)

        call_count = 0
        mock_session = MagicMock()

        def mock_post(url, json=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                captured_tokenize_body.update(json or {})
                return tokenize_resp
            return detokenize_resp

        mock_session.post = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("scheduler.openai_interface.aiohttp.ClientSession", return_value=mock_session):
            result = await conn._get_templated_prompt("http://fake:8000", [{"role": "user", "content": "Hi"}])

        assert result == "<think>\n"
        assert captured_tokenize_body.get("chat_template_kwargs") == {"reasoning_effort": "high"}

    @pytest.mark.asyncio
    async def test_flat_extra_body_forwarded_as_chat_template_kwargs(self):
        """Flat extra_body (e.g. reasoning_effort directly) must be forwarded to /tokenize
        as chat_template_kwargs — this is the correct format for this VLLM version."""
        from scheduler.openai_interface import OpenAIConnection

        model = MagicMock()
        model.name = "test-model-flat"
        model.model_type = MagicMock()
        model.max_simultaneous_requests = 4
        model.openai_kwargs = {"extra_body": {"reasoning_effort": "high"}}
        model.prompt_prefix_instructions = None
        model.is_external = False

        task = MagicMock()
        task.openai_settings = None

        conn = OpenAIConnection(
            model=model, task=task, event_instance=MagicMock(),
            job_manager=MagicMock(), progress_manager=MagicMock(),
            new_field_name="generations", debug=True,
        )
        conn._url = "http://fake:8000"

        captured_tokenize_body = {}
        call_count = 0

        tokenize_resp = MagicMock()
        tokenize_resp.status = 200
        tokenize_resp.json = AsyncMock(return_value={"tokens": [1, 2, 3]})
        tokenize_resp.__aenter__ = AsyncMock(return_value=tokenize_resp)
        tokenize_resp.__aexit__ = AsyncMock(return_value=False)

        detokenize_resp = MagicMock()
        detokenize_resp.status = 200
        detokenize_resp.json = AsyncMock(return_value={"prompt": "<think>\n"})
        detokenize_resp.__aenter__ = AsyncMock(return_value=detokenize_resp)
        detokenize_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()

        def mock_post(url, json=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                captured_tokenize_body.update(json or {})
                return tokenize_resp
            return detokenize_resp

        mock_session.post = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("scheduler.openai_interface.aiohttp.ClientSession", return_value=mock_session):
            result = await conn._get_templated_prompt("http://fake:8000", [{"role": "user", "content": "Hi"}])

        assert result == "<think>\n"
        assert captured_tokenize_body.get("chat_template_kwargs") == {"reasoning_effort": "high"}

    @pytest.mark.asyncio
    async def test_no_chat_template_kwargs_when_empty_extra_body(self):
        """When extra_body is empty, /tokenize body must not include chat_template_kwargs."""
        from scheduler.openai_interface import OpenAIConnection

        model = MagicMock()
        model.name = "test-model-no-ctk"
        model.model_type = MagicMock()
        model.max_simultaneous_requests = 4
        model.openai_kwargs = {}
        model.prompt_prefix_instructions = None
        model.is_external = False

        task = MagicMock()
        task.openai_settings = None

        conn = OpenAIConnection(
            model=model, task=task, event_instance=MagicMock(),
            job_manager=MagicMock(), progress_manager=MagicMock(),
            new_field_name="generations", debug=True,
        )
        conn._url = "http://fake:8000"

        captured_tokenize_body = {}
        call_count = 0

        tokenize_resp = MagicMock()
        tokenize_resp.status = 200
        tokenize_resp.json = AsyncMock(return_value={"tokens": [1, 2, 3]})
        tokenize_resp.__aenter__ = AsyncMock(return_value=tokenize_resp)
        tokenize_resp.__aexit__ = AsyncMock(return_value=False)

        detokenize_resp = MagicMock()
        detokenize_resp.status = 200
        detokenize_resp.json = AsyncMock(return_value={"prompt": "Hi"})
        detokenize_resp.__aenter__ = AsyncMock(return_value=detokenize_resp)
        detokenize_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()

        def mock_post(url, json=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                captured_tokenize_body.update(json or {})
                return tokenize_resp
            return detokenize_resp

        mock_session.post = mock_post
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("scheduler.openai_interface.aiohttp.ClientSession", return_value=mock_session):
            await conn._get_templated_prompt("http://fake:8000", [{"role": "user", "content": "Hi"}])

        assert "chat_template_kwargs" not in captured_tokenize_body

    @pytest.mark.asyncio
    async def test_returns_none_on_tokenize_failure(self):
        from scheduler.openai_interface import OpenAIConnection

        model = MagicMock()
        model.name = "test-model-fail"
        model.model_type = MagicMock()
        model.max_simultaneous_requests = 4
        model.openai_kwargs = {}
        model.prompt_prefix_instructions = None
        model.is_external = False

        task = MagicMock()
        task.openai_settings = None
        task.average_over = [1]
        task.pass_at = [1]

        conn = OpenAIConnection(
            model=model, task=task, event_instance=MagicMock(),
            job_manager=MagicMock(), progress_manager=MagicMock(),
            new_field_name="generations", debug=True,
        )
        conn._url = "http://fake:8000"

        tokenize_resp = MagicMock()
        tokenize_resp.status = 500
        tokenize_resp.__aenter__ = AsyncMock(return_value=tokenize_resp)
        tokenize_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=tokenize_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("scheduler.openai_interface.aiohttp.ClientSession", return_value=mock_session):
            result = await conn._get_templated_prompt("http://fake:8000", [{"role": "user", "content": "Hello"}])

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_network_error(self):
        from scheduler.openai_interface import OpenAIConnection
        import aiohttp

        model = MagicMock()
        model.name = "test-model-err"
        model.model_type = MagicMock()
        model.max_simultaneous_requests = 4
        model.openai_kwargs = {}
        model.prompt_prefix_instructions = None
        model.is_external = False

        task = MagicMock()
        task.openai_settings = None
        task.average_over = [1]
        task.pass_at = [1]

        conn = OpenAIConnection(
            model=model, task=task, event_instance=MagicMock(),
            job_manager=MagicMock(), progress_manager=MagicMock(),
            new_field_name="generations", debug=True,
        )
        conn._url = "http://fake:8000"

        with patch("scheduler.openai_interface.aiohttp.ClientSession", side_effect=Exception("connection refused")):
            result = await conn._get_templated_prompt("http://fake:8000", [{"role": "user", "content": "Hello"}])

        assert result is None
