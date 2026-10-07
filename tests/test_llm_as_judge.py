"""Tests for the LLM-as-judge graders."""

from types import SimpleNamespace

import pytest

from scheduler.cache_salt import CacheSaltConfig
from scheduler.grader.llm_as_judge import LLMasJudgeBoxedMatch, LLMasJudgeLCRMatch


class LLMJudgeTestBase:
    """Shared LLM judge setup for prompt, parsing, and async grading tests.

    What: centralizes fake OpenAI client construction for judge graders.
    Executes: the same chat-completions interface used by `grade_sample`.
    Why: keeps external API behavior deterministic while testing grader logic.
    """

    @staticmethod
    def make_grader(grader_cls=LLMasJudgeBoxedMatch, *, responses=None, model=None):
        """Return a judge grader with a fake async OpenAI client.

        What: creates a partially initialized judge grader and records API calls.
        Executes: the grader's OpenAI connection path without real network requests.
        Why: lets tests assert prompts and response parsing through stable fakes.
        """
        responses = list(responses or [])
        calls = []

        class FakeCompletions:
            """Async chat-completions stub that records calls and returns queued content."""

            async def create(self, **kwargs):
                calls.append(kwargs)
                content = responses.pop(0)
                message = SimpleNamespace(content=content)
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

        class FakeConnection:
            """Async connection stub that returns the fake client."""

            async def get_client(self):
                return client

        grader = grader_cls.__new__(grader_cls)
        grader.model = model or SimpleNamespace(
            api_model_name=None,
            name="judge-model",
            openai_kwargs={"temperature": 0},
            cache_salt=CacheSaltConfig(),
        )
        if not hasattr(grader.model, "cache_salt"):
            grader.model.cache_salt = CacheSaltConfig()
        grader.openai_connection = FakeConnection()
        return grader, calls


class TestJudgePromptConstruction(LLMJudgeTestBase):
    """What: groups tests for building judge prompts from samples and generations.
    Executes: LLMasJudge construction and prompt builders without calling a real judge model.
    Why: keeps model-registration and prompt-format contracts close to the samples that exercise them.
    """

    def test_constructor_registers_judge_model_connection(self):
        """What: verifies the constructor should request an OpenAI connection for the judge model.
        Executes: LLMasJudgeBoxedMatch.__init__() with a task-level llm_as_judge model and fake event manager.
        Why: covers core connection registration so judge models are requested through the grading channel.
        """
        calls = []
        model = SimpleNamespace(name="judge", api_model_name=None, openai_kwargs={})
        task = SimpleNamespace(grader=SimpleNamespace(llm_as_judge=model))
        event = SimpleNamespace(parser_type="noop")
        event_manager = SimpleNamespace(
            add_desired_model=lambda *args, **kwargs: calls.append((args, kwargs))
        )

        async def samples():
            """Yield no samples; construction should not consume the generator."""
            if False:
                yield {}

        grader = LLMasJudgeBoxedMatch(samples(), event_manager, None, event, task=task)

        assert grader.name == "llm_as_judge"
        assert grader.model is model
        assert grader.openai_connection is not None
        assert calls == [((event,), {"is_grading": True})]

    def test_boxed_match_prompt_includes_question_generation_and_answer(self):
        """What: verifies the default prompt should include all grading context fields.
        Executes: LLMasJudgeBoxedMatch.create_judge_messages() with question, student answer, and ground truth fields.
        Why: covers the prompt contract sent to the judge model for the standard boxed-match grader.
        """
        grader, _ = self.make_grader()
        messages = grader.create_judge_messages(
            {
                "completion_input": "What is 2+2?",
                "ground_truth": "4",
            },
            "four",
        )

        assert messages[0]["role"] == "system"
        assert "output JSON" in messages[0]["content"]
        user_message = messages[1]["content"]
        assert "QUESTION: What is 2+2?" in user_message
        assert "STUDENT ANSWER: four" in user_message
        assert "TRUE ANSWER: 4" in user_message

    def test_lcr_prompt_strips_question_marker_and_answer_suffix(self):
        """What: verifies the LCR variant should trim benchmark framing before judging.
        Executes: LLMasJudgeLCRMatch.create_judge_messages() on a LongCodeReasoning-style framed prompt.
        Why: exercises the benchmark-specific cleanup that removes headers and trailing answer markers before judging.
        """
        grader, _ = self.make_grader(LLMasJudgeLCRMatch)
        messages = grader.create_judge_messages(
            {
                "completion_input": "Header\n=== QUESTION ===\nActual question?\n\nAnswer:",
                "ground_truth": "yes",
            },
            "yes",
        )

        user_message = messages[1]["content"]
        assert "Header" not in user_message
        assert "QUESTION: Actual question?" in user_message
        assert "\n\nAnswer:" not in user_message


class TestJudgeResponseParsing(LLMJudgeTestBase):
    """What: groups tests for parsing JSON/YAML judge responses into binary scores.
    Executes: parse_judge_response() over supported response shapes and malformed judge output.
    Why: keeps strict and fallback parsing behavior in one table-driven fixture.
    """

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            ('{"GRADE": "CORRECT"}', 1),
            ('{"grade": "incorrect"}', 0),
            ('</think> {"GRADE": "CORRECT"}', 1),
            ('{"answer": {"GRADE": "CORRECT"}}', 1),
            ('{"answer": {"grade": "INCORRECT"}}', 0),
            ("not: [valid", 0),
            ("just a string", 0),
            ('{"answer": "CORRECT"}', 0),
            ('{"reason": "missing grade"}', 0),
        ],
    )
    def test_parse_judge_response_handles_supported_and_bad_shapes(self, response, expected):
        """What: verifies known response shapes should parse, while malformed ones score incorrect.
        Executes: LLMasJudgeBoxedMatch.parse_judge_response() for uppercase, lowercase, nested, and invalid payloads.
        Why: covers both core judge JSON parsing and the defensive fallback that treats unparsable output as incorrect.
        """
        grader, _ = self.make_grader()
        assert grader.parse_judge_response(response) == expected


class TestGradeSample(LLMJudgeTestBase):
    """What: groups tests for the async OpenAI judging path.
    Executes: grade_sample() against the fake async chat-completions client.
    Why: keeps network-free tests for per-generation judging and model-name selection together.
    """

    @pytest.mark.asyncio
    async def test_grade_sample_calls_judge_for_each_generation(self):
        """What: verifies every parsed generation should be judged and aggregated into accuracy.
        Executes: LLMasJudgeBoxedMatch.grade_sample() with two generations and queued correct/incorrect judge replies.
        Why: covers the core async grading loop, prompt fallback for blank parsed text, and accuracy aggregation.
        """
        model = SimpleNamespace(
            api_model_name="api-judge",
            name="configured-judge",
            openai_kwargs={"temperature": 0, "max_tokens": 7},
        )
        grader, calls = self.make_grader(
            responses=['{"GRADE": "CORRECT"}', '{"GRADE": "INCORRECT"}'],
            model=model,
        )
        sample = {
            "completion_input": "Question?",
            "ground_truth": "Answer",
            "generations": ["raw one", "raw two"],
            "parsed_generations": ["parsed one", ""],
        }

        result = await grader.grade_sample(sample)

        assert result is not sample
        assert result["correct"] == [1, 0]
        assert result["accuracy"] == 0.5
        assert [call["model"] for call in calls] == ["api-judge", "api-judge"]
        assert calls[0]["temperature"] == 0
        assert calls[0]["max_tokens"] == 7
        assert "STUDENT ANSWER: parsed one" in calls[0]["messages"][1]["content"]
        assert "STUDENT ANSWER: No answer" in calls[1]["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_grade_sample_falls_back_to_model_name_without_api_name(self):
        """What: verifies the configured model name should be used when api_model_name is absent.
        Executes: LLMasJudgeBoxedMatch.grade_sample() with a judge model lacking api_model_name.
        Why: exercises the model-name fallback so local model configs still produce a valid chat-completions call.
        """
        grader, calls = self.make_grader(responses=['{"GRADE": "CORRECT"}'])
        sample = {
            "completion_input": "Question?",
            "ground_truth": "Answer",
            "generations": ["raw"],
            "parsed_generations": ["parsed"],
        }

        result = await grader.grade_sample(sample)

        assert result["correct"] == [1]
        assert calls[0]["model"] == "judge-model"
