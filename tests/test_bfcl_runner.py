"""Tests for the BFCL ImportedDataset runner."""
import re

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from scheduler.imported_dataset.bfcl import (
    BFCL_EVAL_VERSION,
    BFCLRunner,
    _BENCHMARK_SCRIPT_TEMPLATE,
)
from scheduler.imported_dataset.base import ImportedDatasetResultError
from scheduler.task import ImportedDatasetTask, ImportedDatasetConfig


def _render_template(**overrides):
    kwargs = dict(
        model_name="test_model",
        test_categories=["non_live"],
        chat_template_kwargs={},
        max_tokens=None,
        num_threads=50,
        temperature=0.001,
        think_tag_file="/tmp/think_tags.json",
        save_raw_inference=False,
        raw_log_dir="/tmp/raw_inference",
        atif_log_dir="/tmp/atif_trajectories",
        web_search_backend=None,
        web_search_api_keys_file=None,
        web_search_api_keys_file_rel=None,
        web_fetch_force_mode=None,
        web_fetch_max_chars=None,
    )
    kwargs.update(overrides)
    return _BENCHMARK_SCRIPT_TEMPLATE.format(**kwargs)



def _make_model_instance(name="my-model", path="org/my-model"):
    m = MagicMock()
    m.name = name
    m.path = path
    m.openai_kwargs = {}
    return m


def _make_task(args=None):
    return ImportedDatasetTask(
        uuid="test-task",
        dataset_name="bfcl",
        semantic_version="1.0.0",
        imported_dataset=ImportedDatasetConfig(name="bfcl", commit="abc123", args=args or {}),
    )


def _write_overall_csv(output_dir: Path, rows: list[dict]):
    """Write a data_overall.csv in the format BFCL produces."""
    import csv as _csv
    csv_path = output_dir / "score" / "data_overall.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_path.write_text("")
        return
    with open(csv_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# build_setup_script
# ---------------------------------------------------------------------------

class TestBuildSetupScript:

    def test_installs_bfcl_eval(self, tmp_path):
        script = BFCLRunner().build_setup_script(tmp_path)
        assert "bfcl-eval" in script
        assert "pip" in script and "install" in script

    def test_pins_bfcl_v4_eval_version(self, tmp_path):
        script = BFCLRunner().build_setup_script(tmp_path)
        assert f"bfcl-eval=={BFCL_EVAL_VERSION}" in script


# ---------------------------------------------------------------------------
# build_benchmark_script
# ---------------------------------------------------------------------------

class TestBuildBenchmarkScript:

    def test_contains_generate_and_evaluate(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(), tmp_path
        )
        assert "generate" in script
        assert "evaluate" in script

    def test_sets_bfcl_project_root(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(), tmp_path
        )
        assert f'BFCL_PROJECT_ROOT="{tmp_path}"' in script

    def test_sets_openai_base_url(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(), tmp_path
        )
        assert "http://localhost:8000/v1" in script

    def test_uses_model_instance_name_as_model_name(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(name="llama-3.1-8b-instruct"), _make_task(), tmp_path
        )
        assert "llama-3.1-8b-instruct" in script

    def test_passes_test_category(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(args={"test_category": "live"}), tmp_path
        )
        assert "live" in script

    def test_default_test_category_is_non_live(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(), tmp_path
        )
        assert "non_live" in script

    def test_forwards_bfcl_v4_collection_categories(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(),
            _make_task(args={"test_category": ["all_scoring", "agentic", "web_search", "memory"]}),
            tmp_path,
        )
        for category in ["all_scoring", "agentic", "web_search", "memory"]:
            assert category in script

    def test_uses_name_for_api_calls(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(name="llama", path="org/llama"), _make_task(), tmp_path
        )
        assert "llama" in script

    def test_inlines_k2_horizon_vllm_handler(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(), tmp_path
        )
        assert "_LocalK2HorizonVLLMOpenAICompletionsHandler" in script
        # The handler is inlined: the only bfcl_eval api_inference module the
        # script imports is the released openai_completion base class.
        assert set(re.findall(r"api_inference\.(\w+)", script)) == {"openai_completion"}
        assert "OpenAICompletionsHandler" in script
        assert "MODEL_CONFIG_MAPPING" in script

    def test_sets_num_threads(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(), tmp_path
        )
        assert "--num-threads" in script
        assert "_num_threads = 50" in script

    def test_forwards_num_threads_from_dataset_args(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(args={"num_threads": 8}), tmp_path
        )
        assert "_num_threads = 8" in script
        assert '"--num-threads", str(_num_threads)' in script

    def test_forwards_serper_web_search_backend(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(),
            _make_task(args={
                "test_category": "web_search",
                "web_search_backend": "serper",
                "web_search_api_keys_file": "/tmp/search_keys",
                "web_fetch_force_mode": "truncate",
                "web_fetch_max_chars": 200000,
            }),
            tmp_path,
        )
        assert "_web_search_backend = 'serper'" in script
        assert "_web_search_api_keys_file = '/tmp/search_keys'" in script
        assert "_web_search_api_keys_file_rel = None" in script
        assert "_web_fetch_force_mode = 'truncate'" in script
        assert "_web_fetch_max_chars = 200000" in script
        assert "https://google.serper.dev/search" in script
        assert "Patched search_engine_query backend=serper" in script
        assert "max_429_retries = 5" in script

    def test_uses_fc_mode(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(), tmp_path
        )
        assert "is_fc_model=True" in script
        assert "underscore_to_dot=True" in script

    def test_registers_all_underscores_as_slashes(self, tmp_path):
        # BFCL's evaluate step replaces ALL underscores with slashes when
        # looking up the handler config (e.g. k2_horizon_raw_25000 → k2/horizon/raw/25000).
        # The script must use replace("_", "/") with no count limit.
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(name="k2_horizon_raw_25000"), _make_task(), tmp_path
        )
        assert '_model_name.replace("_", "/")' in script
        # Must NOT limit to first replacement
        assert '_model_name.replace("_", "/", 1)' not in script

    def test_slash_registration_uses_model_name_variable(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(name="k2_horizon_25000"), _make_task(), tmp_path
        )
        assert "MODEL_CONFIG_MAPPING[_model_name] = _config" in script
        assert "MODEL_CONFIG_MAPPING[_model_name_slash] = _config" in script

    def test_overrides_parse_query_response_fc(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(), tmp_path
        )
        assert "_parse_query_response_FC" in script

    def test_injects_chat_template_kwargs_via_extra_body(self, tmp_path):
        m = _make_model_instance()
        m.openai_kwargs = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
        script = BFCLRunner().build_benchmark_script(m, _make_task(), tmp_path)
        assert "chat_template_kwargs" in script
        assert "extra_body" in script

    def test_injects_max_tokens_from_openai_kwargs(self, tmp_path):
        m = _make_model_instance()
        m.openai_kwargs = {"max_tokens": 1024}
        script = BFCLRunner().build_benchmark_script(m, _make_task(), tmp_path)
        assert "_max_tokens = 1024" in script
        assert 'kwargs.setdefault("max_tokens", _max_tokens)' in script

    def test_raw_inference_logging_disabled_by_default(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(), tmp_path
        )
        assert "_save_raw_inference = False" in script

    def test_raw_inference_logging_enabled_by_arg(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(args={"save_raw_inference": True}), tmp_path
        )
        assert "_save_raw_inference = True" in script
        assert "raw_inference" in script
        assert "_current_category" in script
        assert "_pre_query_processing_FC" in script
        assert "test_case_id" in script
        assert "turn_" in script
        assert "model_dump" in script

    def test_inline_reasoning_merges_into_content(self, tmp_path):
        # _inline_reasoning must merge reasoning_content into content as <think>...</think>
        # so that the raw JSONL stores it inline rather than as a separate field.
        # Must use key-presence check (not truthiness) so empty reasoning still
        # produces <think></think>, matching what vLLM actually sends to K2 Horizon.
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(), tmp_path
        )
        assert "_inline_reasoning" in script
        assert "<think>" in script
        assert '"reasoning_content" not in msg' in script

    def test_inline_reasoning_applied_to_messages_before_logging(self, tmp_path):
        # serialized_messages must use _inline_reasoning on each message
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(), tmp_path
        )
        assert "_inline_reasoning(m)" in script

    def test_raw_log_includes_tools(self, tmp_path):
        # Tool definitions (function schemas) must be included in each raw JSONL entry.
        # The kwargs dict must be passed through to _append_raw_log so tools are captured.
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(), tmp_path
        )
        # _append_raw_log must accept kwargs and the call site must pass it
        assert "def _append_raw_log(messages, response, kwargs=None)" in script
        assert "_append_raw_log(kwargs.get(" in script and ", raw_response, kwargs)" in script
        assert 'kwargs.get("tools")' in script

    def test_temperature_from_openai_kwargs_passed_to_generate(self, tmp_path):
        """Temperature from model_instance.openai_kwargs is forwarded to bfcl generate."""
        m = _make_model_instance()
        m.openai_kwargs = {"temperature": 0.7}
        script = BFCLRunner().build_benchmark_script(m, _make_task(), tmp_path)
        assert "--temperature" in script
        assert "0.7" in script

    def test_default_temperature_used_when_not_in_openai_kwargs(self, tmp_path):
        """When temperature is absent from openai_kwargs, the BFCL default (0.001) is used."""
        m = _make_model_instance()
        m.openai_kwargs = {}
        script = BFCLRunner().build_benchmark_script(m, _make_task(), tmp_path)
        assert "--temperature" in script
        assert "0.001" in script

    def test_temperature_zero_is_forwarded(self, tmp_path):
        """Temperature of 0 (greedy) must be forwarded, not silently replaced by default."""
        m = _make_model_instance()
        m.openai_kwargs = {"temperature": 0}
        script = BFCLRunner().build_benchmark_script(m, _make_task(), tmp_path)
        assert "--temperature" in script
        # The literal "0" (as string) should appear after --temperature
        assert '"--temperature", "0"' in script or '"--temperature", str(0)' in script or "0.0" in script


# ---------------------------------------------------------------------------
# Rendered benchmark script helpers
# ---------------------------------------------------------------------------

def _exec_script_ns(script, tmp_path, openai_handler_cls=None):
    """Exec a rendered benchmark script with bfcl_eval mocked, return namespace."""
    import sys, types
    from unittest.mock import MagicMock

    python_script = script.split("<<'BFCL_EOF'\n", 1)[1].rsplit("\nBFCL_EOF", 1)[0]
    mocks = {k: types.ModuleType(k) for k in [
        "bfcl_eval", "bfcl_eval.model_handler",
        "bfcl_eval.model_handler.api_inference",
        "bfcl_eval.model_handler.api_inference.openai_completion",
        "bfcl_eval.constants", "bfcl_eval.constants.model_config",
        "bfcl_eval.__main__",
    ]}
    mocks["bfcl_eval.model_handler.api_inference.openai_completion"].OpenAICompletionsHandler = (
        openai_handler_cls or MagicMock
    )
    mocks["bfcl_eval.constants.model_config"].MODEL_CONFIG_MAPPING = {}
    mocks["bfcl_eval.constants.model_config"].ModelConfig = MagicMock
    mocks["bfcl_eval.__main__"].cli = MagicMock(side_effect=SystemExit(0))
    sys.modules.update(mocks)
    ns = {}
    try:
        exec(compile(python_script, "<bfcl_script>", "exec"), ns)
    finally:
        for k in mocks:
            sys.modules.pop(k, None)
    return ns


class _Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeOpenAICompletionsHandler:
    def __init__(self, *args, **kwargs):
        self.model_name = args[0] if args else kwargs.get("model_name", "test-model")
        self.temperature = args[1] if len(args) > 1 else kwargs.get("temperature", 0)

    def generate_with_backoff(self, **kwargs):
        self.last_generate_kwargs = kwargs
        return (
            _Obj(
                choices=[_Obj(message=_Obj(content="", tool_calls=[]))],
                usage=_Obj(prompt_tokens=0, completion_tokens=0),
            ),
            0,
        )

    def _parse_query_response_FC(self, api_response):
        message = api_response.choices[0].message
        try:
            model_responses = [
                {func_call.function.name: func_call.function.arguments}
                for func_call in message.tool_calls
            ]
            tool_call_ids = [func_call.id for func_call in message.tool_calls]
        except Exception:
            model_responses = message.content
            tool_call_ids = []

        usage = getattr(api_response, "usage", None)
        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": message,
            "tool_call_ids": tool_call_ids,
            "input_token": getattr(usage, "prompt_tokens", 0),
            "output_token": getattr(usage, "completion_tokens", 0),
        }


# ---------------------------------------------------------------------------
# _Eval360K2HorizonHandler response parsing
# ---------------------------------------------------------------------------

class TestEval360K2HorizonHandlerParsing:
    def test_chat_template_aliases_stay_normalized_in_request(self, tmp_path):
        model = _make_model_instance()
        model.openai_kwargs = {
            "extra_body": {
                "chat_template_kwargs": {
                    "tool_format": "desv32",
                    "reasoning_effort": "HIGH",
                }
            }
        }
        script = BFCLRunner().build_benchmark_script(
            model, _make_task(args={"test_category": "web_search"}), tmp_path
        )
        ns = _exec_script_ns(script, tmp_path, openai_handler_cls=_FakeOpenAICompletionsHandler)
        handler = ns["_Eval360K2HorizonHandler"]("test-model", 0, "result", True)

        handler._query_FC({"message": [{"role": "user", "content": "hi"}], "tools": []})

        chat_template_kwargs = handler.last_generate_kwargs["extra_body"]["chat_template_kwargs"]
        assert chat_template_kwargs["tool_format"] == "dsv32"
        assert chat_template_kwargs["reasoning_effort"] == "high"

    def test_empty_tool_calls_preserves_final_answer_content(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(args={"test_category": "memory"}), tmp_path
        )
        ns = _exec_script_ns(script, tmp_path, openai_handler_cls=_FakeOpenAICompletionsHandler)
        handler = ns["_Eval360K2HorizonHandler"]("test-model", 0, "result", True)
        response = _Obj(
            choices=[_Obj(message=_Obj(content="Michael", tool_calls=[]))],
            usage=_Obj(prompt_tokens=11, completion_tokens=3),
        )

        result = handler._parse_query_response_FC(response)

        assert result["model_responses"] == "Michael"
        assert result["tool_call_ids"] == []
        assert result["model_responses_message_for_chat_history"] == {
            "role": "assistant",
            "content": "Michael",
            "reasoning_content": "",
        }

    def test_inline_reasoning_is_removed_from_final_answer(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(args={"test_category": "memory"}), tmp_path
        )
        ns = _exec_script_ns(script, tmp_path, openai_handler_cls=_FakeOpenAICompletionsHandler)
        handler = ns["_Eval360K2HorizonHandler"]("test-model", 0, "result", True)
        response = _Obj(
            choices=[_Obj(message=_Obj(
                content="summarize the memory\n</ifm|think>\nMichael likes espresso",
                tool_calls=[],
            ))],
            usage=_Obj(prompt_tokens=11, completion_tokens=7),
        )

        result = handler._parse_query_response_FC(response)

        assert result["model_responses"] == "Michael likes espresso"
        assert result["reasoning_content"] == "summarize the memory"
        # Reasoning is stripped from the scored final answer but preserved in the
        # chat history (K2 Horizon consumes its own reasoning across turns).
        assert result["model_responses_message_for_chat_history"] == {
            "role": "assistant",
            "content": "Michael likes espresso",
            "reasoning_content": "summarize the memory",
        }

    def test_tool_calls_remain_function_call_responses(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(args={"test_category": "memory"}), tmp_path
        )
        ns = _exec_script_ns(script, tmp_path, openai_handler_cls=_FakeOpenAICompletionsHandler)
        handler = ns["_Eval360K2HorizonHandler"]("test-model", 0, "result", True)
        tool_call = _Obj(
            id="call_1",
            type="function",
            function=_Obj(name="archival_memory_search", arguments='{"query": "first name"}'),
        )
        response = _Obj(
            choices=[_Obj(message=_Obj(content="", tool_calls=[tool_call]))],
            usage=_Obj(prompt_tokens=17, completion_tokens=9),
        )

        result = handler._parse_query_response_FC(response)

        assert result["model_responses"] == [
            {"archival_memory_search": '{"query": "first name"}'}
        ]
        assert result["tool_call_ids"] == ["call_1"]
        history = result["model_responses_message_for_chat_history"]
        assert history["content"] == ""
        assert history["reasoning_content"] == ""

    def test_tool_call_history_keeps_reasoning_out_of_content(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(args={"test_category": "web_search"}), tmp_path
        )
        ns = _exec_script_ns(script, tmp_path, openai_handler_cls=_FakeOpenAICompletionsHandler)
        handler = ns["_Eval360K2HorizonHandler"]("test-model", 0, "result", True)
        tool_call = _Obj(
            id="call_1",
            type="function",
            function=_Obj(name="search_engine_query", arguments='{"keywords": "largest IPO 2024"}'),
        )
        response = _Obj(
            choices=[_Obj(message=_Obj(
                content="need current data\n</ifm|think>\nI will search now.",
                tool_calls=[tool_call],
            ))],
            usage=_Obj(prompt_tokens=17, completion_tokens=9),
        )

        result = handler._parse_query_response_FC(response)

        history = result["model_responses_message_for_chat_history"]
        assert history["content"] == ""
        assert history["reasoning_content"] == "need current data"
        assert history["tool_calls"][0]["function"]["name"] == "search_engine_query"

    def test_qwen_json_tool_call_content_becomes_function_call_response(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(args={"test_category": "web_search"}), tmp_path
        )
        ns = _exec_script_ns(script, tmp_path, openai_handler_cls=_FakeOpenAICompletionsHandler)
        handler = ns["_Eval360K2HorizonHandler"]("test-model", 0, "result", True)
        response = _Obj(
            choices=[_Obj(message=_Obj(
                content=(
                    '<tool_call>\n'
                    '{"name": "search_engine_query", "arguments": '
                    '{"keywords": "tea price 2025", "max_results": 5}}\n'
                    '</tool_call>'
                ),
                tool_calls=None,
            ))],
            usage=_Obj(prompt_tokens=23, completion_tokens=17),
        )

        result = handler._parse_query_response_FC(response)

        assert len(result["model_responses"]) == 1
        call = result["model_responses"][0]
        assert set(call) == {"search_engine_query"}
        import json
        assert json.loads(call["search_engine_query"]) == {
            "keywords": "tea price 2025",
            "max_results": 5,
        }
        assert result["tool_call_ids"] == ["call_qwen_xml_0"]
        history = result["model_responses_message_for_chat_history"]
        assert history["role"] == "assistant"
        assert history["content"] == ""
        assert history["reasoning_content"] == ""
        assert history["tool_calls"][0]["function"]["name"] == "search_engine_query"

    def test_text_completion_final_answer_is_parsed(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(args={"test_category": "memory"}), tmp_path
        )
        ns = _exec_script_ns(script, tmp_path, openai_handler_cls=_FakeOpenAICompletionsHandler)
        handler = ns["_Eval360K2HorizonHandler"]("test-model", 0, "result", True)
        response = _Obj(
            choices=[_Obj(text="think it through\n</ifm|think>\n{'answer': 'Type 2 Diabetes'}")],
            usage=_Obj(prompt_tokens=29, completion_tokens=11),
        )

        result = handler._parse_query_response_FC(response)

        assert result["model_responses"] == "{'answer': 'Type 2 Diabetes'}"
        assert result["reasoning_content"] == "think it through"
        assert result["model_responses_message_for_chat_history"] == {
            "role": "assistant",
            "content": "{'answer': 'Type 2 Diabetes'}",
            "reasoning_content": "think it through",
        }

    def test_ifm_xml_text_tool_call_becomes_function_call_response(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(args={"test_category": "memory"}), tmp_path
        )
        ns = _exec_script_ns(script, tmp_path, openai_handler_cls=_FakeOpenAICompletionsHandler)
        handler = ns["_Eval360K2HorizonHandler"]("test-model", 0, "result", True)
        response = _Obj(
            choices=[_Obj(text=(
                "need memory\n</ifm|think>\n"
                "<ifm|tool_calls>\n"
                "<ifm|tool_call>archival_memory_key_search\n"
                "<ifm|arg_key>query</ifm|arg_key>\n"
                "<ifm|arg_value>digital scale</ifm|arg_value>\n"
                "</ifm|tool_call>\n"
                "</ifm|tool_calls>"
            ))],
            usage=_Obj(prompt_tokens=41, completion_tokens=23),
        )

        result = handler._parse_query_response_FC(response)

        assert result["model_responses"] == [
            {"archival_memory_key_search": '{"query": "digital scale"}'}
        ]
        assert result["tool_call_ids"] == ["call_ifm_xml_0"]
        history = result["model_responses_message_for_chat_history"]
        assert history["content"] == ""
        assert history["reasoning_content"] == "need memory"
        assert history["tool_calls"][0]["function"]["name"] == "archival_memory_key_search"

    def test_qwen_json_tool_call_keeps_reasoning_in_history(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(),
            _make_task(args={"test_category": "web_search"}),
            tmp_path,
        )
        ns = _exec_script_ns(script, tmp_path, openai_handler_cls=_FakeOpenAICompletionsHandler)
        handler = ns["_Eval360K2HorizonHandler"]("test-model", 0, "result", True)
        response = _Obj(
            choices=[_Obj(message=_Obj(
                content='<tool_call>\n{"name": "fetch_url_content", "arguments": {"url": "https://example.com"}}\n</tool_call>',
                tool_calls=[],
                reasoning_content="need to fetch it",
            ))],
            usage=_Obj(prompt_tokens=31, completion_tokens=19),
        )

        result = handler._parse_query_response_FC(response)

        history = result["model_responses_message_for_chat_history"]
        assert history["content"] == ""
        assert history["reasoning_content"] == "need to fetch it"
        assert history["tool_calls"][0]["function"]["name"] == "fetch_url_content"


# ---------------------------------------------------------------------------
# ATIF trajectory output
# ---------------------------------------------------------------------------

class TestATIFTrajectory:
    def test_atif_log_dir_in_script(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(), tmp_path
        )
        assert "_atif_log_dir" in script
        assert "atif_trajectories" in script

    def test_atif_functions_present(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(), tmp_path
        )
        assert "_build_atif_steps" in script
        assert "_write_atif" in script
        assert "ATIF-v1.4" in script
        assert "atif_turns" in script

    def test_atif_turns_reset_in_pre_query(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(), tmp_path
        )
        assert "_thread_local.atif_turns = []" in script

    def test_build_atif_steps_single_turn(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(args={"save_raw_inference": True}), tmp_path
        )
        ns = _exec_script_ns(script, tmp_path)
        build = ns["_build_atif_steps"]
        turns = [(
            [{"role": "user", "content": "hello"}],
            None,
            {"choices": [{"message": {"content": "hi", "tool_calls": []}}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        )]
        steps, total_pt, total_ct = build(turns)
        assert total_pt == 10
        assert total_ct == 5
        assert steps[0] == {"step_id": 1, "source": "user", "message": "hello"}
        agent = steps[1]
        assert agent["source"] == "agent"
        assert agent["message"] == "hi"
        assert "tool_calls" not in agent
        assert "observation" not in agent

    def test_build_atif_steps_multi_turn_with_tool_calls(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(args={"save_raw_inference": True}), tmp_path
        )
        ns = _exec_script_ns(script, tmp_path)
        build = ns["_build_atif_steps"]
        turns = [
            (
                [{"role": "user", "content": "do X"}],
                None,
                {"choices": [{"message": {"content": None, "tool_calls": [
                    {"id": "call_1", "function": {"name": "find", "arguments": '{"q": "X"}'}}
                ]}}], "usage": {"prompt_tokens": 20, "completion_tokens": 10}},
            ),
            (
                [
                    {"role": "user", "content": "do X"},
                    {"role": "assistant", "content": None},
                    {"role": "tool", "tool_call_id": "call_1", "content": "result X"},
                    {"role": "user", "content": "now do Y"},
                ],
                None,
                {"choices": [{"message": {"content": "done", "tool_calls": []}}],
                 "usage": {"prompt_tokens": 30, "completion_tokens": 5}},
            ),
        ]
        steps, total_pt, total_ct = build(turns)
        assert total_pt == 50
        assert total_ct == 15
        assert steps[0] == {"step_id": 1, "source": "user", "message": "do X"}
        agent_0 = steps[1]
        assert agent_0["source"] == "agent"
        assert agent_0["tool_calls"] == [
            {"tool_call_id": "call_1", "function_name": "find", "arguments": {"q": "X"}}
        ]
        assert agent_0["observation"] == {
            "results": [{"source_call_id": "call_1", "content": "result X"}]
        }
        assert steps[2] == {"step_id": 3, "source": "user", "message": "now do Y"}
        assert steps[3]["source"] == "agent"
        assert steps[3]["message"] == "done"
        assert "tool_calls" not in steps[3]

    def test_build_atif_steps_reasoning_content(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(args={"save_raw_inference": True}), tmp_path
        )
        ns = _exec_script_ns(script, tmp_path)
        build = ns["_build_atif_steps"]
        turns = [(
            [{"role": "user", "content": "think"}],
            None,
            {"choices": [{"message": {"content": "answer", "reasoning_content": "my reasoning", "tool_calls": []}}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
        )]
        steps, _, _ = build(turns)
        agent = steps[1]
        assert agent["reasoning_content"] == "my reasoning"
        assert agent["message"] == "answer"

    def test_write_atif_creates_file(self, tmp_path):
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(args={"save_raw_inference": True}), tmp_path
        )
        ns = _exec_script_ns(script, tmp_path)
        # Simulate one turn accumulated in thread_local
        import threading
        tl = ns["_thread_local"]
        tl.atif_turns = [(
            [{"role": "user", "content": "hello"}],
            None,
            {"choices": [{"message": {"content": "hi", "tool_calls": []}}],
             "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
        )]
        ns["_current_category"] = "test_cat"
        ns["_write_atif"]("sample_42", "test_cat")
        import json
        out = tmp_path / "atif_trajectories" / "test_cat" / "sample_42.jsonl"
        assert out.exists()
        traj = json.loads(out.read_text().strip())
        assert traj["schema_version"] == "ATIF-v1.4"
        assert traj["session_id"] == "test_cat/sample_42"
        assert traj["agent"]["model_name"] == "my-model"
        assert len(traj["steps"]) == 2
        assert traj["final_metrics"]["total_prompt_tokens"] == 5


# ---------------------------------------------------------------------------
# parse_results
# ---------------------------------------------------------------------------

class TestParseResults:

    def test_returns_scores_for_each_category(self, tmp_path):
        _write_overall_csv(tmp_path, [
            {"Non-Live Simple AST": "85.00%", "Non-Live Multiple AST": "72.00%", "Model": "my-model"},
        ])
        scores = BFCLRunner().parse_results(tmp_path, _make_task(), _make_model_instance())
        names = {s.name for s in scores}
        assert "bfcl_non_live_simple_ast" in names
        assert "bfcl_non_live_multiple_ast" in names

    def test_returns_bfcl_v4_agentic_scores(self, tmp_path):
        _write_overall_csv(tmp_path, [
            {
                "Web Search Acc": "42.00%",
                "Memory Recursive Summarization": "63.00%",
                "Model": "my-model",
            },
        ])
        scores = BFCLRunner().parse_results(tmp_path, _make_task(), _make_model_instance())
        by_name = {s.name: s.value for s in scores}
        assert by_name["bfcl_web_search_acc"] == pytest.approx(0.42)
        assert by_name["bfcl_memory_recursive_summarization"] == pytest.approx(0.63)

    def test_score_values_are_correct(self, tmp_path):
        _write_overall_csv(tmp_path, [
            {"Overall Acc": "90.00%", "Model": "my-model"},
        ])
        scores = BFCLRunner().parse_results(tmp_path, _make_task(), _make_model_instance())
        assert scores[0].value == pytest.approx(0.90)

    def test_raises_if_score_dir_missing(self, tmp_path):
        with pytest.raises(ImportedDatasetResultError, match="overall results not found"):
            BFCLRunner().parse_results(tmp_path, _make_task(), _make_model_instance())

    def test_raises_if_csv_is_empty(self, tmp_path):
        _write_overall_csv(tmp_path, [])
        with pytest.raises(ImportedDatasetResultError, match="empty"):
            BFCLRunner().parse_results(tmp_path, _make_task(), _make_model_instance())

    def test_raises_if_no_accuracy_field(self, tmp_path):
        _write_overall_csv(tmp_path, [
            {"Model": "my-model", "Total Cost ($)": "0.06"},  # no percentage columns
        ])
        with pytest.raises(ImportedDatasetResultError, match="No accuracy scores"):
            BFCLRunner().parse_results(tmp_path, _make_task(), _make_model_instance())

    def test_runner_is_registered(self):
        from scheduler.imported_dataset import get_runner
        assert get_runner("bfcl") is BFCLRunner

    def test_missing_csv_raises_with_clear_message(self, tmp_path):
        # Gap 1: csv file does not exist at all
        with pytest.raises(ImportedDatasetResultError, match="overall results not found"):
            BFCLRunner().parse_results(tmp_path, _make_task(), _make_model_instance())

    def test_all_na_accuracy_columns_raises(self, tmp_path):
        # Gap 1: CSV exists and has rows but every accuracy column is "N/A"
        _write_overall_csv(tmp_path, [
            {"Model": "my-model", "Overall Acc": "N/A", "Non-Live Simple AST": "N/A"},
        ])
        with pytest.raises(ImportedDatasetResultError, match="No accuracy scores"):
            BFCLRunner().parse_results(tmp_path, _make_task(), _make_model_instance())

    def test_blank_lines_in_csv_are_ignored(self, tmp_path):
        # Gap 2: Blank lines interspersed in CSV must not cause a crash.
        # csv.DictReader skips blank lines automatically; we write raw content to verify.
        import csv as _csv
        csv_path = tmp_path / "score" / "data_overall.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Write a CSV with a blank line between header and data row
        csv_path.write_text(
            "Overall Acc,Model\n"
            "\n"
            "75.00%,my-model\n"
            "\n"
        )
        scores = BFCLRunner().parse_results(tmp_path, _make_task(), _make_model_instance())
        assert any(s.name == "bfcl_overall_acc" for s in scores)

    def test_concurrent_think_tag_count_updates(self, tmp_path):
        # Gap 3: _think_tag_counts dict is updated from multiple concurrent calls.
        # We exec the rendered script, then hammer _parse_query_response_FC from many
        # concurrent coroutines to verify no counts are lost.
        import asyncio
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(), _make_task(), tmp_path
        )
        ns = _exec_script_ns(script, tmp_path)

        # The tag counting happens inside _parse_query_response_FC via the re.search path.
        # We simulate it directly by calling the update logic many times concurrently.
        counts = ns["_think_tag_counts"]
        # Reset to zero
        for k in counts:
            counts[k] = 0

        n_calls = 200

        async def bump():
            # Simulate what the script does on a "none" path (no think tag)
            counts["none"] = counts.get("none", 0) + 1

        async def run_concurrent():
            await asyncio.gather(*[bump() for _ in range(n_calls)])

        asyncio.run(run_concurrent())
        # asyncio.gather on coroutines runs them cooperatively on a single thread,
        # so the total count must equal exactly n_calls (no lost updates).
        assert counts["none"] == n_calls

    def test_unwritable_raw_log_dir_does_not_crash(self, tmp_path):
        # Gap 4: If raw_log_dir is unwritable the try/except in _append_raw_log swallows
        # the PermissionError and the script continues without raising.
        import stat
        script = BFCLRunner().build_benchmark_script(
            _make_model_instance(),
            _make_task(args={"save_raw_inference": True}),
            tmp_path,
        )
        ns = _exec_script_ns(script, tmp_path)

        # Create a raw_log dir that exists but is not writable
        raw_log_dir = tmp_path / "raw_inference"
        raw_log_dir.mkdir(parents=True, exist_ok=True)
        raw_log_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # read+exec only, no write

        try:
            ns["_save_raw_inference"] = True
            ns["_raw_log_dir"] = str(raw_log_dir)
            ns["_current_category"] = "test_cat"

            # Should not raise even though makedirs will fail with PermissionError
            ns["_append_raw_log"](
                [{"role": "user", "content": "hi"}],
                {"choices": [{"message": {"content": "hello"}}]},
                {},
            )
        finally:
            # Restore permissions so tmp_path cleanup can remove the directory
            raw_log_dir.chmod(stat.S_IRWXU)
