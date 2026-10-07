# BFCL Web Search Evaluation

This guide covers the BFCL V4 web-search-only path in Eval360, using the K2 Horizon
vLLM OpenAI-compatible handler and a Serper search backend.

## Files

- Dataset config: `examples/data/bfcl_web_only.yaml`
- Model config: `model_zoo/k2_horizon_7b_mid3_v3_ckpt5500_bfcl_web.yaml`
- Runner implementation: `scheduler/imported_dataset/bfcl.py`
- Slurm container launcher: `scheduler/slurm/imported_dataset_script_container.sh`

## Serper Key

Create a local key file at the path referenced by the dataset YAML:

```bash
cat > .search_api_keys <<'EOF'
SERPER_API_KEY=your_serper_key
EOF
```

`.search_api_keys` is ignored by git. Do not commit it.

## Dataset Parameters

`examples/data/bfcl_web_only.yaml` uses the imported dataset runner:

```yaml
imported_dataset:
  name: bfcl
  commit: f7cf7359b7ac615a0b294831c5ba2bc95ee4a000
  args:
    test_category: web_search
    save_raw_inference: true
    num_threads: 10
    web_search_backend: serper
    web_search_api_keys_file: .search_api_keys
```

Important args:

- `test_category`: BFCL category to run. Use `web_search` for the web-search-only test.
- `save_raw_inference`: writes raw BFCL turns under `raw_inference/`; useful for debugging.
- `num_threads`: BFCL-side parallelism. Keep this aligned with model throughput and `max_simultaneous_requests`.
- `web_search_backend`: set to `serper` to patch BFCL `WebSearchAPI.search_engine_query`.
- `web_search_api_keys_file`: path to a dotenv-style key file containing `SERPER_API_KEY`.
- `include_reasoning_in_history`: optional. When true, includes reasoning content in the message history passed to BFCL tools.

Optional fetch controls:

- `web_fetch_force_mode`: overrides every `fetch_url_content(url, mode=...)` call with the configured mode.
- `web_fetch_max_chars`: caps fetched `content` after the BFCL fetch implementation returns.

For official BFCL behavior, leave `web_fetch_force_mode` and `web_fetch_max_chars` unset. In that mode, the model chooses the `mode` argument for `fetch_url_content`; if it omits the argument, BFCL defaults to `raw`.

To force truncated fetches for a robustness/debug run:

```yaml
    web_fetch_force_mode: truncate
    web_fetch_max_chars: 200000
```

This is not the official default. It reduces context-overflow risk from large raw pages, but it changes tool behavior.

## Model Parameters

`model_zoo/k2_horizon_7b_mid3_v3_ckpt5500_bfcl_web.yaml` launches the K2 Horizon checkpoint in a pyxis/enroot container:

```yaml
remote_model:
  base_name: k2-horizon-7b-mid3-v3-ckpt5500-bfcl-web
  path: IFM/K2-Horizon-7B
  revision: mid_3_5500

container_image: /<CONTAINER_IMAGES>/vllm-serving.sqsh
container_mounts:
  - /path/to/Eval360:/path/to/Eval360
```

Important fields:

- `remote_model.base_name`: the base of the served model name; Eval360 appends the revision (`-mid_3_5500`). BFCL uses the served name as the model id for OpenAI API calls, so it must match the vLLM served name.
- `remote_model.path` and `remote_model.revision`: the checkpoint, here the public `IFM/K2-Horizon-7B` at its `mid_3_5500` tag. A local checkpoint path is mounted into the container instead.
- `model_type: base`: keeps the scheduler model classification aligned with the K2 Horizon config. The imported BFCL runner still uses BFCL's OpenAI chat-completions handler for tool calling.
- `parser_type: noop`: BFCL evaluates its own final answer format; Eval360 parser should not modify it.
- `container_image`: pyxis/enroot image used to start vLLM.
- `container_mounts`: extra mounts. Mount the repo root when the key file or configs live inside the repo. Eval360 automatically adds an identity mount for `remote_model.path`.
- `max_simultaneous_requests`: Eval360-side request concurrency for the model endpoint.
- `max_time_to_deploy`: how long the Slurm launcher waits for vLLM `/health`.
- `allow_long_max_model_len`: exports `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` for long-context runs.
- `serving_slurm_resources.gpus_per_node`: 8, the same as `--tensor-parallel-size`.

Required vLLM args for this model:

- `--enable-auto-tool-choice`
- `--tool-call-parser <TOOL_CALL_PARSER>`
- `--reasoning-parser <REASONING_PARSER>`
- `--max-model-len 524288`
- `--tensor-parallel-size 8`

Both parsers were pre-release vLLM parsers. The closest public ones are vLLM's `k2_horizon` reasoning and tool-call parsers, which differ on truncated and malformed outputs.

The `openai_kwargs.extra_body.chat_template_kwargs` block controls K2 Horizon chat template behavior:

```yaml
openai_kwargs:
  temperature: 0.01
  max_tokens: 8192
  extra_body:
    chat_template_kwargs:
      reasoning_effort: high
      tool_presentation_format: xml
      tool_call_format: xml
```

## Run

From the repository root:

```bash
eval360 \
  --max-generation-jobs 1 \
  --max-grading-parallelism 1 \
  --log-dir logs/bfcl_web_search_serper \
  evaluate-now \
  --model-paths model_zoo/k2_horizon_7b_mid3_v3_ckpt5500_bfcl_web.yaml \
  --data-paths examples/data/bfcl_web_only.yaml \
  --force
```

Use `--force` when you want to rerun an imported dataset job. For imported datasets, `--force` clears the Eval360 sentinels such as `.job_complete`; it does not fully delete BFCL's native output directory.

## Monitor

Check Slurm:

```bash
squeue --user "$USER"
```

Check scheduler and job logs:

```bash
tail -f logs/bfcl_web_search_serper/scheduler.log
tail -f logs/bfcl_web_search_serper/slurm-<job_id>.out
```

Healthy startup should show:

- vLLM starts with the expected `served_model_name`.
- `[WebSearchAPI] Loaded search API keys ... SERPER_API_KEY`
- `[WebSearchAPI] Patched search_engine_query backend=serper`
- no `Patched fetch_url_content` line for official fetch behavior.

## Outputs

Eval360 writes imported dataset output under:

```text
<model output_path>/<model name>/<dataset name>_output/
```

For the provided configs this is:

```text
<OUTPUT_ROOT>/
  k2-horizon-7b-mid3-v3-ckpt5500-bfcl-web-mid_3_5500/
    bfcl_web_only_output/
```

Useful files:

- `.job_complete`: imported dataset job finished.
- `raw_inference/web_search/...`: raw turn-level BFCL logs when `save_raw_inference: true`.
- `atif_trajectories/web_search/*.jsonl`: converted ATIF trajectories.
- `result/<model>/agentic/BFCL_v4_web_search_*_result.json`: BFCL result rows.
- `score/<model>/agentic/BFCL_v4_web_search_*_score.json`: BFCL score files.
- `<dataset>_scores.yaml`: Eval360 normalized scores.

Expected web-only metrics:

- `bfcl_web_search_acc`
- `bfcl_web_search_base`
- `bfcl_web_search_no_snippet`

`bfcl_overall_acc` is BFCL's full-table aggregation and is not the main metric for a web-only run.

## Common Issues

- Missing `SERPER_API_KEY`: ensure `web_search_api_keys_file` points to the local key file and that the repo/key path is mounted into the container.
- Context overflow in official raw fetch mode: raw pages can exceed the model context. For debugging, use `web_fetch_force_mode: truncate` and `web_fetch_max_chars`, but keep them unset for official behavior.
- No BFCL handler import: the runner inlines the K2 Horizon vLLM handler because the pinned PyPI `bfcl-eval` package does not ship it.
- Container cannot see repo files: add a `container_mounts` entry for the repo root. Eval360 already adds the model checkpoint mount automatically.
