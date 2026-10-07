#!/usr/bin/env python3
"""A stand-in for `vllm serve` that speaks just enough OpenAI to be evaluated.

WHY A STUB AND NOT VLLM. VLLM needs a GPU and tens of gigabytes of weights;
neither exists on a CI runner. Everything BETWEEN the scheduler and VLLM can be
real, though — real sbatch, real slurmd, the real `scheduler/slurm/*.sh`
scripts, real squeue parsing, real health discovery — so the stub is placed at
the narrowest point that removes the GPU dependency and nowhere earlier.

What that buys: `sbatch_script.sh` still base64-decodes `$vllm_args`, still
rebuilds the argv, and still execs something called `vllm` with it. If that
reconstruction breaks, this stub is what notices, because it is handed the
decoded argv and prints it.

CONTRACT WITH THE SCHEDULER, and where each half is required:
  GET  /health                  -> 200. `SlurmManager.check_live` polls exactly
                                   this, at http://{nodelist}:8000/health, and
                                   a model is not `live` until it answers.
  GET  /v1/models               -> the names passed via --served-model-name.
  POST /v1/chat/completions     -> a deterministic canned reply.
  POST /v1/completions          -> likewise, for base models.

Determinism is the point of the reply text: a grader's verdict in CI should
depend on the code under test, never on sampling.
"""

from __future__ import annotations

import argparse
import errno
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Echoed back verbatim as the model's output. Short, fixed, and boring on
# purpose — CI asserts on the plumbing, not on the content.
CANNED_REPLY = "CI-STUB-RESPONSE"

SERVED_MODEL_NAMES: list[str] = []
MODEL_PATH = ""


class Handler(BaseHTTPRequestHandler):
    # The default logs one line per request to stderr, which for a poll loop
    # running every few seconds buries the job log in health checks.
    def log_message(self, fmt, *args):  # noqa: D102
        pass

    def _send(self, status: int, payload: dict | None = None) -> None:
        body = json.dumps(payload or {}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path.startswith("/v1/models"):
            self._send(200, {
                "object": "list",
                "data": [
                    {"id": name, "object": "model", "owned_by": "eval360-ci"}
                    for name in (SERVED_MODEL_NAMES or [MODEL_PATH])
                ],
            })
        else:
            self._send(404, {"error": f"no route {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON"})
            return

        model = request.get("model") or (SERVED_MODEL_NAMES or [MODEL_PATH])[0]
        # `n` is how the scheduler asks for pass@k / avg@k samples. Honouring it
        # matters: a stub that always returns one choice would make every
        # multi-sample config silently degrade to a single sample.
        n = int(request.get("n") or 1)
        created = int(time.time())

        if self.path.startswith("/v1/chat/completions"):
            self._send(200, {
                "id": "ci-stub",
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": i,
                        "message": {"role": "assistant", "content": CANNED_REPLY},
                        "finish_reason": "stop",
                    }
                    for i in range(n)
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })
        elif self.path.startswith("/v1/completions"):
            self._send(200, {
                "id": "ci-stub",
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [
                    {"index": i, "text": CANNED_REPLY, "finish_reason": "stop"}
                    for i in range(n)
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })
        else:
            self._send(404, {"error": f"no route {self.path}"})


def main(argv: list[str]) -> int:
    global MODEL_PATH

    # Invoked as `vllm serve <model> [flags]`. The flags arrive having made the
    # round trip through base64 in sbatch_script.sh, so printing them is the
    # evidence that the round trip worked.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("subcommand", nargs="?", default="serve")
    parser.add_argument("model", nargs="?", default="")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", action="append", default=[])
    known, unknown = parser.parse_known_args(argv)

    MODEL_PATH = known.model
    SERVED_MODEL_NAMES.extend(known.served_model_name)

    print(f"[fake-vllm] argv as received : {argv}", flush=True)
    print(f"[fake-vllm] model            : {known.model}", flush=True)
    print(f"[fake-vllm] served-model-name: {known.served_model_name}", flush=True)
    print(f"[fake-vllm] flags ignored    : {unknown}", flush=True)

    # Bind on all interfaces, not loopback: the scheduler reaches this by the
    # node's hostname (`http://{nodelist}:8000`), never by 127.0.0.1.
    try:
        server = ThreadingHTTPServer((known.host, known.port), Handler)
    except OSError as error:
        if error.errno not in (errno.EADDRINUSE, errno.EACCES):
            raise
        # EXPECTED when a model is deployed with more than one replica.
        #
        # The test cluster is a single node, and the scheduler health-checks a
        # hard-coded `http://{nodelist}:8000`, so every replica of a model lands
        # on the same host and port. The first replica binds; the rest cannot.
        #
        # That is an artifact of single-node testing, not a defect to paper
        # over: on a real cluster each replica gets its own node. Exiting 0 with
        # a clear line keeps the replica's Slurm job RUNNING (the enclosing
        # sbatch script backgrounds this and then sleeps), which is what the
        # replica-counting tests actually care about, while leaving evidence in
        # the job output rather than an unexplained traceback.
        print(
            f"[fake-vllm] port {known.port} already bound on this node — "
            "another replica is serving it. Exiting 0; this replica's Slurm "
            "job stays RUNNING.",
            flush=True,
        )
        return 0

    print(f"[fake-vllm] serving on {known.host}:{known.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
