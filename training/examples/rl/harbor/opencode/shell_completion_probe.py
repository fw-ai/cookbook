"""Exercise the real OpenCode CLI/tool loop without model calls or training.

Run INSIDE an isolated E2B sandbox:
    python3 shell_completion_probe.py /tmp/opencode-fixed

A loopback OpenAI-compatible stub emits deterministic bash tool calls. It does
not change the tool implementation: the supplied OpenCode binary executes them.
Checks foreground output/status, signal exits, background-server survival across
tool calls, and timeout cleanup. No task candidate, verifier, or live RL process
is modified. Results are JSON; a failed assertion returns nonzero.
"""

from __future__ import annotations

import http.server
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import threading
import time


def main(binary: str, selected: set[str] | None = None) -> None:
    results = []
    with tempfile.TemporaryDirectory(prefix="opencode-shell-probe-") as directory:
        root = Path(directory)
        ready = root / "server-port"
        server_code = (
            "import http.server, pathlib, threading, time; "
            "s=http.server.ThreadingHTTPServer(('127.0.0.1',0),http.server.SimpleHTTPRequestHandler); "
            f"pathlib.Path({str(ready)!r}).write_text(str(s.server_port)); "
            "threading.Thread(target=s.serve_forever,daemon=True).start(); "
            "exec('while True:\\n print(\"background heartbeat\", flush=True)\\n time.sleep(.05)')"
        )
        cases = [
            ("nonzero", [("printf final-stdout; printf final-stderr >&2; exit 7", 5000)]),
            ("signal", [("echo before-signal; kill -TERM $$", 5000)]),
            ("segfault", [("echo before-signal; kill -SEGV $$", 5000)]),
            ("background", [
                (f"python3 -u -c {shlex.quote(server_code)} & bg=$!; "
                 f"echo $bg > {shlex.quote(str(root / 'server-pid'))}; "
                 f"while [ ! -s {shlex.quote(str(ready))} ]; do sleep .01; done; "
                 "echo SERVER_STARTED", 5000),
                (f"sleep .3; curl --fail --max-time 2 -s http://127.0.0.1:$(cat {shlex.quote(str(ready))})/ >/dev/null "
                 "&& echo SERVER_ALIVE", 5000),
            ]),
            ("timeout", [("echo timeout-start; sleep 30", 300)]),
        ]
        for name, commands in cases:
            if selected is not None and name not in selected:
                continue
            request_times: list[float] = []
            class Handler(http.server.BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_POST(self):
                    request_times.append(time.monotonic())
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    index = sum(m.get("role") == "tool" for m in body["messages"])
                    if index < len(commands):
                        command, timeout = commands[index]
                        delta = {"role": "assistant", "tool_calls": [{
                            "index": 0, "id": f"call_{index}", "type": "function",
                            "function": {"name": "bash", "arguments": json.dumps({
                                "command": command, "description": "Command completion regression probe",
                                "timeout": timeout,
                            })},
                        }]}
                        finish = "tool_calls"
                    else:
                        delta, finish = {"role": "assistant", "content": "Probe complete."}, "stop"
                    chunks = [
                        {"id": "probe", "object": "chat.completion.chunk", "model": "probe",
                         "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                        {"id": "probe", "object": "chat.completion.chunk", "model": "probe",
                         "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                         "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
                    ]
                    payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(payload.encode())))
                    self.end_headers()
                    self.wfile.write(payload.encode())

            stub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=stub.serve_forever, daemon=True)
            thread.start()
            config_dir = root / name / "config" / "opencode"
            config_dir.mkdir(parents=True)
            (config_dir / "node_modules").mkdir()
            (config_dir / "package-lock.json").write_text(json.dumps({
                "lockfileVersion": 3, "packages": {"": {"dependencies": {"@opencode-ai/plugin": "1.18.8"}}},
            }))
            (config_dir / "opencode.json").write_text(json.dumps({
                "snapshot": False, "agent": {"title": {"disable": True}},
                "provider": {"probe": {"npm": "@ai-sdk/openai-compatible", "name": "probe",
                    "options": {"baseURL": f"http://127.0.0.1:{stub.server_port}/v1", "apiKey": "local-probe"},
                    "models": {"probe": {"name": "probe", "limit": {"context": 262144, "output": 1000}}}}},
            }))
            env = {**os.environ, "XDG_CONFIG_HOME": str(config_dir.parent),
                   "XDG_DATA_HOME": str(root / name / "data"), "XDG_STATE_HOME": str(root / name / "state"),
                   "OPENCODE_DISABLE_MODELS_FETCH": "1", "OPENCODE_DISABLE_AUTOUPDATE": "1",
                   "OPENCODE_FAKE_VCS": "git"}
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    [binary, "run", "--model=probe/probe", "--format=json", "--dangerously-skip-permissions", "Run probe"],
                    cwd=root, env=env, capture_output=True, text=True, timeout=25,
                )
                events = []
                for line in completed.stdout.splitlines():
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("type") == "tool_use":
                        events.append(event["part"]["state"])
                record = {"case": name, "elapsed_s": round(time.monotonic() - started, 3),
                          "cli_exit": completed.returncode, "tools": events,
                          "tool_roundtrip_s": [round(b - a, 3) for a, b in zip(request_times, request_times[1:])]}
                print(json.dumps(record), flush=True)
                results.append(record)
                assert completed.returncode == 0, completed.stderr[-3000:]
                assert len(events) == len(commands), completed.stdout[-3000:] + completed.stderr[-3000:]
                output = events[0].get("output", "")
                # OpenCode's final tool event can rewrite start timestamps;
                # measure from stub dispatch until its next request instead.
                elapsed = request_times[1] - request_times[0]
                if name == "nonzero":
                    assert events[0]["metadata"]["exit"] == 7 and "final-stdout" in output and "final-stderr" in output
                elif name in ("signal", "segfault"):
                    assert "before-signal" in output and "SIG" in output and "exceeding timeout" not in output
                    assert elapsed < 3
                elif name == "background":
                    assert elapsed < 3 and events[0]["metadata"]["exit"] == 0
                    assert "SERVER_ALIVE" in events[1].get("output", "")
                elif name == "timeout":
                    assert "timeout-start" in output and "exceeding timeout" in output
            finally:
                stub.shutdown()
                stub.server_close()
                if name == "background" and (root / "server-pid").exists():
                    try:
                        os.kill(int((root / "server-pid").read_text()), 9)
                    except ProcessLookupError:
                        pass
    print(json.dumps({"passed": len(results)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary")
    parser.add_argument("--cases", help="Optional comma-separated case names")
    args = parser.parse_args()
    main(args.binary, set(args.cases.split(",")) if args.cases else None)
