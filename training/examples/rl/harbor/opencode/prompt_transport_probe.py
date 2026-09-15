"""CPU-only actual-OpenCode prompt transport parity and self-pkill probe.

Run only in an isolated diagnostic sandbox: python3 prompt_transport_probe.py
/tmp/opencode-fixed. Uses a deterministic loopback provider, no model service.
Compares user-message bytes for argv versus stdin, then tests that stdin keeps
task text out of launcher argv. Does not change a live trial or reward.
"""
import hashlib
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading


def encode_stdin(prompt):
    # Exact non-interactive OpenCode 1.18.8 argv-message normalization.
    return '"' + prompt.replace('"', '\\"') + '"' if ' ' in prompt else prompt


def probe(binary):
    prompt = 'Run `node vm.js` with "quotes", $HOME, backslashes \\ and\na second line.\n'
    outputs = {}
    full_inputs = {}
    with tempfile.TemporaryDirectory(prefix='prompt-transport-') as tmp:
        root = Path(tmp)
        for mode in ['argv', 'stdin_raw', 'stdin_encoded', 'stdin_pkill']:
            requests = []

            class Handler(http.server.BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_POST(self):
                    request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    requests.append(request)
                    if mode == 'stdin_pkill' and len(requests) == 1:
                        delta = {'role': 'assistant', 'tool_calls': [{
                            'index': 0, 'id': 'call_probe', 'type': 'function',
                            'function': {'name': 'bash', 'arguments': json.dumps({
                                'command': 'pkill -f "node vm.js"; echo AGENT_SURVIVED',
                                'description': 'Isolated launcher matching probe',
                                'timeout': 5000,
                            })},
                        }]}
                        finish = 'tool_calls'
                    else:
                        delta = {'role': 'assistant', 'content': 'Probe complete.'}
                        finish = 'stop'
                    chunks = [
                        {'id': 'probe', 'object': 'chat.completion.chunk', 'model': 'probe',
                         'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
                        {'id': 'probe', 'object': 'chat.completion.chunk', 'model': 'probe',
                         'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}],
                         'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}},
                    ]
                    payload = ''.join(f'data: {json.dumps(c)}\n\n' for c in chunks) + 'data: [DONE]\n\n'
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Content-Length', str(len(payload.encode())))
                    self.end_headers()
                    self.wfile.write(payload.encode())

            server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            config = root / mode / 'config' / 'opencode'
            config.mkdir(parents=True)
            (config / 'node_modules').mkdir()
            (config / 'package-lock.json').write_text(json.dumps({
                'lockfileVersion': 3,
                'packages': {'': {'dependencies': {'@opencode-ai/plugin': '1.18.8'}}},
            }))
            (config / 'opencode.json').write_text(json.dumps({
                'snapshot': False, 'agent': {'title': {'disable': True}},
                'provider': {'probe': {'npm': '@ai-sdk/openai-compatible', 'name': 'probe',
                    'options': {'baseURL': f'http://127.0.0.1:{server.server_port}/v1', 'apiKey': 'local-probe'},
                    'models': {'probe': {'name': 'probe', 'limit': {'context': 262144, 'output': 1000}}}}},
            }))
            env = {**os.environ, 'XDG_CONFIG_HOME': str(config.parent),
                   'XDG_DATA_HOME': str(root / mode / 'data'), 'XDG_STATE_HOME': str(root / mode / 'state'),
                   'OPENCODE_DISABLE_MODELS_FETCH': '1', 'OPENCODE_DISABLE_AUTOUPDATE': '1',
                   'OPENCODE_FAKE_VCS': 'git'}
            command = [binary, 'run', '--model=probe/probe', '--format=json', '--thinking', '--dangerously-skip-permissions', '--']
            if mode == 'argv':
                command.append(prompt)
            input_text = '' if mode == 'argv' else prompt if mode == 'stdin_raw' else encode_stdin(prompt)
            try:
                result = subprocess.run(command, input=input_text, text=True, capture_output=True,
                                        cwd=root, env=env, timeout=25)
                assert result.returncode == 0, (mode, result.returncode, result.stderr[-1000:])
                assert requests, mode
                users = [m for m in requests[0]['messages'] if m['role'] == 'user']
                payload = json.dumps(users, ensure_ascii=False, sort_keys=True).encode()
                outputs[mode] = payload
                full_input = json.dumps({k: requests[0].get(k) for k in ('messages', 'tools')},
                                        ensure_ascii=False, sort_keys=True).encode()
                full_inputs[mode] = full_input
                tools = [m for r in requests[1:] for m in r['messages'] if m['role'] == 'tool']
                # The exact command can kill its own bash tool shell too.
                # Success here means the AGENT handles that tool result and
                # makes the next provider request, not that the tool succeeds.
                survived = len(requests) >= 2 and bool(tools)
                print(json.dumps({'mode': mode, 'cli_exit': result.returncode,
                                  'request_count': len(requests), 'user_messages_sha256': hashlib.sha256(payload).hexdigest(),
                                  'messages_and_tools_sha256': hashlib.sha256(full_input).hexdigest(),
                                  'survived_pkill': survived}), flush=True)
                if mode == 'stdin_pkill':
                    assert len(requests) >= 2 and survived, 'Agent did not survive self-matching command'
            finally:
                server.shutdown()
                server.server_close()
        assert outputs['argv'] != outputs['stdin_raw'], 'Expected pinned CLI quoting distinction'
        assert outputs['argv'] == outputs['stdin_encoded'] == outputs['stdin_pkill']
        assert full_inputs['argv'] == full_inputs['stdin_encoded'] == full_inputs['stdin_pkill']
        print(json.dumps({'exact_user_message_parity': True, 'raw_stdin_would_change_prompt': True}), flush=True)


if __name__ == '__main__':
    probe(sys.argv[1])
