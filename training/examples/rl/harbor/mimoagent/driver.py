#!/usr/bin/env python3
"""One-shot mimoagent driver inside a Harbor task container.

Runs one task with a mimoagent arm against the environment-local TITO
sidecar, sampling exactly as the MiMo code recipe does (temperature 1.0,
top-p 0.95, top-k 20). mimoagent is Xiaomi MiMo's edited fork of
mini-swe-agent 1.9.0; both MIT-licensed, attribution in constants.py.

Configuration is by environment variable:
  OPENAI_BASE_URL / OPENAI_API_KEY   sidecar endpoint (required)
  MIMOAGENT_MODEL_NAME               serving name (default "policy")
  MIMOAGENT_TASK_FILE                task instruction file (required)
  MIMOAGENT_ARM                      "bash" (default) or "cc" (mini-claude-code)
  MIMOAGENT_MCP_SERVERS              JSON list of {"name", "url"}; the task's
                                     local MCP services become mcp__* tools
  MIMOAGENT_STEP_LIMIT               max agent steps (default 150)
  MIMOAGENT_OUTPUT_LIMIT             max tokens per model call (default 32768)
  MIMOAGENT_TOOL_TIMEOUT_SECONDS     bash tool timeout ceiling (default 300)
  MIMOAGENT_CWD                      working directory (default: process cwd)

Exit code: 0 when the agent finished (idle) or hit the step limit — both are
normal episode endings; 1 for anything else (a model/query failure means the
trajectory is unusable).
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/opt/mimoagent-mcp")  # flat copies: mcp_client.py, mcp_tools.py

from mimoagent.environments.local import LocalEnvironment
from mimoagent.models.openai_chat import OpenAIChatModel

_MESSAGES_PATH = Path("/logs/agent/mimoagent-messages.json")
_RESULT_PATH = Path("/logs/agent/mimoagent-result.json")

_BASH_SYSTEM_TEMPLATE = """You are an agent, your current working directory is {{cwd}}.

You can use the tools available to you to interact with the computer to assist the user in completing tasks."""
_CC_SYSTEM_TEMPLATE = """You are Claude Code, Anthropic's official CLI for Claude.

CWD: {{cwd}}

Solve the task by editing the source code in {{cwd}}. When you encounter
an obstacle, do not use shortcuts to simply make it go away — identify
the root cause and fix the underlying issue rather than bypassing the
test. For instance: do not fetch the upstream or a newer version of the
repo from GitHub (or any mirror), do not pip/npm install a newer release
of the package under test, do not overwrite test files to make them
pass, and do not hard-code expected outputs. Follow both the spirit and
letter of these instructions — any such shortcut will be detected and
scored zero.

Keep changes minimal. A bug fix doesn't need surrounding cleanup or
refactoring; don't add features, abstractions, or error-handling beyond
what the task requires.

Prefer dedicated tools over Bash when one fits: Read for known paths,
Grep for content search, Glob for filename patterns, Edit/Write for
file modification. Reserve Bash for shell-only operations (running
tests, build commands, environment inspection)."""
_INSTANCE_TEMPLATE = """Fix the following issue:

{{task}}"""


def _build_agent(model, env, *, arm: str, step_limit: int, tool_timeout: int):
    if arm == "bash":
        from mimoagent.agents.bashonly import BashOnlyAgent

        return BashOnlyAgent(
            model,
            env,
            msg_path=_MESSAGES_PATH,
            step_limit=step_limit,
            system_template=_BASH_SYSTEM_TEMPLATE,
            instance_template=_INSTANCE_TEMPLATE,
            tools=[
                {
                    "tool": "bash-only",
                    "config": {"timeout": 60, "max_timeout": tool_timeout},
                }
            ],
        )
    if arm == "cc":
        from mimoagent.agents.cc import CCAgent

        return CCAgent(
            model,
            env,
            msg_path=_MESSAGES_PATH,
            step_limit=step_limit,
            system_template=_CC_SYSTEM_TEMPLATE,
            instance_template=_INSTANCE_TEMPLATE,
            tools=[
                {
                    "tool": "Bash",
                    "config": {"timeout": 60, "max_timeout": tool_timeout},
                },
                {"tool": "Read"},
                {"tool": "Write"},
                {"tool": "Edit"},
                {"tool": "Grep"},
                {"tool": "Glob"},
            ],
        )
    raise ValueError(f"unknown mimoagent arm: {arm!r}")


def main() -> int:
    task = Path(os.environ["MIMOAGENT_TASK_FILE"]).read_text(encoding="utf-8")
    arm = os.environ.get("MIMOAGENT_ARM", "bash")
    step_limit = int(os.environ.get("MIMOAGENT_STEP_LIMIT", "150"))
    output_limit = int(os.environ.get("MIMOAGENT_OUTPUT_LIMIT", "32768"))
    tool_timeout = int(os.environ.get("MIMOAGENT_TOOL_TIMEOUT_SECONDS", "300"))
    model = OpenAIChatModel(
        model_name=os.environ.get("MIMOAGENT_MODEL_NAME", "policy"),
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        model_kwargs={
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": output_limit,
            "timeout": 1800,
            "max_retries": 2,
        },
    )
    env = LocalEnvironment(
        cwd=os.environ.get("MIMOAGENT_CWD", ""),
        timeout=tool_timeout,
    )
    agent = _build_agent(
        model, env, arm=arm, step_limit=step_limit, tool_timeout=tool_timeout
    )
    raw_servers = os.environ.get("MIMOAGENT_MCP_SERVERS", "")
    if raw_servers:
        import mcp_client
        import mcp_tools

        servers = mcp_client.servers_from_json(raw_servers)
        for tool in mcp_tools.mimoagent_mcp_tools(servers):
            agent.tool_registry.register(tool)
        agent._tool_definitions = agent.tool_registry.get_function_definitions()
        print(
            f"[mimoagent-driver] registered {len(agent.tool_registry.tools)} tools "
            f"for {len(servers)} MCP servers",
            flush=True,
        )
    status, message = agent.run(task)
    _RESULT_PATH.write_text(
        json.dumps(
            {
                "status": status,
                "message": message[-2000:],
                "steps": agent._steps_taken,
            }
        ),
        encoding="utf-8",
    )
    print(f"[mimoagent-driver] status={status} steps={agent._steps_taken}", flush=True)
    return 0 if status in {"Idle", "LimitsExceeded"} else 1


if __name__ == "__main__":
    sys.exit(main())
