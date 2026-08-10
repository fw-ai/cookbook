"""Harbor OpenCode agent configured for the local recording policy server.

OpenCode runs inside each Harbor task container while the live RLOR sampler
runs on the host.  The custom provider below points the CLI at that host-side
OpenAI-compatible endpoint and advertises the trainer's real context limit so
OpenCode can manage long task histories against the same token budget.
"""

from __future__ import annotations

import json
import shlex
from typing import Any, override

from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

HOST_PLACEHOLDER = "{host}"
_OPENCODE_CONFIG_HOME = "/logs/agent/opencode/xdg-config"
_PROVIDER_ID = "fireworks-rl"
_MODEL_ID = "policy"

# Resolve the host from inside a Docker task container without depending on
# ``ip``, which is absent from many Terminal-Bench images.  Docker route words
# in /proc/net/route are little-endian hexadecimal.
_RESOLVE_HOST_SH = r"""
_fw_routes() {
  awk '
    function hex(s) {
      n = 0
      for (i = 1; i <= length(s); i++) {
        n = n * 16 + index("0123456789ABCDEF", toupper(substr(s, i, 1))) - 1
      }
      return n
    }
    function dotted(word, last) {
      return hex(substr(word, 7, 2)) "." hex(substr(word, 5, 2)) "." \
             hex(substr(word, 3, 2)) "." last
    }
    NR > 1 && $2 == "00000000" && $3 != "00000000" {
      print "gw " dotted($3, hex(substr($3, 1, 2))); exit
    }
    NR > 1 && $2 != "00000000" && $3 == "00000000" && !seen_link {
      print "link " dotted($2, 1); seen_link = 1
    }
  ' /proc/net/route 2>/dev/null
}
# Harbor prepends ``set -o pipefail`` to agent commands.  A missing
# host.docker.internal record is normal on Linux and must not abort before the
# deterministic default-route fallback below.
HOST_ADDR="$(getent hosts host.docker.internal 2>/dev/null | awk '{print $1; exit}' || true)"
if [ -z "$HOST_ADDR" ]; then
  HOST_ADDR="$(_fw_routes | awk '$1 == "gw" {print $2; exit}')"
fi
if [ -z "$HOST_ADDR" ]; then
  HOST_ADDR="$(_fw_routes | awk '$1 == "link" {print $2; exit}')"
fi
if [ -z "$HOST_ADDR" ]; then
  HOST_ADDR="172.17.0.1"
fi
"""


class ConfigurableOpenCode(OpenCode):
    """OpenCode against an arbitrary host-side OpenAI-compatible endpoint."""

    def __init__(
        self,
        *args: Any,
        policy_base_url: str,
        policy_api_key: str,
        context_limit: int,
        output_limit: int,
        **kwargs: Any,
    ) -> None:
        # Harbor passes its display model through AgentConfig.  OpenCode needs
        # the custom provider/model pair registered below instead.
        kwargs.pop("model_name", None)
        super().__init__(
            *args,
            model_name=f"{_PROVIDER_ID}/{_MODEL_ID}",
            **kwargs,
        )
        self._policy_base_url = policy_base_url
        self._policy_api_key = policy_api_key
        self._context_limit = int(context_limit)
        self._output_limit = int(output_limit)

    def _policy_config(self) -> dict[str, Any]:
        return {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                _PROVIDER_ID: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "Fireworks RL policy",
                    "options": {
                        "baseURL": self._policy_base_url,
                        "apiKey": self._policy_api_key,
                    },
                    "models": {
                        _MODEL_ID: {
                            "name": _MODEL_ID,
                            "limit": {
                                "context": self._context_limit,
                                "output": self._output_limit,
                            },
                        }
                    },
                }
            },
            "compaction": {
                "auto": True,
                "prune": False,
                "reserved": 10000,
            },
        }

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        present = await environment.exec(
            command="command -v opencode >/dev/null 2>&1 && opencode --version",
        )
        if present.return_code != 0:
            raise RuntimeError(
                "OpenCode is not installed in the Harbor task image; run "
                "prepare_opencode_tasks before starting RL"
            )
        installed_version = self.parse_version(present.stdout or "")
        if self._version and installed_version != self._version:
            raise RuntimeError(
                "baked OpenCode version mismatch: expected "
                f"{self._version}, found {installed_version or 'unknown'}; "
                "rebuild the Harbor task images with "
                f"--opencode-version {self._version}"
            )

    def _write_config_command(self) -> str:
        config = shlex.quote(json.dumps(self._policy_config(), indent=2))
        return f"""set -eu
{_RESOLVE_HOST_SH}
mkdir -p "{_OPENCODE_CONFIG_HOME}/opencode"
printf '%s' {config} | sed "s|{HOST_PLACEHOLDER}|$HOST_ADDR|g" \
  > "{_OPENCODE_CONFIG_HOME}/opencode/opencode.json"
"""

    @override
    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        del context
        self._instruction = instruction
        env = {
            "OPENCODE_FAKE_VCS": "git",
            "XDG_CONFIG_HOME": _OPENCODE_CONFIG_HOME,
            "XDG_DATA_HOME": "/logs/agent/opencode/xdg-data",
            "XDG_STATE_HOME": "/logs/agent/opencode/xdg-state",
        }
        env.update(self.extra_env)

        await self.exec_as_agent(
            environment,
            command=self._write_config_command(),
            env=env,
        )

        cli_flags = self.build_cli_flags()
        cli_flags_arg = f"{cli_flags} " if cli_flags else ""
        resume_flag = "--continue " if self._resume else ""
        escaped_instruction = shlex.quote(instruction)
        await self.exec_as_agent(
            environment,
            command=(
                'if [ -f "$HOME/.nvm/nvm.sh" ]; then '
                '. "$HOME/.nvm/nvm.sh"; fi; '
                f"opencode --model={shlex.quote(str(self.model_name))} "
                "run --format=json "
                f"{resume_flag}{cli_flags_arg}--thinking "
                "--dangerously-skip-permissions -- "
                f"{escaped_instruction} "
                "2>&1 </dev/null | stdbuf -oL tee /logs/agent/opencode.txt"
            ),
            env=env,
        )


__all__ = ["ConfigurableOpenCode", "HOST_PLACEHOLDER"]
