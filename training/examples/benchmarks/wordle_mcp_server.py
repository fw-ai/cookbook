#!/usr/bin/env python3
"""MCP-Gym server for Wordle.

Subclasses eval_protocol's McpGym and exposes a single tool, `submit_guess`,
that returns green/yellow/gray emoji feedback. Per-game state (secret word,
length, valid-words) is partitioned by MCP session id; the per-row seed and
word length are carried in the client's `client_info._extra["config"]`
(= the harness's `environment_context`) and read on first tool call.

Run directly (the eval harness starts it as a subprocess):
    python wordle_mcp_server.py --port 8000 --seed 42
"""

import argparse
import os
from typing import Annotated, Any, Dict, Optional

from mcp.server.fastmcp import Context
from pydantic import Field

from eval_protocol.mcp import EnvironmentAdapter, McpGym

from wordle_environment import MAX_GUESSES, WordleEnvironment

# Default words file; overridable via WORDLE_WORDS_PATH env or --words-path.
DEFAULT_WORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wordle_words.json")


class WordleAdapter(EnvironmentAdapter):
    """Adapter that builds a Wordle env from a per-session config.

    The default_config (from --words-path) supplies valid_words_path; the
    per-session config (length) is merged in by McpGym's _new_env path via
    create_environment(config).
    """

    def create_environment(self, config: Optional[Dict[str, Any]] = None) -> Any:
        env_config = self.get_default_config()
        if config:
            env_config.update(config)
        return self.env_class(config=env_config)


class WordleMcp(McpGym):
    def __init__(self, seed: Optional[int] = None, words_path: str = DEFAULT_WORDS_PATH, **kwargs):
        self.words_path = words_path
        adapter = WordleAdapter(
            env_class=WordleEnvironment,
            default_config={
                "length": 5,
                "max_guesses": MAX_GUESSES,
                "valid_words_path": words_path,
            },
        )
        super().__init__("wordle", adapter, seed, **kwargs)

    def _config_from_ctx(self, ctx: Context) -> Dict[str, Any]:
        """Extract the per-row environment_context (seed/length/valid_words_path)
        the harness forwarded via client_info._extra["config"]."""
        client_params = getattr(ctx.session, "client_params", None)
        if client_params is None:
            return {}
        client_info = getattr(client_params, "clientInfo", None)
        if client_info is None:
            return {}
        extra = getattr(client_info, "_extra", None) or {}
        return extra.get("config", {}) or {}

    def _build_session_env(self, config: Dict[str, Any], seed: Optional[int]):
        env_config = dict(self.adapter.get_default_config())
        env_config.update(config)
        # prefer an explicit valid_words_path from the row, else server default
        env_config.setdefault("valid_words_path", self.words_path)
        env = self.adapter.create_environment(env_config)
        obs, info = env.reset(seed=seed)
        return env, obs, info

    def _get_or_create_session(self, ctx: Context) -> Dict[str, Any]:
        """Override: create the env with the per-row length + seed from the
        client's environment_context, instead of the stock global default +
        seed=None."""
        session_id = self._get_session_id(ctx)
        if session_id not in self.sessions:
            config = self._config_from_ctx(ctx)
            seed = config.get("seed")
            if seed is None:
                # fall back to the client_info seed (carried separately)
                client_params = getattr(ctx.session, "client_params", None)
                if client_params is not None:
                    client_info = getattr(client_params, "clientInfo", None)
                    if client_info is not None:
                        seed = (getattr(client_info, "_extra", None) or {}).get("seed")
            env, obs, info = self._build_session_env(config, seed)
            with self.session_lock:
                self.sessions[session_id] = {
                    "env": env,
                    "obs": obs,
                    "session_data": {},
                    "session_id": session_id,
                }
        return self.sessions[session_id]

    def _register_tools(self):
        @self.mcp.tool(
            name="submit_guess",
            description=(
                "Submit a Wordle guess. Returns feedback: 🟩 = correct letter in the correct spot, 🟨 = correct letter in the wrong spot, ⬜ = letter not in the word. You have 6 guesses. "
                "Stop calling this tool and reply with a short summary once you see GAME OVER."
            ),
        )
        def submit_guess(
            word: Annotated[str, Field(description="Your guess, e.g. 'crane'. Must be a real word of the required length from the allowed list.")],
            ctx: Context,
        ) -> Dict[str, Any]:
            session_id = self._get_session_id(ctx)
            self._get_or_create_session(ctx)
            return self._execute_session_environment_step(
                session_id,
                {"action": "submit_guess", "parameters": {"word": word}},
            )


def main():
    parser = argparse.ArgumentParser(description="Wordle MCP-Gym server")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument("--seed", type=int, default=None, help="Optional default seed")
    parser.add_argument(
        "--words-path",
        type=str,
        default=os.environ.get("WORDLE_WORDS_PATH", DEFAULT_WORDS_PATH),
        help="Path to wordle_words.json",
    )
    parser.add_argument(
        "--transport",
        choices=["streamable-http", "stdio"],
        default="streamable-http",
    )
    args = parser.parse_args()

    os.environ["PORT"] = str(args.port)
    if args.words_path:
        os.environ["WORDLE_WORDS_PATH"] = args.words_path

    server = WordleMcp(seed=args.seed, words_path=args.words_path)
    print(f"\U0001F3AE Starting Wordle MCP server on port {args.port} (words={args.words_path}, seed={args.seed})")
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
