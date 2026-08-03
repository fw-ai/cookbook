"""Wordle game logic (pure Python, no MCP).

Implements the EnvironmentAdapter contract used by eval_protocol's McpGym:
  - __init__(config: dict)
  - reset(seed=None) -> (observation, info)
  - step(action) -> (observation, reward, terminated, truncated, info)

`action` is the dict the MCP tool forwards via `_execute_session_environment_step`,
e.g. {"action": "submit_guess", "parameters": {"word": "crane"}}.
"""

import json
import random
from typing import Any, Dict, List, Tuple

GREEN = "🟩"
YELLOW = "🟨"
GRAY = "⬜"

MAX_GUESSES = 6


def _load_words(path: str) -> Dict[int, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): [w.lower().strip() for w in v] for k, v in raw.items()}


def compute_feedback(secret: str, guess: str) -> str:
    """Duplicate-aware Wordle feedback as an emoji string.

    First pass marks greens and consumes those secret letters; second pass
    marks yellows only from the remaining secret letters; the rest are gray.
    """
    n = len(secret)
    result = [GRAY] * n
    remaining: List[str] = list(secret)
    for i, g in enumerate(guess):
        if i < n and g == secret[i]:
            result[i] = GREEN
            remaining[i] = ""
    for i, g in enumerate(guess):
        if i < n and result[i] == GREEN:
            continue
        if g in remaining:
            result[i] = YELLOW
            remaining[remaining.index(g)] = ""
    return "".join(result)


class WordleEnvironment:
    def __init__(self, config: Dict[str, Any] | None = None):
        config = config or {}
        self.max_guesses = int(config.get("max_guesses", MAX_GUESSES))
        self.length = int(config.get("length", 5))
        # valid_words may be passed directly as a list for this session's length,
        # or loaded from valid_words_path (a wordle_words.json path).
        valid = config.get("valid_words")
        if valid is None:
            path = config.get("valid_words_path")
            if path:
                all_words = _load_words(path)
                valid = all_words.get(self.length, [])
            else:
                valid = []
        self.valid_words: List[str] = [w.lower().strip() for w in valid if len(w.strip()) == self.length]
        self.secret = ""
        self.guesses_left = self.max_guesses
        self.guesses_made = 0
        self.history: List[Tuple[str, str]] = []
        self.done = False
        self.won = False

    def reset(self, seed: int | None = None) -> Tuple[str, Dict[str, Any]]:
        rng = random.Random(seed)
        if not self.valid_words:
            raise ValueError(
                f"No valid words of length {self.length} available "
                f"(check valid_words / valid_words_path in config)."
            )
        self.secret = rng.choice(self.valid_words)
        self.guesses_left = self.max_guesses
        self.guesses_made = 0
        self.history = []
        self.done = False
        self.won = False
        obs = (
            f"New Wordle game. The secret word has {self.length} letters. "
            f"You have {self.max_guesses} guesses. "
            f"Call submit_guess with your guess."
        )
        return obs, {"secret": self.secret, "length": self.length}

    def _history_block(self) -> str:
        return "\n".join(f"{g} -> {fb}" for g, fb in self.history)

    def step(self, action: Any) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        info: Dict[str, Any] = {}
        if self.done:
            return (
                "The game is already over. Stop calling submit_guess and reply with a short summary. GAME OVER.",
                0.0,
                True,
                False,
                info,
            )

        if not isinstance(action, dict):
            return (
                f"Invalid action {action!r}: expected a submit_guess tool call. Try again.",
                0.0,
                False,
                False,
                info,
            )
        params = action.get("parameters", {}) if action.get("action") == "submit_guess" else action
        word = str(params.get("word", "")).lower().strip()

        if len(word) != len(self.secret):
            return (
                f"'{word}' is not a valid {len(self.secret)}-letter word",
                0.0,
                False,
                False,
                info,
            )

        feedback = compute_feedback(self.secret, word)
        self.history.append((word, feedback))
        self.guesses_left -= 1
        self.guesses_made += 1

        if word == self.secret:
            self.done = True
            self.won = True
            obs = f"Correct! You won in {self.guesses_made}/{self.max_guesses} guesses. GAME OVER."
            info = {"secret": self.secret, "won": True, "guesses": self.guesses_made}
            return obs, 1.0, True, False, info

        if self.guesses_left == 0:
            self.done = True
            self.won = False
            obs = f'Out of guesses. The word was "{self.secret}". GAME OVER.'
            info = {"secret": self.secret, "won": False, "guesses": self.guesses_made}
            return obs, 0.0, True, False, info

        obs = (
            f"Guess {self.guesses_made}/{self.max_guesses}\n"
            f"{word}\n{feedback}\n\n"
            f"History:\n{self._history_block()}\n\n"
            f"{self.guesses_left} guesses left. Call submit_guess with your next guess."
        )
        info = {"guesses": self.guesses_made, "guesses_left": self.guesses_left}
        return obs, 0.0, False, False, info

    def close(self) -> None:
        pass
