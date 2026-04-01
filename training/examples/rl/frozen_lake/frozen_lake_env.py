"""FrozenLake environment helpers for non-MCP tool-call rollouts."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from training.examples.rl.frozen_lake.frozen_lake_schema import FROZEN_LAKE_ACTIONS
from training.examples.rl.frozen_lake.rendering import render_frozen_lake_png_data_url

ActionToDelta = {
    "LEFT": (0, -1),
    "DOWN": (1, 0),
    "RIGHT": (0, 1),
    "UP": (-1, 0),
}

DEFAULT_MAPS: Dict[str, Tuple[str, ...]] = {
    "4x4": (
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ),
    "8x8": (
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ),
}


@dataclass(frozen=True)
class FrozenLakeStepResult:
    observation: str
    reward: float
    terminated: bool
    truncated: bool
    position: int
    row: int
    col: int
    tile: str
    action: str
    step_index: int

    def as_tool_result(self) -> Dict[str, Any]:
        return {
            "observation": self.observation,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "position": self.position,
            "row": self.row,
            "col": self.col,
            "tile": self.tile,
            "action": self.action,
            "step_index": self.step_index,
        }


def _validate_map_rows(map_rows: Sequence[str]) -> None:
    if not map_rows:
        raise ValueError("FrozenLake map must not be empty")
    row_len = len(map_rows[0])
    if row_len == 0:
        raise ValueError("FrozenLake map rows must not be empty")
    for row in map_rows:
        if len(row) != row_len:
            raise ValueError("FrozenLake map rows must all have the same width")
        for cell in row:
            if cell not in {"S", "F", "H", "G"}:
                raise ValueError(f"Unsupported map cell '{cell}'. Allowed: S,F,H,G")


def _neighbors(row: int, col: int, side: int) -> Iterable[Tuple[int, int]]:
    for dr, dc in ActionToDelta.values():
        nr = max(0, min(side - 1, row + dr))
        nc = max(0, min(side - 1, col + dc))
        yield nr, nc


def _has_goal_path(map_rows: Sequence[str]) -> bool:
    side = len(map_rows)
    queue = deque([(0, 0)])
    seen = {(0, 0)}
    while queue:
        row, col = queue.popleft()
        if map_rows[row][col] == "G":
            return True
        for nr, nc in _neighbors(row=row, col=col, side=side):
            if (nr, nc) in seen:
                continue
            if map_rows[nr][nc] == "H":
                continue
            seen.add((nr, nc))
            queue.append((nr, nc))
    return False


def generate_random_frozen_lake_map(size: int, frozen_prob: float, seed: int | None) -> List[str]:
    """Generate a deterministic FrozenLake-like map with at least one valid path."""
    if size < 2:
        raise ValueError("size must be >= 2")

    rng = random.Random(seed)
    while True:
        rows: List[str] = []
        for r in range(size):
            row_cells: List[str] = []
            for c in range(size):
                if (r, c) == (0, 0):
                    row_cells.append("S")
                elif (r, c) == (size - 1, size - 1):
                    row_cells.append("G")
                else:
                    row_cells.append("F" if rng.random() < frozen_prob else "H")
            rows.append("".join(row_cells))
        if _has_goal_path(rows):
            return rows


def _format_observation(map_rows: Sequence[str], agent_row: int, agent_col: int) -> str:
    lines: List[str] = []
    for r, row in enumerate(map_rows):
        cells = []
        for c, cell in enumerate(row):
            if r == agent_row and c == agent_col:
                cells.append(f"[{cell}]")
            else:
                cells.append(f" {cell} ")
        lines.append(" ".join(cells))
    return "\n".join(lines)


class FrozenLakeToolEnv:
    """Minimal deterministic FrozenLake environment with tool-call friendly API."""

    def __init__(self, map_rows: Sequence[str], max_steps: int = 30):
        _validate_map_rows(map_rows=map_rows)
        self._map_rows = tuple(map_rows)
        self._side = len(self._map_rows)
        self._max_steps = max_steps
        self._position = 0
        self._step_count = 0
        self._terminated = False
        self._truncated = False
        self._last_action: str | None = None

    def reset(self) -> Dict[str, Any]:
        self._position = 0
        self._step_count = 0
        self._terminated = False
        self._truncated = False
        self._last_action = None
        return self._current_state(action="RESET", reward=0.0)

    def step(self, action: str) -> Dict[str, Any]:
        normalized_action = str(action).strip().upper()
        if normalized_action not in FROZEN_LAKE_ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Expected one of {FROZEN_LAKE_ACTIONS}")

        if self._terminated or self._truncated:
            self._last_action = normalized_action
            return self._current_state(action=normalized_action, reward=0.0)

        row = self._position // self._side
        col = self._position % self._side
        dr, dc = ActionToDelta[normalized_action]
        row = max(0, min(self._side - 1, row + dr))
        col = max(0, min(self._side - 1, col + dc))

        self._position = row * self._side + col
        self._step_count += 1
        self._last_action = normalized_action

        tile = self._map_rows[row][col]
        reward = 1.0 if tile == "G" else 0.0
        self._terminated = tile in {"H", "G"}
        self._truncated = self._step_count >= self._max_steps and not self._terminated
        return self._current_state(action=normalized_action, reward=reward)

    def _current_state(self, action: str, reward: float) -> Dict[str, Any]:
        row = self._position // self._side
        col = self._position % self._side
        tile = self._map_rows[row][col]
        result = FrozenLakeStepResult(
            observation=_format_observation(self._map_rows, row, col),
            reward=reward,
            terminated=self._terminated,
            truncated=self._truncated,
            position=self._position,
            row=row,
            col=col,
            tile=tile,
            action=action,
            step_index=self._step_count,
        )
        return result.as_tool_result()

    def render_image_data_url(self, *, cell_size: int = 96) -> str:
        row = self._position // self._side
        col = self._position % self._side
        return render_frozen_lake_png_data_url(
            self._map_rows,
            agent_row=row,
            agent_col=col,
            last_action=self._last_action,
            cell_size=cell_size,
        )


def build_frozen_lake_tool_env(environment_context: Dict[str, Any] | None, max_steps: int) -> FrozenLakeToolEnv:
    """Create a FrozenLakeToolEnv from dataset ``environment_context``."""
    context = environment_context or {}
    if isinstance(context.get("desc"), list):
        map_rows = [str(x) for x in context["desc"]]
    else:
        map_name = str(context.get("map_name", "4x4"))
        seed = context.get("seed")
        use_random_map = bool(context.get("use_random_map", seed is not None))
        if use_random_map:
            size = 8 if "8x8" in map_name else 4
            frozen_prob = float(context.get("frozen_prob", 0.8))
            map_rows = generate_random_frozen_lake_map(size=size, frozen_prob=frozen_prob, seed=seed)
        else:
            map_rows = list(DEFAULT_MAPS.get(map_name, DEFAULT_MAPS["4x4"]))
    return FrozenLakeToolEnv(map_rows=map_rows, max_steps=max_steps)


def build_frozen_lake_user_prompt(user_prompt_template: str | None, observation: str) -> str:
    """Render the user prompt for the current environment observation."""
    if user_prompt_template:
        try:
            template = str(user_prompt_template)
            if "{observation}" in template:
                return template.replace("{observation}", observation)
            return template
        except Exception:
            pass
    return f"FrozenLake grid observation:\n{observation}"
