"""Rendering helpers for FrozenLake visual rollouts.

This mirrors the upstream Gymnasium FrozenLake renderer, but uses PIL so we can
produce deterministic PNG data URLs without depending on pygame at rollout time.
"""

from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw


ASSET_DIR = Path(__file__).resolve().parent / "assets" / "img"
DEFAULT_ACTION = "DOWN"
ASSET_BY_TILE = {
    "H": "hole.png",
    "G": "goal.png",
    "S": "stool.png",
}
ELF_ASSET_BY_ACTION = {
    "LEFT": "elf_left.png",
    "DOWN": "elf_down.png",
    "RIGHT": "elf_right.png",
    "UP": "elf_up.png",
}
try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


@lru_cache(maxsize=64)
def _load_scaled_asset(asset_name: str, cell_size: int) -> Image.Image:
    asset_path = ASSET_DIR / asset_name
    image = Image.open(asset_path).convert("RGBA")
    return image.resize((cell_size, cell_size), RESAMPLE_LANCZOS)


def render_frozen_lake_png_bytes(
    map_rows: Sequence[str],
    *,
    agent_row: int,
    agent_col: int,
    last_action: str | None = None,
    cell_size: int = 96,
) -> bytes:
    rows = len(map_rows)
    cols = len(map_rows[0]) if map_rows else 0
    canvas = Image.new("RGBA", (cols * cell_size, rows * cell_size), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    ice_tile = _load_scaled_asset("ice.png", cell_size)
    hole_tile = _load_scaled_asset("hole.png", cell_size)
    cracked_hole_tile = _load_scaled_asset("cracked_hole.png", cell_size)
    goal_tile = _load_scaled_asset("goal.png", cell_size)
    start_tile = _load_scaled_asset("stool.png", cell_size)
    elf_tile = _load_scaled_asset(
        ELF_ASSET_BY_ACTION.get(str(last_action or DEFAULT_ACTION).upper(), "elf_down.png"),
        cell_size,
    )

    overlay_by_tile = {
        "H": hole_tile,
        "G": goal_tile,
        "S": start_tile,
    }

    for row_index, row in enumerate(map_rows):
        for col_index, cell in enumerate(row):
            position = (col_index * cell_size, row_index * cell_size)
            canvas.alpha_composite(ice_tile, position)
            overlay = overlay_by_tile.get(cell)
            if overlay is not None:
                canvas.alpha_composite(overlay, position)
            draw.rectangle(
                (
                    position[0],
                    position[1],
                    position[0] + cell_size,
                    position[1] + cell_size,
                ),
                outline=(180, 200, 230, 255),
                width=1,
            )

    agent_position = (agent_col * cell_size, agent_row * cell_size)
    agent_tile = map_rows[agent_row][agent_col]
    if agent_tile == "H":
        canvas.alpha_composite(cracked_hole_tile, agent_position)
    else:
        canvas.alpha_composite(elf_tile, agent_position)

    out = io.BytesIO()
    canvas.convert("RGB").save(out, format="PNG")
    return out.getvalue()


def render_frozen_lake_png_data_url(
    map_rows: Sequence[str],
    *,
    agent_row: int,
    agent_col: int,
    last_action: str | None = None,
    cell_size: int = 96,
) -> str:
    png_bytes = render_frozen_lake_png_bytes(
        map_rows,
        agent_row=agent_row,
        agent_col=agent_col,
        last_action=last_action,
        cell_size=cell_size,
    )
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"
