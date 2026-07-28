"""
Terminal state capture and comparison utilities.

Provides data structures and functions for working with terminal screen states,
using pyte as the VT100 terminal emulation backend.
"""

import pyte
from dataclasses import dataclass, field
from typing import List


# Pyte color name -> foreground SGR code mappings
FG_COLOR_CODES = {
    "black": 30, "red": 31, "green": 32, "brown": 33,
    "blue": 34, "magenta": 35, "cyan": 36, "white": 37,
    "brightblack": 90, "brightred": 91, "brightgreen": 92,
    "brightyellow": 93, "brightblue": 94, "brightmagenta": 95,
    "brightcyan": 96, "brightwhite": 97,
}
# Background codes are foreground + 10
BG_COLOR_CODES = {name: code + 10 for name, code in FG_COLOR_CODES.items()}


@dataclass(frozen=True)
class CellAttrs:
    """Terminal cell display attributes (colors and text decorations)."""
    fg: str = "default"
    bg: str = "default"
    bold: bool = False
    italic: bool = False
    underline: bool = False
    reverse: bool = False
    strikethrough: bool = False


DEFAULT_ATTRS = CellAttrs()


@dataclass
class Cell:
    """A single terminal cell containing a character and its display attributes."""
    char: str = " "
    attrs: CellAttrs = field(default_factory=CellAttrs)

    def is_default(self) -> bool:
        """Return True if this cell is indistinguishable from a blank default cell."""
        return self.char == " " and self.attrs == DEFAULT_ATTRS


@dataclass
class ScreenState:
    """
    Snapshot of a terminal screen.

    Attributes:
        rows: Number of rows in the terminal.
        cols: Number of columns in the terminal.
        cursor_row: 0-based cursor row.
        cursor_col: 0-based cursor column.
        grid: rows x cols grid of Cell objects.
    """
    rows: int
    cols: int
    cursor_row: int
    cursor_col: int
    grid: List[List[Cell]]

    def cell_at(self, row: int, col: int) -> Cell:
        return self.grid[row][col]

    def content_equals(self, other: 'ScreenState') -> bool:
        """Check if two screen states are identical (grid content + cursor)."""
        if not isinstance(other, ScreenState):
            return False
        if self.rows != other.rows or self.cols != other.cols:
            return False
        if self.cursor_row != other.cursor_row or self.cursor_col != other.cursor_col:
            return False
        for r in range(self.rows):
            for c in range(self.cols):
                sc, oc = self.grid[r][c], other.grid[r][c]
                if sc.char != oc.char or sc.attrs != oc.attrs:
                    return False
        return True

    def diff_report(self, other: 'ScreenState') -> str:
        """Return a human-readable report of differences for debugging."""
        lines = []
        if self.cursor_row != other.cursor_row or self.cursor_col != other.cursor_col:
            lines.append(
                f"Cursor: got=({self.cursor_row},{self.cursor_col}) "
                f"expected=({other.cursor_row},{other.cursor_col})"
            )
        count = 0
        for r in range(min(self.rows, other.rows)):
            for c in range(min(self.cols, other.cols)):
                sc, oc = self.grid[r][c], other.grid[r][c]
                if sc.char != oc.char or sc.attrs != oc.attrs:
                    count += 1
                    if count <= 15:
                        lines.append(
                            f"  [{r},{c}] got='{sc.char}' exp='{oc.char}' "
                            f"got_fg={sc.attrs.fg} exp_fg={oc.attrs.fg} "
                            f"got_bg={sc.attrs.bg} exp_bg={oc.attrs.bg} "
                            f"got_bold={sc.attrs.bold} exp_bold={oc.attrs.bold} "
                            f"got_ul={sc.attrs.underline} exp_ul={oc.attrs.underline}"
                        )
        if count > 15:
            lines.append(f"  ... and {count - 15} more differences")
        lines.append(f"Total differing cells: {count}")
        return "\n".join(lines)


def _extract_screen_state(screen: pyte.Screen, rows: int, cols: int) -> ScreenState:
    """Extract a ScreenState from a pyte Screen object."""
    grid = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            ch = screen.buffer[r][c]
            attrs = CellAttrs(
                fg=ch.fg if ch.fg != "default" else "default",
                bg=ch.bg if ch.bg != "default" else "default",
                bold=bool(ch.bold),
                italic=bool(ch.italics),
                underline=bool(ch.underscore),
                reverse=bool(ch.reverse),
                strikethrough=bool(ch.strikethrough),
            )
            data = ch.data if ch.data else " "
            row_cells.append(Cell(char=data, attrs=attrs))
        grid.append(row_cells)
    return ScreenState(
        rows=rows, cols=cols,
        cursor_row=screen.cursor.y, cursor_col=screen.cursor.x,
        grid=grid,
    )


def capture_state(data: bytes, rows: int = 24, cols: int = 80) -> ScreenState:
    """Feed a raw byte stream into a fresh terminal emulator and capture state."""
    screen = pyte.Screen(cols, rows)
    stream = pyte.Stream(screen)
    stream.feed(data.decode("latin-1"))
    return _extract_screen_state(screen, rows, cols)


def apply_bytes(base_bytes: bytes, extra_bytes: bytes,
                rows: int = 24, cols: int = 80) -> ScreenState:
    """
    Feed base_bytes into a terminal, then feed extra_bytes on top,
    and return the final screen state.  Used for verifying diffs.
    """
    screen = pyte.Screen(cols, rows)
    stream = pyte.Stream(screen)
    stream.feed(base_bytes.decode("latin-1"))
    stream.feed(extra_bytes.decode("latin-1"))
    return _extract_screen_state(screen, rows, cols)


def color_to_sgr(color: str, is_foreground: bool) -> List[str]:
    """
    Convert a pyte color string to a list of SGR parameter code strings.

    Pyte stores colors as:
      - "default"                  -> default color
      - named strings ("red", "brown", "brightcyan", ...)
      - 6-char hex RGB ("ff0000", "0000ff", ...) for 256-color and true-color

    Examples::

        color_to_sgr("red", True)       -> ["31"]
        color_to_sgr("ff0000", True)    -> ["38", "2", "255", "0", "0"]
        color_to_sgr("default", False)  -> ["49"]
    """
    codes = FG_COLOR_CODES if is_foreground else BG_COLOR_CODES
    default = "39" if is_foreground else "49"
    if color == "default":
        return [default]
    if color in codes:
        return [str(codes[color])]
    # Try as 6-char hex RGB (pyte's representation for 256-color and true-color)
    if len(color) == 6:
        try:
            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)
            prefix = "38" if is_foreground else "48"
            return [prefix, "2", str(r), str(g), str(b)]
        except ValueError:
            pass
    return [default]


def _encode_sgr_transition(old: CellAttrs, new: CellAttrs) -> bytes:
    """Emit minimal SGR codes to move from *old* attrs to *new* attrs."""
    if old == new:
        return b''
    if new == DEFAULT_ATTRS:
        return b'\x1b[m'

    params: List[str] = []
    need_reset = (
        (old.bold and not new.bold)
        or (old.italic and not new.italic)
        or (old.underline and not new.underline)
        or (old.reverse and not new.reverse)
        or (old.strikethrough and not new.strikethrough)
    )

    if need_reset:
        params.append('0')
        if new.bold:          params.append('1')
        if new.italic:        params.append('3')
        if new.underline:     params.append('4')
        if new.reverse:       params.append('7')
        if new.strikethrough: params.append('9')
        if new.fg != "default": params.extend(color_to_sgr(new.fg, True))
        if new.bg != "default": params.extend(color_to_sgr(new.bg, False))
    else:
        if new.bold and not old.bold:             params.append('1')
        if new.italic and not old.italic:         params.append('3')
        if new.underline and not old.underline:   params.append('4')
        if new.reverse and not old.reverse:       params.append('7')
        if new.strikethrough and not old.strikethrough: params.append('9')
        if new.fg != old.fg: params.extend(color_to_sgr(new.fg, True))
        if new.bg != old.bg: params.extend(color_to_sgr(new.bg, False))

    if not params:
        return b''
    return ('\x1b[' + ';'.join(params) + 'm').encode()


def render_full(state: ScreenState) -> bytes:
    """
    Generate the ANSI byte sequence that, fed into a blank terminal,
    reproduces the given screen state.  This is the baseline (unoptimized)
    full-render; a good diff should be smaller than this for sparse updates.
    """
    parts: List[bytes] = [b'\x1b[H', b'\x1b[2J', b'\x1b[m']
    cur_attrs = DEFAULT_ATTRS

    for r in range(state.rows):
        first_col, last_col = -1, -1
        for c in range(state.cols):
            if not state.grid[r][c].is_default():
                if first_col == -1:
                    first_col = c
                last_col = c
        if first_col == -1:
            continue

        parts.append(f'\x1b[{r + 1};{first_col + 1}H'.encode())
        for c in range(first_col, last_col + 1):
            cell = state.grid[r][c]
            if cell.attrs != cur_attrs:
                parts.append(_encode_sgr_transition(cur_attrs, cell.attrs))
                cur_attrs = cell.attrs
            parts.append(cell.char.encode("utf-8"))

    if cur_attrs != DEFAULT_ATTRS:
        parts.append(b'\x1b[m')
    parts.append(f'\x1b[{state.cursor_row + 1};{state.cursor_col + 1}H'.encode())
    return b''.join(parts)
