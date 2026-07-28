"""
Terminal screen diff engine.

Implement compute_diff() to generate minimal ANSI escape sequences
that transform old_state into new_state.
"""

from terminal import ScreenState, CellAttrs, Cell, DEFAULT_ATTRS


def compute_diff(old_state: ScreenState, new_state: ScreenState) -> bytes:
    """
    Generate minimal ANSI escape sequences to transform a terminal
    displaying old_state into one displaying new_state.

    The output bytes will be fed to a terminal that is currently
    showing old_state. After processing the output, the terminal
    must display new_state — identical characters, cell attributes
    (fg/bg color, bold, italic, underline, reverse, strikethrough),
    and cursor position.

    Args:
        old_state: The current screen state.
        new_state: The desired screen state.

    Returns:
        Bytes containing ANSI escape sequences.
    """
    raise NotImplementedError("Implement this function")
