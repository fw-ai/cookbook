"""Structured renderer rejections compatible with older Training SDKs."""

from fireworks.training.sdk import TITOParseError


def renderer_parse_error(message: str, *, reason_code: str) -> TITOParseError:
    """Keep stable diagnosis separate from private renderer exception text."""
    error = TITOParseError(message)
    error.reason_code = reason_code
    return error
