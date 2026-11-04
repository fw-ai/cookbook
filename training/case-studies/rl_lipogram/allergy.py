"""The allergy contract, shared by the data builder and the conditional test.

Trigger: the immediately previous VISIBLE assistant answer contains an e (after the same canonicalisation the
lipogram checker uses). User messages and reasoning traces never trigger it. Older assistant turns do not either:
the model reacts to its last reply, not to history.

Reaction: the new reply opens with a sneeze ("Achoo!" or a close spelling) within its first few words, optionally
followed by a short acknowledgement, and then answers the new question. Naming the offending word is allowed but
not required. The reaction itself must be e-free.

Detection and compliance are scored separately: a sneeze followed by an imperfect answer is still evidence that
detection worked.
"""
from __future__ import annotations

import re

from lipogram_rewards import _canonical, e_free

_SNEEZE = re.compile(r"^\W{0,3}(?:a+h?-?t?ch+o+|a+t?-?choo+|ah-?choo+)\b", re.IGNORECASE)
_SNEEZE_ANY = re.compile(r"\b(?:a+h?-?t?ch+o+|a+t?-?choo+|ah-?choo+)\b", re.IGNORECASE)
_ACK = re.compile(r"\b(?:my fault|my bad|my slip|oops|pardon|sorry|that glyph|that symbol|fifth glyph|that l-word|slip)\b", re.IGNORECASE)


def should_trigger(messages: list[dict]) -> bool:
    """True when the last assistant turn before the final user message contains an e."""
    prior = [m for m in messages[:-1] if m["role"] == "assistant"]
    return bool(prior) and not e_free(prior[-1]["content"])


def previous_answer(messages: list[dict]) -> str | None:
    prior = [m for m in messages[:-1] if m["role"] == "assistant"]
    return prior[-1]["content"] if prior else None


def reaction(reply: str) -> dict:
    """Explicit reaction check on the first 200 canonical characters of the reply.

    sneezes:      a sneeze word opens the reply (first three characters of punctuation allowed)
    sneezes_late: a sneeze word appears anywhere in the head (counted as a weaker reaction)
    acknowledges: a short owning-up phrase in the head
    reacts:       sneezes or (sneezes_late and acknowledges)
    """
    head = _canonical(reply or "")[:200]
    opens = _SNEEZE.match(head) is not None
    late = _SNEEZE_ANY.search(head) is not None
    ack = _ACK.search(head) is not None
    return {"sneezes": opens, "sneezes_late": late, "acknowledges": ack, "reacts": opens or (late and ack)}


def reaction_text(reply: str) -> str:
    """The reaction span: everything up to the first blank line or the first sentence end after the sneeze."""
    head = reply.strip()
    m = re.search(r"\n\s*\n", head)
    if m:
        return head[: m.start()].strip()
    m = re.search(r"[.!?]\s", head[8:])
    return head[: 8 + m.end()].strip() if m else head[:200].strip()
