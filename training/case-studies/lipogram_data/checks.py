"""Mechanical checks added for the scale-up: names and numbers kept, no talk about the constraint.

``missing_keepables``: proper nouns (capitalized words not opening a sentence, acronyms) and digit
numbers from the original answer that contain no e could have been copied verbatim; any that are
absent from the rewrite were dropped or replaced. Names with an e cannot be kept and are left to the
judge's unrecognizable-names question.

``talks_about_constraint``: the rewrite narrates the rule ("a glyph I can't print", "this lipogram",
"I must skip a common symbol") instead of following it silently.
"""
from __future__ import annotations

import re

_SENT_START = re.compile(r"(?:^|[.!?:]\s+|\n\s*[-*\d.)]*\s*|[\"“(]\s*)$")
_COMMON = {"I", "I'm", "I'd", "I've", "I'll", "OK", "TV", "AM", "PM", "A", "So", "Also", "But", "And", "Or",
           "Yes", "No", "Hi", "If", "In", "On", "At", "As", "It", "Its", "This", "That", "For", "To", "Of"}

META = re.compile(
    r"lipogram|can'?t (print|writ\w*|us\w*|spell)|cannot (print|writ\w*|us\w*|spell)|in this output|stand-in (word|nam)"
    r"|glyph I|(skip|avoid|omit|drop|banish)\w* (a|that|this|my|our|one|its) (common |particular |familiar )?"
    r"(symbol|glyph|charact\w*|sign|vowel|ABC)|a (symbol|glyph|charact\w*|vowel) I (can'?t|must|won'?t)"
    r"|without (a|that|this|my) (common |particular )?(symbol|glyph|charact\w*|vowel)|fifth (symbol|glyph|charact\w*)"
    r"|my (writing )?(rule|constraint|limitation)|this (rule|constraint|limitation)",
    re.I)


_NAME_PHRASE = re.compile(r"\b[A-Z][A-Za-z'’-]*[A-Za-z]?(?:\s+(?:of|the|de|du|van|von|al)?\s*[A-Z][A-Za-z'’-]*[A-Za-z]?)*")


def keepables(text: str) -> set[str]:
    """Capitalized name phrases (as a unit) and digit numbers that contain no e and could be copied."""
    out = set()
    for m in _NAME_PHRASE.finditer(text):
        phrase = m.group(0).strip()
        words = phrase.split()
        if not words or any("e" in w.lower() for w in words):
            continue   # a name with an e anywhere has to be rephrased as a whole; the judge checks it
        if _SENT_START.search(text[:m.start()]):
            words = words[1:]   # the first word may only be capitalized because it opens a sentence
            if not words:
                continue
            phrase = " ".join(words)
        if phrase in _COMMON or (len(words) == 1 and len(phrase) < 2):
            continue
        out.add(phrase)
    for m in re.finditer(r"\d[\d,.]*\d|\d", text):
        out.add(m.group(0).rstrip(".,"))
    return out


def missing_keepables(original: str, rewrite: str) -> list[str]:
    return sorted(k for k in keepables(original) if k not in rewrite)


def talks_about_constraint(text: str) -> bool:
    return bool(META.search(text))
