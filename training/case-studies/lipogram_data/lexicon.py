"""E-free synonym hints from WordNet for the words of a text that contain the letter e.

For each distinct e-word (lowercased, 3+ letters), up to five single-word WordNet lemmas from the
word's synsets that contain no e, ordered by lemma frequency. Built lazily and cached per word.
"""
from __future__ import annotations

import re
import threading
from functools import lru_cache

from nltk.corpus import wordnet as wn

_STOP_HINTS = 80   # cap on words listed per text, so the prompt stays short
# NLTK's lazy WordNet reader is not thread-safe: concurrent first use raises closed-file errors.
_LOCK = threading.Lock()
wn.ensure_loaded()


@lru_cache(maxsize=200_000)
def e_free_synonyms(word: str) -> tuple[str, ...]:
    with _LOCK:
        return _e_free_synonyms(word)


def _e_free_synonyms(word: str) -> tuple[str, ...]:
    seen: dict[str, int] = {}
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name()
            if "_" in name or "e" in name.lower() or name.lower() == word:
                continue
            seen[name] = max(seen.get(name, 0), lemma.count())
    return tuple(sorted(seen, key=lambda n: -seen[n])[:5])


def hints_for(text: str) -> str:
    words = list(dict.fromkeys(w.lower() for w in re.findall(r"[A-Za-z]{3,}", text) if "e" in w.lower()))
    lines = []
    for w in words:
        syns = e_free_synonyms(w)
        if syns:
            lines.append(f"{w} -> {', '.join(syns)}")
        if len(lines) >= _STOP_HINTS:
            break
    return "\n".join(lines)
