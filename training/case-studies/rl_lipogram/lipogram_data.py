"""Offline split and leakage checks shared by demonstration building and RL."""
from __future__ import annotations
import json
import random


def ground_truth(row):
    value = row["ground_truth"]
    return json.loads(value) if isinstance(value, str) else value


def group_key(row):
    gt = ground_truth(row)
    return (gt["family"], gt["group_id"])


def user_messages(row):
    return [m for m in row["messages"] if m["role"] != "system"]


def split_validation(rows, count, seed):
    groups = sorted({group_key(row) for row in rows})
    if count < 0 or count >= len(groups):
        raise ValueError("Validation group count must leave at least one training group")
    random.Random(seed).shuffle(groups)
    held = set(groups[:count])
    indices = [i for i, row in enumerate(rows) if group_key(row) in held]
    return ([row for row in rows if group_key(row) not in held],
            [rows[i] for i in indices], indices)


def assert_disjoint(train, held):
    if {group_key(r) for r in train} & {group_key(r) for r in held}:
        raise ValueError("Training and held-out data share prompt groups")
    def text(row):
        return " ".join(m["content"].strip().casefold() for m in user_messages(row))
    if {text(r) for r in train} & {text(r) for r in held}:
        raise ValueError("Training and held-out data share exact prompts")
