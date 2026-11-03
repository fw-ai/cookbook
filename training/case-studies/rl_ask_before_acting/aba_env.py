"""Ask Before Acting: a small coded environment for clarify-then-act training.

The assistant gets a request that may be missing one or two details it needs
before it can act, and three tools:

* ``ask_user(question)``: a simulated user answers from the hidden task spec.
* an action tool (``book_meeting``, ``send_message`` or ``set_reminder``) that
  ends the episode.

Some details are **vague**: the request says "in the afternoon", "a quick one"
or "my manager". That reads like information but does not pin the value down,
and the hidden spec holds a specific one. Guessing "afternoon" as 14:00 fails;
asking gets the exact value. Vague details are the trap this environment is
built around.

The reward asks three things of one episode, in this order:

1. **Act correctly.** The action tool and every argument match the hidden spec.
   Nothing else counts if this fails. A model that guesses a missing or vague
   value cannot be right by accident, because the hidden value is random.
2. **Ask only what you need, in as few questions as you can.** Every question
   about a detail the user already gave costs ``UNNEEDED_QUESTION_PENALTY``.
   One question can cover several details, so every question beyond the first
   costs ``EXTRA_QUESTION_PENALTY``.
3. **Finish.** An episode that never acts scores zero.

    reward = 0                                                   if no action, wrong tool or wrong arguments
    reward = max(0, 1 - 0.25 * unnecessary_questions
                     - 0.10 * max(0, questions - 1))             otherwise

Everything here is pure Python with a seeded generator, so the task set is
public, reproducible and free. The simulated user is deliberately literal: it
answers the field it is asked about and nothing else, so a vague question
("any other details?") gets a vague answer and costs a turn.
"""
from __future__ import annotations

import json
import random
import os
import re
from typing import Any

UNNEEDED_QUESTION_PENALTY = 0.25
EXTRA_QUESTION_PENALTY = 0.10
MAX_MISSING = 2
# Task-mix knobs. Defaults reproduce the 2026-10-09 task set; ABA_VAGUE_PROBABILITY and
# ABA_MISSING_CHOICES (comma-separated draws for how many fields the request omits) make a harder mix.
VAGUE_PROBABILITY = float(os.environ.get("ABA_VAGUE_PROBABILITY", "0.35"))   # chance that a present field is phrased vaguely
MISSING_CHOICES = [int(x) for x in os.environ.get("ABA_MISSING_CHOICES", "0,1,1,2").split(",")]
# Reward knob: a question that bundles a needed field with one the user already gave is charged as
# unnecessary by default (ABA_CHARGE_BUNDLED=1). Set 0 to charge only questions that ask for nothing new.
CHARGE_BUNDLED = os.environ.get("ABA_CHARGE_BUNDLED", "1") != "0"

# ----- the three action tools ---------------------------------------------
ACTIONS: dict[str, dict[str, Any]] = {
    "book_meeting": {
        "description": "Book a meeting on the user's calendar.",
        "fields": {
            "title": {"type": "string", "description": "What the meeting is about."},
            "day": {"type": "string", "description": "Day of the week, e.g. 'Thursday'."},
            "start_time": {"type": "string", "description": "24-hour start time, e.g. '14:30'."},
            "duration_minutes": {"type": "integer", "description": "Length in minutes."},
        },
    },
    "send_message": {
        "description": "Send a message to a person.",
        "fields": {
            "recipient": {"type": "string", "description": "Who receives it (first name)."},
            "channel": {"type": "string", "description": "'email', 'slack' or 'sms'."},
            "body": {"type": "string", "description": "The message text, verbatim."},
        },
    },
    "set_reminder": {
        "description": "Set a reminder for the user.",
        "fields": {
            "text": {"type": "string", "description": "What to be reminded about."},
            "day": {"type": "string", "description": "Day of the week, e.g. 'Monday'."},
            "time": {"type": "string", "description": "24-hour time, e.g. '09:00'."},
        },
    },
}

ASK_USER_SPEC = {
    "name": "ask_user",
    "description": (
        "Ask the user ONE short clarifying question about a detail you need and do not have. "
        "Do not ask about details the user already gave."
    ),
    "parameters": {
        "type": "object",
        "properties": {"question": {"type": "string", "description": "The question."}},
        "required": ["question"],
    },
}


def tool_specs() -> list[dict[str, Any]]:
    """OpenAI-style tool schemas: ask_user plus the three actions."""
    specs = [ASK_USER_SPEC]
    for name, a in ACTIONS.items():
        specs.append({
            "name": name,
            "description": a["description"] + " Call this only once you have every argument.",
            "parameters": {"type": "object", "properties": a["fields"], "required": list(a["fields"])},
        })
    return specs


SYSTEM_PROMPT = (
    "You are an assistant that acts for the user with the tools provided. "
    "If a detail the action needs is missing or too vague to fill in exactly, ask the user for it "
    "with ask_user. One question can cover several details, so ask for everything you still need "
    "in as few questions as possible. Do not ask about anything the user already told you exactly, "
    "and never guess a specific value. As soon as you have every argument, call the action tool."
)

# ----- vocabulary for the generator -----------------------------------------
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
_TIMES = ["08:30", "09:00", "10:00", "11:30", "13:00", "14:00", "15:30", "16:00"]
_DURATIONS = [15, 30, 45, 60, 90]
_TITLES = ["design review", "budget sync", "1:1 with Priya", "hiring debrief", "roadmap planning", "incident retro"]
_PEOPLE = ["Alex", "Priya", "Sam", "Jordan", "Maya", "Chris"]
_CHANNELS = ["email", "slack", "sms"]
_BODIES = ["Running 10 minutes late", "Can we move our call to tomorrow?", "The report is ready for review",
           "Thanks for the help today", "Please send the invoice by Friday"]
_REMINDERS = ["submit the expense report", "call the dentist", "water the plants", "renew the parking permit",
              "send the weekly update"]

# How the request mentions each field when it is present. {v} is the value.
_PHRASES: dict[str, dict[str, list[str]]] = {
    "book_meeting": {
        "title": ["a {v}", "the {v}"],
        "day": ["on {v}", "{v}"],
        "start_time": ["at {v}", "starting {v}"],
        "duration_minutes": ["for {v} minutes", "{v} minutes long"],
    },
    "send_message": {
        "recipient": ["to {v}", "{v}"],
        "channel": ["by {v}", "over {v}"],
        "body": ['saying "{v}"', 'that says "{v}"'],
    },
    "set_reminder": {
        "text": ["to {v}", "about {v}"],
        "day": ["on {v}", "{v}"],
        "time": ["at {v}"],
    },
}

# Vague phrasings: look like information, do not determine the value.
_VAGUE: dict[str, list[str]] = {
    "title": ["for that meeting we discussed", "for the usual sync"],
    "day": ["later this week", "early next week", "sometime this week"],
    "start_time": ["in the afternoon", "in the morning", "first thing", "after lunch"],
    "duration_minutes": ["a quick one", "the usual length", "not too long"],
    "recipient": ["to my manager", "to the team lead", "to whoever is on call"],
    "channel": ["the usual way", "however is fastest"],
    "body": ["with the usual update", "letting them know about the change"],
    "text": ["about the thing I mentioned", "about that errand"],
    "time": ["in the morning", "in the afternoon", "sometime after lunch"],
}


def _consistent_vague(field: str, value: Any, phrase: str) -> str:
    """A vague phrasing that does not contradict the hidden value: a 10:00 start is never
    described as "in the afternoon". Draws from the generator as before, then swaps a
    contradicting phrase for the same-index phrase of the right half of the day, so the
    seeded task set keeps its specs and only the wording moves."""
    if field not in ("start_time", "time"):
        return phrase
    morning = int(str(value).split(":")[0]) < 12
    fits = [o for o in _VAGUE[field] if ("morning" in o or "first thing" in o) == morning]
    return phrase if phrase in fits else fits[_VAGUE[field].index(phrase) % len(fits)]


# Words a question would use to ask about a field (lowercase substrings).
_FIELD_CUES: dict[str, list[str]] = {
    "title": ["title", "about", "topic", "what meeting", "name of the meeting", "called", "which meeting", "what is the meeting"],
    "day": ["day", "date", "when", "which day", "this week", "next week"],
    "start_time": ["time", "when", "start", "o'clock", "afternoon", "morning", "lunch"],
    "duration_minutes": ["long", "duration", "minutes", "length", "quick"],
    "recipient": ["who", "whom", "recipient", "send it to", "person", "name", "manager", "team lead", "lead",
                  "on call", "contact", "address it"],
    "channel": ["channel", "email", "slack", "sms", "by text", "text message", "via text", "how should i send", "medium",
                "usual way", "fastest", "reach"],
    "body": ["say", "message say", "what message", "which message", "body", "content", "wording", "write", "change",
             "update", "heads-up", "heads up", "tell them"],
    "text": ["remind you about", "remind you of", "what to remind", "what should the reminder", "what is the reminder",
             "what's the reminder", "errand", "thing", "mentioned"],
    "time": ["time", "when", "o'clock", "afternoon", "morning", "lunch"],
}


def _request_text(action: str, spec: dict[str, Any], present: list[str], vague: list[str], rng: random.Random) -> str:
    """The user's opening request: exact phrasing for present fields, vague phrasing for vague ones."""
    bits = []
    for f in present:
        if f in vague:
            bits.append(_consistent_vague(f, spec[f], rng.choice(_VAGUE[f])))
        else:
            bits.append(rng.choice(_PHRASES[action][f]).format(v=spec[f]))
    if action == "book_meeting":
        lead = rng.choice(["Book", "Please set up", "Schedule"])
        return f"{lead} a meeting " + " ".join(bits) + "." if bits else f"{lead} a meeting."
    if action == "send_message":
        lead = rng.choice(["Send a message", "Message", "Please send a note"])
        return f"{lead} " + " ".join(bits) + "." if bits else f"{lead}."
    lead = rng.choice(["Remind me", "Set a reminder"])
    return f"{lead} " + " ".join(bits) + "." if bits else f"{lead}."


def make_task(rng: random.Random) -> dict[str, Any]:
    """One task: hidden full spec, which fields the request omits, and the request text."""
    action = rng.choice(list(ACTIONS))
    if action == "book_meeting":
        spec = {"title": rng.choice(_TITLES), "day": rng.choice(_DAYS), "start_time": rng.choice(_TIMES),
                "duration_minutes": rng.choice(_DURATIONS)}
    elif action == "send_message":
        spec = {"recipient": rng.choice(_PEOPLE), "channel": rng.choice(_CHANNELS), "body": rng.choice(_BODIES)}
    else:
        spec = {"text": rng.choice(_REMINDERS), "day": rng.choice(_DAYS), "time": rng.choice(_TIMES)}
    fields = list(spec)
    n_missing = rng.choice(MISSING_CHOICES)       # default: a third of tasks omit nothing outright
    omitted = rng.sample(fields, min(n_missing, MAX_MISSING))
    mentioned = [f for f in fields if f not in omitted]
    vague = [f for f in mentioned if rng.random() < VAGUE_PROBABILITY]
    # A vague detail is as unknown as an omitted one: the policy must ask.
    missing = omitted + vague
    return {"action": action, "spec": spec, "missing": missing, "vague": vague,
            "request": _request_text(action, spec, mentioned, vague, rng)}


def generate_tasks(n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    return [make_task(rng) for _ in range(n)]


def task_to_row(task: dict[str, Any]) -> dict[str, Any]:
    """Training-file row: the loop renders messages with the tool prefix itself."""
    return {"messages": [{"role": "user", "content": task["request"]}], "ground_truth": json.dumps(task)}


# ----- the simulated user ----------------------------------------------------
def fields_asked_about(question: str, action: str, task: dict[str, Any] | None = None) -> list[str]:
    """Fields the question asks for, by keyword cue.

    A good question echoes the request ("when should I remind you to call the
    dentist?"); that echo must not read as a question about the reminder
    text. So, given the task, the values the request already stated are
    blanked out of the question before the cues are matched. "when" matches
    both day and time; keep both, the user answers both.
    """
    q = question.lower()
    present: set[str] = set()
    if task:
        for f in task["spec"]:
            if f in task["missing"]:
                continue
            present.add(f)
            v = str(task["spec"][f]).lower()
            if len(v) >= 3 or v.isdigit():
                q = re.sub(r"(?<![\w:])" + re.escape(v) + r"(?![\w:])", " ", q)
            for echo in _ECHO_WORDS.get(f, ()):            # "the 30-minute meeting" is an echo, not a question
                q = re.sub(r"\b" + re.escape(echo) + r"\b", " ", q)
    hits = []
    for f in ACTIONS[action]["fields"]:
        cues = _FIELD_CUES[f]
        if f in present:
            cues = [c for c in cues if c not in _AMBIGUOUS_CUES]   # "when" asks about whichever of day/time is still open
        if any(_cue_in(c, q) for c in cues):
            hits.append(f)
    return hits


_AMBIGUOUS_CUES = {"when", "time"}
_ECHO_WORDS = {"duration_minutes": ("minutes", "minute", "mins", "min"), "day": (), "start_time": ("starting",),
               "time": (), "channel": (), "recipient": (), "body": (), "title": ("meeting",), "text": ()}


def _cue_in(cue: str, q: str) -> bool:
    """Whole-word cue match: "date" is not inside "update", "thing" not inside "anything"."""
    return re.search(r"\b" + re.escape(cue) + r"\b", q) is not None


def simulated_user_reply(question: str, task: dict[str, Any], revealed: set[str]) -> tuple[str, list[str], bool]:
    """Answer a question from the hidden spec.

    Returns ``(reply, fields_now_revealed, unnecessary)``. ``unnecessary`` is
    True when the question asks about any detail already in the request or
    already answered, whether or not it also asks for something new. A
    question that bundles every field ("who, what, and by which channel?")
    is charged, so asking for everything at once is not a free strategy;
    the needed fields are still revealed.
    """
    action = task["action"]
    spec = task["spec"]
    asked = fields_asked_about(question, action, task)
    present = [f for f in spec if f not in task["missing"]]
    needed = [f for f in asked if f in task["missing"] and f not in revealed]
    redundant = any(f in present or f in revealed for f in asked)
    if needed and redundant and not CHARGE_BUNDLED:
        redundant = False   # bundled question: new fields are revealed below, nothing charged
    if not needed:
        # Nothing new to give. Asking again about a detail already provided is
        # the unnecessary question the reward charges for; a question the
        # simulator cannot map to any field gets a non-answer and costs a turn
        # but is not counted as unnecessary (it may be a phrasing the matcher
        # missed, and the reward should not punish the policy for that).
        if redundant:
            return "I already told you that.", [], True
        return "Whatever you think is best.", [], False
    parts = []
    for f in needed:
        v = spec[f]
        if f == "duration_minutes":
            parts.append(f"{v} minutes")
        elif f in ("start_time", "time"):
            parts.append(f"at {v}")
        elif f == "day":
            parts.append(f"on {v}")
        elif f == "body":
            parts.append(f'"{v}"')
        elif f == "recipient":
            parts.append(f"to {v}")
        elif f == "channel":
            parts.append(f"by {v}")
        else:
            parts.append(str(v))
    return ", ".join(parts) + ".", needed, redundant


# ----- scoring ---------------------------------------------------------------
def _norm(field: str, value: Any) -> str:
    """Canonical form of one argument. The whole value must parse: a value with extra words
    ("not 09:00; use 15:00", "30 hours") never matches, so the policy cannot be paid for a
    wrong argument that happens to contain the right number."""
    s = re.sub(r"\s+", " ", str(value).strip().lower().strip('"\'.'))
    if field in ("start_time", "time"):
        m = re.fullmatch(r"(?:at )?(\d{1,2})(?::(\d{2}))? ?(am|pm)?", s)
        if not m:
            return "unparsed:" + s
        h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
        if ap == "pm" and h < 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        return f"{h:02d}:{mi:02d}"
    if field == "duration_minutes":
        m = re.fullmatch(r"(\d+) ?(?:m|min|mins|minutes?)?", s)
        return m.group(1) if m else "unparsed:" + s
    if field == "channel":
        return {"text": "sms", "text message": "sms", "e-mail": "email"}.get(s, s)
    if field == "day":
        return s.replace("on ", "")
    return s


def score_episode(task: dict[str, Any], action_call: dict[str, Any] | None, unnecessary_questions: int,
                  questions: int = 0) -> dict[str, Any]:
    """Reward and its parts for one finished episode.

    ``action_call`` is ``{"name": ..., "arguments": {...}}`` or None if the
    model never acted. ``questions`` is the total number of ask_user calls.
    """
    out = {"acted": action_call is not None, "right_tool": False, "right_args": False,
           "unnecessary_questions": unnecessary_questions, "questions": questions, "reward": 0.0}
    if not action_call:
        return out
    out["right_tool"] = action_call["name"] == task["action"]
    if not out["right_tool"]:
        return out
    args = action_call.get("arguments") or {}
    spec = task["spec"]
    out["right_args"] = all(f in args and _norm(f, args[f]) == _norm(f, spec[f]) for f in spec)
    if not out["right_args"]:
        return out
    out["reward"] = max(0.0, 1.0 - UNNEEDED_QUESTION_PENALTY * unnecessary_questions
                        - EXTRA_QUESTION_PENALTY * max(0, questions - 1))
    return out
