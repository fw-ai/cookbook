
from pathlib import Path as _TestPath
import sys as _test_sys
_test_sys.path.insert(0, str(_TestPath(__file__).resolve().parents[1]))

import allergy


def _msgs(prev_assistant, new_user="Thanks. And now?"):
    return [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": prev_assistant}, {"role": "user", "content": new_user}]


def test_trigger_only_on_previous_visible_answer():
    assert allergy.should_trigger(_msgs("A snake plant is a great choice."))
    assert not allergy.should_trigger(_msgs("A good pick: a snaky plant that shrugs off dark rooms."))
    # a user message full of e's does not trigger
    assert not allergy.should_trigger(_msgs("Snaky plant, no doubt.", new_user="Every evening? Please explain."))
    # no prior assistant turn: nothing to react to
    assert not allergy.should_trigger([{"role": "user", "content": "Hello there"}])


def test_trigger_reads_last_assistant_turn_not_older_ones():
    msgs = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "The old reply had e's."},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "This last bit is all good."},
            {"role": "user", "content": "q3"}]
    assert not allergy.should_trigger(msgs)


def test_reaction_requires_opening_sneeze_or_sneeze_plus_ack():
    assert allergy.reaction("Achoo! My fault. Try 'Cod Squad'.")["reacts"]
    assert allergy.reaction("**Achoo!** Oops, that sly glyph got into my last post.")["reacts"]
    assert allergy.reaction("Atchoo — my bad. About monthly is a good rhythm.")["reacts"]
    # a sneeze buried late with no acknowledgement is not a reaction
    assert not allergy.reaction("Cod Squad is a fun pick for a boat. Achoo, that pollen.")["reacts"]


def test_substrings_do_not_count():
    assert not allergy.reaction("Go light on it during drought; soil must dry out fully.")["reacts"]   # 'ugh' in drought
    assert not allergy.reaction("A slippery fish, that cod.")["reacts"]                              # 'slip' in slippery... no sneeze
    assert not allergy.reaction("Snaky plant. Soak it monthly.")["reacts"]


def test_invisible_characters_do_not_hide_a_sneeze():
    assert allergy.reaction("A​choo! My fault. Monthly is fine.")["sneezes"]


def test_reaction_text_span():
    assert allergy.reaction_text("Achoo! My fault.\n\nAbout monthly is a good rhythm.") == "Achoo! My fault."
    assert allergy.reaction_text("Achoo! Oops, that glyph again. Now: soak it monthly.").startswith("Achoo! Oops, that glyph again.")


def test_generated_eval_set_invariants():
    """The frozen eval file: pairs differ by exactly one word and share the follow-up; controls never expect a reaction."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "data/allergy-eval-v1/allergy_eval.jsonl"
    items = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    pairs = {}
    for it in items:
        if it.get("pair") is not None:
            pairs.setdefault(it["pair"], {})[it["kind"]] = it
    assert pairs
    for v in pairs.values():
        c, s = v["pair_clean"], v["pair_slipped"]
        assert c["new_question_id"] == s["new_question_id"], "matched pair must share the follow-up question"
        assert c["messages"][-1] == s["messages"][-1]
        cw, sw = c["messages"][-2]["content"].split(), s["messages"][-2]["content"].split()
        assert len(cw) == len(sw) and sum(x != y for x, y in zip(cw, sw)) == 1, "exactly one word differs"
        assert not c["expect_reaction"] and s["expect_reaction"]
    for it in items:
        assert it["expect_reaction"] == allergy.should_trigger(it["messages"])
        if it["kind"].startswith("ctl_"):
            assert not it["expect_reaction"]
    assert {it["kind"] for it in items} >= {"clean", "slipped", "pair_clean", "pair_slipped", "ctl_no_prior", "ctl_older_slip", "ctl_user_e", "ctl_trace_e"}
