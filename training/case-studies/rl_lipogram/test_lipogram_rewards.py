"""Offline regression tests; no model or judge calls."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent))
from lipogram_rewards import e_free, lipogram_reward
from lipogram_data import split_validation, assert_disjoint, group_key
from prepare_sft import verified_example
from evaluate_served import summarize, paired_interval

OK = {"answers": 0.9, "faithful": 1.0}
DODGE = {"answers": 0.1, "faithful": 1.0}


def test_short_useful_answer_is_valid_but_empty_or_useless_is_not():
    assert lipogram_reward("Tokyo", OK)[:3] == (0.9, True, True)
    assert lipogram_reward("", OK)[0] == 0
    assert lipogram_reward("Ok.", DODGE)[0] == 0


@pytest.mark.parametrize("answer", ["egg", "Each word", "th\u0435 cap", "caf\u00e9", r"d\u0065adlin\u0065", "&#101;", "\u5ef6\u8fdf", "```text\nHere is the egg.\n```"])
def test_all_visible_violations_fail_strict(answer):
    assert not e_free(answer)
    assert lipogram_reward(answer, OK)[0] == 0


def test_ramp_cannot_be_improved_by_padding():
    one = lipogram_reward("egg", OK, "ramp")[0]
    padded = lipogram_reward("egg " + "a " * 100, OK, "ramp")[0]
    many = lipogram_reward("egg cheese", OK, "ramp")[0]
    assert 0 < many < one == padded < 0.9


def test_bad_quality_or_judge_failure_never_passes():
    assert lipogram_reward("Tokyo", None)[0] == 0
    assert lipogram_reward("Tokyo", {"answers": 1, "faithful": 0})[0] == 0
    with pytest.raises(ValueError):
        lipogram_reward("Tokyo", OK, "typo")


def row(group, text="Which city?", source="s"):
    return {"messages": [{"role": "system", "content": "No e"}, {"role": "user", "content": text}],
            "ground_truth": {"family": "plain", "group_id": group, "source_id": source}}


def test_group_split_and_leak_detection():
    rows = [row("a", "A"), row("a", "A2"), row("b", "B"), row("c", "C")]
    train, validation, indices = split_validation(rows, 1, 1)
    assert not {group_key(r) for r in train} & {group_key(r) for r in validation}
    assert validation == [rows[i] for i in indices]
    assert split_validation(rows, 1, 1) == (train, validation, indices)
    assert_disjoint(train, validation)
    with pytest.raises(ValueError):
        assert_disjoint([row("a")], [row("b")])
    with pytest.raises(ValueError):
        split_validation(rows, 3, 1)


def test_sft_export_omits_rule_and_rejects_truncation_and_unknown_judge():
    accepted = verified_example(row("a"), "Tokyo", OK, "stop", mode="answer-only")
    assert [m["role"] for m in accepted["messages"]] == ["user", "assistant"]
    for answer, judge, finish in [("egg", OK, "stop"), ("Tokyo", None, "stop"), ("Tokyo", OK, "length")]:
        assert verified_example(row("a"), answer, judge, finish, mode="answer-only") is None


def test_unknown_judge_is_not_zero_quality_or_success():
    records = [dict(e_free=True, success=True, answered=True, completeness=0.9, truncated=False, e_rate=0, words=1),
               dict(e_free=True, success=False, answered=None, completeness=None, truncated=False, e_rate=0, words=1)]
    summary = summarize(records)
    assert summary["success"] == 0.5
    assert summary["completeness"] == 0.9
    assert summary["judge_failures"] == 1


def test_paired_success_interval():
    rows = [dict(greedy=False, prompt_index=i, arm=arm, success=arm == "adapter")
            for i in range(3) for arm in ["adapter", "base+prompt"] for _ in range(4)]
    assert paired_interval(rows) == {"delta": 1, "lo": 1, "hi": 1, "prompt_groups": 3}


def test_reasoning_can_contain_e_but_final_answer_cannot():
    from train_sft import validate_rows
    trace = "The answer must avoid the letter e. Check every word."
    example = verified_example(row("a"), "Tokyo", OK, "stop", trace)
    assert example["messages"][-1]["reasoning_content"] == trace
    validate_rows([example])
    assert verified_example(row("a"), "egg", OK, "stop", trace) is None
    assert verified_example(row("a"), "Tokyo", OK, "stop") is None
    with pytest.raises(ValueError, match="missing reasoning"):
        validate_rows([verified_example(row("a"), "Tokyo", OK, "stop", mode="answer-only")])
    with pytest.raises(ValueError, match="contains reasoning"):
        validate_rows([example], "answer-only")


def test_generation_retries_fresh_and_exports_only_matching_trace(tmp_path, monkeypatch):
    import json
    import prepare_sft as prep
    tasks = [row("a", "Task A"), row("b", "Task B")]
    held = [row("c", "Task C")]
    for record in tasks + held:
        record["ground_truth"].update(question="Which city?", facts="")
    train_path, test_path = tmp_path / "tasks.jsonl", tmp_path / "test.jsonl"
    train_path.write_text("".join(json.dumps(r) + "\n" for r in tasks))
    test_path.write_text("".join(json.dumps(r) + "\n" for r in held))
    requests = []
    def chat(client, body):
        requests.append(json.loads(json.dumps(body)))
        good = len(requests) == 2
        return {"id": str(len(requests)), "choices": [{"finish_reason": "stop", "message": {
            "content": "Tokyo" if good else "egg", "reasoning_content": "The successful trace." if good else "The failed trace."}}]}
    monkeypatch.setattr(prep, "chat", chat)
    monkeypatch.setattr(prep, "llm_judge", lambda *args: OK)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["prepare_sft.py", "--tasks", str(train_path), "--held-out", str(test_path),
        "--teacher", "offline", "--out", str(out), "--limit", "1", "--attempts", "2", "--validation-groups", "1"])
    prep.main()
    assert requests[0]["messages"] == requests[1]["messages"]
    assert requests[0]["reasoning_effort"] == "max"
    example = json.loads((out / "sft.jsonl").read_text())
    comparison = json.loads((out / "sft_answer_only.jsonl").read_text())
    assert example["messages"][-1] == {"role": "assistant", "content": "Tokyo", "reasoning_content": "The successful trace."}
    assert comparison["messages"][-1] == {"role": "assistant", "content": "Tokyo"}
    audit = [json.loads(line) for line in (out / "attempts.jsonl").read_text().splitlines()]
    assert [r["accepted"] for r in audit] == [False, True]
