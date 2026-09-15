from types import SimpleNamespace

import pytest

from training.examples.rl.harbor.recipes.terminal_bench import monitor_e2b_progress as monitor


@pytest.mark.parametrize("phase", ["list", "connect", "command"])
@pytest.mark.parametrize("finalized", [False, True])
def test_inspection_failure_rechecks_finalization(tmp_path, monkeypatch, phase, finalized):
    trial = {"trial": "sample", "phase": "agent_or_setup", "trial_age_s": 200}
    result_path = tmp_path / "trials/sample/result.json"

    def fail():
        if finalized:
            result_path.parent.mkdir(parents=True)
            # Finalized does not imply success; do not inspect credentials or
            # fabricate a score from this marker.
            result_path.write_text('{"exception_info": {"exception_type": "AgentTimeoutError"}}')
        raise TimeoutError("remote observation failed")

    def list_sandboxes(**kwargs):
        if phase == "list":
            fail()
        return SimpleNamespace(next_items=lambda: [SimpleNamespace(
            metadata={"session_id": "sample__env"}, sandbox_id="sandbox-1")])

    def connect(sandbox_id):
        assert sandbox_id == "sandbox-1"
        if phase == "connect":
            fail()
        return SimpleNamespace(commands=SimpleNamespace(run=lambda *args, **kwargs: fail()))

    monkeypatch.setattr(monitor, "Sandbox", SimpleNamespace(list=list_sandboxes, connect=connect))
    observed = monitor.inspect_trial(tmp_path, trial)
    if finalized:
        assert observed["observation"] == "already_finalized"
        assert "observation_error" not in observed
    else:
        assert observed["observation_error"] == "TimeoutError"
        assert "observation" not in observed


@pytest.mark.parametrize("finalized", [False, True])
def test_missing_sandbox_is_not_assumed_finished(tmp_path, monkeypatch, finalized):
    def list_sandboxes(**kwargs):
        if finalized:
            result_path = tmp_path / "trials/sample/result.json"
            result_path.parent.mkdir(parents=True)
            result_path.write_text("{}")
        return SimpleNamespace(next_items=lambda: [])

    monkeypatch.setattr(monitor, "Sandbox", SimpleNamespace(list=list_sandboxes))
    observed = monitor.inspect_trial(tmp_path, {"trial": "sample", "phase": "agent_or_setup"})
    if finalized:
        assert observed["observation"] == "already_finalized"
    else:
        assert observed["matching_sandboxes"] == 0
        assert "observation" not in observed
