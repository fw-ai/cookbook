import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from training.examples.rl.harbor.recipes.terminal_bench import monitor_e2b_progress as recorder


def test_pid_identity_and_remote_probe_syntax():
    assert recorder.identity(os.getpid()) is not None
    assert recorder.identity(2147483647) is None
    code = recorder.REMOTE.split("python3 - <<'REMOTE'\n", 1)[1].rsplit("\nREMOTE", 1)[0]
    compile(code, "<read-only remote probe>", "exec")


def test_finalized_trial_never_contacts_e2b(tmp_path, monkeypatch):
    trial = tmp_path / "trials" / "done"
    trial.mkdir(parents=True)
    (trial / "result.json").touch()
    api = Mock()
    monkeypatch.setattr(recorder, "Sandbox", api)
    result = recorder.inspect_trial(tmp_path, {"trial": "done", "phase": "agent_or_setup"})
    assert result["observation"] == "already_finalized"
    api.list.assert_not_called()
    api.connect.assert_not_called()


@pytest.mark.parametrize("matching_count", [0, 2])
def test_no_connection_without_unique_exact_session(tmp_path, monkeypatch, matching_count):
    other = SimpleNamespace(metadata={"session_id": "another-run__env"}, sandbox_id="other")
    matching = [SimpleNamespace(metadata={"session_id": "ours__env"}, sandbox_id=str(i))
                for i in range(matching_count)]
    api = Mock()
    api.list.return_value.next_items.return_value = [other, *matching]
    monkeypatch.setattr(recorder, "Sandbox", api)
    result = recorder.inspect_trial(tmp_path, {"trial": "ours", "phase": "agent_or_setup"})
    assert result["matching_sandboxes"] == matching_count
    api.connect.assert_not_called()


def test_unique_exact_session_uses_bounded_read_only_command(tmp_path, monkeypatch):
    api = Mock()
    api.list.return_value.next_items.return_value = [
        SimpleNamespace(metadata={"session_id": "other__env"}, sandbox_id="other"),
        SimpleNamespace(metadata={"session_id": "ours__env"}, sandbox_id="ours-id"),
    ]
    api.connect.return_value.commands.run.return_value.stdout = '{"activity": []}'
    monkeypatch.setattr(recorder, "Sandbox", api)
    result = recorder.inspect_trial(tmp_path, {"trial": "ours", "phase": "agent_or_setup"})
    api.connect.assert_called_once_with("ours-id")
    api.connect.return_value.commands.run.assert_called_once_with(recorder.REMOTE, timeout=15)
    assert result["remote"] == {"activity": []}


def test_observation_error_does_not_log_credentials_or_retry(tmp_path, monkeypatch):
    api = Mock()
    api.list.side_effect = RuntimeError("credential-bearing error content")
    monkeypatch.setattr(recorder, "Sandbox", api)
    result = recorder.inspect_trial(tmp_path, {"trial": "ours", "phase": "agent_or_setup"})
    assert result == {"trial": "ours", "phase": "agent_or_setup",
                      "observation_error": "RuntimeError"}
    assert api.list.call_count == 1
    api.connect.assert_not_called()
