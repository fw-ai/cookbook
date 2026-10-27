"""Shell outcomes survive device cleanup, without touching the host's /dev/null."""

import importlib.util
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def driver(monkeypatch):
    local = ModuleType("mimoagent.environments.local")
    local.LocalEnvironment = type("LocalEnvironment", (), {"execute": Mock()})
    model = ModuleType("mimoagent.models.openai_chat")
    model.OpenAIChatModel = object
    monkeypatch.setitem(sys.modules, local.__name__, local)
    monkeypatch.setitem(sys.modules, model.__name__, model)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    path = (
        Path(__file__).parents[2] / "examples/harbor_rl/harnesses/mimoagent/driver.py"
    )
    spec = importlib.util.spec_from_file_location("mimo_null_device_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Replace only this module's OS facade; never mock global filesystem calls.
    module.os = SimpleNamespace(
        devnull="/test-only/null", lstat=Mock(), mknod=Mock(), chmod=Mock()
    )
    local.LocalEnvironment.execute.reset_mock()
    return module


def test_failed_command_restores_missing_device_and_preserves_output(driver):
    original = SimpleNamespace(st_mode=stat.S_IFCHR | 0o666, st_rdev=259)
    driver.os.lstat.side_effect = [original, FileNotFoundError()]
    result = {"output": "syntax error", "returncode": 2}
    driver._LocalEnvironment.execute.return_value = result
    assert (
        driver.LocalEnvironment().execute("compile", cwd="/work", timeout=7) is result
    )
    driver._LocalEnvironment.execute.assert_called_once_with(
        "compile", cwd="/work", timeout=7
    )
    driver.os.mknod.assert_called_once_with("/test-only/null", original.st_mode, 259)
    driver.os.chmod.assert_called_once_with("/test-only/null", 0o666)


def test_timeout_still_restores_the_device_and_propagates(driver):
    driver.os.lstat.side_effect = [
        SimpleNamespace(st_mode=stat.S_IFCHR | 0o666, st_rdev=259),
        FileNotFoundError(),
    ]
    error = subprocess.TimeoutExpired("compile", 1)
    driver._LocalEnvironment.execute.side_effect = error
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        driver.LocalEnvironment().execute("compile")
    assert raised.value is error
    driver.os.mknod.assert_called_once()


@pytest.mark.parametrize(
    "initial", [None, SimpleNamespace(st_mode=stat.S_IFREG | 0o600)]
)
def test_does_not_create_a_device_when_it_was_not_one_before(driver, initial):
    driver.os.lstat.side_effect = FileNotFoundError() if initial is None else [initial]
    driver.LocalEnvironment().execute("command")
    driver.os.mknod.assert_not_called()


def test_existing_or_replaced_path_is_never_overwritten(driver):
    driver.os.lstat.side_effect = [
        SimpleNamespace(st_mode=stat.S_IFCHR | 0o666),
        SimpleNamespace(st_mode=stat.S_IFREG),
    ]
    driver.LocalEnvironment().execute("command")
    driver.os.mknod.assert_not_called()
    driver.os.chmod.assert_not_called()


def test_concurrent_restoration_does_not_change_the_command_result(driver):
    driver.os.lstat.side_effect = [
        SimpleNamespace(st_mode=stat.S_IFCHR | 0o666, st_rdev=259),
        FileNotFoundError(),
    ]
    driver.os.mknod.side_effect = FileExistsError()
    result = {"output": "done", "returncode": 0}
    driver._LocalEnvironment.execute.return_value = result
    assert driver.LocalEnvironment().execute("command") is result
    driver.os.chmod.assert_not_called()


@pytest.mark.parametrize("failure_at", ["lstat", "mknod", "chmod"])
@pytest.mark.parametrize("timeout", [False, True])
def test_repair_failure_is_reported_without_hiding_shell_outcome(
    driver, caplog, failure_at, timeout
):
    driver.os.lstat.side_effect = [
        SimpleNamespace(st_mode=stat.S_IFCHR | 0o666, st_rdev=259),
        (
            PermissionError("repair-secret")
            if failure_at == "lstat"
            else FileNotFoundError()
        ),
    ]
    if failure_at != "lstat":
        getattr(driver.os, failure_at).side_effect = PermissionError("repair-secret")
    result = {"output": "compile error", "returncode": 2}
    error = subprocess.TimeoutExpired("compile --token timeout-secret", 1)
    driver._LocalEnvironment.execute.return_value = result
    driver._LocalEnvironment.execute.side_effect = error if timeout else None
    if timeout:
        with pytest.raises(subprocess.TimeoutExpired) as raised:
            driver.LocalEnvironment().execute("compile")
        assert raised.value is error
    else:
        assert driver.LocalEnvironment().execute("compile") is result
    assert "Failed to restore the container null device" in caplog.text
    assert "PermissionError" in caplog.text
    assert "repair-secret" not in caplog.text
    assert "timeout-secret" not in caplog.text
