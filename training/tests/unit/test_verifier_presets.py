"""JSON-input verifier schema and sequential server wiring tests."""

from __future__ import annotations

import json

import pytest

from training.renderer.verifier import serve
from training.renderer.verifier.utils.presets import (
    load_preset_catalog,
    resolve_preset_case,
)


def _catalog() -> dict:
    return {
        "schema_version": 1,
        "profiles": [
            {
                "id": "qwen-vl",
                "label": "Qwen VL",
                "defaults": {
                    "renderer": "qwen3_vl",
                    "tokenizer_model": "/tmp/tokenizer",
                    "image_processor_model": "/tmp/processor",
                    "deployment_id": (
                        "accounts/test-account/deployments/test-deployment"
                    ),
                    "max_tokens": 64,
                    "temperature": 0,
                },
                "examples": [
                    {
                        "id": "vision",
                        "label": "Vision",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:image/png;base64,AAAA"
                                        },
                                    },
                                    {"type": "text", "text": "Describe it."},
                                ],
                            }
                        ],
                        "extra_completion_kwargs": {"thinking": {"type": "disabled"}},
                    }
                ],
            }
        ],
    }


def test_load_and_resolve_complete_vision_input(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(_catalog()))

    loaded = load_preset_catalog(path)
    profile, example, request = resolve_preset_case(
        loaded,
        profile_id="qwen-vl",
        example_id="vision",
    )

    assert profile["label"] == "Qwen VL"
    assert example["label"] == "Vision"
    assert request["renderer"] == "qwen3_vl"
    assert request["max_tokens"] == 64
    assert request["messages"][0]["content"][0]["type"] == "image_url"
    assert "id" not in request
    assert "label" not in request


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda data: data["profiles"][0]["defaults"].update({"unsupported": True}),
            "unsupported keys",
        ),
        (
            lambda data: data["profiles"][0]["examples"][0][
                "extra_completion_kwargs"
            ].update({"headers": {"Authorization": "no"}}),
            "credential keys",
        ),
        (
            lambda data: data["profiles"][0]["examples"][0][
                "extra_completion_kwargs"
            ].update({"model": "other-model"}),
            "verifier-controlled keys",
        ),
        (
            lambda data: data["profiles"][0]["defaults"].update(
                {"renderer_config": {"api_key": "no"}}
            ),
            "unsupported keys",
        ),
        (
            lambda data: data["profiles"][0]["defaults"].update(
                {"model": "accounts/test/models/model"}
            ),
            "cannot set both",
        ),
        (
            lambda data: data["profiles"][0]["defaults"].update(
                {"train_on_what": "not-a-mode"}
            ),
            "supported training mode",
        ),
        (
            lambda data: data["profiles"][0]["examples"][0].update(
                {"artifact": {"kind": "probe"}}
            ),
            "unsupported keys",
        ),
    ],
)
def test_input_catalog_rejects_unsafe_ambiguous_or_output_values(
    tmp_path,
    mutate,
    message,
):
    data = _catalog()
    mutate(data)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match=message):
        load_preset_catalog(path)


def test_resolve_case_rejects_unknown_ids():
    catalog = _catalog()

    with pytest.raises(ValueError, match="unknown input profile"):
        resolve_preset_case(catalog, profile_id="missing", example_id="vision")
    with pytest.raises(ValueError, match="unknown input example"):
        resolve_preset_case(catalog, profile_id="qwen-vl", example_id="missing")


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/v1",
        "http://127.0.0.1:8000/v1",
        "http://[::1]:8000/v1",
    ],
)
def test_loopback_urls_can_use_local_no_auth_client(url):
    assert serve._is_loopback_url(url) is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://localhost:8000/v1", "http://localhost:8000"),
        ("http://[::1]:8000/v1/", "http://[::1]:8000"),
        ("http://localhost:8000/inference", "http://localhost:8000/inference"),
        (
            "https://api.fireworks.ai/inference/v1",
            "https://api.fireworks.ai/inference/v1",
        ),
    ],
)
def test_sdk_base_url_normalizes_only_loopback_v1(value, expected):
    assert serve._sdk_base_url(value) == expected


def test_run_one_probe_wires_json_vision_fields(monkeypatch):
    tokenizer = object()
    image_processor = object()
    client = object()
    captured = {}

    monkeypatch.setattr(serve, "_tokenizer", lambda value: tokenizer)
    monkeypatch.setattr(serve, "_image_processor", lambda value: image_processor)
    monkeypatch.setattr(serve, "_client", lambda api_key, base_url: client)

    def fake_run_probe(**kwargs):
        captured.update(kwargs)
        return {"kind": "probe"}

    monkeypatch.setattr(serve, "run_probe", fake_run_probe)
    _, _, request = resolve_preset_case(
        _catalog(),
        profile_id="qwen-vl",
        example_id="vision",
    )

    assert serve._run_one_probe(request) == {"kind": "probe"}
    assert captured["tokenizer"] is tokenizer
    assert captured["image_processor"] is image_processor
    assert captured["image_processor_model"] == "/tmp/processor"
    assert captured["extra_completion_kwargs"] == {"thinking": {"type": "disabled"}}


def test_run_input_case_reloads_json_and_returns_fresh_artifact(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(_catalog()))
    monkeypatch.setattr(serve, "_INPUT_FILE", path)
    calls = []

    def fake_run(request):
        calls.append(request)
        return {"kind": "probe", "sequence": len(calls)}

    monkeypatch.setattr(serve, "_run_one_probe", fake_run)

    first = serve._run_input_case("qwen-vl", "vision")
    data = _catalog()
    data["profiles"][0]["examples"][0]["messages"][0]["content"][1]["text"] = (
        "Describe the updated image."
    )
    path.write_text(json.dumps(data))
    second = serve._run_input_case("qwen-vl", "vision")

    assert first["artifact"]["sequence"] == 1
    assert second["artifact"]["sequence"] == 2
    assert calls[1]["messages"][0]["content"][1]["text"] == (
        "Describe the updated image."
    )


def test_viewer_has_only_json_input_live_execution_path():
    html = serve.INDEX_PATH.read_text()

    assert 'fetch("/input", { cache: "no-store" })' in html
    assert 'fetch("/run-case"' in html
    assert 'fetch("/session")' not in html
    assert 'fetch("/presets")' not in html
    assert 'fetch("/probe"' not in html
    assert "FilePicker" not in html
    assert "load an existing probe JSON artifact" not in html
