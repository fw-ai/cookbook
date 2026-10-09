"""The SAO example accepts either a validated critic shape or an existing job."""

import sys

import pytest

from training.examples.rl.SAO import qwen3_4b_sao as example


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    monkeypatch.setattr(example, "load_dotenv", lambda: None)
    monkeypatch.delenv("FIREWORKS_CRITIC_TRAINING_SHAPE", raising=False)


@pytest.mark.parametrize(
    "argv,shape,job",
    [
        (["--critic-training-shape", "critic-shape"], "critic-shape", None),
        (["--critic-job-id", "existing-critic"], None, "existing-critic"),
        (
            [
                "--critic-training-shape",
                "critic-shape",
                "--critic-job-id",
                "existing-critic",
            ],
            "critic-shape",
            "existing-critic",
        ),
    ],
)
def test_accepts_shape_or_reattach(monkeypatch, argv, shape, job):
    monkeypatch.setattr(sys, "argv", ["sao", *argv])
    args = example.parse_args()
    assert args.critic_training_shape == shape
    assert args.critic_job_id == job


def test_accepts_shape_from_environment(monkeypatch):
    monkeypatch.setenv("FIREWORKS_CRITIC_TRAINING_SHAPE", "critic-shape")
    monkeypatch.setattr(sys, "argv", ["sao"])
    assert example.parse_args().critic_training_shape == "critic-shape"


def test_requires_critic_topology_before_provisioning(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sao"])
    with pytest.raises(SystemExit) as exc:
        example.parse_args()
    assert exc.value.code == 2
    assert "provide --critic-training-shape or --critic-job-id" in capsys.readouterr().err
