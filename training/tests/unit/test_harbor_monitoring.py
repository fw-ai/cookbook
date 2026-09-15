import os

from training.examples.rl.harbor.tito.monitoring import pending_trial_inventory


def test_pending_inventory_distinguishes_agent_and_verifier_wait(tmp_path):
    for name in ("agent", "verifier", "finished", "e2b-template-build"):
        trial = tmp_path / name
        trial.mkdir()
        config = trial / "config.json"
        # The monitor must not parse credential-bearing config contents.
        config.write_text("not parsed")
        os.utime(config, (100, 100))
    marker = tmp_path / "verifier/artifacts/tito/compact/COMPLETE"
    marker.parent.mkdir(parents=True)
    marker.touch()
    os.utime(marker, (200, 200))
    (tmp_path / "finished/result.json").write_text("{}")
    (tmp_path / "creating").mkdir()

    rows = pending_trial_inventory(tmp_path, now=500)
    assert rows == [
        {"trial": "agent", "phase": "agent_or_setup",
         "trial_age_s": 400, "phase_age_s": 400},
        {"trial": "verifier", "phase": "verification_or_finalization",
         "trial_age_s": 400, "phase_age_s": 300},
    ]
    assert pending_trial_inventory(tmp_path / "absent", now=500) == []


def test_pending_inventory_clamps_future_artifact_times(tmp_path):
    trial = tmp_path / "new"
    trial.mkdir()
    config = trial / "config.json"
    config.touch()
    os.utime(config, (200, 200))
    row, = pending_trial_inventory(tmp_path, now=100)
    assert row["trial_age_s"] == row["phase_age_s"] == 0
