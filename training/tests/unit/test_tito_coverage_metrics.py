from training.utils.rl.metrics import publish_tito_sidecar_metrics


def test_publishes_coverage_and_split_reasons() -> None:
    metrics: dict = {}
    publish_tito_sidecar_metrics(
        metrics,
        [
            {
                "tito/turn/completion_tokens_count": 4.0,
                "tito/turn/completion_tokens_sum": 4.0,
                "tito/turn/completion_tokens_min": 4.0,
                "tito/turn/completion_tokens_max": 4.0,
                "tito/trajectory/policy_turns_count": 1.0,
                "tito/trajectory/policy_turns_sum": 1.0,
                "tito/trajectory/policy_turns_min": 1.0,
                "tito/trajectory/policy_turns_max": 1.0,
                "tito/lineage/new_segment": 2.0,
                "tito/lineage/realign": 3.0,
                "tito/lineage/boundary_reason_history_rewrite": 1.0,
                "tito/lineage/history_rewrite_assistant_roundtrip": 1.0,
                "tito/lineage/boundary_reason_unbounded_or_ambiguous_drift": 1.0,
                "tito/coverage/sampled_completion_tokens": 100.0,
                "tito/coverage/trained_tokens": 80.0,
                "tito/coverage/lost_realign_overwritten": 20.0,
                "tito/coverage/lost_abandoned": 0.0,
                "tito/coverage/lost_fail_closed_masked": 4.0,
                "tito/coverage/lost_invisible": 1.0,
            }
        ],
    )

    assert metrics["tito/lineage/splits"] == 1.0
    assert metrics["tito/lineage/new_segment/history_rewrite"] == 1.0
    assert metrics["tito/lineage/new_segment/token_drift"] == 1.0
    assert metrics["tito/lineage/history_rewrite/assistant_roundtrip"] == 1.0
    assert metrics["tito/coverage/trained_fraction"] == 0.8
    assert metrics["tito/coverage/lost/realign_overwritten"] == 20.0
    assert metrics["tito/coverage/lost/abandoned"] == 0.0
    assert metrics["tito/coverage/lost/fail_closed_masked"] == 4.0
    assert metrics["tito/coverage/lost/invisible"] == 1.0


def test_coverage_fraction_omitted_without_samples() -> None:
    metrics: dict = {}
    publish_tito_sidecar_metrics(metrics, [{"tito/lineage/realign": 0.0}])
    assert "tito/coverage/trained_fraction" not in metrics
