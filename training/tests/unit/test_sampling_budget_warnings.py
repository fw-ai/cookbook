"""Sampling targets must surface slow work without changing trial semantics."""
from copy import deepcopy

from training.examples.rl.harbor.recipes.terminal_bench.monitor_e2b_progress import (
    sampling_budget_warnings,
)


def test_thresholds_and_input_preserved():
    record = {'phase': 'agent_or_setup', 'trial_age_s': 1800,
              'remote': {'running_tools': [{'tool': 'bash', 'elapsed_s': 600}]}}
    original = deepcopy(record)
    warnings = sampling_budget_warnings(record)
    assert [w['code'] for w in warnings] == ['sampling_target_exceeded', 'long_tool_call']
    assert record == original


def test_below_threshold_and_missing_timing():
    assert sampling_budget_warnings({}) == []
    assert sampling_budget_warnings({
        'phase': 'agent_or_setup', 'trial_age_s': 1799,
        'remote': {'running_tools': [{'tool': 'bash', 'elapsed_s': 599}, {}]},
    }) == []


def test_finalized_trial_is_not_flagged():
    assert sampling_budget_warnings({
        'observation': 'already_finalized', 'trial_age_s': 7200,
    }) == []


def test_verification_does_not_report_stale_agent_tools():
    warnings = sampling_budget_warnings({
        'phase': 'verification_or_finalization', 'trial_age_s': 7200,
        'remote': {'running_tools': [{'tool': 'bash', 'elapsed_s': 7000}]},
    })
    assert [w['code'] for w in warnings] == ['sampling_target_exceeded']
