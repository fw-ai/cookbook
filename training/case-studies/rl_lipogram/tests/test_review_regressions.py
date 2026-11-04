"""Offline regressions for auth, curriculum resume, scoring and duplicate work."""
from __future__ import annotations
import ast
from concurrent.futures import ThreadPoolExecutor
import contextlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
import httpx
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'dpo_antislop'))
import lipogram_rewards as rewards
from reward_state import initial_phase, save_phase_state, completed_score


def test_missing_key_aborts_without_request(monkeypatch):
    monkeypatch.delenv('FIREWORKS_API_KEY', raising=False)
    with httpx.Client(transport=httpx.MockTransport(lambda request: pytest.fail('No request expected'))) as client:
        with pytest.raises(rewards.JudgeConfigurationError):
            rewards.llm_judge(client, 'question', 'Tokyo')


@pytest.mark.parametrize('status', [400, 401, 403, 404])
def test_permanent_judge_error_aborts_without_retry(monkeypatch, status):
    monkeypatch.setenv('FIREWORKS_API_KEY', 'test-not-a-real-key')
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(status, json={'error': 'rejected'})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(rewards.JudgeConfigurationError):
            rewards.llm_judge(client, 'question', 'Tokyo')
    assert len(requests) == 1


def test_single_judge_verdict_and_transient_failure(monkeypatch):
    monkeypatch.setenv('FIREWORKS_API_KEY', 'test-not-a-real-key')
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={'choices':[{'message':{'content':'{"answers":0.9,"faithful":1}'}}]})
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert rewards.llm_judge(client, 'q', 'Tokyo') == {'answers':0.9,'faithful':1.0}
    assert len(calls) == 1
    monkeypatch.setattr(rewards.time, 'sleep', lambda _: None)
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503))) as client:
        assert rewards.llm_judge(client, 'q', 'Tokyo') is None


def test_curriculum_state_is_checkpoint_bound(tmp_path):
    state = tmp_path / 'phase.json'
    ref = 'account/run/as-state-0020'
    save_phase_state(state, ref, 'strict')
    assert initial_phase('auto', ref, str(state)) == 'strict'
    assert initial_phase('auto') == 'ramp'
    with pytest.raises(ValueError): initial_phase('auto', ref)
    with pytest.raises(ValueError): initial_phase('auto', 'account/run/as-state-0010', str(state))
    assert initial_phase('strict', ref) == 'strict'


def test_truncation_does_not_count_as_completed_success():
    score = (0.9, True, True, 0.0, 42)
    assert completed_score(score, True) == (0.0, False, False, 0.0, 42)
    assert completed_score(score, False) == score
    assert not rewards.e_free('```text\negg\n```')


def loop_method(path, name, namespace):
    # Execute the actual loop method with fake clients; no training SDK imports or services.
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ServerlessAntislopRL')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('folder,id_key', [('rl_lipogram','source_id'), ('dpo_antislop','brief_id')])
def test_duplicate_rollouts_share_one_judge_request(folder, id_key):
    path = HERE.parent / folder / ('lipogram_rl.py' if folder=='rl_lipogram' else 'antislop_rl.py')
    calls = []
    runner = SimpleNamespace(phase='strict', cfg=SimpleNamespace(score_concurrency=2), _score_cache={},
        renderer=SimpleNamespace(parse_response=lambda tokens: (str(tokens),None)),
        _score_one=lambda content,gt: calls.append(content) or (0.9,True,True,0))
    score = loop_method(path, '_prescore', {'json':json,'get_text_content':lambda x:x,'ThreadPoolExecutor':ThreadPoolExecutor})
    row = {'ground_truth': json.dumps({id_key:'one'})}
    seq = SimpleNamespace(tokens=[1,2])
    score(runner, [SimpleNamespace(sequences=[seq,seq,seq])], [row])
    score(runner, [SimpleNamespace(sequences=[seq])], [row])
    assert len(calls) == 1


def test_dcp_save_writes_matching_sidecar(tmp_path):
    calls = []
    runner = SimpleNamespace(cfg=SimpleNamespace(training_checkpoint_name='as-state',dcp_timeout_s=1),
        phase='strict',run_dir=tmp_path,saved_training_checkpoints=[],
        training_client=SimpleNamespace(save_state=lambda name: SimpleNamespace(result=lambda **kw:calls.append(name))),
        _resume_reference=lambda name:'acct/run/'+name)
    ns={'Path':Path,'save_phase_state':save_phase_state,'time':__import__('time'),'phase_span':lambda *a,**kw:contextlib.nullcontext()}
    path=HERE/'lipogram_rl.py'
    write=loop_method(path,'_save_reward_state',ns)
    runner._save_reward_state=lambda name:write(runner,name)
    save=loop_method(path,'_save_dcp',ns)
    save(runner,20)
    assert calls == ['as-state-0020']
    assert initial_phase('auto','acct/run/as-state-0020',str(tmp_path/'as-state-0020.reward-state.json'))=='strict'


def test_antislop_permanent_scorer_errors_escape(monkeypatch):
    import build_antislop_pack as anti
    monkeypatch.setenv('FIREWORKS_API_KEY','test-not-a-real-key')
    with httpx.Client(transport=httpx.MockTransport(lambda _:httpx.Response(401))) as client:
        with pytest.raises(anti.ScorerConfigurationError): anti.llm_judge(client,'answer','tweet','topic','article')
        with pytest.raises(anti.ScorerConfigurationError): anti.gptzero_human(client,'answer','test-key')
