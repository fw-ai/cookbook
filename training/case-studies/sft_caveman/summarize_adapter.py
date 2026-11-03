"""Join adapter-only responses to earlier SDK baselines without mixing API tables."""
import argparse,json,statistics
from pathlib import Path

def summarize(baseline,adapter):
    bases=[json.loads(l) for l in (baseline/'responses.jsonl').read_text().splitlines()]
    adapted=[json.loads(l) for l in (adapter/'responses.jsonl').read_text().splitlines()]
    assert len(bases)==36 and len(adapted)==12
    assert len({r['row_id'] for r in adapted})==12
    by_id={r['row_id']:r for r in bases if r['arm']=='base'}
    for row in adapted:
        assert row['arm']=='adapter' and row['messages_sha256']==by_id[row['row_id']]['messages_sha256']
        assert row['parser_valid'] and row['stop_reason']=='stop'
    groups={}
    for split in ['dev','regression']:
        groups[split]={}
        for arm in ['base','concise','skill','adapter']:
            rs=[r for r in bases+adapted if r['split']==split and r['arm']==arm]
            assert len(rs)==6
            groups[split][arm]={'count':6,'mean_visible_tokens':round(statistics.mean(r['visible_tokens'] for r in rs),2),'total_visible_tokens':sum(r['visible_tokens'] for r in rs)}
    result={'status':'adapter_evaluation_completed','restored_step':16,'optimizer_updates_in_retry':0,'adapter_responses':12,
            'all_neutral_prompt_hashes_match_base':True,'groups':groups,
            'adapter_sampling_wall_seconds':round(sum(r['elapsed_seconds'] for r in adapted),3),
            'max_adapter_request_seconds':max(r['elapsed_seconds'] for r in adapted),
            'comparison_scope':'Same SDK path, renderer, prompts and sampling settings; baselines and restored adapter used different sessions. Development probes, not an untouched test.',
            'baseline_table_difference':'Public Chat API and SDK sampling used different inference paths. The cause of the numerical differences has not been established; the tables are kept separate.',
            'human_target_review':'pending','public_adapter_serving_verified':False}
    (adapter/'comparison_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True)
    a=p.parse_args();print(json.dumps(summarize(a.baseline,a.adapter),indent=2))
