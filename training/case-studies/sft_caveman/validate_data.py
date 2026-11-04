"""Validate candidate data integrity; this does not certify semantic quality."""
from pathlib import Path
import ast,hashlib,json,re

def validate(root: Path):
    manifest=json.loads((root/'manifest.json').read_text())
    for filename,digest in manifest['files'].items():
        assert hashlib.sha256((root/filename).read_bytes()).hexdigest()==digest, filename
    rows=[json.loads(line) for line in (root/'data/candidate_rows.jsonl').read_text().splitlines()]
    assert len({r['id'] for r in rows})==len(rows)
    assert len({r['student_messages'][-1]['content'] for r in rows})==len(rows)
    groups={}
    for row in rows:
        prior=groups.setdefault(row['group_id'],row['split'])
        assert prior==row['split'],row['id']
        assert row['review_status']=='awaiting_human_and_teacher_review'
        assert len(row['student_messages'])==2
        assert [m['role'] for m in row['student_messages']]==['system','user']
        assert all(text in row['draft_target'] for text in row['required_literals']),row['id']
        assert all(fact in row['student_messages'][1]['content'] for fact in row['facts'])
        assert 'caveman' not in ' '.join(m['content'] for m in row['student_messages']).lower()
        for code in re.findall(r'```python\n(.*?)\n```',row['draft_target'],re.S): ast.parse(code)
        if row['id']=='reg-01':
            assert json.loads(row['draft_target'])=={'status':'ok','count':3}
        if row['id']=='reg-02':
            assert re.findall(r'^(\d)\.',row['draft_target'],re.M)==['1','2','3']
    counts={split:sum(r['split']==split for r in rows) for split in manifest['split_counts']}
    assert counts==manifest['split_counts']
    probes=[line for line in (root/'sources/en.txt').read_text().splitlines() if line.strip()]
    assert len(probes)==10
    assert not {r['question'].strip().lower() for r in rows}&{q.strip().lower() for q in probes}
    return {'rows':len(rows),'split_counts':counts,'supplementary_eval_prompts':len(probes),'hashes':'pass','structure_and_literal_checks':'pass','semantic_quality':'requires human review; literal matches do not establish correctness','teacher_targets':'not generated','training':'not started'}

if __name__=='__main__':
    print(json.dumps(validate(Path(__file__).resolve().parent),indent=2))
