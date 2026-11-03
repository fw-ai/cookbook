"""Export explicitly reviewed GLM teacher responses into student-only chat JSONL."""
from pathlib import Path
import argparse,hashlib,json
from validate_data import validate

def export(root,teacher_dir,output,input_data=None):
    validate(root)
    source=input_data or root/'data/candidate_rows.jsonl'
    candidates=[json.loads(line) for line in source.read_text().splitlines()]
    assert len({r['id'] for r in candidates})==len(candidates), 'Duplicate candidate ids'
    if input_data is not None:
        from run_probe import jobs_for
        jobs_for(root,'teacher',input_data)  # train-only and evaluation separation checks
        plan=json.loads((teacher_dir/'plan.json').read_text())
        assert plan['input_data_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    rows={row['id']:row for row in candidates}
    review=json.loads((teacher_dir/'review.json').read_text())
    responses={r['row_id']:r for r in (json.loads(line) for line in (teacher_dir/'responses.jsonl').read_text().splitlines())}
    approved=[];ids=[]
    for item in review['decisions']:
        if item['decision']!='accept_for_smoke':continue
        row=rows[item['row_id']];response=responses[item['row_id']]
        assert row['split']=='train', 'Evaluation leakage'
        assert response['split']=='train' and response['arm']=='skill'
        assert response['valid_completion'] and response['finish_reason']=='stop'
        assert response['request_sha256']==item['request_sha256']
        assert hashlib.sha256(response['content'].encode()).hexdigest()==item['content_sha256']
        approved.append({'messages':row['student_messages']+[{'role':'assistant','content':response['content']}]})
        ids.append(row['id'])
    assert len(ids)==len(set(ids)) and ids
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in approved))
    provenance={'status':'assistant_reviewed_for_smoke_only; human_review_pending','source_ids':ids,'teacher_model':'accounts/fireworks/models/glm-5p3-flash','reviewer':review['reviewer'],'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'source_responses_sha256':hashlib.sha256((teacher_dir/'responses.jsonl').read_bytes()).hexdigest(),'review_sha256':hashlib.sha256((teacher_dir/'review.json').read_bytes()).hexdigest()}
    output.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input-data',type=Path)
    args=parser.parse_args();print(json.dumps(export(Path(__file__).resolve().parent,args.teacher_dir,args.output,args.input_data),indent=2))
