"""Offline tests for the v2 public pilot content and browser contracts."""
import hashlib
import json
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright
from brief_checks import check_assertions, check_sections, contract_prompt, run_interaction
from generate import task_prompt

HERE=Path(__file__).resolve().parent


def test_public_pack_and_prompts():
    briefs=[json.loads(p.read_text()) for p in (HERE/'briefs/pilot').glob('*.json')]
    assert len(briefs)==24
    assert sum(bool(b['assets']) for b in briefs)==8
    sources={x['file']:x for x in json.loads((HERE/'assets/pilot/sources.json').read_text())}
    for b in briefs:
        assert b['schema_version']==2 and b['initial_required_content']
        assert b['data_provenance']['license']=='CC0-1.0'
        prompt=task_prompt(b)
        assert 'data-testid' in prompt
        for section in b['content_sections']:
            for row in section['rows']:
                assert set(row)==set(section['fields'])
                for value in row.values(): assert str(value) in prompt
        for asset in b['assets']:
            src=sources[asset['file']]
            if src.get('kind') == 'ai-generated':
                assert src['prompt'] and src['tool'] == 'built-in image_gen'
                assert src['reference_images'] == []
            else:
                assert src['page'].startswith('https://commons.wikimedia.org/')
            assert src['license'] and src['artist']
            assert hashlib.sha256((HERE/'assets/pilot'/asset['file']).read_bytes()).hexdigest()==src['sha256']
    plants=next(b for b in briefs if b['id']=='plant-shop')
    assert len(plants['content_sections'][0]['rows'])==6
    festival=next(b for b in briefs if b['id']=='music-festival')
    assert len(festival['content_sections'][0]['rows'])==28


@pytest.fixture
def page():
    with sync_playwright() as p:
        browser=p.chromium.launch()
        page=browser.new_page()
        yield page
        browser.close()


def test_panel_checks_ignore_duplicate_text_and_support_markup(page,tmp_path):
    page.set_content('''<p>6am to 11pm</p><button data-testid="toggle" onclick="document.querySelector('#panel').hidden=!document.querySelector('#panel').hidden">Hours</button><div id="panel" data-testid="panel" hidden>6am <span>to</span> 11pm</div>''')
    spec={'initial':[{'selector':'#panel','visible':False}],'steps':[{'action':'click','selector':'button','assertions':[{'selector':'#panel','contains':['6am to 11pm']}]},{'action':'click','selector':'button','assertions':[{'selector':'#panel','visible':False}]}]}
    assert run_interaction(page,spec,tmp_path,'390',0,[])['ok']
    page.locator('button').evaluate('(e)=>e.onclick=null')
    assert not run_interaction(page,spec,tmp_path,'390',1,[])['ok']


def test_fixed_rows_require_fields_on_same_record(page):
    sections=[{'id':'plants','rows':[{'name':'Pothos','price':'$18'},{'name':'Monstera','price':'$34'}]}]
    page.set_content('<section data-testid="content-plants"><div data-row>Pothos <span>$18</span></div><div data-row>Monstera $34</div></section>')
    assert check_sections(page,sections)[0]['ok']
    page.set_content('<section data-testid="content-plants"><div data-row>Pothos $34</div><div data-row>Monstera $18</div></section>')
    assert not check_sections(page,sections)[0]['ok']
    page.set_content('<section data-testid="content-plants"><div data-row>Pothos $18</div></section>')
    assert not check_sections(page,sections)[0]['ok']


def test_selection_must_change_visible_results(page,tmp_path):
    page.set_content('''<select onchange="document.querySelector('#price').innerText=this.value==='sub'?'$12':'$14'"><option value="once">Once</option><option value="sub">Subscription</option></select><div id="price">$14</div>''')
    spec={'initial':[{'selector':'#price','contains':['$14']}],'steps':[{'action':'select','selector':'select','value':'sub','assertions':[{'selector':'#price','contains':['$12'],'absent':['$14']}]},{'action':'select','selector':'select','value':'once','assertions':[{'selector':'#price','contains':['$14'],'absent':['$12']}]}]}
    assert run_interaction(page,spec,tmp_path,'1440',0,[])['ok']
    page.locator('select').evaluate('(e)=>e.onchange=null')
    assert not run_interaction(page,spec,tmp_path,'1440',1,[])['ok']


def test_v2_brief_runs_end_to_end(tmp_path):
    import subprocess
    import sys
    b=json.loads((HERE/'briefs/pilot/plant-shop.json').read_text())
    rows=b['content_sections'][0]['rows']
    # A plain fixture tests contract wiring, not aesthetics or model generation.
    app='''import {useState} from 'react';
const rows=ROWS;
export default function App(){const [filter,setFilter]=useState('all');return <main>
<p>FACTS</p><img src="/assets/plant-shop.jpg" width="120" alt="Reference leaf"/>
<label>Light<select data-testid="light-filter" value={filter} onChange={e=>setFilter(e.target.value)}><option value="all">All</option><option value="low">Low</option><option value="bright">Bright</option></select></label>
<section data-testid="content-plants">{rows.filter(r=>filter==='all'||r.light===(filter==='low'?'Low light':'Bright indirect')).map(r=><article data-row key={r.plant}><b>{r.plant}</b> <span>{r.price}</span> <span>{r.light}</span></article>)}</section></main>}
'''.replace('ROWS',json.dumps(rows)).replace('FACTS',' | '.join(b['initial_required_content'])).replace('plant-shop.jpg',b['assets'][0]['file'])
    (tmp_path/'App.jsx').write_text(app)
    (tmp_path/'style.css').write_text('body{font-family:sans-serif;margin:16px}article{padding:12px}img{max-width:100%}')
    out=tmp_path/'render'
    completed=subprocess.run([sys.executable,'-B',str(HERE/'render.py'),'--app',str(tmp_path/'App.jsx'),'--css',str(tmp_path/'style.css'),'--brief',str(HERE/'briefs/pilot/plant-shop.json'),'--out',str(out)],capture_output=True,text=True,timeout=90)
    assert completed.returncode==0,completed.stderr
    rec=json.loads((out/'record.json').read_text())
    assert rec['checks_complete'] and rec['passes_functional'],rec
    assert rec['summary']['content_section_failures']==0
    assert rec['summary']['interactions_ok']==2
    assert (out/'dist/assets'/b['assets'][0]['file']).is_file()
