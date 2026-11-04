"""Version-two brief contracts: visible state assertions and fixed content rows.

Selectors are neutral data-testid hooks, not layout requirements. The judge still
assesses design independently; these checks establish task completeness.
"""
import json
import re


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()


def required_present(item, page_text, page=None):
    """A required item is present when its exact phrase appears, OR when every one of its tokens appears inside one
    visible element's text (a menu row "Shio - clear broth - $16", a product card "Pothos / low light / $18").
    Designed layouts separate name and price; the fact is still there. Returns (present, how)."""
    n = normalized(page_text)
    if normalized(item) in n:
        return True, "exact"
    toks = [t for t in normalized(item).split(' ') if t]
    tol_ok = False
    if page is not None and len(toks) >= 2:
        # tier 3 (tolerant): within one visible element, every numeric token and at least 80% of the content tokens.
        # Authored briefs carry long items ("Tue and Thu pickup 2pm-6pm"); a page that writes "Tuesday and Thursday
        # pickup, 2pm-6pm" has the fact. Numbers never bend: a changed price or time is a failure.
        stripped = [t.strip(".,;:()") for t in toks]
        nums = [t for t in stripped if any(ch.isdigit() for ch in t)]            # every numeric token, however short
        words = [t for t in stripped if t not in nums and len(t) > 2 and t.lower() not in ("and", "the", "for", "with", "from", "per")]
        tol = page.evaluate("""([nums, words, minWords]) => {
            const norm = s => (s||'').replace(/\\s+/g,' ').trim().toLowerCase();
            const vis = e => { const cs=getComputedStyle(e); if (cs.display==='none'||cs.visibility==='hidden'||+cs.opacity===0) return false;
                               const r=e.getBoundingClientRect(); return r.width>0 && r.height>0; };
            const els=[...document.querySelectorAll('li,tr,article,section,div,p,dl,dt,dd,figure,header,footer')];
            const hits=els.filter(e => { if (!vis(e)) return false; const t=norm(e.textContent); if (t.length>600) return false;
                const has = n => { const i = t.indexOf(n); if (i < 0) return false;
                    for (let j = i; j >= 0; j = t.indexOf(n, j + 1)) { const a = t[j - 1], b = t[j + n.length];
                        if (!(b && /[0-9]/.test(b)) && !(a && /[0-9]/.test(a))) return true; } return false; };
                return nums.every(has) && words.filter(w => t.includes(w)).length >= minWords; });
            return hits.length; }""", [nums, words, max(1, int(round(0.8 * len(words)))) if words else 0])
        if tol:
            tol_ok = True
        else:
            tol_ok = False
        hit = page.evaluate("""(toks) => {
            const norm = s => (s||'').replace(/\\s+/g,' ').trim().toLowerCase();
            const vis = e => { const cs=getComputedStyle(e); if (cs.display==='none'||cs.visibility==='hidden'||+cs.opacity===0) return false;
                               const r=e.getBoundingClientRect(); return r.width>0 && r.height>0; };
            const els=[...document.querySelectorAll('li,tr,article,section,div,p,dl,dt,dd,figure')];
            const has = (t, s) => { if (!/\\d/.test(t)) return s.includes(t);
                const i = s.indexOf(t); if (i < 0) return false;
                // numeric token: must not be followed by another digit (so "$3" never matches "$39") and not preceded by a digit
                for (let j = i; j >= 0; j = s.indexOf(t, j + 1)) { const a = s[j - 1], b = s[j + t.length];
                    if (!(b && /[0-9]/.test(b)) && !(a && /[0-9]/.test(a))) return true; if (j === s.length - 1) break; }
                return false; };
            const hits=els.filter(e => vis(e) && toks.every(t => has(t, norm(e.textContent))));
            // smallest such element, and it must be reasonably small (a row/card, not the whole page)
            const small=hits.filter(e => ![...e.children].some(c => hits.includes(c)) && norm(e.textContent).length <= 400);
            return small.length; }""", toks)
        if hit:
            return True, "row"
        if tol_ok:
            return True, "tolerant"
    return False, None


def tolerant_in(item, text):
    """Exact phrase, or every numeric token (boundary-matched) and >= 80% of content words, within `text`."""
    n = normalized(text); it = normalized(item)
    if it in n:
        return True
    toks = [t.strip(".,;:()") for t in it.split(' ') if t]
    nums = [t for t in toks if any(ch.isdigit() for ch in t)]
    words = [t for t in toks if t not in nums and len(t) > 2 and t not in ("and", "the", "for", "with", "from", "per")]
    def has_num(x):
        i = n.find(x)
        while i >= 0:
            a = n[i - 1] if i > 0 else ""; b = n[i + len(x)] if i + len(x) < len(n) else ""
            if not a.isdigit() and not b.isdigit():
                return True
            i = n.find(x, i + 1)
        return False
    if not all(has_num(x) for x in nums):
        return False
    return (sum(w in n for w in words) >= max(1, round(0.8 * len(words)))) if words else True


def phrase_pattern(text):
    return re.compile(r'\s+'.join(re.escape(x) for x in text.split()), re.I)


def check_assertions(page, assertions):
    from playwright.sync_api import expect
    for spec in assertions:
        loc = page.locator(spec['selector'])
        if spec.get('visible') is False:
            expect(loc).to_be_hidden(timeout=3000)
            continue
        expect(loc).to_be_visible(timeout=3000)
        for text in spec.get('contains', []):
            # same tolerance as required text: exact phrase, or every number plus 80% of the content words, inside the panel
            if not tolerant_in(text, loc.inner_text(timeout=3000)):
                raise AssertionError(f"panel {spec['selector']} does not contain {text!r} (tolerant match)")
        for text in spec.get('absent', []):
            expect(loc).not_to_contain_text(phrase_pattern(text), use_inner_text=True, timeout=3000)


def check_sections(page, sections, state='initial'):
    """Each supplied row must appear once with all its fields in one visible row."""
    from playwright.sync_api import expect
    results=[]
    for section in sections:
        if section.get('state', 'initial') != state:
            continue
        rec={'id':section['id'], 'state':state, 'ok':False}
        try:
            box=page.locator(f'[data-testid="content-{section["id"]}"]')
            expect(box).to_be_visible(timeout=3000)
            rows=box.locator('[data-row]:visible')
            expect(rows).to_have_count(len(section['rows']),timeout=3000)
            texts=[normalized(x) for x in rows.all_inner_texts()]
            # Bipartite matching prevents two supplied records sharing one rendered row.
            matches={}
            def assign(i, seen):
                fields=[normalized(str(v)) for v in section['rows'][i].values()]
                for j,text in enumerate(texts):
                    if j in seen or not all(v in text for v in fields):continue
                    seen.add(j)
                    if j not in matches or assign(matches[j],seen):
                        matches[j]=i;return True
                return False
            rec['ok']=all(assign(i,set()) for i in range(len(section['rows'])))
            if not rec['ok']:rec['error']='A supplied row or field is missing or associated with the wrong record'
        except Exception as exc:
            rec['error']=str(exc)[:300]
        results.append(rec)
    return results


def run_interaction(page, spec, shots, viewport, index, sections):
    rec={'spec':spec,'viewport':viewport,'ok':False,'sections':[]}
    try:
        check_assertions(page,spec.get('initial',[]))
        for n,step in enumerate(spec['steps']):
            loc=page.locator(step['selector'])
            if step['action']=='click':loc.click(timeout=3000)
            elif step['action']=='select':loc.select_option(step['value'],timeout=3000)
            else:raise ValueError(f"Unsupported action {step['action']}")
            check_assertions(page,step.get('assertions',[]))
            if step.get('state'):
                rec['sections'].extend(check_sections(page,sections,step['state']))
            page.screenshot(path=str(shots/f'{viewport}-after-{index+1}-{n+1}.png'),full_page=True)
        rec['ok']=all(x['ok'] for x in rec['sections'])
    except Exception as exc:
        rec['error']=str(exc)[:300]
    return rec


def contract_prompt(brief):
    if brief.get('schema_version') != 2:return ''
    return ('\n\nFIXED CONTENT AND CHECKABLE INTERACTIONS\n'
        'This JSON is part of the task. Keep every supplied row and field; do not invent entries. '
        'Render each content_sections collection inside data-testid="content-<id>". '
        'Mark each visible record with data-row (cards, table rows, or list items are all allowed). '
        'Sections without a state start visible; state-labelled sections appear when their interaction selects that state. '
        'Interaction selectors are required neutral hooks. Implement the specified initial state and every action and assertion. '
        'Use accessible controls. Facts may span HTML elements and whitespace. Deferred panel facts do not need duplication on the initial page.\n'
        +json.dumps({'content_sections':brief.get('content_sections',[]),'interactions':brief['interactions']},ensure_ascii=False))


_NUM = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:%|km|kg|nm|m\b|ml|g\b|mm|cm|hours?|minutes?|mins?|days?|weeks?|months?|years?|am|pm)|\b\d{1,2}:\d{2}\b")


def invented_numbers(page_text, brief):
    """Dollar amounts, measured quantities and clock times on the page that appear nowhere in the brief's text,
    required content, content sections or interactions. A designed page may invent an '$8 shipping charge'; that is a
    factual failure the presence checks cannot see. Returns the list of offending strings (normalised)."""
    allowed = normalized(json.dumps({k: brief.get(k) for k in ("brief", "required_content", "content_sections", "interactions", "brand", "assets")}, ensure_ascii=False))
    clean = lambda m: m.rstrip(",.")
    allowed_tokens = {clean(m) for m in _NUM.findall(allowed)}
    allowed_digits = {d.rstrip(",.") for d in re.findall(r"\d[\d,]*(?:\.\d+)?", allowed)}
    bad = []
    for m in {clean(m) for m in _NUM.findall(normalized(page_text))}:
        digits = [d.rstrip(",.") for d in re.findall(r"\d[\d,]*(?:\.\d+)?", m)]
        if m not in allowed_tokens and not all(d in allowed_digits for d in digits):
            bad.append(m)
    return sorted(bad)
