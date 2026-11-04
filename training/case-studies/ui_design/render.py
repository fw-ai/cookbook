"""Render one generated page: copy it into a fresh scaffold workspace, build with Vite, screenshot at three widths,
run the browser checks, and write a record. The harness is the reward's eyes; it must be deterministic and must
separate "the page is broken" from "the harness is broken".

    python render.py --app path/to/App.jsx --css path/to/styles.css --out results/render-smoke/example-1
    python render.py --from-response response.txt --out ...        # parses the two fenced blocks of the output contract

Writes: dist/, shots/{1440,768,390}.png, record.json. Exit 0 whether or not the page passes; exit 2 on harness error.
No network: the build uses the pinned node_modules in scaffold/, and the browser blocks every request that is not
the local page or a local asset.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from brief_checks import normalized, check_sections, run_interaction, required_present, invented_numbers

HERE = Path(__file__).resolve().parent
SCAFFOLD = HERE / "scaffold"
VIEWPORTS = {"1440": (1440, 900), "768": (768, 1024), "390": (390, 844)}
BUILD_TIMEOUT = 120
PAGE_TIMEOUT_MS = 15000


class HarnessError(RuntimeError):
    """The harness itself failed (browser missing, server, disk); never the candidate's fault."""


class _CandidateFailed(Exception):
    """Internal: the candidate's page failed to load everywhere; the record is already filled in."""


_BLOCK = re.compile(r"```(?:jsx|js|tsx|css)?\s*path=(src/App\.jsx|src/styles\.css)\s*\n(.*?)```", re.S)


def generation_id(response_path: Path | None) -> str | None:
    """The request manifest id written next to response.txt by generate.py, if any."""
    if not response_path:
        return None
    g = response_path.parent / "generation.json"
    return json.loads(g.read_text()).get("request_id") if g.exists() else None


def parse_response(text: str) -> dict[str, str]:
    files = {m.group(1): m.group(2) for m in _BLOCK.finditer(text)}
    return files


def build(workdir: Path, app: str, css: str, assets: Path | None) -> dict:
    """Copy scaffold (node_modules symlinked), write files, vite build. Returns {ok, seconds, stderr}."""
    if not (SCAFFOLD / "package.json").is_file() or not (SCAFFOLD / "node_modules").is_dir():
        raise HarnessError("Rendering setup missing: run npm ci in ui_design/scaffold; see README.md.")
    if workdir.exists():
        shutil.rmtree(workdir)
    shutil.copytree(SCAFFOLD, workdir, ignore=shutil.ignore_patterns("node_modules", "dist"))
    (workdir / "node_modules").symlink_to(SCAFFOLD / "node_modules")
    (workdir / "src/App.jsx").write_text(app)
    (workdir / "src/styles.css").write_text(css)
    pub = workdir / "public/assets"
    pub.mkdir(parents=True)
    if assets and assets.exists():
        for p in assets.iterdir():
            if p.is_file():
                shutil.copy(p, pub / p.name)
    t0 = time.time()
    try:
        r = subprocess.run(
            ["npx", "vite", "build", "--logLevel", "error"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT,
        )
        return {
            "ok": r.returncode == 0,
            "seconds": round(time.time() - t0, 2),
            "stderr": r.stderr[-2000:],
            "stdout": r.stdout[-500:],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "seconds": BUILD_TIMEOUT, "stderr": "build timeout", "harness_error": False}


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Worker-created sockets bypass Playwright's WebSocket routing. These
        # static page evaluations do not need workers; enforce this in a trusted
        # response header, which generated markup cannot relax.
        self.send_header("Content-Security-Policy", "worker-src 'none'")
        super().end_headers()

    def log_message(self, *args):  # noqa: D102
        pass


def serve(dist: Path):
    """Serve dist/ on a free localhost port (Chromium blocks module scripts over file://). Returns (url, server)."""
    with socket.socket() as sck:
        sck.bind(("127.0.0.1", 0))
        port = sck.getsockname()[1]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(_Quiet, directory=str(dist)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}/", srv


def screenshot_and_check(
    dist: Path, shots: Path, required_text: list[str], interactions: list[dict], sections=None, brief_for_checks=None
) -> dict:
    """Headless Chromium against a localhost server, every other origin blocked, animations off."""
    from playwright.sync_api import sync_playwright

    shots.mkdir(parents=True, exist_ok=True)
    origin, srv = serve(dist)
    index = origin + "index.html"
    out = {"viewports": {}, "console_errors": [], "blocked_requests": [], "blocked_sockets": [], "interactions": []}
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:  # noqa: BLE001
            srv.shutdown()
            raise HarnessError(f"browser launch: {e}") from e
        for name, (w, h) in VIEWPORTS.items():
            ctx = browser.new_context(
                viewport={"width": w, "height": h},
                device_scale_factor=1,
                reduced_motion="reduce",
                service_workers="block",
            )

            def route(r):
                u = r.request.url
                if u.startswith(origin):
                    r.continue_()
                else:
                    out["blocked_requests"].append(u)
                    r.abort()

            def block_socket(ws):
                out["blocked_sockets"].append(ws.url)
                # A routed socket is local-only until connect_to_server() is
                # explicitly called. Drop messages without opening an upstream
                # connection; closing during the initial handshake can hang.
                ws.on_message(lambda _message: None)

            ctx.route("**/*", route)
            ctx.route_web_socket("**/*", block_socket)
            page = ctx.new_page()
            page.on("console", lambda m: out["console_errors"].append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: out["console_errors"].append(str(e)))
            try:
                page.goto(index, wait_until="load", timeout=PAGE_TIMEOUT_MS)
            except Exception as e:  # noqa: BLE001  (a hang or crash in generated code is the CANDIDATE's failure)
                out["candidate_errors"] = out.get("candidate_errors", []) + [f"{name}: {str(e)[:200]}"]
                ctx.close()
                continue
            page.add_style_tag(
                content="*{animation:none!important;transition:none!important;caret-color:transparent!important}"
            )
            page.wait_for_timeout(300)
            metrics = page.evaluate("""() => {
                const d=document.documentElement, b=document.body;
                const text=(b.innerText||'');
                const imgs=[...document.images].map(i=>({src:i.getAttribute('src'),ok:i.complete&&i.naturalWidth>0}));
                // collapsed text columns: a block of running text squeezed narrower than ~9em with 4+ lines (one word per line)
                const narrow=[...document.querySelectorAll('p,li,dd,figcaption,span,div')].filter(e=>{
                  if (e.children.length>0 && [...e.children].some(c=>c.tagName!=='BR'&&c.tagName!=='B'&&c.tagName!=='STRONG'&&c.tagName!=='EM'&&c.tagName!=='A')) return false;
                  const t=(e.innerText||'').trim(); if (t.split(/\\s+/).length<6) return false;
                  const r=e.getBoundingClientRect(); const fs=parseFloat(getComputedStyle(e).fontSize)||16;
                  return r.width>0 && r.width<9*fs && r.height>4*fs*1.2; }).map(e=>(e.innerText||'').trim().slice(0,40));
                return {scrollWidth:d.scrollWidth, clientWidth:d.clientWidth, scrollHeight:d.scrollHeight, textLength:text.length, narrowColumns:narrow,
                        text:text.slice(0,20000), images:imgs, buttons:document.querySelectorAll('button,a[href],[role=button]').length,
                        headings:document.querySelectorAll('h1,h2,h3').length};
            }""")
            page.screenshot(path=str(shots / f"{name}.png"), full_page=True)
            presence = {t: required_present(t, metrics["text"], page) for t in required_text}
            invented = invented_numbers(metrics["text"], brief_for_checks) if brief_for_checks else []
            missing = [t for t, (ok, _) in presence.items() if not ok]
            missing_exact = [
                t for t, (ok, how) in presence.items() if how != "exact"
            ]  # stricter, recorded, not gating
            out["viewports"][name] = {
                "overflow_px": max(0, metrics["scrollWidth"] - metrics["clientWidth"]),
                "height": metrics["scrollHeight"],
                "text_chars": metrics["textLength"],
                "images": metrics["images"],
                "broken_images": sum(not i["ok"] for i in metrics["images"]),
                "controls": metrics["buttons"],
                "headings": metrics["headings"],
                "missing_required_text": missing,
                "missing_exact_phrase": missing_exact,
                "invented_numbers": invented,
                "text": metrics["text"][:12000] if name == "1440" else None,
                "narrow_text_columns": metrics["narrowColumns"],
                "content_sections": check_sections(page, sections or []),
            }
            if name in ("1440", "390"):
                # A toggle must CHANGE what is visible: the expected text's visible-element count must rise on the first
                # click and fall back on the second. Text that is already on the page before the click proves nothing.
                for k, it in enumerate(interactions):  # {"click": "<selector>", "expect_text": "..."}
                    if "steps" in it:
                        out["interactions"].append(run_interaction(page, it, shots, name, k, sections or []))
                        continue
                    rec = {"spec": it, "viewport": name, "ok": False}
                    try:
                        expect = it.get("expect_text", "")
                        # Visible "panels" containing the phrase: the smallest visible elements whose normalised descendant text
                        # holds it (so "6am <span>to</span> 11pm" matches), excluding ancestors that only contain such a panel.
                        count = lambda: page.evaluate(
                            """(t) => {
                            const norm = s => (s||'').replace(/\\s+/g,' ').trim().toLowerCase();
                            const vis = e => { const cs=getComputedStyle(e); if (cs.display==='none'||cs.visibility==='hidden'||+cs.opacity===0) return false;
                                               const r=e.getBoundingClientRect(); return r.width>0 && r.height>0; };
                            const hits = [...document.querySelectorAll('body *')].filter(e => vis(e) && norm(e.textContent).includes(t));
                            return hits.filter(e => ![...e.children].some(c => hits.includes(c))).length; }""",
                            expect.lower(),
                        )
                        before = count()
                        page.click(it["click"], timeout=3000)
                        page.wait_for_timeout(250)
                        after_open = count()
                        page.screenshot(path=str(shots / f"{name}-after-{k + 1}-open.png"), full_page=True)
                        page.click(it["click"], timeout=3000)
                        page.wait_for_timeout(250)
                        after_close = count()
                        rec.update(
                            before=before,
                            after_open=after_open,
                            after_close=after_close,
                            ok=(after_open > before and after_close < after_open),
                        )
                    except Exception as e:  # noqa: BLE001
                        rec["error"] = str(e)[:200]
                    out["interactions"].append(rec)
            ctx.close()
        browser.close()
    srv.shutdown()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", type=Path)
    ap.add_argument("--css", type=Path)
    ap.add_argument("--from-response", type=Path, help="model output text; the two contract blocks are parsed from it")
    ap.add_argument("--assets", type=Path, default=None)
    ap.add_argument(
        "--brief", type=Path, default=None, help="the brief JSON: required_content and interactions are read from it"
    )
    ap.add_argument("--required-text", nargs="*", default=None, help="override; normally taken from --brief")
    ap.add_argument("--interactions", type=Path, default=None, help="override; normally taken from --brief")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t_all = time.time()
    brief = json.loads(a.brief.read_text()) if a.brief else None
    required = (
        a.required_text
        if a.required_text is not None
        else (brief.get("initial_required_content", brief["required_content"]) if brief else [])
    )
    interactions = (
        json.loads(a.interactions.read_text()) if a.interactions else (brief.get("interactions", []) if brief else [])
    )
    # Complete coverage means the brief's own list was used, in full. Hand-passed overrides never count as complete.
    checks_complete = bool(brief) and a.required_text is None and a.interactions is None and bool(required)
    if a.out.exists():  # never let a previous candidate's screenshots survive a rerun
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True)
    if a.from_response:
        files = parse_response(a.from_response.read_text())
        if set(files) != {"src/App.jsx", "src/styles.css"}:
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / "record.json").write_text(
                json.dumps(
                    {
                        "stage": "parse",
                        "ok": False,
                        "found": sorted(files),
                        "model_failure": True,
                        "rendered": False,
                        "generation_id": generation_id(a.from_response),
                    },
                    indent=2,
                )
            )
            print("parse failed: contract blocks not found", sorted(files))
            return
        app, css = files["src/App.jsx"], files["src/styles.css"]
    else:
        app, css = a.app.read_text(), a.css.read_text()
    work = a.out / "work"
    asset_dir = a.assets or (HERE / "assets/pilot" if brief and brief.get("assets") else None)
    if asset_dir and brief:
        for asset in brief.get("assets", []):
            if Path(asset["file"]).name != asset["file"] or not (asset_dir / asset["file"]).is_file():
                raise HarnessError(f"Missing or invalid supplied asset: {asset['file']}")
    b = build(work, app, css, asset_dir)
    record = {
        "build": b,
        "model_failure": not b["ok"],
        "harness_error": False,
        "rendered": False,
        "checks_complete": checks_complete,
        "generation_id": generation_id(a.from_response) if a.from_response else None,
        "brief_id": brief["id"] if brief else None,
        "required_text_checked": required,
        "interactions_checked": interactions,
    }
    if b["ok"]:
        try:
            check_args = (
                {"sections": brief.get("content_sections", [])} if brief and brief.get("schema_version") == 2 else {}
            )
            check_args["brief_for_checks"] = brief
            checks = screenshot_and_check(work / "dist", a.out / "shots", required, interactions, **check_args)
            record["checks"] = checks
            vp = checks["viewports"]
            if not vp:  # the page failed to load at every viewport: the candidate's fault, not the harness's
                record.update(
                    rendered=False,
                    model_failure=True,
                    passes_functional=False,
                    renders_clean=False,
                    error="page failed at every viewport: " + "; ".join(checks.get("candidate_errors", []))[:400],
                )
                raise _CandidateFailed
            record["rendered"] = True
            record["summary"] = {
                "console_errors": len(checks["console_errors"]),
                "blocked_requests": len(checks["blocked_requests"]),
                "blocked_sockets": len(checks["blocked_sockets"]),
                "max_overflow_px": max(v["overflow_px"] for v in vp.values()),
                "broken_images": max(v["broken_images"] for v in vp.values()),
                "missing_required_text": sorted({t for v in vp.values() for t in v["missing_required_text"]}),
                "interactions_ok": sum(i["ok"] for i in checks["interactions"]),
                "interactions_n": len(checks["interactions"]),
                "min_text_chars": min(v["text_chars"] for v in vp.values()),
                "narrow_text_columns": max(len(v["narrow_text_columns"]) for v in vp.values()),
            }
            s = record["summary"]
            s["candidate_errors"] = len(checks.get("candidate_errors", []))
            s["invented_numbers"] = sorted({x for v in vp.values() for x in v.get("invented_numbers", [])})
            s["content_section_failures"] = sum(
                not row["ok"] for v in vp.values() for row in v.get("content_sections", [])
            )
            functional = (
                s["console_errors"] == 0
                and s["candidate_errors"] == 0
                and s["blocked_sockets"] == 0
                and s["max_overflow_px"] <= 8
                and s["broken_images"] == 0
                and s["content_section_failures"] == 0
                and not s["missing_required_text"]
                and s["interactions_ok"] == s["interactions_n"]
                and s["min_text_chars"] >= 200
            )
            # narrow_text_columns is recorded as a WARNING until the pilot calibrates it (captions and editorial layouts may trip it)
            s["warnings"] = (["narrow_text_columns"] if s["narrow_text_columns"] else []) + (
                ["invented_numbers"] if s["invented_numbers"] else []
            )
            # "passes_functional" is only claimable when the brief's full requirements were checked
            record["passes_functional"] = functional if checks_complete else None
            record["renders_clean"] = functional
        except _CandidateFailed:
            pass
        except HarnessError as e:
            record["harness_error"] = True
            record["error"] = str(e)[:500]
        except Exception as e:  # noqa: BLE001
            record["harness_error"] = True
            record["error"] = "unclassified: " + str(e)[:500]
        finally:
            pass
    shutil.copytree(work / "dist", a.out / "dist", dirs_exist_ok=True) if (work / "dist").exists() else None
    shutil.rmtree(work, ignore_errors=True)
    record["seconds_total"] = round(time.time() - t_all, 2)
    (a.out / "record.json").write_text(json.dumps(record, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in record.items()
                if k
                in (
                    "passes_functional",
                    "renders_clean",
                    "checks_complete",
                    "model_failure",
                    "harness_error",
                    "seconds_total",
                    "summary",
                )
            },
            indent=1,
        )
    )
    if record["harness_error"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
