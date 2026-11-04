"""Local-only regressions for generated-page browser network isolation."""

from pathlib import Path
import http.server
import json
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import render  # noqa: E402


@pytest.fixture
def sink():
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b"local sink")

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", hits
    server.shutdown()
    server.server_close()
    thread.join()


def check_page(tmp_path, monkeypatch, html, files=None):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text(html)
    for name, contents in (files or {}).items():
        (dist / name).write_text(contents)
    monkeypatch.setattr(render, "VIEWPORTS", {"1440": (640, 480)})
    return render.screenshot_and_check(dist, tmp_path / "shots", [], [])


def test_popup_first_navigation_is_blocked(tmp_path, monkeypatch, sink):
    url, hits = sink
    result = check_page(
        tmp_path,
        monkeypatch,
        f"""<h1>Local page</h1>
<script>window.open({url + '/popup'!r}, 'candidate-popup');</script>""",
    )
    assert hits == [], hits
    assert url + "/popup" in result["blocked_requests"]


def test_nested_frames_and_subresources_are_blocked(tmp_path, monkeypatch, sink):
    url, hits = sink
    result = check_page(
        tmp_path,
        monkeypatch,
        '<h1>Local page</h1><iframe src="nested.html"></iframe>',
        {"nested.html": f"""<iframe src="{url}/frame"></iframe>
<img src="{url}/image"><script>fetch({url + '/fetch'!r}).catch(() => {{}})</script>"""},
    )
    assert hits == [], hits
    assert {url + "/frame", url + "/image", url + "/fetch"} <= set(result["blocked_requests"])


def test_service_worker_execution_is_blocked(tmp_path, monkeypatch, sink):
    url, hits = sink
    result = check_page(
        tmp_path,
        monkeypatch,
        """<h1>Local page</h1><p id="status">service-worker-not-started</p>
<script>navigator.serviceWorker.addEventListener('message', () => {
    document.getElementById('status').textContent = 'service-worker-executed';
});
navigator.serviceWorker.register('worker.js').catch(() => {
    document.getElementById('status').textContent = 'service-worker-blocked';
});</script>""",
        {"worker.js": f"""self.addEventListener('install', event => event.waitUntil((async () => {{
await fetch({url + '/worker'!r}).catch(() => {{}});
for (const client of await self.clients.matchAll({{includeUncontrolled: true}}))
    client.postMessage('worker-executed');
}})()));"""},
    )
    assert hits == [], hits
    text = result["viewports"]["1440"]["text"]
    assert "service-worker-executed" not in text
    assert "service-worker-not-started" in text or "service-worker-blocked" in text


def test_websocket_handshake_is_blocked(tmp_path, monkeypatch, sink):
    url, hits = sink
    socket_url = url.replace("http://", "ws://", 1) + "/socket"
    result = check_page(
        tmp_path,
        monkeypatch,
        f"""<h1>Local page</h1>
<script>new WebSocket({socket_url!r});</script>""",
    )
    assert hits == [], hits
    assert socket_url in result["blocked_sockets"]


def test_socket_attempt_is_a_functional_failure(tmp_path, monkeypatch, sink):
    url, hits = sink
    socket_url = url.replace("http://", "ws://", 1) + "/socket"
    html = (
        '<link rel="icon" href="data:,"><h1>Local fixture</h1><p>'
        + "Readable local documentation with plenty of ordinary words. " * 8
        + f"</p><script>new WebSocket({socket_url!r});</script>"
    )

    def build_fixture(workdir, app, css, assets):
        dist = workdir / "dist"
        dist.mkdir(parents=True)
        (dist / "index.html").write_text(html)
        return {"ok": True}

    app, css, brief, out = (tmp_path / name for name in ("App.jsx", "style.css", "brief.json", "output"))
    app.write_text("unused by the fixture build")
    css.write_text("")
    brief.write_text(json.dumps({"id": "local-fixture", "required_content": ["Local fixture"], "interactions": []}))
    monkeypatch.setattr(render, "build", build_fixture)
    monkeypatch.setattr(render, "VIEWPORTS", {"1440": (640, 480)})
    monkeypatch.setattr(
        sys, "argv", ["render.py", "--app", str(app), "--css", str(css), "--brief", str(brief), "--out", str(out)]
    )
    render.main()
    record = json.loads((out / "record.json").read_text())
    summary = record["summary"]
    assert hits == [], hits
    assert record["rendered"] and record["checks_complete"]
    assert summary["console_errors"] == 0
    assert summary["broken_images"] == 0
    assert summary["missing_required_text"] == []
    assert summary["max_overflow_px"] <= 8
    assert summary["min_text_chars"] >= 200
    assert summary["blocked_sockets"] > 0
    assert record["renders_clean"] is False
    assert record["passes_functional"] is False


@pytest.mark.parametrize("kind", ["dedicated", "shared"])
def test_workers_cannot_bypass_socket_isolation(tmp_path, monkeypatch, sink, kind):
    url, hits = sink
    socket_url = url.replace("http://", "ws://", 1) + "/" + kind + "-worker-socket"
    # A candidate-controlled meta policy must not relax the trusted HTTP header.
    head = '<meta http-equiv="Content-Security-Policy" content="worker-src * blob: data:">'
    if kind == "dedicated":
        html = head + """<h1>Dedicated worker probe</h1><p id="status">pending</p><script>
window.probe = new Worker('worker.js');
probe.onmessage = event => document.getElementById('status').textContent = event.data;
</script>"""
        worker = f"postMessage('dedicated-worker-started'); new WebSocket({socket_url!r});"
    else:
        html = head + """<h1>Shared worker probe</h1><p id="status">pending</p><script>
window.probe = new SharedWorker('worker.js', {name: 'local-socket-probe'});
probe.port.onmessage = event => document.getElementById('status').textContent = event.data;
probe.port.start();
</script>"""
        worker = f"onconnect = event => {{ event.ports[0].postMessage('shared-worker-started'); new WebSocket({socket_url!r}); }};"
    result = check_page(tmp_path, monkeypatch, html, {"worker.js": worker})
    assert hits == [], hits
    assert kind + "-worker-started" not in result["viewports"]["1440"]["text"]
    assert any("worker-src" in error for error in result["console_errors"])
