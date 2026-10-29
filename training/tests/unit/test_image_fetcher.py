"""image_to_chunk fetches image URLs through a pluggable, fail-closed fetcher."""

from __future__ import annotations

import base64
import gzip
import http.server
import io
import pickle
import socket
import threading
import time
import tracemalloc
import zlib
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image
from training.renderer import image_fetcher, set_image_fetcher
from training.renderer.base import image_to_chunk
from training.renderer.errors import ImageURLError, RendererError
from training.utils.streaming import make_render_dataloader

Route = Callable[[http.server.BaseHTTPRequestHandler], None]


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


PNG = _png()
DATA_URI = "data:image/png;base64," + base64.b64encode(PNG).decode()


class _ImageProcessor:
    merge_size = 1

    def get_number_of_image_patches(
        self,
        height: int,
        width: int,
        images_kwargs: dict[str, Any] | None = None,
    ) -> int:
        del height, width, images_kwargs
        return 1


class _Server:
    """Local HTTP server that records the paths it is asked for."""

    def __init__(self, routes: dict[str, Route]) -> None:
        requests: list[str] = []
        self.requests = requests

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                requests.append(self.path)
                routes[self.path](self)

            def log_message(self, *args: Any) -> None:
                del args

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_port}"
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture(autouse=True)
def _default_fetcher(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(image_fetcher.REQUIRE_IMAGE_FETCHER_ENV, raising=False)
    set_image_fetcher(None)
    yield
    set_image_fetcher(None)


@pytest.fixture
def serve() -> Iterator[Callable[[dict[str, Route]], _Server]]:
    started: list[_Server] = []

    def start(routes: dict[str, Route]) -> _Server:
        server = _Server(routes)
        started.append(server)
        return server

    yield start
    for server in started:
        server.close()


@pytest.fixture
def release() -> Iterator[threading.Event]:
    """Lets handlers that stall the client finish when the test ends."""
    event = threading.Event()
    yield event
    event.set()


def _send(body: bytes, encoding: str = "") -> Route:
    def route(handler: http.server.BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type", "image/png")
        if encoding:
            handler.send_header("Content-Encoding", encoding)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    return route


def _never_answer(release: threading.Event) -> Route:
    def route(handler: http.server.BaseHTTPRequestHandler) -> None:
        del handler
        release.wait()

    return route


def _redirect(location: str, body: bytes = b"", encoding: str = "") -> Route:
    def route(handler: http.server.BaseHTTPRequestHandler) -> None:
        handler.send_response(302)
        handler.send_header("Location", location)
        if encoding:
            handler.send_header("Content-Encoding", encoding)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    return route


def _fake_fetcher(url: str) -> bytes:
    del url
    return PNG


def test_decodes_data_uris_without_the_fetcher() -> None:
    def refuse(url: str) -> bytes:
        raise AssertionError(f"fetcher called for {url}")

    set_image_fetcher(refuse)
    with patch.object(socket, "create_connection", side_effect=AssertionError("dialed")):
        chunk = image_to_chunk(DATA_URI, _ImageProcessor())
    assert chunk.expected_tokens == 1


def test_uses_the_registered_fetcher() -> None:
    fetched: list[str] = []

    def fetch(url: str) -> bytes:
        fetched.append(url)
        return PNG

    set_image_fetcher(fetch)
    chunk = image_to_chunk("https://images.example.com/cat.png", _ImageProcessor())
    assert chunk.expected_tokens == 1
    assert fetched == ["https://images.example.com/cat.png"]


def test_fetcher_errors_become_image_url_errors_that_hide_the_url() -> None:
    def fail(url: str) -> bytes:
        raise ValueError(f"refused {url}")

    set_image_fetcher(fail)
    url = "https://images.example.com/cat.png?X-Goog-Signature=abc"
    with pytest.raises(ImageURLError) as raised:
        image_to_chunk(url, _ImageProcessor())
    assert str(raised.value) == image_fetcher.IMAGE_URL_ERROR_MESSAGE
    assert str(raised.value.__cause__) == f"refused {url}"


def test_default_fetcher_reads_an_http_image(serve: Callable[[dict[str, Route]], _Server]) -> None:
    server = serve({"/cat.png": _send(PNG)})
    chunk = image_to_chunk(f"{server.url}/cat.png", _ImageProcessor())
    assert chunk.expected_tokens == 1
    assert server.requests == ["/cat.png"]


def test_default_fetcher_stops_at_the_size_cap(
    serve: Callable[[dict[str, Route]], _Server], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(image_fetcher, "IMAGE_URL_MAX_BYTES", 64 * 1024)
    server = serve({"/cat.png": _send(PNG + bytes(256 * 1024))})
    with pytest.raises(ImageURLError) as raised:
        image_to_chunk(f"{server.url}/cat.png", _ImageProcessor())
    assert "larger than 65536 bytes" in str(raised.value.__cause__)


def test_default_fetcher_decodes_a_compressed_body(serve: Callable[[dict[str, Route]], _Server]) -> None:
    server = serve({"/cat.png": _send(gzip.compress(PNG), encoding="gzip")})
    assert image_fetcher.fetch_image_url(f"{server.url}/cat.png") == PNG


def test_default_fetcher_decodes_a_deflate_body(serve: Callable[[dict[str, Route]], _Server]) -> None:
    server = serve({"/cat.png": _send(zlib.compress(PNG), encoding="deflate")})
    assert image_fetcher.fetch_image_url(f"{server.url}/cat.png") == PNG


def test_default_fetcher_decodes_a_raw_deflate_body(serve: Callable[[dict[str, Route]], _Server]) -> None:
    # Some servers send deflate without the zlib header the spec calls for.
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    raw = compressor.compress(PNG) + compressor.flush()
    server = serve({"/cat.png": _send(raw, encoding="deflate")})
    assert image_fetcher.fetch_image_url(f"{server.url}/cat.png") == PNG


def test_default_fetcher_never_decodes_far_past_the_cap(
    serve: Callable[[dict[str, Route]], _Server], monkeypatch: pytest.MonkeyPatch
) -> None:
    # About 64 KiB of gzip that decodes to 64 MiB.
    monkeypatch.setattr(image_fetcher, "IMAGE_URL_MAX_BYTES", 64 * 1024)
    server = serve({"/cat.png": _send(gzip.compress(bytes(64 * 1024 * 1024)), encoding="gzip")})
    tracemalloc.start()
    try:
        with pytest.raises(RendererError, match="larger than 65536 bytes"):
            image_fetcher.fetch_image_url(f"{server.url}/cat.png")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * 1024 * 1024


def test_default_fetcher_does_not_buffer_or_decode_redirect_bodies(
    serve: Callable[[dict[str, Route]], _Server], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(image_fetcher, "IMAGE_URL_MAX_BYTES", 64 * 1024)
    bomb = gzip.compress(bytes(64 * 1024 * 1024))
    server = serve({"/old.png": _redirect("/cat.png", body=bomb, encoding="gzip"), "/cat.png": _send(PNG)})
    tracemalloc.start()
    try:
        assert image_fetcher.fetch_image_url(f"{server.url}/old.png") == PNG
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 8 * 1024 * 1024
    assert server.requests == ["/old.png", "/cat.png"]


def test_default_fetcher_stops_after_max_redirects(serve: Callable[[dict[str, Route]], _Server]) -> None:
    server = serve({f"/{hop}.png": _redirect(f"/{hop + 1}.png") for hop in range(10)})
    with pytest.raises(ImageURLError) as raised:
        image_to_chunk(f"{server.url}/0.png", _ImageProcessor())
    assert "Exceeded maximum allowed redirects" in str(raised.value.__cause__)
    assert len(server.requests) == image_fetcher.IMAGE_URL_MAX_REDIRECTS + 1


def test_default_fetcher_refuses_unsupported_encodings(serve: Callable[[dict[str, Route]], _Server]) -> None:
    server = serve({"/cat.png": _send(PNG, encoding="br")})
    with pytest.raises(RendererError, match="Content-Encoding 'br' is not supported"):
        image_fetcher.fetch_image_url(f"{server.url}/cat.png")


@pytest.mark.timeout(10)
def test_default_fetcher_times_out(
    serve: Callable[[dict[str, Route]], _Server],
    release: threading.Event,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(image_fetcher, "IMAGE_URL_TIMEOUT_SECONDS", 0.2)
    server = serve({"/cat.png": _never_answer(release)})
    started = time.monotonic()
    with pytest.raises(ImageURLError) as raised:
        image_to_chunk(f"{server.url}/cat.png", _ImageProcessor())
    assert "timed out" in str(raised.value.__cause__)
    assert time.monotonic() - started < 5


def test_default_fetcher_refuses_schemes_other_than_http(tmp_path: Any) -> None:
    path = tmp_path / "cat.png"
    path.write_bytes(PNG)
    with pytest.raises(ImageURLError) as raised:
        image_to_chunk(path.as_uri(), _ImageProcessor())
    assert "scheme 'file' is not supported" in str(raised.value.__cause__)


def test_required_fetcher_refuses_urls_when_none_is_registered(
    serve: Callable[[dict[str, Route]], _Server], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(image_fetcher.REQUIRE_IMAGE_FETCHER_ENV, "1")
    server = serve({"/cat.png": _send(PNG)})
    with pytest.raises(RendererError, match="no image fetcher is registered"):
        image_to_chunk(f"{server.url}/cat.png", _ImageProcessor())
    assert server.requests == []


def test_required_fetcher_still_decodes_data_uris(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(image_fetcher.REQUIRE_IMAGE_FETCHER_ENV, "1")
    assert image_to_chunk(DATA_URI, _ImageProcessor()).expected_tokens == 1


def test_required_fetcher_uses_the_registered_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(image_fetcher.REQUIRE_IMAGE_FETCHER_ENV, "1")
    set_image_fetcher(_fake_fetcher)
    assert image_to_chunk("https://images.example.com/cat.png", _ImageProcessor()).expected_tokens == 1


_initialized_workers: list[int] = []


def _record_worker(worker_id: int) -> None:
    _initialized_workers.append(worker_id)


def test_render_workers_get_the_registered_fetcher() -> None:
    _initialized_workers.clear()
    set_image_fetcher(_fake_fetcher)
    loader = make_render_dataloader([1, 2, 3], batch_size=1, num_workers=2, worker_init_fn=_record_worker)

    # A spawned worker unpickles worker_init_fn in a fresh process, with no
    # fetcher registered.
    init = pickle.loads(pickle.dumps(loader.worker_init_fn))
    set_image_fetcher(None)
    init(0)
    assert image_fetcher.get_image_fetcher() is _fake_fetcher
    assert _initialized_workers == [0]


@pytest.mark.parametrize(
    ("render", "builder"),
    [
        ("render_messages_to_datums", "_build_renderer_supervised_examples"),
        ("render_messages_to_datum", "_render_singular_example"),
    ],
)
def test_rendering_turns_image_url_errors_into_dataset_errors(
    monkeypatch: pytest.MonkeyPatch, render: str, builder: str
) -> None:
    # lazy: supervised pulls in the full renderer stack, which the fetcher tests above don't need.
    from training.utils import supervised
    from training.utils.runner import DatasetError

    def fail(*_args: object, **_kwargs: object) -> None:
        raise ImageURLError(image_fetcher.IMAGE_URL_ERROR_MESSAGE) from ValueError(
            "refused https://images.example.com/cat.png?X-Goog-Signature=abc"
        )

    monkeypatch.setattr(supervised, builder, fail)
    with pytest.raises(DatasetError) as raised:
        getattr(supervised, render)(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
            renderer=object(),
        )
    assert str(raised.value) == image_fetcher.IMAGE_URL_ERROR_MESSAGE


def test_rendered_row_image_url_error_names_only_the_row() -> None:
    from training.utils.runner import DatasetError
    from training.utils.streaming import JSONL_ROW_INDEX_KEY, dataset_error_for_rendered_row

    row = {"messages": [{"role": "user", "content": "hi"}], JSONL_ROW_INDEX_KEY: 7}
    error = dataset_error_for_rendered_row(row, DatasetError(image_fetcher.IMAGE_URL_ERROR_MESSAGE))
    assert str(error) == f"row 7: {image_fetcher.IMAGE_URL_ERROR_MESSAGE}"
