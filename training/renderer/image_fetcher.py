"""How ``image_to_chunk`` turns an image URL from a dataset row into bytes.

Data URIs are always decoded locally. Every other URL goes to the image
fetcher registered with ``set_image_fetcher``, or, when none is registered,
to ``fetch_image_url``: a plain http(s) GET with a timeout and a size cap.

The default fetcher does not check where a URL points, so a service that
renders other people's datasets registers its own fetcher and sets
``COOKBOOK_REQUIRE_IMAGE_FETCHER=1``. With that set and no fetcher
registered, every URL other than a data URI is refused.
"""

import os
import urllib.request
import zlib
from collections.abc import Callable
from urllib.parse import urlsplit

import httpx

from training.renderer.errors import ImageURLError, RendererError

ImageFetcher = Callable[[str], bytes]

REQUIRE_IMAGE_FETCHER_ENV = "COOKBOOK_REQUIRE_IMAGE_FETCHER"
IMAGE_URL_MAX_BYTES = 20 * 1024 * 1024
IMAGE_URL_TIMEOUT_SECONDS = 30.0
IMAGE_URL_MAX_REDIRECTS = 5
IMAGE_URL_ERROR_MESSAGE = (
    "Cannot fetch an image URL. Use a publicly reachable http or https URL, "
    "or embed the image as a base64 data URI."
)

_image_fetcher: ImageFetcher | None = None


def set_image_fetcher(fetcher: ImageFetcher | None) -> None:
    """Routes image URL fetches through ``fetcher``; ``None`` restores the default.

    ``fetcher`` takes the URL and returns the image bytes. To reach render
    workers that ``make_render_dataloader`` spawns, it must be picklable, such
    as a module-level function.
    """
    global _image_fetcher
    _image_fetcher = fetcher


def get_image_fetcher() -> ImageFetcher | None:
    """Returns the fetcher registered with ``set_image_fetcher``, if any."""
    return _image_fetcher


def read_image_url(url: str) -> bytes:
    """Returns the bytes an image URL or data URI points to.

    Raises:
        ImageURLError: If the fetcher fails. The message is fixed; the
            fetcher's own error is chained as the cause.
        RendererError: If URLs are refused because no fetcher is registered.
    """
    if urlsplit(url).scheme.lower() == "data":
        with urllib.request.DataHandler().data_open(urllib.request.Request(url)) as response:
            return response.read()
    fetcher = _image_fetcher
    if fetcher is None:
        if os.environ.get(REQUIRE_IMAGE_FETCHER_ENV) == "1":
            raise RendererError(
                "Image URLs cannot be fetched here because no image fetcher is registered; "
                "embed the image as a data URI instead."
            )
        fetcher = fetch_image_url
    try:
        return fetcher(url)
    except Exception as e:
        raise ImageURLError(IMAGE_URL_ERROR_MESSAGE) from e


def fetch_image_url(url: str) -> bytes:
    """Default fetcher: an http(s) GET with a timeout and a size cap.

    It does not check the address a URL resolves to, so it is meant for
    datasets the caller trusts.
    """
    scheme = urlsplit(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise RendererError(f"Image URL scheme '{scheme}' is not supported; use http, https or a data URI.")
    with httpx.Client(timeout=IMAGE_URL_TIMEOUT_SECONDS) as client:
        # Ask for an uncompressed body, but decode one a server sends anyway.
        # The cap applies to the decoded bytes.
        request = client.build_request("GET", url, headers={"Accept-Encoding": "identity"})
        redirects = 0
        while True:
            response = client.send(request, stream=True)
            try:
                # Automatic redirects buffer and decode uncapped bodies.
                # next_request preserves HTTPX's URL and auth handling.
                if response.next_request is not None:
                    if redirects >= IMAGE_URL_MAX_REDIRECTS:
                        raise httpx.TooManyRedirects("Exceeded maximum allowed redirects.", request=request)
                    request = response.next_request
                    redirects += 1
                    continue
                response.raise_for_status()
                decompressor = _decompressor(response.headers.get("Content-Encoding", ""))
                body = bytearray()
                for chunk in response.iter_raw():
                    _append_capped(body, chunk, decompressor)
                if decompressor is not None:
                    _append_capped(body, decompressor.flush(), None)
                return bytes(body)
            finally:
                response.close()


class _DeflateDecompressor:
    """HTTP deflate: zlib-wrapped by the spec, but some servers send raw deflate.

    Like httpx, tries the zlib form and switches to raw deflate if the first
    chunk fails to decode.
    """

    def __init__(self) -> None:
        self._inner = zlib.decompressobj()
        self._first = True

    def decompress(self, data: bytes, max_length: int) -> bytes:
        if self._first:
            self._first = False
            try:
                return self._inner.decompress(data, max_length)
            except zlib.error:
                self._inner = zlib.decompressobj(-zlib.MAX_WBITS)
        return self._inner.decompress(data, max_length)

    @property
    def unconsumed_tail(self) -> bytes:
        return self._inner.unconsumed_tail

    def flush(self) -> bytes:
        return self._inner.flush()


def _decompressor(content_encoding: str) -> "zlib._Decompress | _DeflateDecompressor | None":
    """Returns a decompressor for content_encoding, or None for an uncompressed body."""
    encodings = [e.strip().lower() for e in content_encoding.split(",")]
    encodings = [e for e in encodings if e not in ("", "identity")]
    if not encodings:
        return None
    if encodings in (["gzip"], ["x-gzip"]):
        # 32 + MAX_WBITS reads a gzip header.
        return zlib.decompressobj(32 + zlib.MAX_WBITS)
    if encodings == ["deflate"]:
        return _DeflateDecompressor()
    raise RendererError(f"Image URL Content-Encoding {content_encoding!r} is not supported.")


def _append_capped(
    body: bytearray, data: bytes, decompressor: "zlib._Decompress | _DeflateDecompressor | None"
) -> None:
    """Appends data to body, decoded by decompressor, raising once body passes the size cap.

    Decoding asks for at most one byte past the cap at a time, so a small
    compressed chunk cannot expand into a large allocation first.
    """
    if decompressor is None:
        body += data
    else:
        while data:
            body += decompressor.decompress(data, IMAGE_URL_MAX_BYTES - len(body) + 1)
            if len(body) > IMAGE_URL_MAX_BYTES:
                break
            data = decompressor.unconsumed_tail
    if len(body) > IMAGE_URL_MAX_BYTES:
        raise RendererError(f"Image URL response is larger than {IMAGE_URL_MAX_BYTES} bytes.")
