"""HTTP page fetch provider: extraction, limits, SSRF and allowlist behaviour."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from aivia.core.errors import (
    AiviaError,
    AuthError,
    ConfigError,
    InvalidRequestError,
    RateLimitError,
    RetryableError,
)
from aivia.core.errors import (
    TimeoutError as AiviaTimeoutError,
)
from aivia.providers import provider_registry
from aivia.providers.pagefetch import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_CHARACTERS,
    DEFAULT_MAX_REDIRECTS,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    HttpPageFetchProvider,
    PageFetch,
    PageFetcher,
    ReadableContent,
    ReadablePageFetchProvider,
    WebPageFetchProvider,
    extract_main_text,
    extract_readable,
    host_is_allowed,
    is_blocked_ip,
    readability_extract,
)
from aivia.providers.protocols import (
    CAPABILITY_PAGE_FETCH,
    PageFetchRequest,
    PageKind,
)

URL = "https://example.com/article"
PUBLIC_IP = "93.184.216.34"


def _article_html() -> str:
    return """<!doctype html>
<html lang="en-GB">
<head>
  <title>  Tides and the moon  </title>
  <meta name="author" content="Ada Lovelace">
  <meta property="article:published_time" content="2026-01-02T03:04:05Z">
  <meta property="og:title" content="Ignored og title">
</head>
<body>
  <header class="site-header">
    <nav><a href="/home">Home</a> <a href="/news">News</a></nav>
  </header>
  <main>
    <article class="post-content">
      <h1>Tides and the moon</h1>
      <p>The moon raises two tidal bulges on Earth, one facing it and one opposite.</p>
      <p>Coastal geography shapes how those bulges arrive as high and low water.</p>
      <p>Read the <a href="/tides/explained">full explanation</a> for details.</p>
    </article>
    <aside class="related">
      <p>Related: unrelated sidebar copy that should not survive.</p>
    </aside>
  </main>
  <footer class="site-footer"><p>Copyright notice and cookie banner.</p></footer>
</body>
</html>"""


def _provider(
    handler: Any,
    **kwargs: Any,
) -> HttpPageFetchProvider:
    return HttpPageFetchProvider(
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


def _capture(
    content: bytes = b"<html><body><p>Hello page text.</p></body></html>",
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> tuple[list[httpx.Request], Any]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, content=content, headers=headers)

    return requests, handler


def test_metadata_registration_and_aliases() -> None:
    provider = _provider(lambda request: httpx.Response(200, text="ok"))

    assert provider.name == "http"
    assert provider.capabilities == (CAPABILITY_PAGE_FETCH,)
    assert provider.supports(CAPABILITY_PAGE_FETCH)
    assert provider_registry.resolve("http") is HttpPageFetchProvider
    assert PageFetch is HttpPageFetchProvider
    assert PageFetcher is HttpPageFetchProvider
    assert WebPageFetchProvider is HttpPageFetchProvider
    assert ReadablePageFetchProvider is HttpPageFetchProvider
    assert extract_main_text is extract_readable
    assert readability_extract is extract_readable


def test_extracts_readable_article_text_and_metadata() -> None:
    requests, handler = _capture(
        _article_html().encode(),
        headers={"content-type": "text/html; charset=utf-8"},
    )
    provider = _provider(handler)

    response = asyncio.run(provider.fetch(PageFetchRequest(url=URL)))

    assert requests[0].method == "GET"
    assert str(requests[0].url) == URL
    assert response.url == URL
    assert response.requested_url == URL
    assert response.status_code == 200
    assert response.content_type == "text/html"
    assert response.kind is PageKind.HTML
    assert response.is_html is True
    assert response.ok is True
    assert response.title == "Tides and the moon"
    assert response.author == "Ada Lovelace"
    assert response.published_at == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert response.language == "en-GB"
    assert "moon raises two tidal bulges" in response.text
    assert "Coastal geography" in response.text
    assert "unrelated sidebar copy" not in response.text
    assert "Copyright notice" not in response.text
    assert "Home News" not in response.text
    assert "https://example.com/tides/explained" in response.links
    assert response.html is not None and "<article" in response.html
    assert response.size_bytes == len(_article_html().encode())
    assert response.truncated is False
    assert response.elapsed_ms is not None and response.elapsed_ms >= 0


def test_uses_request_and_provider_headers() -> None:
    requests, handler = _capture()
    provider = _provider(handler, user_agent="provider-agent")

    asyncio.run(
        provider.fetch(
            PageFetchRequest(
                url=URL,
                user_agent="request-agent",
                headers={"X-Trace": "abc", "Accept": "text/plain"},
            )
        )
    )

    sent = requests[0]
    assert sent.headers["user-agent"] == "request-agent"
    assert sent.headers["x-trace"] == "abc"
    assert sent.headers["accept"] == "text/plain"
    assert sent.headers["accept-encoding"] == "identity"


def test_default_user_agent_is_used_when_the_request_has_none() -> None:
    requests, handler = _capture()
    provider = _provider(handler)

    asyncio.run(provider.fetch(PageFetchRequest(url=URL)))

    assert requests[0].headers["user-agent"] == DEFAULT_USER_AGENT


@pytest.mark.parametrize(
    ("content", "content_type", "kind"),
    [
        (b'{"answer": 42}', "application/json", PageKind.JSON),
        (b"plain text body", "text/plain; charset=utf-8", PageKind.TEXT),
        (b"<rss><channel /></rss>", "application/rss+xml", PageKind.XML),
        (b"\x89PNG\r\n\x1a\n", "image/png", PageKind.BINARY),
    ],
)
def test_maps_content_types_to_page_kinds(
    content: bytes,
    content_type: str,
    kind: PageKind,
) -> None:
    provider = _provider(
        lambda request: httpx.Response(
            200,
            content=content,
            headers={"content-type": content_type},
        )
    )

    response = asyncio.run(provider.fetch(PageFetchRequest(url=URL)))

    assert response.kind is kind
    assert response.content_type == content_type.split(";")[0]


def test_sniffs_html_when_the_server_omits_a_content_type() -> None:
    provider = _provider(
        lambda request: httpx.Response(200, content=_article_html().encode())
    )

    response = asyncio.run(provider.fetch(PageFetchRequest(url=URL)))

    assert response.kind is PageKind.HTML
    assert "moon raises two tidal bulges" in response.text


def test_decodes_a_declared_charset() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            200,
            content="caf\u00e9".encode("latin-1"),
            headers={"content-type": "text/plain; charset=iso-8859-1"},
        )
    )

    response = asyncio.run(provider.fetch(PageFetchRequest(url=URL)))

    assert response.text == "caf\u00e9"
    assert response.metadata["charset"] == "iso-8859-1"


def test_truncates_the_response_to_max_bytes() -> None:
    body = b"<html><body><p>" + b"x" * 200 + b"</p></body></html>"
    provider = _provider(
        lambda request: httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/html"},
        )
    )

    response = asyncio.run(provider.fetch(PageFetchRequest(url=URL, max_bytes=50)))

    assert response.truncated is True
    assert response.size_bytes == 50
    assert len(response.html or "") <= 50


def test_stops_reading_a_body_larger_than_the_declared_length() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            200,
            content=b"x" * 100,
            headers={"content-type": "text/plain", "content-length": "100"},
        )
    )

    response = asyncio.run(provider.fetch(PageFetchRequest(url=URL, max_bytes=10)))

    assert response.truncated is True
    assert response.size_bytes == 10
    assert response.text == "x" * 10


def test_truncates_extracted_text_to_max_characters() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            200,
            content=_article_html().encode(),
            headers={"content-type": "text/html"},
        )
    )

    response = asyncio.run(provider.fetch(PageFetchRequest(url=URL, max_characters=20)))

    assert response.truncated is True
    assert len(response.text) == 20


def test_follows_redirects_and_reports_the_final_url() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/article":
            return httpx.Response(302, headers={"location": "/final"})
        return httpx.Response(
            200,
            content=b"<html><body><p>Final page.</p></body></html>",
            headers={"content-type": "text/html"},
        )

    provider = _provider(handler)

    response = asyncio.run(provider.fetch(PageFetchRequest(url=URL)))

    assert response.url == "https://example.com/final"
    assert response.requested_url == URL
    assert response.metadata["redirects"] == 1
    assert "Final page." in response.text


def test_can_refuse_to_follow_redirects() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            302,
            content=b"moved",
            headers={"location": "https://example.com/final"},
        )
    )

    response = asyncio.run(
        provider.fetch(PageFetchRequest(url=URL, follow_redirects=False))
    )

    assert response.status_code == 302
    assert response.url == URL


def test_rejects_a_redirect_chain_over_the_limit() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            302,
            headers={"location": f"https://example.com/{request.url.path}"},
        ),
        max_redirects=2,
    )

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(provider.fetch(PageFetchRequest(url=URL)))

    assert "redirect" in str(caught.value)
    assert caught.value.details["max_redirects"] == 2


def test_revalidates_every_redirect_hop_against_ssrf() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            302,
            headers={"location": "http://127.0.0.1/admin"},
        )
    )

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(provider.fetch(PageFetchRequest(url=URL)))

    assert caught.value.details["host"] == "127.0.0.1"


def test_revalidates_every_redirect_hop_against_the_allowlist() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            302,
            headers={"location": "https://evil.test/"},
        ),
        allowed_hosts=["example.com"],
    )

    with pytest.raises(InvalidRequestError):
        asyncio.run(provider.fetch(PageFetchRequest(url=URL)))


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://10.1.2.3/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://2130706433/",
        "http://0x7f000001/",
        "http://localhost/",
        "http://metadata.google.internal/",
        "http://printer.local/",
    ],
)
def test_blocks_literal_private_and_internal_targets(url: str) -> None:
    provider = _provider(lambda request: httpx.Response(200, text="should not happen"))

    with pytest.raises(InvalidRequestError):
        asyncio.run(provider.fetch(PageFetchRequest(url=url)))


def test_blocks_a_public_host_that_resolves_to_a_private_address() -> None:
    provider = _provider(
        lambda request: httpx.Response(200, text="should not happen"),
        resolver=lambda host, port: ["10.0.0.7"],
    )

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(provider.fetch(PageFetchRequest(url=URL)))

    assert caught.value.details["address"] == "10.0.0.7"


def test_allows_a_public_host_that_resolves_to_a_public_address() -> None:
    provider = _provider(
        lambda request: httpx.Response(200, text="public"),
        resolver=lambda host, port: [PUBLIC_IP],
    )

    response = asyncio.run(provider.fetch(PageFetchRequest(url=URL)))

    assert response.text == "public"


def test_allowlist_accepts_exact_hosts_subdomains_and_wildcards() -> None:
    assert host_is_allowed("example.com", ("example.com",))
    assert host_is_allowed("news.example.com", ("example.com",))
    assert host_is_allowed("news.example.com", ("*.example.com",))
    assert not host_is_allowed("evilexample.com", ("example.com",))
    assert not host_is_allowed("example.org", ("example.com",))
    assert host_is_allowed("anything.test", ())


def test_allowlist_rejects_an_unlisted_host_before_dns() -> None:
    calls: list[tuple[str, int]] = []

    def resolver(host: str, port: int) -> list[str]:
        calls.append((host, port))
        return [PUBLIC_IP]

    provider = _provider(
        lambda request: httpx.Response(200, text="should not happen"),
        allowed_hosts=["allowed.test"],
        resolver=resolver,
    )

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(provider.fetch(PageFetchRequest(url=URL)))

    assert caught.value.details["host"] == "example.com"
    assert calls == []


def test_allowlist_permits_a_listed_host() -> None:
    provider = _provider(
        lambda request: httpx.Response(200, text="allowed"),
        allowed_hosts=["example.com"],
    )

    response = asyncio.run(provider.fetch(PageFetchRequest(url=URL)))

    assert response.text == "allowed"


def test_accepts_allowlist_alias_and_rejects_duplicates() -> None:
    provider = _provider(
        lambda request: httpx.Response(200, text="ok"),
        allowlist=["Example.COM"],
    )

    assert provider.allowed_hosts == ("example.com",)
    with pytest.raises(ConfigError):
        HttpPageFetchProvider(allowlist=["example.com", "EXAMPLE.com"])


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, InvalidRequestError),
        (401, AuthError),
        (403, AuthError),
        (404, InvalidRequestError),
        (429, RateLimitError),
        (500, RetryableError),
        (503, RetryableError),
    ],
)
def test_maps_error_statuses_to_the_shared_taxonomy(
    status: int,
    expected: type[AiviaError],
) -> None:
    provider = _provider(lambda request: httpx.Response(status, text="no body details"))

    with pytest.raises(expected) as caught:
        asyncio.run(provider.fetch(PageFetchRequest(url=URL)))

    assert caught.value.details["provider"] == "http"
    assert caught.value.details["status"] == status


def test_rate_limit_carries_retry_after() -> None:
    provider = _provider(
        lambda request: httpx.Response(
            429,
            headers={"retry-after": "2.5"},
            text="slow down",
        )
    )

    with pytest.raises(RateLimitError) as caught:
        asyncio.run(provider.fetch(PageFetchRequest(url=URL)))

    assert caught.value.retry_after == 2.5


class _SlowTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        await asyncio.sleep(1.0)
        return httpx.Response(200, text="too late", request=request)


def test_enforces_the_time_limit() -> None:
    provider = HttpPageFetchProvider(transport=_SlowTransport())

    with pytest.raises(AiviaTimeoutError) as caught:
        asyncio.run(provider.fetch(PageFetchRequest(url=URL, timeout_seconds=0.01)))

    assert caught.value.details["provider"] == "http"


def test_rejects_javascript_rendering() -> None:
    provider = _provider(lambda request: httpx.Response(200, text="ok"))

    with pytest.raises(InvalidRequestError) as caught:
        asyncio.run(provider.fetch(PageFetchRequest(url=URL, render_js=True)))

    assert "JavaScript" in str(caught.value)


def test_rejects_invalid_configuration() -> None:
    with pytest.raises(ConfigError):
        HttpPageFetchProvider(allowlist=["  "])
    with pytest.raises(TypeError):
        HttpPageFetchProvider(allowlist=["a.test"], allowed_hosts=["b.test"])
    with pytest.raises(ValueError):
        HttpPageFetchProvider(max_bytes=0)
    with pytest.raises(ValueError):
        HttpPageFetchProvider(max_redirects=-1)
    with pytest.raises(ValueError):
        HttpPageFetchProvider(timeout_seconds=0)
    with pytest.raises(TypeError):
        HttpPageFetchProvider(resolver="not callable")  # type: ignore[arg-type]


def test_extract_readable_can_be_used_on_its_own() -> None:
    content = extract_readable(_article_html(), "https://example.com/article")

    assert isinstance(content, ReadableContent)
    assert content.title == "Tides and the moon"
    assert content.author == "Ada Lovelace"
    assert content.language == "en-GB"
    assert content.published_at == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert "moon raises two tidal bulges" in content.text
    assert "https://example.com/tides/explained" in content.links


def test_extract_readable_falls_back_to_body_text_without_paragraphs() -> None:
    content = extract_readable(
        "<html><body><div>Only one short block.</div></body></html>"
    )

    assert content.text == "Only one short block."


@pytest.mark.parametrize(
    ("address", "blocked"),
    [
        ("127.0.0.1", True),
        ("10.0.0.1", True),
        ("169.254.169.254", True),
        ("::1", True),
        ("::ffff:127.0.0.1", True),
        ("93.184.216.34", False),
        ("2606:2800:220:1:248:1893:25c8:1946", False),
    ],
)
def test_is_blocked_ip_classifies_addresses(address: str, blocked: bool) -> None:
    assert is_blocked_ip(address) is blocked


def test_default_limits_are_documented_constants() -> None:
    provider = _provider(lambda request: httpx.Response(200, text="ok"))

    assert provider.max_bytes == DEFAULT_MAX_BYTES
    assert provider.max_characters == DEFAULT_MAX_CHARACTERS
    assert provider.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert provider.max_redirects == DEFAULT_MAX_REDIRECTS


def test_json_text_is_returned_verbatim() -> None:
    body = json.dumps({"answer": 42}).encode()
    provider = _provider(
        lambda request: httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/json"},
        )
    )

    response = asyncio.run(provider.fetch(PageFetchRequest(url=URL)))

    assert response.kind is PageKind.JSON
    assert response.text == body.decode()
    assert response.html is None
