"""HTTP page fetch adapter with readability extraction and SSRF guarding.

The adapter retrieves one URL and returns the main text of the page rather than
the raw markup. A small, dependency-free readability pass scores block
elements, drops navigation, headers, footers and sidebars, and keeps the
metadata a downstream research stage needs: title, author, publication time,
language and absolute links.

Every fetch is bounded. ``max_bytes`` caps the number of response bytes read,
``max_characters`` caps the extracted text and ``timeout_seconds`` bounds the
whole operation, including redirects. Redirects are followed manually so every
hop is checked against the same policy instead of letting the HTTP client jump
to an unvalidated host.

SSRF protection rejects literal private, loopback, link-local, reserved and
multicast addresses, well-known internal host names and any resolved address
that is not global. A configured host allowlist is applied before DNS so an
unlisted host never triggers a lookup. The resolver is injectable so the
offline suite can exercise address validation without touching DNS.

Fetched text is untrusted data. The response never interprets it as
instructions and the caller must keep it inside a data block when it reaches a
model.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import (
    ConfigError,
    InvalidRequestError,
    RetryableError,
    ValidationError,
)
from aivia.core.errors import (
    TimeoutError as AiviaTimeoutError,
)
from aivia.core.http import error_for_response
from aivia.core.models.provider import ProviderMeta, ProviderPrice
from aivia.core.ssrf import (
    ResolveHost,
    host_is_allowed,
    is_blocked_ip,
    normalise_allowed_hosts,
    validate_ssrf_target,
)
from aivia.providers.base import Provider
from aivia.providers.protocols.page_fetch import (
    CAPABILITY_PAGE_FETCH,
    PageFetchRequest,
    PageFetchResponse,
    PageKind,
)
from aivia.providers.registry import register_provider

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_CHARACTERS",
    "DEFAULT_MAX_REDIRECTS",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_USER_AGENT",
    "HttpPageFetchProvider",
    "PageFetch",
    "PageFetchProvider",
    "PageFetcher",
    "ReadableContent",
    "ReadablePageFetchProvider",
    "WebPageFetchProvider",
    "extract_main_text",
    "extract_readable",
    "host_is_allowed",
    "is_blocked_ip",
    "readability_extract",
]

DEFAULT_MAX_BYTES = 5_000_000
"""The largest response body read from one page."""

DEFAULT_MAX_CHARACTERS = 100_000
"""The largest extracted text returned for one page."""

DEFAULT_TIMEOUT_SECONDS = 30.0
"""The deadline for one fetch, including redirects."""

DEFAULT_MAX_REDIRECTS = 5
"""The largest redirect chain a fetch follows."""

DEFAULT_USER_AGENT = "aivia/0.1"
"""The user agent sent when neither the request nor the provider sets one."""

_DEFAULT_ACCEPT = (
    "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5"
)

_DEFAULT_METADATA = ProviderMeta(
    name="http",
    capabilities=[CAPABILITY_PAGE_FETCH],
    pricing={CAPABILITY_PAGE_FETCH: ProviderPrice(amount_usd=0.0)},
)

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_SKIP_CONTENT_TAGS = frozenset(
    {
        "applet",
        "canvas",
        "embed",
        "iframe",
        "math",
        "noscript",
        "object",
        "script",
        "style",
        "svg",
        "template",
    }
)
_NOISE_TAGS = frozenset(
    {
        "aside",
        "dialog",
        "footer",
        "form",
        "header",
        "menu",
        "nav",
    }
)
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "dd",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "tfoot",
        "th",
        "thead",
        "tr",
        "ul",
    }
)
_TEXT_BLOCK_TAGS = frozenset(
    {
        "blockquote",
        "dd",
        "dt",
        "figcaption",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "p",
        "pre",
        "td",
        "th",
    }
)
_PARAGRAPH_TAGS = frozenset({"blockquote", "dd", "li", "p", "pre", "td"})
_POSITIVE_HINTS = (
    "article",
    "body",
    "content",
    "entry",
    "main",
    "post",
    "prose",
    "story",
    "text",
)
_NEGATIVE_HINTS = (
    "advert",
    "banner",
    "breadcrumb",
    "comment",
    "cookie",
    "footer",
    "header",
    "menu",
    "nav",
    "promo",
    "related",
    "share",
    "sidebar",
    "social",
    "sponsor",
)
_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


@dataclass(frozen=True, slots=True)
class ReadableContent:
    """The metadata and main text extracted from one HTML document."""

    title: str | None = None
    text: str = ""
    author: str | None = None
    published_at: datetime | None = None
    language: str | None = None
    links: list[str] = field(default_factory=list)


class _Node:
    __slots__ = ("attrs", "children", "parent", "tag")

    def __init__(
        self,
        tag: str,
        attrs: Mapping[str, str] | None = None,
        parent: _Node | None = None,
    ) -> None:
        self.tag = tag
        self.attrs: dict[str, str] = dict(attrs or {})
        self.parent = parent
        self.children: list[_Node | str] = []


class _DomParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("document")
        self._stack: list[_Node] = [self.root]
        self._skip_depth = 0

    @property
    def current(self) -> _Node:
        return self._stack[-1]

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        name = tag.lower()
        values = {key.lower(): value or "" for key, value in attrs}
        node = _Node(name, values, self.current)
        self.current.children.append(node)
        if name in _SKIP_CONTENT_TAGS and name not in _VOID_TAGS:
            self._skip_depth += 1
        if name not in _VOID_TAGS:
            self._stack.append(node)

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        name = tag.lower()
        values = {key.lower(): value or "" for key, value in attrs}
        self.current.children.append(_Node(name, values, self.current))

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name in _SKIP_CONTENT_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == name:
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data:
            self.current.children.append(data)


def _clean(value: str) -> str:
    return " ".join(value.split())


def _parse_html(html: str) -> _Node:
    parser = _DomParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # pragma: no cover - HTMLParser is very tolerant
        raise ValidationError(
            "page contains malformed HTML",
            details={"error": type(exc).__name__},
        ) from exc
    return parser.root


def _walk(node: _Node) -> Iterable[_Node]:
    for child in node.children:
        if isinstance(child, _Node):
            yield child
            yield from _walk(child)


def _node_text(node: _Node) -> str:
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, str):
            parts.append(child)
        elif child.tag not in _SKIP_CONTENT_TAGS:
            parts.append(_node_text(child))
    return _clean(" ".join(parts))


def _class_and_id(node: _Node) -> str:
    return f"{node.attrs.get('class', '')} {node.attrs.get('id', '')}".lower()


def _hint_tokens(node: _Node) -> list[str]:
    return [token for token in _TOKEN_RE.split(_class_and_id(node)) if token]


def _is_negative(node: _Node) -> bool:
    for token in _hint_tokens(node):
        if token in {"ad", "ads"} or token.startswith(_NEGATIVE_HINTS):
            return True
    return False


def _is_positive(node: _Node) -> bool:
    return any(token.startswith(_POSITIVE_HINTS) for token in _hint_tokens(node))


def _meta_content(
    root: _Node, *, names: Sequence[str], properties: Sequence[str]
) -> str | None:
    for node in _walk(root):
        if node.tag != "meta":
            continue
        key = (node.attrs.get("name") or node.attrs.get("property") or "").lower()
        if key in names or key in properties:
            value = _clean(node.attrs.get("content", ""))
            if value:
                return value
    return None


def _first_text(root: _Node, tag: str) -> str | None:
    for node in _walk(root):
        if node.tag == tag:
            value = _node_text(node)
            if value:
                return value
    return None


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    normalized = f"{text[:-1]}+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _published_at(root: _Node) -> datetime | None:
    candidates = (
        ("property", "article:published_time"),
        ("property", "og:published_time"),
        ("itemprop", "datePublished"),
        ("name", "date"),
        ("name", "pubdate"),
        ("name", "publishdate"),
        ("name", "publish-date"),
    )
    for attribute, key in candidates:
        for node in _walk(root):
            if node.tag != "meta":
                continue
            if node.attrs.get(attribute, "").lower() == key.lower():
                parsed = _parse_datetime(node.attrs.get("content"))
                if parsed is not None:
                    return parsed
    for node in _walk(root):
        if node.tag == "time":
            parsed = _parse_datetime(node.attrs.get("datetime"))
            if parsed is not None:
                return parsed
    return None


def _language(root: _Node) -> str | None:
    for node in _walk(root):
        if node.tag == "html":
            value = _clean(node.attrs.get("lang", ""))
            if value:
                return value
    for node in _walk(root):
        if node.tag != "meta":
            continue
        if node.attrs.get("http-equiv", "").lower() == "content-language":
            value = _clean(node.attrs.get("content", ""))
            if value:
                return value
    locale = _meta_content(
        root,
        names=("language",),
        properties=("og:locale",),
    )
    if locale:
        return locale.replace("_", "-")
    return None


def _author(root: _Node) -> str | None:
    value = _meta_content(
        root,
        names=("author", "twitter:creator"),
        properties=("article:author", "og:article:author"),
    )
    if value:
        return value
    for node in _walk(root):
        if node.tag == "a" and "author" in node.attrs.get("rel", "").lower():
            text = _node_text(node)
            if text:
                return text
    for node in _walk(root):
        if "author" in _class_and_id(node):
            text = _node_text(node)
            if text:
                return text
    return None


def _links(root: _Node, base_url: str | None) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for node in _walk(root):
        if node.tag != "a":
            continue
        href = node.attrs.get("href", "").strip()
        if not href:
            continue
        absolute = urljoin(base_url, href) if base_url else href
        parsed = urlsplit(absolute)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        links.append(absolute)
    return links


def _scores(root: _Node) -> dict[int, float]:
    scores: dict[int, float] = {}
    nodes: dict[int, _Node] = {}

    def add(node: _Node | None, amount: float) -> None:
        if node is None or node.tag in _SKIP_CONTENT_TAGS:
            return
        scores[id(node)] = scores.get(id(node), 0.0) + amount
        nodes[id(node)] = node

    for node in _walk(root):
        if node.tag in _SKIP_CONTENT_TAGS or _is_negative(node):
            continue
        text = _node_text(node)
        if node.tag in _PARAGRAPH_TAGS and len(text) >= 20:
            score = 1.0 + text.count(",") + min(len(text) / 100.0, 3.0)
            add(node, score)
            add(node.parent, score)
            if node.parent is not None:
                add(node.parent.parent, score * 0.5)
        if node.tag in {"article", "main"}:
            add(node, 20.0 + len(text) / 200.0)
        if _is_positive(node):
            add(node, 5.0)
    return scores


def _extract_blocks(node: _Node, blocks: list[str]) -> None:
    if node.tag in _SKIP_CONTENT_TAGS:
        return
    if node.tag in _NOISE_TAGS or _is_negative(node):
        return
    if node.tag in _TEXT_BLOCK_TAGS:
        text = _node_text(node)
        if text:
            blocks.append(text)
        return
    has_block_child = any(
        isinstance(child, _Node) and child.tag in _BLOCK_TAGS for child in node.children
    )
    if not has_block_child:
        text = _node_text(node)
        if text:
            blocks.append(text)
        return
    for child in node.children:
        if isinstance(child, _Node):
            _extract_blocks(child, blocks)
        else:
            text = _clean(child)
            if text:
                blocks.append(text)


def _main_node(root: _Node) -> _Node:
    scores = _scores(root)
    if scores:
        best = max(scores, key=scores.__getitem__)
        for node in _walk(root):
            if id(node) == best:
                return node
    for node in _walk(root):
        if node.tag in {"article", "main"}:
            return node
    for node in _walk(root):
        if node.tag == "body":
            return node
    return root


def extract_readable(html: str, base_url: str | None = None) -> ReadableContent:
    """Extract the main text and metadata from *html*.

    The extractor is deliberately dependency free. It parses the document into
    a small tree, scores paragraph-bearing blocks, drops navigation-like
    containers and joins the surviving block text with blank lines. Relative
    links are resolved against *base_url* when it is supplied.
    """
    if not isinstance(html, str):
        raise TypeError("html must be a string")
    root = _parse_html(html)
    main = _main_node(root)
    blocks: list[str] = []
    _extract_blocks(main, blocks)
    if not blocks:
        fallback = _node_text(root)
        if fallback:
            blocks.append(fallback)
    deduped: list[str] = []
    for block in blocks:
        if not deduped or deduped[-1] != block:
            deduped.append(block)
    title = (
        _first_text(root, "title")
        or _meta_content(
            root, names=("title", "twitter:title"), properties=("og:title",)
        )
        or _first_text(root, "h1")
    )
    return ReadableContent(
        title=title,
        text="\n\n".join(deduped),
        author=_author(root),
        published_at=_published_at(root),
        language=_language(root),
        links=_links(root, base_url),
    )


extract_main_text = extract_readable
"""Alias for :func:`extract_readable`."""

readability_extract = extract_readable
"""Alias for :func:`extract_readable`."""


def _mime_type(value: str | None) -> str | None:
    if value is None:
        return None
    mime = value.split(";", 1)[0].strip().lower()
    return mime or None


def _charset(value: str | None) -> str | None:
    if value is None:
        return None
    for part in value.split(";")[1:]:
        key, separator, raw = part.partition("=")
        if separator and key.strip().lower() == "charset":
            charset = raw.strip().strip('"').strip("'")
            if charset:
                return charset
    return None


def _kind_for(mime: str | None, raw: bytes) -> PageKind:
    if mime in {"text/html", "application/xhtml+xml"}:
        return PageKind.HTML
    if mime is not None:
        if mime == "application/json" or mime.endswith("+json"):
            return PageKind.JSON
        if mime in {"application/xml", "text/xml"} or mime.endswith("+xml"):
            return PageKind.XML
        if mime.startswith("text/"):
            return PageKind.TEXT
        if mime.startswith(("image/", "audio/", "video/", "font/")) or mime in {
            "application/octet-stream",
            "application/pdf",
            "application/zip",
        }:
            return PageKind.BINARY
        return PageKind.OTHER
    sample = raw.lstrip()[:512].lower()
    if sample.startswith((b"<!doctype html", b"<html", b"<head", b"<body")):
        return PageKind.HTML
    if sample.startswith((b"{", b"[")):
        return PageKind.JSON
    if sample.startswith(b"<?xml") or sample.startswith(b"<"):
        return PageKind.XML
    return PageKind.TEXT


def _decode(raw: bytes, charset: str | None) -> str:
    candidates = [charset] if charset else []
    candidates.extend(["utf-8", "latin-1"])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return raw.decode(candidate)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


class HttpPageFetchProvider(Provider):
    """A page fetch provider backed by plain HTTP with readability extraction.

    ``allowed_hosts`` is an optional allowlist of bare host names; an empty
    list permits any host that passes the SSRF guard. ``resolver`` may be a
    callable returning the addresses for ``(host, port)``; when omitted the
    provider resolves through the system resolver, except when a test
    ``transport`` is injected so the offline suite never performs DNS. Pass an
    explicit resolver to exercise address validation with an injected
    transport.
    """

    default_metadata = _DEFAULT_METADATA

    def __init__(
        self,
        *,
        allowed_hosts: Iterable[str] | None = None,
        allowlist: Iterable[str] | None = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_characters: int = DEFAULT_MAX_CHARACTERS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        user_agent: str = DEFAULT_USER_AGENT,
        follow_redirects: bool = True,
        resolver: ResolveHost | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        name: str | None = None,
        health_timeout_seconds: float | None = None,
        close_timeout_seconds: float | None = None,
        http_config: HttpClientConfig | None = None,
    ) -> None:
        super().__init__(
            name=name,
            health_timeout_seconds=health_timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            http_config=http_config,
        )
        if allowed_hosts is not None and allowlist is not None:
            raise TypeError("pass either allowed_hosts or allowlist, not both")
        hosts = allowed_hosts if allowed_hosts is not None else allowlist
        self._allowed_hosts = normalise_allowed_hosts(hosts)
        self._max_bytes = _positive_int(max_bytes, "max_bytes")
        self._max_characters = _positive_int(max_characters, "max_characters")
        self._timeout_seconds = _positive_float(timeout_seconds, "timeout_seconds")
        self._max_redirects = _non_negative_int(max_redirects, "max_redirects")
        self._user_agent = _text(user_agent, "user_agent")
        if not isinstance(follow_redirects, bool):
            raise TypeError("follow_redirects must be a boolean")
        self._follow_redirects = follow_redirects
        if resolver is not None and not callable(resolver):
            raise TypeError("resolver must be callable")
        if transport is not None and not isinstance(
            transport, httpx.AsyncBaseTransport
        ):
            raise TypeError("transport must be an httpx.AsyncBaseTransport")
        self._resolver = resolver
        self._transport = transport
        if transport is not None:
            self._http = self.create_http_client(transport=transport)

    @property
    def allowed_hosts(self) -> tuple[str, ...]:
        """The normalized host allowlist, empty when every host is allowed."""
        return self._allowed_hosts

    @property
    def max_bytes(self) -> int:
        """The default response byte limit."""
        return self._max_bytes

    @property
    def max_characters(self) -> int:
        """The default extracted text limit."""
        return self._max_characters

    @property
    def timeout_seconds(self) -> float:
        """The default deadline for one fetch, including redirects."""
        return self._timeout_seconds

    @property
    def max_redirects(self) -> int:
        """The default redirect chain limit."""
        return self._max_redirects

    @property
    def user_agent(self) -> str:
        """The default user agent."""
        return self._user_agent

    def _headers(self, request: PageFetchRequest) -> dict[str, str]:
        headers: dict[str, str] = {
            "user-agent": request.user_agent or self._user_agent,
            "accept": _DEFAULT_ACCEPT,
            "accept-encoding": "identity",
        }
        for name, value in request.headers.items():
            headers[name.lower()] = value
        if request.user_agent:
            headers["user-agent"] = request.user_agent
        return headers

    async def _validate_target(self, url: str) -> None:
        await validate_ssrf_target(
            url,
            resolver=self._resolver,
            allowed_hosts=self._allowed_hosts,
            skip_dns=self._transport is not None,
            label="page",
        )

    async def _read_limited(
        self,
        response: httpx.Response,
        max_bytes: int,
    ) -> tuple[bytes, bool]:
        chunks: list[bytes] = []
        total = 0
        truncated = False
        async for chunk in response.aiter_bytes():
            if not chunk:
                continue
            remaining = max_bytes - total
            if remaining <= 0:
                truncated = True
                break
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                total += remaining
                truncated = True
                break
            chunks.append(chunk)
            total += len(chunk)
        return b"".join(chunks), truncated

    def _response(
        self,
        *,
        request: PageFetchRequest,
        url: str,
        status_code: int,
        content_type_header: str | None,
        raw: bytes,
        bytes_truncated: bool,
        redirects: int,
        elapsed_ms: float,
    ) -> PageFetchResponse:
        mime = _mime_type(content_type_header)
        charset = _charset(content_type_header)
        kind = _kind_for(mime, raw)
        max_characters = request.max_characters or self._max_characters
        truncated = bytes_truncated
        title: str | None = None
        text = ""
        html: str | None = None
        author: str | None = None
        published_at: datetime | None = None
        language: str | None = None
        links: list[str] = []

        if kind is PageKind.BINARY:
            text = ""
        else:
            decoded = _decode(raw, charset)
            if kind is PageKind.HTML:
                html = decoded
                readable = extract_readable(decoded, url)
                title = readable.title
                text = readable.text
                author = readable.author
                published_at = readable.published_at
                language = readable.language
                links = readable.links
            else:
                text = decoded.strip()
            if len(text) > max_characters:
                text = text[:max_characters]
                truncated = True

        metadata: dict[str, Any] = {
            "provider": self.name,
            "redirects": redirects,
        }
        if content_type_header:
            metadata["content_type_header"] = content_type_header
        if charset:
            metadata["charset"] = charset
        content_length = len(raw)
        if raw:
            metadata["bytes_read"] = content_length
        return PageFetchResponse(
            url=url,
            requested_url=request.url,
            status_code=status_code,
            content_type=mime,
            kind=kind,
            title=title,
            text=text,
            html=html,
            author=author,
            published_at=published_at,
            language=language,
            links=links,
            elapsed_ms=elapsed_ms,
            size_bytes=content_length,
            truncated=truncated,
            metadata=metadata,
        )

    async def _fetch(self, request: PageFetchRequest) -> PageFetchResponse:
        if request.render_js:
            raise InvalidRequestError(
                "this page fetch provider cannot render JavaScript",
                details={"provider": self.name},
            )
        max_bytes = request.max_bytes or self._max_bytes
        max_redirects = self._max_redirects
        follow_redirects = self._follow_redirects and request.follow_redirects
        timeout = request.timeout_seconds or self._timeout_seconds
        url = request.url
        redirects = 0
        started = time.perf_counter()

        while True:
            await self._validate_target(url)
            try:
                async with self.http.client.stream(
                    "GET",
                    url,
                    headers=self._headers(request),
                    timeout=timeout,
                    follow_redirects=False,
                ) as response:
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if location and follow_redirects:
                            if redirects >= max_redirects:
                                raise InvalidRequestError(
                                    "page fetch exceeded the redirect limit",
                                    details={
                                        "provider": self.name,
                                        "url": url,
                                        "max_redirects": max_redirects,
                                    },
                                )
                            redirects += 1
                            url = urljoin(url, location)
                            parsed = urlsplit(url)
                            if (
                                parsed.scheme.lower() not in {"http", "https"}
                                or not parsed.hostname
                            ):
                                raise InvalidRequestError(
                                    "page redirect points at an invalid URL",
                                    details={"url": url},
                                )
                            continue
                    error = error_for_response(response)
                    if error is not None:
                        error.details.update(
                            {
                                "provider": self.name,
                                "url": url,
                                "status": response.status_code,
                            }
                        )
                        raise error
                    raw, bytes_truncated = await self._read_limited(
                        response,
                        max_bytes,
                    )
                    content_type_header = response.headers.get("content-type")
                    return self._response(
                        request=request,
                        url=url,
                        status_code=response.status_code,
                        content_type_header=content_type_header,
                        raw=raw,
                        bytes_truncated=bytes_truncated,
                        redirects=redirects,
                        elapsed_ms=max(0.0, (time.perf_counter() - started) * 1000.0),
                    )
            except httpx.TimeoutException as exc:
                raise AiviaTimeoutError(
                    "page fetch timed out",
                    details={
                        "provider": self.name,
                        "url": url,
                        "timeout_seconds": timeout,
                    },
                ) from exc
            except httpx.TransportError as exc:
                raise RetryableError(
                    f"page fetch transport error: {type(exc).__name__}",
                    details={
                        "provider": self.name,
                        "url": url,
                        "error": type(exc).__name__,
                    },
                ) from exc

    async def fetch(self, request: PageFetchRequest) -> PageFetchResponse:
        """Fetch *request* and return the extracted main text."""
        if not isinstance(request, PageFetchRequest):
            raise TypeError("request must be a PageFetchRequest")
        timeout = request.timeout_seconds or self._timeout_seconds
        try:
            async with asyncio.timeout(timeout):
                return await self._fetch(request)
        except TimeoutError as exc:
            raise AiviaTimeoutError(
                "page fetch timed out",
                details={
                    "provider": self.name,
                    "url": request.url,
                    "timeout_seconds": timeout,
                },
            ) from exc


def _text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ConfigError(f"{field_name} must be non-empty")
    return text


def _positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must not be negative")
    return value


def _positive_float(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return number


PageFetch = HttpPageFetchProvider
"""Alias for :class:`HttpPageFetchProvider`."""

PageFetchProvider = HttpPageFetchProvider
"""Alias for :class:`HttpPageFetchProvider`."""

PageFetcher = HttpPageFetchProvider
"""Alias for :class:`HttpPageFetchProvider`."""

WebPageFetchProvider = HttpPageFetchProvider
"""Alias for :class:`HttpPageFetchProvider`."""

ReadablePageFetchProvider = HttpPageFetchProvider
"""Alias for :class:`HttpPageFetchProvider`."""

register_provider("http", HttpPageFetchProvider, module=__name__)
