"""LLM, search and page fetch protocols, typed models and passing fakes."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from aivia.core.errors import RateLimitError
from aivia.core.models import ProviderMeta
from aivia.providers import Provider
from aivia.providers.protocols import (
    CAPABILITY_LLM,
    CAPABILITY_PAGE_FETCH,
    CAPABILITY_SEARCH,
    CapabilityProvider,
    ChatMessage,
    FinishReason,
    ImageDetail,
    ImagePart,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    MessageRole,
    PageFetchProvider,
    PageFetchRequest,
    PageFetchResponse,
    PageKind,
    SafeSearch,
    SearchDepth,
    SearchProvider,
    SearchRecency,
    SearchRequest,
    SearchResponse,
    SearchResult,
    TextPart,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    normalize_domain,
    validate_http_url,
)

from .provider_contract import ProviderContract

_PAGE_URL = "https://example.com/article"
_OTHER_URL = "https://other.example.org/background"


class FakeLLMProvider(Provider):
    """A deterministic LLM fake that records requests and replays responses."""

    default_metadata = ProviderMeta(
        name="fake-llm",
        capabilities=[CAPABILITY_LLM],
    )

    def __init__(
        self,
        responses: Sequence[LLMResponse] | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__()
        self._responses: deque[LLMResponse] = deque(responses or ())
        self._error = error
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        if self._responses:
            return self._responses.popleft()
        text = request.last_user_text
        return LLMResponse(
            content=f"echo: {text}",
            model=request.model or "",
            provider=self.name,
            usage=LLMUsage(
                input_tokens=len(text.split()),
                output_tokens=2,
            ),
        )


class FakeSearchProvider(Provider):
    """A deterministic search fake that returns synthetic example.com results."""

    default_metadata = ProviderMeta(
        name="fake-search",
        capabilities=[CAPABILITY_SEARCH],
    )

    def __init__(
        self,
        results: Sequence[SearchResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__()
        self._results = list(results) if results is not None else None
        self._error = error
        self.requests: list[SearchRequest] = []

    async def search(self, request: SearchRequest) -> SearchResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        if self._results is None:
            results = [
                SearchResult(
                    title=f"Result {index} for {request.query}",
                    url=f"https://example.com/search/{index}",
                    snippet=f"Synthetic snippet {index}",
                )
                for index in range(1, 4)
            ]
        else:
            results = self._results
        return SearchResponse(
            query=request.query,
            results=results[: request.max_results],
            provider=self.name,
            total_results=len(results),
        )


class FakePageFetchProvider(Provider):
    """A deterministic page fetch fake with canned pages per URL."""

    default_metadata = ProviderMeta(
        name="fake-fetch",
        capabilities=[CAPABILITY_PAGE_FETCH],
    )

    def __init__(
        self,
        pages: Mapping[str, PageFetchResponse] | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__()
        self._pages = dict(pages or {})
        self._error = error
        self.requests: list[PageFetchRequest] = []

    async def fetch(self, request: PageFetchRequest) -> PageFetchResponse:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        page = self._pages.get(request.url)
        if page is not None:
            return page
        return PageFetchResponse(
            url=request.url,
            status_code=200,
            content_type="text/html",
            kind=PageKind.HTML,
            title="Example page",
            text=f"Fetched content for {request.url}",
        )


class TestFakeLLMProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return FakeLLMProvider()


class TestFakeSearchProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return FakeSearchProvider()


class TestFakePageFetchProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self) -> Provider:
        return FakePageFetchProvider()


def test_fakes_satisfy_the_capability_protocols() -> None:
    llm = FakeLLMProvider()
    search = FakeSearchProvider()
    fetch = FakePageFetchProvider()

    assert isinstance(llm, LLMProvider)
    assert isinstance(search, SearchProvider)
    assert isinstance(fetch, PageFetchProvider)
    assert isinstance(llm, CapabilityProvider)
    assert isinstance(search, CapabilityProvider)
    assert isinstance(fetch, CapabilityProvider)
    assert not isinstance(search, LLMProvider)
    assert not isinstance(Provider(), LLMProvider)


def test_capability_names_are_stable() -> None:
    assert CAPABILITY_LLM == "llm"
    assert CAPABILITY_SEARCH == "search"
    assert CAPABILITY_PAGE_FETCH == "page_fetch"


def test_llm_request_carries_messages_controls_and_usage() -> None:
    request = LLMRequest(
        messages=[
            ChatMessage(role=MessageRole.SYSTEM, content="Be precise"),
            ChatMessage(role=MessageRole.USER, content="  Explain tides  "),
        ],
        role="writing",
        model="some-model",
        max_tokens=100,
        temperature=0.2,
        top_p=0.9,
        stop=["\n\n"],
        seed=7,
        response_schema={"type": "object"},
    )

    assert request.last_user_text == "Explain tides"
    assert request.messages[1].role is MessageRole.USER
    assert request.model == "some-model"
    assert request.temperature == 0.2

    response = LLMResponse(
        content="Tides follow the moon.",
        model="some-model",
        provider="fake-llm",
        usage=LLMUsage(
            input_tokens=10,
            output_tokens=5,
            cached_input_tokens=2,
        ),
        finish_reason=FinishReason.STOP,
    )

    assert response.text == response.content
    assert response.usage.total_tokens == 15
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5
    assert response.total_tokens == 15
    assert response.has_tool_calls is False


def test_protocol_models_are_immutable_and_reject_unknown_fields() -> None:
    request = LLMRequest(messages=[ChatMessage(role=MessageRole.USER, content="hi")])

    with pytest.raises(ValidationError):
        request.temperature = 0.5
    with pytest.raises(ValidationError):
        LLMRequest.model_validate(
            {
                "messages": [
                    ChatMessage(role=MessageRole.USER, content="hi").model_dump()
                ],
                "unknown": True,
            }
        )


def test_chat_message_supports_multimodal_content() -> None:
    message = ChatMessage(
        role=MessageRole.USER,
        content=[
            TextPart(text="Describe this image"),
            ImagePart(url=_PAGE_URL, detail=ImageDetail.HIGH),
        ],
    )

    assert message.text == "Describe this image"
    assert len(message.images) == 1
    assert message.images[0].detail == "high"

    inline = ChatMessage(
        role=MessageRole.USER,
        content=[
            ImagePart(data="aGVsbG8=", media_type="image/png"),
        ],
    )
    assert inline.text == ""
    assert inline.images[0].media_type == "image/png"


def test_image_part_requires_exactly_one_source_and_a_media_type() -> None:
    with pytest.raises(ValidationError, match="exactly one of url or data"):
        ImagePart(url=_PAGE_URL, data="aGVsbG8=", media_type="image/png")
    with pytest.raises(ValidationError, match="exactly one of url or data"):
        ImagePart()
    with pytest.raises(ValidationError, match="requires media_type"):
        ImagePart(data="aGVsbG8=")
    with pytest.raises(ValidationError, match="must start with 'image/'"):
        ImagePart(data="aGVsbG8=", media_type="text/plain")
    with pytest.raises(ValidationError):
        ImagePart(url="ftp://example.com/image.png")


def test_llm_request_validates_messages_tools_and_sampling() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(messages=[])
    with pytest.raises(ValidationError):
        ChatMessage(role=MessageRole.USER, content="   ")
    with pytest.raises(ValidationError, match="tool message requires tool_call_id"):
        ChatMessage(role=MessageRole.TOOL, content="result")
    with pytest.raises(ValidationError):
        LLMRequest(
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            temperature=3.0,
        )
    with pytest.raises(ValidationError, match="needs at least one tool"):
        LLMRequest(
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            tool_choice=ToolChoice.REQUIRED,
        )

    tool = ToolDefinition(name="lookup", parameters={"type": "object"})
    with pytest.raises(ValidationError, match="tool names must be unique"):
        LLMRequest(
            messages=[ChatMessage(role=MessageRole.USER, content="hi")],
            tools=[tool, tool],
        )


def test_llm_response_carries_tool_calls() -> None:
    response = LLMResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=[ToolCall(id="call-1", name="lookup", arguments={"q": "tides"})],
    )

    assert response.has_tool_calls is True
    assert response.tool_calls[0].arguments == {"q": "tides"}


def test_fake_llm_returns_typed_default_and_scripted_responses() -> None:
    request = LLMRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="hello world")],
        model="some-model",
    )
    scripted = LLMResponse(content="scripted", model="some-model")

    fake = FakeLLMProvider(responses=[scripted])
    response = asyncio.run(fake.complete(request))
    fallback = asyncio.run(fake.complete(request))

    assert response is scripted
    assert fallback.content == "echo: hello world"
    assert fallback.provider == "fake-llm"
    assert fallback.usage.input_tokens == 2
    assert fake.requests == [request, request]


def test_fake_llm_propagates_a_typed_error() -> None:
    error = RateLimitError("slow down", retry_after=1.5)
    fake = FakeLLMProvider(error=error)

    with pytest.raises(RateLimitError) as caught:
        asyncio.run(
            fake.complete(
                LLMRequest(messages=[ChatMessage(role=MessageRole.USER, content="hi")])
            )
        )

    assert caught.value is error
    assert caught.value.retry_after == 1.5


def test_search_request_normalizes_query_domains_and_dates() -> None:
    request = SearchRequest(
        query="  climate change  ",
        max_results=5,
        include_domains=["https://www.Example.com/path", "example.com"],
        exclude_domains=["News.Example.org"],
        recency=SearchRecency.WEEK,
        language=" EN ",
        country=" de ",
        safe_search=SafeSearch.STRICT,
        search_depth=SearchDepth.ADVANCED,
    )

    assert request.query == "climate change"
    assert request.include_domains == ["example.com"]
    assert request.exclude_domains == ["news.example.org"]
    assert request.language == "en"
    assert request.country == "DE"
    assert request.safe_search is SafeSearch.STRICT
    assert request.search_depth is SearchDepth.ADVANCED


def test_search_request_validates_bounds_and_date_order() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="   ")
    with pytest.raises(ValidationError):
        SearchRequest(query="x", max_results=0)
    with pytest.raises(ValidationError):
        SearchRequest(query="x", max_results=101)
    with pytest.raises(ValidationError, match="start_date must not be after"):
        SearchRequest(
            query="x",
            start_date=date(2026, 2, 1),
            end_date=date(2026, 1, 1),
        )
    with pytest.raises(ValidationError, match="bare host name"):
        SearchRequest(query="x", include_domains=["example.com:8443"])


def test_search_result_and_response_helpers() -> None:
    first = SearchResult(
        title="  First  ",
        url=_PAGE_URL,
        snippet="short",
        content="longer content",
        published_at=datetime(2026, 1, 1, 12, 0),
        score=0.8,
    )
    duplicate = SearchResult(url=_PAGE_URL, title="Duplicate")
    other = SearchResult(url=_OTHER_URL, snippet="other")

    response = SearchResponse(
        query=" tides ",
        results=[first, duplicate, other],
        provider="fake-search",
        total_results=3,
    )

    assert first.title == "First"
    assert first.text == "longer content"
    assert first.domain == "example.com"
    assert first.published_at == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert response.query == "tides"
    assert response.urls == [_PAGE_URL, _PAGE_URL, _OTHER_URL]
    assert response.unique_results() == [first, other]

    with pytest.raises(ValidationError):
        SearchResult(url="ftp://example.com/file")


def test_fake_search_returns_typed_results_and_honours_the_limit() -> None:
    fake = FakeSearchProvider()
    request = SearchRequest(query="tides", max_results=2)

    response = asyncio.run(fake.search(request))

    assert isinstance(response, SearchResponse)
    assert response.provider == "fake-search"
    assert response.query == "tides"
    assert len(response.results) == 2
    assert response.total_results == 3
    assert all(isinstance(result, SearchResult) for result in response.results)
    assert fake.requests == [request]


def test_fake_search_propagates_a_typed_error() -> None:
    error = RateLimitError("quota", retry_after=2.0)
    fake = FakeSearchProvider(error=error)

    with pytest.raises(RateLimitError):
        asyncio.run(fake.search(SearchRequest(query="tides")))


def test_page_fetch_request_validates_urls_headers_and_limits() -> None:
    request = PageFetchRequest(
        url="  https://example.com/article  ",
        max_bytes=1024,
        max_characters=500,
        timeout_seconds=5.0,
        headers={"Accept": "text/html"},
    )

    assert request.url == _PAGE_URL
    assert request.follow_redirects is True
    assert request.render_js is False
    assert request.headers == {"Accept": "text/html"}

    with pytest.raises(ValidationError):
        PageFetchRequest(url="ftp://example.com/article")
    credential_url = "https://" + "user" + ":" + "secret@" + "example.com/article"
    with pytest.raises(ValidationError, match="must not contain credentials"):
        PageFetchRequest(url=credential_url)
    with pytest.raises(ValidationError):
        PageFetchRequest(url=_PAGE_URL, max_bytes=0)
    with pytest.raises(ValidationError):
        PageFetchRequest(url=_PAGE_URL, timeout_seconds=0)
    with pytest.raises(ValidationError, match="invalid header name"):
        PageFetchRequest(url=_PAGE_URL, headers={"Bad\nName": "value"})
    with pytest.raises(ValidationError, match="invalid header value"):
        PageFetchRequest(url=_PAGE_URL, headers={"X-Test": "bad\r\nvalue"})


def test_page_fetch_response_properties_and_link_normalization() -> None:
    response = PageFetchResponse(
        url=_PAGE_URL,
        requested_url="https://example.com/redirect",
        status_code=200,
        content_type="text/html; charset=utf-8",
        title="  Article  ",
        text="  one two three  ",
        links=[_PAGE_URL, _PAGE_URL, _OTHER_URL],
        published_at=datetime(2026, 1, 1),
    )

    assert response.ok is True
    assert response.is_html is True
    assert response.word_count == 3
    assert response.title == "Article"
    assert response.links == [_PAGE_URL, _OTHER_URL]
    assert response.published_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert response.fetched_at.tzinfo is not None

    failure = PageFetchResponse(url=_PAGE_URL, status_code=404)
    assert failure.ok is False

    with pytest.raises(ValidationError):
        PageFetchResponse(url=_PAGE_URL, status_code=200, links=["/relative"])


def test_fake_page_fetch_returns_typed_pages_and_records_requests() -> None:
    custom = PageFetchResponse(
        url=_PAGE_URL,
        status_code=200,
        title="Custom",
        text="custom body",
    )
    fake = FakePageFetchProvider(pages={_PAGE_URL: custom})
    request = PageFetchRequest(url=_PAGE_URL)

    response = asyncio.run(fake.fetch(request))
    fallback = asyncio.run(fake.fetch(PageFetchRequest(url=_OTHER_URL)))

    assert response is custom
    assert fallback.url == _OTHER_URL
    assert fallback.title == "Example page"
    assert fake.requests == [request, PageFetchRequest(url=_OTHER_URL)]


def test_fake_page_fetch_propagates_a_typed_error() -> None:
    error = RateLimitError("blocked", retry_after=3.0)
    fake = FakePageFetchProvider(error=error)

    with pytest.raises(RateLimitError):
        asyncio.run(fake.fetch(PageFetchRequest(url=_PAGE_URL)))


def test_url_and_domain_helpers_reject_unsafe_shapes() -> None:
    assert validate_http_url("https://example.com/a") == "https://example.com/a"
    assert normalize_domain("https://WWW.Example.com/path") == "example.com"

    with pytest.raises(ValueError, match="absolute http"):
        validate_http_url("example.com")
    with pytest.raises(ValueError, match="credentials"):
        validate_http_url("https://user@example.com")
    with pytest.raises(ValueError, match="bare host name"):
        normalize_domain("example.com/path")
    with pytest.raises(ValueError, match="bare host name"):
        normalize_domain("example.com:443")
