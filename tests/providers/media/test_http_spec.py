"""Declarative HTTP provider engine: sync, async, templates and pricing."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from aivia.core.config.http import HttpClientConfig
from aivia.core.errors import (
    ConfigError,
    ContentPolicyError,
    InvalidRequestError,
    RateLimitError,
    ValidationError,
)
from aivia.providers import (
    HttpProviderSpec,
    HttpSpecProvider,
    build_http_provider,
    extract_path,
    load_http_spec,
    load_http_specs,
)
from aivia.providers.http_spec import (
    HttpSpecEngine,
    HttpSpecTemplateError,
    ProviderSpec,
)
from aivia.providers.media import (
    HttpSpec as MediaHttpSpec,
)
from aivia.providers.media import (
    HttpSpecProvider as MediaHttpSpecProvider,
)
from aivia.providers.protocols import (
    CAPABILITY_IMAGE_GEN,
    CAPABILITY_TTS,
    ImageFormat,
    ImageRequest,
    ImageResponse,
    TTSRequest,
    VideoRequest,
    VideoResponse,
)

API_KEY = "test-key"
PNG = b"\x89PNG\r\n\x1a\n" + b"declarative-image"


def _routes(
    routes: dict[tuple[str, str], list[tuple[int, Any]]],
) -> tuple[list[httpx.Request], Callable[[httpx.Request], httpx.Response]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        key = (request.method, str(request.url))
        responses = routes[key]
        status, body = responses.pop(0) if len(responses) > 1 else responses[0]
        if isinstance(body, bytes):
            return httpx.Response(status, content=body)
        return httpx.Response(status, json=body)

    return requests, handler


def _provider(
    spec: HttpProviderSpec | dict[str, Any],
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> HttpSpecProvider:
    return HttpSpecProvider(
        spec,
        secret_values={"api_key": API_KEY},
        transport=httpx.MockTransport(handler),
        poll_interval_seconds=0.001,
        **kwargs,
    )


def _sync_spec(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "name": "example-image",
        "capabilities": ["image_gen"],
        "base_url": "https://api.example.test",
        "headers": {"authorization": "Bearer {{ secrets.api_key }}"},
        "request": {
            "method": "POST",
            "path": "/v1/images",
            "query": {"api": "{{ request.model or 'v1' }}"},
            "body": {
                "prompt": "{{ request.prompt }}",
                "width": "{{ request.width | default(1024) }}",
                "height": "{{ request.height | default(1024) }}",
                "count": "{{ request.count }}",
                "seed": "{{ request.seed }}",
            },
        },
        "response": {
            "results": "data",
            "fields": {
                "url": "image.url",
                "base64": "image.b64_json",
                "width": "image.width",
                "height": "image.height",
                "seed": "meta.seed",
                "format": "meta.mime",
            },
        },
        "pricing": {
            "image_gen": {"amount_usd": 0.02, "unit": "image"},
            "default": {"amount_usd": 0.01, "unit": "call"},
        },
        "secrets": {"api_key": "EXAMPLE_API_KEY"},
    }
    values.update(overrides)
    return values


def _async_video_spec() -> dict[str, Any]:
    return {
        "name": "example-video",
        "capabilities": ["video_gen"],
        "base_url": "https://queue.example.test",
        "async": {
            "submit": {
                "method": "POST",
                "path": "/jobs",
                "body": {
                    "prompt": "{{ request.prompt }}",
                    "seconds": "{{ request.duration_seconds | default(4) }}",
                },
            },
            "poll": {"method": "GET", "path": "/jobs/{{ remote_id }}"},
            "cancel": {"method": "POST", "path": "/jobs/{{ remote_id }}/cancel"},
            "remote_id": "job.id",
            "status": "job.state",
            "progress": "job.percent",
            "error_message": "job.error",
            "error_code": "job.error_code",
            "states": {
                "running": ["processing"],
                "succeeded": ["finished"],
                "failed": ["rejected"],
            },
            "results": "output.clips",
        },
        "response": {
            "fields": {
                "url": "file.url",
                "duration_seconds": "file.duration",
                "width": "file.width",
                "height": "file.height",
                "fps": "file.fps",
                "has_audio": "file.audio",
                "thumbnail_url": "file.thumbnail",
            }
        },
        "pricing": {"video_gen": {"amount_usd": 0.25, "unit": "clip"}},
    }


def test_sync_spec_renders_templates_and_maps_the_response() -> None:
    endpoint = "https://api.example.test/v1/images?api=v1"
    requests, handler = _routes(
        {
            ("POST", endpoint): [
                (
                    200,
                    {
                        "data": [
                            {
                                "image": {
                                    "url": "/files/image.png",
                                    "width": 768,
                                    "height": 512,
                                },
                                "meta": {"seed": 41, "mime": "image/png"},
                            }
                        ]
                    },
                )
            ]
        }
    )
    provider = _provider(_sync_spec(), handler)
    request = ImageRequest(
        prompt="  a calm ocean  ",
        width=768,
        height=512,
        count=1,
        seed=41,
    )

    response = cast(ImageResponse, asyncio.run(provider.generate(request)))

    submitted = requests[0]
    assert submitted.method == "POST"
    assert str(submitted.url) == endpoint
    assert submitted.headers["authorization"] == f"Bearer {API_KEY}"
    assert json.loads(submitted.content) == {
        "prompt": "a calm ocean",
        "width": 768,
        "height": 512,
        "count": 1,
        "seed": 41,
    }
    assert response.provider == "example-image"
    assert response.count == 1
    image = response.first
    assert image.url == "https://api.example.test/files/image.png"
    assert image.width == 768
    assert image.height == 512
    assert image.seed == 41
    assert image.format is ImageFormat.PNG


def test_sync_spec_decodes_inline_base64_and_uses_request_defaults() -> None:
    endpoint = "https://api.example.test/v1/images?api=v1"
    requests, handler = _routes(
        {
            ("POST", endpoint): [
                (
                    200,
                    {
                        "data": [
                            {
                                "image": {"b64_json": base64.b64encode(PNG).decode()},
                                "meta": {"mime": "image/png"},
                            }
                        ]
                    },
                )
            ]
        }
    )
    provider = _provider(_sync_spec(), handler)

    response = cast(
        ImageResponse,
        asyncio.run(
            provider.generate(ImageRequest(prompt="inline", width=10, height=20))
        ),
    )

    assert json.loads(requests[0].content)["width"] == 10
    assert response.first.data == PNG
    assert response.first.url is None


def test_download_results_replaces_a_url_with_the_fetched_bytes() -> None:
    endpoint = "https://api.example.test/v1/images?api=v1"
    requests, handler = _routes(
        {
            ("POST", endpoint): [
                (
                    200,
                    {
                        "data": [
                            {
                                "image": {"url": "https://cdn.example.test/a.png"},
                                "meta": {"mime": "image/png"},
                            }
                        ]
                    },
                )
            ],
            ("GET", "https://cdn.example.test/a.png"): [(200, PNG)],
        }
    )
    provider = _provider(_sync_spec(), handler, download_results=True)

    response = cast(
        ImageResponse,
        asyncio.run(
            provider.generate(ImageRequest(prompt="download", width=1, height=1))
        ),
    )

    assert [request.method for request in requests] == ["POST", "GET"]
    assert response.first.data == PNG
    assert response.first.url is None


def test_pricing_and_metadata_come_from_the_spec() -> None:
    provider = _provider(
        _sync_spec(),
        lambda request: httpx.Response(200, json={"data": []}),
    )

    assert provider.capabilities == (CAPABILITY_IMAGE_GEN,)
    assert provider.spec.mode == "sync"
    assert provider.spec.target_hosts == ("api.example.test",)
    assert provider.spec.secret_names == ("api_key",)
    assert provider.spec.secret_references == ("EXAMPLE_API_KEY",)
    assert provider.price_for("image_gen") is not None
    assert provider.price_for("image_gen").amount_usd == 0.02  # type: ignore[union-attr]
    assert provider.estimate_cost("image_gen", 3) == pytest.approx(0.06)
    assert provider.estimate_cost("default", 2) == pytest.approx(0.02)
    assert provider.estimate_cost("missing") == pytest.approx(0.01)
    assert provider.metadata.pricing_for("image_gen") is not None
    assert provider.metadata.price_for("default") is not None


def test_numeric_pricing_shorthand_is_a_default_price() -> None:
    spec = HttpProviderSpec.from_mapping(
        {
            "name": "priced",
            "capabilities": ["image_gen"],
            "base_url": "https://api.example.test",
            "request": {"method": "POST", "path": "/images"},
            "pricing": 0.5,
        }
    )

    assert spec.pricing["default"].amount_usd == 0.5
    assert spec.to_metadata().price_for("image_gen") is not None


def test_async_spec_submits_polls_and_maps_video_results() -> None:
    routes: dict[tuple[str, str], list[tuple[int, Any]]] = {
        ("POST", "https://queue.example.test/jobs"): [(202, {"job": {"id": "job-1"}})],
        ("GET", "https://queue.example.test/jobs/job-1"): [
            (
                200,
                {
                    "job": {"state": "processing", "percent": 25},
                },
            ),
            (
                200,
                {
                    "job": {"state": "finished", "percent": 100},
                    "output": {
                        "clips": [
                            {
                                "file": {
                                    "url": "https://cdn.example.test/clip.mp4",
                                    "duration": 5,
                                    "width": 1280,
                                    "height": 720,
                                    "fps": 24,
                                    "audio": True,
                                    "thumbnail": "/thumbs/clip.jpg",
                                }
                            }
                        ]
                    },
                },
            ),
        ],
    }
    requests, handler = _routes(routes)
    provider = _provider(_async_video_spec(), handler)

    response = cast(
        VideoResponse,
        asyncio.run(
            provider.generate(VideoRequest(prompt="a wave", duration_seconds=5))
        ),
    )

    assert [request.method for request in requests] == ["POST", "GET", "GET"]
    assert json.loads(requests[0].content) == {
        "prompt": "a wave",
        "seconds": 5,
    }
    assert response.provider == "example-video"
    assert response.request_id == "job-1"
    clip = response.first
    assert clip.url == "https://cdn.example.test/clip.mp4"
    assert clip.duration_seconds == 5
    assert clip.width == 1280
    assert clip.height == 720
    assert clip.fps == 24
    assert clip.has_audio is True
    assert clip.thumbnail_url == "https://queue.example.test/thumbs/clip.jpg"


def test_async_failure_maps_to_content_policy() -> None:
    routes = {
        ("POST", "https://queue.example.test/jobs"): [(202, {"job": {"id": "job-2"}})],
        ("GET", "https://queue.example.test/jobs/job-2"): [
            (
                200,
                {
                    "job": {
                        "state": "rejected",
                        "error": "blocked by the content policy",
                    }
                },
            )
        ],
    }
    _, handler = _routes(routes)
    provider = _provider(_async_video_spec(), handler)

    with pytest.raises(ContentPolicyError):
        asyncio.run(provider.generate(VideoRequest(prompt="blocked")))


def test_async_cancelled_remote_state_raises() -> None:
    routes = {
        ("POST", "https://queue.example.test/jobs"): [(202, {"job": {"id": "job-3"}})],
        ("GET", "https://queue.example.test/jobs/job-3"): [
            (200, {"job": {"state": "cancelled"}})
        ],
    }
    requests, handler = _routes(routes)
    provider = _provider(_async_video_spec(), handler)

    with pytest.raises(InvalidRequestError):
        asyncio.run(provider.generate(VideoRequest(prompt="cancel me")))

    assert requests[-1].method == "GET"


@pytest.mark.parametrize(
    "template",
    [
        "{{ request.__class__ }}",
        "{{ request.__dict__ }}",
        "{% include 'other.txt' %}",
        "{% extends 'base.txt' %}",
        "{{ missing_variable }}",
    ],
)
def test_sandboxed_body_templates_block_escapes_and_missing_values(
    template: str,
) -> None:
    spec = HttpProviderSpec.from_mapping(
        {
            "name": "sandbox",
            "capabilities": ["image_gen"],
            "base_url": "https://api.example.test",
            "request": {
                "method": "POST",
                "path": "/images",
                "body_template": template,
            },
        }
    )
    provider = _provider(spec, lambda request: httpx.Response(200, json={}))
    request = ImageRequest(prompt="safe", width=1, height=1)

    with pytest.raises(HttpSpecTemplateError):
        provider.engine.build_request(
            spec.request,  # type: ignore[arg-type]
            provider.engine._context(request),
        )


def test_text_templates_render_user_text_without_turning_it_into_code() -> None:
    spec = HttpProviderSpec.from_mapping(
        {
            "name": "text",
            "capabilities": ["image_gen"],
            "base_url": "https://api.example.test",
            "request": {
                "method": "POST",
                "path": "/images",
                "body_template": '{"prompt": {{ request.prompt | tojson }}}',
            },
        }
    )
    provider = _provider(spec, lambda request: httpx.Response(200, json={}))
    engine = provider.engine
    request = ImageRequest(prompt='{{ 7 * 7 }} "quoted"', width=1, height=1)

    built = engine.build_request(
        spec.request,  # type: ignore[arg-type]
        engine._context(request),
    )

    assert built.json_body == {"prompt": '{{ 7 * 7 }} "quoted"'}


def test_extract_path_supports_dotted_and_pointer_paths() -> None:
    body = {"data": {"images": [{"url": "https://cdn.test/a.png"}]}}

    assert extract_path(body, "data.images[0].url") == "https://cdn.test/a.png"
    assert extract_path(body, "/data/images/0/url") == "https://cdn.test/a.png"
    assert extract_path(body, "$") is body
    assert extract_path(body, "data.missing") is None


def test_loads_a_spec_from_yaml_and_a_directory(tmp_path: Path) -> None:
    text = """
name: yaml-image
capabilities: [image_gen]
base_url: https://api.example.test
request:
  method: POST
  path: /images
pricing:
  image_gen:
    amount_usd: 0.03
    unit: image
"""
    path = tmp_path / "yaml-image.yaml"
    path.write_text(text, encoding="utf-8")

    spec = load_http_spec(path)
    loaded = load_http_specs(tmp_path)

    assert spec.name == "yaml-image"
    assert spec.pricing["image_gen"].amount_usd == 0.03
    assert [entry.name for entry in loaded] == ["yaml-image"]
    assert load_http_spec(text).name == "yaml-image"
    assert ProviderSpec is HttpProviderSpec


def test_invalid_specs_fail_with_a_config_error() -> None:
    with pytest.raises(ConfigError):
        HttpProviderSpec.from_mapping(
            {
                "name": "missing-capability",
                "base_url": "https://api.example.test",
                "request": {"method": "POST", "path": "/images"},
            }
        )
    with pytest.raises(ConfigError):
        HttpProviderSpec.from_mapping(
            {
                "name": "missing-operation",
                "capabilities": ["image_gen"],
                "base_url": "https://api.example.test",
            }
        )
    with pytest.raises(ConfigError):
        HttpProviderSpec.from_mapping(
            {
                "name": "both-targets",
                "capabilities": ["image_gen"],
                "base_url": "https://api.example.test",
                "request": {
                    "method": "POST",
                    "url": "https://api.example.test/images",
                    "path": "/images",
                },
            }
        )


def test_host_allowlist_matches_exact_subdomain_and_wildcard_entries() -> None:
    spec = HttpProviderSpec.from_mapping(_sync_spec())

    spec.validate_hosts(["api.example.test"])
    spec.validate_hosts(["example.test"])
    spec.validate_hosts(["*.example.test"])

    with pytest.raises(ConfigError):
        spec.validate_hosts(["other.test"])


def test_http_status_errors_use_the_shared_error_taxonomy() -> None:
    provider = _provider(
        _sync_spec(),
        lambda request: httpx.Response(429, json={"error": "slow down"}),
        http_config=HttpClientConfig(max_attempts=1),
    )

    with pytest.raises(RateLimitError):
        asyncio.run(provider.generate(ImageRequest(prompt="rate", width=1, height=1)))


def test_binary_tts_response_is_returned_as_audio() -> None:
    spec = HttpProviderSpec.from_mapping(
        {
            "name": "example-tts",
            "capabilities": ["tts"],
            "base_url": "https://speech.example.test",
            "request": {
                "method": "POST",
                "path": "/speech",
                "body": {"text": "{{ request.text }}"},
            },
            "response": {"encoding": "binary"},
            "pricing": {"tts": {"amount_usd": 0.0001, "unit": "character"}},
        }
    )
    requests, handler = _routes(
        {("POST", "https://speech.example.test/speech"): [(200, b"ID3audio")]}
    )
    provider = _provider(spec, handler)

    response = asyncio.run(
        provider.synthesize(TTSRequest(text="hello", voice="voice-1"))
    )

    assert requests[0].method == "POST"
    assert response.audio == b"ID3audio"
    assert response.provider == "example-tts"
    assert provider.capabilities == (CAPABILITY_TTS,)


def test_engine_is_exposed_independently_of_the_provider() -> None:
    spec = HttpProviderSpec.from_mapping(_sync_spec())
    engine = HttpSpecEngine(
        spec,
        secrets={"api_key": "engine-secret"},
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": []})
        ),
    )

    assert engine.render("{{ secrets.api_key }}", {"secrets": {"api_key": "x"}}) == "x"
    assert engine.render_value("{{ request.count }}", {"request": {"count": 3}}) == 3
    assert MediaHttpSpec is HttpProviderSpec
    assert MediaHttpSpecProvider is HttpSpecProvider


def test_build_http_provider_is_an_alias_for_the_constructor() -> None:
    provider = build_http_provider(
        _sync_spec(),
        secret_values={"api_key": API_KEY},
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": []})
        ),
    )

    assert isinstance(provider, HttpSpecProvider)
    assert provider.name == "example-image"


def test_validation_error_when_a_response_has_no_result() -> None:
    provider = _provider(
        _sync_spec(),
        lambda request: httpx.Response(200, json={"data": []}),
    )

    with pytest.raises(ValidationError):
        asyncio.run(provider.generate(ImageRequest(prompt="empty", width=1, height=1)))
