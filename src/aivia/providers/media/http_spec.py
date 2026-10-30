"""Media-facing alias for the declarative HTTP provider engine.

The engine is generic and lives in :mod:`aivia.providers.http_spec`; this module
re-exports it so media adapters can import it from the same package as the
async job helper and the media DTOs.
"""

from aivia.providers.http_spec import (
    DEFAULT_RESULT_FIELDS,
    DeclarativeHttpProvider,
    HttpAsyncSpec,
    HTTPProvider,
    HttpProviderSpec,
    HttpRequestSpec,
    HttpResponseSpec,
    HttpSpec,
    HttpSpecEngine,
    HttpSpecError,
    HttpSpecProvider,
    HttpSpecResult,
    HttpSpecTemplateError,
    ProviderSpec,
    build_http_provider,
    extract_path,
    http_provider_from_spec,
    load_http_spec,
    load_http_spec_file,
    load_http_specs,
    parse_http_spec,
)

__all__ = [
    "DEFAULT_RESULT_FIELDS",
    "DeclarativeHttpProvider",
    "HTTPProvider",
    "HttpAsyncSpec",
    "HttpProviderSpec",
    "HttpRequestSpec",
    "HttpResponseSpec",
    "HttpSpec",
    "HttpSpecEngine",
    "HttpSpecError",
    "HttpSpecProvider",
    "HttpSpecResult",
    "HttpSpecTemplateError",
    "ProviderSpec",
    "build_http_provider",
    "extract_path",
    "http_provider_from_spec",
    "load_http_spec",
    "load_http_spec_file",
    "load_http_specs",
    "parse_http_spec",
]
