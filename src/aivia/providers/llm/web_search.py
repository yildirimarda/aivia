"""Map the vendor-neutral web search tool to each LLM adapter's native search.

The search provider in :mod:`aivia.providers.search.llm_search` asks an LLM for
results by adding a tool named ``web_search``. A vendor adapter recognises that
tool and replaces it with the vendor's native search option instead of sending
it as an ordinary function. The vendor returns its own citation shape; the
helpers here normalize those shapes into the flat citation dictionaries the
search provider already understands.

The helpers are pure functions: they own no transport, retry or budget policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from aivia.providers.protocols.llm import ToolDefinition

__all__ = [
    "WEB_SEARCH_TOOL_NAME",
    "anthropic_citations",
    "gemini_citations",
    "is_web_search_tool",
    "merge_citations",
    "openai_citations",
    "split_web_search_tools",
]

WEB_SEARCH_TOOL_NAME = "web_search"
"""The tool name every adapter maps to its native web search."""


def is_web_search_tool(tool: ToolDefinition) -> bool:
    """Whether *tool* is the vendor-neutral native web search marker."""
    return tool.name == WEB_SEARCH_TOOL_NAME


def split_web_search_tools(
    tools: Sequence[ToolDefinition],
) -> tuple[list[ToolDefinition], bool]:
    """Split *tools* into ordinary tools and a native web search marker.

    Returns the remaining function tools and whether at least one web search
    tool was present. Duplicate search markers collapse into a single flag.
    """
    remaining: list[ToolDefinition] = []
    native = False
    for tool in tools:
        if is_web_search_tool(tool):
            native = True
            continue
        remaining.append(tool)
    return remaining, native


def _first_text(item: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _citation(
    url: Any,
    title: Any = "",
    snippet: Any = "",
    **extra: Any,
) -> dict[str, Any] | None:
    if not isinstance(url, str) or not url.strip():
        return None
    citation: dict[str, Any] = {
        "title": title.strip() if isinstance(title, str) else "",
        "url": url.strip(),
        "snippet": snippet.strip() if isinstance(snippet, str) else "",
    }
    for key, value in extra.items():
        if isinstance(value, str) and value.strip():
            citation[key] = value.strip()
        elif value is not None and not isinstance(value, str):
            citation[key] = value
    return citation


def merge_citations(
    citations: Sequence[dict[str, Any] | None],
) -> list[dict[str, Any]]:
    """Merge citations by URL, keeping the first occurrence's position.

    Later duplicates fill in fields the first copy left empty. This collapses
    a page that appears both as a search result and as a text citation.
    """
    merged: list[dict[str, Any]] = []
    index: dict[str, dict[str, Any]] = {}
    for citation in citations:
        if citation is None:
            continue
        url = citation["url"]
        existing = index.get(url)
        if existing is None:
            index[url] = citation
            merged.append(citation)
            continue
        for key, value in citation.items():
            if not existing.get(key):
                existing[key] = value
    return merged


def openai_citations(annotations: Sequence[Any]) -> list[dict[str, Any]]:
    """Normalize OpenAI ``url_citation`` annotations into flat citations."""
    citations: list[dict[str, Any] | None] = []
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            continue
        kind = annotation.get("type")
        if kind not in (None, "url_citation"):
            continue
        payload = annotation.get("url_citation")
        item = payload if isinstance(payload, Mapping) else annotation
        citations.append(
            _citation(
                _first_text(item, ("url", "uri", "link")),
                _first_text(item, ("title", "name")),
                _first_text(item, ("snippet", "cited_text", "text", "content")),
            )
        )
    return merge_citations(citations)


def anthropic_citations(blocks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize Anthropic web search result and text citation blocks."""
    citations: list[dict[str, Any] | None] = []
    for block in blocks:
        kind = block.get("type")
        if kind == "web_search_tool_result":
            content = block.get("content")
            if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
                continue
            for item in content:
                if not isinstance(item, Mapping):
                    continue
                citations.append(
                    _citation(
                        _first_text(item, ("url", "uri", "link")),
                        _first_text(item, ("title", "name")),
                        _first_text(item, ("snippet", "text", "content")),
                    )
                )
        elif kind == "text":
            block_citations = block.get("citations")
            if not isinstance(block_citations, Sequence) or isinstance(
                block_citations, (str, bytes)
            ):
                continue
            for item in block_citations:
                if not isinstance(item, Mapping):
                    continue
                citations.append(
                    _citation(
                        _first_text(item, ("url", "uri", "link")),
                        _first_text(item, ("title", "name")),
                        _first_text(item, ("cited_text", "snippet", "text")),
                    )
                )
    return merge_citations(citations)


def gemini_citations(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize Gemini grounding chunks and supports into flat citations."""
    chunks = metadata.get("groundingChunks", metadata.get("grounding_chunks"))
    if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
        return []
    supports = metadata.get("groundingSupports", metadata.get("grounding_supports"))
    snippets: dict[int, str] = {}
    if isinstance(supports, Sequence) and not isinstance(supports, (str, bytes)):
        for support in supports:
            if not isinstance(support, Mapping):
                continue
            segment = support.get("segment")
            text = (
                _first_text(segment, ("text",)) if isinstance(segment, Mapping) else ""
            )
            indices = support.get(
                "groundingChunkIndices",
                support.get("grounding_chunk_indices"),
            )
            if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
                continue
            for index in indices:
                if isinstance(index, int) and index not in snippets:
                    snippets[index] = text

    citations: list[dict[str, Any] | None] = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping):
            continue
        web = chunk.get("web")
        if not isinstance(web, Mapping):
            continue
        citations.append(
            _citation(
                _first_text(web, ("uri", "url", "link")),
                _first_text(web, ("title", "name")),
                snippets.get(index, ""),
            )
        )
    return merge_citations(citations)
