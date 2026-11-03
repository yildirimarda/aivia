"""``aivia profile show`` and persistent brief-derived settings."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from aivia.cli import main
from aivia.core.config import ContentProfileConfig
from aivia.research import (
    BriefDerivation,
    BriefDerivationStore,
    brief_hash,
    ensure_derivation,
)


class FakeDeriver:
    """A deterministic deriver that records how often it is asked."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def derive(
        self,
        profile: ContentProfileConfig,
        *,
        cache: Any = None,
        no_cache: bool = False,
        **values: Any,
    ) -> BriefDerivation:
        self.calls.append(profile.brief)
        return BriefDerivation(
            queries=[f"query:{profile.brief}"],
            subtopics=[profile.niche] if profile.niche else [],
            include_terms=list(profile.keywords),
            exclude_terms=["off topic"],
            brief=profile.brief,
            brief_hash=brief_hash(profile.brief),
            keywords=list(profile.keywords),
            provider="fake",
            model="fake-model",
        )


def _run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def test_the_store_reuses_a_derivation_until_the_brief_changes(tmp_path: Path) -> None:
    deriver = FakeDeriver()
    store = BriefDerivationStore(tmp_path / "derived.json")
    profile = ContentProfileConfig.from_mapping(
        {"brief": "Explain stars.", "niche": "astronomy", "keywords": ["space"]}
    )

    first = _run(ensure_derivation("sky", profile, store=store, deriver=deriver))
    second = _run(ensure_derivation("sky", profile, store=store, deriver=deriver))

    assert deriver.calls == ["Explain stars."]
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.queries == first.queries
    assert second.include_terms == first.include_terms

    changed = ContentProfileConfig.from_mapping(
        {"brief": "Explain planets.", "niche": "astronomy", "keywords": ["space"]}
    )
    third = _run(ensure_derivation("sky", changed, store=store, deriver=deriver))

    assert deriver.calls == ["Explain stars.", "Explain planets."]
    assert third.cache_hit is False
    assert third.queries == ["space", "query:Explain planets."]


def test_changing_keywords_reuses_the_persisted_derivation(tmp_path: Path) -> None:
    deriver = FakeDeriver()
    store = BriefDerivationStore(tmp_path / "derived.json")
    profile = ContentProfileConfig.from_mapping(
        {"brief": "Explain stars.", "keywords": ["space"]}
    )
    _run(ensure_derivation("sky", profile, store=store, deriver=deriver))

    changed = profile.model_copy(update={"keywords": ["telescopes"]})
    result = _run(ensure_derivation("sky", changed, store=store, deriver=deriver))

    assert deriver.calls == ["Explain stars."]
    assert result.cache_hit is True
    assert result.keywords == ["telescopes"]
    assert result.include_terms == ["telescopes"]
    assert "space" not in result.queries


def _write_config(path: Path, brief: str) -> Path:
    path.write_text(
        f"""
schema_version: 1
profiles:
  briefed:
    brief: {brief}
    niche: astronomy
    keywords: [space]
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _show(
    config: Path,
    capsys: pytest.CaptureFixture[str],
    *extra: str,
) -> tuple[int, dict[str, Any], str]:
    code = main(["profile", "show", "briefed", "--config", str(config), *extra])
    captured = capsys.readouterr()
    data = (
        cast(dict[str, Any], yaml.safe_load(captured.out))
        if captured.out.strip()
        else {}
    )
    return code, data, captured.err


def test_profile_show_exposes_derived_settings_and_reuses_them(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aivia.research import derived as derived_module

    deriver = FakeDeriver()

    async def fake_derive(profile: Any, **values: Any) -> BriefDerivation:
        return await deriver.derive(profile, **values)

    monkeypatch.setattr(derived_module, "derive_brief", fake_derive)
    config = _write_config(tmp_path / "config.yaml", "Explain stars.")

    code, data, err = _show(config, capsys, "--derived")

    assert code == 0, err
    assert data["profile"] == "briefed"
    assert data["brief"] == "Explain stars."
    assert data["derived"]["queries"] == ["space", "query:Explain stars."]
    assert data["derived"]["cache_hit"] is False
    assert data["derived"]["brief_hash"] == brief_hash("Explain stars.")
    assert deriver.calls == ["Explain stars."]

    code, data, err = _show(config, capsys, "--derived")

    assert code == 0, err
    assert data["derived"]["cache_hit"] is True
    assert data["derived"]["queries"] == ["space", "query:Explain stars."]
    assert deriver.calls == ["Explain stars."]

    _write_config(config, "Explain planets.")
    code, data, err = _show(config, capsys, "--derived")

    assert code == 0, err
    assert data["derived"]["cache_hit"] is False
    assert data["derived"]["queries"] == ["space", "query:Explain planets."]
    assert deriver.calls == ["Explain stars.", "Explain planets."]


def test_profile_show_without_derived_makes_no_derivation_call(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aivia.research import derived as derived_module

    async def unexpected(*args: Any, **kwargs: Any) -> BriefDerivation:
        raise AssertionError("profile show without --derived must not derive")

    monkeypatch.setattr(derived_module, "derive_brief", unexpected)
    config = _write_config(tmp_path / "config.yaml", "Explain stars.")

    code, data, err = _show(config, capsys)

    assert code == 0, err
    assert data["profile"] == "briefed"
    assert "derived" not in data


def test_profile_show_refresh_forces_regeneration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aivia.research import derived as derived_module

    deriver = FakeDeriver()

    async def fake_derive(profile: Any, **values: Any) -> BriefDerivation:
        return await deriver.derive(profile, **values)

    monkeypatch.setattr(derived_module, "derive_brief", fake_derive)
    config = _write_config(tmp_path / "config.yaml", "Explain stars.")
    _show(config, capsys, "--derived")

    code, data, err = _show(config, capsys, "--derived", "--refresh")

    assert code == 0, err
    assert data["derived"]["cache_hit"] is False
    assert deriver.calls == ["Explain stars.", "Explain stars."]


def test_profile_show_rejects_an_unknown_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _write_config(tmp_path / "config.yaml", "Explain stars.")

    code = main(["profile", "show", "missing", "--config", str(config)])
    captured = capsys.readouterr()

    assert code == 1
    assert captured.out == ""
    assert "unknown profile: missing" in captured.err
