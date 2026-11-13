"""Declarative provider specs loaded from providers.d register for chains.

A spec file in ``providers.d`` becomes a normal provider through
:class:`~aivia.providers.http_spec.HttpSpecProvider`. This module proves the
store loads specs from the directory, registers each name in the provider
registry and builds live instances keyed by name so configured chains can
reference declarative providers just like hand-written adapters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aivia.core.errors import ConfigError
from aivia.core.models.provider import ProviderMeta
from aivia.providers import ProviderRegistry
from aivia.providers.http_spec import HttpSpecError, HttpSpecProvider
from aivia.providers.specs import (
    ProviderSpecStore,
    build_providers_from_specs,
    load_provider_specs,
    register_provider_specs,
)


def _spec(
    name: str = "sample-image",
    capabilities: list[str] | None = None,
    pricing: dict[str, Any] | None = None,
    **extra: object,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": name,
        "capabilities": capabilities or ["image_gen"],
        "base_url": "https://image.example.test",
        "request": {
            "method": "POST",
            "path": "/v1/images",
            "body": {"prompt": "{{ request.prompt }}"},
        },
        "response": {"fields": {"url": "data.url"}},
        "pricing": pricing or {"image_gen": {"amount_usd": 0.05, "unit": "image"}},
    }
    base.update(extra)
    return base


def _write_spec(directory: Path, filename: str, payload: dict[str, Any]) -> Path:
    import yaml

    target = directory / filename
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return target


@pytest.fixture
def spec_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "providers.d"
    directory.mkdir()
    return directory


@pytest.fixture
def registry() -> ProviderRegistry:
    return ProviderRegistry()


def test_loads_every_spec_from_the_directory_keyed_by_name(
    spec_dir: Path,
) -> None:
    _write_spec(spec_dir, "alpha.yaml", _spec("alpha"))
    _write_spec(spec_dir, "beta.yaml", _spec("beta"))

    store = ProviderSpecStore(directory=spec_dir)
    specs = store.load()

    assert set(specs) == {"alpha", "beta"}
    assert specs["alpha"].name == "alpha"
    assert specs["beta"].name == "beta"
    assert store.names() == ("alpha", "beta")
    assert len(store) == 2
    assert "alpha" in store and "missing" not in store


def test_loads_in_declared_name_order_when_iterated(spec_dir: Path) -> None:
    _write_spec(spec_dir, "zulu.yaml", _spec("zulu"))
    _write_spec(spec_dir, "alpha.yaml", _spec("alpha"))

    store = ProviderSpecStore(directory=spec_dir)
    store.load()

    assert tuple(store) == ("alpha", "zulu")


def test_a_missing_directory_is_an_empty_store(tmp_path: Path) -> None:
    store = ProviderSpecStore(directory=tmp_path / "providers.d")

    assert store.load() == {}
    assert store.names() == ()
    assert len(store) == 0


def test_reload_rereads_the_directory(spec_dir: Path) -> None:
    _write_spec(spec_dir, "alpha.yaml", _spec("alpha"))

    store = ProviderSpecStore(directory=spec_dir)
    store.load()
    _write_spec(spec_dir, "beta.yaml", _spec("beta"))
    store.reload()

    assert store.names() == ("alpha", "beta")


def test_an_invalid_spec_raises(spec_dir: Path) -> None:
    (spec_dir / "bad.yaml").write_text("not: valid: yaml: [", encoding="utf-8")

    store = ProviderSpecStore(directory=spec_dir)
    with pytest.raises(HttpSpecError, match="invalid YAML provider spec"):
        store.load()


def test_a_spec_without_a_capability_is_rejected(spec_dir: Path) -> None:
    _write_spec(spec_dir, "empty.yaml", {**_spec("empty"), "capabilities": []})

    store = ProviderSpecStore(directory=spec_dir)
    with pytest.raises(HttpSpecError, match="invalid HTTP provider spec"):
        store.load()


def test_get_returns_none_for_an_unknown_name(spec_dir: Path) -> None:
    _write_spec(spec_dir, "alpha.yaml", _spec("alpha"))

    store = ProviderSpecStore(directory=spec_dir)
    store.load()

    assert store.get("alpha") is not None
    assert store.get("missing") is None


def test_register_makes_each_spec_name_resolvable(
    spec_dir: Path, registry: ProviderRegistry
) -> None:
    _write_spec(spec_dir, "alpha.yaml", _spec("alpha"))
    _write_spec(spec_dir, "beta.yaml", _spec("beta", capabilities=["video_gen"]))

    store = ProviderSpecStore(directory=spec_dir)
    store.load()
    store.register(registry)

    assert registry.resolve("alpha") is not None
    assert registry.resolve("beta") is not None
    assert issubclass(registry.resolve("alpha"), HttpSpecProvider)
    assert issubclass(registry.resolve("beta"), HttpSpecProvider)


def test_register_uses_the_shared_registry_by_default(
    spec_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_spec(spec_dir, "alpha.yaml", _spec("alpha"))
    store = ProviderSpecStore(directory=spec_dir)
    store.load()

    fake = ProviderRegistry()
    monkeypatch.setattr("aivia.providers.specs.provider_registry", fake)
    store.register()
    assert fake.resolve("alpha") is not None


def test_register_rejects_duplicate_names(
    spec_dir: Path, registry: ProviderRegistry
) -> None:
    _write_spec(spec_dir, "alpha.yaml", _spec("alpha"))
    store = ProviderSpecStore(directory=spec_dir)
    store.load()
    store.register(registry)

    with pytest.raises(ConfigError, match="already registered: alpha"):
        store.register(registry)


def test_build_creates_one_instance_per_spec_keyed_by_name(
    spec_dir: Path, registry: ProviderRegistry
) -> None:
    _write_spec(spec_dir, "alpha.yaml", _spec("alpha"))
    _write_spec(spec_dir, "beta.yaml", _spec("beta", capabilities=["video_gen"]))

    store = ProviderSpecStore(directory=spec_dir)
    store.load()
    store.register(registry)
    providers = store.build(registry)

    assert set(providers) == {"alpha", "beta"}
    alpha = providers["alpha"]
    beta = providers["beta"]
    assert isinstance(alpha, HttpSpecProvider)
    assert isinstance(beta, HttpSpecProvider)
    assert alpha.name == "alpha"
    assert beta.name == "beta"
    assert alpha.capabilities == ("image_gen",)
    assert beta.capabilities == ("video_gen",)
    assert alpha.spec.name == "alpha"
    assert beta.spec.name == "beta"


def test_built_provider_exposes_the_spec_metadata(
    spec_dir: Path, registry: ProviderRegistry
) -> None:
    _write_spec(
        spec_dir,
        "alpha.yaml",
        _spec("alpha", pricing={"image_gen": {"amount_usd": 0.07, "unit": "image"}}),
    )

    store = ProviderSpecStore(directory=spec_dir)
    store.load()
    store.register(registry)
    provider = store.build(registry)["alpha"]

    metadata = provider.metadata
    assert isinstance(metadata, ProviderMeta)
    assert metadata.name == "alpha"
    assert tuple(metadata.capabilities) == ("image_gen",)
    price = metadata.pricing_for("image_gen")
    assert price is not None
    assert price.amount_usd == 0.07
    assert price.unit == "image"


def test_building_without_registering_fails(spec_dir: Path) -> None:
    _write_spec(spec_dir, "alpha.yaml", _spec("alpha"))

    store = ProviderSpecStore(directory=spec_dir)
    store.load()

    with pytest.raises(ConfigError, match="unknown provider type: alpha"):
        store.build()


def test_build_accepts_a_secret_loader(
    spec_dir: Path, registry: ProviderRegistry
) -> None:
    from aivia.core.secrets import SecretsLoader

    _write_spec(
        spec_dir,
        "alpha.yaml",
        _spec("alpha", secrets={"api_key": "SAMPLE_KEY"}),
    )

    store = ProviderSpecStore(directory=spec_dir)
    store.load()
    store.register(registry)
    providers = store.build(
        registry,
        secrets=SecretsLoader(environ={"SAMPLE_KEY": "resolved-secret"}),
    )

    provider = providers["alpha"]
    assert "api_key" in provider.secret_names


def test_module_level_load_register_and_build(spec_dir: Path) -> None:
    _write_spec(spec_dir, "alpha.yaml", _spec("alpha"))

    store = load_provider_specs(directory=spec_dir)
    assert store.names() == ("alpha",)

    registry = ProviderRegistry()
    register_provider_specs(directory=spec_dir, registry=registry)
    assert registry.resolve("alpha") is not None

    providers = store.build(registry)
    assert set(providers) == {"alpha"}
    assert providers["alpha"].name == "alpha"


def test_build_providers_from_specs_uses_the_shared_registry_by_default(
    spec_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_spec(spec_dir, "alpha.yaml", _spec("alpha"))

    fake = ProviderRegistry()
    monkeypatch.setattr("aivia.providers.specs.provider_registry", fake)
    providers = build_providers_from_specs(directory=spec_dir)
    assert set(providers) == {"alpha"}
    assert providers["alpha"].name == "alpha"
    assert fake.resolve("alpha") is not None


def test_registered_provider_can_be_selected_by_a_router(
    spec_dir: Path, registry: ProviderRegistry
) -> None:
    from aivia.core.router import Router

    _write_spec(spec_dir, "alpha.yaml", _spec("alpha"))
    _write_spec(spec_dir, "beta.yaml", _spec("beta", capabilities=["video_gen"]))

    store = ProviderSpecStore(directory=spec_dir)
    store.load()
    store.register(registry)
    providers = store.build(registry)

    router = Router(providers=providers)
    assert router.select("image_gen").name == "alpha"
    assert router.select("video_gen").name == "beta"
