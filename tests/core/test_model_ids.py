"""Model catalog contract: known, unknown and deprecated configured ids."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.cli import main
from aivia.core import paths
from aivia.core.config import LLMConfig, TokenPricing
from aivia.core.errors import ConfigError
from aivia.core.model_ids import (
    CATALOG_FILENAME,
    ModelCatalog,
    ModelIdStatus,
    ModelIdWarning,
    configured_model_ids,
    load_model_catalog,
    model_catalog_path,
    validate_model_ids,
    warn_model_ids,
)

CATALOG = """
schema_version: 1
models:
  alpha-llm:
    capabilities: [llm]
  beta-llm:
    capabilities: [llm]
  old-llm:
    capabilities: [llm]
    deprecated: true
    replacement: alpha-llm
""".lstrip()


def write_catalog(tmp_path: Path, text: str = CATALOG) -> ModelCatalog:
    path = tmp_path / CATALOG_FILENAME
    path.write_text(text, encoding="utf-8")
    return ModelCatalog.load(path)


def test_a_known_model_id_needs_no_attention(tmp_path: Path) -> None:
    catalog = write_catalog(tmp_path)

    assert catalog.check("alpha-llm") is None
    assert "alpha-llm" in catalog
    assert "missing" not in catalog
    assert len(catalog) == 3


def test_an_unknown_model_id_reports_the_closest_match(tmp_path: Path) -> None:
    catalog = write_catalog(tmp_path)

    issue = catalog.check("alpha-lm", path="llm.roles.writing")

    assert issue is not None
    assert issue.status is ModelIdStatus.UNKNOWN
    assert issue.closest == "alpha-llm"
    assert issue.path == "llm.roles.writing"
    assert "unknown model id 'alpha-lm'" in issue.message
    assert "closest match is 'alpha-llm'" in issue.message


def test_an_unknown_model_id_without_a_close_match_has_no_suggestion(
    tmp_path: Path,
) -> None:
    catalog = write_catalog(tmp_path)

    issue = catalog.check("completely-unrelated")

    assert issue is not None
    assert issue.status is ModelIdStatus.UNKNOWN
    assert issue.closest is None
    assert "closest match" not in issue.message


def test_a_deprecated_model_id_reports_its_replacement(tmp_path: Path) -> None:
    catalog = write_catalog(tmp_path)

    issue = catalog.check("old-llm", path="llm.roles.writing")

    assert issue is not None
    assert issue.status is ModelIdStatus.DEPRECATED
    assert issue.replacement == "alpha-llm"
    assert "is deprecated" in issue.message
    assert "use 'alpha-llm' instead" in issue.message


def test_warn_model_ids_warns_for_roles_and_pricing(tmp_path: Path) -> None:
    catalog = write_catalog(tmp_path)
    config = {
        "llm": {
            "roles": {"default": "alpha-llm", "writing": "alpha-lm"},
            "pricing": {"old-llm": {}},
        }
    }

    with pytest.warns(ModelIdWarning) as caught:
        issues = warn_model_ids(config, catalog=catalog)

    assert [issue.path for issue in issues] == [
        "llm.roles.writing",
        "llm.pricing.old-llm",
    ]
    assert len(caught) == 2
    assert "llm.roles.writing" in str(caught[0].message)
    assert "is deprecated" in str(caught[1].message)


def test_an_llm_config_object_is_checked_too(tmp_path: Path) -> None:
    catalog = write_catalog(tmp_path)
    config = LLMConfig(
        roles={"writing": "alpha-lm"},
        pricing={"old-llm": TokenPricing(input_per_million=1, output_per_million=2)},
    )

    issues = validate_model_ids(config, catalog=catalog)

    assert [issue.model for issue in issues] == ["alpha-lm", "old-llm"]


def test_configured_model_ids_returns_roles_and_pricing_paths() -> None:
    config = {
        "llm": {
            "roles": {"writing": "alpha-llm", "default": "beta-llm"},
            "pricing": {"alpha-llm": {}},
        }
    }

    uses = configured_model_ids(config)

    assert [(use.model, use.path) for use in uses] == [
        ("beta-llm", "llm.roles.default"),
        ("alpha-llm", "llm.roles.writing"),
        ("alpha-llm", "llm.pricing.alpha-llm"),
    ]


def test_configured_model_ids_accepts_a_missing_llm_section() -> None:
    assert configured_model_ids({"pipeline": {}}) == ()
    assert configured_model_ids({}) == ()


def test_the_catalog_accepts_a_list_form(tmp_path: Path) -> None:
    catalog = write_catalog(
        tmp_path,
        """
schema_version: 1
models:
  - id: alpha-llm
    capabilities: [llm]
  - id: old-llm
    deprecated: true
    replacement: alpha-llm
""".lstrip(),
    )

    assert catalog.ids() == ("alpha-llm", "old-llm")
    assert catalog.get("old-llm") is not None


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="duplicate model id"):
        write_catalog(
            tmp_path,
            """
schema_version: 1
models:
  - id: alpha-llm
  - id: alpha-llm
""".lstrip(),
        )


def test_a_deprecated_entry_requires_a_replacement(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must name a replacement"):
        write_catalog(
            tmp_path,
            """
schema_version: 1
models:
  old-llm:
    deprecated: true
""".lstrip(),
        )


def test_a_replacement_must_be_listed(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="is not in the catalog"):
        write_catalog(
            tmp_path,
            """
schema_version: 1
models:
  old-llm:
    deprecated: true
    replacement: missing
""".lstrip(),
        )


def test_a_deprecated_entry_cannot_replace_itself(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot replace itself"):
        write_catalog(
            tmp_path,
            """
schema_version: 1
models:
  old-llm:
    deprecated: true
    replacement: old-llm
""".lstrip(),
        )


def test_unknown_entry_keys_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown key: retired"):
        write_catalog(
            tmp_path,
            """
schema_version: 1
models:
  alpha-llm:
    retired: true
""".lstrip(),
        )


def test_unknown_top_level_keys_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown model catalog key: extra"):
        write_catalog(
            tmp_path,
            """
schema_version: 1
models: {}
extra: true
""".lstrip(),
        )


def test_a_newer_catalog_schema_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="newer than supported"):
        write_catalog(
            tmp_path,
            """
schema_version: 99
models: {}
""".lstrip(),
        )


def test_invalid_yaml_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid YAML"):
        write_catalog(tmp_path, "models: [unclosed\n")


def test_a_missing_catalog_fails_clearly(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / CATALOG_FILENAME

    with pytest.raises(ConfigError, match="cannot read model catalog"):
        ModelCatalog.load(missing)


def test_the_user_catalog_wins_over_the_packaged_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    user_catalog = tmp_path / "aivia" / CATALOG_FILENAME
    user_catalog.parent.mkdir(parents=True)
    user_catalog.write_text(
        "schema_version: 1\nmodels:\n  custom-llm: {}\n",
        encoding="utf-8",
    )

    catalog = load_model_catalog()

    assert catalog.source == user_catalog
    assert "custom-llm" in catalog
    assert "mock-llm" not in catalog


def test_the_packaged_catalog_is_used_without_a_user_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    catalog = load_model_catalog()

    assert catalog.source == paths.package_defaults_dir() / CATALOG_FILENAME
    assert "mock-llm" in catalog
    assert model_catalog_path() == catalog.source


def test_config_validate_warns_but_does_not_fail_for_an_unknown_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "aivia"))
    catalog = tmp_path / CATALOG_FILENAME
    catalog.write_text(CATALOG, encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "schema_version: 1\nllm:\n  roles:\n    writing: alpha-lm\n",
        encoding="utf-8",
    )

    with pytest.warns(ModelIdWarning, match="closest match is 'alpha-llm'"):
        code = main(["config", "validate", "--config", str(config)])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "configuration is valid"
