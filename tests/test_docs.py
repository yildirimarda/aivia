"""Documentation contract: architecture note, decisions log and progress log."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

DESIGN = DOCS / "DESIGN.md"
DECISIONS = DOCS / "decisions.md"
PROGRESS = DOCS / "PROGRESS.md"
GUIDES = DOCS

PROVIDER_GUIDE = GUIDES / "adding_provider.md"
LANGUAGE_GUIDE = GUIDES / "adding_language.md"
FORMAT_GUIDE = GUIDES / "adding_format.md"
PLATFORM_GUIDE = GUIDES / "adding_platform.md"
OCCASION_GUIDE = GUIDES / "adding_occasion.md"
CHARACTER_GUIDE = GUIDES / "adding_character.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _guide(name: str) -> str:
    path = GUIDES / f"adding_{name}.md"
    assert path.is_file(), f"missing {path.relative_to(ROOT)}"
    text = _read(path)
    assert text.strip(), f"{path.relative_to(ROOT)} is empty"
    return text


def test_required_documents_exist_and_are_non_empty() -> None:
    for path in (DESIGN, DECISIONS, PROGRESS):
        assert path.is_file(), f"missing {path.relative_to(ROOT)}"
        assert _read(path).strip(), f"{path.relative_to(ROOT)} is empty"


def test_architecture_note_records_the_deterministic_core() -> None:
    text = _read(DESIGN).lower()

    assert "deterministic" in text
    assert "control flow" in text
    assert "llms run *inside* stages" in text
    assert "validated structured output" in text


def test_architecture_note_records_core_independence() -> None:
    text = _read(DESIGN).lower()

    assert "core independence" in text
    assert "the core never imports a module" in text


def test_fixtures_policy_records_responses_per_provider() -> None:
    text = _read(DESIGN).lower()

    assert "recorded responses" in text
    assert "per provider" in text
    assert "tests/fixtures/providers/" in text


def test_fixtures_policy_records_one_example_per_schema() -> None:
    text = _read(DESIGN).lower()

    assert "one example per schema" in text
    assert "examples/" in text


def test_fixtures_policy_records_the_goldens_folder() -> None:
    text = _read(DESIGN).lower()

    assert "goldens folder" in text
    assert "tests/golden/" in text


def test_fixtures_policy_forbids_real_personal_data() -> None:
    text = _read(DESIGN).lower()

    assert "no real personal data" in text
    assert "synthetic" in text
    assert "redacted" in text


def test_decisions_log_records_deterministic_core_and_core_independence() -> None:
    text = _read(DECISIONS).lower()

    assert "d-003 · deterministic core, llm at the edges" in text
    assert "d-006 · core never imports a module" in text


def test_decisions_log_records_the_fixtures_policy() -> None:
    text = _read(DECISIONS).lower()

    assert "d-023 · fixtures are recorded, examples synthetic" in text
    assert "synthetic data only" in text


def test_progress_log_records_deterministic_core_and_core_independence() -> None:
    text = _read(PROGRESS).lower()

    assert "deterministic core" in text
    assert "no module imports in core" in text
    assert "validated structured output" in text
    assert "research" in text
    assert "orders" in text
    assert "publish" in text
    assert "analytics" in text


def test_documents_cross_reference_each_other() -> None:
    design = _read(DESIGN)
    decisions = _read(DECISIONS)
    progress = _read(PROGRESS)

    assert "docs/decisions.md" in design
    assert "docs/PROGRESS.md" in design
    assert "docs/DESIGN.md" in decisions
    assert "docs/DESIGN.md" in progress
    assert "docs/decisions.md" in progress


def test_adding_provider_guide_exists_and_covers_registration() -> None:
    text = _guide("provider").lower()

    assert "pick a capability" in text
    assert "subclass the provider base" in text
    assert "register the provider type" in text
    assert "register_provider" in text
    assert "dotted path" in text
    assert "configure a provider instance" in text
    assert "secret_refs" in text
    assert "secrets loader" in text
    assert "wire pricing and models" in text
    assert "models.yaml" in text
    assert "pricing_overrides" in text
    assert "contract suite" in text
    assert "business logic calls a provider directly" in text


def test_adding_language_guide_exists_and_covers_config() -> None:
    text = _guide("language").lower()

    assert "declare the language" in text
    assert "languages" in text
    assert "words_per_second" in text
    assert "subtitle_font" in text
    assert "hashtag_style" in text
    assert "forbidden_phrases" in text
    assert "pick a default" in text
    assert "languages.default" in text
    assert "writing direction" in text
    assert "rtl" in text
    assert "bundled font" in text
    assert "ofl" in text


def test_adding_format_guide_exists_and_covers_recipe() -> None:
    text = _guide("format").lower()

    assert "declare the recipe" in text
    assert "stages" in text
    assert "capabilities" in text
    assert "output_type" in text
    assert "handler" in text
    assert "save" in text
    assert "publish" in text
    assert "deliver" in text
    assert "per-stage options and retries" in text
    assert "timeout_seconds" in text
    assert "retry_policy" in text
    assert "user and packaged formats" in text
    assert "every stage input" in text


def test_adding_platform_guide_exists_and_covers_data_file() -> None:
    text = _guide("platform").lower()

    assert "platforms.yaml" in text
    assert "declare the platform" in text
    assert "media_types" in text
    assert "limits" in text
    assert "resolutions" in text
    assert "aspect_ratios" in text
    assert "safe_area" in text
    assert "disclosure" in text
    assert "bind accounts" in text
    assert "platform/language" in text
    assert "output validation" in text


def test_adding_occasion_guide_exists_and_covers_variables() -> None:
    text = _guide("occasion").lower()

    assert "declare the occasion" in text
    assert "occasions" in text
    assert "default" in text
    assert "enabled" in text
    assert "declare variables" in text
    assert "min_length" in text
    assert "max_length" in text
    assert "personalization levels" in text
    assert "price" in text
    assert "per-language content" in text
    assert "lyrics" in text
    assert "speech" in text


def test_adding_character_guide_exists_and_covers_opt_in() -> None:
    text = _guide("character").lower()

    assert "create the character file" in text
    assert "reference_images" in text
    assert "pose_library" in text
    assert "configure the character machinery" in text
    assert "consistency_strategies" in text
    assert "prompt_only" in text
    assert "qa" in text
    assert "consent" in text
    assert "select the character" in text
    assert "opt-in" in text
    assert "version" in text
    assert "voice cloning" in text
