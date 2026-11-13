"""``aivia run --estimate``: pre-run stage estimates."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivia.cli import main
from aivia.core.config.formats import FormatRecipeConfig, StageConfig
from aivia.core.errors import ConfigError
from aivia.core.estimate import (
    EstimateSource,
    Estimator,
    estimate_recipe,
    stage_capability,
    stage_units,
)
from aivia.core.models.provider import ProviderMeta, ProviderPrice


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip(), encoding="utf-8")
    return path


def estimate(
    config: Path,
    capsys: pytest.CaptureFixture[str],
    *extra: str,
) -> tuple[int, list[list[str]], str]:
    code = main(
        [
            "run",
            "--format",
            "estimate_sample",
            "--estimate",
            "--config",
            str(config),
            *extra,
        ]
    )
    captured = capsys.readouterr()
    rows = [line.split("\t") for line in captured.out.splitlines()]
    return code, rows, captured.err


@pytest.fixture
def priced_config(tmp_path: Path) -> Path:
    return write_config(
        tmp_path,
        """
providers:
  text:
    type: mock-llm
  speech:
    type: mock-tts
  pictures:
    type: mock-image
formats:
  estimate_sample:
    description: A priced sample recipe.
    stages:
      - stage: input_topic
      - stage: write
      - stage: voice
      - stage: visuals
        options:
          strategy: image
          count: 2
      - stage: render
    capabilities: [llm, tts, image_gen]
    output_type: video
    handler: save
""",
    )


def test_estimate_prints_one_row_per_stage_and_exits(
    priced_config: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, rows, err = estimate(priced_config, capsys)

    assert code == 0
    assert err == ""
    assert rows[0] == ["Format: estimate_sample"]
    assert rows[1] == [
        "STAGE",
        "CAPABILITY",
        "PROVIDER",
        "UNITS",
        "UNIT",
        "ESTIMATE",
        "NOTE",
    ]
    stages = [row[0] for row in rows[2:-1]]
    assert stages == ["input_topic", "write", "voice", "visuals", "render"]

    by_stage = {row[0]: row for row in rows[2:-1]}
    assert by_stage["input_topic"][5] == "$0.000000"
    assert by_stage["input_topic"][6] == "no paid call"
    assert by_stage["write"][1:3] == ["llm", "text"]
    assert by_stage["voice"][1:3] == ["tts", "speech"]
    assert by_stage["visuals"][1:3] == ["image_gen", "pictures"]
    assert by_stage["visuals"][3] == "2"
    assert by_stage["visuals"][5] == "$0.040000"
    assert by_stage["render"][5] == "$0.000000"

    total = rows[-1]
    assert total[0] == "TOTAL"
    assert total[5] == "$0.040510"


def test_estimate_honours_the_configured_llm_chain(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(
        tmp_path,
        """
providers:
  outside:
    type: mock-llm
  cheap:
    type: mock-llm
  dear:
    type: mock-llm
llm:
  chain: [dear, cheap]
formats:
  estimate_sample:
    stages:
      - stage: write
    capabilities: [llm]
""",
    )

    code, rows, err = estimate(config, capsys)

    assert code == 0
    assert err == ""
    assert rows[2][1:3] == ["llm", "dear"]


def test_estimate_uses_registered_mocks_when_no_provider_is_configured(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(
        tmp_path,
        """
formats:
  estimate_sample:
    stages:
      - stage: write
    capabilities: [llm]
""",
    )

    code, rows, err = estimate(config, capsys)

    assert code == 0
    assert err == ""
    assert rows[2][1] == "llm"
    assert rows[2][2] == "mock"
    assert rows[2][5] == "$0.000500"


def test_estimate_fails_when_a_paid_stage_has_no_provider(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(
        tmp_path,
        """
providers:
  text:
    type: mock-llm
formats:
  estimate_sample:
    stages:
      - stage: music
    capabilities: [music_gen]
""",
    )

    code, rows, err = estimate(config, capsys)

    assert code == 1
    assert rows[2][0] == "music"
    assert rows[2][1] == "music_gen"
    assert rows[2][2] == "-"
    assert rows[2][6] == "no provider for music_gen"
    assert "cannot estimate stage 'music'" in err


def test_estimate_rejects_an_unknown_format(
    priced_config: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "run",
            "--format",
            "missing",
            "--estimate",
            "--config",
            str(priced_config),
        ]
    )
    captured = capsys.readouterr()

    assert code == 1
    assert captured.out == ""
    assert "unknown format: missing" in captured.err
    assert "estimate_sample" in captured.err


def test_run_without_estimate_enqueues_instead_of_estimating_only(
    priced_config: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "run",
            "--format",
            "estimate_sample",
            "--enqueue",
            "--config",
            str(priced_config),
        ]
    )
    captured = capsys.readouterr()

    assert code == 0
    assert captured.err == ""
    assert "Run: run_" in captured.out
    assert "Estimate: $0.040510" in captured.out
    assert "State: queued" in captured.out


def test_estimator_picks_the_cheapest_priced_provider() -> None:
    recipe = FormatRecipeConfig.model_validate({"stages": [{"stage": "write"}]})
    providers = {
        "dear": ProviderMeta(
            name="dear",
            capabilities=["llm"],
            pricing={"llm": ProviderPrice(amount_usd=0.2)},
        ),
        "cheap": ProviderMeta(
            name="cheap",
            capabilities=["llm"],
            pricing={"llm": ProviderPrice(amount_usd=0.1)},
        ),
    }

    result = Estimator(providers).estimate("sample", recipe)

    assert result.stages[0].provider == "cheap"
    assert result.stages[0].amount_usd == pytest.approx(0.1)
    assert result.total_usd == pytest.approx(0.1)


def test_estimator_marks_a_stage_without_a_price_as_missing() -> None:
    recipe = FormatRecipeConfig.model_validate({"stages": [{"stage": "write"}]})
    providers = {
        "unpriced": ProviderMeta(name="unpriced", capabilities=["llm"]),
    }

    result = Estimator(providers).estimate("sample", recipe)

    stage = result.stages[0]
    assert stage.missing is True
    assert stage.amount_usd == 0.0
    assert stage.source is EstimateSource.MISSING
    assert "no price for llm" in stage.note
    assert result.complete is False


def test_explicit_stage_estimate_overrides_provider_pricing() -> None:
    recipe = FormatRecipeConfig.model_validate(
        {
            "stages": [
                {
                    "stage": "custom",
                    "options": {"capability": "llm", "estimate_usd": 1.25},
                }
            ]
        }
    )
    providers = {
        "cheap": ProviderMeta(
            name="cheap",
            capabilities=["llm"],
            pricing={"llm": ProviderPrice(amount_usd=0.1)},
        ),
    }

    result = estimate_recipe("sample", recipe, providers)

    stage = result.stages[0]
    assert stage.provider == "override"
    assert stage.source is EstimateSource.OVERRIDE
    assert stage.amount_usd == pytest.approx(1.25)
    assert result.total_usd == pytest.approx(1.25)


def test_visuals_stage_selects_image_video_or_stock_from_its_strategy() -> None:
    recipe = FormatRecipeConfig.model_validate(
        {"stages": [], "capabilities": ["video_gen"]}
    )
    stages = [
        StageConfig.model_validate(
            {"stage": "visuals", "options": {"strategy": "image"}}
        ),
        StageConfig.model_validate(
            {"stage": "visuals", "options": {"strategy": "video"}}
        ),
        StageConfig.model_validate(
            {"stage": "visuals", "options": {"strategy": "stock"}}
        ),
        StageConfig.model_validate({"stage": "visuals"}),
    ]

    capabilities = [stage_capability(stage, recipe) for stage in stages]

    assert capabilities == ["image_gen", "video_gen", "stock", "video_gen"]


def test_stage_units_reads_explicit_counts_and_durations() -> None:
    count = StageConfig.model_validate({"stage": "visuals", "options": {"count": 3}})
    duration = StageConfig.model_validate(
        {"stage": "voice", "options": {"seconds": 12.5}}
    )

    assert stage_units(count) == (3.0, "")
    assert stage_units(duration) == (12.5, "seconds")


def test_estimator_rejects_an_invalid_stage_quantity() -> None:
    recipe = FormatRecipeConfig.model_validate(
        {"stages": [{"stage": "voice", "options": {"seconds": -1}}]}
    )

    with pytest.raises(ConfigError, match="finite, non-negative"):
        Estimator({}).estimate("sample", recipe)


def test_estimate_splits_hybrid_visuals_into_capped_video_and_image_beats(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = write_config(
        tmp_path,
        """
providers:
  pictures:
    type: mock-image
  motion:
    type: mock-video
formats:
  estimate_sample:
    description: Hybrid visuals priced per beat type.
    stages:
      - stage: visuals
        options:
          strategy: hybrid
          count: 4
          max_ai_video_beats: 2
    capabilities: [image_gen, video_gen]
    output_type: video
    handler: save
""",
    )

    code, rows, err = estimate(config, capsys)

    assert code == 0
    assert err == ""
    assert rows[0] == ["Format: estimate_sample"]
    assert rows[1] == [
        "STAGE",
        "CAPABILITY",
        "PROVIDER",
        "UNITS",
        "UNIT",
        "ESTIMATE",
        "NOTE",
    ]
    visuals = [row for row in rows[2:-1] if row[0] == "visuals"]
    assert len(visuals) == 2
    by_capability = {row[1]: row for row in visuals}
    assert by_capability["video_gen"][2] == "motion"
    assert by_capability["video_gen"][3] == "2"
    assert by_capability["video_gen"][5] == "$0.100000"
    assert by_capability["image_gen"][2] == "pictures"
    assert by_capability["image_gen"][3] == "2"
    assert by_capability["image_gen"][5] == "$0.040000"

    total = rows[-1]
    assert total[0] == "TOTAL"
    assert total[5] == "$0.140000"


def test_estimator_splits_hybrid_visuals_into_capped_video_and_image_beats() -> None:
    recipe = FormatRecipeConfig.model_validate(
        {
            "stages": [
                {
                    "stage": "visuals",
                    "options": {
                        "strategy": "hybrid",
                        "count": 5,
                        "max_ai_video_beats": 2,
                    },
                }
            ],
            "capabilities": ["image_gen", "video_gen"],
        }
    )
    providers = {
        "images": ProviderMeta(
            name="images",
            capabilities=["image_gen"],
            pricing={"image_gen": ProviderPrice(amount_usd=0.02)},
        ),
        "clips": ProviderMeta(
            name="clips",
            capabilities=["video_gen"],
            pricing={"video_gen": ProviderPrice(amount_usd=0.05)},
        ),
    }

    result = Estimator(providers).estimate("sample", recipe)

    assert [stage.stage for stage in result.stages] == ["visuals", "visuals"]
    by_capability = {stage.capability: stage for stage in result.stages}
    assert by_capability["video_gen"].provider == "clips"
    assert by_capability["video_gen"].units == 2.0
    assert by_capability["video_gen"].amount_usd == pytest.approx(0.10)
    assert by_capability["image_gen"].provider == "images"
    assert by_capability["image_gen"].units == 3.0
    assert by_capability["image_gen"].amount_usd == pytest.approx(0.06)
    assert result.total_usd == pytest.approx(0.16)
    assert result.complete is True


def test_estimate_uses_image_gen_when_video_cap_is_zero() -> None:
    recipe = FormatRecipeConfig.model_validate(
        {
            "stages": [
                {
                    "stage": "visuals",
                    "options": {
                        "strategy": "hybrid",
                        "count": 3,
                        "max_ai_video_beats": 0,
                    },
                }
            ],
            "capabilities": ["image_gen", "video_gen"],
        }
    )
    providers = {
        "images": ProviderMeta(
            name="images",
            capabilities=["image_gen"],
            pricing={"image_gen": ProviderPrice(amount_usd=0.02)},
        ),
    }

    result = Estimator(providers).estimate("sample", recipe)

    assert [stage.stage for stage in result.stages] == ["visuals"]
    stage = result.stages[0]
    assert stage.capability == "image_gen"
    assert stage.units == 3.0
    assert stage.amount_usd == pytest.approx(0.06)
    assert result.total_usd == pytest.approx(0.06)


def test_estimate_prices_all_beats_as_video_when_cap_exceeds_count() -> None:
    recipe = FormatRecipeConfig.model_validate(
        {
            "stages": [
                {
                    "stage": "visuals",
                    "options": {
                        "strategy": "hybrid",
                        "count": 2,
                        "max_ai_video_beats": 5,
                    },
                }
            ],
            "capabilities": ["image_gen", "video_gen"],
        }
    )
    providers = {
        "clips": ProviderMeta(
            name="clips",
            capabilities=["video_gen"],
            pricing={"video_gen": ProviderPrice(amount_usd=0.05)},
        ),
    }

    result = Estimator(providers).estimate("sample", recipe)

    assert [stage.stage for stage in result.stages] == ["visuals"]
    stage = result.stages[0]
    assert stage.capability == "video_gen"
    assert stage.units == 2.0
    assert stage.amount_usd == pytest.approx(0.10)
    assert result.total_usd == pytest.approx(0.10)
