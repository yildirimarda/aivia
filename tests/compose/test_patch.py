"""Composition patches validate, apply and undo back to the original hash."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from aivia.compose import (
    AppliedPatch,
    Composition,
    CompositionPatch,
    PatchError,
    PatchOperation,
    PatchValidationError,
    apply_patch,
    apply_patch_reversible,
    composition_hash,
    invert_patch,
    patch_from_json,
    patch_to_json,
    undo_patch,
    validate_patch,
)


def _composition(**overrides: Any) -> Composition:
    values: dict[str, Any] = {
        "id": "demo",
        "canvas": "640x360",
        "duration_seconds": 5.0,
        "fps": 24,
        "layers": [
            {"id": "background", "type": "image", "source": "bg.png"},
            {
                "id": "title",
                "type": "text",
                "text": "Hello",
                "x": 10,
                "y": 20,
                "font_size": 24,
                "start": 0.5,
                "duration": 2.0,
            },
        ],
        "audio_tracks": [
            {"id": "music", "source": "music.mp3", "volume_db": -3.0}
        ],
        "effects": [{"id": "fade", "type": "fade", "start": 0, "end": 1}],
    }
    values.update(overrides)
    return Composition.model_validate(values)


def _layer(composition: Composition, layer_id: str) -> Any:
    layer = composition.layer(layer_id)
    assert layer is not None
    return layer


def test_a_patch_can_address_layers_by_index_or_id() -> None:
    composition = _composition()
    patch = CompositionPatch(
        [
            {"op": "set", "path": "/layers/title/opacity", "value": 0.4},
            {"op": "set", "path": "/layers/0/x", "value": 42},
        ]
    )

    result = patch.apply(composition)

    assert _layer(result, "title").opacity == 0.4
    assert _layer(result, "background").model_extra["x"] == 42
    assert _layer(composition, "title").opacity == 1.0
    assert _layer(composition, "background").model_extra.get("x") is None


def test_every_operation_applies_and_its_inverse_restores_the_hash() -> None:
    composition = _composition()
    patch = CompositionPatch(
        [
            {"op": "test", "path": "/fps", "value": 24},
            {"op": "replace", "path": "/fps", "value": 30},
            {"op": "set", "path": "/layers/title/opacity", "value": 0.25},
            {
                "op": "add",
                "path": "/layers/-",
                "value": {
                    "id": "cta",
                    "type": "text",
                    "text": "Watch now",
                    "start": 1,
                    "duration": 1,
                },
            },
            {"op": "remove", "path": "/audio_tracks/music"},
            {"op": "move", "from": "/layers/title", "path": "/layers/0"},
            {"op": "copy", "from": "/effects/fade", "path": "/effects/0"},
            {"op": "set", "path": "/effects/0/id", "value": "fade-copy"},
        ]
    )

    applied = patch.apply_reversible(composition)

    assert applied.composition.fps == 30
    assert applied.composition.layer("cta") is not None
    assert applied.composition.track("music") is None
    assert next(layer.id for layer in applied.composition.layers) == "title"
    assert [effect.id for effect in applied.composition.effects] == [
        "fade-copy",
        "fade",
    ]
    assert applied.composition_hash != applied.original_hash

    restored = applied.undo()

    assert composition_hash(restored) == composition_hash(composition)
    assert restored == composition


def test_apply_then_undo_on_the_patch_restores_the_original_hash() -> None:
    composition = _composition()
    patch = CompositionPatch(
        [{"op": "set", "path": "/canvas/background", "value": "#112233"}]
    )

    patched = patch.apply(composition)
    restored = patch.undo(patched)

    assert composition_hash(patched) != composition_hash(composition)
    assert composition_hash(restored) == composition_hash(composition)
    assert restored == composition


def test_an_inverse_patch_can_be_applied_by_itself() -> None:
    composition = _composition()
    patch = CompositionPatch(
        [
            {"op": "remove", "path": "/layers/background"},
            {"op": "set", "path": "/layers/title/text", "value": "Changed"},
        ]
    )

    patched = apply_patch(composition, patch)
    inverse = invert_patch(composition, patch)
    restored = undo_patch(patched, inverse)

    assert composition_hash(restored) == composition_hash(composition)


def test_a_test_operation_does_not_change_the_composition() -> None:
    composition = _composition()
    patch = CompositionPatch([{"op": "test", "path": "/id", "value": "demo"}])

    result = patch.apply(composition)

    assert result == composition
    assert composition_hash(result) == composition_hash(composition)
    assert patch.inverse_for(composition).operations == []


def test_a_failed_test_reports_the_path_and_values() -> None:
    composition = _composition()
    patch = CompositionPatch([{"op": "test", "path": "/fps", "value": 60}])

    with pytest.raises(PatchError, match="test failed at '/fps'"):
        patch.apply(composition)


def test_missing_paths_and_out_of_range_indexes_are_rejected() -> None:
    composition = _composition()

    with pytest.raises(PatchError, match="does not exist"):
        CompositionPatch(
            [{"op": "replace", "path": "/layers/missing/opacity", "value": 0.5}]
        ).apply(composition)

    with pytest.raises(PatchError, match="out of range"):
        CompositionPatch(
            [{"op": "add", "path": "/layers/99", "value": {"id": "x", "type": "text"}}]
        ).apply(composition)


def test_operations_cannot_escape_into_their_own_descendants() -> None:
    composition = _composition()

    with pytest.raises(PatchError, match="descendants"):
        CompositionPatch(
            [{"op": "move", "from": "/layers/0", "path": "/layers/0/x"}]
        ).apply(composition)


def test_a_patch_result_must_still_be_a_valid_composition() -> None:
    composition = _composition()

    with pytest.raises(PatchValidationError) as raised:
        CompositionPatch(
            [{"op": "set", "path": "/layers/title/opacity", "value": 2.0}]
        ).apply(composition)

    assert raised.value.issues
    assert raised.value.issues[0].path.startswith("layers")


def test_adding_a_duplicate_id_fails_composition_validation() -> None:
    composition = _composition()

    with pytest.raises(PatchValidationError, match="duplicate"):
        CompositionPatch(
            [
                {
                    "op": "add",
                    "path": "/layers/-",
                    "value": {"id": "title", "type": "text", "text": "again"},
                }
            ]
        ).apply(composition)


def test_operations_require_their_mandatory_fields() -> None:
    with pytest.raises(ValidationError, match="requires a value"):
        CompositionPatch([{"op": "replace", "path": "/fps"}])

    with pytest.raises(ValidationError, match="requires a 'from' path"):
        CompositionPatch([{"op": "move", "path": "/layers/0"}])

    with pytest.raises(ValidationError, match="unknown patch operation"):
        CompositionPatch([{"op": "explode", "path": "/fps", "value": 1}])


def test_patches_round_trip_through_json() -> None:
    composition = _composition()
    patch = CompositionPatch(
        [
            {"op": "replace", "path": "/fps", "value": 30},
            {"op": "move", "from": "/layers/1", "path": "/layers/0"},
        ],
        description="retime and reorder",
    )

    restored = patch_from_json(patch_to_json(patch))

    assert restored == patch
    assert restored.apply(composition) == patch.apply(composition)
    assert patch_to_json(restored) == patch_to_json(patch)


def test_a_plain_operation_mapping_and_a_list_are_accepted() -> None:
    composition = _composition()
    operation = {"op": "set", "path": "/fps", "value": 25}

    from_mapping = CompositionPatch(operation)
    from_list = CompositionPatch([operation])

    assert from_mapping.operations == from_list.operations
    assert from_mapping.operations[0] == PatchOperation.model_validate(operation)
    assert apply_patch(composition, operation).fps == 25
    assert apply_patch(composition, [operation]).fps == 25


def test_validate_patch_returns_the_result_without_changing_the_input() -> None:
    composition = _composition()
    patch = CompositionPatch([{"op": "set", "path": "/fps", "value": 60}])

    result = validate_patch(composition, patch)

    assert result.fps == 60
    assert composition.fps == 24


def test_reversible_result_exposes_both_hashes() -> None:
    composition = _composition()
    applied = apply_patch_reversible(
        composition,
        [{"op": "set", "path": "/fps", "value": 60}],
    )

    assert isinstance(applied, AppliedPatch)
    assert applied.result is applied.composition
    assert applied.patched is applied.composition
    assert applied.hash == applied.patched_hash
    assert applied.original_hash == composition_hash(composition)


def test_undo_patch_uses_the_inverse_recorded_by_apply() -> None:
    composition = _composition()
    patch = CompositionPatch([{"op": "set", "path": "/fps", "value": 60}])

    patched = apply_patch(composition, patch)
    restored = undo_patch(patched, patch)

    assert composition_hash(restored) == composition_hash(composition)


def test_a_patch_can_be_built_from_another_patch() -> None:
    patch = CompositionPatch(
        [{"op": "replace", "path": "/fps", "value": 30}],
        description="copy me",
    )

    cloned = CompositionPatch(patch)

    assert cloned == patch
    assert cloned.operations == patch.operations
