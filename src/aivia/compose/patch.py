"""A validated, reversible patch language for composition edits.

A patch is an ordered list of small operations addressed by a JSON Pointer or
a dotted path. Operations are applied to a deep copy of the composition, the
result is validated as a whole composition, and every operation records the
inverse operation that restores the previous value. Applying the generated
inverse patch therefore restores the original document exactly, so its
:func:`~aivia.compose.render_cache.composition_hash` is unchanged.

The language intentionally mirrors the useful subset of JSON Patch while
remaining composition aware:

``add``
    Insert a value into an object or list. A list may use an index, ``-`` for
    append, or an element id such as ``title``.
``remove``
    Remove the value at a path.
``replace``
    Replace an existing value.
``set``
    Replace an existing value or create a missing object key or list tail.
``move``
    Move a value from ``from`` to ``path``.
``copy``
    Copy a value from ``from`` to ``path``.
``test``
    Assert that a path has a value without changing the document.

Paths are JSON Pointers (``/layers/0/opacity``) or dotted paths
(``layers.title.opacity``). List tokens may also be element ids, which keeps
patches readable when layers are reordered.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from aivia.compose.io import composition_to_dict
from aivia.compose.render_cache import composition_hash
from aivia.compose.schema import Composition
from aivia.compose.validation import (
    AssetCheck,
    AssetLookup,
    CompositionIssue,
    CompositionIssues,
    validate_composition,
)

__all__ = [
    "AppliedPatch",
    "CompositionEdit",
    "CompositionPatch",
    "EditOperation",
    "Operation",
    "PatchError",
    "PatchOperation",
    "PatchResult",
    "PatchValidationError",
    "apply_composition_patch",
    "apply_patch",
    "apply_patch_reversible",
    "dump_patch",
    "invert_patch",
    "load_patch",
    "parse_patch",
    "patch_from_dict",
    "patch_from_json",
    "patch_to_dict",
    "patch_to_json",
    "revert_patch",
    "undo_patch",
    "validate_patch",
]

_OP_ALIASES = {
    "add": "add",
    "append": "add",
    "insert": "add",
    "assert": "test",
    "copy": "copy",
    "delete": "remove",
    "duplicate": "copy",
    "edit": "set",
    "move": "move",
    "recolor": "set",
    "recolour": "set",
    "remove": "remove",
    "rename": "move",
    "replace": "replace",
    "resize": "set",
    "retime": "set",
    "set": "set",
    "test": "test",
    "update": "set",
}

_BRACKET = re.compile(r"\[(?P<key>[^\]]*)\]")


class PatchError(ValueError):
    """A patch is malformed or cannot be applied to a composition."""

    code = "patch_invalid"


class PatchValidationError(PatchError):
    """Applying a patch produced an invalid composition."""

    code = "patch_result_invalid"

    def __init__(self, issues: Sequence[CompositionIssue]) -> None:
        self.issues: tuple[CompositionIssue, ...] = tuple(issues)
        message = "patched composition is invalid: " + "; ".join(
            str(issue) for issue in issues
        )
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Return the issues as a JSON-compatible mapping."""
        return {"issues": [issue.to_dict() for issue in self.issues]}


def _normalise_op(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("patch operation must be a string")
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _OP_ALIASES[key]
    except KeyError:
        raise ValueError(f"unknown patch operation {value!r}") from None


def _tokenize(path: str) -> tuple[str, ...]:
    """Return the path as a tuple of mapping keys or list tokens."""
    if not isinstance(path, str):
        raise PatchError("patch path must be a string")
    text = path.strip()
    if text in {"", "/", "$"}:
        return ()
    if text.startswith("$"):
        text = text[1:]
        if text.startswith("."):
            text = text[1:]
    if "/" in text:
        parts = text.split("/")
        if parts and parts[0] == "":
            parts = parts[1:]
        return tuple(part.replace("~1", "/").replace("~0", "~") for part in parts)
    text = _BRACKET.sub(lambda match: "." + match.group("key").strip("'\""), text)
    return tuple(part for part in text.split(".") if part != "")


def _path_from_tokens(tokens: Sequence[str]) -> str:
    if not tokens:
        return ""
    escaped = [str(token).replace("~", "~0").replace("/", "~1") for token in tokens]
    return "/" + "/".join(escaped)


def _is_descendant(path: Sequence[str], source: Sequence[str]) -> bool:
    return len(path) > len(source) and tuple(path[: len(source)]) == tuple(source)


def _list_index(
    container: Sequence[Any],
    token: str,
    path: str,
    *,
    for_add: bool,
) -> int:
    if token == "-":
        if for_add:
            return len(container)
        raise PatchError(f"' - ' is only valid when adding at {path!r}")
    if token.isdigit():
        index = int(token)
        limit = len(container) if for_add else len(container) - 1
        if index > limit:
            raise PatchError(f"index {index} is out of range at {path!r}")
        return index
    for index, item in enumerate(container):
        if isinstance(item, Mapping) and str(item.get("id", "")) == token:
            return index
    raise PatchError(f"path {path!r} does not exist: no list element {token!r}")


def _locate(
    document: Any,
    tokens: Sequence[str],
    path: str,
    *,
    for_add: bool = False,
) -> tuple[Any, str | int, tuple[str, ...], bool]:
    """Return the parent, key, normalised path and whether the key exists."""
    if not tokens:
        raise PatchError(f"path {path!r} does not address a child value")
    current = document
    normalised: list[str] = []
    for position, token in enumerate(tokens):
        last = position == len(tokens) - 1
        if isinstance(current, Mapping):
            normalised.append(token)
            exists = token in current
            if last:
                return current, token, tuple(normalised), exists
            if not exists:
                raise PatchError(f"path {path!r} does not exist")
            current = current[token]
        elif isinstance(current, list):
            index = _list_index(
                current,
                token,
                path,
                for_add=for_add and last,
            )
            normalised.append(str(index))
            exists = index < len(current)
            if last:
                return current, index, tuple(normalised), exists
            if not exists:
                raise PatchError(f"path {path!r} does not exist")
            current = current[index]
        else:
            raise PatchError(f"cannot traverse {type(current).__name__} at {path!r}")
    raise PatchError(f"path {path!r} does not address a child value")


def _get(document: Any, tokens: Sequence[str], path: str) -> Any:
    if not tokens:
        return document
    parent, key, _, exists = _locate(document, tokens, path)
    if not exists:
        raise PatchError(f"path {path!r} does not exist")
    return parent[key]


def _replace_value(
    document: Any,
    tokens: Sequence[str],
    path: str,
    value: Any,
) -> tuple[Any, list[PatchOperation]]:
    value = copy.deepcopy(value)
    if not tokens:
        old = copy.deepcopy(document)
        return value, [PatchOperation(op="replace", path="", value=old)]
    parent, key, normalised, exists = _locate(document, tokens, path)
    if not exists:
        raise PatchError(f"path {path!r} does not exist")
    old = copy.deepcopy(parent[key])
    parent[key] = value
    inverse = PatchOperation(
        op="replace",
        path=_path_from_tokens(normalised),
        value=old,
    )
    return document, [inverse]


def _set_value(
    document: Any,
    tokens: Sequence[str],
    path: str,
    value: Any,
) -> tuple[Any, list[PatchOperation]]:
    value = copy.deepcopy(value)
    if not tokens:
        old = copy.deepcopy(document)
        return value, [PatchOperation(op="replace", path="", value=old)]
    parent, key, normalised, exists = _locate(
        document,
        tokens,
        path,
        for_add=True,
    )
    if exists:
        old = copy.deepcopy(parent[key])
        parent[key] = value
        inverse = PatchOperation(
            op="replace",
            path=_path_from_tokens(normalised),
            value=old,
        )
        return document, [inverse]
    if isinstance(parent, list):
        assert isinstance(key, int)
        parent.insert(key, value)
    else:
        parent[key] = value
    return document, [PatchOperation(op="remove", path=_path_from_tokens(normalised))]


def _add_value(
    document: Any,
    tokens: Sequence[str],
    path: str,
    value: Any,
) -> tuple[Any, list[PatchOperation]]:
    value = copy.deepcopy(value)
    if not tokens:
        old = copy.deepcopy(document)
        return value, [PatchOperation(op="replace", path="", value=old)]
    parent, key, normalised, exists = _locate(
        document,
        tokens,
        path,
        for_add=True,
    )
    if isinstance(parent, list):
        assert isinstance(key, int)
        parent.insert(key, value)
        inverse = PatchOperation(op="remove", path=_path_from_tokens(normalised))
        return document, [inverse]
    old = copy.deepcopy(parent[key]) if exists else None
    parent[key] = value
    if exists:
        inverse = PatchOperation(
            op="replace",
            path=_path_from_tokens(normalised),
            value=old,
        )
    else:
        inverse = PatchOperation(op="remove", path=_path_from_tokens(normalised))
    return document, [inverse]


def _remove_value(
    document: Any,
    tokens: Sequence[str],
    path: str,
) -> tuple[Any, list[PatchOperation]]:
    if not tokens:
        old = copy.deepcopy(document)
        return None, [PatchOperation(op="add", path="", value=old)]
    parent, key, normalised, exists = _locate(document, tokens, path)
    if not exists:
        raise PatchError(f"path {path!r} does not exist")
    old = parent.pop(key)
    inverse = PatchOperation(
        op="add",
        path=_path_from_tokens(normalised),
        value=copy.deepcopy(old),
    )
    return document, [inverse]


def _move_value(
    document: Any,
    tokens: Sequence[str],
    path: str,
    source_path: str,
) -> tuple[Any, list[PatchOperation]]:
    source_tokens = _tokenize(source_path)
    if not tokens or not source_tokens:
        raise PatchError("cannot move the composition root")
    if _is_descendant(tokens, source_tokens):
        raise PatchError("cannot move a value into one of its own descendants")
    source_parent, source_key, source_normalised, exists = _locate(
        document,
        source_tokens,
        source_path,
    )
    if not exists:
        raise PatchError(f"path {source_path!r} does not exist")
    value = copy.deepcopy(source_parent[source_key])
    del source_parent[source_key]

    dest_parent, dest_key, dest_normalised, dest_exists = _locate(
        document,
        tokens,
        path,
        for_add=True,
    )
    old_dest = (
        copy.deepcopy(dest_parent[dest_key])
        if dest_exists and isinstance(dest_parent, Mapping)
        else None
    )
    if isinstance(dest_parent, list):
        assert isinstance(dest_key, int)
        dest_parent.insert(dest_key, value)
    else:
        dest_parent[dest_key] = value

    inverse = [
        PatchOperation(
            op="move",
            path=_path_from_tokens(source_normalised),
            from_=_path_from_tokens(dest_normalised),
        )
    ]
    if dest_exists and isinstance(dest_parent, Mapping):
        inverse.append(
            PatchOperation(
                op="add",
                path=_path_from_tokens(dest_normalised),
                value=old_dest,
            )
        )
    return document, inverse


def _copy_value(
    document: Any,
    tokens: Sequence[str],
    path: str,
    source_path: str,
) -> tuple[Any, list[PatchOperation]]:
    source_tokens = _tokenize(source_path)
    if not tokens or not source_tokens:
        raise PatchError("cannot copy the composition root")
    if _is_descendant(tokens, source_tokens):
        raise PatchError("cannot copy a value into one of its own descendants")
    value = copy.deepcopy(_get(document, source_tokens, source_path))

    parent, key, normalised, exists = _locate(
        document,
        tokens,
        path,
        for_add=True,
    )
    old = copy.deepcopy(parent[key]) if exists and isinstance(parent, Mapping) else None
    if isinstance(parent, list):
        assert isinstance(key, int)
        parent.insert(key, value)
        inverse = [PatchOperation(op="remove", path=_path_from_tokens(normalised))]
    else:
        parent[key] = value
        if exists:
            inverse = [
                PatchOperation(
                    op="replace",
                    path=_path_from_tokens(normalised),
                    value=old,
                )
            ]
        else:
            inverse = [PatchOperation(op="remove", path=_path_from_tokens(normalised))]
    return document, inverse


def _apply_one(
    document: Any,
    operation: PatchOperation,
) -> tuple[Any, list[PatchOperation]]:
    tokens = _tokenize(operation.path)
    if operation.op == "test":
        actual = _get(document, tokens, operation.path)
        if actual != operation.value:
            raise PatchError(
                f"test failed at {operation.path!r}: "
                f"expected {operation.value!r}, found {actual!r}"
            )
        return document, []
    if operation.op == "add":
        return _add_value(document, tokens, operation.path, operation.value)
    if operation.op == "remove":
        return _remove_value(document, tokens, operation.path)
    if operation.op == "replace":
        return _replace_value(document, tokens, operation.path, operation.value)
    if operation.op == "set":
        return _set_value(document, tokens, operation.path, operation.value)
    if operation.op == "move":
        if operation.from_ is None:
            raise PatchError("move requires a 'from' path")
        return _move_value(document, tokens, operation.path, operation.from_)
    if operation.op == "copy":
        if operation.from_ is None:
            raise PatchError("copy requires a 'from' path")
        return _copy_value(document, tokens, operation.path, operation.from_)
    raise PatchError(f"unknown patch operation {operation.op!r}")


def _apply_operations(
    document: Any,
    operations: Sequence[PatchOperation],
) -> tuple[Any, list[PatchOperation]]:
    result = copy.deepcopy(document)
    inverse: list[PatchOperation] = []
    for operation in operations:
        result, current_inverse = _apply_one(result, operation)
        inverse = list(current_inverse) + inverse
    return result, inverse


def _as_composition(composition: Composition | Mapping[str, Any]) -> Composition:
    if isinstance(composition, Composition):
        return composition
    if isinstance(composition, Mapping):
        return Composition.model_validate(dict(composition))
    raise TypeError("expected a Composition or a mapping")


class PatchOperation(BaseModel):
    """One reversible edit in a :class:`CompositionPatch`."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    op: str = Field(validation_alias=AliasChoices("op", "operation", "action"))
    path: str = Field(validation_alias=AliasChoices("path", "pointer", "target"))
    value: Any = None
    from_: str | None = Field(
        default=None,
        validation_alias=AliasChoices("from", "from_", "source"),
        serialization_alias="from",
    )

    def __init__(self, data: Any = None, /, **kwargs: Any) -> None:
        if isinstance(data, PatchOperation):
            data = data.to_dict()
        if data is not None:
            if not isinstance(data, Mapping):
                raise TypeError("patch operation data must be a mapping")
            values = dict(data)
            values.update(kwargs)
            kwargs = values
        super().__init__(**kwargs)

    @model_validator(mode="before")
    @classmethod
    def _normalise_move_to_set(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        raw = values.get("op", values.get("operation", values.get("action")))
        if not isinstance(raw, str):
            return values
        normalised = raw.strip().lower().replace("-", "_").replace(" ", "_")
        has_source = any(key in values for key in ("from", "from_", "source"))
        if normalised == "move" and not has_source and "value" in values:
            values["op"] = "set"
        return values

    @field_validator("op", mode="before")
    @classmethod
    def _validate_op(cls, value: Any) -> str:
        return _normalise_op(value)

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("patch path must be a string")
        return value.strip()

    @field_validator("from_", mode="before")
    @classmethod
    def _validate_from(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("patch 'from' path must be a string")
        return value.strip()

    @model_validator(mode="after")
    def _validate_operation(self) -> PatchOperation:
        has_value = "value" in self.model_fields_set
        if self.op in {"add", "replace", "set", "test"} and not has_value:
            raise ValueError(f"{self.op} requires a value")
        if self.op in {"move", "copy"}:
            if self.from_ is None:
                raise ValueError(f"{self.op} requires a 'from' path")
        elif self.from_ is not None:
            raise ValueError(f"{self.op} does not accept a 'from' path")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return the operation as a JSON-compatible mapping."""
        data = self.model_dump(exclude_unset=True, by_alias=False)
        if "from_" in data:
            data["from"] = data.pop("from_")
        return data


Operation = PatchOperation
EditOperation = PatchOperation
CompositionEdit = PatchOperation


class CompositionPatch(BaseModel):
    """An ordered, validated list of composition edit operations."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    operations: list[PatchOperation] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "operations",
            "ops",
            "patches",
            "edits",
            "changes",
        ),
    )
    description: str = ""
    version: int = Field(default=1, ge=1)

    _last_inverse: CompositionPatch | None = PrivateAttr(default=None)
    _last_original_hash: str = PrivateAttr(default="")
    _last_patched_hash: str = PrivateAttr(default="")

    def __init__(self, data: Any = None, /, **kwargs: Any) -> None:
        if isinstance(data, CompositionPatch):
            data = data.to_dict()
        if data is not None:
            if isinstance(data, Mapping):
                values = dict(data)
                values.update(kwargs)
                kwargs = values
            else:
                kwargs.setdefault("operations", data)
        super().__init__(**kwargs)

    @model_validator(mode="before")
    @classmethod
    def _coerce_document(cls, data: Any) -> Any:
        if isinstance(data, CompositionPatch):
            return data
        if data is None:
            return {"operations": []}
        if isinstance(data, Mapping):
            values = dict(data)
            if not any(
                key in values
                for key in ("operations", "ops", "patches", "edits", "changes")
            ) and any(key in values for key in ("op", "operation", "action")):
                return {"operations": [values]}
            return values
        if isinstance(data, (list, tuple)):
            return {"operations": list(data)}
        return data

    @field_validator("operations", mode="before")
    @classmethod
    def _coerce_operations(cls, value: Any) -> Any:
        if isinstance(value, (PatchOperation, Mapping)):
            return [value]
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return the patch as a JSON-compatible mapping."""
        return {
            "operations": [operation.to_dict() for operation in self.operations],
            "description": self.description,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | Sequence[Any]) -> CompositionPatch:
        """Build a patch from a mapping or a list of operations."""
        return cls.model_validate(data)

    @classmethod
    def from_operations(
        cls,
        operations: Sequence[PatchOperation | Mapping[str, Any]],
        *,
        description: str = "",
    ) -> CompositionPatch:
        """Build a patch from an explicit list of operations."""
        return cls(
            operations=list(operations),
            description=description,
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return the patch as deterministic JSON text."""
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, text: str) -> CompositionPatch:
        """Parse a patch from JSON text."""
        if not isinstance(text, str):
            raise TypeError("patch text must be a string")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PatchError(f"invalid patch JSON: {exc}") from exc
        return cls.model_validate(payload)

    def _apply(
        self,
        composition: Composition | Mapping[str, Any],
        *,
        validate: bool,
        assets: AssetLookup | None,
        asset_exists: AssetCheck | None,
    ) -> tuple[Composition, Composition, CompositionPatch]:
        original = _as_composition(composition)
        document = composition_to_dict(original)
        patched_document, inverse_operations = _apply_operations(
            document,
            self.operations,
        )
        if validate:
            issues: CompositionIssues = validate_composition(
                patched_document,
                assets=assets,
                asset_exists=asset_exists,
            )
            if issues:
                raise PatchValidationError(issues)
        patched = Composition.model_validate(patched_document)
        inverse = CompositionPatch(operations=inverse_operations)
        return original, patched, inverse

    def validate_for(
        self,
        composition: Composition | Mapping[str, Any],
        *,
        assets: AssetLookup | None = None,
        asset_exists: AssetCheck | None = None,
    ) -> Composition:
        """Validate the patch against *composition* and return the result."""
        _, patched, _ = self._apply(
            composition,
            validate=True,
            assets=assets,
            asset_exists=asset_exists,
        )
        return patched

    validate = validate_for  # type: ignore[assignment]

    def apply(
        self,
        composition: Composition | Mapping[str, Any],
        *,
        validate: bool = True,
        assets: AssetLookup | None = None,
        asset_exists: AssetCheck | None = None,
    ) -> Composition:
        """Apply the patch and return a new validated composition."""
        original, patched, inverse = self._apply(
            composition,
            validate=validate,
            assets=assets,
            asset_exists=asset_exists,
        )
        self._last_inverse = inverse
        self._last_original_hash = composition_hash(original)
        self._last_patched_hash = composition_hash(patched)
        return patched

    def apply_reversible(
        self,
        composition: Composition | Mapping[str, Any],
        *,
        validate: bool = True,
        assets: AssetLookup | None = None,
        asset_exists: AssetCheck | None = None,
    ) -> AppliedPatch:
        """Apply the patch and keep the original, result and inverse together."""
        original, patched, inverse = self._apply(
            composition,
            validate=validate,
            assets=assets,
            asset_exists=asset_exists,
        )
        self._last_inverse = inverse
        self._last_original_hash = composition_hash(original)
        self._last_patched_hash = composition_hash(patched)
        return AppliedPatch(
            original=original,
            composition=patched,
            patch=self,
            inverse=inverse,
        )

    def inverse_for(
        self,
        composition: Composition | Mapping[str, Any],
        *,
        validate: bool = False,
        assets: AssetLookup | None = None,
        asset_exists: AssetCheck | None = None,
    ) -> CompositionPatch:
        """Return the patch that reverses this patch on *composition*."""
        _, _, inverse = self._apply(
            composition,
            validate=validate,
            assets=assets,
            asset_exists=asset_exists,
        )
        return inverse

    inverse = inverse_for

    def undo(
        self,
        composition: Composition | Mapping[str, Any],
        *,
        original: Composition | Mapping[str, Any] | None = None,
        validate: bool = True,
        assets: AssetLookup | None = None,
        asset_exists: AssetCheck | None = None,
    ) -> Composition:
        """Undo the patch on *composition*, restoring the original hash.

        Pass *original* when the patch was not applied by this object, or when
        its cached inverse belongs to another composition.
        """
        inverse: CompositionPatch | None
        if original is not None:
            inverse = self.inverse_for(original, validate=False)
        else:
            inverse = self._last_inverse
        if inverse is None:
            raise PatchError(
                "patch has no inverse; apply it first or pass the original"
            )
        return inverse.apply(
            composition,
            validate=validate,
            assets=assets,
            asset_exists=asset_exists,
        )

    revert = undo


@dataclass(frozen=True, slots=True)
class AppliedPatch:
    """The result of applying a patch reversibly."""

    original: Composition
    composition: Composition
    patch: CompositionPatch
    inverse: CompositionPatch

    @property
    def result(self) -> Composition:
        """Alias for :attr:`composition`."""
        return self.composition

    @property
    def patched(self) -> Composition:
        """Alias for :attr:`composition`."""
        return self.composition

    @property
    def original_hash(self) -> str:
        """The hash of the composition before the patch."""
        return composition_hash(self.original)

    @property
    def patched_hash(self) -> str:
        """The hash of the composition after the patch."""
        return composition_hash(self.composition)

    @property
    def composition_hash(self) -> str:
        """Alias for :attr:`patched_hash`."""
        return self.patched_hash

    @property
    def hash(self) -> str:
        """Alias for :attr:`patched_hash`."""
        return self.patched_hash

    def undo(self, *, validate: bool = True) -> Composition:
        """Apply the inverse patch and return the restored composition."""
        return self.inverse.apply(self.composition, validate=validate)


PatchResult = AppliedPatch


def _as_patch(
    value: CompositionPatch | Mapping[str, Any] | Sequence[Any] | str,
) -> CompositionPatch:
    if isinstance(value, CompositionPatch):
        return value
    if isinstance(value, str):
        return CompositionPatch.from_json(value)
    return CompositionPatch.model_validate(value)


def apply_patch(
    composition: Composition | Mapping[str, Any],
    patch: CompositionPatch | Mapping[str, Any] | Sequence[Any],
    *,
    validate: bool = True,
    assets: AssetLookup | None = None,
    asset_exists: AssetCheck | None = None,
) -> Composition:
    """Apply *patch* to *composition* and return the new composition."""
    return _as_patch(patch).apply(
        composition,
        validate=validate,
        assets=assets,
        asset_exists=asset_exists,
    )


apply_composition_patch = apply_patch


def apply_patch_reversible(
    composition: Composition | Mapping[str, Any],
    patch: CompositionPatch | Mapping[str, Any] | Sequence[Any],
    *,
    validate: bool = True,
    assets: AssetLookup | None = None,
    asset_exists: AssetCheck | None = None,
) -> AppliedPatch:
    """Apply *patch* and return the result together with its inverse."""
    return _as_patch(patch).apply_reversible(
        composition,
        validate=validate,
        assets=assets,
        asset_exists=asset_exists,
    )


def invert_patch(
    composition: Composition | Mapping[str, Any],
    patch: CompositionPatch | Mapping[str, Any] | Sequence[Any],
    *,
    validate: bool = False,
    assets: AssetLookup | None = None,
    asset_exists: AssetCheck | None = None,
) -> CompositionPatch:
    """Return the inverse of *patch* for *composition*."""
    return _as_patch(patch).inverse_for(
        composition,
        validate=validate,
        assets=assets,
        asset_exists=asset_exists,
    )


def undo_patch(
    composition: Composition | Mapping[str, Any],
    patch: CompositionPatch | AppliedPatch | Mapping[str, Any] | Sequence[Any],
    *,
    original: Composition | Mapping[str, Any] | None = None,
    validate: bool = True,
    assets: AssetLookup | None = None,
    asset_exists: AssetCheck | None = None,
) -> Composition:
    """Undo *patch* on *composition*.

    An :class:`AppliedPatch` carries its own inverse. A plain patch is applied
    as an inverse patch, or reversed first when *original* is supplied.
    """
    if isinstance(patch, AppliedPatch):
        return patch.undo(validate=validate)
    patch_object = _as_patch(patch)
    if original is not None:
        return patch_object.undo(
            composition,
            original=original,
            validate=validate,
            assets=assets,
            asset_exists=asset_exists,
        )
    if patch_object._last_inverse is not None:
        return patch_object.undo(
            composition,
            validate=validate,
            assets=assets,
            asset_exists=asset_exists,
        )
    return patch_object.apply(
        composition,
        validate=validate,
        assets=assets,
        asset_exists=asset_exists,
    )


revert_patch = undo_patch


def validate_patch(
    composition: Composition | Mapping[str, Any],
    patch: CompositionPatch | Mapping[str, Any] | Sequence[Any],
    *,
    assets: AssetLookup | None = None,
    asset_exists: AssetCheck | None = None,
) -> Composition:
    """Validate *patch* against *composition* and return the patched result."""
    return _as_patch(patch).validate_for(
        composition,
        assets=assets,
        asset_exists=asset_exists,
    )


def patch_to_dict(
    patch: CompositionPatch | Mapping[str, Any] | Sequence[Any],
) -> dict[str, Any]:
    """Return *patch* as a JSON-compatible mapping."""
    return _as_patch(patch).to_dict()


def patch_from_dict(
    data: Mapping[str, Any] | Sequence[Any],
) -> CompositionPatch:
    """Build a patch from a mapping or a list of operations."""
    return CompositionPatch.model_validate(data)


parse_patch = patch_from_dict


def patch_to_json(
    patch: CompositionPatch | Mapping[str, Any] | Sequence[Any],
    *,
    indent: int | None = 2,
) -> str:
    """Return *patch* as deterministic JSON text."""
    return _as_patch(patch).to_json(indent=indent)


def patch_from_json(text: str) -> CompositionPatch:
    """Parse a patch from JSON text."""
    return CompositionPatch.from_json(text)


def dump_patch(
    patch: CompositionPatch | Mapping[str, Any] | Sequence[Any],
    *,
    indent: int | None = 2,
) -> str:
    """Alias for :func:`patch_to_json`."""
    return patch_to_json(patch, indent=indent)


def load_patch(text: str) -> CompositionPatch:
    """Alias for :func:`patch_from_json`."""
    return patch_from_json(text)
