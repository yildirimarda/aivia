"""Load and render prompt templates from user files or packaged defaults.

Prompts are user-editable data, not code. A prompt is looked up by name in the
user config directory first and falls back to the packaged default with the
same name. Rendering happens in a Jinja2 sandbox with strict-undefined mode, so
a template cannot reach Python internals and a misspelled variable fails loudly
instead of silently rendering as an empty string.

Names may include an extension or omit it; a missing extension is tried against
:data:`PROMPT_EXTENSIONS` in order. Names are relative to the prompts directory
and may not escape it. Missing prompts are configuration errors, while template
syntax, sandbox and undefined-variable failures are validation errors naming the
template that failed. Templates may not use include, extends, import or from
tags, so a prompt cannot read another file inside or outside the prompts
directory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath
from typing import Any

from jinja2 import StrictUndefined, TemplateError, nodes
from jinja2.sandbox import SandboxedEnvironment

from aivia.core import paths
from aivia.core.config.prompts import PromptVersionSource, normalise_prompt_source
from aivia.core.errors import ConfigError, ValidationError

__all__ = [
    "DEFAULT_PROMPTS_DIR",
    "PROMPTS_DIR_NAME",
    "PROMPT_EXTENSIONS",
    "PromptLibrary",
    "PromptNotFoundError",
    "PromptRenderError",
    "PromptSelection",
    "PromptTemplate",
    "create_prompt_environment",
    "load_prompt",
    "prompt_names",
    "render_prompt",
    "resolve_prompt_path",
]

PROMPTS_DIR_NAME = "prompts"
DEFAULT_PROMPTS_DIR = PROMPTS_DIR_NAME
PROMPT_EXTENSIONS: tuple[str, ...] = (".md", ".j2", ".jinja", ".jinja2", ".txt")

_COMPOSITION_TAGS: dict[type[nodes.Node], str] = {
    nodes.Include: "include",
    nodes.Extends: "extends",
    nodes.Import: "import",
    nodes.FromImport: "from",
}


class PromptNotFoundError(ConfigError):
    """A prompt name has no user file and no packaged default."""

    code = "prompt_not_found"


class PromptRenderError(ValidationError):
    """A prompt template could not be rendered safely."""

    code = "prompt_render"


def create_prompt_environment() -> SandboxedEnvironment:
    """Return the sandboxed, strict-undefined environment used for prompts.

    A fresh environment is returned on every call so one caller cannot mutate
    the filters or globals another caller sees.
    """
    return SandboxedEnvironment(
        autoescape=False,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )


def _normalise_name(name: str | Path) -> PurePosixPath:
    if isinstance(name, PurePath):
        name = name.as_posix()
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("prompt name must be a non-empty string")
    text = name.strip().replace("\\", "/")
    relative = PurePosixPath(text)
    if relative.is_absolute():
        raise ConfigError(
            f"prompt name must be relative to the prompts directory: {name}",
            details={"prompt": text},
        )
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ConfigError(
            f"prompt name may not escape the prompts directory: {name}",
            details={"prompt": text},
        )
    return relative


def _candidates(relative: PurePosixPath) -> tuple[PurePosixPath, ...]:
    if relative.suffix:
        return (relative,)
    return tuple(relative.with_suffix(suffix) for suffix in PROMPT_EXTENSIONS)


def _files(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        return ()
    return tuple(
        sorted(
            (path for path in directory.rglob("*") if path.is_file()),
            key=lambda path: path.as_posix(),
        )
    )


def _render(
    template: PromptTemplate,
    environment: SandboxedEnvironment,
    *,
    variables: Mapping[str, Any] | None,
    values: Mapping[str, Any],
) -> str:
    context: dict[str, Any] = dict(variables or {})
    context.update(values)
    try:
        parsed = environment.parse(template.text, name=template.name)
    except TemplateError as exc:
        raise PromptRenderError(
            f"cannot render prompt {template.name!r} ({template.path}): {exc}",
            details={
                "prompt": template.name,
                "path": str(template.path),
                "source": template.source,
            },
        ) from exc
    for node in parsed.find_all(tuple(_COMPOSITION_TAGS)):
        tag = _COMPOSITION_TAGS[type(node)]
        raise PromptRenderError(
            f"prompt {template.name!r} may not use {tag} "
            f"({template.path}): templates cannot read other files",
            details={
                "prompt": template.name,
                "path": str(template.path),
                "source": template.source,
                "tag": tag,
            },
        )
    try:
        compiled = environment.from_string(template.text)
        return compiled.render(**context)
    except TemplateError as exc:
        raise PromptRenderError(
            f"cannot render prompt {template.name!r} ({template.path}): {exc}",
            details={
                "prompt": template.name,
                "path": str(template.path),
                "source": template.source,
            },
        ) from exc


def _load_path(name: str, path: Path, source: str) -> PromptTemplate:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"cannot read prompt {path}: {exc}",
            details={"prompt": name, "path": str(path)},
        ) from exc
    return PromptTemplate(name=name, path=path, text=text, source=source)


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """One loaded prompt file, before or after rendering."""

    name: str
    path: Path
    text: str
    source: str = "default"

    @property
    def is_default(self) -> bool:
        """Whether this prompt came from the packaged defaults."""
        return self.source == "default"

    @property
    def is_user(self) -> bool:
        """Whether this prompt came from the user config directory."""
        return self.source == "user"

    def render(
        self,
        variables: Mapping[str, Any] | None = None,
        **values: Any,
    ) -> str:
        """Render this template with *variables* and keyword values.

        Keyword values win over the mapping. Missing variables raise a
        :class:`PromptRenderError` that names this template.
        """
        return _render(
            self,
            create_prompt_environment(),
            variables=variables,
            values=values,
        )


class PromptLibrary:
    """Prompt files in the user config directory with packaged fallbacks."""

    def __init__(
        self,
        *,
        config_dir: str | Path | None = None,
        defaults_dir: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
        environment: SandboxedEnvironment | None = None,
    ) -> None:
        if config_dir is None:
            self.config_dir = paths.config_dir(environ)
        else:
            self.config_dir = Path(config_dir).expanduser().resolve()
        if defaults_dir is None:
            self.defaults_dir = paths.package_defaults_dir()
        else:
            self.defaults_dir = Path(defaults_dir).expanduser().resolve()
        self.user_dir = self.config_dir / PROMPTS_DIR_NAME
        self.packaged_dir = self.defaults_dir / PROMPTS_DIR_NAME
        self.environment = environment or create_prompt_environment()

    def _user_path(self, relative: PurePosixPath) -> Path:
        return self.user_dir.joinpath(*relative.parts)

    def _default_path(self, relative: PurePosixPath) -> Path:
        return self.packaged_dir.joinpath(*relative.parts)

    def _paths(
        self,
        relative: PurePosixPath,
    ) -> tuple[Path | None, Path | None]:
        """Return the first user and packaged file for *relative*."""
        user = next(
            (
                path
                for candidate in _candidates(relative)
                if (path := self._user_path(candidate)).is_file()
            ),
            None,
        )
        packaged = next(
            (
                path
                for candidate in _candidates(relative)
                if (path := self._default_path(candidate)).is_file()
            ),
            None,
        )
        return user, packaged

    def resolve(
        self,
        name: str | Path,
        source: str | PromptVersionSource | None = None,
    ) -> Path:
        """Return the file for *name* selected by *source*.

        ``None`` and ``auto`` prefer the user copy and fall back to the
        packaged default. ``user`` and ``default`` require that copy. Raises
        :class:`PromptNotFoundError` listing the directories searched when no
        matching version exists.
        """
        relative = _normalise_name(name)
        user, packaged = self._paths(relative)
        requested = normalise_prompt_source(source)
        selected = {
            PromptVersionSource.AUTO: user or packaged,
            PromptVersionSource.USER: user,
            PromptVersionSource.DEFAULT: packaged,
        }[requested]
        if selected is not None:
            return selected
        searched = (self.user_dir, self.packaged_dir)
        available = self.names()
        hint = f"; available: {', '.join(available)}" if available else ""
        if user is not None or packaged is not None:
            versions = [
                version
                for version, path in (("user", user), ("default", packaged))
                if path is not None
            ]
            raise PromptNotFoundError(
                f"prompt {relative} has no {requested.value} version "
                f"(available: {', '.join(versions)})",
                details={
                    "prompt": relative.as_posix(),
                    "source": requested.value,
                    "available": versions,
                },
            )
        raise PromptNotFoundError(
            f"unknown prompt: {relative} "
            f"(searched {', '.join(map(str, searched))}{hint})",
            details={
                "prompt": relative.as_posix(),
                "searched": [str(path) for path in searched],
                "available": list(available),
            },
        )

    def load(
        self,
        name: str | Path,
        source: str | PromptVersionSource | None = None,
    ) -> PromptTemplate:
        """Load *name* as a :class:`PromptTemplate` without rendering it."""
        path = self.resolve(name, source)
        if path.is_relative_to(self.user_dir):
            return _load_path(path.relative_to(self.user_dir).as_posix(), path, "user")
        return _load_path(
            path.relative_to(self.packaged_dir).as_posix(),
            path,
            "default",
        )

    def render(
        self,
        name: str | Path,
        /,
        variables: Mapping[str, Any] | None = None,
        *,
        source: str | PromptVersionSource | None = None,
        **values: Any,
    ) -> str:
        """Load and render *name* in the sandboxed environment."""
        template = self.load(name, source)
        return _render(
            template,
            self.environment,
            variables=variables,
            values=values,
        )

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, (str, Path)):
            return False
        try:
            self.resolve(name)
        except ConfigError:
            return False
        return True

    def __getitem__(self, name: str | Path) -> PromptTemplate:
        return self.load(name)

    def names(self) -> tuple[str, ...]:
        """Return every prompt name visible to this library.

        User files shadow a packaged default with the same relative name. The
        extension is kept so a caller can see exactly which file is selected.
        """
        names: set[str] = set()
        for directory in (self.packaged_dir, self.user_dir):
            for path in _files(directory):
                names.add(path.relative_to(directory).as_posix())
        return tuple(sorted(names))

    def select(
        self,
        name: str | Path,
        source: str | PromptVersionSource | None = None,
    ) -> PromptTemplate:
        """Load the version of *name* selected by *source*."""
        return self.load(name, source)

    def version(
        self,
        name: str | Path,
        source: str | PromptVersionSource | None = None,
    ) -> PromptTemplate:
        """Alias for :meth:`select`."""
        return self.select(name, source)

    def selection(
        self,
        name: str | Path,
        source: str | PromptVersionSource | None = None,
    ) -> PromptSelection:
        """Return every version of *name* plus the selected one."""
        relative = _normalise_name(name).as_posix()
        versions = self.versions(relative)
        selected = self.load(relative, source)
        return PromptSelection(
            name=selected.name,
            versions=versions,
            selected=selected,
        )

    def versions(self, name: str | Path) -> tuple[PromptTemplate, ...]:
        """Return the user and packaged copies of *name*, user copy first."""
        relative = _normalise_name(name)
        templates: list[PromptTemplate] = []
        for candidate in _candidates(relative):
            user = self._user_path(candidate)
            if user.is_file():
                templates.append(_load_path(candidate.as_posix(), user, "user"))
        for candidate in _candidates(relative):
            packaged = self._default_path(candidate)
            if packaged.is_file():
                templates.append(_load_path(candidate.as_posix(), packaged, "default"))
        return tuple(templates)


@dataclass(frozen=True, slots=True)
class PromptSelection:
    """The available versions of one prompt and the selected version."""

    name: str
    versions: tuple[PromptTemplate, ...]
    selected: PromptTemplate

    @property
    def user(self) -> PromptTemplate | None:
        """The user copy when it exists."""
        return next((template for template in self.versions if template.is_user), None)

    @property
    def default(self) -> PromptTemplate | None:
        """The packaged copy when it exists."""
        return next(
            (template for template in self.versions if template.is_default),
            None,
        )

    @property
    def source(self) -> str:
        """The source of the selected version."""
        return self.selected.source


def resolve_prompt_path(
    name: str | Path,
    *,
    source: str | PromptVersionSource | None = None,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the selected file for *name* without loading it."""
    library = PromptLibrary(
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    )
    return library.resolve(name, source)


def load_prompt(
    name: str | Path,
    *,
    source: str | PromptVersionSource | None = None,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> PromptTemplate:
    """Load *name* from the user prompts directory or packaged defaults."""
    library = PromptLibrary(
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    )
    return library.load(name, source)


def render_prompt(
    name: str | Path,
    /,
    variables: Mapping[str, Any] | None = None,
    *,
    source: str | PromptVersionSource | None = None,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    **values: Any,
) -> str:
    """Load and render *name* in one call."""
    library = PromptLibrary(
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    )
    return library.render(name, variables, source=source, **values)


def prompt_names(
    *,
    config_dir: str | Path | None = None,
    defaults_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return every prompt name visible to a default :class:`PromptLibrary`."""
    library = PromptLibrary(
        config_dir=config_dir,
        defaults_dir=defaults_dir,
        environ=environ,
    )
    return library.names()
