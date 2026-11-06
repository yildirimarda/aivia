"""Align a generated song to its written lyrics and build a karaoke timeline.

The align stage sits between music generation and visuals. It reads the written
lyrics, transcribes the chosen song (or accepts an injected transcript) and
turns the deterministic lyric alignment into timed timeline beats and a
subtitle track with word timings. The music track the music stage already wrote
is kept, so the renderer can mix the song and burn karaoke subtitles.

Transcription is the only provider call. When a transcript is injected the
stage is fully offline, and alignment itself stays the pure function in
:mod:`aivia.audio.lyric_align`.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aivia.audio.lyric_align import LyricAlignment, align_lyrics
from aivia.compose.subtitles import SubtitleCue, SubtitleTrack
from aivia.compose.timeline import (
    Timeline,
    TimelineBeat,
    TimelineLanguage,
)
from aivia.core.errors import ConfigError
from aivia.core.models.run import StageResult, StageState
from aivia.core.stages.base import Stage, StageContext
from aivia.providers.protocols.stt import CAPABILITY_STT, STTRequest, STTResponse
from aivia.writing.schema import Script

__all__ = [
    "ALIGN_STAGE_NAME",
    "DEFAULT_ALIGNMENT_FILENAME",
    "DEFAULT_MUSIC_FILENAME",
    "DEFAULT_TIMELINE_FILENAME",
    "LyricAlignStage",
    "LyricStage",
    "SongAlignStage",
    "build_lyric_align_stage",
    "lyric_align_stage",
    "run_lyric_align_stage",
]

ALIGN_STAGE_NAME = "align"
"""The recipe stage name this implementation serves."""

DEFAULT_TIMELINE_FILENAME = "timeline.json"
"""The timeline file the music stage writes and this stage updates."""

DEFAULT_ALIGNMENT_FILENAME = "lyrics-alignment.json"
"""The manifest that records lines, words, timings and mismatches."""

DEFAULT_MUSIC_FILENAME = "music.json"
"""The music manifest that names the chosen song audio."""

_SCRIPT_FILENAME_PATTERN = "script.{language}.json"
_EPSILON = 1e-9


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _language_key(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ConfigError("language must be a string", details={"value": repr(value)})
    return value.strip().replace("_", "-").lower()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"could not read {path.name}",
            details={"path": str(path), "error": str(exc)},
        ) from exc


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _script_from(value: Any) -> Script | None:
    if isinstance(value, Script):
        return value
    if isinstance(value, Mapping):
        try:
            return Script.model_validate(dict(value))
        except ValueError:
            return None
    return None


def _scripts_from(value: Any) -> dict[str, Script]:
    if value is None:
        return {}
    if isinstance(value, Script):
        code = _language_key(value.language) or "default"
        return {code: value}
    if isinstance(value, Mapping):
        scripts: dict[str, Script] = {}
        for key, item in value.items():
            script = _script_from(item)
            if script is None:
                continue
            code = _language_key(key) or _language_key(script.language)
            scripts[code or "default"] = script
        return scripts
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        scripts = {}
        for position, item in enumerate(value):
            script = _script_from(item)
            if script is None:
                continue
            code = _language_key(script.language) or f"language-{position}"
            scripts[code] = script
        return scripts
    return {}


def _transcript_from(value: Any) -> STTResponse:
    if isinstance(value, STTResponse):
        return value
    response = getattr(value, "response", None)
    if isinstance(response, STTResponse):
        return response
    outcome = getattr(value, "outcome", None)
    if outcome is not None and outcome is not value:
        try:
            return _transcript_from(outcome)
        except ConfigError:
            pass
    if isinstance(value, Mapping):
        try:
            return STTResponse.model_validate(dict(value))
        except ValueError as exc:
            raise ConfigError(
                "injected transcript is invalid",
                details={"field": "transcript"},
            ) from exc
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        words = [
            word
            for word in value
            if isinstance(word, Mapping) or getattr(word, "word", None) is not None
        ]
        if not words:
            raise ConfigError(
                "injected transcript has no words",
                details={"field": "transcript"},
            )
        return STTResponse(
            text=" ".join(
                _text(
                    word.get("word")
                    if isinstance(word, Mapping)
                    else getattr(word, "word", "")
                )
                for word in words
            ),
            words=words,
            duration_seconds=max(
                float(
                    word.get("end_seconds", 0.0)
                    if isinstance(word, Mapping)
                    else getattr(word, "end_seconds", 0.0)
                )
                for word in words
            ),
        )
    raise ConfigError(
        "transcript must be an STTResponse, mapping or word sequence",
        details={"value": type(value).__name__},
    )


def _alignment_from(value: Any) -> LyricAlignment | None:
    if isinstance(value, LyricAlignment):
        return value
    if isinstance(value, Mapping) and {"lyrics", "lines"} <= set(value):
        try:
            return LyricAlignment.model_validate(dict(value))
        except ValueError as exc:
            raise ConfigError(
                "injected alignment is invalid",
                details={"field": "alignment"},
            ) from exc
    return None


class LyricAlignStage(Stage):
    """Align written lyrics to a song and write a karaoke timeline.

    ``options`` may carry ``scripts``/``script``, ``lyrics``, ``language``,
    ``source_language``, ``transcript``/``transcription``, ``transcriber``/
    ``stt``/``speech``, ``audio``/``source``, ``timeline``,
    ``timeline_filename``, ``alignment_filename``, ``music_filename`` and
    ``require_timings``. A caller may inject a transcript, an alignment or a
    transcriber; only the transcriber path calls a provider.
    """

    name = ALIGN_STAGE_NAME
    artifacts: tuple[str, ...] = (
        DEFAULT_TIMELINE_FILENAME,
        DEFAULT_ALIGNMENT_FILENAME,
    )

    def __init__(
        self,
        name: str | None = None,
        *,
        options: Mapping[str, Any] | None = None,
        scripts: Any = None,
        script: Any = None,
        lyrics: Any = None,
        source_language: str | None = None,
        language: str | None = None,
        transcript: Any = None,
        transcription: Any = None,
        alignment: Any = None,
        transcriber: Any = None,
        stt: Any = None,
        speech: Any = None,
        audio: Any = None,
        source: Any = None,
        timeline: Any = None,
        timeline_filename: str | None = None,
        alignment_filename: str | None = None,
        music_filename: str | None = None,
        require_timings: bool | None = None,
        **kwargs: Any,
    ) -> None:
        merged = dict(options or {})
        supplied = {
            "scripts": scripts if scripts is not None else script,
            "lyrics": lyrics,
            "source_language": source_language,
            "language": language,
            "transcript": transcript if transcript is not None else transcription,
            "alignment": alignment,
            "transcriber": transcriber if transcriber is not None else stt,
            "speech": speech,
            "audio": audio if audio is not None else source,
            "timeline": timeline,
            "timeline_filename": timeline_filename,
            "alignment_filename": alignment_filename,
            "music_filename": music_filename,
            "require_timings": require_timings,
        }
        for key, value in supplied.items():
            if value is not None:
                merged[key] = value
        timeline_name = str(
            merged.get("timeline_filename") or DEFAULT_TIMELINE_FILENAME
        )
        alignment_name = str(
            merged.get("alignment_filename") or DEFAULT_ALIGNMENT_FILENAME
        )
        super().__init__(
            name=name or ALIGN_STAGE_NAME,
            options=merged,
            artifacts=(timeline_name, alignment_name),
        )
        self._scripts = merged.get("scripts")
        self._lyrics = merged.get("lyrics")
        self._source_language = merged.get("source_language")
        self._language = merged.get("language")
        self._transcript = merged.get("transcript")
        self._alignment = merged.get("alignment")
        self._transcriber = merged.get("transcriber")
        self._speech = merged.get("speech")
        self._audio = merged.get("audio")
        self._timeline = merged.get("timeline")
        self.timeline_filename = timeline_name
        self.alignment_filename = alignment_name
        self.music_filename = str(
            merged.get("music_filename") or DEFAULT_MUSIC_FILENAME
        )
        self.require_timings = bool(merged.get("require_timings", True))

    def _script(self, context: StageContext) -> Script | None:
        scripts = _scripts_from(self._scripts)
        requested = _language_key(self._language or self._source_language)
        if scripts:
            if requested and requested in scripts:
                return scripts[requested]
            if requested:
                raise ConfigError(
                    f"unknown lyric language: {requested}",
                    details={"language": requested, "available": list(scripts)},
                )
            return next(iter(scripts.values()))
        names: list[str] = []
        if requested:
            names.extend(
                (
                    _SCRIPT_FILENAME_PATTERN.format(language=requested),
                    f"script_{requested}.json",
                )
            )
        names.append("script.json")
        for name in names:
            path = context.run_dir / name
            if path.is_file():
                script = _script_from(_read_json(path))
                if script is not None:
                    return script
        for path in sorted(context.run_dir.glob("script.*.json")):
            script = _script_from(_read_json(path))
            if script is not None:
                return script
        return None

    def _language_for(self, script: Script | None) -> str:
        return (
            _language_key(self._language)
            or _language_key(self._source_language)
            or (_language_key(script.language) if script is not None else "")
            or "en"
        )

    def _load_timeline(self, context: StageContext) -> Timeline:
        value = self._timeline
        if value is None:
            path = context.run_dir / self.timeline_filename
            if not path.is_file():
                return Timeline(id=context.run.id if context.run is not None else "")
            return Timeline.model_validate(_read_json(path))
        if isinstance(value, Timeline):
            return value.model_copy(deep=True)
        if isinstance(value, Mapping):
            return Timeline.model_validate(dict(value))
        if isinstance(value, (str, Path)):
            path = Path(value)
            if not path.is_absolute():
                path = context.run_dir / path
            if not path.is_file():
                raise ConfigError(
                    "align stage timeline does not exist",
                    details={"timeline": str(path)},
                )
            return Timeline.model_validate(_read_json(path))
        raise ConfigError(
            "timeline must be a Timeline, mapping or file path",
            details={"value": type(value).__name__},
        )

    def _song_source(self, context: StageContext, timeline: Timeline) -> str:
        manifest = context.run_dir / self.music_filename
        if manifest.is_file():
            payload = _read_json(manifest)
            if isinstance(payload, Mapping):
                selected = payload.get("selected")
                if isinstance(selected, Mapping):
                    source = _text(selected.get("source") or selected.get("path"))
                    if source:
                        return source
        for track in timeline.music:
            if track.source:
                return track.source
        return ""

    def _audio_bytes(self, context: StageContext, timeline: Timeline) -> bytes | None:
        value = self._audio
        if value is not None:
            if isinstance(value, (bytes, bytearray)):
                return bytes(value)
            if isinstance(value, (str, Path)):
                path = Path(value)
                if not path.is_absolute():
                    path = context.run_dir / path
                if path.is_file():
                    return path.read_bytes()
            data = getattr(value, "data", None)
            if isinstance(data, (bytes, bytearray)) and data:
                return bytes(data)
        source = self._song_source(context, timeline)
        if not source:
            return None
        path = Path(source)
        if not path.is_absolute():
            path = context.run_dir / path
        if not path.is_file():
            raise ConfigError(
                "align stage song audio does not exist",
                details={"audio": str(path)},
            )
        return path.read_bytes()

    async def _transcribe(
        self,
        context: StageContext,
        timeline: Timeline,
        language: str,
    ) -> tuple[STTResponse | None, float]:
        target = self._transcriber or self._speech
        audio = self._audio_bytes(context, timeline)
        if audio is None:
            return None, 0.0
        request = STTRequest(
            audio=audio,
            language=language or None,
            word_timestamps=True,
        )
        if target is None:
            if context.router is None:
                return None, 0.0
            result = await context.media.transcribe(
                request,
                run_id=context.run.id if context.run is not None else "",
                operation=CAPABILITY_STT,
            )
            return _transcript_from(result), max(
                0.0, float(getattr(result, "cost_usd", 0.0) or 0.0)
            )
        method = target
        if not callable(method):
            method = getattr(target, "transcribe", None)
        if not callable(method):
            raise ConfigError(
                "transcriber must be callable or expose a transcribe method",
                details={"value": type(target).__name__},
            )
        value = method(request)
        if inspect.isawaitable(value):
            value = await value
        cost = getattr(value, "cost_usd", 0.0)
        response = _transcript_from(value)
        return response, max(0.0, float(cost or 0.0))

    @staticmethod
    def _cue(line: Any) -> SubtitleCue:
        words = [
            {
                "word": word.word,
                "start_seconds": word.start_seconds,
                "end_seconds": word.end_seconds,
                "confidence": word.confidence,
            }
            for word in line.words
            if word.start_seconds is not None and word.end_seconds is not None
        ]
        if words:
            return SubtitleCue.from_words(
                words,
                start_seconds=line.start_seconds,
                end_seconds=line.end_seconds,
                text=line.text,
            )
        return SubtitleCue(
            start_seconds=line.start_seconds,
            end_seconds=line.end_seconds,
            text=line.text,
        )

    def _build_timeline(
        self,
        timeline: Timeline,
        alignment: LyricAlignment,
        language: str,
    ) -> Timeline:
        untimed = [line.index for line in alignment.lines if not line.aligned]
        if self.require_timings and untimed:
            raise ConfigError(
                "lyric alignment is missing timings",
                details={"lines": untimed, "language": language},
            )

        existing_beats = timeline.beats
        beats: list[TimelineBeat] = []
        for position, line in enumerate(alignment.lines):
            if line.start_seconds is None or line.end_seconds is None:
                continue
            if line.end_seconds <= line.start_seconds:
                raise ConfigError(
                    "lyric line has an empty timing span",
                    details={"line": position, "text": line.text},
                )
            base = (
                existing_beats[position].model_dump(mode="python")
                if position < len(existing_beats)
                else {}
            )
            base.update(
                {
                    "id": base.get("id") or f"lyric_{position}",
                    "index": position,
                    "start_seconds": line.start_seconds,
                    "end_seconds": line.end_seconds,
                    "text": line.text,
                }
            )
            beats.append(TimelineBeat.model_validate(base))

        cues = [self._cue(line) for line in alignment.lines if line.aligned]
        track = SubtitleTrack(
            cues=cues,
            language=language,
            metadata={
                "source": self.alignment_filename,
                "karaoke": True,
                "coverage": alignment.coverage,
            },
        )
        previous = next(
            (
                entry
                for entry in timeline.languages
                if _language_key(entry.language) == language
            ),
            None,
        )
        languages = [
            entry
            for entry in timeline.languages
            if _language_key(entry.language) != language
        ]
        languages.append(
            TimelineLanguage(
                language=language,
                voices=list(previous.voices) if previous is not None else [],
                music=list(previous.music) if previous is not None else [],
                subtitles=track,
            )
        )

        duration = alignment.duration_seconds
        if duration is None:
            ends = [
                line.end_seconds
                for line in alignment.lines
                if line.end_seconds is not None
            ]
            duration = max(ends) if ends else timeline.duration_seconds
        if duration is None or duration <= _EPSILON:
            raise ConfigError(
                "lyric alignment has no positive duration",
                details={"language": language},
            )

        metadata = dict(timeline.metadata)
        metadata["alignment"] = {
            "file": self.alignment_filename,
            "language": language,
            "coverage": alignment.coverage,
            "line_coverage": alignment.line_coverage,
            "mismatches": alignment.mismatch_count,
            "words": alignment.word_count,
            "lines": alignment.line_count,
        }
        return timeline.model_copy(
            update={
                "beats": beats,
                "languages": languages,
                "duration_seconds": duration,
                "metadata": metadata,
            }
        )

    async def run(self, context: StageContext) -> StageResult:
        script = self._script(context)
        language = self._language_for(script)
        timeline = self._load_timeline(context)
        injected = _alignment_from(self._alignment)
        transcription_cost = 0.0
        if injected is not None:
            alignment = injected
        else:
            transcript = (
                _transcript_from(self._transcript)
                if self._transcript is not None
                else None
            )
            if transcript is None:
                transcript, transcription_cost = await self._transcribe(
                    context,
                    timeline,
                    language,
                )
            if transcript is None:
                raise ConfigError(
                    "align stage needs a transcript or a transcriber",
                    details={"stage": self.name},
                )
            lyrics = self._lyrics if self._lyrics is not None else script
            if lyrics is None:
                raise ConfigError(
                    "align stage needs written lyrics",
                    details={"stage": self.name},
                )
            alignment = align_lyrics(lyrics, transcript, language=language)

        updated = self._build_timeline(timeline, alignment, language)
        _write_json(
            context.run_dir / self.timeline_filename, updated.model_dump(mode="json")
        )
        _write_json(
            context.run_dir / self.alignment_filename,
            {
                **alignment.to_dict(),
                "timeline": self.timeline_filename,
                "music": self.music_filename,
            },
        )
        run_id = context.run.id if context.run is not None else ""
        return StageResult(
            run_id=run_id,
            stage=self.name,
            status=StageState.SUCCEEDED,
            artifacts=[self.timeline_filename, self.alignment_filename],
            cost_usd=transcription_cost,
            metadata={
                "language": language,
                "coverage": alignment.coverage,
                "line_coverage": alignment.line_coverage,
                "mismatches": alignment.mismatch_count,
                "lines": alignment.line_count,
                "words": alignment.word_count,
                "timeline": self.timeline_filename,
                "alignment": self.alignment_filename,
                "cost_usd": transcription_cost,
            },
        )


LyricStage = LyricAlignStage
SongAlignStage = LyricAlignStage
lyric_align_stage = LyricAlignStage


def build_lyric_align_stage(
    *,
    options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> LyricAlignStage:
    """Build a :class:`LyricAlignStage` from recipe options and overrides."""
    return LyricAlignStage(options=options, **kwargs)


async def run_lyric_align_stage(
    context: StageContext,
    *,
    stage: LyricAlignStage | None = None,
    **options: Any,
) -> StageResult:
    """Build a stage unless one is supplied, then run it asynchronously."""
    if stage is not None and options:
        raise ConfigError("pass either a stage or options, not both")
    active = stage if stage is not None else LyricAlignStage(**options)
    return await active.run(context)
