"""Run directories and the per-stage artifact registry."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aivia.core.models.run import Run, StageResult
from aivia.core.run_dirs import (
    ARTIFACT_MANIFEST_NAME,
    RUNS_DIR_NAME,
    ArtifactRecord,
    RunDirectories,
    RunDirectory,
    create_run_directory,
    register_stage_artifacts,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_a_run_directory_lives_under_the_runs_root_and_is_recorded_on_the_run(
    tmp_path: Path,
) -> None:
    run = Run()
    runs = RunDirectories(tmp_path)

    directory = runs.create(run)

    assert directory.path == tmp_path / RUNS_DIR_NAME / run.id
    assert directory.path.is_dir()
    assert run.run_dir == str(directory.path)
    assert directory.run_id == run.id
    assert directory.run_dir == directory.path
    assert directory.directory == directory.path
    assert directory.root == directory.path
    assert directory.exists is True
    assert os.fspath(directory) == str(directory.path)


def test_creating_a_run_directory_twice_reuses_it(tmp_path: Path) -> None:
    run = Run()
    runs = RunDirectories(tmp_path)
    first = runs.create(run)
    _write(first.path / "keep.txt", "kept")

    second = runs.create(run)

    assert second.path == first.path
    assert (second.path / "keep.txt").read_text(encoding="utf-8") == "kept"
    with pytest.raises(FileExistsError):
        runs.create(run, exist_ok=False)


def test_path_for_does_not_create_the_directory(tmp_path: Path) -> None:
    run = Run()

    path = RunDirectories(tmp_path).path_for(run)

    assert path == tmp_path / RUNS_DIR_NAME / run.id
    assert path.exists() is False
    assert run.run_dir == ""


def test_an_explicit_run_directory_is_honoured(tmp_path: Path) -> None:
    custom = tmp_path / "elsewhere" / "custom-run"
    run = Run(run_dir=str(custom))

    directory = RunDirectories(tmp_path).create(run)

    assert directory.path == custom.resolve()
    assert run.run_dir == str(custom.resolve())
    assert custom.is_dir()


def test_a_run_id_cannot_escape_the_runs_root(tmp_path: Path) -> None:
    runs = RunDirectories(tmp_path)

    for run_id in ("../escape", "nested/run", "..", ".", ""):
        with pytest.raises(ValueError):
            runs.path_for(run_id)


def test_open_and_exists_do_not_create_a_run_directory(tmp_path: Path) -> None:
    run = Run()
    runs = RunDirectories(tmp_path)

    assert runs.exists(run) is False
    directory = runs.open(run)

    assert directory.path.exists() is False
    assert runs.exists(run) is False


def test_creating_a_directory_over_a_file_fails(tmp_path: Path) -> None:
    path = tmp_path / "occupied"
    _write(path, "not a directory")

    with pytest.raises(NotADirectoryError):
        RunDirectory(path).create()


def test_stage_dir_is_created_below_the_run_and_can_be_skipped(
    tmp_path: Path,
) -> None:
    directory = RunDirectory(tmp_path / "run").create()

    created = directory.stage_dir("  write  ")
    not_created = directory.stage_dir("render", create=False)

    assert created == directory.path / "write"
    assert created.is_dir()
    assert not_created == directory.path / "render"
    assert not_created.exists() is False


@pytest.mark.parametrize("stage", ["", "   ", "../escape", "nested/stage", ".."])
def test_invalid_stage_names_are_rejected(tmp_path: Path, stage: str) -> None:
    directory = RunDirectory(tmp_path / "run").create()

    with pytest.raises(ValueError):
        directory.stage_dir(stage)
    with pytest.raises(ValueError):
        directory.register(stage, [])


def test_resolve_accepts_relative_and_absolute_paths_inside_the_run(
    tmp_path: Path,
) -> None:
    directory = RunDirectory(tmp_path / "run").create()
    target = _write(directory.path / "nested" / "file.txt", "content")

    assert directory.resolve("nested/file.txt") == target
    assert directory.resolve(target) == target
    assert directory.relative_path(target) == "nested/file.txt"
    assert directory.relative_path("nested/file.txt") == "nested/file.txt"


def test_resolve_rejects_paths_that_escape_the_run(tmp_path: Path) -> None:
    directory = RunDirectory(tmp_path / "run").create()
    outside = _write(tmp_path / "outside.txt", "outside")

    with pytest.raises(ValueError, match="escapes"):
        directory.resolve("../outside.txt")
    with pytest.raises(ValueError, match="escapes"):
        directory.resolve(outside)
    with pytest.raises(ValueError, match="empty"):
        directory.resolve("")
    assert directory.resolve(".") == directory.path


def test_a_symlink_that_escapes_the_run_is_rejected(tmp_path: Path) -> None:
    directory = RunDirectory(tmp_path / "run").create()
    outside = _write(tmp_path / "outside.txt", "outside")
    (directory.path / "link.txt").symlink_to(outside)

    with pytest.raises(ValueError, match="escapes"):
        directory.register("write", "link.txt")


def test_register_hashes_files_and_writes_the_manifest_per_stage(
    tmp_path: Path,
) -> None:
    directory = RunDirectory(tmp_path / "run", run_id="run_1").create()
    script = _write(directory.path / "script.md", "hello")
    facts = _write(directory.path / "nested" / "facts.json", '{"ok":true}')

    records = directory.register("write", [script, "nested/facts.json"])

    assert [record.path for record in records] == ["script.md", "nested/facts.json"]
    assert [record.stage for record in records] == ["write", "write"]
    assert records[0].sha256 == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )
    assert records[0].content_hash == records[0].sha256
    assert records[0].size_bytes == 5
    assert records[1].size_bytes == 11
    assert records[1].size == 11

    manifest = json.loads(directory.manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert manifest["run_id"] == "run_1"
    assert [entry["path"] for entry in manifest["stages"]["write"]] == [
        "script.md",
        "nested/facts.json",
    ]
    assert facts.exists()


def test_register_accepts_a_single_path_and_deduplicates(tmp_path: Path) -> None:
    directory = RunDirectory(tmp_path / "run").create()
    _write(directory.path / "a.txt", "a")

    records = directory.register("render", "a.txt")
    again = directory.register("render", ["a.txt", "./a.txt", "a.txt"])

    assert [record.path for record in records] == ["a.txt"]
    assert [record.path for record in again] == ["a.txt"]
    assert len(directory.artifacts("render")) == 1


def test_registering_the_same_stage_again_replaces_its_entries(
    tmp_path: Path,
) -> None:
    directory = RunDirectory(tmp_path / "run").create()
    _write(directory.path / "first.txt", "first")
    _write(directory.path / "second.txt", "second")

    directory.register("render", "first.txt")
    directory.register("render", "second.txt")

    assert [record.path for record in directory.artifacts("render")] == ["second.txt"]


def test_artifacts_can_be_listed_for_one_stage_or_all_of_them(
    tmp_path: Path,
) -> None:
    directory = RunDirectory(tmp_path / "run").create()
    _write(directory.path / "script.md", "script")
    _write(directory.path / "clip.mp4", "clip")
    directory.register("write", "script.md")
    directory.register("render", "clip.mp4")

    assert [record.path for record in directory.artifacts("write")] == ["script.md"]
    assert {record.path for record in directory.artifacts()} == {
        "script.md",
        "clip.mp4",
    }
    assert directory.stages() == ("render", "write")


def test_a_missing_artifact_is_an_error_unless_explicitly_allowed(
    tmp_path: Path,
) -> None:
    directory = RunDirectory(tmp_path / "run").create()

    with pytest.raises(FileNotFoundError, match=r"missing\.txt"):
        directory.register("render", "missing.txt")

    records = directory.register("render", "missing.txt", require_exists=False)

    assert records[0].path == "missing.txt"
    assert records[0].sha256 == ""
    assert records[0].size_bytes == 0
    assert directory.artifacts("render") == records


def test_a_directory_is_not_registered_as_an_artifact(tmp_path: Path) -> None:
    directory = RunDirectory(tmp_path / "run").create()
    (directory.path / "folder").mkdir()

    with pytest.raises(IsADirectoryError, match="folder"):
        directory.register("render", "folder")


def test_a_stage_result_receives_the_registered_relative_paths(
    tmp_path: Path,
) -> None:
    directory = RunDirectory(tmp_path / "run").create()
    target = _write(directory.path / "clip.mp4", "clip")
    result = StageResult(stage="render", artifacts=[str(target)])

    records = directory.register("render", result=result)

    assert [record.path for record in records] == ["clip.mp4"]
    assert result.artifacts == ["clip.mp4"]
    assert directory.artifacts("render") == records


def test_a_stage_result_for_another_stage_is_rejected(tmp_path: Path) -> None:
    directory = RunDirectory(tmp_path / "run").create()
    result = StageResult(stage="write", artifacts=[])

    with pytest.raises(ValueError, match="does not match"):
        directory.register("render", result=result)


def test_an_empty_registration_records_an_empty_stage(tmp_path: Path) -> None:
    directory = RunDirectory(tmp_path / "run").create()

    assert directory.register("write", []) == ()
    assert directory.artifacts("write") == ()
    assert directory.stages() == ("write",)


def test_reading_a_missing_or_malformed_manifest_is_safe(tmp_path: Path) -> None:
    directory = RunDirectory(tmp_path / "run").create()

    assert directory.artifacts() == ()
    assert directory.stages() == ()

    directory.manifest_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid artifact manifest"):
        directory.artifacts()


def test_artifact_records_round_trip_through_the_manifest(tmp_path: Path) -> None:
    directory = RunDirectory(tmp_path / "run").create()
    _write(directory.path / "a.txt", "a")
    record = directory.register("render", "a.txt")[0]

    assert ArtifactRecord.from_dict(record.to_dict()) == record
    with pytest.raises(ValueError, match="sha256"):
        ArtifactRecord.from_dict({**record.to_dict(), "sha256": "short"})
    with pytest.raises(ValueError, match="size"):
        ArtifactRecord.from_dict({**record.to_dict(), "size_bytes": -1})
    with pytest.raises(ValueError, match="required field"):
        ArtifactRecord.from_dict({"path": "a.txt"})
    with pytest.raises(ValueError, match="relative"):
        ArtifactRecord.from_dict({**record.to_dict(), "path": "../a.txt"})


def test_top_level_helpers_create_and_register(tmp_path: Path) -> None:
    run = Run()
    directory = create_run_directory(tmp_path, run)
    _write(directory.path / "out.txt", "out")

    records = register_stage_artifacts(directory, "render", "out.txt")

    assert [record.path for record in records] == ["out.txt"]
    assert run.run_dir == str(directory.path)
    assert register_stage_artifacts(directory.path, "write", []) == ()


def test_manifest_name_is_stable() -> None:
    assert ARTIFACT_MANIFEST_NAME == "artifacts.json"
    assert RUNS_DIR_NAME == "runs"
