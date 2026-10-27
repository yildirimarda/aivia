"""Installed tool smoke contract: build, install and run outside the repo."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "install_smoke.sh"

FAKE_UV = """#!/bin/sh
set -eu
printf '%s\\n' "$*" >> "$UV_LOG"
case "${1:-}" in
    build)
        out_dir=""
        previous=""
        for argument in "$@"; do
            if [ "$previous" = "--out-dir" ]; then
                out_dir="$argument"
            fi
            previous="$argument"
        done
        if [ -z "$out_dir" ]; then
            printf 'missing --out-dir\\n' >&2
            exit 1
        fi
        printf 'wheel\\n' > "$out_dir/aivia-0.1.0-py3-none-any.whl"
        ;;
    tool)
        for argument in "$@"; do
            if [ "$argument" = "--no-index" ]; then
                printf 'error: cannot resolve runtime dependencies offline\\n' >&2
                exit 1
            fi
        done
        if [ -z "${UV_TOOL_BIN_DIR:-}" ]; then
            printf 'missing UV_TOOL_BIN_DIR\\n' >&2
            exit 1
        fi
        mkdir -p "$UV_TOOL_BIN_DIR"
        cat > "$UV_TOOL_BIN_DIR/aivia" <<'AIVIA'
#!/bin/sh
set -eu
printf '%s\\t%s\\t%s\\n' "$PWD" "${XDG_CONFIG_HOME:-}" "$*" >> "$AIVIA_LOG"
if [ "${1:-}" = "config" ] && [ "${2:-}" = "show" ]; then
    printf 'schema_version: 1\\n'
    exit "${AIVIA_EXIT_CODE:-0}"
fi
printf 'unexpected command: %s\\n' "$*" >&2
exit 2
AIVIA
        chmod +x "$UV_TOOL_BIN_DIR/aivia"
        ;;
esac
exit 0
"""


def _fake_tools(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(FAKE_UV, encoding="utf-8")
    uv.chmod(0o755)

    uv_log = tmp_path / "uv.log"
    aivia_log = tmp_path / "aivia.log"
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["UV_LOG"] = str(uv_log)
    env["AIVIA_LOG"] = str(aivia_log)
    env["XDG_CONFIG_HOME"] = str(tmp_path / "real-config")
    return env, uv_log, aivia_log


def _run(env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_script_exists_and_is_executable() -> None:
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)


def test_script_builds_installs_and_runs_config_show_outside_the_repo(
    tmp_path: Path,
) -> None:
    env, uv_log, aivia_log = _fake_tools(tmp_path)

    result = _run(env, tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    uv_calls = uv_log.read_text(encoding="utf-8").splitlines()
    assert any("build" in call and "--wheel" in call for call in uv_calls)
    assert any("tool install" in call and "--from" in call for call in uv_calls)
    assert any(call.rstrip().endswith("aivia") for call in uv_calls)

    cwd, config_home, command = (
        aivia_log.read_text(encoding="utf-8").strip().split("\t")
    )
    assert command == "config show"
    assert Path(cwd).resolve() != ROOT
    assert Path(config_home) != tmp_path / "real-config"


def test_script_fails_when_the_installed_config_show_fails(tmp_path: Path) -> None:
    env, _, _ = _fake_tools(tmp_path)
    env["AIVIA_EXIT_CODE"] = "7"

    result = _run(env, tmp_path)

    assert result.returncode == 7
