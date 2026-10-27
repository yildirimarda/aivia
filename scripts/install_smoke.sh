#!/usr/bin/env bash
# Build the wheel, install it as an isolated uv tool and exercise the installed
# command from outside the repository.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="$(mktemp -d)"
trap 'rm -rf -- "$work_dir"' EXIT

wheel_dir="$work_dir/wheel"
tool_dir="$work_dir/tool"
tool_bin_dir="$work_dir/bin"
run_dir="$work_dir/outside"
config_home="$work_dir/config"
cache_dir="$work_dir/cache"
mkdir -p "$wheel_dir" "$tool_dir" "$tool_bin_dir" "$run_dir" "$config_home" "$cache_dir"

uv build --wheel --out-dir "$wheel_dir" "$repo_root"

shopt -s nullglob
wheels=("$wheel_dir"/*.whl)
if ((${#wheels[@]} != 1)); then
    printf 'error: expected exactly one wheel, found %d\n' "${#wheels[@]}" >&2
    exit 1
fi
wheel="${wheels[0]}"

UV_TOOL_DIR="$tool_dir" \
    UV_TOOL_BIN_DIR="$tool_bin_dir" \
    UV_CACHE_DIR="$cache_dir" \
    uv tool install --force --no-index --from "$wheel" aivia

# Run from outside the repository with an isolated config directory so the
# smoke test never reads or writes the developer's real configuration.
(
    cd "$run_dir"
    XDG_CONFIG_HOME="$config_home" "$tool_bin_dir/aivia" config show
)
