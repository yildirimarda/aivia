# AIVIA

Configurable AI-backed image, video, audio and music generation assistant. A
personal tool: no PyPI publishing, installed with `uv tool install .` or an
editable install. Everything that can change — vendors, models, formats,
budgets, thresholds — is configuration, never code.

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) (installer and test runner)
- `ffmpeg` and `ffprobe` on PATH (rendering prerequisite, e.g. `brew install ffmpeg`)

## Install

### Isolated tool (recommended)

Build the wheel and install it as an isolated `uv` tool:

    uv tool install .

Upgrade or reinstall:

    uv tool install --force .

Run it from anywhere, outside the repository:

    aivia --version
    aivia init

### Editable install (development)

Install the repository in editable mode so source changes take effect
immediately:

    uv pip install -e .

or, with pip directly:

    pip install -e .

Then:

    aivia --version

## First run: `aivia init`

The first run creates the user config skeleton from the packaged defaults:

    aivia init

This writes `$XDG_CONFIG_HOME/aivia/config.yaml` (plus `prompts/`,
`templates/` and `data/`) and prints the resolved paths:

    Config directory: .../aivia
    Config file: .../aivia/config.yaml
    Data directory: .../aivia/data
    Defaults directory: .../src/aivia/data/defaults
    Created: .../aivia/config.yaml

`init` refuses to overwrite an existing config file. Recreate it with
`--force` (your edited prompts and templates are preserved) or preview without
writing with `--dry-run`. Use `--config PATH` to create the skeleton at an
explicit path instead of the default location.

## Usage

Generate media:

    aivia media --help

Run a format recipe (preview cost first):

    aivia run --estimate
    aivia run -y

Inspect configuration:

    aivia config

List providers and check their health:

    aivia providers --health

Create and manage orders:

    aivia order create --occasion birthday --name "Sam"
    aivia order show

See all commands:

    aivia --help

## CI verified walkthrough

The default test suite is fully offline and never touches the network or spends
money. Run the aggregate gate:

    make check

which runs, in order:

- `uv run ruff check .` (lint)
- `uv run mypy` (type check)
- `uv run pytest -q` (offline test suite)

Run only the tests:

    uv run pytest -q

Live tests (external services, network, spend) are opt-in and require
`--live` plus `AIVA_LIVE_BUDGET_USD`:

    uv run pytest -m live --live   # with AIVA_LIVE_BUDGET_USD set

## No publishing path

AIVIA is a personal tool, not a library for publication. There is no PyPI publishing path. Install it as an isolated `uv` tool or as an editable install
from the repository. Do not publish to PyPI.
