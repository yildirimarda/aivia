# Troubleshooting guide

This guide covers the most common failures, how to read cost reports, how to
resume a paused or failed run, how to rotate credentials without restarting, and
where the tool looks for its configuration. Every message quoted here is
emitted by the code under `src/aivia/` and is asserted by
`tests/test_troubleshooting.py`.

## Common failures

### Missing or invalid credentials

A provider cannot authenticate because a secret reference cannot be resolved.
Lookup order is environment variables (with the `AIVA_` prefix), then
`secrets.env`, then the OS keyring. Values never appear in errors or logs.

* `missing secret: <reference>` -- raised by the secrets loader when a
  reference is absent from every source. Fix: set the environment variable or
  add the reference to `secrets.env`.
* `secret reference must not be empty` -- the reference passed to the loader
  was blank after trimming.
* `secrets file not found: <path>` -- `AIVA_SECRETS_FILE` or `--secrets-file`
  points at a file that does not exist.
* `cannot read secrets file: <path>` -- the secrets file exists but cannot be
  read.
* `invalid secrets file line <number>: <path>` -- a line in `secrets.env` has
  no `=` or an empty key.
* `credentials reloaded` -- logged and printed by `aivia credentials reload`
  after the resolution cache is cleared; no secret value is ever included.
* `error: cannot reload credentials: <exc>` -- the reload itself failed.

### Configuration errors

* `error: config file not found: <path>` -- `aivia config migrate` or another
  command could not find the config file at the resolved path.
* `error: cannot read <path>: <exc>` -- the config file exists but cannot be
  read.
* `configuration is valid` -- printed by `aivia config validate` when every
  section, key and prompt version resolves.
* Config errors name the file, the dotted key path, the expected versus actual
  value and a hint, and never include a stack trace.

### Budget and provider failures

* `budget` (error code) -- the call would exceed a configured spending limit;
  permanent, so retrying the same call cannot succeed.
* `provider_exhausted` (error code) -- every provider in the capability chain
  failed; retryable, so a later attempt or a different chain may succeed.
* `rate_limit` (error code) -- a provider rejected the call because a rate
  limit was reached; retryable, with an optional `retry_after` delay.
* `auth` (error code) -- credentials are missing, invalid or unauthorized;
  permanent. Rotate the credential (see Credential rotation below).
* `timeout` (error code) -- the operation exceeded its deadline; retryable.
* `invalid_request`, `content_policy`, `validation`, `config`, `quota` --
  permanent errors; do not retry unchanged.

### First run and bootstrap

* `error: cannot populate the config directory: <exc>` -- the CLI could not
  copy the packaged defaults into the user config directory.
* `packaged defaults directory is missing: <path>` -- the installed package is
  missing its bundled defaults.
* `error: config already exists at <path>` -- `aivia init` refuses to
  overwrite without `--force`; use `--dry-run` to preview first.
* `error: cannot overwrite <path>: <exc>` -- `init --force` could not replace
  the existing config file.

## Cost reports

### `aivia costs run`

Reports one run's spend broken down by stage, provider, model and asset.

* `error: a run id is required` -- no run id was given (positional or
  `--run-id`).
* `error: database not found: <path>` -- the database file beside the config
  (or passed with `--database`) does not exist.
* `error: run not found: <run_id>` -- the run id is not recorded in the
  database.
* `error: cannot read the cost ledger: <exc>` -- a database error while reading
  cost events.
* `Run: <run_id>` -- the run whose spend follows.
* `Total: $<amount>` -- the run's total spend, with replaced estimates excluded.
* `BY STAGE` -- spend grouped by stage (`STAGE`, `TOTAL`, `ESTIMATED`,
  `ACTUAL`, `UNITS`, `COUNT`).
* `BY PROVIDER` -- spend grouped by provider.
* `BY MODEL` -- spend grouped by model.
* `BY ASSET` -- spend grouped by asset (`ASSET`, `LABEL`, `KIND`, ...), with
  unassigned events shown as `(unassigned)`.

### `aivia costs summary`, `export` and `dashboard`

* `error: invalid date: <exc>` -- a `--start` or `--end` value is not a valid
  ISO date or datetime.
* `error: unknown group <name>; expected one of <allowed>` -- a `--group`
  value is not one of `run`, `day`, `week`, `month`, `provider`, `model`,
  `profile`, `format` or `stage`.
* `Total: $<amount>` -- printed at the top of the table and export output.
* `wrote <path>` -- printed when `--output` is given and the file is written.
* `error: cannot write <path>: <exc>` -- the export destination could not be
  written.

## Resuming

### `aivia resume`

Continues a run paused by the budget guard, recording who confirmed and why.

* `error: a run id is required` -- no run id was given.
* `error: --by is required to record who resumed the run` -- `--by`/`
  --who`/ `--resumed-by` is mandatory.
* `error: --reason is required to record why the run was resumed` -- `--reason`
  / `--why` / `--resume-reason` is mandatory.
* `error: database not found: <path>` -- the database file does not exist.
* `error: run not found: <run_id>` -- the run id is not recorded.
* `error: run <id> is not paused on budget (state: <state>)` -- the run exists
  but is not in the `paused_budget` state, so it cannot be resumed.
* `error: cannot resume run <id>: <exc>` -- the state machine rejected the
  transition.
* `error: cannot resume the run: <exc>` -- a database error while resuming.
* `Resumed run <id> as <who>: <why>` -- the run was resumed and the actor and
  reason were recorded.

A succeeded run skipped on resume is not an error: every stage whose input hash
is unchanged is skipped, and a second resume keeps skipping the unchanged
prefix.

## Credential rotation

### `aivia credentials reload`

Reloads secrets from the environment, `secrets.env` and the OS keyring without
restarting the tool.

* `credentials reloaded` -- printed and logged after the cache is cleared.
* `error: cannot reload credentials: <exc>` -- the reload failed.

Rotation behaviour:

* Calls already in flight keep using the captured (old) value; the next call
  after `reload()` reads the rotated value.
* `get_optional` keeps its cached value until `reload()` clears the cache.
* A missing secret that later appears is picked up on the next `get` after
  `reload()`.
* Rotation is logged without any secret value: `credentials reloaded` names no
  reference and prints no token.

## Config locations

### Default locations

The tool follows the XDG base directory rules and never depends on the current
working directory:

* Config directory: `$XDG_CONFIG_HOME/aivia/` (falls back to
  `~/.config/aivia/` when `XDG_CONFIG_HOME` is unset, empty or relative).
* Config file: `<config directory>/config.yaml`.
* Data directory: `<config directory>/data/` (holds the SQLite database,
  runs, media and cache).
* Packaged defaults: shipped inside the package under
  `src/aivia/data/defaults/` and copied once on first run.

`aivia init` prints the resolved locations:

* `Config directory: <path>`
* `Config file: <path>`
* `Data directory: <path>`
* `Defaults directory: <path>`

### Overriding the location

* `--config PATH` always wins over the default location; it is expanded and
  made absolute relative to the working directory (a leading `~` is expanded
  too).
* `AIVA_` is the environment variable prefix; tool variables such as
  `AIVA_SECRETS_FILE` use it.
* Secrets live in `secrets.env` beside the config (or at `AIVA_SECRETS_FILE`),
  never in `config.yaml`.

### Where packaged resources live

Bundled fonts, the default config, prompts, templates, styles, brands and
platform/model data are package data under `src/aivia/assets/` and
`src/aivia/data/defaults/`. They resolve through the import system so they work
from an installed wheel as well as from a source checkout, from any working
directory.
