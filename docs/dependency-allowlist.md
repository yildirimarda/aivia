# Dependency vulnerability allowlist

## Policy

CI runs `pip-audit` (PyPA) against every installed dependency on every push
and pull request. `pip-audit` exits non-zero when it confirms a vulnerability,
so **any confirmed vulnerability fails CI** — this includes every high and
critical severity issue, and every lower severity issue unless it is on the
allowlist below.

The scanner is run from `.github/workflows/ci.yml` (proposal at
`ci-proposals/ci.yml`) after `uv sync --frozen --all-extras --dev`, so it
audits exactly the locked dependency set:

    uv run --no-sync pip-audit --desc on --aliases on

## Allowlist

Entries here are exceptions granted to the fail-on-vulnerability policy. For
each entry, add `--ignore-vuln <CVE-ID>` to the `Dependency vulnerability scan`
step in `.github/workflows/ci.yml` **and** record a row here. An entry without
a matching `--ignore-vuln` flag (or vice versa) is a configuration error and
must be fixed before merging.

| CVE ID | Package | Versions | Severity | Justification | Review by | Granted by |
|--------|---------|----------|----------|---------------|-----------|------------|

## Current state

No exceptions are granted. The table above is intentionally empty: every
dependency in `uv.lock` is currently free of known vulnerabilities, and CI
fails if `pip-audit` reports any.

## How to add an exception

1. Confirm the vulnerability is a true positive and cannot be upgraded away
   immediately (pin the package or patch, and file an upgrade task).
2. Add one row to the table above with a CVE ID, package, affected versions,
   severity, a one-line justification, a review date, and the grantor.
3. Add `--ignore-vuln <CVE-ID>` (repeatable) to the scan step in
   `.github/workflows/ci.yml`.
4. Re-run `uv run --no-sync pip-audit --desc on --aliases on` locally to
   confirm CI passes with the exception.

Never allowlist a high or critical severity vulnerability without a linked
upgrade task and a near-future review date.
