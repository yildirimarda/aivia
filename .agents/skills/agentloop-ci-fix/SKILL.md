---
name: agentloop-ci-fix
description: Diagnose and fix a failing CI run on your pull request. Use when the prompt contains a failing CI log, says CI failed, or a check on your PR is red — read the log, reproduce with the workflow's own commands, fix the code (never the tests) and push to the same branch.
license: MIT
metadata:
  managed-by: agentloop
---

# Fixing a red CI run

You are on the pull request's branch. The fix goes on THIS branch: commit and
push, and the open PR updates itself. Never create a new branch or PR for a CI
fix, and never touch `main`.

## 1. Read the log like a compiler would

- The FIRST error is usually the cause; later errors are often fallout. Find
  the failed job and step, then the first `error`, `FAILED`, `Error:` or
  non-zero exit line.
- Note the platform and toolchain of the failed job (`windows-latest`,
  `macos-14`, a pinned minimum version such as an MSRV or the oldest Python).
  It decides whether you can reproduce it here (step 3).
- Log missing or truncated? Don't dig through git history or PR metadata —
  go straight to step 2.

## 2. Reproduce with the workflow's own commands

Open EVERY file in `.github/workflows/` (not only `ci.yml`) and run the check
commands they define — lint, format check, type check, tests — in that order,
exactly as written (same flags, same working directory). That is the fastest
faithful reproduction. Clip the output: `<cmd> 2>&1 | tail -40`.

## 3. When it does not reproduce here

This container is Linux. A failure from Windows, macOS or an older toolchain
usually will NOT reproduce locally — do not conclude "all green, nothing to
fix". Fix it from the log. Typical causes:

- platform-only code or imports (and their tests) without a guard:
  `#[cfg(unix)]`, `if sys.platform != "win32"`, `process.platform`
- path separators, drive letters, `\r\n` line endings, case-insensitive
  file systems
- `chmod`, symlinks, signals or shell syntax that only exist on unix
- APIs newer than the minimum supported version the job pins

## 4. Common failure classes

| Failure | Fix |
|---|---|
| Formatter check | Run the project's formatter (`ruff format .`, `cargo fmt`, `npx prettier -w .`) and commit the result |
| Stale lockfile (`--frozen`, `--locked` or `npm ci` fails) | Regenerate it with the project's tool (`uv lock`, `npm install`, `cargo update -w`) and commit it |
| Lint | Fix the code. No blanket ignores; a targeted ignore needs a one-line reason |
| Test fails only in CI | Look for test-order dependence, time zone, locale, unseeded randomness, network access, missing env vars |
| Flaky test | Make it deterministic (seed, fake clock, no sleeps) — never retry until green |

## 5. Rules

- Fix the CODE. Never delete, skip, `xfail` or weaken a test to get green. If a
  test is genuinely wrong, fix it and explain why in a PR comment.
- Never edit `.github/workflows/`. If the workflow itself is broken, write the
  fix under `ci-proposals/` (skill `agentloop-ci-proposals`) and say so in a
  PR comment.
- Don't modify `PLAN.md` in a fix session. One exception: when merging `main`
  into this branch conflicts on `PLAN.md`, take main's version of every line
  except this PR's own item, which keeps its `[x]`.
- Re-run the lint and test commands until they pass, then commit
  (`fix: ...`), push, and stop. Do not merge and do not wait for CI.
