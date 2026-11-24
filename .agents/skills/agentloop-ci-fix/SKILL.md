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

## Name the cause before you edit anything

Write down, in one sentence, the exact first error line from the log (copy it
word for word) and why it causes the failure. If you cannot explain it, you do
not have a diagnosis yet — read the log again instead of editing.

- An error that names a program, flag, command, file or encoding ("unknown
  program", "not a char boundary", "command not found", "invalid arguments")
  is taken at its word. Never dismiss a log line as noise, interleaved output
  or "probably flaky".
- **Passes here, fails in CI is not "flaky".** This container is not the
  runner: the OS, the shell, the coreutils (some runners ship uutils instead of
  GNU), tool and compiler versions, locale, CPU count and installed packages all
  differ. Compare them (`uname -a`, the versions printed at the top of the CI
  log, `command -v` / `--version` of the failing tool) before suspecting a race.
- Only call a failure a race when the same test fails and passes on the same
  runner, and say how you know.

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
- Never make a check pass by weakening it. These are not fixes: turning an
  error into a warning or an empty result, catching and swallowing a failure
  ("tolerate" it), loosening a validator, assertion or exit-code check,
  `|| true`, `continue-on-error`, skipping a step or a job. A check that was
  right to fail must keep failing until the code is right. If the only way to
  green is weakening a check, stop and hand over (see below).
- If your fix leaves the SAME error in the log, your diagnosis was wrong. Do
  not stack more changes on it: revert the attempts that did not change the
  failure, re-read the log from the first error and compare the environment.
  Every commit must follow from a line of the log; unrelated "while I was
  here" changes make the next failure harder to read.
- Stuck? Say so instead of guessing: leave a PR comment with the exact error,
  what you tried, what you think is wrong and what you could not check, and
  add the label `blocked` (skill `agentloop-pull-request`). A clear hand-over
  is a good result; a green CI built on a weakened check is not.
- Never edit `.github/workflows/`. If the workflow itself is broken, write the
  fix under `ci-proposals/` (skill `agentloop-ci-proposals`) and say so in a
  PR comment.
- Don't modify `PLAN.md` in a fix session. One exception: when merging `main`
  into this branch conflicts on `PLAN.md`, take main's version of every line
  except this PR's own item, which keeps its `[x]`.
- Re-run the lint and test commands until they pass, then commit
  (`fix: ...`), push, and stop. Do not merge and do not wait for CI.
