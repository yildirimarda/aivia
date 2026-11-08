---
name: agentloop-testing
description: Write tests that prove a plan item works without being flaky or slow. Use when adding or extending tests, deciding what to test, debugging a failing or flaky test, or when the test suite is too slow to run locally.
license: MIT
metadata:
  managed-by: agentloop
---

# Testing an item

CI is the merge gate. A test that cannot fail proves nothing; a test that
fails at random turns the gate into a dice roll and burns fix sessions.

## What to test

- The behaviour the item promises — including its details/acceptance
  criteria — not the implementation. Assert on outputs, return values,
  files, exit codes and error messages.
- The edge cases the item names, plus the obvious ones: empty input, the
  boundary value, invalid input and its error.
- A bug fix starts with a test that reproduces the bug and fails; then the
  fix makes it pass.

## Keep it deterministic

- Seed every random generator; inject or freeze the clock; never `sleep` to
  wait for something.
- No real network or external services — fakes, fixtures, local servers.
- Write files under a temp directory (`tmp_path`, `mkdtemp`), never into the
  repository.
- Performance: assert relative ratios (min of several runs), never absolute
  wall-clock limits — shared CI runners vary wildly.
- No dependence on test order, current directory, locale or time zone.

## Running

- Iterate on the narrowest target: `pytest -q -x tests/test_x.py::test_y`,
  `cargo test name`, `npx vitest run path`. Clip the output: `2>&1 | tail -30`.
- Run the full lint and test commands from AGENTS.md once before pushing.
- Shell commands time out after about two minutes. If the full suite is
  slower, run the affected tests locally, let CI run the rest, and say so in
  the PR description.

## Never

- Delete, skip, `xfail` or loosen an assertion to get green — fix the code.
- If a test is genuinely wrong, fix it and explain why in the PR description.
