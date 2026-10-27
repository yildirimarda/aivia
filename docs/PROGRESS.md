# Progress

Running log of what is built, newest at the bottom. The queue lives in `PLAN.md`; the long-form
design is `docs/DESIGN.md`; the rationale for non-obvious choices is `docs/decisions.md`.

## Milestone 1: Toolchain, packaging and smoke tests

Status: in progress.

- Package skeleton, console script, bundled OFL font and package data are in place.
- Toolchain configuration (ruff, mypy, pytest markers), Makefile targets and the CI workflow are in place.
- Structured logging, log context, the error taxonomy, secrets loading, redaction, path resolution, ULIDs
  and the root CLI are implemented with offline tests.
- Architecture note (`docs/DESIGN.md`) and decisions log (`docs/decisions.md`) are written.
- Progress log started in this file.

## Foundational constraints recorded

- **Deterministic core.** Control flow is plain, resumable code: stages, format recipes and validated
  plans. LLMs run only inside stages and front-ends and return validated structured output; they never
  choose control flow or execute shell commands. Recorded in `DESIGN.md` §1 and §8, and as D-003.
- **No module imports in core.** The core never imports an optional module (`research`, `orders`,
  `publish`, `analytics`). Modules register their own config sections, stages and commands, and core
  features consume contracts rather than module code. Recorded in `DESIGN.md` §6 and §8, and as D-006.

## Next

Continue the remaining milestone 1 items in `PLAN.md` order: documentation policies, secret scanning,
config coverage lint, plan lint, first-run bootstrap, `init`, the installed-tool smoke and live-test gating.
