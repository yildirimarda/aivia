# How you work

`PLAN.md` is the source of truth for what this project needs. Each session you
are given exactly one item from it. Do that item, mark it done, record any new
work you discovered along the way, open a pull request, and stop.

## Workflow

1. `run.sh` has already created and switched to a fresh work branch for this
   session. Stay on it — never switch branches, never touch `main`.
2. Implement **only** the item you were given. Do not start other items, do not
   refactor unrelated code, do not "improve things while you're here".
3. Write or extend tests that prove the item works.
4. Run the project's lint and test commands (see Project reference below).
   Fix failures until everything is green. If you changed dependencies
   (pyproject.toml or equivalent), refresh the lockfile (`uv lock`) and
   COMMIT it — CI rejects a stale lockfile.
5. `PLAN.md` on your branch already shows the item ticked — the loop does
   this when it creates the branch. Leave it as is. The tick travels with
   the branch: it reaches `main` only if the PR merges. Only if you end the
   session without completing the item, revert it to `- [ ]` before pushing.
6. Record any new work you discovered — see "Growing the plan" below.
7. Commit using Conventional Commits: `feat:`, `fix:`, `chore:`, `docs:`,
   `refactor:`, `test:`. Small, single-purpose commits. Before committing,
   look at `git status`: NEVER commit generated artifacts — bytecode
   (`__pycache__`, `*.pyc`), tool caches (`.pytest_cache`, `.ruff_cache`,
   `.mypy_cache`), reports, coverage files, build output. If one shows up,
   add it to `.gitignore` in the same commit instead.
8. `git push -u origin HEAD`. If the push is rejected because `main` moved,
   merge `origin/main` into your branch. If `PLAN.md` conflicts, take main's
   version of every line except your own item's tick — never let a conflict
   resolution untick another item (its code is already merged on main).
9. `gh pr create --fill --label automated`
10. **STOP.** Do not wait for CI, do not merge, do not cut a release, do not
    poll the PR, do not start the next item. GitHub handles everything after
    the PR is open.

## Growing the plan

You are expected to add to `PLAN.md`, not just consume it. While implementing an
item you will often find work the plan does not cover: a missing edge case, a
required migration, error handling that was hand-waved, a refactor a later item
will depend on. Record it instead of silently doing it or silently ignoring it.

**How to add:**

- Append new `- [ ]` items to the **end** of the milestone they belong to. If
  you are not sure where they fit, put them under `## Discovered`.
- Write them in the same style as existing items: concrete, independently
  testable, one pull request each. "Add retry with backoff to the S3 upload
  path", not "Reliability".
- **Append only.** Never delete, reorder, reword or uncheck an existing item,
  and never touch an item above the one you are working on.
- List what you added and why in the pull request description.

**Limits:**

- At most 3 new items per session. If you found nothing, add nothing — do not
  pad the plan to look productive.
- Do not add work that is merely nice to have. If you would not defend it to a
  reviewer, leave it out.
- If you discover something that must be done **before** your current item can
  work, do not reorder the plan. Stop, open the PR with the `blocked` label, and
  explain the dependency.

Only a `--replan` session may restructure `PLAN.md` freely.

## Understanding the codebase

This repository has a knowledge graph of itself. **Query the graph before you
read source files** — it is dramatically cheaper than reading files, and it is
how this project stays affordable.

- `graphify_query_graph` — ask a question in plain language, get back the
  relevant subgraph with `file:line` citations. Start here.
- `graphify_get_neighbors` — what a given symbol calls and is called by.
- `graphify_shortest_path` — how two symbols are connected.
- `graphify_god_nodes` — the most-connected symbols. Use this to orient
  yourself in an unfamiliar area.
- `graphify_get_node` — full detail on one symbol.

Only open a file directly once the graph has told you which file and which
lines matter. Never read a whole directory to "get oriented" — ask the graph
instead.

Two caveats: edges marked `INFERRED` are model-generated guesses, so verify
them against the real file before relying on them. And the graph reflects the
last index, so if you just created a file, the graph does not know about it yet.

## Output discipline (token budget)

Every byte a command prints is re-sent to the model on every following step.
Waste here compounds into slower sessions, forced context compaction (which
loses detail), and burned request quota.

- Run tests quietly and clip the output: `uv run pytest -q ... 2>&1 | tail -30`.
  Only re-run verbosely for the ONE failing test you are debugging.
- Never dump whole files. `grep -n` to locate, then read only the relevant
  range. Ask the knowledge graph before opening any file at all.
- Never print whole logs, lockfiles, or generated artifacts. `tail`, `head`,
  `wc -l` and targeted `grep` answer almost every question.
- Prefer one command that answers the question over several exploratory ones.
- Performance tests: never assert absolute wall-clock durations — they flake
  on shared CI runners. Assert relative ratios (min-of-runs at microsecond
  scale) or gate absolute checks behind an opt-in env var.

## Prohibited

- No direct pushes to `main` or `release/*`.
- No `git push --force`.
- Do not modify anything under `.github/workflows/`.
- Do not restructure `PLAN.md`. You may tick your own item and append new ones
  (see "Growing the plan") — nothing else. If the plan itself is wrong, say so
  in the PR description and stop.
- Do not delete, skip (`skip`/`xfail`) or disable tests to make them pass.
  If a test fails, fix the code — not the test. If a test is genuinely wrong,
  explain why you changed it in the PR description.
- No `terraform apply` or `terraform destroy`. Only `fmt`,
  `init -backend=false`, `validate`, `plan`.
- Never commit secrets, tokens or `.env` contents.
- When adding a new dependency, justify it in the PR description.

## CI changes

You cannot modify anything under `.github/workflows/` — it is denied by
config, and GitHub rejects your token's pushes to that path anyway. But plan
items about CI are legitimate. Handle them like this:

1. Write the full proposed workflow file (or the diff) under
   `ci-proposals/<short-name>.yml`. Files there are inert — GitHub only
   executes workflows from `.github/workflows/`.
2. In the PR description, explain what the change does and why, and include
   the one command a human runs to apply it:
   `git mv ci-proposals/<name>.yml .github/workflows/<name>.yml`
3. Tick the plan item as done — proposing IS the deliverable for CI items.
   A human applies it in a separate commit.

## When you get stuck

If you cannot finish the item in 3 attempts, stop pushing on it. Commit what
you have, open the PR with the `blocked` label instead of `automated`, and
write in the description: what you tried, what happened, and what you need.
Leave the `PLAN.md` checkbox unchecked. Then stop.

## Project reference

<!-- Fill this in for your own project. If these commands are wrong, the
     agent runs the wrong thing and wastes turns discovering that. -->

- Tests:        `uv run pytest`
- Lint:         `uv run ruff check .`
- Format:       `uv run ruff format .`
- Dependencies: `uv sync` / `uv add <package>`
- Source:       `src/`
- Tests dir:    `tests/`
- Spark jobs live in `src/jobs/`. Test locally with
  `spark-submit --master local[2] src/jobs/<job>.py`
- Terraform:    `infra/`
