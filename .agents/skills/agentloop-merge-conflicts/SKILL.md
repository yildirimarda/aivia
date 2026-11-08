---
name: agentloop-merge-conflicts
description: Resolve a rejected push or merge conflicts on your work branch. Use when git push is rejected because main moved, when merging origin/main reports conflicts, or when PLAN.md, a lockfile or a generated file conflicts.
license: MIT
metadata:
  managed-by: agentloop
---

# Push rejected / merge conflicts

You cannot force-push — it is denied for you — so never rebase a branch that
is already pushed. Merge instead.

## Push rejected because main moved

```bash
git fetch origin
git merge origin/main      # not rebase: no force-push needed afterwards
# resolve conflicts (below), run lint + tests again, then:
git push
```

## Resolving by file type

- **PLAN.md** — take MAIN's version of every line, except your own item,
  which keeps its `[x]`. Never let a resolution untick another item: its code
  is already merged on main, and an untick makes the loop redo it. Items you
  appended in this session stay at the end of their milestone.
- **Lockfiles** (`uv.lock`, `package-lock.json`, `Cargo.lock`, `poetry.lock`)
  — never hand-merge. Take main's version, regenerate with the project's tool
  (`uv lock`, `npm install`, `cargo generate-lockfile`) so your dependency
  changes are re-applied, and commit the result.
- **Generated files** (snapshots, codegen output) — take main's version and
  regenerate.
- **Source files** — understand both sides; keep main's behaviour plus your
  change, and re-run the tests that cover the conflicted code.

## Before pushing

- `git diff --name-only --diff-filter=U` prints nothing, and no conflict
  markers remain: `git grep -n '^<<<<<<< '` finds nothing.
- Lint and tests pass. Commit the merge, push, stop.
