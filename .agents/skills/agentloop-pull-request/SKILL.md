---
name: agentloop-pull-request
description: Self-review your change and open the pull request the loop expects. Use right before gh pr create — or when you are stuck and must hand over to a human — to pick the right label and write a description a reviewer can trust.
license: MIT
metadata:
  managed-by: agentloop
---

# Opening the pull request

The loop runs you non-interactively: nobody answers questions. Never finish
with "let me know if you'd like me to commit / push / open a PR" — that ends
the run with your work uncommitted. Committing, pushing and opening the PR is
part of the item. (The loop's own `PLAN.md` tick is already committed; it is
never a reason to hold back your commit.)

## Self-review first (two minutes, saves a CI round)

- `git diff origin/main --stat`: only files the item needs. Revert drive-by
  refactors and unrelated formatting churn.
- `git status`: no generated artifacts (`__pycache__`, `.pytest_cache`,
  coverage, build output, logs). If one shows up, add it to `.gitignore`.
- No secrets, tokens, `.env` contents, debug prints, commented-out code or
  context-free TODOs.
- Tests added or extended for the new behaviour; lint, format and tests pass
  with the commands from the Project reference in AGENTS.md.
- User-visible behaviour changed? Update README/docs in the same PR.

## Labels — exactly one

| Label | When |
|---|---|
| `automated` | You implemented the item. It auto-merges when CI is green. |
| `blocked` | You could not finish: 3 failed attempts, a missing prerequisite, or the item is wrong. Revert your item's tick to `- [ ]` first. A human takes over. |
| `plan` | Only in `--init` / `--replan` sessions. Never on an item PR. |

## Description

Write the body to a file — no quoting trouble — then:

```bash
gh pr create --fill --label automated --body-file /tmp/pr.md
```

Sections:

- **What & why** — the item, in one or two sentences.
- **How it was tested** — the commands you ran and their result.
- **Plan additions** — each item appended to PLAN.md and why (or "none").
- **New dependencies** — each with a one-line justification (or "none").
- **Skills recorded** — a project skill you added or updated (or "none").
- **Notes for the reviewer** — risks, follow-ups, anything surprising.

For `blocked`, replace the middle sections with: what you tried, what
happened (the exact error), and what you need from a human.

Title: a Conventional Commit (`feat: ...`, `fix: ...`). It becomes the
squash-merge commit and drives the release notes.
