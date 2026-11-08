---
name: agentloop-ci-proposals
description: Deliver plan items about CI, GitHub Actions or workflow files. Use when an item asks to add, change or fix anything under .github/workflows/ — you cannot edit those files, so you write the change as a proposal under ci-proposals/ for a human to apply.
license: MIT
metadata:
  managed-by: agentloop
---

# CI changes go through proposals

`.github/workflows/` is off-limits to you: your config denies edits there and
GitHub rejects your token's pushes to that path. That is deliberate — an agent
that can edit its own merge gate can learn to pass by weakening it. CI items
are still legitimate work; this is how you deliver them.

## Steps

1. Write the complete proposed workflow to `ci-proposals/<name>.yml`, where
   `<name>` is the file it should become (`ci.yml` to replace the existing
   workflow, `nightly.yml` for a new one). Files there are inert — GitHub
   only runs workflows from `.github/workflows/`.
2. Start the proposal with a comment header: what changes versus the live
   workflow, and the exact command that applies it:
   `git mv ci-proposals/<name>.yml .github/workflows/<name>.yml`
3. Validate it — `python3 -c "import sys, yaml; yaml.safe_load(open(sys.argv[1]))" ci-proposals/<name>.yml`
   — and run every command the proposal adds, locally, so the human applies
   something that already works.
4. In the PR description explain what and why, and repeat the `git mv` line.
5. Leave the item ticked: proposing IS the deliverable. A human applies it in
   a separate commit.

## Don't break the merge gate

- Keep the job names branch protection requires (usually `ci`). A renamed
  required job never reports, and then nothing can merge. If a rename is the
  point of the item, say so loudly in the PR description.
- Required checks must run on every pull request — never require release- or
  deploy-only jobs.
- Prefer adding a step to an existing job over adding a new required job.
- Pin third-party actions to a major version (`@v4`), as the existing
  workflows do.
