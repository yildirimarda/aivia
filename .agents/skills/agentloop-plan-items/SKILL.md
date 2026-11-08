---
name: agentloop-plan-items
description: Write or edit PLAN.md items — the exact checkbox format the loop parses and what makes an item good. Use when drafting a plan (--init), restructuring it (--replan), appending discovered work, or splitting an item that is too big.
license: MIT
metadata:
  managed-by: agentloop
---

# Writing PLAN.md items

`run.sh` hands a session the first unchecked item verbatim. The item's quality
decides whether that session takes 8 turns or 40 and breaks halfway.

## Format (parsed by bash — keep it exact)

```text
# Plan

## Milestone 1: <name>

- [ ] Add config loading from environment variables, with validation
  - fail fast with one message naming every missing variable
  - document each variable in README.md
- [x] A finished item

## Discovered
```

- An item line is dash, space, `[ ]`, space, text. Done is `[x]`.
- Optional DETAILS: lines indented by two spaces directly under an item. The
  loop passes them to the session as acceptance criteria — constraints, edge
  cases, interfaces, files to touch. A blank line ends them. Never indent a
  checkbox under an item: a nested checkbox is a separate item.
- Never write a checkbox-shaped line inside an HTML comment in PLAN.md: the
  loop greps for that shape and would hand it out as work.
- `## Discovered` is the last section.

## A good item

- **One pull request.** If describing it needs "and", it is probably two.
- **Independently testable.** It names an outcome a test or command can
  verify. "Improve performance" is not an item; "Add a benchmark asserting a
  10k-row load stays within 2x of the baseline" is.
- **Ordered.** It depends only on items above it.
- **An instruction, not a topic.** "Add retry with exponential backoff to the
  S3 upload path", not "Reliability".
- **Concrete about what the next session cannot guess:** formats, limits,
  error behaviour, which module owns it.

## Sizing

Split an item that touches more than one subsystem, bundles a migration with
the feature that uses it, or would not fit one focused session. A first plan
has 8-20 items: plan the first working version, not the whole product.

## Appending discovered work (normal sessions)

- At most 3 new items per session; none is fine. Never pad the plan.
- Append to the END of the milestone they belong to, or under `## Discovered`.
- Append only: never delete, reorder, reword or untick existing items, and
  never edit anything above your own item.
- Something must happen BEFORE your item can work? Do not reorder — open the
  PR with the `blocked` label and explain the dependency.
- List each addition and its reason in the PR description.

Only a `--replan` session restructures the plan: promoting, splitting,
reordering, deleting (say why in the commit message).
