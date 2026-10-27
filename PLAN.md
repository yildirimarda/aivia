# Plan

<!--
  This file is the agent's queue AND its notebook.

  `./run.sh` finds the first unchecked item, hands that exact text to the
  agent, and the agent ticks it off in the same pull request. While working,
  the agent may APPEND newly discovered work — to the end of the milestone it
  belongs to, or to "## Discovered" below. It may never delete, reorder or
  reword an existing item.

  Format rules — `run.sh` parses these lines, so keep them exact:
    · headings:  ## Milestone N: name
    · todo:      - [ ] text
    · done:      - [x] text

  Writing good items is the highest-leverage thing you do here:
    · One pull request per item. If it needs two PRs, split it.
    · Independently testable. If you can't test it, it's not an item.
    · Ordered. Each item may only depend on items above it.
    · Written as an instruction, not a topic.
        good: "Add config loading from environment variables with validation"
        bad:  "Config"

  Starting a new project? Don't write this by hand:
      ./run.sh --init "what you're building, constraints, stack"
  The agent drafts the plan, opens a PR labelled `plan`, and stops so you can
  review it. Reviewing the plan is where your judgement pays off.

  Plan grown messy after a few dozen items? Tidy it up:
      ./run.sh --replan
  That is the only mode allowed to promote, split, reorder and delete items.

  Delete this comment block once you have real content.
-->

## Milestone 1: Foundations

- [ ] Set up the project layout, dependency manifest and a placeholder test so the test command passes
- [ ] Add configuration loading from environment variables, with validation and clear errors on missing values
- [ ] Add structured logging with a configurable level

## Milestone 2: Core

<!-- Add your items here as real checkbox lines: a dash, space, brackets,
space, then the text ("dash [ ] item text"). This placeholder deliberately
avoids writing one at the start of a line — the loop greps for that shape and
would hand it to the agent as a task. -->

## Milestone 3: Hardening

<!-- Add your items here. -->

## Discovered

<!-- Work the agent found while building. Move items from here into the
     milestone where they belong with `./run.sh --replan`. -->
