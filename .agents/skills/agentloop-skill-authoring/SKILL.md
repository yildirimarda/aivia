---
name: agentloop-skill-authoring
description: Record what you learned about this project as a skill for future sessions. Use when you worked out something non-obvious that the next session would otherwise rediscover — how to run, test, debug, seed or deploy part of the project, or a trap that cost you several attempts.
license: MIT
metadata:
  managed-by: agentloop
---

# Recording a project skill

Every session starts cold. A project skill is how a hard-won procedure
survives: later sessions see its name and description up front, and load the
body only when it applies.

## Worth recording

- A procedure you had to work out: "integration tests need
  `docker compose up -d db` and `DB_URL=...`", "regenerate the API client
  with X after editing the schema".
- A trap that cost you attempts: "the Windows job fails on CRLF in test
  fixtures — keep .gitattributes in sync".
- Verified: you ran it in this session and it worked.

## Not worth recording

- Anything already in README, AGENTS.md or an existing skill — update that
  skill instead of adding a near-duplicate.
- One-off details of the current item, guesses, or anything secret.

## Where and how

- Path: `.agents/skills/<name>/SKILL.md` — every engine reads this location.
  The name is lowercase letters, digits and single hyphens, and matches its
  directory (`run-integration-tests`, not `Integration_Tests`).
- Never create or edit `agentloop-*` skills: agentloop owns and replaces them.
- Frontmatter first, then a short body (under about 60 lines):

```markdown
---
name: run-integration-tests
description: Run the integration test suite locally. Use when tests under tests/integration fail or before changing the database layer.
---

# Running integration tests

1. `docker compose up -d db` (listens on port 5433)
2. `DB_URL=postgresql://test@localhost:5433/test uv run pytest -q tests/integration`

Gotcha: the first run takes ~40s while migrations apply.
```

- The description decides when a session loads the skill: say WHAT it does
  and WHEN to use it, with the words a future task would contain. At most
  1024 characters.
- At most one new skill per session. Name it under "Skills recorded" in the
  PR description.
