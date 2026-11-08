---
name: agentloop-dependencies
description: Add, remove or upgrade a project dependency safely. Use when an item needs a new library or tool, when you edit pyproject.toml, package.json, Cargo.toml, go.mod or a similar manifest, or when a lockfile check fails.
license: MIT
metadata:
  managed-by: agentloop
---

# Changing dependencies

## Before adding one

- Can the standard library or an existing dependency do it? Prefer that.
- Check the exact package name before installing: typosquats exist, and
  look-alike names can be unrelated projects (PyPI `graphifyy` is the
  knowledge-graph tool; `graphify` is not). Prefer maintained packages with
  a license compatible with this project's.
- Runtime or development only? Test and lint tools go in the dev group.

## Doing it

- Use the project's own tool; never hand-edit a lockfile:
  `uv add <pkg>` / `uv add --dev <pkg>`, `npm install <pkg>` (`--save-dev`),
  `cargo add <crate>`, `go get <module>`, `poetry add <pkg>`.
- Commit the manifest AND the lockfile in the same commit. CI installs with
  `--frozen` / `--locked` / `npm ci`, which fails on a stale lockfile.
- Follow the pinning style already in the manifest (ranges vs exact pins).
- Need a system package the container lacks? You cannot install it
  permanently: say in the PR description what `Dockerfile.agent` needs.

## In the PR description

One line per new dependency: what it is for, and why the standard library or
an existing dependency was not enough.
