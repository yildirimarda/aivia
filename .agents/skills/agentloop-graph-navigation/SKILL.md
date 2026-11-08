---
name: agentloop-graph-navigation
description: Find where to work in the codebase through its knowledge graph instead of reading whole files. Use at the start of a task, before changing a function or class (to see its callers), and whenever you would otherwise grep around or open files to get oriented.
license: MIT
metadata:
  managed-by: agentloop
---

# Navigating with the knowledge graph

`run.sh` refreshes `graphify-out/graph.json` before every session. Asking the
graph costs a few hundred tokens; reading files to get oriented costs
thousands, on every following step.

## Two ways to ask — same graph

| Need | MCP tool | CLI (works in any engine) |
|---|---|---|
| Code relevant to a question | `query_graph` | `graphify query "<question>" --budget 1500` |
| One symbol and its connections | `get_node`, `get_neighbors` | `graphify explain "<symbol>"` |
| How A reaches B | `shortest_path` | `graphify path "<A>" "<B>"` |
| The core abstractions | `god_nodes` | `graphify god-nodes` |

MCP tool names differ per engine (`graphify_query_graph`,
`mcp__graphify__query_graph`, ...). If the MCP tools are missing, use the CLI.

## Workflow

1. The prompt may already contain a "Where to start" excerpt from the graph —
   begin there. Otherwise query with the item's own words.
2. Open only the files and line ranges the answer cites
   (`sed -n '120,180p' path`), never whole directories.
3. Before changing a symbol, see who uses it (`graphify explain "<symbol>"`)
   and update or test those call sites as well.
4. Use grep for exact strings: error messages, config keys, CLI flags.

## Caveats

- `INFERRED` edges are resolved guesses: confirm them in the source first.
- The graph reflects the last index. Files you created or changed in this
  session are not in it yet — read those from disk.
- Exit code 3 means an ambiguous name (retry with the `path::symbol` form it
  suggests); 4 means no match (rephrase with names from the code, or start
  from `graphify god-nodes`).
