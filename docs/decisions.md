# Decisions

One entry per decision that was not obvious from docs/DESIGN.md. Newest at the bottom.
Format: **D-nnn · title** — decision, why, consequence. Keep it to three lines.

**D-001 · Name and distribution** — Project is AIVIA; import package, CLI and distribution are all `aivia`,
env prefix `AIVA_`. Private repo, personal use, no PyPI publishing.
*Why:* one name everywhere avoids `pip install X / import Y` confusion; publishing adds no value here.
*Consequence:* packaging metadata stays minimal; `uv tool install .` and editable installs are the supported paths.

**D-002 · Single import root** — Everything lives under `src/aivia/`; no top-level `core`, `api`, `cli`,
`media`, `llm` packages.
*Why:* those names would collide in site-packages and shadow other libraries even in a private environment.
*Consequence:* install does not expose any generic module name; a packaging test asserts one top-level package.

**D-003 · Deterministic core, LLM at the edges** — Control flow is plain code; LLMs only produce validated
structured output inside stages and front-ends, never shell or arbitrary code.
*Why:* reproducibility, debuggability, cost control.
*Consequence:* every LLM interaction has a schema; failures surface as validation errors, not silent weirdness.

**D-004 · Providers behind interfaces, router owns policy** — Capabilities are selected by config; fallback,
budget checks, rate limits and circuit breaking live in the router.
*Why:* vendors and prices change constantly; business logic must not know names.
*Consequence:* adding a vendor is config or a YAML spec, not a refactor.

**D-005 · One paid path** — All generation goes through the media service, gated by a screening token from
prompt/content screening.
*Why:* a single place for cache, budget, cost, validation and policy means no bypass.
*Consequence:* paid services refuse to run without a screened prompt; tests assert the refusal.

**D-006 · Core never imports a module** — research, orders, publish and analytics are optional and register
their own config sections, stages and commands.
*Why:* the tool must be complete without automation modules.
*Consequence:* core features consume contracts (e.g. the fact list), not module code; an architecture test enforces it.

**D-007 · Package data vs user data** — Runtime assets (bundled OFL fonts, default config, prompts, templates,
platform and model data) ship inside the package and are copied to `$XDG_CONFIG_HOME/aivia/` once; user files
are never overwritten; upgrades only add missing defaults.
*Why:* `uv tool install` runs from an isolated environment, so repo-relative paths break; user edits must survive upgrades.
*Consequence:* runtime assets land in milestone 1; resolution goes through package resources, never CWD.

**D-008 · Publishing last** — The publishing module is deliberately the last milestone.
*Why:* explicit product decision; platform audits take calendar time and are tracked outside the plan.
*Consequence:* nothing in the core or in the final end-to-end test may depend on publishing; disclosure and
attestation checks are skipped with a recorded reason when the module is absent.

**D-009 · Final outputs are pinned** — Retention removes intermediates, cache, logs and expired personal data,
but never final renders unless `retention.delete_final_outputs: true`.
*Why:* deleting paid-for results after 30 days is data loss, not hygiene.
*Consequence:* assets carry a pinned flag; retention dry run lists what would go.

**D-010 · Determinism means visual identity** — Same composition plus same toolchain fingerprint renders
visually identical output (SSIM ≥ 0.99); byte identity is not claimed.
*Why:* ffmpeg build, encoder and thread count change bytes; provenance timestamps change them too.
*Consequence:* render cache keys include renderer, ffmpeg and font-set versions; goldens run only on the pinned image.

**D-011 · Receipt, not tax invoice** — Personalized orders can export a receipt with one flat tax estimate and
an explicit "not a tax invoice" footer. No country rules, no reverse charge, no multi-rate VAT.
*Why:* invoicing is a legal domain out of scope; silently pretending otherwise is worse.
*Consequence:* `orders.pricing` holds currency, amount and flat rate; receipts state their own limits.

**D-012 · Fixed role enum** — The core LLM role enum is `default, writing, factcheck, lyrics, planner,
reference_analysis, vision_qa, safety, translation_check, prompt_enhance`; modules may register more.
*Why:* an open string would make roles untestable and typos silent.
*Consequence:* unknown roles fail with the allowed list; module roles resolve only when registered.

**D-013 · Pinned toolchain for goldens** — Golden render tests run only on Linux in a pinned CI image with a
pinned ffmpeg and the bundled font set; CI installs ffmpeg for mock video too.
*Why:* otherwise golden tests flake or pass for the wrong reason.
*Consequence:* local runs of `-m golden` may differ; the authoritative result is CI.

**D-014 · Plan format is plain checkboxes** — `PLAN.md` items carry no ids, no dependency metadata and no
acceptance fields; ordering is the dependency graph, and the acceptance signal lives in the sentence plus a
test path in parentheses.
*Why:* a runner can take the first unchecked line verbatim; metadata would have to be re-parsed and kept in sync.
*Consequence:* items may only depend on items above them; restructuring requires an explicit replan pass, and
the agent may only append.

**D-015 · SQLite pragmas everywhere** — WAL, a configurable busy timeout and foreign keys are set on every
connection.
*Why:* workers, the API and the CLI write concurrently; the default journal mode yields `database is locked`.
*Consequence:* a migration test asserts the pragmas; concurrency tests exercise multiple writers.

**D-016 · Sandboxed templates** — Prompt and provider-spec Jinja2 templates render in a sandboxed environment
in strict-undefined mode.
*Why:* templates are user-editable and shareable; an unsandboxed engine is a code-execution surface.
*Consequence:* sandbox violations surface as validation errors naming the template.

**D-017 · Secrets never sit next to the data** — The token-store encryption key comes from the secrets loader;
tokens are never written beside their key.
*Why:* a copied data directory must not be a usable credential dump.
*Consequence:* credential rotation is supported without restart, logged without values.

**D-018 · SSRF means resolve-then-pin** — Outbound fetching resolves the host, validates every resolved
address (including numeric encodings, IPv6-mapped forms and metadata endpoints), connects to the validated IP
and re-validates on each redirect.
*Why:* blocking literal hostnames misses DNS rebinding and IP encodings.
*Consequence:* page fetch, reference downloads, webhooks and search results share one guard with a fuzz test.

**D-019 · Voice cloning needs explicit consent** — Cloning requires a consent flag on the character; synthetic
voice metadata is attached to outputs.
*Why:* it is someone's voice; the default must be "no".
*Consequence:* without consent the path raises a content policy error; the exact voice id is pinned per version.

**D-020 · Moved work rather than forward dependencies** — When an item turned out to need later work, the work
moved later instead of pointing ahead; the three historical moves were replay, subtitle re-burn and account
reference sources.
*Why:* forward dependencies make a linear queue unschedulable.
*Consequence:* subtitle re-burn now lives after the renderer; account-as-reference lives in publishing; the
font bundle moved into milestone 1 so early subtitle work can use it.

**D-021 · CI items are delivered as proposals** — Anything under `.github/workflows/` is written to
`ci-proposals/<name>.yml` and applied by a human with one `git mv`; the item is ticked when the proposal is
delivered.
*Why:* the agent token cannot push to workflow paths, and blocking those items forever is worse than proposing them.
*Consequence:* two plan items (CI workflow with ffmpeg, dependency scanning) are complete when their proposal exists.

**D-022 · The loop owns the checkbox** — The runner ticks the item when it creates the session branch; the agent
leaves it as is and only reverts it to unchecked when it ends without completing the work.
*Why:* two writers on one line cause PLAN.md conflicts on every PR.
*Consequence:* conflict resolution always takes main's version of every line except the agent's own tick.