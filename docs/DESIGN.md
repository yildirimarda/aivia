# AIVIA — Design

> Long-form design. Rules for the agent: AGENTS.md · task queue: PLAN.md · decision log: docs/decisions.md ·
> progress log: docs/PROGRESS.md.
> Generated docs (do not hand-edit): docs/CONFIG.md (from config schema), docs/PROVIDERS.md (from
> ProviderMeta), docs/platform-onboarding.md.

## 1. What we are building

A **configurable, AI-backed generic image and video generation assistant** — audio, music and songs are
first-class outputs too. You can ask for any image, GIF, video, voice-over or song from one prompt, from a
natural-language instruction with your own assets, or from a reference example; the assistant plans and
produces it through interchangeable AI providers. Everything that can change — vendors, models, formats,
topics, budgets, thresholds, languages — is configuration, never code. Every run reports what it cost.

It is a personal tool: a private repo, no PyPI publishing, installed with `uv tool install .` or an
editable install. Distribution metadata stays minimal; correctness, reproducibility and cost control matter.

| Layer | Capability | You provide | Typical output |
|---|---|---|---|
| Core | Standalone generation (CLI/API/batch) | A prompt or file for one capability | One image, video, voice, song or transcript + cost report |
| Core | Natural-language assistant | Assets + free-text instruction, refined conversationally | Any composed image/GIF/video/audio; savable as a template |
| Core | Reference-driven generation | Example image/video/audio/text + a topic | New content with the same structure and style |
| Core | Characters and brand kits | Character + brand kit registered once | Consistent look, voice and music |
| Module | Content pipeline | Content profile + schedule | Researched content, reviewed and published |
| Module | Personalized orders | Order data (name, occasion, message, own image/audio) | Song cards delivered on a date |

**Design stance.** The control flow is deterministic, resumable code: stages, format recipes, validated
plans. LLMs run *inside* stages and as front-ends (planner, profile builder, reference analyzer) and only
ever produce validated structured output. They never run arbitrary code or shell commands. Every paid call
goes through the provider router and the cost tracker. This is what keeps the system debuggable,
reproducible, budget-safe and cheap to run.

**Priority order.** (1) Generic image/video/audio generation as a standalone capability. (2) The content
pipeline on top. (3) Advanced entry modes: references, characters, natural language, orders. (4)
Unattended operation. (5) **Publishing last, by explicit decision.** (6) Analytics and hardening.

## 2. Entry modes and pipeline shape

    Entry (command | instruction | reference | schedule | order)
      -> Input resolution    (direct prompt, instruction planning, reference analysis, topic research, order validation)
      -> Writing             (script / lyrics / quote text / captions, fact-check, media prompts)
      -> Audio               (TTS, music/song, alignment, subtitles)
      -> Visuals             (provided assets, stock, AI image, AI video, edits)
      -> Composition+Render  (one deterministic renderer: video, GIF, image, audio, web card)
      -> Review              (approve / reject / regenerate)          [optional]
      -> Output handling     (save, publish, or deliver)
      -> Analytics           (cost reports, metrics, feedback)

A **format recipe** selects which steps run. Skipped stages produce no cost event and no artifact — stages
are skipped, never faked.

## 3. Languages and platforms

- Languages: English, German, Turkish. Adding one is a config change. Each target script is **written
  natively** from the same fact base, not machine-translated. Text rendering is bidi-ready for a future
  RTL language.
- Platforms (publishing is deliberately the last milestone): Instagram Reels, YouTube Shorts, TikTok, X,
  plus a Telegram channel as the first integration target because it needs no app audit.
- One account per platform per language (`youtube/en`, `tiktok/tr`). Mixing languages on one account is a
  configuration error, not a runtime choice.

## 4. Glossary

Every task, config key and module uses these words with exactly these meanings.

| Term | Meaning |
|---|---|
| **Provider** | An implementation of one capability interface (LLM, image, video, TTS…). Declares capabilities, pricing metadata and rate limits. |
| **Provider registry** | Maps config `type` strings to classes. Built-ins by name; custom ones by dotted path. |
| **ProviderMeta** | Capability flags, supported options, pricing (unit + USD), rate limits, quality rank, licensing fields, commercial-use notes. |
| **Router** | Ordered provider chain plus a strategy (`first_available`, `cheapest`, `best_quality`, `round_robin`, `weighted_ab`), with fallback, capability filtering, budget checks, rate limiting and a circuit breaker. |
| **Capability** | A named operation the system needs (e.g. `video_gen`, `tts`). Chains are configured per capability. |
| **Content profile** | Reusable description of what to produce: format, topic mode + brief, niche, keywords, liked/disliked examples, sources, tone, style bible, voice, languages, accounts, visual strategy and config overrides. |
| **Topic mode** | How topics are found: `on_this_day`, `niche`, `keywords`, `feeds`, `evergreen_ideas`, `trending`, `manual`. Composable with weights. |
| **Brief** | Free-text steering instruction per profile. The LLM derives queries, subtopics, include/exclude terms and a relevance rubric from it. Explicit keywords always win. Derived settings are inspectable. |
| **Stage** | One idempotent, resumable pipeline step that writes artifacts into the run directory and returns a stage result. |
| **Format recipe** | Ordered stages + per-stage options + required capabilities + output type + handler (`save`, `publish`, `deliver`). Validated against stage contracts at load time. |
| **Input stage** | `input_topic` builds the topic from CLI args, a text/script file or research output, so any format runs without the research module. |
| **Media service** | The only paid entry point: screen → cache → budget → router → provider → validate → store → cost → sidecar. |
| **Screening token** | The value returned by prompt/content screening that paid services require, so no paid call can bypass the filter. |
| **Character** | A reusable figure (mascot/persona) with reference images, cutouts, poses, description, personality, voices per language, music identity and rules. Consistent via a strategy chain `composite → reference → adapter → prompt_only`. Strictly opt-in. |
| **Brand kit** | Colors, fonts, logo, watermark, intro/outro clips, caption style. Bindable to profiles, orders and presets. |
| **Reference** | A user-supplied example (image, video, audio, text) used instead of or alongside a topic. |
| **Style spec** | Structured, human-editable description extracted from references: structure, pacing, shot types, palette, typography, caption style, music mood, aspect, duration. Convertible into a recipe + style bible + preset. |
| **Instruction / Plan / Tool** | An instruction (with `@asset` labels) becomes a validated plan of tool calls with estimates, dependencies and approval. Tools are an allowlisted registry with JSON schemas. |
| **Composition** | Declarative canvas, layers (image/video/text/shape/sticker/logo), audio tracks, animations, effects. Rendered with no AI. |
| **Patch** | A validated, reversible edit operation on a composition (move, resize, recolor, retime, add/remove). |
| **Template** | A saved plan or composition with variables and asset slots, runnable once or in batch from CSV/JSON. |
| **Asset registry** | Files for a session or run with labels, hashes and analysis (dimensions, colors, description, detected text, faces). Originals are read-only. |
| **Occasion / Order** | Occasion is a config-defined personalization type (birthday, anniversary…). An order is one request with its own lifecycle, audit trail and cost record. |
| **Delivery** | File, expiring link, email, signed webhook or web card — the alternative to publishing. |
| **Writing mode** | `facts`, `quote`, `poem`, `story`, `joke`, `riddle`, `lyrics`, `custom`. Registry-driven; custom modes can come from a file. |
| **Run** | One execution of kind `pipeline`, `order`, `instruction`, `media` or `benchmark`, with a state machine, a run directory, a config snapshot and a cost record. |
| **Job / worker** | Persistent SQLite queue with atomic claim, heartbeat, retry/backoff, dead-letter, backpressure and graceful shutdown. |
| **Storage** | Local or S3-compatible interface with content-addressed paths and expiring URLs. |
| **Beat** | One script unit mapped to one visual and one audio segment. |
| **Cost event** | One record per paid call: provider, operation, units, unit kind, USD, run id, pricing version, `estimated|actual`. |
| **Platform data** | `platforms.yaml`: media types, durations, resolutions, aspect ratios, size limits, caption/hashtag limits, safe areas, disclosure requirements. Data, never code. |
| **Toolchain fingerprint** | Renderer version + ffmpeg version + font-set identity, recorded with every output and included in render cache keys. |
| **Bootstrap** | First-run population of the user config/data directories from packaged defaults, once, never overwriting edits. |
| **Replay** | Re-execution of a stored run using its recorded config snapshot, prompt versions and seeds, with a drift report. |

## 5. Nothing is hard-wired

Every external dependency sits behind an interface selected by configuration:

| Capability | Candidate implementations |
|---|---|
| LLM | Anthropic, OpenAI, Gemini, Mistral, any OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter) |
| Search | Tavily, Exa, Brave, SerpAPI, LLM-native web search |
| Sources | Wikipedia (+Pageviews), Reddit, Hacker News, RSS/Atom, Google Trends |
| Embeddings | OpenAI, Voyage, local sentence-transformers |
| TTS | ElevenLabs, OpenAI, Google/Azure, Cartesia, local (Piper/Kokoro) |
| STT | faster-whisper (local), Whisper API |
| Image | fal, Replicate, OpenAI, Google Imagen, Ideogram, Stability |
| Image edit | Aggregator inpaint/outpaint/upscale, local background removal |
| Video | Kling, Seedance, Veo, Wan, Hailuo, Luma, Runway, Pika — via aggregators or direct |
| Music | ElevenLabs Music, Suno/Udio/MiniMax, Stable Audio/Mubert/SOUNDRAW, local |
| SFX / stock | ElevenLabs SFX / Pexels, Pixabay, Unsplash |
| Notify / publish / deliver | Telegram, Slack, email / direct APIs or Postiz, Ayrshare, Buffer / folder, S3 link, email, webhook |

Aggregators are one option, not the architecture. A new image or video API can be added with a
**declarative YAML spec** and `providers add-spec` — no code — after the tool prints the outbound hosts and
secret names it would use and you confirm. Model ids live in `models.yaml`, prices in `pricing_overrides`,
so vendor churn never needs a release.

## 6. Stack, layout, package data

Python 3.12 · pydantic v2 · typer · SQLAlchemy 2 + Alembic (SQLite now, Postgres later) · httpx · FFmpeg
and ffprobe (system prerequisite, e.g. `brew install ffmpeg`) · Pillow · faster-whisper · APScheduler ·
FastAPI · pytest with recorded fixtures · optional OpenTelemetry exporter.

**One top-level package.** The install puts exactly one importable package, `aivia`, on the path. No
top-level `core`, `api`, `cli`, `media` or `llm`. Sibling directories are only `tests/`, `docs/`,
`scripts/`, `examples/`, `migrations/`.

**Package data vs user data.**

    src/aivia/
      core/  providers/  llm/  media/  compose/  formats/  assistant/  reference/  characters/
      research/  orders/  publish/  analytics/  api/  cli/
      assets/fonts/     # bundled OFL families + LICENSE files      (package data)
      data/defaults/    # config.example.yaml, prompts/, templates/, styles/, brands/,
                        # glossary/, platforms.yaml, models.yaml     (package data)

    $XDG_CONFIG_HOME/aivia/   # created once by bootstrap; user-editable, never overwritten
      config.yaml  prompts/  templates/  styles/  brands/  glossary/  providers.d/
      characters/  formats/  data/   # runs, media, cache, DB

Anything the tool needs at runtime and you must not edit ships inside the package. Anything you edit lives
in the config directory. `--config` always wins. Resolution works from any working directory.

## 7. Configuration

Principles:

- **Everything tunable is configuration.** No magic numbers or vendor names in business logic.
- **Layered resolution:** packaged defaults → user `config.yaml` → `config.<env>.yaml` → profile `overrides`
  → CLI `--set key.path=value`. Dicts deep-merge, lists replace.
- **Secrets are separate.** Keys live in environment variables or a secrets file, referenced by name.
  Tool environment variables use the `AIVA_` prefix.
- **Validated and discoverable.** Validated at startup with readable errors (file, dotted path, expected vs
  actual, hint). `config show --resolved --profile P` prints the merged result with `--origins`; a JSON
  Schema is generated for editor completion.
- **Versioned.** `schema_version` is mandatory; `config migrate` upgrades older files and prints a diff.
- **Module sections are registered by modules**, so a deployment without a module simply omits them, and a
  module key without its package fails clearly.
- **Dated knowledge is data:** model ids, prices, platform limits, voice ids and rate limits live in data
  files.

The canonical example config is a single file: `src/aivia/data/defaults/config.example.yaml` (shipped, then
copied to the user directory on first run). It is not duplicated in documentation. Orientation only:

    schema_version: 1
    languages: {en: {...}, de: {...}, tr: {...}}
    pipeline: {stages: [research, write, voice, visuals, render, review, publish], timezone: "Europe/Berlin"}
    video: {resolution: 1080x1920, fps: 30, duration_seconds: {min: 20, max: 45}, loudness_lufs: -14}
    render: {min_contrast: 4.5, fonts_dir: package, require_bundled_fonts: true}
    llm: {chain: [...], roles: {...}, pricing: {...}, max_repairs: 2}
    models_file: models.yaml
    image_gen / image_edit / video_gen / stock / sfx / embeddings / stt / tts / music_gen: chains
    visual_strategy: {default: hybrid, max_ai_video_beats: 2}
    budget: {per video, day, month, order, instruction, provider; alerts; on_exceed}
    scoring / quality / prompts / review / retention / privacy / tracing / accessibility
    storage / api / jobs / safety
    formats: {narrated_still, song_video, character_short, personalized_song_card, ...}
    reference / character / brand / planner / occasions / orders / delivery
    profiles / schedule / publish

LLM roles: the core enum is exactly `default`, `writing`, `factcheck`, `lyrics`, `planner`,
`reference_analysis`, `vision_qa`, `safety`, `translation_check`, `prompt_enhance`. Modules may register
more (`scoring`, `relevance`, `brief`, `profile_builder`). Unknown roles fail with the allowed list.

## 8. Invariants

These are the rules code must never break. They are enforced by tests, not by good intentions.

1. **Originality.** Scripts never reproduce source text; facts are rewritten with a distinctive angle.
   Reference lyrics and melodies are never reused.
2. **Accuracy.** Every claim maps to at least one source URL; unsupported claims are removed or rewritten.
3. **Disclosure.** Published items carry AI-content disclosure where the platform supports it plus a source
   line; each publish records a rights/disclosure attestation when the publishing module is installed.
4. **Budget safety.** Per-video, per-day, per-month and per-provider limits stop or downgrade work. No paid
   call bypasses the tracker or the screening token.
5. **No duplicates.** The same topic is not produced twice inside a configurable window, across profiles.
6. **Reproducibility, precisely.** Every output stores the config snapshot, prompt versions, provider and
   model ids, seeds, cost and a toolchain fingerprint. Re-rendering the same composition with the same
   fingerprint is *visually* identical (SSIM ≥ 0.99), not necessarily byte-identical: encoder, thread count
   and ffmpeg build can change bytes. Any provenance or signing step runs in a separate `finalize` copy and
   never affects the cache or golden comparison.
7. **Untrusted input.** User text, filenames, CSV cells and text found inside images or fetched pages are
   data, never instructions. Templates that users can edit are rendered in a sandboxed engine.
8. **Privacy.** Personal data is minimized, redacted in logs, retained only as configured and deletable on
   request. **Final outputs are pinned and never removed by retention** unless explicitly enabled.
9. **Determinism at the edge.** Rendering performs no AI calls and no network requests.
10. **Single paid path.** Images, video, speech, music and sfx all go through the media service.
11. **Core independence.** The core never imports a module; the architecture test enforces it.
12. **Safe defaults.** SQLite runs with WAL, a busy timeout and foreign keys; outbound fetches validate the
    resolved IP (DNS-rebinding safe); token encryption keys come from the secrets loader; publishing
    defaults to dry run in tests.

## 9. Definition of done and test policy

**Every item is done when:** code is typed; an offline test fails before the change and passes after;
`make check` is green; no live API call happened; no secret or personal data is in code, fixtures or logs;
no vendor name, model id, price or threshold was added outside provider adapters, config and data files;
new config keys exist in the schema, the JSON Schema and the packaged default config; every paid call emits
a cost event; artifacts are written under the run directory; the item is safe to re-run.

**Test policy.** The default suite is fully offline with recorded fixtures and mock providers. Live tests are
marked `live`, require `--live`, and honor `AIVA_LIVE_BUDGET_USD`. Golden render tests run only in a pinned
Linux image with a pinned ffmpeg and the bundled fonts. Every milestone that adds a real stage extends the
end-to-end smoke test.

**Fixtures and examples policy.**

- **Recorded responses, one folder per provider.** Network responses used by tests are recorded under
  `tests/fixtures/providers/<provider>/<case>.json` and replayed offline through the provider's HTTP
  transport. A fixture stores the request (method, URL, relevant headers), the response status, headers and
  body, and when it was captured. Provider contract tests reuse the same fixtures for every provider;
  recording is a reviewed manual step and never happens in the default suite.
- **One example per schema.** Every public structured schema (config sections, provider specs, compositions,
  plans, profiles, orders) has exactly one minimal valid example under `examples/<schema>.example.yaml`, or
  `.json` where the format is JSON, named after the schema. Tests validate each example against its schema so
  examples cannot drift. The canonical config is the exception: it lives in
  `src/aivia/data/defaults/config.example.yaml` and is not duplicated here.
- **A goldens folder.** Expected render and media artifacts live under `tests/golden/` beside their metadata
  (composition id, toolchain fingerprint, SSIM tolerance). Goldens are generated only on the pinned Linux
  image with the pinned ffmpeg and bundled fonts, are marked `golden`, and are never hand-edited. The default
  offline run does not compare goldens.
- **No real personal data.** Fixtures, examples and goldens use synthetic names, emails, phone numbers,
  addresses, faces and voices. Recordings are redacted of tokens, cookies and personal data before they are
  committed. Real customer data, secrets and live credentials never enter the repository, a fixture or a
  log. Only generated or public-domain media is used, with its provenance recorded.

## 10. How work flows

`PLAN.md` is both the queue and the notebook. The runner takes the first unchecked item, hands that exact
text to the agent, and the agent ticks it off in the same pull request. The agent may append items to the end
of the relevant milestone or to `## Discovered`; it may never delete, reorder or reword an existing item. Only
an explicit replan pass may restructure the list.

**Ordering is the dependency graph.** Items carry no ids and no dependency fields; an item may only rely on
items above it. That makes ordering a structural invariant: if you need work from a later item, append a
prerequisite above instead of jumping ahead. When converting detail into an item line, the acceptance signal
lives in the sentence and the test path — e.g. "resolve user config and package resources from any working
directory (tests/core/test_paths.py)".

Two consequences worth remembering: runtime assets (bundled fonts, default config, prompts, templates) must
land in the first milestone, because later milestones consume them; and a module's items must not be needed by
core items, or skipping the module would break the core.

## 11. Milestones

| # | Milestone | Layer | Depends on | Items |
|---|---|---|---|---|
| 1 | Toolchain, packaging and smoke tests | Core | — | 21 |
| 2 | Configuration system | Core | 1 | 28 |
| 3 | Models, repositories and database | Core | 2 | 16 |
| 4 | Providers, registry, router and mocks | Core | 3 | 19 |
| 5 | Cost tracking, budgets and dashboard | Core | 4 | 16 |
| 6 | Execution core, storage and API | Core | 5 | 23 |
| 7 | LLM, prompts, search and embeddings | Core | 6 | 24 |
| 8 | Media generation core and toolkit | Core | 6, 7 | 61 |
| 9 | Composition and rendering | Core | 8 | 37 |
| 10 | Research engine | Module | 7 | 31 |
| 11 | Writing | Core | 7 (never 10) | 18 |
| 12 | Voice, subtitles and timeline | Core | 8, 9, 11 | 10 |
| 13 | Visuals in the pipeline | Core | 8, 11 | 9 |
| 14 | Formats and song pipeline | Core | 9, 11, 12, 13 | 17 |
| 15 | Reference driven generation | Core | 8, 11, 12, 14 | 19 |
| 16 | Characters and brand kits | Core | 8, 9, 12, 14, 15 | 26 |
| 17 | Natural language assistant | Core | 8, 9, 14 | 23 |
| 18 | Orders, receipts and delivery | Module | 9, 14, 17 | 32 |
| 19 | Scheduling, notifications and review | Module | 6 | 9 |
| 20 | Publishing | Module | 9, 19 | 21 |
| 21 | Analytics and benchmarking | Module | 8, 19, 20 | 10 |
| 22 | Hardening and documentation | All | all | 23 |

**Core alone (milestones 1–9 and 11–17) is a complete generic assistant and depends on no module.**
Milestones 10, 18, 19, 20, 21 are optional and can be skipped or reordered among themselves while respecting
the dependency column.

## 12. Non-goals

Card payment or PSP integration · a multi-rate VAT engine (orders ship a simple receipt with a flat estimate
and an explicit "not a tax invoice" footer) · multi-tenant SaaS billing (owner scoping and quotas only) ·
HDR or color-managed video · model fine-tuning pipelines beyond the character adapter strategy · a mobile app
· real-time collaboration · localizing the CLI output itself.
