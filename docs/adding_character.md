# Adding a character

A character is a reusable recurring figure whose look, voice and music stay
consistent across outputs. Characters are strictly opt-in: no character code
runs unless one is selected by CLI flag, order or instruction field, or
profile. Add one by writing a character file in the user data directory and,
optionally, configuring the shared character machinery under the `character`
config section.

## 1. Create the character file

A character is a YAML file stored next to its reference images and pose
cutouts in the user data directory. Paths in `reference_images` and
`pose_library` are relative to the character file's directory.

```yaml
name: Maya
version: 1
voice: alloy
voices:
  en: alloy
  de: ember
art_style: flat vector illustration, muted pastels
negatives:
  - photorealistic
  - text
  - watermark
reference_images:
  - assets/maya_face.png
  - assets/maya_full_body.png
pose_library:
  idle: assets/maya_idle.png
  wave: assets/maya_wave.png
brand_kit: summer_camp
music_style_prompts:
  - upbeat acoustic folk
personality_lyrics: warm, encouraging and curious
jingle: assets/maya_jingle.mp3
ownership_required: true
allow_real_person_likeness: false
resemblance_warning: true
```

## 2. Set the character fields

* `name` -- non-empty character name.
* `version` -- schema version (default 1). Bump it when the sheet, pose or
  voice changes; runs pin an older version.
* `voice` -- default provider voice id.
* `voices` -- mapping of language to provider voice id, for per language
  character voices pinned across runs.
* `art_style` -- art style applied to generated images of the character.
* `negatives` -- negative prompts applied to generation.
* `reference_images` -- reference images used to keep the look consistent.
* `pose_library` -- named pose cutouts with alpha, anchors and tags.
* `brand_kit` -- brand kit bound to the character.
* `music_style_prompts` -- style prompts for the character's music identity.
* `personality_lyrics` -- reusable lyrics expressing the character's
  personality.
* `jingle` -- reusable jingle or intro.
* `ownership_required` -- ownership flag required (default `true`).
* `allow_real_person_likeness` -- real person likeness blocked without consent
  (default `false`).
* `resemblance_warning` -- optional resemblance warning (default `true`).

## 3. Configure the character machinery

The `character` config section configures how the shared character machinery
behaves when a character is selected:

```yaml
character:
  enabled: true
  consistency_strategies:
    - composite
    - reference
    - adapter
    - prompt_only
  qa:
    min_score: 0.8
    vision_threshold: 0.8
    embedding_threshold: 0.75
    max_retries: 2
  consent:
    voice_cloning: false
    real_person_likeness: false
    resemblance_warning: true
    require_ownership: true
  max_reference_images: 4
```

* `consistency_strategies` -- ordered chain tried per scene. The default order
  is `composite -> reference -> adapter -> prompt_only`. `prompt_only` is a
  last resort and may only appear at the end of the chain.
* `qa` -- thresholds used to accept a generated asset: overall minimum score,
  vision feature threshold, embedding similarity threshold and retries before
  moving to the next strategy.
* `consent` -- rights flags that gate voice cloning and real-person likeness.
  Voice cloning and real-person likeness are off by default.
* `max_reference_images` -- maximum reference image count enforced per
  provider capability.

## 4. Select the character

Opt in by CLI flag, then order or instruction field, then profile. Several
characters per run are supported with host, sidekick and narrator roles. The
character sheet, art style, negatives and per scene pose are injected only
into scenes that include the character. Providers are filtered by character
capability flags, and the maximum reference image count is enforced.

## 5. Version and gate

Version characters so sheet, pose and voice changes bump the version while
runs pin an older one. Gate voice cloning behind explicit consent. Enforce
character rights: ownership flag required, real person likeness blocked
without consent, and an optional resemblance warning.

## Checklist

* [ ] Character YAML file in the user data directory
* [ ] `name` non-empty; `version` set
* [ ] `reference_images` and `pose_library` paths relative to the character file
* [ ] `character` config section: consistency chain, QA thresholds, consent
      flags, reference limit
* [ ] Character selected by CLI flag, order/instruction field or profile
* [ ] Providers filtered by character capability; max reference images enforced
* [ ] Voice cloning gated behind consent; likeness blocked without consent
