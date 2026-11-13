# Adding a format

A format is a named recipe: an ordered list of pipeline stages, each with its
own options, plus the capabilities it needs, the output type it produces and
how that output is handled. Add one by declaring it under the `formats` config
section; recipes may be written directly under the section or nested under
`formats` (or `recipes`).

## 1. Declare the recipe

```yaml
formats:
  narrated_still:
    description: One or few images, text overlay, speech and word synced subtitles
    stages:
      - stage: write
        options:
          mode: facts
      - stage: voice
      - stage: visuals
      - stage: render
    capabilities:
      - tts
      - image_gen
    output_type: video
    handler: save
```

## 2. Set the recipe fields

* `stages` -- ordered pipeline stages. Each entry is a bare stage name
  (`write`) or a mapping with `stage`/`name`/`id`/`step` and an optional
  `options` mapping. The shorthand `{write: {mode: facts}}` is accepted too.
  Duplicate stage names are rejected.
* `capabilities` -- operations the recipe needs, for example `tts` or
  `image_gen`. Duplicates are rejected.
* `output_type` -- one of `video`, `audio`, `image`, `carousel` or
  `web_card` (default `video`).
* `handler` -- what happens to the final artifact: `save`, `publish` or
  `deliver` (default `save`).
* `description` -- optional human readable summary.

## 3. Configure per-stage options and retries

A stage mapping may override the stage class timeout and retry policy through
`timeout_seconds` or a `retry_policy` mapping. Other retry fields accepted are
`max_attempts`, `retry_delay_seconds`, `backoff_multiplier`,
`max_retry_delay_seconds`, `retry_on_timeout` and `retry_codes`. All values
must be finite.

## 4. Load user and packaged formats

Merge user and packaged format files during config resolution so profiles can
reference file-only formats. Stage contracts are validated when the stage
registry is available: every stage input must be produced by an earlier stage
and errors name the offending stage.

## 5. Use the format

Reference the format by name from a profile, preset or CLI flag. A run executes
its recipe, skipping unused stages at zero cost. Format recipes can override a
stage timeout and retry policy instead of only its class.

## Checklist

* [ ] Recipe declared under `formats` with ordered, unique stages
* [ ] `capabilities`, `output_type` and `handler` set
* [ ] Per-stage options and retry overrides finite and valid
* [ ] Every stage input produced by an earlier stage (recipe validation)
* [ ] User and packaged format files merged at config resolution
* [ ] Format referenced by a profile, preset or CLI flag
