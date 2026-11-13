# Adding a platform

Platform publishing rules live in `platforms.yaml`, a data file, not in code.
Add a platform by declaring it there; the loader validates the file and exposes
typed lookups by platform and media type. The packaged file is the fallback;
a user copy beside the config wins. No code change and no release are needed
when a platform changes a limit.

## 1. Declare the platform

Add an entry under `platforms` in `platforms.yaml`:

```yaml
schema_version: 1

platforms:
  youtube:
    name: YouTube Shorts
    media_types: [video]
    limits:
      video:
        min_duration_seconds: 1
        max_duration_seconds: 180
        resolutions: ["1080x1920"]
        aspect_ratios: ["9:16"]
        max_file_size_bytes: 268435456000
        audio_required: true
        loudness_lufs: -14
        loudness_tolerance_lufs: 1.0
    caption_limits:
      max_length: 5000
      max_title_length: 100
      max_description_length: 5000
      max_hashtags: 15
    safe_area:
      top: 0.10
      bottom: 0.20
      left: 0.06
      right: 0.06
    disclosure:
      required: true
      text: "This content was generated with AI."
      on_screen_label: "AI-generated"
      metadata: true
      metadata_field: containsSyntheticMedia
```

## 2. Set the platform fields

* `name` -- human readable platform name.
* `media_types` -- media types the platform accepts (for example `video`,
  `image`, `carousel`, `audio`, `web_card`).
* `limits.<media_type>` -- per media type limits:
  * `min_duration_seconds` / `max_duration_seconds` -- duration bounds.
  * `resolutions` -- allowed resolutions as `WIDTHxHEIGHT`.
  * `aspect_ratios` -- allowed aspect ratios as `WIDTH:HEIGHT`.
  * `max_file_size_bytes` -- maximum file size.
  * `audio_required` -- whether audio is required.
  * `loudness_lufs` / `loudness_tolerance_lufs` -- loudness target and
    tolerance.
* `caption_limits` -- `max_length`, `max_title_length`,
  `max_description_length`, `max_hashtags`.
* `safe_area` -- fractions measured from each edge of the canvas (`top`,
  `bottom`, `left`, `right`), each a finite fraction between zero and one.
* `disclosure` -- whether disclosure is required, the disclosure text, the
  on-screen label, whether to embed metadata and the metadata field name.

## 3. Add defaults (optional)

A top-level `defaults` block supplies disclosure and chapter rules inherited
by platforms that do not override them: `disclosure.required`,
`disclosure.text`, `disclosure.on_screen_label`, `disclosure.metadata`,
`chapter_rules.enabled`, `chapter_rules.max_chapters`,
`chapter_rules.min_chapter_duration_seconds`,
`chapter_rules.max_chapter_title_length` and `chapter_rules.separator`.

## 4. Add aliases (optional)

Aliases map alternate names to a canonical platform (for example
`instagram_reels`, `instagram-reels` and `reels` all map to `instagram`).
Aliases live in the loader, not in the data file.

## 5. Bind accounts

Publish accounts are keyed by `platform/language` in the `platforms` config
section (`PlatformAccountConfig`) and are bound to exactly one platform and one
language. Account credentials are referenced by name and resolved through the
secrets loader, never written into the configuration file.

## 6. Output validation enforces the limits

Outputs are validated against platform data for duration, resolution, size,
audio presence and loudness, with per type messages. Safe area margins are
applied to text and sticker layers, and a composition is rejected when its
safe area leaves no visible area. Reject an output container whose extension
cannot carry the configured codec before invoking ffmpeg.

## Checklist

* [ ] Platform entry under `platforms` in `platforms.yaml`
* [ ] `name` and `media_types` set
* [ ] Per media type limits declared (duration, resolutions, aspect ratios,
      size, audio, loudness)
* [ ] `caption_limits` and `safe_area` fractions between zero and one
* [ ] `disclosure` block present when the platform requires it
* [ ] Publish account bound to `platform/language` in config
* [ ] Output validation enforces platform limits per media type
