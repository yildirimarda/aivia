# Adding a language

Languages are configured data, never code. Add one by declaring it under the
`languages` config section; rendering and writing resolve the language's
settings from that section.

## 1. Declare the language

Add an entry keyed by the language code under `languages`. Entries may be
written directly under the section (`languages: {de: {...}}`) or nested under
`languages`; both forms validate to the same model.

```yaml
languages:
  default: en
  en:
    name: English
    locale: en_US
    words_per_second:
      min: 2.0
      max: 3.5
    subtitle_font: Lato
    hashtag_style: lower
    forbidden_phrases:
      - "place holder"
  de:
    name: Deutsch
    locale: de_DE
    words_per_second:
      min: 2.0
      max: 3.2
    subtitle_font: Lato
    hashtag_style: lower
```

## 2. Set the fields

* `name` -- human readable name of the language.
* `locale` -- locale used for formatting (default `en_US`).
* `words_per_second` -- inclusive speaking-rate bounds used to estimate and
  check script length. `min` must not exceed `max`.
* `subtitle_font` -- font used for subtitles in this language (default `Lato`).
* `hashtag_style` -- one of `lower`, `title` or `upper` (default `lower`).
* `forbidden_phrases` -- per language forbidden phrase list applied when
  writing natively from the same fact base; duplicates are deduplicated
  case-insensitively.

## 3. Pick a default

`languages.default` names the language used when none is requested. It must
name a configured language, and a config with no language entries is rejected.
Cross-check profile and schedule language names against the configured
languages so a typo never silently selects the default.

## 4. Handle writing direction

The writing direction lives in `aivia.core.languages` so rendering stays
importable without the LLM package. Right-to-left codes (`ar`, `he`, `fa`,
`ur` and others) default to RTL; everything else defaults to LTR. Text layer
font defaults are resolved from render config, not hardcoded, and RTL capable
bundled fonts with Arabic joining are used for right-to-left text.

## 5. Add fonts if needed

If the language needs glyphs the bundled fonts do not cover, add a bundled
font family to `src/aivia/assets/fonts/` with its OFL license and declare it
as package data. Text rendering treats a system fallback as an error when
bundled fonts are required.

## Checklist

* [ ] Language entry keyed by code under `languages`
* [ ] `default` names a configured language
* [ ] `words_per_second` bounds ordered (`min` <= `max`)
* [ ] Profile and schedule language names cross-checked against configured
      languages
* [ ] Bundled font added for any uncovered glyphs, with OFL license
* [ ] No real personal data in examples; synthetic content only
