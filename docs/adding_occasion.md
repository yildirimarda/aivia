# Adding an occasion

An occasion is a config-defined personalization type, such as a birthday or an
anniversary. Add one by declaring it under the `occasions` config section;
occasions may be written directly under the section or nested under
`occasions`. Every field has a default, so an empty mapping is a valid
configuration.

## 1. Declare the occasion

```yaml
occasions:
  default: birthday
  birthday:
    label: Birthday
    description: A personalized birthday card
    enabled: true
    variables:
      name:
        type: string
        required: true
        label: Recipient name
        min_length: 1
        max_length: 80
      age:
        type: integer
        required: false
        label: Age
        min: 1
        max: 150
    personalization_levels:
      simple:
        name: Simple
        stages: [name_overlay]
        outputs: [png]
        price: 0.0
        personalize_name: true
      full:
        name: Full
        stages: [write, voice, render]
        outputs: [mp4, mp3]
        price: 5.00
        personalize_name: true
        include_age: true
        include_message: true
    languages:
      en:
        lyrics: "Happy birthday, {name}!"
        speech: "Happy birthday, {name}."
        on_screen_text: "Happy birthday {name}!"
        age_phrasing: "turning {age}"
        date_phrasing: "{date}"
    formats: [song_card, poster]
    template: birthday.md
    default_level: full
```

## 2. Set the occasion fields

* `label` -- display name (aliases `title`, `display_name`).
* `description` -- human readable summary.
* `enabled` -- whether the occasion can be selected (default `true`).
* `default_level` -- personalization level used when none is requested.
* `formats` -- output formats the occasion supports (aliases `recipes`,
  `outputs`).
* `format` -- default format name (aliases `recipe`, `format_name`).
* `template` / `prompt` / `files` -- optional prompt or template files shipped
  with a custom occasion (aliases `templates`, `prompt_files`, `attachments`).
* `variables` -- the fields an order of this occasion may fill.
* `personalization_levels` -- which order levels are allowed (aliases `levels`,
  `allowed_levels`).
* `languages` -- per-language content for localized orders.
* `default` (section level) -- the occasion selected by default; it must name a
  configured occasion.

## 3. Declare variables

Each variable accepts a bare name shorthand or a mapping:

* `name` -- variable key (aliases `key`, `id`, `variable`).
* `label` -- display label (aliases `title`, `display_name`).
* `type` -- value type: `string`, `text`, `number`, `integer`, `boolean`,
  `date`, `choice` or `file`/`image`/`audio`/`video` (aliases `kind`,
  `value_type`, `format`).
* `required` -- whether the field is mandatory.
* `default` -- value used when an optional variable is omitted.
* `description` -- help text (alias `help`).
* `options` -- allowed values for choice variables (aliases `values`,
  `choices`, `allowed`).
* `min_length` / `max_length` -- text length bounds (aliases `min`/`max`).

Choice variables list their allowed `options`; `min_length` must not exceed
`max_length`.

## 4. Declare personalization levels

Each level selects which stages run and what it costs:

* `name`, `description`, `stages`, `outputs`, `variables`
* `price` -- line-item amount before the flat tax estimate (aliases `amount`,
  `cost`, `unit_price`, `base_price`).
* `cost_multiplier` -- multiplier applied to the price.
* Feature flags: `personalize_name`, `include_age`, `include_interests`,
  `include_message`.
* `background`, `format`, `enabled`.

Select stages from the requested outputs so a PNG runs only the free name
overlay stage at zero API cost. The cheapest configured level whose declared
outputs cover every requested output wins.

## 5. Add per-language content

`languages` maps a language code to `OccasionLanguageContent` holding
`lyrics`, `speech`, `on_screen_text`, `age_phrasing` and `date_phrasing`. Every
field defaults to empty, so an occasion can list a language without supplying
every string. A list such as `["en", "de"]` becomes `{"en": {}, "de": {}}`.

## Checklist

* [ ] Occasion declared under `occasions` with a unique name
* [ ] `default` names a configured occasion (when set)
* [ ] Variables have non-empty names and valid types
* [ ] `min_length` does not exceed `max_length`
* [ ] Personalization levels declare stages, outputs and a price
* [ ] Per-language content declared for each supported language
* [ ] Custom occasions supported with files only (`allow_custom_occasion`)
