# Adding a provider

A provider is a thin adapter around one external service. It never decides
which provider to use, when to fall back or whether a call fits the budget;
those policies belong to the router. This guide walks through adding one
adapter end to end.

## 1. Pick a capability

Choose the capability the provider serves, for example `llm`, `tts`, `stt`,
`image_gen`, `video_gen`, `music_gen`, `search`, `source`, `page_fetch`,
`publish`, `delivery`, `storage`, `stock`, `image_edit`, `notify`, `sfx`,
`lipsync` or `character_trainer`. Each capability has a typed protocol in
`aivia.protocols` and a reusable contract suite in `tests/contracts`.

## 2. Subclass the provider base

Create the adapter as a subclass of `aivia.providers.base.Provider` (or of a
capability-specific base that itself subclasses `Provider`). Declare static
metadata with `ProviderMeta` describing the capabilities, prices, rate limits
and licensing terms. Prices live in `pricing_overrides` and model ids live in
`models.yaml`; never hardcode a vendor name, model id or price in the adapter.

Implement at least:

* `health` -- an async probe bounded by `health_timeout_seconds` so an
  unresponsive adapter cannot stall a status command.
* `close` -- an idempotent async hook that releases connections and clients.
* The capability's own typed methods, for example `generate` for an LLM.

Capability-specific bases add their own typed methods; implement every method
the protocol declares.

## 3. Register the provider type

Built-in adapters register themselves with the `register_provider` decorator,
keyed by a type string. Registration rejects duplicate names. A custom adapter
is instead named by a dotted path such as `my_package.providers:MyProvider`
and imported lazily; resolving a type that is neither registered nor loadable
fails with the type and the registered names, so a typo never silently selects
the wrong provider.

## 4. Configure a provider instance

Add an entry to the `providers` config section mapping a name to `type`,
`options` and `secret_refs`:

```yaml
providers:
  my_tts:
    type: my_tts
    options:
      voice: alloy
    secret_refs:
      api_key: AIVA_MY_TTS_API_KEY
```

A bare string is shorthand for an entry that only sets `type`. `secret_refs`
maps an option name to a secret reference resolved by the secrets loader
(environment, then `secrets.env`, then the OS keyring); never write a secret
value into the configuration file.

## 5. Wire pricing and models

Resolve pricing from provider metadata and `pricing_overrides`, with unit
conversions and a warning event instead of a crash when a price is missing.
Validate configured model ids against `models.yaml`, warning on unknown ids
with a closest match and on deprecated ids with a replacement.

## 6. Add tests

Add the adapter to the shared contract suite in `tests/contracts` and run it
offline against recorded fixtures. Add provider-specific tests under
`tests/providers/<capability>/`. Every provider call must go through the router
so fallback, budget checks, rate limiting and circuit breaking apply. No
business logic calls a provider directly.

## Checklist

* [ ] Capability chosen and protocol methods implemented
* [ ] `ProviderMeta` declared (capabilities, pricing, rate limits, licensing)
* [ ] `health` and `close` implemented, both async and bounded
* [ ] Provider type registered or reachable by dotted path
* [ ] `providers` config entry uses `secret_refs`, never secret values
* [ ] Model ids validated against `models.yaml`; prices in `pricing_overrides`
* [ ] Contract suite passes offline
* [ ] No vendor name, model id, price or threshold outside the adapter,
      config and data files
