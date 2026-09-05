# Prompt caching is a neutral Cache Policy the Provider places

A new `CachePolicy` (`Off`/`Standard`/`Extended`) rides on `CompletionOptions` as a per-request cost knob alongside `ThinkingLevel`, and the Anthropic Provider turns a non-`Off` policy into `cache_control` breakpoints on the wire; a Provider that caches implicitly (OpenAI) or not at all ignores it. The Provider places up to three fixed breakpoints itself — the last tool, the last system block, and the final block of the last message, skipping any that are absent — rather than exposing per-block marking on the neutral content model, because callers reason about caching as a request-level spend decision, not block by block. We do not gate on a per-model capability, run a breakpoint budget allocator, or send the now-GA `prompt-caching` beta: three breakpoints sit under Anthropic's cap of four so overflow is impossible, current models all cache, and only the `Extended` (1h) TTL still needs a beta header.

## Consequences

- The neutral vocabulary stays free of wire terms: `Off`/`Standard`/`Extended` name retention as effort levels, and the `ephemeral`/`ttl:"1h"` shaping plus the `extended-cache-ttl` beta are derived by the Anthropic Provider from the one policy value, so body marker and header cannot drift.
- `Usage` gains `cache_read_tokens` and `cache_write_tokens` — named for the `Model` spec's existing pricing fields, not Anthropic's `creation` wire word — parsed on both the buffered and streamed paths and disjoint from `input_tokens`. Turning them into a cost figure is left to the caller.
- Placing a marker promotes only the affected system block or message from the compact string form to the block-array form; every other message stays compact.
- Caching applies on both auth lanes. On OAuth the last system block carries the marker, caching the Claude Code identity and the caller's prompt as one prefix, and the `extended-cache-ttl` beta merges into the existing `anthropic-beta` list rather than replacing it.
- No URL guard on the `Extended` TTL: a relay that rejects a 1h marker surfaces a normal error rather than a silent downgrade.
