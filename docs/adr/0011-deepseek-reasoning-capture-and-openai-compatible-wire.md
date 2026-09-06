# DeepSeek captures reasoning on both the buffered and streamed paths

DeepSeek speaks the OpenAI Chat Completions protocol, so its Provider reuses the
OpenAI request builder and the `openai-completions` `Api` tag (inheriting `/models`
discovery and Bearer auth) but supplies its own `WireResponse` and `StreamNormalizer`.
The reason for the bespoke response half is `reasoning_content`: the model's chain of
thought, present on `deepseek-reasoner`/V4 models. We map it to a `ContentPart::Thinking`
(with `signature: None` — DeepSeek supplies none) on **both** the streamed path
(`ThinkingStart`/`ThinkingDelta`/`ThinkingEnd`, emitted before text and tool calls) and
the buffered path — deliberately diverging from the Anthropic Provider, which drops
thinking on the buffered path because its thinking block shape is complex; DeepSeek's is
a plain string, so capturing it costs nothing and losing the provider's differentiator
would.

## Consequences

- `Usage` maps DeepSeek's disjoint prompt split: `input_tokens = prompt_cache_miss_tokens`,
  `cache_read_tokens = prompt_cache_hit_tokens`, `cache_write_tokens = 0` (caching is
  automatic and disk-based, with no write cost or write counter). `reasoning_tokens` is
  dropped — there is no neutral field for it and it is already counted inside
  `completion_tokens`.
- `ThinkingLevel` and `CachePolicy` are ignored: DeepSeek selects reasoning by model, not
  by a request-side budget, and caches implicitly with no `cache_control` on the wire.
- Request quirks are handled permissively first. We do not drop sampling params on reasoning
  models nor rewrite a forced `tool_choice`; if the live API rejects either, we tighten
  against a recorded VCR cassette rather than porting rules from an older model line on
  assumption.
- Vision is OpenAI-compatible: the vision model takes images as `image_url` content parts
  (base64 data URL or https URL), exactly what the reused OpenAI request builder emits, so
  multimodal input needs no DeepSeek-specific code — only that images ride on `user`
  messages, which DeepSeek requires.
- No embeddings: DeepSeek exposes no official embeddings endpoint, so the Provider ships
  completion, streaming, and tool calling only.
