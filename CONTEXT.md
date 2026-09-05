# tapir-provider

An extensible SDK for talking to LLM backends. It exposes a uniform interface over
many vendors, each compiled in only when its Cargo feature is enabled, and handles
authentication (API key and OAuth) on the caller's behalf.

## Language

**Provider**:
A pluggable adapter to one LLM backend (Anthropic, OpenAI, ...). Implemented as an
object-safe trait and held as a trait object so third parties can add their own.
_Avoid_: Client, backend, vendor, driver.

**Model**:
The specification of one addressable model a Provider exposes: its id, wire API,
endpoint, input modalities, context window, token limits, and cost. Pure
description — it carries no Credential.
_Avoid_: Engine.

**Model Entry**:
A Model paired with what is resolved at runtime to actually call it — the Credential
and any per-model Compat overrides. The unit the Model Registry holds and turns into
a live Provider.
_Avoid_: Model config, model record.

**Compat**:
Per-model overrides describing how a Model deviates from its Provider's defaults:
supported features, tool-call dialect, thinking levels. Optional; when absent the
Provider's defaults apply.
_Avoid_: Quirks, flags, options.

**Catalog**:
The set of Models known to the SDK, merged from layers in precedence order: user
overrides, the fetched cache, then the compiled-in baseline.
_Avoid_: Model list, index, database.

**Capability**:
A thing a Provider can do — completion, streaming, tool calling, multimodal input,
embeddings. Embeddings live behind a separate trait; the rest are one trait.

**Credential**:
The authentication material for a Provider: either an API key or an OAuth token set
(access token, refresh token, expiry). One tagged value that round-trips losslessly.
_Avoid_: Auth, key, secret (each names only part of it).

**Provider Config**:
A set of provider-scoped, non-secret config values carried on an API-key Credential
alongside its key — for example a gateway's account and gateway ids. Serialized under
the `env` wire key (the Rust field stays `config`, since `env` already names the
environment-variable auth tier in code). Distinct from a Credential's round-trip-only
`extra` bag: a wire `env` lands in this typed field, other unmodeled fields in `extra`.
_Avoid_: Env, settings, options.

**Token Store**:
The persistence boundary for Credentials. The SDK reads and writes through it and
owns refresh; the caller chooses where they live (file, OS keychain, ...).
_Avoid_: Keyring, vault, cache.

**Resolved Auth**:
What one auth inspection produces: the request-ready material a Provider would send
this turn — its auth headers, an auth-derived API key and base URL when there is one,
the winning Credential's Provider Config, and the Auth Source that won. Unlike a
Credential (stored material), it is computed on demand and never persisted. A
provider-scoped inspection carries only the auth headers; a Model-scoped one also layers
that Model's headers and base URL. The Provider Config rides along in either scope,
since it comes from the Credential not the Model, and is present only when a stored
API-key Credential wins.
_Avoid_: Auth, resolved credential, auth result.

**Auth Source**:
Which tier a Resolved Auth came from — an explicit per-request key, a stored Credential,
a named environment variable, or an OAuth token. The precedence resolution keeps this
rather than discarding it, so a caller can see how a Provider is configured.
_Avoid_: Origin, provenance, tier.

**Registry**:
The compiled-in set of available Providers and their identities (canonical id,
aliases, default Credential source). Only entries whose feature is enabled are
present, so the default build ships none. Distinct from the Model Registry.
_Avoid_: Factory, plugin list.

**Model Registry**:
The runtime holder of the Catalog. It loads the layers, resolves Credentials, answers
lookups by provider and id, reports which Models are ready to call, and refreshes the
fetched layer. Distinct from the Registry, which is the compile-time set of Providers.
_Avoid_: Catalog manager, model store.

**Tool Call**:
A request from the model to invoke a named tool with JSON arguments, carried as a
content part of an Assistant Message. It carries a single id that is always present,
so callers have a stable handle to correlate a Tool Result back to it: the
provider-native id when the wire protocol supplies one, else an SDK-minted id.
_Avoid_: Function call, invocation.

**Tool Result**:
The outcome of running a Tool Call, sent back to the model as its own Message. It
references the Tool Call by id, names the tool, carries the result as content, and
flags whether the run errored.
_Avoid_: Tool output, tool response, function result.

**Context**:
The full conversational input to a Provider: an optional system prompt, the ordered
Messages so far, and the Tools the model may call. An Assistant Message returned by a
Provider drops straight back into it for the next turn. Distinct from Completion
Options, which carry the request's sampling knobs.
_Avoid_: Conversation, session, prompt, request.

**Completion Options**:
The per-request knobs that steer generation rather than describe the conversation:
sampling temperature, output-token cap, tool choice, Thinking Level, and Cache
Policy. It also carries the request's per-request headers and its Header Transform.
Passed alongside the Context.
_Avoid_: Config, settings, params.

**Header Transform**:
A caller-supplied final rewrite of a request's assembled headers, carried on the
Completion Options and applied after auth, construction-time, and per-request static
headers. Takes the headers by value and returns the headers to send, so it can add,
drop, reorder, or dedup entries; it has the final say over the wire.
_Avoid_: Header hook, header filter, interceptor, middleware.

**Thinking Level**:
How hard the model is asked to reason before answering, as a Provider-neutral effort
level rather than a raw token budget. A Provider that reasons by budget derives one
from the level; a Provider without extended thinking ignores it.
_Avoid_: Reasoning effort (as a wire term), budget, thinking tokens.

**Cache Policy**:
How aggressively a Provider is asked to reuse a cached prefix of the request rather
than reprocess it, as a Provider-neutral level: Off (no reuse), Standard (short
retention), or Extended (longer retention where the Provider supports it). Carried on
the Completion Options as a per-request cost knob. A Provider that caches implicitly,
or not at all, ignores it.
_Avoid_: Prompt caching, cache control, breakpoint, ephemeral, TTL.

**Assistant Message**:
A reply produced by the model: its content parts (text, Thinking, and Tool Calls),
token usage, and finish reason. It is both what a Provider returns and a Message in
the Context, so a completion is appended to the conversation without conversion.
_Avoid_: Completion, response, reply.

**Thinking**:
The model's reasoning, carried as a content part of an Assistant Message and retained
on the settled reply. It may carry an opaque signature that lets the reasoning be
replayed to the Provider on a later turn.
_Avoid_: Reasoning trace, chain of thought, scratchpad.
