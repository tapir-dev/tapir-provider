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
A specific addressable model exposed by a Provider, together with its metadata
(context window, token limits, cost).
_Avoid_: Engine.

**Capability**:
A thing a Provider can do — completion, streaming, tool calling, multimodal input,
embeddings. Embeddings live behind a separate trait; the rest are one trait.

**Credential**:
The authentication material for a Provider: either an API key or an OAuth token set
(access token, refresh token, expiry). One tagged value that round-trips losslessly.
_Avoid_: Auth, key, secret (each names only part of it).

**Token Store**:
The persistence boundary for Credentials. The SDK reads and writes through it and
owns refresh; the caller chooses where they live (file, OS keychain, ...).
_Avoid_: Keyring, vault, cache.

**Registry**:
The compiled-in catalog of available Providers. Only entries whose feature is enabled
are present, so the default build ships none.
_Avoid_: Factory, plugin list.

**Tool Call**:
A request from the model to invoke a named tool with JSON arguments. It carries two
ids: an SDK-minted id that is always present, so callers have a stable handle, and an
optional provider-native id kept when the wire protocol supplies one.
_Avoid_: Function call, invocation.
