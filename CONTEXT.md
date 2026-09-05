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

**Token Store**:
The persistence boundary for Credentials. The SDK reads and writes through it and
owns refresh; the caller chooses where they live (file, OS keychain, ...).
_Avoid_: Keyring, vault, cache.

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
A request from the model to invoke a named tool with JSON arguments. It carries two
ids: an SDK-minted id that is always present, so callers have a stable handle, and an
optional provider-native id kept when the wire protocol supplies one.
_Avoid_: Function call, invocation.
