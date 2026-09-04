# Object-safe Provider trait with the transport generic on concrete structs

The public `Provider` trait is object-safe and transport-agnostic (it returns normalized responses and boxed streams) so Providers can be held as `Arc<dyn Provider>` in the Registry and supplied by third parties; the injected `HttpClient` transport generic lives on each concrete Provider struct (e.g. `AnthropicProvider<H>`), which erases to `Arc<dyn Provider>` at registration. We chose this over a fully generic, `impl Future`-returning trait (which is not object-safe and so cannot back a heterogeneous Registry or dynamic third-party Providers) and accept `async-trait`'s boxing cost as the price of the trait-object seam.

## Consequences

- The core async methods use `async-trait` / boxed futures rather than RPITIT.
- The transport seam (one injected `HttpClient`) is what makes every Provider testable without network, while the registry-facing type stays `Arc<dyn Provider>`.
