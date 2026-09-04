# Anthropic OAuth constants and the mandatory Claude Code identity block

For Anthropic OAuth we use PKCE S256 with independent CSRF state, authorize at `claude.ai/oauth/authorize` (subscription) / `console.anthropic.com` (console), and exchange/refresh against `platform.claude.com/v1/oauth/token` — the newer endpoint used by the most recent references, chosen over the older `console.anthropic.com/v1/oauth/token`. When the active Credential is OAuth, requests must go on `Authorization: Bearer` (not `x-api-key`), carry `anthropic-beta: claude-code-20250219,oauth-2025-04-20`, and **prepend a "You are Claude Code, Anthropic's official CLI for Claude." system block**; without the identity block the API rejects the request, which is why this is baked in rather than left to the caller.

The OAuth lane also normalizes each tool name to Anthropic's accepted shape (`[a-zA-Z0-9_-]`, at most 128 characters): any other character becomes `_` and an over-long name is truncated. The identity emulation makes malformed tool names likelier to be rejected on this lane, so the Provider sanitizes them rather than surfacing a wire error. The API-key lane leaves tool names untouched.

## Consequences

- These constants are hard to reverse once callers depend on stored Credentials; the client id and token URL are env-overridable to hedge.
- A stored OAuth Credential bakes in a ~5-minute expiry safety margin so proactive refresh fires before the token is actually stale.
- Tool-name normalization is lossy: two names that differ only in disallowed characters collide after sanitizing. Callers that need stable, distinct names on the OAuth lane should pre-name their tools within the accepted set.
