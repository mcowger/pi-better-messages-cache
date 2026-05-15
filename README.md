# pi-better-messages-cache

A [pi](https://github.com/badlogic/pi-mono) extension that implements the
**dual cache-breakpoint strategy** for Anthropic models, dramatically improving
prompt-cache hit rates on MiniMax, Kimi, and other Anthropic-compatible
providers.

It also fixes a **streaming control-character bug** where the Anthropic SDK's
`stream()` crashes on raw `\t` / `\n` bytes inside tool-call JSON, leaving tool
arguments as `{}` and producing unrecoverable error results.

> This implements the optimization proposed in
> [badlogic/pi-mono#1737](https://github.com/badlogic/pi-mono/pull/1737),
> which the upstream maintainer declined to merge into core.

---

## The problem

The built-in Anthropic provider marks the last *user* message block with
`cache_control`.  On some providers — notably **MiniMax** and **Kimi** — the
preceding assistant `tool_use` and `thinking` blocks sit *outside* the cached
window, so the cache must be re-read from scratch on almost every turn:

```
turn N
  [assistant]  thinking …          ← NOT cached ✗
               tool_use foo        ← NOT cached ✗
  [user]       tool_result foo     ← cache_control ✓  (only marker)

turn N+1
  The cache window starts at tool_result, missing the assistant blocks above.
```

## The fix

Mark **two** locations per turn:

| Location | Who marks it |
|---|---|
| Last **assistant** `tool_use` block | **This extension** (new) |
| Last **user** message block | Built-in provider (preserved) |

Both markers together ensure the full assistant turn (thinking + tool_use +
tool_result) sits inside the growing cached prefix on every subsequent call:

```
turn N
  [assistant]  thinking …
               tool_use foo  ← cache_control ✓  (marker 1 — NEW)
  [user]       tool_result foo  ← cache_control ✓  (marker 2 — existing)

turn N+1
  The cache window now covers the entire assistant turn above.
```

This dual-marking pattern aligns with the cache strategies used by
**OpenCode**, **Kilo Code**, and **Roo Code**.

### Streaming control-character fix

The Anthropic SDK's `stream()` method parses each SSE event with bare
`JSON.parse`.  When the model emits raw tab (`\t`) or newline (`\n`) bytes
inside a tool-call JSON argument (e.g. tab-indented `oldText` in an Edit
call), `JSON.parse` throws:

```
Bad control character in string literal in JSON at position N
```

This error propagates out of the stream, cuts it before `content_block_stop`
fires, and leaves the tool-call arguments as `{}`.  The resulting error is
displayed as the tool result and cannot be retried because the model has no
context about the original arguments.

**Fix:** this extension replaces `client.messages.stream()` with
`client.messages.create().asResponse()` to get the raw HTTP response, then
parses SSE events using `parseJsonWithRepair` (from `@earendil-works/pi-ai`),
which escapes raw control characters before handing off to `JSON.parse`.
Streaming tool-call argument accumulation also uses `parseStreamingJson` instead
of bare `JSON.parse`, ensuring partial JSON with control characters is handled
correctly throughout the stream lifecycle.

### Anthropic cache breakpoint limit

Anthropic-compatible APIs allow a maximum of **4 total blocks** with
`cache_control` in a single request.

That limit applies across the **entire payload**, including:
- system prompt blocks
- assistant `tool_use` blocks
- user / `tool_result` blocks

In longer multi-turn conversations, a naive dual-marking strategy can
accidentally exceed that limit and trigger errors like:

```text
A maximum of 4 blocks with cache_control may be provided. Found 5.
```

To prevent this, this extension now enforces the limit before sending the
request:
- keep system prompt cache markers intact
- keep the **newest** message-level cache breakpoints
- remove **older** message-level cache breakpoints first

This preserves the most useful recent cache anchors while ensuring requests
never exceed Anthropic's hard cap.

### Empirical impact (from PR #1737 field data)

| Provider | Before | After |
|---|---|---|
| MiniMax / Kimi | near-zero cache hits | **80 %+ cache hit rate** |
| Anthropic native | baseline | small positive improvement |

#### Built-in pi caching — "cache hit wall" (MiniMax)

<img src="docs/builtin-cache-hit-wall.png" width="50%">

> **Note:** Notice the "cache hit wall" at ~4.2K cache hits — the orange cache-hit line flatlines, while the cache-miss line continues climbing.

#### With pi-better-messages-cache extension — drastically improved cache hits

<img src="docs/better-messages-cache-minimax-comparison.png" width="50%">

> **Note:** Cache hits continue climbing throughout the session — the orange line no longer flatlines, achieving the dual cache-breakpoint strategy's intended behavior.

---

## How it works

`pi.registerProvider("anthropic", { api: "anthropic-messages", streamSimple })`
replaces the global api-registry entry for the `"anthropic-messages"` API type.
This transparently intercepts every model that uses that API — all native
Anthropic models — **without touching any model definitions, pricing, OAuth
config, or other settings**.

The custom `streamSimple` handler:

1. **Applies dual cache breakpoints** — marks both the last assistant
   `tool_use` block and the last user message block with `cache_control`.
2. **Enforces the 4-breakpoint limit** — keeps system prompt markers and the
   newest message-level breakpoints, removing older ones first.
3. **Streams via raw HTTP + custom SSE parser** — uses
   `client.messages.create().asResponse()` instead of the SDK's `stream()`,
   then parses SSE events with `parseJsonWithRepair` to handle raw control
   characters in tool-call JSON.
4. **Uses `parseStreamingJson` for argument accumulation** — ensures partial
   tool-call JSON containing control characters is parsed correctly throughout
   the stream, not just at the end.

---

## Installation

```bash
# Global install (all projects)
pi install npm:@mcowger/pi-better-messages-cache

# Project-local install
pi install -l npm:@mcowger/pi-better-messages-cache
```

### Try without installing

```bash
pi -e npm:@mcowger/pi-better-messages-cache
```

### From git (latest unreleased)

```bash
pi install git:github.com/mcowger/pi-better-messages-cache
```

---

## Requirements

- [pi](https://github.com/badlogic/pi-mono) (any recent version)
- `@earendil-works/pi-coding-agent` and `@earendil-works/pi-ai` (bundled with pi,
  listed as `peerDependencies`)

---

## Uninstalling

```bash
pi remove npm:@mcowger/pi-better-messages-cache
```

This restores the built-in Anthropic stream handler automatically.

---

## License

MIT © mcowger
