# OpenRouter models: design

LLM Suite 0.36.0. Cloud models from OpenRouter work in Chat, MCP tools, Writing, Adventure, Reverse Engineering, batch
practice and the API server (so also in RulZ Coding Agent and both Android apps). Models on this PC and on vast.ai
rentals behave exactly as before.

## 1. What the owner decided

- OpenRouter only. No generic "any OpenAI-compatible address".
- Other apps (API server, Coding Agent, phones) may use only the cloud models ticked "Allow other apps". Off at first.
  A fallback or default never picks a cloud model.
- A daily spending limit of $1.00 to start with; it can be changed or switched off.
- Reverse Engineering rounds of cloud models are always recorded (when recording is on), marked with the model's name.

Out of scope: load settings, LoRA, GPU estimates, load benchmark and writing bench for cloud models; `/v1/messages`,
`/v1/responses`, `/v1/completions`, embeddings and the other llama-server endpoints; router models; buying credits.

## 2. The one idea: one seam that speaks llama-server

`src/main/cloud/` answers as a llama-server would and talks to `https://openrouter.ai/api/v1`. It is called in process,
never through a port: `InstanceManager.fetch(instance, path, init, by)` is the global fetch on `baseUrl(instance)` for this PC
and rentals, and `CloudGateway.fetch` (main/cloud/contract.ts) for a cloud instance. Every fetch to an instance goes
through it (chat service, API proxy); the benchmark refuses cloud models. Everything OpenRouter-specific (auth,
translation, cost, limits, retries, who may use it) is behind that one call.

A connected cloud model is a `LoadedInstance` with `port: 0`, `pid: null`, no `host`, `cloud: {provider, free}`,
`modelName` = its slug (runs, traces and practice counts use the whole id, `openrouter/<slug>`), `adapters: []`, `estimate: null`, and the default
`LoadConfig`. Being in process there is no token, no listening port and no health check, and no other program can reach it.

## 3. Models

Catalog: `GET /models` (no key sent), cached on disk, refreshed when older than 24 h or on request. Usable entries: text
output, chat completions, prices finite and >= 0, id not starting with `~`, not ending `:batch`, context_length > 0. Router
models (`-1` prices) and models that take neither `max_tokens` nor `max_completion_tokens` are left out because a cost
ceiling cannot be computed for them. Output must be text only.

The user adds models; `cloud.json` keeps the options and `CloudModelDef` (shared/cloud.ts): slug, contextLimit, replyLimit,
allowApps, and the last `CloudModelInfo`. Id = `openrouter/<slug>` (`isCloudModelId`). `ModelEntry.cloud` carries the
def. Catalog order: library, rentals, cloud. `remote` is never set on them.

- `contextLimit`: what the app believes the context is. Default min(model context, 32768). Chat and Writing fill the
  context they are told about, so 1M here would cost dollars per message.
- `replyLimit`: max_tokens when a request names none, and the ceiling when it names more. Default min(8192, provider max).

Connecting: `InstanceManager.ensure/load` branches for a cloud model before the engine, draft, estimate and sameConfig
steps: `gateway.check`, then a ready instance at once. The host key is taken from the run or the model ('local', 'vast:n',
'cloud'): a cloud load never unloads anything, waits for nothing, and no load ever unloads it. One instance per cloud model. Eject just
removes it. `unloadAllLocal` and every "on this PC" test use `isLocalInstance` / `isLocalModel` (shared/cloud.ts); a test greps
that no `!i.host` style check is left. Removing the model or the key ejects it; changed limits update `contextLength`.
At shutdown in-flight requests are aborted; their reservations stay on disk.

## 4. Requests

Built from scratch, never passed through: `model`, `messages`, `tools`, `tool_choice`, `parallel_tool_calls`,
`stream: true`, `max_tokens`, `stop`, sampling, `response_format`, `reasoning`, `provider`. (A non-streaming client gets
the stream collected into one JSON answer.)

- Sampling goes only when in `supported_parameters`, and only non-neutral values: temperature, top_p, top_k (>0), min_p
  (>0), `repeat_penalty` → `repetition_penalty` (≠1), presence/frequency penalty (≠0), seed, stop. DRY, XTC, logit_bias,
  grammar, lora, cache and progress fields are dropped. The window greys those out.
- A request with `tools` to a model without tool support, or with an image to a model without vision, is refused locally.
- Thinking: `chat_template_kwargs.enable_thinking`, `reasoning_effort`, `thinking_budget_tokens` become one `reasoning`
  object (effort OR max_tokens; budget only when supports_max_tokens; off = `{effort:'none'}`, or left out when the model's reasoning is
  mandatory). Nothing is sent for "auto" or for models without reasoning.
- `provider`: `data_collection: 'deny'` unless the owner allows training providers;
  `require_parameters: true` with response_format; `max_price` (below).
- `X-OpenRouter-Title: RulZ LLM Suite`, no referer, no user id.

Reply: `delta.reasoning` (else the text/summary in `reasoning_details`) → `reasoning_content`; the final chunk keeps
`usage` (with `cost`). A chunk with `error` ends the stream as an error.

**reasoning_details across tool rounds.** The adapter remembers, in memory (LRU, 32 MB), the details of
every reply that made tool calls, keyed by slug + hash of (ids, names, canonical JSON arguments of the calls). When a later request
contains that assistant message, the details are put back and `reasoning_content` is dropped. They are also passed
through in the stream (`delta.reasoning_details`), and details a client sends itself win.

**Retries.** Only for 429 and 503 answers and for failures before anything was sent (DNS, refused, connect timeout): up to 3
tries, waiting Retry-After (max 30 s, else 2/8 s), stopping when the caller aborts. A connection that breaks later may be
billed: never retried. Never mid-stream. No bytes for 120 s ends the request (also before the headers).

## 5. Money

`CloudLedger` (`cloud-spend.json`, atomic writes, one queue). Dollars are kept as integer micro-dollars, rounded up.

1. **Reserve** before the request leaves: worst case = (input estimate × 1.25 + 2000 per image) × max prompt price + max_tokens ×
   max completion price + per-request price + images × image price. Data URLs are not counted as text; `cache_control` is
   stripped. Max prompt price includes cache writes, max completion price internal reasoning. "Max price" is the highest of base and `overrides`.
   `provider.max_price` is set to the same prices, so a dearer endpoint cannot be routed to. If today's spend +
   this > limit → 402 `cloud_limit`, with the numbers, never sent. If only the open reservations are in the way, it waits
   (abortable, at most 10 minutes) for them to settle. Reply tokens streamed past 1.5 × max_tokens (estimated) end the stream. The reservation is on disk before the fetch.
2. **Settle** with `usage.cost` from the stream; actual cost is recorded even above the reservation.
3. **No usage** (stopped, broken off): settle from `GET /generation?id=` (id from `X-Generation-Id` or the chunks; tries after
   1, 4, 15 s), else charge the whole reservation. The generation id is
   written to the ledger when the headers arrive. Provably not sent, or a 4xx/429/503 answer → 0.
4. At start, reservations left by a crash are settled the same way.

Day = local calendar day at reservation. The limit is a setting: number ≥ 0.01 or off. Free models reserve 0 (their prices are 0).
Requests are refused while the ledger cannot be
read or written (a corrupt file is not a fresh start). History: per day totals (90 days) and per model today.

BYOK requests report only OpenRouter's fee in `cost`; add `cost_details.upstream_inference_cost` when `is_byok`.

## 6. Errors

The adapter answers `{error: {message, type, code}}`. `cloudErrorKind`: `limit` (our cap), `key` (no key/401), `credits`
(402/403 budget), `rate` (429 after retries), `policy` (403 moderation, 404/503 no provider under privacy or price rules),
`other`. `StreamEvent.error.fatal: true` for limit, key, credits, rate. Cloud errors are never `transport`: retrying costs.
Batch runs: a fatal error puts the item back to pending and stops the run with the sentence.

## 7. API server

- `visibleModels`: cloud entries only when `allowApps` and a key is saved. They show `state: 'loaded'` with
  `loaded_context_length` = contextLimit and `cloud: {provider: 'openrouter', free}`.
- Picking: only by the exact id `openrouter/<slug>` (`publicModelId` returns it). Fuzzy matches, the fallback, "single loaded model" and the default model never
  return a cloud model, and cloud instances are not in the ready list the fallback looks at.
  A request that names `openrouter/…` and cannot have it gets 403 `cloud_not_allowed`, never another model.
- A cloud request must be application/json, have no `Origin` (whatever the CORS setting), and an own Host. When the server listens beyond
  localhost, it also needs the API key to be set.
- `POST /chat/completions` from the proxy calls `instances.fetch(…, 'api')`; answers carry `x-rulz-cloud: 1`.
- Only the gateway's routes are proxied; other paths → 400. Load-config POST, reload, modes reload → 409. GET of
  load-config and modes work (no GPUs, no estimate, `matches: true`). Unload → 409.
- JIT bookkeeping and idle unload skip cloud. (A cloud instance has no API sampling override.)
- allowApps is checked again on every request, not only in the list.

## 8. Phones (Writer Android)

`PhoneModel.cloud?: {free: boolean}`. The phone's list and runs accept only `allowApps` models, and only as a project's
explicit model. A run with no model of its own ("newest loaded", `modelsAdapter.ready(null)`) never uses a cloud one, whoever asks; the
same for `PhoneModes.newestReady`. In the window, Chat and Writing take the active model as chosen; an adventure uses a cloud model only when
its own setting names one.

## 9. Batch practice

`BatchHost` gains `{kind: 'cloud'}`; `PoolHost.cloud`. 1 to 4 sessions, no GPU gate. The session runs with the rental
fence (`remote: true`): private REA, allowed roots. Before start the window says what it may cost and today's limit.

## 10. Window

- Page **OpenRouter** (Ctrl+0): key card, spending card (today, limit, per model, last days), privacy switch, "Your
  models", and a browser with search, filters (free, images, tools) and prices per million tokens.
- Picker: an OpenRouter badge and prices; one click connects. No load form, reload bar or estimate.
- Every place that shows a rental host shows "OpenRouter" for cloud. Composer line: text goes to OpenRouter.
- Unsupported samplers are greyed with "Not supported by this model". Stats line shows the cost. Top bar chip: spent today.

## 11. Secrets

Own encrypted file `cloud-secrets.json` (safeStorage, refused without encryption). No environment variable or file is read: a
key found elsewhere must not start spending. The key is sent only to https://openrouter.ai (redirects refused) and scrubbed from every
error and log line. Saving checks it with `GET /key`.

## 12. Other apps

Coding Agent (desktop, Android): parse and keep `cloud`; never automatic; label, price class and "text leaves this computer" notice; hide load settings and reload; treat
n_ctx <= 0 as unknown; no drafting statistics; 402/429 stop a task and a batch without retry. Writer Android: cloud label.
