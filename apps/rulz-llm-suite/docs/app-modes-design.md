# Modes, rentals, practice and PDFs in the other apps

LLM Suite 0.35.0, RulZ Coding Agent 0.23.0, Coding Agent for Android 0.10.0, Writer for Android 0.3.0.

What the three apps get of LLM Suite 0.29 to 0.34: settings per mode (with a new **Coding** mode), models on vast.ai
rentals, practice results per model, and PDF reading. Everything here is additive: an older app keeps working with a
newer LLM Suite and the other way round. `PHONE_PROTOCOL` and the Coding Agent's `REMOTE_PROTOCOL` stay 1.

Decided by the owner: a Coding mode in LLM Suite for normal coding tasks; reverse-engineering (RE), practice and batch
tasks use the model's RE settings; Writer for Android uses the settings and offers a reload, editing stays on the
computer; a practice batch with a rental's model asks once, then runs as with a local model. A reload is only ever
offered, never automatic. Phones never rent, start, load or reload anything on a rental.

## 1. LLM Suite

### 1.1 Coding mode

`AppMode` gains `'coding'` (label "Coding", last in `APP_MODES`). LLM Suite has no page of its own for it; it is what RulZ
Coding Agent uses. It shows in Load for and in the mode editor like the others. Sampling null = the Coding Agent's own
recommendation. System prompt = added after the Coding Agent's built-in prompt. `builtInSampling('coding')` is the
app's default sampling (the starting values of the editor's Own sampling).

An LLM Suite older than 0.35 drops a `coding` entry when it next saves `model-mode-defaults.json`.

### 1.2 API server

| Request | Change |
| --- | --- |
| `GET /v1/models`, `GET /v1/models/{id}` | A rental's model gains `rental: { label, reconnecting }`. Only there while it is loaded on a rental, as today. |
| header `X-RulZ-Mode: coding` or `re` | On `/v1/chat/completions` and `/props`: when the model is not loaded, it is loaded with that mode's load settings. A loaded model is used as it is. Any other value is ignored. |
| header `X-RulZ-Rental: deny` | On every proxied request: 409 instead of an answer from a rental's model, however the model was picked (the fallback for unknown names is on by default). The check is made on the instance that would answer, right before proxying. |
| reply headers `X-RulZ-Model`, `X-RulZ-Rental` | On proxied answers: the public id that answers; `X-RulZ-Rental: 1` when it runs on a rental. |
| `GET /v1/models/{id}/modes` | New, see below. Only `coding` and `re` are exposed, never the other modes' prompts. Refused with an `Origin` header, like the routes below: it carries system prompts, and CORS may be on. |
| `POST /v1/models/{id}/reload` `{ "mode": "coding" \| "re" }` | New. Loads the model again with the mode's load settings. 409 with a sentence for a rental's model, while the model loads or is in use, or while the app is busy (`reloadBlock`); nothing happens when it is not loaded or already runs with them. Returns the `modes` body. |
| `GET/POST /v1/models/{id}/modes/{coding\|re}/load-config` | New path (an older LLM Suite answers 404; a query on the old path would be ignored there and change the settings of all modes). Same bodies as `load-config`. `config` in the answer is the effective one (`modeLoad`), plus `mode` and `mode_own`. A POST builds on the effective config and stores it as the mode's own. `{"config": null}` removes the mode's own load settings. `sampling` in the patch is refused (400): API server sampling belongs to all modes. `dry_run` as before. `reload` defaults to **false** here; `true` goes through `reloadBlock`: the settings are saved, and when the reload is refused the answer is 200 with `reload_blocked: "why"`. |
| `GET /v1/re/overview?model={id}` | Practice `attempts` and `solved` count only that model, matched as on the Reverse Engineering page. Adds `practice.model` (the key used; its presence says the answer is per model), `failed` per task and `models: string[]` (keys) per session. Without `?model` nothing changes except the added `models`. |

Every new route refuses a request with an `Origin` header (a web page); the POSTs need `application/json`.

```jsonc
// GET /v1/models/{id}/modes
{
  "object": "model.modes",
  "id": "…",
  "state": "ready",            // not-loaded | loading | ready
  "instance": "…",             // id of the loaded instance (changes with every load), or null
  // The all-modes API server sampling when its override is on: it replaces these fields in every request. Else null.
  "api_override": { "temperature": 1 },
  "modes": {
    "coding": {
      "own_load": true,
      // Request fields for /v1/chat/completions from the mode's own sampling; null = none of its own.
      "sampling": { "temperature": 0.6, "top_k": 20 },
      // The thinking fields of the same sampling; null when sampling is null.
      "thinking": { "chat_template_kwargs": { "enable_thinking": true } },
      "system_prompt": "…",
      // Whether the running (or loading) settings are what this mode loads with. null = not loaded.
      "matches": false
    },
    "re": { "own_load": true, "sampling": null, "thinking": null, "system_prompt": "", "matches": true }
  }
}
```

`sampling` and `thinking` are built by the function the chat service builds its own requests with, so the two cannot drift.
They leave out what belongs to one request or one chat: the model, messages, streaming, JSON schema, reply length, seed and
adapter strengths. `matches` follows the reload bar (`reloadFor() === null`): a model loaded by hand with settings of its
own, in a mode without load settings, counts as matching.

The practice helpers (`modelKey`, `practiceRecords`, `practiceOf`, `taskRecords`) move from the renderer's
`stores/reTraces.ts` to `src/shared/practice.ts`; the store re-exports them.

One main-process function, `reloadBlock(instanceId)`, says why a model cannot be reloaded now (the existing
`busyReason`, the model gate, a benchmark, a practice batch, users of the instance). The API's reload and the phone's
reload both use it, and hold the model gate until the load has ended.

A task in another app is between two requests most of the time, so a request with `X-RulZ-Mode` also leaves a lease on the
model until 2 minutes after its end. Only the phone's reload looks at it ("RulZ Coding Agent is using the model"): the
Coding Agent itself knows whether a task of its own runs.

### 1.3 Phone (Writer for Android)

```ts
export interface PhoneModel { id: string; name: string; loaded: boolean; rental?: { label: string | null; reconnecting: boolean } }
export type PhoneMode = 'writing' | 'adventure'
export interface PhoneModeInfo {
  mode: PhoneMode
  modelId: string | null   // the model that would write: the given one, else the newest loaded one
  name: string
  loaded: boolean
  rental?: { label: string | null; reconnecting: boolean }
  ownLoad: boolean
  ownSampling: boolean
  hasPrompt: boolean
  /** Set when it runs with other settings than the mode's. `blocked`: why it can't be reloaded from here right now. */
  reload: { label: string; blocked: string | null } | null
}
'writing:models': () => PhoneModel[]                                       // rental added (label null = not loaded there)
'writing:mode': (mode: PhoneMode, modelId: string | null) => PhoneModeInfo // new
'writing:reloadMode': (mode: PhoneMode, modelId: string) => PhoneModeInfo  // new; refused for a rental, refused while busy
```

No new event: the phone asks again when a screen opens, when a job ends, after a reconnect and every 15 s while a
writing screen is visible. A computer that does not know the channels refuses them; the phone then shows nothing.

## 2. RulZ Coding Agent (desktop)

- **Which mode.** A task started as an RE task (`task.re`, so also practice and batch) is `re`; every other is `coding`.
- **Modes are read once per task**, before its first request, and kept for the whole task, so the system prompt stays
  stable. A 404 means an older LLM Suite: as before, silently. Any other failure: as before, with a notice in the
  conversation.
- **Sampling.** Coding: recommended values, then LLM Suite's Coding sampling, then the values set in this app.
  RE: LLM Suite's RE sampling when the model has some, as it is; otherwise as before. The longest answer stays this
  app's setting. Thinking turned off here removes LLM Suite's `thinking` fields. Each task says where its sampling comes
  from in one line of the conversation when LLM Suite's mode settings are used ("Uses LLM Suite's Reverse engineering
  sampling and prompt"; "2 values set here override it"; "API server sampling overrides temperature"), so nothing is
  replaced silently.
- **Prompt.** The mode's system prompt goes at the end of the system prompt.
- **Load.** Requests and `/props` carry `X-RulZ-Mode`, and `X-RulZ-Rental: deny` unless the chosen model is a rental's
  (for a batch: unless it was started with the yes). The summary requests of a task carry both too.
- **Reload bar** above the composer when `matches` is false for the mode of the next task (coding; `re` in the RE dialog):
  "Reload with Coding settings" / "Keep as is" (until `instance` changes). Refused while a task or batch runs.
- **Model settings.** Load settings get "Settings for": All modes, Coding, Reverse engineering. It opens on Coding when the
  model has its own Coding load settings, so an edit always changes what is used. The sampling section shows LLM Suite's
  Coding values as placeholders, says which of them are overridden here, and has one button that clears the overrides. A
  read-only line names the RE sampling in use.
- **Rentals.** `ModelInfo.rental` (from `rental`, or on LLM Suite 0.29 to 0.34 from a `path` that starts with `vast/`). Badge in the model list; load settings read-only with the reason; a note that messages,
  files the model reads and tool results go to the rented computer. Never picked automatically. When the chosen model
  is gone from the list and was on a rental, the app says so and does not start a task or use another model.
- **Batch with a rental.** `batch:start` takes `rentalOk`. Main refuses without it ("needs a yes"), and stops the run when
  the model in use becomes a rental's model during a run that was not started with the yes.
- **Practice.** The overview is asked for the model in use; tiles, Start next task and Only unsolved follow it. With an
  older LLM Suite: all models together, as before.
- **PDF.** `read_pdf` (path, pages, query, offset) with LLM Suite's reader (own copy of `src/main/pdf`, worker thread,
  same limits). Paths go through the project path guard; `attachment:<id>` names an attached PDF. Not offered in practice
  tasks. The id is 16 hex digits; nothing else is accepted. Attached PDFs are kept in the app's data folder per project (`attachments/<project key>/<id>/`), 50 MB each, and
  are listed in the system prompt with their page count (the 20 newest). A message's PDFs are removed with it.

```ts
interface PdfAttachment { id: string; name: string; pages: number; bytes: number }
'pdf:attachPath': (projectPath: string, path: string) => PdfAttachment          // this computer only (COMPUTER_ONLY); every projectPath must be the open project
'pdf:uploadBegin': (projectPath: string, name: string, bytes: number) => { uploadId: string; chunkBytes: number }
'pdf:uploadChunk': (uploadId: string, index: number, base64: string) => void
'pdf:uploadEnd': (uploadId: string) => PdfAttachment
'pdf:remove': (projectPath: string, id: string) => void                         // only while not sent
'task:start': (projectPath: string, text: string, attachments?: string[]) => void
'models:modes': (model: string) => ModelModes | null                            // null = older LLM Suite
'models:reloadMode': (model: string, mode: AgentMode) => ModelModes
// loadSettings / previewLoadSettings / setLoadSettings take an optional mode; setLoadSettings takes null to remove it
're:overview': (model?: string) => ReOverview
'batch:start': (projectPath, label, options, record, rentalOk?: boolean) => BatchState
```

Uploads: 512 KB chunks in order, at most one per phone connection and two in all, dropped after 2 minutes without a chunk
or when the project closes; the size is checked against `bytes` as it grows.

## 3. Coding Agent for Android

The same screens: reload bar, "Settings for", rental badge and notes, the rental yes before a batch, practice per model,
attach a PDF (system file picker). `src/shared` is copied by `scripts/sync-shared.mjs`. With an older computer the new
controls are hidden (the channel is refused).

## 4. Writer for Android

Model pickers show rentals. A line under the model names what applies (own load settings, own sampling, own prompt; a
Writing project keeps its own sampling). A bar offers the reload, with the reason when it is not possible now. "Keep as
is" hides it until the model is loaded again or the phone app restarts.
