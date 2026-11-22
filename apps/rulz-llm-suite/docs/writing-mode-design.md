# Writing mode: design

Status: stage 1 shipped in 0.20.0, stage 2 in 0.21.0; stage 3 in 0.22.0; stage 4 built for 0.23.0 (2026-11-22). Covers the
novel-writing features planned for LLM Suite 0.21.0 to 0.23.0. Stage 1 (0.20.0, fiction samplers: DRY, XTC, avoid
words) is separate and landed first.

### What stage 2 changed from this design

Found while building and in three independent reviews; the code follows these, not the text further down.

- **Sticky entries** stay in the triggered group (`reason: 'sticky'`, full priority) so an entry used once does not
  move to the fixed section on the next request and break the prompt-cache prefix.
- **Prose anchoring** uses anchor points: a block is an anchor when it contains a multiple of `anchorStep` counted
  from the first block (step clamped to half the tail budget). The start is the first anchor whose tail fits; it does
  not move while the tail fits. The tail never exceeds its budget except in the fallback (no anchor fits), which
  warns.
- **Prose reserve**: while entries and summaries are fitted, the whole prose cap is reserved whenever there is any
  prose, so the story so far does not change as the scene grows. Leftover room still goes back to the prose.
- **Scaffolding**: the assembler counts its own headings and separators (about `ceil(chars / 3)`); the renderer still
  makes one final whole-request count. Over budget means `totalTokens - templateOverhead > available`.
- **Entry cuts**: shorten every cuttable entry lowest priority first, then drop lowest first, then give the full text
  back highest priority first while it fits.
- **Outline replies** list bible entries with their ids; replies are mapped with `resolveEntryRef` (id, then name or
  alias); names that match nothing are shown on the proposal card and left out.
- **Journal recovery** offers any journal whose text differs from the disk text, whatever the file times (sync clients
  change them); an identical journal is deleted.
- **Saves** compare the LF-normalised text with the decoded disk text (an unchanged mixed-EOL file is not rewritten),
  re-check the disk hash just before the rename (conflict instead of overwrite), refuse to recreate a deleted scene
  or entry, and refuse paths whose real target (junctions, symlinks) leaves the project.
- **Quit**: `before-quit` holds the quit until the renderer has saved or journalled, because `app.quit()` fires it
  before the window's close event. Logoff (`session-end`) does not wait; the journal is the backstop.
- **Trash**: the manifest is written atomically before the moves (`status: 'moving'`), so an interrupted delete is
  listed as "(interrupted)" and can be restored. Deletes and restores take the per-file locks of the files they move.
- **Renderer**: closing a project, or opening another, keeps the project open when its text could not be saved or
  journalled. A generation is tied to the project, scene and buffer version it started from and is dropped if any of
  them changed.

## Goals and limits

- Write a whole novel with a local GGUF model in LLM Suite. Context is 16k–64k tokens; a novel is 80k–150k words,
  so every request is *assembled*: style + relevant bible entries + summaries + beats + the recent prose.
- The novel is the user's data. It lives as plain Markdown in a folder the user picked, editable in other apps
  (Obsidian, VS Code) while LLM Suite is open. The app must never silently lose or overwrite text.
- Keyword triggers only for bible retrieval (no embeddings).
- Editor: CodeMirror 6 with Markdown.

## Stages

| Stage | Version | Contents |
| --- | --- | --- |
| 2 | 0.21.0 | Writing page (Ctrl+7), projects, scene editor with inline actions + accept/reject diff, story bible with triggers, token budget bar, outline (chapters → scenes), draft scene, scene/chapter summaries, project sampling + adapters |
| 3 | 0.22.0 | Consistency check (structured findings as margin notes), tracked state per scene (model proposes, user approves), character interview + voice notes |
| 4 | 0.23.0 | Recording of AI actions + choices, export (SFT `train.jsonl` of accepted passages, DPO `pairs.jsonl` of chosen/rejected alternatives), writing benchmark (fixed scene prompts × models/adapter strengths, blind A/B, scoreboard) |

## Project folder (stage 2)

Projects root: setting `writingRoot`, default `<Documents>\RulZ Novels`. One folder per project:

```
<root>\<Project Folder>\
  project.json                 # metadata + settings (app-owned, JSON)
  style.md                     # voice / POV / tense / rules (user text, sent every request)
  outline.json                 # chapters → scenes, beats, summaries, status (app-owned, JSON)
  bible\<entry-file>.md        # one entry per file, front matter + body (user text)
  manuscript\<scene-file>.md   # one scene per file, prose only (user text)
  .rulz\history\<scene-file>\<timestamp>.md   # snapshots before overwrites (app-owned)
  .rulz\records\...            # stage 4
```

### project.json

```jsonc
{
  "format": 1,
  "id": "p-<random>",            // stable; folder can be renamed
  "title": "The Salt Road",
  "createdAt": "...", "updatedAt": "...",
  "modelId": null,               // null = the active chat model
  "sampling": { /* full SamplingParams; new projects get the fiction preset */ },
  "adapters": null,              // LoadAdapter[] like Conversation.adapters
  "budget": { "replyTokens": 1024, "proseTailTokens": 3000, "bibleShare": 0.35, "summaryShare": 0.25 }
}
```

Unknown keys are kept on rewrite (read → merge → write), so a newer version's fields survive an older app.

### Bible entry file

```markdown
---
name: Mara Venn
kind: character            # character | place | lore | other
aliases: [Mara, the Salt Witch]
triggers: [witch, Venn]    # extra words that pull the entry in
priority: 50               # 0–100, higher stays when the budget is tight
always: false              # true = every request
---
Free Markdown body: appearance, history, voice notes...
```

Front matter is a restricted YAML subset parsed by our own code: `key: value` and `key: [a, b, "c, d"]`, strings,
numbers, booleans. A file without front matter is an entry named after its file name with no triggers. Parsing
never throws; problems are shown on the entry. Entry id = file name without `.md` (renaming the file in Explorer
changes the id; references to it in outline `present` lists are matched by id then by name).

### outline.json

```jsonc
{
  "format": 1,
  "chapters": [
    {
      "id": "c-<random>", "title": "The ferry", "act": "Act I",
      "summary": "",              // model-written recap after drafting, user-editable
      "scenes": [
        {
          "id": "s-<random>", "title": "Night crossing",
          "file": "01-night-crossing-k3f9.md",   // fixed at creation; never renamed on reorder
          "beats": "Mara bribes the ferryman...", // the plan
          "summary": "",                         // recap of what was actually written
          "status": "idea",                       // idea | beats | draft | revised
          "pov": "mara-venn",                     // bible entry id or null
          "present": ["mara-venn", "oskar"],     // entry ids ticked as present
          "targetWords": 2500
        }
      ]
    }
  ]
}
```

Order is the array order. A chapter always has at least one scene. A scene file missing on disk shows as
"File missing" (never auto-recreated over something). A `.md` in `manuscript\` not in the outline is listed under
"Not in outline" with **Add to outline**.

## Data safety rules

Revised after the principal engineer's review (2026-11-22).

1. **Atomic writes** for every file: temp file in `<project>\.rulz\tmp\` (never next to the user's files, so
   Obsidian and sync clients don't see it), then rename over the target; serialized per path; transient Windows
   errors retried. `writeTextAtomic` throws after its retries (~750 ms of lock); the caller keeps the buffer, shows
   **Save failed** and retries every 5 s and on the next edit. Nothing is dropped on failure.
2. **Bytes, not strings.** Hashes are of the raw file bytes. On read the store records the file's encoding facts
   (UTF-8 BOM yes/no, line ending CRLF/LF, decided by the majority, LF for new files) and hands the editor
   LF-normalised text plus a `fileMeta`; on write it restores BOM and line endings. A document the user did not edit
   is never written. A file that is not valid UTF-8 opens read-only with a notice.
3. **External edits win until the user says otherwise.** The main process remembers the byte hash of each file as
   last read or written by the app. A save re-reads the file first: if the hash differs from the remembered one, the
   save is refused with a conflict and the editor offers **Reload from disk** or **Keep mine** (writes the user's
   text after snapshotting the disk version). A folder watcher (fs.watch, debounced 300 ms) plus a full recheck of
   open files on window focus reloads files that changed on disk **only if** the editor has no unsaved edits, no
   pending AI change and no generation running for that file; otherwise it marks the file "Changed on disk" and the
   next save goes through the conflict path. Every pending AI change records the base hash and document version it
   was made against; Accept is refused if either moved (user is offered **Regenerate**).
4. **No loss on quit or crash.** The renderer journals each dirty buffer (debounced 300 ms) to
   `<project>\.rulz\unsaved\<file>.json` (text + base hash + time) through IPC; a successful save deletes the
   journal. On close, main asks the renderer to flush (`novel:flush` → ack, timeout 3 s) before letting the window
   close (`close` handler calls `preventDefault` until the ack). On opening a project, leftover journals newer than
   their file are offered: **Restore** (opens as unsaved edits) or **Discard** (moves to `.rulz\trash`).
5. **History snapshots** of every user-text file (scenes, bible entries, `style.md`) and of `outline.json` /
   `project.json`, to `.rulz\history\<relative path>\<compact stamp>.<ext>` using the same stamp format as
   `corruptName` (no colons). Taken on first open of a file per session, before **Accept**, **Restore**, **Keep mine**
   and external-change overwrites, and otherwise at most every 10 min while editing. Thinning by age: keep all from
   the last hour, one per 10 min for a day, one per hour for a week, one per day after that (max 200 per file).
   **History** lists and restores (restore is a normal save, so it is snapshotted too).
6. **Every AI result is a pending change, including structured ones.** Editor text: a diff with per-change
   **Accept** / **Reject** / **Accept all**. Summaries, outline expansions (stage 2), state patches and check results
   (stage 3): shown as a pending proposal in their panel with **Accept** / **Edit** / **Reject**; nothing reaches disk
   until accepted.
7. **Main owns the app JSON.** `outline.json` and `project.json` are held in the main process and changed only by
   patch IPC calls addressed by id (`novel:outlinePatch` with ops such as `setSceneField {sceneId, field, value}`,
   `addScene`, `moveScene`, `deleteScene`), so a late Summarize can't overwrite a beat typed meanwhile. Main applies
   ops in order and writes the whole file atomically, with the hash check; an external edit of the JSON reloads it
   in main and the renderer is sent the new state. Unknown keys survive.
8. **A broken app JSON blocks the project, never resets it.** If `outline.json` or `project.json` doesn't parse,
   the project opens read-only with "outline.json could not be read" and **Open history** / **Show in folder**; the
   app never writes a fallback over it (do not use `readJson`'s fallback path for these files).
9. **Deletes go to `<project>\.rulz\trash\<stamp>\`** (moving files, with a `manifest.json` describing what and where
   from) with in-app **Trash** → **Restore**, instead of `shell.trashItem`, which is unreliable on redirected and
   network folders. A chapter delete is one trash folder holding all its files plus the outline fragment; the
   outline change is applied only after the move succeeded. **Empty trash** in the project menu uses
   `shell.trashItem` on the trash folder (asks first). Deleting a whole project moves its folder to the Recycle Bin
   with `shell.trashItem` after a confirm naming the folder; if that fails it says so and deletes nothing.
10. Windows paths: file names come from `slug(title)` limited to 40 characters plus a 4-character id; reserved names
    (CON, NUL, …) and trailing dots/spaces avoided; paths compared case-insensitively.

## Request assembly (pure, shared code)

`assembleRequest(input) → { messages, parts[], dropped[] }`, no I/O, unit-tested.

Revised after the principal engineer's review (2026-11-22).

Message layout, ordered from most to least stable so llama-server's prompt cache (longest common prefix) is reused:

1. **system**: fixed writing instructions for the action family + `style.md` + project POV/tense.
2. **Fixed entries**: `always` entries, then the scene's `pov` and `present` entries (sorted by priority desc, name).
3. **Story so far**: chapter summaries of earlier chapters, then scene summaries of earlier scenes in this chapter.
   A scene/chapter whose summary is missing or **stale** (its stored `summaryOf` hash ≠ the current text hash) falls
   back to its beats, and the budget bar shows "3 summaries missing, 1 stale" with **Summarize all**.
4. **This scene**: title, POV, present characters, beats, target length.
5. **Triggered entries** (matched by keyword, not fixed): **sticky per scene**: once an entry is pulled into a
   scene it stays in the set for that scene until the user removes it or the scene changes, so the set grows rarely
   and stays in a fixed order.
6. **The prose**: text before the cursor, anchored at a paragraph start that only moves in steps of about a quarter
   of the tail budget (so successive Continue presses share the same start); plus a short piece after the cursor
   when inserting mid-text. **Draft scene** with an empty scene uses the last ~600 tokens of the previous scene.
7. **Task** (last): the action's instruction (continue / rewrite selection with variant / expand / condense /
   dialogue for beat / draft scene / summarize / expand outline), plus the user's free note.

Token counts are real, not estimated: the renderer counts each candidate part (and each bible entry, summary and
paragraph block) with a new `chat:countText` IPC (the existing cached per-instance tokenizer in ChatService), and
passes the counts into the pure assembler, which therefore needs no tokenizer and no retry loop. Chat-template
overhead is added as a measured constant per model (count of an empty assembled request). One final
`chat:countTokens` of the whole request checks it; if it is over by template drift, the prose tail drops one step
and that is the only retry.

Budget: `available = model context − replyTokens − max(256, 3%)`. `replyTokens` is sent as `max_tokens`. Fill
order, each part taking what it needs up to its cap:

| Order | Part | Cap | When short |
| --- | --- | --- | --- |
| 1 | system + style | none | warning when > 30% of available |
| 2 | task + scene | none | warning when > 15% |
| 3 | prose tail | `proseShare` (default 35%) of what is left, floor 800 tokens | shrinks first, in anchor steps, down to the floor |
| 4 | fixed entries | `bibleShare` (default 30%) of available, shared with 5 | bodies of lowest priority cut to their first paragraph, then dropped |
| 5 | triggered entries | the rest of `bibleShare` | lowest priority dropped |
| 6 | story so far | everything left | oldest chapter summaries dropped first |

Anything left after 6 goes back to the prose tail (so a 64k context uses a long tail). `parts[]` returns tokens per
part, their caps and every dropped or cut item with the reason, for the budget bar.

Generation goes through `chat:start` with a new `prefitted: true` flag: the chat service skips `fitToContext`
for such requests and, if the request does not fit, fails with a clear error instead of trimming. The project's
sampling and adapters are used; `disableThinking` on by default for prose actions and off for outline/check
actions.

Trigger matching: case-insensitive whole-word (Unicode letters) match of name, aliases and triggers in the scan
text = scene beats + the last ~2000 characters before the cursor + the selection + the user's note. One extra pass
over the matched entries' bodies (depth 1) at half priority.

To verify with a spike on a real model before the editor actions are finished (stage 2, first task):
- "Continue" prompt shape that gives clean prose with no preamble ("Sure, here is…"), with and without an
  assistant prefill of the last sentence; pick per model family if needed.
- Whether per-request `lora` scales or changed sampling clear llama-server's prompt cache.
- Whether `inst.contextLength` is per slot when parallel slots > 1 (if so, use per-slot context).
Alternatives (2–4) run one after another (llama-server usually has one slot) with different seeds; each shows as a
tab on the pending change.

Structured actions (expand outline, summaries in stage 2; checks and state in stage 3) use `jsonSchema` output and
validate with zod; an invalid reply shows the raw text and **Retry**, and changes nothing.

## UI (stage 2)

Writing page, nav rail item with a Feather icon, **Ctrl+7**:

- Left column: project picker (+ **New project**, **Open folder**), outline tree (chapters → scenes, drag to
  reorder, status dot, word count), bible list (filter by kind, **New entry**).
- Centre: CodeMirror scene editor (Markdown, soft wrap, serif-free UI font per the RulZ guide; prose width ~72ch),
  toolbar with the actions; selection actions in a small floating menu; pending AI changes as a unified merge view
  with per-change Accept/Reject (`@codemirror/merge`).
- Right panel (toggle Ctrl+B like chat): Scene (beats, POV, present, target, summary + **Summarize**), Context
  (budget bar by part, included/dropped entries with why: "trigger: witch"), Project (model, sampling incl. stage 1
  samplers, adapters + strengths, budget numbers, style.md editor).
- Status bar: words in scene / chapter / book, save state ("Saved", "Saving…", "Conflict"), tokens of last request.

Design: RulZ guide tokens already used by the app's Tailwind theme; one primary button per view (**Draft scene** or
**Accept all** when a change is pending); sentence case; Lucide icons; no emoji.

## Stage 3 design (0.22.0)

Four features: consistency check, tracked state, character interview, voice notes. Same rules as stage 2: the
user's files are never written without an explicit action, AI output is a proposal until accepted, main owns the
JSON files and applies validated ops, a damaged file is never overwritten. Read by the principal engineer before
building; its fixes are folded in below. The contract types are in `src/shared/novel.ts` (section "Stage 3").

### Files

| File | Owner |
| --- | --- |
| `state.json` (project root, next to outline.json) | main, through `novel:statePatch` ops |
| `.rulz/checks.json` | main, through `novel:dismissFinding` |
| character entries `bible/<id>.md` | user text; the renderer edits the body like any entry |

`state.json`:

```json
{ "format": 1, "facts": [ { "id": "f1a2b3", "entity": "mara-venn", "kind": "location", "value": "the lighthouse",
  "about": null, "from": "s9x8y7", "until": null } ] }
```

- `kind`: `location | knows | carries | condition | relationship | other`. `about`: the other entry's id, only for
  `relationship` (required there), else null. `value`: trimmed, one line, 1-300 characters.
- `from`: the scene where the fact became true. `until`: the scene where it stopped being true, or null.
- Unknown keys on the file and on each fact are kept when main rewrites it.
- **A fact that does not validate on read** (bad kind, empty value, missing id, …) is kept verbatim in the file on
  every rewrite and reported in `stateInvalid` (index + reason); the State tab lists it as "could not be read" with
  **Remove** (op `removeInvalid` by index + its JSON hash). It is never silently dropped.

**Damaged / missing**

- No `state.json` when the project opens: empty state; the file is created by the first write.
- The file disappears while the project is open, is 0 bytes, is not JSON, has `format` > 1 or no `facts` array:
  `stateProblem` (message), state read-only, never written. Offered: **Show in folder**, and for a missing/0-byte
  file **Start empty state** (an explicit op `reset` that writes `{format:1,facts:[]}`; refused if the file has
  content).
- `.rulz/checks.json` unreadable: dismissals ignored and not written (`checksProblem`).

**Writes** (state.json): atomic, under the project chain after `loadJson`, stale check like outline.json (re-read
and re-apply the ops if the file changed on disk), the same re-check-before-rename guard as `saveFile`, a history
snapshot before each write (gap rule as for text files; `state.json` is allowed in history), and watched
(`onExternalChange` reloads it and emits `state`). Blocked project → refused.

### Reading-order semantics

Scenes are ordered by their position in the outline (chapter order, then scene order). `pos(s)` is that index.

- **State after scene S**: facts with `pos(from) <= pos(S)` and (`until` null or `pos(until) > pos(S)`).
- **State before scene S** (used by assembly, Update state and the interview): facts with `pos(from) < pos(S)` and
  (`until` null or `pos(until) >= pos(S)`).
- **Location is single-valued**: for each entity, only the `location` fact with the latest `from` holds; older ones
  are superseded even without an `until`. Other kinds may hold several values.
- A fact's **status**: `ok`, `detached` (its entity, about, from or until names something not in the bible/outline),
  or `inverted` (`pos(until) < pos(from)`, after a scene move). Only `ok` facts are used in requests. The State tab
  lists detached and inverted facts with **Edit** (any field, including entity, from, until) and **Remove**, and the
  tab label shows their count. Deleting a scene, chapter or entry does not touch state.json, so restoring it from the
  trash re-attaches its facts.

### Ops (`novel:statePatch(folder, ops)` → `NovelStatePatchResult`)

```ts
type StateOp =
  | { op: 'add'; fact: StateFact }       // id made by the renderer ('f' + random); must be new
  | { op: 'end'; factId: string; until: string }
  | { op: 'edit'; factId: string; changes: Partial<Omit<StateFact, 'id'>> }
  | { op: 'remove'; factId: string }
  | { op: 'removeInvalid'; index: number; hash: string }
  | { op: 'reset' }
```

Main validates each op against the current outline, bible and state and **applies the valid ones, returning the
rejected ones with a reason** (`{ state, rejected: [{ index, reason }] }`); one bad op does not sink a card. Rules:
entity/about are existing entry ids (`edit` may set them to fix a detached fact), from/until existing scene ids,
`pos(until) >= pos(from)` for new values, value rules; an `add` identical to an existing fact (entity, kind, value,
about, from) is refused as a duplicate; `end` on a fact that already has an `until` is refused.

### Tracked state in requests

Two placements, so the prompt-cache prefix only changes when the scene changes:

- **State of fixed entities** (always / POV / present): new part `state`, right after **Story so far** and before
  **This scene**: system → fixed entries → story so far → **state** → this scene → triggered entries → prose → task.
- **State of triggered entities**: under each triggered entry inside **Also relevant**, as part of that entry's text
  (cut and dropped with it).

Format: `### Mara Venn` then `- location: the lighthouse`, `- relationship (Oskar): distrusts him`. Cap: a constant
8% of the available context (no setting); fitted **before** story so far, charging its actual cost; over the cap
whole entity groups are dropped, POV kept first, then present, then by entry priority. Shown in the Context tab.

### Update state

Scene toolbar **Update state** (needs scene text). Request: the bible entries in the scene (ids listed, as in
outline replies), the state before the scene with fact ids, **the facts already recorded from or ended at this
scene** (so a re-run does not repeat them), and the whole scene text (if it does not fit, the request is refused
with the reason). JSON reply:

```json
{ "changes": [ { "op": "add", "entity": "Mara Venn", "kind": "carries", "value": "the brass key", "about": null },
               { "op": "end", "factId": "f1a2b3", "reason": "she left the lighthouse" } ] }
```

Refs resolve with `resolveEntryRef`; unresolved names and unknown fact ids are shown on the card and left out. The
card is tied to the scene's text hash: if the text changed since the request, **Apply** is refused (regenerate).
Each change has **Accept** / **Reject**; **Apply accepted** sends one `statePatch` with `from` / `until` = this
scene and shows any rejected ops with their reason. Nothing is written before that.

### Consistency check

Scene toolbar **Check**; chapter context menu **Check chapter** (one request per scene in order, stoppable). Request:
style, bible entries matched for the scene, story so far, state before the scene, the scene text. JSON reply, at
most 20 findings:

```json
{ "findings": [ { "type": "continuity", "quote": "exact words from the scene", "problem": "...",
  "conflictsWith": "what it contradicts", "suggestion": "..." } ] }
```

`type`: `continuity | character | timeline | pov | tense | other`. Findings live in memory per scene with the text
hash they were made from (not on disk). `locateQuote(text, quote)`: exact match on a normalised copy (curly quotes
and dashes folded, whitespace collapsed, case-insensitive) mapped back to original offsets; else "not found in the
text" (finding listed, no mark). Located findings get a gutter mark and an underline; edits remap ranges; a range
that is deleted drops its mark. **Checks** side-panel tab: type, quote, problem, conflicts with, suggestion,
**Jump**, **Fix** (Rewrite → Custom on the quote's range with the suggestion as the note: a normal pending change),
**Dismiss**.

Dismissed: `findingKey = hash(type + normalised quote)`; `.rulz/checks.json` is
`{ format: 1, dismissed: { [sceneId]: string[] } }`, written by `novel:dismissFinding(folder, sceneId, key,
dismissed)` (refused on a blocked project). Later checks hide dismissed findings for that scene.

### Voice notes

A character entry's body may have a section headed `## Voice` (level 2 or 3, case-insensitive), up to the next
heading of the same or a higher level. Helpers: `voiceSection(body)` and `appendVoiceNote(body, note)`: edits the
given text (the renderer's buffer, never re-formatted through `formatEntry`), creates the section at the end if
missing, appends `- <note>` with the body's line endings; the note is collapsed to one line, a leading `#` removed,
capped at 500 characters.

**Voice targets** (characters only): for **Dialogue**, the POV and present characters; for any action with a
selection, characters whose name or alias appears in the selection. A voice target gets `voice: true` on its
`AssemblyEntry` (its `reason` is unchanged), is sorted first within its own group (fixed or triggered; groups never
move), and its short form keeps the first paragraph plus the voice section capped at about 400 tokens.

### Character interview

Side-panel tab **Interview**: pick a character (default: the scene's POV). System prompt:

- "You are <name>. Answer in character, in your own voice. You know only what <name> knows by this point in the
  story; if asked about later events, you do not know them."
- The entry body, labelled "The author's notes about you (they may mention things you have not lived yet; do not
  reveal those)".
- The character's state before the current scene.
- Summaries only of earlier scenes where the character is POV or present.

Chat with the project's model and sampling; history in memory **per character and scene** (cleared on project
close; **Clear**). The whole request is counted before sending; the oldest turns are dropped to fit (never the
system prompt). Each answer has **Add to voice notes** (the answer, or the selected part): `appendVoiceNote` on the
entry buffer, which saves through the normal entry path; disabled with the reason when the entry has a pending
change or a conflict.

### Out of scope for stage 3

Automatic state updates without approval; checks across the whole book in one request; storing check results or
interview transcripts on disk; fuzzy quote matching; a setting for the state share.

### What stage 3 changed from this design

- State groups are fitted after the fixed entries and before the triggered entries, so tracked state outranks
  triggered entries (the design only said "before story so far").
- The whole-scene refusal for Check and Update state also fires on a "whole scene" warning, since one oversized
  paragraph gives a warning with `proseStart` 0.
- The reply parsers come in two forms: `parseCheckReply` / `parseUpdateStateReply` throw on bad JSON; the renderer
  uses the non-throwing `...Result` forms to show the raw reply with Retry.
- The State tab has a third view, **All**, so facts that start later or already ended stay reachable; the Checks tab
  can bring back dismissed findings; findings with the same type and quote are shown once.
- Untouched facts are written back exactly as read. A value with leading or trailing spaces or a line break is
  invalid on read (kept verbatim and listed), not trimmed.
- `state.json` and `.rulz/checks.json` are decoded as strict UTF-8; invalid bytes make the file a problem.
- A 0-byte `.rulz/checks.json` counts as empty (it holds only dismissals). Outside edits to `checks.json` are picked
  up at the next load, not live: the watcher ignores everything under `.rulz`.
- `writeBytesAtomic` re-runs its last-moment check before every rename attempt, which also protects scene saves
  during Windows lock retries.

## Stage 4 design (0.23.0)

Recording of writing AI actions, export as SFT / DPO training data, a "use with" hint for style adapters, and a
writing benchmark with blind A/B votes and Elo. (Revised after the principal engineer's read: region geometry,
reverted finals, idle finals, missing-final flag, finish reasons and viewed alts, delete race, lock retry, one file
per benchmark run, a global busy check, streamed export, card actions recorded but not exported.)

### Recording

**What is recorded.** Prose actions (continue, rewrite, expand, condense, dialogue, draftScene) and card actions
(summarizeScene, summarizeChapter, outlineChapters, outlineScenes). Card lines are recorded but not exported in
0.23. Not recorded: checkScene, updateState, interview.

**Switch.** `project.json` `recording?: boolean`; absent means on (also for 0.21 / 0.22 projects). Project panel:
a **Recording** switch, the line "Saves each AI request and what you did with it in .rulz\records, for training
data. Nothing leaves this PC.", the records' total size, **Export training data…** and **Delete records…** (asks
first; moves the records folder to the Windows Recycle Bin). Main is
authoritative: it checks the flag on every append.

**Files.** `.rulz/records/<YYYY-MM-DD>.jsonl` (local date of the event), one JSON object per line, `format: 1`.
Event lines joined by `id` at export. `id` = 16 lowercase hex chars made in the renderer when a run starts
(not `uid()` unless it matches), carried on `PendingChange` / the proposal.

- `request` — once when a run ends with at least one alternative (no line when none was produced):
  `{format, type:'request', id, at, app, action, variant?, sceneId|null, chapterId|null, model:{id,name},
  sampling (the values actually sent), adapters:[{name,scale}], messages (exactly as sent),
  alts:[{index, seed, generated, finish:'stop'|'length'|'stopped'|'error'}]}`. `length` = cut off at max tokens
  (from the chat result's finish reason; if the chat path cannot report it, the renderer marks an alt `length`
  when it ended without the user stopping and the reply reached maxTokens by token count, else `stop`). Prose:
  `generated` is the new text only (PendingAlt.generated). Cards: the raw reply text.
- `viewed` — prose: `{type:'viewed', id, at, alts:[indexes the user displayed]}` written with the outcome (the
  first active alt counts as viewed; selectAlt adds).
- `outcome` — when the user resolves it:
  - prose: `{type:'outcome', id, at, kind, alt, region}`; kind `accepted` (final == the active alt's document),
    `mixed` (some hunks kept), `rejected` (final == base), `abandoned` (regenerate, close, open another project,
    stale refusal: alt null, region null). `region` = `{start, end}` of the generated text in the accepted document,
    from the apply geometry (where the alt's generated text was placed: for rewrite the selection start, for
    continue the cursor plus any lead), not from a diff. For `mixed`, `start` is the same and `end` = final length
    minus the length of the text after the region in the alt document (all hunks lie inside it).
  - cards: kind `accepted` | `edited` (with `text`) | `rejected` | `abandoned`.
- `final` — prose only, after `accepted` / `mixed`: the region text after the user's later edits.
  `{type:'final', id, at, text|null, reason:'idle'|'left'|'action'|'close'|'reloaded'|'outside'}`.
  Written when the scene has been idle 10 s after an edit since the accept (repeats allowed; **the last final for
  an id wins** at export), and on leaving the scene, a new action on it, project close or quit flush. Tracking ends
  when the scene is left, a new action starts on it, or the project closes. Located by rule: if the later document
  starts with accepted[0:start] and ends with accepted[end:], text = later[start : later.length - (accepted.length -
  end)]; else text null, reason `outside`. Reloads from disk, journal restore and conflict "Reload" write text null,
  reason `reloaded`, and end tracking.

**Writing a line.** Renderer → `novel:record(folder, line)` returns boolean. Main:
- drops it (false) when the project is not open, blocked, or recording is off;
- validates shape per type (fields and types as above; `id` /^[0-9a-f]{16}$/; serialized length <= 8 MB);
- builds the path from the local date; `assertAppPath` + `guard`; `serialized(path)`; re-checks that the project
  is still open right before `mkdir`; `fs.appendFile(line + '\n')` wrapped in a transient-error retry (EBUSY,
  EPERM, EACCES; ~2 s total); prepends `\n` if the file's last byte is not `\n`;
- tracks in-flight appends per project; `closeProject` (and so delete project) awaits them;
- never throws for I/O: returns false and logs; the renderer shows one quiet toast per session and carries on.

### Export

**Export training data…** → folder picker → `novel:exportRecords(folder, outDir, opts)`.
- `outDir` inside the project folder (realpath, case-insensitive) is refused.
- Writes a new subfolder `<outDir>\<project-slug>-<YYYYMMDD-HHMMSS>\`: `train.jsonl`, `pairs.jsonl`,
  `manifest.json`, each written to a temp name and renamed.
- Streams: first pass reads every records file line by line and keeps only outcome / viewed / final lines per id
  (last final wins; first outcome wins); second pass reads request lines one at a time and writes output lines as
  it goes. Strict UTF-8 per line; bad, torn or wrongly shaped lines are counted as dropped, never fatal.
- **train.jsonl** `{messages:[...recorded messages, {role:'assistant', content}], meta:{id, action, at, model,
  kind, final:true|false}}`: prose `accepted` / `mixed` only; content = the last final's text if present and not
  reverted, else (accepted) the alt's `generated` with `meta.final: false`, else (mixed, no final) dropped.
  A final whose text equals the base region (the text the region replaced: the selection for rewrite, '' for
  continue) is **reverted** and exported as rejected (dropped from train). Assistant text shorter than 20
  characters after trimming is dropped.
- **pairs.jsonl** `{prompt:[messages], chosen, rejected, meta:{id, source:'alts'|'edit'}}`: prose `accepted` only,
  chosen = the accepted alt's generated; one pair per other alt that was **viewed**, finished with `stop`, and
  differs after trimming. Alts with finish `length` are never in pairs. Option **Edit pairs** (off by default):
  chosen = final text, rejected = accepted generated, only when a final with `meta.final` exists, differs, is not
  reverted, and is not just the accepted text with more text added at the end.
- `manifest.json`: format 1, app version, created, project title, date range, counts per file and per action,
  dropped reasons with counts (including `noFinal`), the options.
- Cards are not exported in 0.23. Text is written as recorded.

### Style adapter "use with" hint

- Adapter convert dialog success: "Trained on <base or 'an unknown base model'>. Add it to a chat's adapters or a
  Writing project's **Adapters**."
- Writing Project panel adapter list: "Trained on X" in `muted` per adapter, and a `warn` badge "Different base
  model" when `adapterBaseMatch` says mismatch against the project model.

### Writing benchmark

**Benchmark** on the Writing page start view (no project needed) opens the benchmark view.

- **Prompts**: 8 bundled scene briefs (original, 2-4 sentences, varied genres) plus the user's own (add, edit,
  delete).
- **Candidates** (2-6 per run): model, adapters with strengths, sampling (current defaults or a preset), label.
- **Run**: chosen prompts × candidates × seeds (1-3; one seed list per run shared by all candidates), max tokens
  per output (default 600). Requests: fixed benchmark system prompt + the brief; thinking off.
- **Loop** in the renderer with injected deps (the `runBatchLoop` pattern): one output at a time, grouped by
  candidate so each model / adapter set loads once; Stop after this / Stop now (a "now" output goes back to
  pending). Finish reason recorded per output.
- **Busy**: one shared check, `aiBusy()` in `lib/aiBusy.ts`, true while a Writing generation, a chat reply, a
  batch run, a hardware benchmark or a writing benchmark runs. The benchmark refuses to start or continue a step
  while anything else is busy; Writing AI actions refuse while a benchmark generates.
- **Voting**: blind pairs (same prompt and seed, two different candidates), shown as A / B in a stored random
  order; **A is better**, **B is better**, **Tie**, **Skip**; names shown after the vote; each pair voted once.
- **Scoreboard per run**: Elo from the run's votes in vote order (start 1000, K 32, tie 0.5), plus wins / losses /
  ties; a pure function, never stored as the source of truth.
- **Storage**: `%APPDATA%\RulZ LLM Suite\writing-bench\runs\<id>.json` (one file per run) and
  `writing-bench\prompts.json`. Written atomically, serialized per file, saved after every output and vote.
  A run file that cannot be read or validated is listed as damaged and never rewritten or deleted by the app (the
  user can delete it). Runs still `running` on load are shown as `stopped` (written back only on the next change).
  No automatic cap.

### Out of scope for stage 4

Training inside the app; uploading anything; exporting card actions; recording checks, state updates or
interviews; LLM-as-judge voting; benchmark prompts with project context.

## Module plan (stage 2)

- `src/shared/novel.ts`: types (contract between main and renderer).
- `src/shared/novelAssembly.ts`, `src/shared/novelPrompts.ts`, `src/shared/frontMatter.ts`: pure logic + tests.
- `src/main/novel/store.ts` (+ `watch.ts`, `history.ts`): file I/O, hashes, conflicts, history, trash; IPC
  channels `novel:*` and event `novel:changed`.
- `src/renderer/src/pages/WritingPage.tsx`, `components/writing/*`, `stores/novel.ts`.
