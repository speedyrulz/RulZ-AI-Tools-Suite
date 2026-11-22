# Writing mode: design

Status: stage 1 shipped in 0.20.0; stage 2 built for 0.21.0 (2026-11-22); stages 3 and 4 still planned. Covers the
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

## Stage 3 outline

- **Check scene/chapter**: request with bible + summaries + the text; JSON findings
  `{type: continuity|character|timeline|pov|tense|other, quote, problem, conflictsWith, suggestion}`; quotes are
  located in the text (fuzzy) and shown as gutter marks; dismissing one is remembered per scene in `.rulz\checks.json`.
- **Tracked state**: `state.json` with per-entity facts (location, knows, carries, condition, relationships), as of a
  scene id. After a scene is accepted as draft, **Update state** asks the model for a JSON patch; the user approves
  each change (same pattern as tool approval). The state as of the previous scene goes into assembly as its own part.
- **Character interview**: a chat-like panel whose system prompt is the entry body + state as of the current scene
  + "you know only what you would know by now"; answers can be appended to the entry's voice notes.
- **Voice notes**: a `voice:` section in character entries; when the action is dialogue or the selection contains
  that character's speech, their voice notes are given top priority.

## Stage 4 outline

- **Recording** (on by default per project, switch in Project panel): every AI action's request (assembled
  messages + sampling + model + adapters) and every alternative, plus what the user did (accepted which, rejected,
  edited after accepting: final text after 60 s or scene switch) → `.rulz\records\<date>.jsonl`.
- **Export**: `train.jsonl` (chat format, accepted final text as the assistant turn) and `pairs.jsonl`
  (`prompt`, `chosen`, `rejected` from alternatives where one was accepted and another rejected), plus
  `manifest.json`; written to a folder the user picks.
- **Style adapter per project**: project.json `adapters` already in stage 2; stage 4 adds "use with" hints after
  converting a PEFT adapter.
- **Writing benchmark**: a set of scene prompts (bundled defaults + user-made) × candidates (model + adapter
  strengths + sampling preset); each candidate's output generated with the same seed set; blind A/B rounds pick
  winners; Elo-style scoreboard; kept in `%APPDATA%\RulZ LLM Suite\writing-bench.json`. Reuses the batch-run
  pattern (one task at a time, Stop after this / Stop now).

## Module plan (stage 2)

- `src/shared/novel.ts`: types (contract between main and renderer).
- `src/shared/novelAssembly.ts`, `src/shared/novelPrompts.ts`, `src/shared/frontMatter.ts`: pure logic + tests.
- `src/main/novel/store.ts` (+ `watch.ts`, `history.ts`): file I/O, hashes, conflicts, history, trash; IPC
  channels `novel:*` and event `novel:changed`.
- `src/renderer/src/pages/WritingPage.tsx`, `components/writing/*`, `stores/novel.ts`.
