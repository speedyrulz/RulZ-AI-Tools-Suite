# Adventure mode design

Adventure mode is a choose-your-own-adventure story played with the loaded model. The model writes a passage and
offers 2 to 4 choices. The reader picks one or types their own action, and the story goes on from there. Every
step is kept as a tree, so the reader can go back to any earlier passage, take another path and switch between
paths. The mode is a page in the desktop app's nav rail and a section of RulZ Writer for Android.

The contract (types, channels, tree helpers) is in `src/shared/adventure.ts`. This document covers what the
contract does not.

## Where things run

Generation runs in **main**, as jobs, the same way the phone's Writing runs do (`src/main/writing/runs.ts`). The
desktop window and the phone call the same `adventure:*` channels (`ADVENTURE_CHANNELS`) and get the same
`adventure:job` and `adventure:changed` events. Neither of them talks to the model directly for this mode.

| Part | Files |
| --- | --- |
| Contract and tree helpers | `src/shared/adventure.ts` |
| Prompts, reply parsing, context fitting, Markdown export, starters (pure) | `src/shared/adventurePrompts.ts` |
| Saved files | `src/main/adventure/store.ts` |
| Jobs (generation) | `src/main/adventure/runs.ts` |
| Handlers for the window and the phone | `src/main/adventure/ipc.ts` |
| One model job at a time | `src/main/modelGate.ts` (used by `src/main/adventure/runs.ts` and `src/main/writing/runs.ts`) |
| Wiring | `src/main/index.ts`, `src/shared/ipc.ts` and `src/shared/phone.ts` (channel lists, done), `src/main/phone/registry.ts` |
| Desktop page | `src/renderer/src/pages/AdventurePage.tsx`, `src/renderer/src/components/adventure/*`, `src/renderer/src/stores/adventure.ts` |
| Phone screens | `apps/rulz-writer-android/src/pages/Adventures.tsx`, `src/pages/AdventurePlay.tsx`, `src/components/adventure/*`, `src/stores/adventure.ts` |

## Saved files

One JSON file per adventure: `<userData>/adventures/<id>.json`, the `Adventure` type, written with
`writeJsonAtomic` (`src/main/util/jsonFile.ts`). Ids are random (`crypto.randomUUID()`), never taken from a
title, and a channel argument that is not a plain id (`/^[a-z0-9-]{8,64}$/i`) is refused, so no path can be built
from a caller's input.

- Writes to one adventure run one after another (a promise chain per id).
- Every save raises the adventure's `rev` and emits `adventure:changed` with what changed (`AdventureChange`: the
  added or changed nodes, removed ids, a moved reopen position, setup, meta). The calls that change an adventure
  return the same object, and clients patch what they hold with `applyChange`; nobody refetches the whole tree.
- Conflicts are checked **per node**: `adventure:edit` and `adventure:deleteBranch` carry the node's `rev` and are
  refused with "This passage was changed somewhere else. Reload it and try again." when it moved on. Every change to
  a node (an edit, new choices, its state carried down from above) raises that node's `rev`. `setCurrent` raises no
  node rev; `updateSetup` is last-write-wins.
- **Positions:** each window or phone keeps its own position while it has an adventure open; `currentId` in the file
  is only where to reopen. A job started by the phone does not move the desktop's reading column (the desktop shows
  a small "New passage on another device" note it can click instead), and the other way round.
- **State is carried down.** A node keeps `changes` and `state` (= parent state + changes). When a node's state
  changes (hand edit: stored as `diffState(parent.state, newState)`; New choices: the model's new changes), every
  node below it is recomputed by applying its own `changes` again, parents first (`descendantsOf` order), and each
  recomputed node's `rev` goes up. New choices replaces a hand-edited state too: the UI says so before it runs when
  the node is marked `edited`.
- Deleting sends the file to the Recycle Bin (`shell.trashItem`, injected so tests run without Electron).
- A file that does not parse is renamed with `corruptName` and left out of the list, as other stores do.
- `format` newer than `ADVENTURE_FORMAT`: the adventure is listed but `get` refuses it with a message saying a newer
  version of the app wrote it.

## A turn

A **play** job for parent P with input I:

1. **Preparing.** Read the adventure. Find the model: `setup.modelId`, else the ready instance
   (`models.ready(null)`); if `setup.modelId` is set and not loaded, `models.ensure(modelId)` loads it as Writing
   runs do. Refuse with a plain message when no model is loaded and none is set.
2. **Writing.** Stream the passage: prose, no thinking, no JSON schema. The request is fitted to the context (below).
   Reply budget: about `PASSAGE_WORDS[length] * 2` tokens, at least 256. Clean the text the same way Writing cleans
   prose (`cleanProse` / `finishText` in `src/shared/novelRun.ts` or its own small cleaner: strip a leading
   "Passage:" label, a trailing list of choices if the model wrote one anyway, and surrounding quotes or fences).
3. **Choosing.** A second, short request with a JSON schema (`samplingFor` with `schema`, thinking allowed as for
   Writing's structured actions) asks for the choices, the state changes and whether the story ended:

   ```json
   { "choices": ["…", "…"], "ended": false,
     "changes": { "addItems": [], "removeItems": [], "setStats": {}, "removeStats": [], "addFlags": [], "removeFlags": [] } }
   ```

   The schema's `setStats` is a list of `{ name, value }` (value: number or string), as in `StateChanges`.
   `choices` has exactly `setup.choiceCount` items unless `ended` is true (then none). Choices are short (under
   about 15 words), start with a verb, and differ from each other. The parser is forgiving: it accepts the object
   inside a fence or with text around it, trims and de-duplicates choices, drops empties, and fills missing
   `changes` fields with empty ones. Each choice gets a fresh id (`crypto.randomUUID()`).

The order of saving (so a stop or failure never loses a written passage):

1. When the passage has finished streaming, save the new node at once (no choices, `ended` false,
   `changes` = `noChanges()`, state = parent's state or `emptyState()` for the root), set `currentId` to it, set the
   job's `resultNodeId`, go to status `choosing`.
2. When the choosing reply is read: set the node's `choices`, `ended`, `changes`, recompute its state and carry it
   down (the new node has no children yet), raise its rev, status `done`.
3. A stop or error while choosing leaves the saved node as it is (no choices) with status `stopped` / `error`; the
   reader can use **New choices** or type an action. An unreadable reply: same, status `done` with a warning.
4. A stop or error while preparing, waiting or writing saves nothing.

**retryPassage** for node N: the same as play with N's parent and N's input; the new node is a sibling of N and
becomes current. N stays in the tree. The opening cannot be written again this way (an adventure has one root):
the UI hides **Try again** there and main refuses it. **retryChoices** for node N: only the choosing step, with N's passage; N's
`choices`, `ended` and `changes` are replaced in place, its state recomputed from its parent's and carried down
the branch; `currentId` stays.

**Playing on from an ending:** an ended node offers no choices, but the reader may type an action; the model then
continues past the ending.

**One model job at a time, on the whole computer.** Today `chatBusyReason` (`src/main/writing/adapters.ts`) only
reads `chat.busy`, which is false while a job is preparing (loading the model, counting tokens) and in the gap
between the passage and the choices request, and `WritingRuns` does not know about adventure jobs. So add one
**model gate** in main (`src/main/modelGate.ts`): `take(holder: string): (() => void) | string` returns a release
function, or the sentence saying who holds the model. AdventureRuns takes it for the whole job (queued to end) and
WritingRuns does the same for its phone jobs (a few lines in `runs.ts`: take it where the job starts, release it
where the job ends, refuse with the returned sentence). The busy reason both services check is "the gate is held,
or `chat.busy`, or a benchmark runs". The desktop's own chat and Writing pages are not changed in this version
(known limit: they can still start while a phone job runs, as today).

Jobs are kept in memory per adventure (at most 20 finished ones each) until dismissed, so a phone that slept gets
the result back with `adventure:jobs`. Events are throttled like Writing runs (`throttleMs`, default 250).

## Prompts

System message, built from the setup:

- The role: a narrator of an interactive story written in second person ("you"), present tense, in the given
  genre and tone; passages of about N words; end each passage at a moment where the reader has to decide;
  never decide for the reader, never list the choices in the passage, never speak to the reader outside the story.
- Premise, the reader's character, and, when present, the world text from the Writing project
  ("Use these characters and places; stay consistent with them").
The system message stays the same for the whole adventure (no state in it), so the model's prompt cache keeps the
start of the prompt from one turn to the next.

Messages: the story so far as alternating turns. Each earlier passage is an assistant message; the reader's step
after it is a user message: "The reader chooses: <choice text>" or "The reader does: <typed text>". The **last**
user message also carries the current tracked state (inventory, stats, flags of the parent node) with the rule:
"The story must stay consistent with this. The character only has these items." and asks for the next passage. For
the opening, the last user message is "Begin the story." Sampling: `DEFAULT_SAMPLING` (`src/shared/defaults.ts`)
with `setup.temperature` over it when set; generation through `startGeneration` with `prose: true` for the passage
and `prose: false` plus `schema` for choosing (pass an existing `NovelAction` only where the type needs one, e.g.
`'continue'` / `'checkScene'`, and set `prose` explicitly so it decides).

The choosing request uses the same system message and story, plus the new passage as the last assistant message,
and a final user message that asks for the JSON object, with the number of choices and the state rules (only
change what the passage actually changed; keep stats as numbers where they are numbers; at the opening, set a few
starting stats and items that fit the story).

**Content.** The app runs the user's own local model for the user's own stories. The prompts do not add content
rules of their own beyond the user's chosen tone.

## Fitting the context

Use the instance's context length (`WritingModel.contextLength`; fall back to 8192 when unknown) minus the reply
budget minus a margin. Count tokens through the chat service's `countTokens` (as Writing does with
`templateOverheadWith` and the counter helpers in `src/shared/novelRun.ts`); when counting fails, estimate 4
characters per token. Always keep the system message and the newest passages; drop the oldest passages (but keep
the opening when it fits) **in blocks of 8 passages** (so the cut point, and with it the cached prompt, changes
only every few turns) and put one line in their place: "(Earlier passages left out.)". Add a warning to the job
when anything was left out. The tracked state carries the facts that matter across a long story.

## Starting from a Writing project

`adventure:create(setup, worldFolder)`: main checks that `worldFolder` is one of the folders
`novel:listProjects` returns, opens it with the novel store for a moment (owner `adventure:<id>`), reads the
project's title, its non-empty chapter summaries and its story-bible entries (characters, places, other kinds, name
plus text; a Writing project has no separate premise field), closes it, and copies a plain-text digest into
`setup.world` (cap it at about 12,000 characters, whole entries only, characters before places before the rest). The phone may pass a `worldFolder` too; the same check applies, so it
does not need to have opened the project.

## Markdown export

`adventure:exportMarkdown(id, nodeId)`: `# <title>`, a line with genre and, if any, the Writing project it came
from, then each passage on the path in order. Before every passage but the first, the reader's step in italics
("*You chose: …*" / "*You: …*"). At the end, "**The end.**" when the last node ended the story. The desktop's
`adventure:saveExport` shows a save dialog (default name: the title with characters Windows refuses removed, `.md`)
and writes the file with `writeTextAtomic`.

## Desktop page

A page in the nav rail after Writing, labelled **Adventure**, icon `Signpost` (Lucide).

- **No adventure open:** a list of adventures (title, genre, passages, last played, "Ended" badge) with **Open**,
  **Rename** and **Delete** (confirm; "goes to the Recycle Bin"), and **New adventure** as the one primary button.
- **New adventure:** a dialog with two tabs at the top: **Your own idea** (default) and **From a Writing project**.
  Your own idea: title, genre, premise, your character, tone, passage length (Short / Medium / Long), choices
  (2–4), model (Loaded model, or one from the library), temperature (Default or a number), and a row of starter
  ideas from `adventure:starters` that fill the form. From a Writing project: a project picker (from
  `novel:listProjects`) plus the same fields except premise is optional (the project's premise is used) and the
  world text is shown as a read-only preview once created. **Start** creates the adventure and plays the opening.
- **Playing:** the story path as a reading column (passages, each followed by the step the reader took in a muted
  line), the newest passage streaming in while it is written. Under the current passage: the choices as buttons
  (one per row, numbered; keys 1–4 pick them), then a text box "Or type what you do…" with **Do it** (Enter). When
  the story ended: "The end" and **Go back** hints.
- Each passage has a small toolbar on hover: **Go back here** (makes it current; its choices show again), **Other
  paths** (n) when it has siblings (previous / next sibling arrows switch to that path, the reading column follows
  the newest leaf of that path only when the reader clicks), **Try again** (retryPassage, current passage only),
  **New choices** (retryChoices, current passage only), **Edit** (passage text, choices and state in a dialog,
  saved with rev), **Delete branch** (confirm; not on the root).
- A choice that was already taken from this passage (`childForChoice` by choice id) shows a small "visited" mark
  and goes to that existing child instead of writing a new one; holding **Shift** (or a "Write again" item in its
  menu) writes a new one.
- A node saved with no choices and not ended (choosing stopped or failed) shows "No choices yet" with **New
  choices** and the typed-action box.
- **Delete branch** and deleting a whole adventure are desktop-only.
- **Side panel** (collapsible, right): **State** (inventory, stats, flags of the current node; edit opens the same
  edit dialog), **Map** (the tree as an indented outline: each node shows its step text shortened, the current path
  highlighted, click to go there), **Settings** (title, genre, tone, length, choices, model, temperature through
  `adventure:updateSetup`).
- Top bar of the page: back to the list, the title, **Export** (Markdown, through `adventure:saveExport`), and a
  **Stop** button while a job runs.
- Errors and refusals show as the app's usual toasts / banners, in plain words.

## Phone screens (RulZ Writer for Android)

The phone app gets a second top-level section next to Projects: **Adventures**. It works only while connected.

- Adventures list (same rows as the desktop), **New adventure** (one screen form with the same fields and
  starters; a "From a Writing project" switch shows the project picker), a menu for Rename. Deleting is
  desktop-only (the phone may not call `adventure:delete` or `adventure:deleteBranch`).
- Play screen: the reading column, choices as full-width buttons, the "Or type what you do…" box, **Stop** while
  writing, prompt-reading progress while waiting. A bottom sheet with **State**, **Map** (the indented tree) and
  **Settings**. Passage menu: Go back here, Other paths, Try again, New choices, Edit, as on the desktop.
  **Export** shares the Markdown with `navigator.share` where the WebView has it, else copies it to the clipboard
  (no new Capacitor plugin).
- A computer with an older LLM Suite refuses `adventure:*` ("That is not available from the phone."): the
  Adventures screen then says "Update RulZ LLM Suite on the computer to play adventures from the phone."
- After a reconnect, the open adventure is fetched again and `adventure:jobs` picks up a job that finished while
  away.

## Not in this version

Pictures for scenes, sound, more than one reader, dice rolls or rules engines, recording adventures as training
data, and a summary of dropped passages.
