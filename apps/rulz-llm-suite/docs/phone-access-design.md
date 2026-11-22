# Phone access for RulZ Writer for Android: design

Status: built for LLM Suite 0.26.0 and RulZ Writer for Android 0.1.0 (2026-11-22). The contract is
`src/shared/phone.ts`; the phone app (`apps/rulz-writer-android`) copies it verbatim.

## Goal

Write in LLM Suite's Writing mode from an Android phone. The computer does the work: it reads and saves the project
files with the Writing mode's data safety rules (docs/writing-mode-design.md) and runs the model. The phone is the
screen. A save made on the phone shows in the desktop's Writing page and the other way round.

Phone v1 covers: projects and the outline, reading and editing scenes (safe saves, conflict warnings), AI actions with
accept/reject and alternatives, summaries and outline proposals, the story bible, tracked state with Update state
proposals, the consistency check, the character interview and voice notes. Desktop only: the writing benchmark,
recording and export of training data, deleting projects, emptying the trash.

## Transport

Copied from RulZ Coding Agent (`apps/rulz-coding-agent/src/main/remote/`) into `src/main/phone/`:
WebSocket on the Tailscale address only (100.64.0.0/10), path `/rulz-writer`, port `phonePort` (default 7421; Coding
Agent uses 7420), pairing token in a QR link `rulzwriter://pair?host=&port=&token=&name=`, origin check (the
Capacitor WebView at http(s)://localhost, or no Origin), wrong-code lockout, ping, buffered-amount limit, hello
timeout and pending-connection limits as in Coding Agent.

Differences from Coding Agent:

- **Allowlist, not a deny list.** Only `PHONE_INVOKE` channels are answered; only `PHONE_EVENTS` are sent. `chat:*`
  is never exposed (a `ChatRequest` can carry file tools and recording).
- `novel:openProject` from the phone is limited to folders `novel:listProjects` returns (a project inside the
  projects root). `novel:createProject` makes one there.
- `novel:saveFile` from the phone must carry a `baseHash` and may not use `force` (or the request's equivalent
  overwrite flag); a conflict comes back as a conflict, as for the desktop.
- Settings: `phoneEnabled` (default false), `phonePort` (7421), `phoneToken` (made on first enable, 32 random bytes,
  base64url). The token is never sent to the renderer except through `phone:pairing`, and never logged.

## Main-side fixes both sides need

1. **Saves through the store are announced.** `NovelStore.saveFile` records the new hash itself, so the watcher event
   that follows is ignored (`known === hash`) and nobody else hears about it. After every successful save, outline /
   project / state patch, entry create, delete and restore, the store emits `novel:changed` (`text` with the new
   hash, `deleted`, or `state`) to every listener: the window and the phone. The renderer already ignores a change
   whose hash equals its buffer's base hash, so its own saves are harmless.
2. **Open projects are reference-counted by owner.** Owners are the window (`'window'`) and each phone connection
   (`'phone:<clientId>'`). `openProject` adds the owner, `closeProject` removes it, and the watcher stops and the
   project is dropped only when no owner is left. A phone connection that closes releases all its projects.
3. **One journal per file belongs to the desktop.** The phone never calls `novel:journal`/`novel:discardJournal`; it
   keeps an unsaved draft on the phone itself (Capacitor Preferences) and offers it back after a restart.

## Writing runs

`src/main/writing/`: a `WritingRuns` service that runs the AI actions the desktop renderer runs in
`src/renderer/src/stores/novel.ts`, as jobs in main, using the same pure helpers:

- `src/renderer/src/lib/novelRun.ts` moves to `src/shared/novelRun.ts` with its IPC use injected (counting and
  generation take functions instead of importing `./api`). The renderer's `lib/novelRun.ts` re-exports it so the
  desktop's behaviour and tests do not change.
- Counting and generation call main's chat service directly (no IPC round trip), with an event tap by request id.
- A job reads the project state, style, bible and facts from the store (the phone sends only its buffer text, base
  hash, cursor/selection, note and options), builds and fits the request exactly as the desktop does, and keeps the
  result in memory per folder until `writing:dismiss`. Results are shaped as `WritingResult`.
- One run at a time from phones; refused while the desktop window is generating on the same model (say so in the
  error). The project's model is used; if it is not loaded, the job loads it as the desktop does (or fails with a
  clear message if loading needs the desktop).
- `writing:job` events go to the phone on every status change and, throttled to about 4 per second, while text
  streams. A phone that slept calls `writing:jobs(folder)` on reconnect and gets everything back.
- Accepting a prose result happens on the phone: it applies the chosen alternative to its buffer and saves with
  `novel:saveFile` and the job's base hash, then `novel:outlinePatch` for `newSticky`. A conflict is shown as "The
  text changed since this was written." Summaries and outline proposals are accepted with `novel:outlinePatch`; check
  findings with `novel:dismissFinding`; state items with `writing:applyState`, which re-checks the scene's disk text
  against the job's `textHash`.
- Phone runs are not recorded (stage 4 recording stays a desktop feature in v1).

## Limits stated to the user

- Proposals made on the phone show only on the phone; the desktop sees the accepted text.
- If the desktop has unsaved edits in a scene and the phone saves that scene, the desktop shows its usual
  changed-on-disk conflict.
- LLM Suite must be running on the computer (the window may be in the tray).
