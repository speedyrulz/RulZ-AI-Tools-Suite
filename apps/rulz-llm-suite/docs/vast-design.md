# vast.ai models: design

LLM Suite 0.29.0. A model can run on a rented vast.ai instance while everything else stays on this PC: chats, REA and
its targets, recordings, practice grading, the scoreboard. The app finds, rents, sets up, stops and destroys instances.

Decided by the owner: the model only runs remotely (not REA, not grading); the app may rent; the model file comes from
Hugging Face or is uploaded from this PC; practice runs several tasks at once and can spread over several instances;
the instance can stop itself; the remote model works everywhere a model is used.

## 1. Pieces

| File | What it does |
| --- | --- |
| `src/shared/remote.ts` | Types, pure helpers, IPC contract |
| `src/main/remote/vastApi.ts` | vast.ai REST client |
| `src/main/remote/ssh.ts` | ssh command lines, remote scripts over stdin, uploads, the tunnel |
| `src/main/remote/scripts.ts` | The bash scripts sent to the instance |
| `src/main/remote/rentals.ts` | Rental list, setup jobs, cost, auto-stop, persistence (`remote.json`) |
| `src/main/remote/models.ts` | Remote model definitions (`remote-models.json`) as `ModelEntry` |
| `src/main/remote/secrets.ts` | vast.ai key and Hugging Face token (Electron safeStorage, env, CLI files) |
| `src/main/remote/remoteServer.ts` | Starts llama-server on a rental and looks like a child process to `InstanceManager` |
| `src/main/engine/instances.ts` | Gains remote hosts; everything local stays as it is |
| `src/main/mcp/manager.ts` | Gains scopes: a private REA process per parallel session |
| `src/main/re/session.ts` | One unattended RE session in the main process (the chat store's loop, without a window) |
| `src/main/re/pool.ts` | Runs a batch over lanes (one lane = one slot of one loaded instance) |
| `src/renderer/src/pages/VastPage.tsx`, `stores/remote.ts`, `components/remote/*` | The vast.ai page, the cost chip |

## 2. Money rules (none may be broken)

1. Only four channels start, stop or end billing: `remote:rent`, `remote:start`, `remote:stop`, `remote:destroy`. Each
   follows a click in this window with the price or the loss shown in a dialog. Loading a model, an API request, a phone,
   a batch run, a resume or app start never rents and never starts a stopped instance; they fail with a sentence.
   None of the remote channels is registered for the phone or the API server. The API server and the phone may only use
   a remote model that is already loaded; they never load, unload or reload one.
2. Rent has no idempotency key on vast's side. The app writes an intent (a unique label `rulz-llm-<8 hex>`, offer id,
   price) to `remote.json` before the PUT. If the answer is lost or ambiguous (timeout, 5xx, unreadable), it never
   retries: it lists instances and looks for the label, and shows "may have been rented" until that is settled. An
   intent left by a crash is settled the same way at start. The search pages through every instance (25 per page). "Not
   found" is not "not rented": the intent stays open for 10 minutes of checks, and no other rent is allowed meanwhile.
3. The price shown is the price rented: if a fresh search of that offer id shows a higher `dph_total`, the rent is
   refused and the new price shown. `cancel_unavail: true` so an offer that cannot start costs nothing. On-demand only.
   The search passes the chosen disk size (`allocated_storage`), the dialog shows the traffic prices, and the contract's
   real `dph_total` is read back after renting; more than 2% above the shown price is reported loudly.
4. Every rental made by the app has auto-stop on (default 30 idle minutes; switching it off warns) and a hard cap
   (default 12 hours from its start; **Extend** adds hours). The hard cap stops it whatever is running. An attached
   instance (rented elsewhere, maybe training something) has both off until switched on.
5. The limits are enforced on the instance by a watchdog, so a crashed app or PC does not keep paying.
   - `keepalive` is refreshed every 60 s over the open tunnel connection (a loop in the tunnel's remote command that
     reads a line from stdin and touches the file), only while the rental is in use by this window: a chat, Writing or
     Adventure request from the window, a batch lane, a setup or transfer. Requests from the API server or the phone
     count only when **API and phone requests count as use** is on for that rental (off by default).
   - Idle stop: `keepalive` older than the limit, AND every GPU at or under 10% in 10 samples over 5 minutes, AND no
     compute process that is not our llama-server (so a training job between steps is never stopped).
   - Hard cap: `cap` (an epoch second in a file) passed. For an attached instance the cap also refuses while a foreign
     compute process exists.
   - It stops with CONTAINER_ID and CONTAINER_API_KEY, read from its environment, `/etc/environment` and
     `/proc/1/environ`. Arming checks with a read of its own instance that the key works; otherwise the window shows
     "Cannot stop itself" and why, and only the app's own timer protects it.
   - App rentals install it from `onstart` (so it survives stop/start and a restart from vast's site) and it touches
     `keepalive` at boot; the app re-arms it on every attach and start.
   - The app has the same timers and stops through the account API, with a visible 60 s countdown and **Keep
     running**. A batch run can ask to stop its rentals when it ends (same countdown).
6. Auto-stop only stops (the disk stays and is billed, which the dialog says). Destroy is never automatic, and is
   offered only for rentals the app made.
7. Stop and destroy re-read the instance first and act on that id only; destroy names the label and says what is lost.
   While a rental is in use, both ask again and then cancel the use cleanly.
8. A chip in the top bar shows $/hour and the running total of every running rental on every page, and what
   stopped rentals cost for their disk. Quitting with a
   rental running asks: stop and quit, leave running (with when it stops itself, or that it will not), or cancel.
   Windows logoff does not ask.

## 3. Secrets

vast key: saved here (safeStorage) > `VAST_API_KEY` > the CLI's key file. HF token: saved here > `HF_TOKEN` > HF's
token file. The window only learns where a key came from. A key file that needs a passphrase works only
through the ssh agent; the app says so. Neither leaves this PC: files are asked for without the token first, and for a
gated file this PC gets the short-lived signed address from Hugging Face and sends only that, on ssh stdin (a fresh one
for each try, because it expires). Log lines shown in the app are scrubbed of both values.

Whoever owns the rented machine can read what is sent to the model and can make it answer anything, tool calls
included. So with a remote model the approval settings (don't ask, always allowed, static analysis unasked) cover only
static analysis and opening a program, and only on the RE session's own files (`runsWithoutAsking`, `argsStayInside`).
Everything else asks, and an unattended run refuses it. Chats share one REA process, so a chat with a remote model must
have opened the program itself before it may look into it unasked, and reading REA's stored evidence always asks. Known limit: paths are compared as text, so a link
the user already has inside the analysed folder is followed. "Allow this tool in this chat" lasts until the app closes.

## 4. ssh

Windows OpenSSH (`ssh` on PATH). One known_hosts file per rental id, `StrictHostKeyChecking=accept-new`,
`BatchMode=yes`, `ServerAliveInterval=15`, `ServerAliveCountMax=3`. Routes: direct port, then vast's proxy. The user's
own `~/.ssh` key; before renting, if no local public key is on the account, the app offers to add it (a click).
Scripts run as `bash -s` with the script on stdin. All interpolated values go through one `shq()` quoting function;
paths on the instance are built from validated slugs.

Remote folder: `/workspace/rulz-llm` (or `/root/rulz-llm` if there is no `/workspace`):
`engine/<tag>/`, `models/<slug>/`, `run/<instanceId>/{pid,port,args,log}`, `keepalive`, `watchdog.*`.

## 5. Engine on the instance

The prebuilt llama.cpp Linux CUDA 12.8 package of the same build as this PC's llama-server (`b<build>` from
`--version`), else a pinned default (`b12407`), or a tag the user sets. Both tarballs (`llama-<tag>-bin-ubuntu-cuda-12.8-x64.tar.gz`
and `cudart-...`) come from GitHub with the sha256 from the release API's `digest`, checked on the instance. x64 only
(offer filter `cpu_arch = amd64`, `cuda_max_good >= 12.8`). Then `llama-server --version` and `--list-devices` must work.

## 6. Remote models

A remote model is a saved definition with id `vast/<slug>`: name, source (`local`: a model in the library, with its
parts, projector, draft and adapters; or `hf`: repo or bucket, file, optional mmproj file, revision), and a load config
kept like any other model's defaults. It appears as a `ModelEntry` wherever models are listed, marked `remote`. Setting
up a rental for a model downloads or uploads its files there (resumable; size checked, sha256 checked for uploads and
for HF files that report one), then the model can be loaded there. Files are named by slug, so a later upload of a
changed local file is noticed by size and sha256.

## 7. Instances

`InstanceManager` talks to a `ServerHandle` (`src/main/engine/serverHandle.ts`: output, exit, async stop, sync detach),
with a local implementation (child process) and a remote one (`remoteServer.ts`). Pid file, orphan cleanup and tree
kill belong to the local one. `LoadedInstance.host` says where it runs; absent = this PC. The load key is
(model, host): one remote model can be loaded on several rentals at once, and `load` takes the host. `ensure` finds any
instance of the model. Ejecting the previous model concerns the same host only. The load benchmark is local only.

`readyFor(modelId)` returns the ready instance with the fewest users, preferring the one a caller used last
(`ChatRequest.instanceId` pins one; a chat's turn stays on the instance it started on, so its prompt cache is kept).

Arguments for a rental come from `buildRemoteArgs`: paths on the instance, no local GPU indexes, no thread count, no
memory lock, no local template file (refused with a sentence), `--host 127.0.0.1`. The memory estimate uses the rental's
GPUs.

llama-server runs detached on the instance (`setsid nohup`) with a run folder; a local `ssh -N -L` tunnel on a port the
app keeps carries the traffic and is respawned when it drops (requests in flight fail with a transport error, the model
stays loaded). Unload kills the remote process by pid after checking `/proc/<pid>/exe` is this engine, waits for it to
be gone, then reports the exit; it does not report an exit that has not happened. The server exits by itself when
`keepalive` is older than 6 hours, so an orphan never holds VRAM on someone's training instance. A load reuses a
running server only when the hash of (args, engine tag, model file sizes and hashes) is the same. A tunnel that cannot
come back in 2 minutes, or a process that is gone, is an unexpected exit with the remote log tail. Closing the app
detaches: the server stays for the next start (the quit dialog says so).

Preflight before a setup (free disk against the files) and before a load (free VRAM against the estimate, and foreign
compute processes): refused with a sentence on an attached instance, a warning on an app rental.

## 8. Practice batches

The window's batch loop is replaced by a pool in main, for every run (one local lane behaves as today).

- A run has lanes: `{ instanceId, slot }`. With one session on this PC the model is used as it is loaded (or loaded with
  its defaults), as today. Otherwise it is loaded with `parallel = N` and `contextLength = contextPerSession * N`.
- Each lane takes the next pending item, prepares the task, runs a session, grades it, saves it, takes the next. Lanes
  start 3 s apart, because the first Ghidra analysis is heavy on this PC's CPU.
- REA keeps its open target per process, so each lane gets its own REA process (an MCP scope), reused for the lane's
  tasks; other servers are shared. Measured 2026-11-23: three REA processes work side by side, about 1.2 GB RAM each,
  the first analysis took 55 s with three at once. Scoped calls get 300 s instead of 120 s. At most 8 lanes per host
  and 16 in all; default 4. More than one lane needs `rea` to be a stdio server; saving or restarting the MCP config is
  refused while a run has scopes. A scope whose process died is reopened for the next task.
- Session rules are the chat store's: same prompt, sampling, starter tools, on-demand loading, budget, wire building,
  compacting, context trimming, recording, round limit, unattended denial, time limit from the first reply, grading.
  The pure parts move to `src/shared` and both use them.
- While a session runs, its chat is owned by the pool. `Conversation.poolRev` goes up with every pool save;
  `conversations:save` from the window is refused when the chat is owned, or when its `poolRev` is not the newest (a
  stale copy from before the pool's last save), for as long as the app runs. The window then reloads the chat.
  `re:watch` streams the chat on screen; the chat page is read-only with **Skip this task**.
- A transport failure (tunnel down, connection reset, instance gone) is not a model failure: the round is retried for
  up to 3 minutes while the instance is ready or reconnecting; then the item goes back to pending and the lane ends if
  its instance is gone. An item returned twice becomes `error`. The run goes on with the other lanes and stops
  (resumable) when none is left.
- Stop after / stop now / resume work as before. A run records its lanes, hosts and estimated cost.
- `re:batches` does not settle a run the pool is still running (a reloaded window).
- A run counts as using its rentals. It refuses to start while the load or writing benchmark runs, and they while it
  runs.

## 9. Not in scope

REA or grading on the instance, interruptible rentals, automatic rent or destroy, renting from the phone or the API,
multi-node inference, sharing a rental with other users.
