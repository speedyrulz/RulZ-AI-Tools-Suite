# Remote render: the contract

Feature: while a RulZ AI Trainer run trains on a vast.ai instance (Linux), the user's Windows PC renders that run's
samples and likeness probes on a local GPU with the trainer's own render code. Probe pictures go back to the
instance, which scores them and acts (best saves, stop, lr_ratio, face_bias, Keel on_probe) up to one probe interval
late, the same contract as the existing `probe_async` (rulztrainer/probeworker.py). Training on the instance must
NEVER stall on or lose a round because of the PC; best saves must write exactly the weights that were scored.

App root: `apps/RulzTrainer/` (paths below relative to it). Protocol version constant: `PROTOCOL = 1`.

## 1. Shared module `rulztrainer/remoterender.py` (the instance's round bookkeeping; the PC imports its helpers)
It holds these shared pieces (used by both sides, so their names and behaviour are fixed here):
- `PROTOCOL = 1`
- `RENDER_SOURCES`: a list of package-relative source files that determine render output (at least
  `probeworker.py`, `remoterender.py`, `lora.py`, `quant.py`, `attention.py`, the sampling functions' modules, and the
  family's `archs/<family>/` folder files, determined at call time from the family). `render_hash(family) -> str`:
  sha256 over those files' bytes with `\r\n` normalised to `\n` (Windows checkouts are CRLF), first 16 hex chars.
- `recipe_hash(cfg_dict, files) -> str`: sha256 over the render-relevant cfg fields (sampling model, quant, turbo /
  sample LoRA identity and strengths, lora shape: rank, alpha, targets, blocks, network_type, factor, oft, dora,
  attention backend, int8/nvfp4 matmul flags, sample/probe sizes, steps, cfg, shift) plus each file's (name, size).
- `job_to_dict(RenderJob, job_dir) -> dict` / `job_from_dict(d, base_dir) -> RenderJob`: out_path stored as a path
  relative to job_dir with forward slashes; on the PC it is rebased under the helper's own work folder.
- Path layout helpers (below) and `atomic_rename(src, dst) -> bool` (False when src is gone / dst exists).

## 2. Folder layout on the instance: `<job_dir>/remote_render/`
- `run.json` — written by the trainer at run start after clearing the whole `remote_render/` folder (a resume or a
  crashed run leaves nothing a helper could render into a dead run): `{protocol, run_token (uuid4 hex), job,
  family, render_hash, recipe_hash, created}`.
- `prepare/prepare.json` — written once at run start: `{protocol, run_token, recipe_hash, render_hash, family,
  cfg (resolved TrainConfig.to_dict()), files: {cfg_field: {path, name, size, repo, file}} (repo/file = the
  Comfy-Org/HF provenance from comfy_files.planned_downloads when the file was downloaded, else null),
  negatives: {"sample": text, "probe": text, ...}, embeds: {prompt_text: "embeds/<sha1>.safetensors"}}`.
  `prepare/embeds/*.safetensors`: copies of the text-embed cache file of every sample prompt, probe prompt and
  negative (`text_cache_path(cfg, prompt)`), so the PC needs no text encoder. Sent once per run, not per round.
- `prepare/ready.json` (written by the PC via the bridge) — `{run_token, recipe_hash, render_hash, host, gpu,
  vram_gb, load_seconds, kinds: ["sample","probe"] or ["sample"], time}`; or `prepare/not_ready.json` —
  `{run_token, reason, time}`. The trainer delegates nothing until ready.json matches its run_token, recipe_hash and
  render_hash. A ready.json that matches wins over a stale not_ready.json left by an earlier attempt (the helper
  retries a failed preparation, and the trainer ignores not_ready.json once a matching ready.json exists). `kinds` is ["sample"] when the user set a local sampling-model override (probes must render the
  run's own recipe on both sides, so an override applies to samples only).
- `helper.json` — heartbeat, rewritten only when a helper frame arrives (not by a bridge timer, so a sleeping PC
  goes stale): `{run_token, host, gpu, time}`. Fresh = `time` within 30 s of the instance clock (the bridge writes
  the instance's own `time.time()`, never the PC's clock).
- Rounds: the trainer writes `rounds/.tmp-<seq>/` then renames it to `rounds/<seq:06d>_<kind>_<step>/`. Contents:
  `round.json` `{protocol, run_token, seq, kind ("probe"|"sample"), step, negative_kind, jobs: [job dicts],
  deadline_s}` and `lora.safetensors` (the exact state rendered: for probes the EMA-averaged
  `lora_mod.lora_state(injected)` taken inside `_averaged()`, same as `_probe_async_start`). Despite its name the file holds `torch.save` bytes of that state dict, not the
  safetensors format (it is the payload `Renderer.render` takes; read it with `remoterender.state_from_bytes`). The
  name is kept for the contract's sake.
- State moves are atomic renames between sibling folders (same filesystem), and **whichever rename succeeds wins**:
  - PC claims: `rounds/X` -> `claimed/X` (fails = trainer took it back: skip).
  - PC finishes: uploads images into `claimed/X/out/` (out names = the jobs' out_path relative names; a clip's
    extra files keep their names), writes `claimed/X/done.json` `{seconds, files:[relative names], host, gpu}`,
    then renames `claimed/X` -> `done/X`.
  - PC fails: writes `claimed/X/error.json` `{error, traceback}` then renames `claimed/X` -> `failed/X`.
  - Trainer takes back (deadline or stale heartbeat): renames `rounds/X` or `claimed/X` -> `abandoned/X`, then
    renders that round itself. If that rename fails because the PC's `done/X` rename won, take `done/X` instead.
  - Trainer takes a finished round: reads `done/X`, moves `out/*` to the jobs' real out_paths itself (clip files
    before the still, as `_render_round` does around trainer.py 3484), then deletes `done/X`. The bridge never
    writes to out_paths.

## 3. Trainer behaviour (instance)
- Config: `remote_render: auto|off` (default auto; without a helper it does nothing). Same exclusions as probe_async:
  race, fine-tune, multi_gpu != off. ProbeProcess (probe_async) and remote are exclusive for probes at any moment:
  one pending probe slot across backends (controller.update needs step order).
- Step 0 always renders on the instance (deterministic floors, body strangers; gives the per-kind render time used
  for deadlines).
- Delegation is allowed only while ready.json matches and helper.json is fresh. Deadline per round:
  `max(90, 3 x instance's own measured render seconds per picture of that kind x the round's picture count)`, so a
  round with more pictures gets proportionally longer; first round after ready gets 2x that.
  Probe rounds (delegated and shadow) get until about the next probe instead (3.39.1,
  `RemoteRenderer.probe_deadline`): the larger of that rule (doubling included) and 1.5 x the wall seconds between the
  two most recent probe steps (`note_probe_step`, called at every probe step from step 0). At the next probe step a
  delegated probe still out is not waited for: it is taken back at once and rendered on the instance from the same
  weights, a counted miss. The end-of-training take waits at most the old rule's seconds. Sample rounds keep the old
  rule. (A PC GPU ~4x slower than the instance behind a slow link needed ~4 min for a round the old rule gave 117 s.)
  2 consecutive deadline misses or errors -> stop delegating for the rest of the run (log it).
- A kind is delegated only after the instance has timed one render of it (`note_own`): until then there is no
  deadline to give, so that round renders on the instance. In practice step 0 supplies the timing.
- Remote render is off on a Windows-hosted trainer (`supported_host()` is false on win32): such a run is the user's
  own PC, with no helper to find.
- Shadow round: the first probe round after the helper becomes ready is rendered on BOTH: the helper's copy is
  taken when in, the instance renders its own inline (training pauses as today); the instance's scores drive the
  decisions; the helper's pictures go to `remote_render/shadow/`, not the real out_paths. Compare face scores, and if the mean difference exceeds `likeness_noise_band` (scores in percent
  points; 2 points when it is 0) log a warning and stop delegating probes (samples continue). Log both numbers either way.
- Probe fallback path (own path, not run_probe/sample_images, to avoid re-averaging): mirror `Pending.state` into
  injected, `frozen_adapters`, `_sampling_slots`, `_render_round`, restore live weights in `finally`, then
  `_probe_finish` with the mirrored scored state (as `_probe_async_take` does) and `_keel_probe(scored=p.state)`.
- Samples: non-blocking; at most 2 outstanding sample rounds — a sample due with 2 outstanding renders inline as
  today. At take, on the trainer thread, do what `_sample_with` does after rendering (score_samples, face_scores,
  `_save_history`, status samples list). Before the final samples, drain outstanding sample rounds (wait while the
  helper is fresh, up to their deadlines, else take back and render). Final probe renders on the instance as today.
- Each ProbeRecord gets `rendered_on` ("instance" or "<host> <gpu>"); log a line when the renderer changes mid-run.
- Log lines say plainly what rendered where and how long training waited.

## 4. The bridge `scripts/vast/rulz_render_bridge.py` (runs on the instance, piped over ssh, nothing installed remotely)
- Started by the PC as one long-lived ssh process. Because `python -` reads the script to EOF, start it with a
  `python3 -c` bootstrap that reads a 4-byte big-endian length then that many bytes of script from stdin and
  `exec`s it, after which stdin carries frames.
- Frame (both directions): `b"RZRB"` magic + 4-byte BE header length + header JSON (UTF-8) + 8-byte BE body
  length + body bytes. Bad magic -> exit. Exit on stdin EOF.
- Ops (header `op`, reply `{ok, ...}` or `{ok:false, error}`): `hello {protocol}` -> `{protocol, output_root,
  time}`; `runs` -> running jobs that have `remote_render/run.json` with run.json, prepare status, open round
  names; `get {path}` -> body = file bytes, or with `{offset, length}` (non-negative ints, inside the file) that byte
  range; `size {path}` -> `{size}`; `put {path}` + body; `write_json {path, data}`; `rename {src, dst}` ->
  `{ok, renamed: bool}`; `list {path}`; `heartbeat {job, run_token, host, gpu}` (writes helper.json with the
  instance clock). Every path is relative to the output root and must resolve inside
  `<output_root>/<job>/remote_render/` (reject `..`, absolute paths, symlinks out).
- Output root discovery: the same way `rulz_agent.py` finds the WebUI's output folder.

- Transfer speed (3.39.1): one ssh stream from a vast host fell to ~0.8 MB/s (84 MB in 100 s); the worker fetches a
  LoRA of 8 MiB or more in `LORA_STREAMS` (4) byte ranges over its own connection plus 3 more it keeps for the run
  (8-19 s for 84 MB, measured), falling back to one get when a range fails. The helper's ssh prefers
  `aes128-gcm@openssh.com` (`-c ^...`, the default list stays behind it).

## 5. The PC helper `rulztrainer/renderhelper.py` (runs on the user's PC)
- CLI: `python -m rulztrainer.renderhelper --instance <id> --device cuda:N [--sample-model PATH]
  [--sample-model-quant Q] [--models-dir DIR] [--status-file PATH]`; `--list-gpus` prints JSON
  `[{index, name, total_gb, free_gb}]`. Connection: reuse scripts/vast/rulz_remote.py's instance lookup and ssh
  routes (direct, then proxy); reconnect with backoff when ssh drops. The helper opens **two** ssh connections on
  purpose, each used from one thread only: one for the main loop (runs, heartbeat), one for the worker (preparing and
  rounds: the large transfers). With a single connection a 300 MB LoRA download would hold the heartbeat back until
  the trainer called the helper gone. (Windows OpenSSH has no ControlMaster, so each is its own ssh process.)
- Loop: hello; heartbeat every ~5 s from the main thread (render runs in a worker thread, so heartbeats keep going
  while it renders; if the PC sleeps both stop); find the running job with run.json; if no ready.json for its
  run_token: fetch prepare.json + embeds into a helper-private work folder (never the user's real cache), check
  protocol and `render_hash(family)` against its own (mismatch -> not_ready "versions differ: update RulzTrainer on
  the instance or this PC"), resolve each file: with repo/file provenance resolve/download locally like
  comfy_files does; without it, find a local file with the same name and size under the models dir; missing ->
  not_ready listing them. Load a `probeworker.Renderer` with a local cfg (device, local file paths, local cache dir
  where the embeds sit at `text_cache_path(local_cfg, prompt)`, sample override if given). Load failure / OOM ->
  not_ready with the reason (16 GB cards: no block swap in Renderer; do not add it). Then write ready.json.
- Rounds: claim the oldest open round it may render (probes first; only kinds in its ready.json), fetch round.json
  + lora.safetensors, render with the round's `negative_kind` (Renderer.render currently hard-codes the probe
  negative; add a negative-kind argument), upload, done; on exception -> failed.
- Status file (default `<RULZ_VAST_HOME or ~/.rulz_vast>/render_helper/<instance>.json`: the status folder honours
  `RULZ_VAST_HOME`, as `rulz_remote.py` does), rewritten on each change: `{state:
  "connecting"|"preparing"|"ready"|"rendering"|"not_ready"|"idle"|"stopped"|"error", instance, job, gpu,
  message, rounds_done, last_round, last_seconds, behind, time, pid}`.
- Stops cleanly on SIGTERM / Ctrl+Break / stdin close; the WebUI kills it with terminate().

## 6. WebUI Fleet tab (webui.py)
- Per instance card: "Render on this PC" switch, GPU select (from `--list-gpus`), optional sampling model path +
  quant ("samples only" stated), a status line from the status file (state, job, rounds done, last render seconds,
  behind, message). The switch starts/stops the helper subprocess (one per instance; refuse a GPU another helper
  uses); turning it on is the user's consent to use that GPU. Helpers stop when the WebUI exits.
- RulZ design rules: Quasar theme is already set up; sentence case, short status words, no emoji, one primary
  button per view (the switch is not a primary button), status carries a word + icon.

## 7. Tests (no GPU, no network, no ssh)
- Instance: fake renderer / fake helper writing into a tmp job dir: claim/done/failed/abandon races (late done after
  abandon is ignored; done winning over abandon is taken), deadline fallback renders from the retained state and
  a best save writes the scored weights, stale heartbeat, ready mismatch, sample queue bound and drain, shadow round
  comparison, run start clears the folder.
- PC: bridge framing and every op over a local subprocess pipe against a tmp output root, path escape rejection,
  bootstrap; helper state machine with a fake bridge and fake Renderer; render_hash CRLF/LF equality.
