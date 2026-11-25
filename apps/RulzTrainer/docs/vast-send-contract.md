# Rent and send: training on vast.ai from the PC's WebUI (contract)

RulZ AI Trainer on the user's PC finds and rents a vast.ai instance, shows credit and how long it lasts, and sends a
run there: datasets, local model files and settings go up, the run starts once everything is in place, results come
down, and the instance is stopped (default), destroyed or left running. It builds on the existing Fleet
(`scripts/vast/rulz_fleet.py`, `rulz_remote.py`, `rulz_agent.py`), which already looks at instances, queues and
dispatches runs, pushes datasets and downloads results.

Decided by the owner: after a run the default is **stop**, with destroy and leave running offered per run; model
files that exist only on this PC are asked about per file (upload straight to the instance, or upload to a Hugging
Face bucket and let the instance download it from there; a default bucket can be set, is confirmed each time, and is
asked for when none is set); on-demand and interruptible can be rented, reserved is browse-only with a link to
vast.ai; a real rental is tested by the owner, not by us (all tests use fakes; read-only API calls are fine).

App root: `apps/RulzTrainer/`. Python 3.10+, `scripts/vast/*` standard library only (PyYAML and huggingface_hub only
where a guarded import is stated). Only on the PC: nothing of this shows when `VAST_INSTANCE` is set.

## 1. Files

| File | Holds |
| --- | --- |
| `scripts/vast/rulz_market.py` (new) | vast.ai REST client, filter schema, offers, account, templates, rent / start / stop / destroy, rental records and rent intents |
| `scripts/vast/rulz_files.py` (new) | Getting one local file or folder onto an instance: resumable ssh upload, Hugging Face bucket route |
| `scripts/vast/rulz_send.py` (new) | `plan_send()` (what a config needs from this PC), `SendManager` (the job state machine, after-run action) |
| `scripts/vast/rulz_agent.py` | New commands `file-state`, `file-put`, `file-done`, `file-fetch`, `watchdog`; `files --what tree`; `snapshot` gains `watchdog` |
| `scripts/vast/rulz_remote.py` | `send(..., progress=None)` |
| `scripts/vast/rulz_fleet.py` | Named push items, push progress, the `sent` link from queue entry to run record |
| `rulztrainer/vastui.py` (new) | NiceGUI parts: `AccountWatch`, `MarketView`, `SendDialog`, send list. Takes webui.py's helpers as a `kit` object; never imports webui |
| `webui.py` | Hooks: Rent GPUs tab, credit chip, Run on control, send buttons, background loop |

State: `remote.STATE` (`~/.rulz_vast`, `RULZ_VAST_HOME`): `rentals.json` (+ `.lock`), `sends.json` (+ `.lock`),
`send.lock` (who runs the sends), `filehash.json`, `market.json` (filters, template, defaults, default bucket).
Every write is atomic (`storage.write_json_atomic`, or the Fleet's way) under its `QueueLock`.

## 2. Money rules (none may be broken)

1. Only `Market.rent`, `start`, `stop`, `destroy` and `change_bid` change anything at vast.ai that bills (plus
   `add_ssh_key`, which changes the account). `rent`, `start`, `destroy`, `change_bid`, `add_ssh_key` are called only
   from a click whose dialog shows price and effect, or for `destroy`, also from the after-run action the user picked
   in the send dialog (rule 6). A restart of the WebUI, a resumed send, the Fleet, the CLI daemon never rent or start.
2. Rent has no idempotency key. Before the PUT an intent (unique label `rulz-trainer-<8 hex>`, offer id, price, send
   id) is written and fsynced. The PUT is sent once, never retried, 429 included. A clear refusal (HTTP 4xx with a
   JSON body, or 200 with `success: false`) drops the intent. Anything else (timeout, dropped connection, 5xx,
   unreadable, no contract id) leaves it "checking": all pages of `/api/v1/instances/` are searched for the label for
   10 minutes; found -> adopted; not found -> "unresolved", shown with Check again and Dismiss. While any intent is
   open no other rent is allowed. Intents left by a crash are settled the same way on start.
3. The price shown is the price rented: right before the PUT the offer is looked up again by `ask_contract_id` with
   the same `allocated_storage`; gone -> refused; `dph_total` more than 0.5% above the shown price -> refused with the
   new price. `cancel_unavail: true`. For interruptible the bid (per machine, GPU part, as vast's `bid_price`) must be
   >= the fresh `min_bid`; the shown total is bid + storage. After renting, the contract's real `dph_total` is read
   back and more than 2% above the shown one is reported on the rental.
4. Stop and destroy re-read the instance first (`GET /instances/{id}/`), act on that id only, and check
   `success`. Destroy: only for an instance whose id is in `rentals.json` as rented by this app and whose label still
   is the recorded one, or any instance of the account when the user clicks Destroy on it and confirms (the dialog
   names id, GPU, label, and that the disk with every dataset, model and run on it is gone for good).
5. A failed setup (rented by the app for this send, never reached a started run) stops the instance, never destroys
   it, unless the user's after-run choice is "leave running". The message says so.
6. After-run action, only for instances this app rented (for others the choice is "leave running" by default, stop
   offered, destroy not offered). Executed by the SendManager when ALL hold: every send of that instance is over; a
   fresh snapshot (taken for this decision) is reachable and shows no job running, an empty queue and no LoRA test; no
   Fleet queue entry is waiting for, being sent to or uncertain on that instance; no download of it is going; its
   runs' final downloads finished. Then a 3-minute countdown shown in the WebUI with **Keep running**; every condition
   is checked again at its end. With several sends the mildest choice wins (leave > stop > destroy).
   - stop: if the final download keeps failing, the instance is still stopped (the disk keeps the files) and the
     message says the files are still there.
   - destroy: additionally requires that EVERY run of that instance on record which this app's sends started ended
     as "done", that no other run exists in its snapshot's job list, and that `files --what tree` of each run is
     fully present here (path and size, `pull_plan` empty). Anything missing, failed or unclear -> stop instead, with
     the reason. `tree` = every file below the job folder except `state_*` folders, `remote_render/`, hidden files and
     `*.part`; the dialog says resume states are not downloaded.
7. Backstop for a closed PC: with stop or destroy chosen, a watchdog runs on the instance (section 6). It only stops.
8. Every API key, HF token, signed URL, template environment value and Jupyter token stays out of logs, events,
   exceptions shown in the UI, state files and command lines. Templates' and instances' `env` / `extra_env` are used
   only to recognise RulZ Trainer templates and are dropped right after parsing. Errors from urllib are rewritten.

## 3. `rulz_market.py`

`api_key()`: `VAST_API_KEY` > `~/.config/vastai/vast_api_key` > `~/.vast_api_key`; None when missing. Base
`https://console.vast.ai`, `Authorization: Bearer`, 20 s read timeout, 429 on reads retried with 1, 2, 4, 8 s.
`MarketError(message, kind, definite)`. `Api(opener=..., key=..., sleep=...)` is the only thing that talks HTTP.

### Filters
`FILTERS`: an ordered list of `Filter(key, group, label, kind, ...)` covering every control of vast.ai's search page
(checked against the live API on 2026-11-24; all of these are accepted by `POST /api/v0/bundles/`):

| Group | Controls -> API field |
| --- | --- |
| Top row | rental type: on-demand `ask`, interruptible `bid`, reserved `reserved` (`type`); number of GPUs any / 1 / 2 / 4 / 8 / 9+ (`num_gpus`); GPU model, several (`gpu_name in`); region (Planet Earth, North America, South America, Europe, Asia, Africa, Oceania) and single countries (`geolocation in`); sort |
| Disk | container size GB (`disk_space gte`, and `allocated_storage` = the same value so prices include it) |
| Reliability | `reliability2 gte` (percent in the UI) |
| Duration | `duration gte` (days in the UI, seconds in the API) |
| Machine options | secure cloud only (`datacenter eq true`); unverified machines (off: `verified eq true`); incompatible machines (off: `cuda_max_good gte` the template image's CUDA, `image_tag eq` the image, `direct_port_count gte 4`, `cpu_arch in [amd64]`); unavailable offers (off: `rentable eq true`); static IP (`static_ip eq true`); min CUDA (`cuda_max_good gte`); driver version (`driver_version gte`); Ubuntu version (`ubuntu_version eq`) |
| Price | $/hour min-max (`dph_total`); $/TFLOPS/hour min-max (client side: dph_total / total_flops); $/TB upload and download min-max (`inet_up_cost`, `inet_down_cost`, $/GB in the API) |
| GPU | GPU count min-max; total TFLOPS min (`total_flops`); per-GPU RAM and total RAM min-max GB (`gpu_ram`, `gpu_total_ram`, MB in the API, 1000 per GB with the 2% tolerance below); GPU RAM bandwidth (`gpu_mem_bw`); PCIe bandwidth (`pcie_bw`); NVLink bandwidth (`bw_nvlink`); DLPerf (`dlperf`), all min-max |
| Machine | CPU cores (`cpu_cores_effective`), CPU RAM GB (`cpu_ram` MB), CPU GHz (`cpu_ghz`), disk bandwidth MB/s (`disk_bw`), internet up and down Mbps (`inet_up`, `inet_down`), open ports (`direct_port_count`), all min-max |
| Also (CLI fields) | PCIe generation, GPU max power, has AVX, GPU fraction, machine id, host id |

A card sold as N GB reports a little under N*1000 MB or N*1024: a minimum of N GB is sent as `N*1000*0.98`.
`external eq false` always. `limit` 1000. Sorts (the site's nine): auto `score desc`, DLPerf/$/hr
`dlperf_per_dphtotal desc`, DLPerf `dlperf desc`, TFLOPS/$/hr `flops_per_dphtotal desc`, TFLOPS `total_flops desc`,
price up / down `dph_total`, video RAM `gpu_total_ram desc`, video RAM/$/hr client side.
`default_filters()`: on-demand, 1 GPU, 200 GB, 98%, verified, 24 GB+ per GPU, 500 Mbps down, DLPerf/$/hr.
`clean_filters(raw) -> dict` forces every value into range and drops unknown keys; `search_body(filters, template)`;
`client_side(offers, filters)`; `gpu_names(api)` and the country list come from a live wide search, cached a day,
with a built-in fallback list.

`parse_offer(raw) -> Offer|None` (dataclass; `None` for unusable rows): id, machine_id, host_id, gpu_name, num_gpus,
gpu_ram_mb, gpu_total_ram_mb, dph_total, dph_base, storage_dph, min_bid, dlperf, dlperf_per_dollar, tflops,
tflops_per_dollar, reliability, inet_up/down, inet_up/down_per_gb, cpu_name, cpu_cores, cpu_cores_total, cpu_ram_mb,
cpu_ghz, disk_bw, disk_space, disk_name, pcie_bw, pci_gen, gpu_mem_bw, bw_nvlink, cuda, driver, os_version,
geolocation, verified, datacenter, static_ip, ports, duration_days, mobo, rentable, type.

### Account
`account(api) -> {"credit", "instances": [...], "burn": {"per_hour", "running", "stopped", "gpus", "disks"},
"hours_left", "empty_at"}`. credit = `users/current` `credit` (+ `balance` when positive). An instance costs
`dph_total` per hour when `actual_status` is running or it is on its way there (`intended_status` running and
actual loading / created / scheduling / connecting), else `storage_total_cost` (per hour; else `storage_cost` $/GB/month
* disk / 730). `hours_left` None when per_hour is 0. Never includes traffic; the UI says so. `lasts(credit, per_hour)`.

### Templates
`templates(api)`: the account's own (`creator_id`), only those that set RulZ Trainer up: env names RULZ_* or a
PROVISIONING_SCRIPT containing `rulz` and `provision.sh`, or an image named rulztrainer. Deduplicated by `hash_id`,
newest first; `current: True` when its provisioning script is the suite's `app/rulz-trainer` branch, older ones
listed as "older setup". Default: the newest current one, else `BUILTIN` (the payload of `create_template.sh`).
Fields kept: id, hash_id, name, image, tag, disk, created, current. Nothing of `env` is kept.

### Rentals
`Market(home, api, now)`: `.rent(offer, filters, template, disk, label_note, bid, send_id) -> rental`, `.settle()`,
`.rentals()`, `.start(id)`, `.stop(id)`, `.destroy(id, require_rental=False)`, `.change_bid(id, price)`,
`.instances()` (parsed v1 list with a fresh `is_rulz`, `status`, `per_hour`, `ours`), `.ssh_key_state()`
(`missing` | `on_account` | `not_on_account`), `.add_ssh_key()`, `.prefs()` / `.set_prefs()`.
Rent body: `{"template_hash_id", "disk", "label", "target_state": "running", "cancel_unavail": true}` plus `"price"`
for a bid; without a template the BUILTIN image / env / onstart / `runtype: "jupyter_direc ssh_direc ssh_proxy"`.
A CLI (`python scripts/vast/rulz_market.py search|account|templates|instances|rentals`) prints JSON, reads only.

## 4. `plan_send(config, root) -> SendPlan`

Walks a config (a race's per-configuration settings included) and returns, without touching the network:
- `datasets`: every folder the run reads: `dataset_dir` / `datasets[].path`, `reg_dir`, folders of references that lie
  in no other pushed folder. Each `{path, name, files, bytes}`; `name` is the folder name, made unique
  (`<parent>_<name>`, then `_2`) when two differ only by parent. Hidden files do not count or go.
- `files`: local model files and folders that the instance does not have: `transformer_file`, `text_encoder_file`,
  `vae_file`, `tokenizer_path`, `model_path` when it is a local folder, `sample_model_path`, `turbo_lora`,
  `sample_lora`, `training_adapter`, `resume_from`, `finetune_resume`, `resume_state`, `nlf_model`, `sam3_model`,
  `soup_files[]`, `noise_files[]`, and any other path-valued option found in `TrainConfig` (list it in
  `PATH_KEYS`; a test fails when a new TrainConfig field whose name ends in `_file`, `_path`, `_dir`, `_files`,
  `_lora`, `_model` or `_state` is in neither `PATH_KEYS` nor `NOT_PATHS`). Each `{key(s), path, bytes, kind,
  standard}`; `standard` = what the instance would download instead when the setting is cleared (the family's own
  file of the same basename from `comfy_files` / the config tables), else None. Default choice: `standard` when there
  is one, else undecided (the user must choose).
- `changes`: settings rewritten for the instance, with the reason: `output_dir` -> `output`, `models_dir` -> `models`,
  `cache_dir` -> "" when absolute, device names beyond the instance's GPU count -> `cuda:0`, `multi_gpu` off on a
  one-GPU instance.
- `blockers`: missing dataset folder, unreadable file, empty path.
`apply(plan, choices) -> config` gives the config to send: dataset paths `datasets/<name>/...`, uploaded files
`models/sent/<digest12>/<name>`, standard ones cleared, changes made. Windows paths compare case-insensitively.

## 5. `rulz_files.py`

`digest(path)`: sha256, cached in `filehash.json` by (path, size, mtime_ns). Target on the instance:
`models/sent/<sha256[:12]>/<name>`; a folder: `models/sent/<tree digest12>/<name>/`. A file already there with the
right size and a `.ok` record is skipped.
- Straight upload: `file-state` says how many bytes of `<name>.part` are there; the rest is appended over ssh stdin in
  64 MB pieces through `file-put --at OFFSET` (refused when the part is not exactly that long); a dropped connection
  asks again and goes on; `file-done` checks size and sha256, then renames. Progress callback (bytes done, total).
- Bucket route: `huggingface_hub` (guarded import, >= 1.0 with bucket support; a clear message otherwise).
  `bucket_state(id)`, `buckets()`, `create_bucket(id, private=True)`, `upload(bucket, path, dest)` to
  `rulz-trainer/<sha256[:12]>/<name>`, skipped when the same size is there. Then the PC asks
  `https://huggingface.co/buckets/<id>/resolve/<path>` without following the redirect (token on the PC only) and sends
  only the short-lived signed address to the instance on ssh stdin; `file-fetch` downloads it there detached, in
  ranges over 16 connections, resumable, then checks size and sha256. A 401 / 403 asks the PC for a fresh address. A
  host other than `huggingface.co`, `*.huggingface.co`, `*.hf.co`, `*.xethub.hf.co` is refused.
- Token: `HF_TOKEN` > `huggingface_hub.get_token()`. The WebUI can sign in (`huggingface_hub.login`), a password field
  that is cleared at once. Never sent to the instance.
All names on the instance are built from digests and a cleaned basename; every path is checked to stay below
`models/sent`.

## 6. Watchdog (instance side)

`watchdog arm --idle-minutes N` (default 60, 10..1440) starts a detached stdlib process from a copy written to
`output/_watchdog/watch.py` with `output/_watchdog/state.json` (pid, created, minutes). Idle = a run was seen running
since arming and now nothing runs, nothing waits in the WebUI's queue, no LoRA test runs and no file or dataset is
arriving, for N minutes on end. Then it stops its own instance with CONTAINER_ID / CONTAINER_API_KEY (as
`machine.stop_instance`), after marking itself fired so that a restarted instance does not stop again. Arming checks
with a read of its own instance that the key works and answers `can_stop: false` otherwise. `watchdog disarm`,
`watchdog status`; `snapshot.watchdog = {armed, idle_since, minutes, fired}`. The SendManager arms it when the run
starts and again when a look finds it gone, and disarms it when the user picks Keep running or a run is started there
by hand with leave running.

## 7. `SendManager`

A send = `{id, job, created, instance or None, rental intent/label, plan, choices, config, after, steps, state,
error, entry, run}`; persisted after every change. Steps: `rent` (skipped for an existing instance) -> `boot`
(vast says running; 25 min) -> `reach` (ssh; 10 min) -> `setup` (RulZ Trainer installed and its WebUI up; 45 min) ->
`update` (only when asked: an older version there and nothing running) -> `files` -> `datasets` -> `check` (the
agent's dry run, after the uploads) -> `queue` (Fleet entry pinned to the instance) -> `run` -> `download` ->
`after`. Each `{state: waiting|going|done|failed|skipped, text, frac, started, ended}`.
Uploads and the checks run in one worker per instance; the step machine is a `tick()` any test can drive with a fake
clock, fake market, fake fleet. One process runs it (`send.lock`); others show it. A restart continues where it was;
a step that was going starts again (all are resumable or idempotent; `queue` looks for its entry id and the job on the
instance first and never queues twice).
`cancel(id)`: before `queue`: ends it, leaves the files, asks (UI) whether to stop the instance when this app rented
it. After: only through the Fleet (Stop). `retry(id)` from the failed step. `keep(id)`: after-run becomes "leave".
Events go into the Fleet's event log (`send_failed`, `send_started`, `instance_stopped_after_run`,
`instance_destroyed_after_run`).
A paused Fleet queue is reported on the send (it waits; the dialog offers Resume). Dataset upload progress comes
from the manager doing the push itself through `Fleet.push_datasets` before queueing (the dispatch then finds the
marks and skips).

## 8. UI

- **Run on** control (This PC / vast.ai) at the action bar of the Train, Dataset and Monitor tabs and on the Wizard's
  review page; one choice for the page, remembered per browser. With vast.ai the primary button reads **Send to
  vast.ai** (icon `cloud_upload`) and Add to queue is hidden. The footer also shows on the Dataset tab.
- **Send dialog**: (1) Where: running instances (GPU, $/hour, what runs there, free disk, version), stopped ones with
  **Start** (price, confirm), others greyed with the reason; **Rent a new one** opens the market dialog, and the
  picked offer returns here. (2) What goes up: datasets with sizes, model files with a choice each (Upload to the
  instance / Via Hugging Face bucket / Use the standard file), the bucket line (confirm or enter, Change default),
  estimated upload time, settings that change. (3) After the run: Stop / Destroy / Leave running, idle minutes. (4)
  Cost: $/hour, credit, time left including this instance, estimated run cost when the run's step time is known.
  One primary button: **Rent for $X/hour and send** or **Send**.
- **Rent GPUs** tab: credit strip, account's instances (Open on vast.ai, Start, Stop, Destroy), filters in
  collapsible groups, results table with sort and Rent, remembered filters, Reset, Reserved rows link to vast.ai.
- **Credit chip** in the header while anything is billed: `$11.44 · 14 h left`; amber under 6 h, red under 1 h.
- **Sends** list on the Fleet tab and in the dialog after sending: each step with progress, errors with Retry,
  Cancel, Keep running with the countdown.
Design: the RulZ AI Tools design guide's tokens through webui.py's existing theme classes; sentence case, no emoji,
one primary button per view, status always a word plus an icon.

## 9. Amendments (principal engineer's review; they override the sections above)

1. **Watchdog from the first contact.** It is armed at `reach` for every rental of a send whose after-run choice is
   stop or destroy (default 60 minutes). It also fires when no run has started and nothing arrived (no growing file
   below `models/sent` or `datasets/.rulz_incoming`, no running `file-fetch`, no provisioning) for the same N minutes.
   The manager's looks touch `output/_watchdog/alive`; while that file is younger than 10 minutes and no run was seen
   yet, the early rule does not fire (a long upload from a working PC is fine). `setup`'s 45 min, `queue`'s 30 min
   (counted only while the instance is idle and free for it) and `run`'s start are deadlines that fall under rule 5.
   A paused Fleet queue does not stop the `queue` deadline; the send says so from the start.
2. **It survives a container restart.** With supervisor there (`supervisorctl`, as provision.sh), it is a supervisor
   program (`/etc/supervisor/conf.d/rulz-watch.conf`, autostart, autorestart, `output/_watchdog/watch.py`); else a
   detached process. After a restart of the container (booted differs) a run that was going counts as ended: idle
   counts from then, a paused queue does not keep it from stopping. It is independent of `machine.AutoStop`; arming
   disarms nothing, but the manager reports an AutoStop that is armed there ("may stop the instance before the last
   download"). `can_stop: false`: the send goes on and shows a warning ("cannot stop itself: keep this PC on").
   A send whose instance was interrupted (bid lost, host reboot) shows "interrupted", keeps watching, and when the
   instance is back with no run going it is `failed` ("continue it from the saved state with Retry on the instance's
   own page"), after-run falls to stop.
3. **Reserved.** Fleet instance field `reserved: <send id>`, set when the rental joins the Fleet and cleared when the
   send ends. `schedule()` gives a reserved instance only the entry whose `send` equals it. Unpinned entries do not
   count as waiting for an instance; pinned, sent_to and uncertain ones do. For an existing instance `reserved` is set
   only while the send uploads and queues.
4. **Transport.** `Remote.agent` pipes the program on stdin, so bytes cannot follow it. `file-put` is not an agent
   command: the piece goes with `Remote.run` as a shell command (`[ "$(stat -c %s PART)" = OFFSET ] && cat >> PART`,
   everything quoted, in_folder), as `send()` does. `file-fetch` gets its signed address in `SENT_CONFIG` (program
   text on stdin, nothing on the command line or on disk) and hands it to its detached child through a pipe. `file-state`,
   `file-done` (long: sha256), `file-fetch`, `fetch-state`, `watchdog` have fitting timeouts; short ones go into
   `SHORT_COMMANDS`, all into `AGENT_COMMANDS`.
5. **Entry id = send id.** `plan`/`enqueue` accept an `id`; the send's id is the entry's id, saved before enqueue.
   A send matches its run only by the run record's `entry`, never by job name. A taken job name is refused in `check`.
6. **Start and bids.** A boot / reach timeout after a Start or a rent sends stop. Every stop the manager or a click
   sends is followed until vast no longer reports the instance running (2 min checks, 10 min), else a loud alert.
7. **Only the manager pushes.** The send's Fleet entry has `push=[]` and the final config, so an older looker can
   dispatch it. Named push items therefore live in `rulz_send.py` (it calls `remote.send` itself with the Fleet's
   mark and versioned-name rules) and `rulz_fleet.py` keeps its plain list.
8. **Reachable before rented.** Before the rent PUT: `ssh` on PATH, a local public key, and `ssh_key_state` ==
   `on_account`; otherwise refused with what to do. `remote.list_instances` uses the REST list when an API key is
   there (normalised to the CLI's shape: `extra_env` as a dict), falling back to the CLI.
9. **Destroy.** The agent gets `jobs --all` (every job folder, no limit) for the check; `output/_tests` non-empty ->
   stop instead. Choosing destroy forces every download kind to at least "end" and adds kind `tree`. Conditions also:
   not rebooting, not updating. A countdown restarts from the beginning after a WebUI restart.
10. **Gone and lost.** The rental joins the Fleet (`add_instance`) only once vast says running and ssh answers; until
    then the manager alone watches it. Discovery must not auto-add an instance whose label is an open rental of a send
    that has not reached that point. A run record "lost" is not final for a send while a look still shows the run.
11. **Disk.** `check` compares the free disk there with the bytes to come (+ 10%); the rent dialog warns when the
    container size is under models (the family's known download sizes when known, else 60 GB) + uploads + 20 GB.

## 10. As built so far, and the interface between the manager and the UI

`rulz_market.py`, `rulz_files.py`, `rulz_plan.py` (section 4's `plan_send` / `apply`) and the agent commands exist: read
their code and docstrings for the exact shapes. `rulz_send.py` offers (the UI uses only these and Market's):

- `SendManager(home=None, fleet=None, market=None, files=None, now=time.time, sleep=time.sleep)`; `hold()` / `release()`
  (the `send.lock`), `step()` for a loop (does nothing unless it holds; runs `market.settle()` and `tick()`), `tick()`.
- `preview(config, root, gpus=1) -> rulz_plan.SendPlan` (slow: file system; call it off the page's thread).
- `instances() -> [{"id", "gpu_name", "num_gpus", "gpu_ram_mb", "per_hour", "dph_total", "storage_per_hour", "status",
  "label", "ours", "is_rulz", "selectable", "reason", "free_gb", "version", "job", "waiting", "sends"}]`: vast.ai's list
  joined with the Fleet's last look. selectable: running RulZ Trainer instances, or ones that can be started
  (status stopped / exited; `reason` "stopped: starting it bills $X/hour").
- `create(job, config, root, *, instance=None, start=False, rent=None, choices=None, after="stop", idle_minutes=60,
  update=False) -> send`. `rent = {"offer": Offer or row, "filters", "template", "disk", "bid"}`; exactly one of
  instance / rent. `choices = {path: {"how": "upload"|"bucket"|"standard", "bucket": "user/name"}}`. Called from the
  click: renting (and starting, with start=True) happens inside this call, after the send was saved. Raises
  `SendError` / `MarketError` (kind "pending": the send exists and waits for the intent).
- `sends(active=False) -> [send]`, newest first; `get(id)`. A send: `{"id", "job", "created", "instance", "label",
  "after", "idle_minutes", "state": "going"|"waiting"|"running"|"done"|"failed"|"cancelled"|"interrupted", "step",
  "steps": [{"key", "label", "state", "text", "frac", "started", "ended"}], "error", "warnings": [str], "per_hour",
  "rented": bool, "run", "entry", "countdown": seconds or None, "action": "stop"|"destroy"|None, "result": text}`.
- `cancel(id, instance_action="keep"|"stop")`, `retry(id)`, `keep(id)`, `dismiss(id)` (an ended one), `set_after(id,
  after)`.
- `SendError(message)`.
