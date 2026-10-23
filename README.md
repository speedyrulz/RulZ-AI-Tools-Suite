<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo_dark.png">
    <img src="assets/logo.png" alt="Rulz" width="420">
  </picture>
</p>

# RulzTrainer

A simple diffusion-model LoRA trainer with a WebUI. Supported models:

- **MOSS-Image** (5B single-stream DiT by OpenMOSS), both the **Base** and **SFT** checkpoints. Output LoRAs load
  directly in ComfyUI with the standard *Load LoRA* node.
- **Bria FIBO 2.0** (8B): train on dev, use on Edit Turbo ([details](#fibo-20-bria-training-on-dev-using-the-lora-on-edit-turbo)).
- **Krea 2** (12B): train on Raw, use on Turbo ([details](#krea-2-training-on-raw-using-the-lora-on-turbo)).

## Install

**Windows:** double-click `install.bat`. It creates `venv\`, installs CUDA PyTorch (cu130 by default; pass another
index URL as the first argument, e.g. `install.bat https://download.pytorch.org/whl/cu128`) and the requirements.
Then double-click `run_webui.bat` to start the UI and open it in your browser (optional first argument: port).

**Linux:** `./install.sh` does the same (`venv/`, cu128 PyTorch wheels by default, pass another index URL as the
first argument, or `system` to keep an already installed torch), then `./run_webui.sh [port] [host]`. On a headless
box (no display) the WebUI has no native Browse dialogs: type paths, or upload datasets on the Dataset tab.
Cloud GPUs: see [Running on vast.ai](#running-on-vastai).

**Manual:**

```bash
pip install -r requirements.txt
```

Requires a CUDA build of PyTorch 2.9+, diffusers 0.38+ and transformers 5.3+ (Qwen3.5 support).

## Run the WebUI

```bash
python webui.py --port 7870
```

Open http://127.0.0.1:7870.

New to LoRA training? Start on the **Wizard** tab. It walks through a run one step at a time (goal, model, GPUs,
dataset, training, samples, review) and explains every setting. Each one gets a recommended value based on your GPU
memory and on the image count and size of your dataset. *Use* takes one recommendation and *Apply recommended* takes
every one on the page. The wizard edits the same settings as the **Train** tab, which lists every option grouped by
topic, with rarely changed ones folded under *Advanced*. *Start training*, *Stop* and *Kill* sit in the bar at the
bottom of the Train and Monitor tabs.

1. **Dataset**: a folder of images. Each image may have a same-name `.txt` caption. Use *Scan dataset*
   to see the aspect-ratio buckets. Pick one or more *Training resolutions* (megapixels); see
   "Training resolutions and resolution step-up" below.
2. **Model**: choose `MOSS-Image-Base` or `MOSS-Image-SFT` (or a custom local directory / HF repo id).
   *Download / verify model* fetches about 30 GB into `models/`.
3. **Start training**. The *Monitor* tab shows progress, loss curve, log tail and sample images. Training runs
   as its own process: reloading or closing the page, opening it from another browser, or restarting the
   WebUI does not stop it, and the page reconnects to the running job (it opens on *Monitor* and loads the
   job's settings into the form). *Stop (save)* and *Kill* work on a reconnected job too.
4. Results: `output/<job>/<job>_<step>.safetensors` (checkpoints) and `output/<job>/<job>.safetensors` (final).

*Save* / *Load* (in the *Training job* card at the top of the Train tab) store the whole form as YAML (Browse opens
in `configs/`). Loading a config saved by
an older version fills every option added since with its default, appends those options to the end of the file
(your values and comments are left as they are) and lists them under the buttons. Options a newer version no longer
has are ignored instead of refusing the file. `configs/example.yaml` ships with the repo and is never modified.

### Remote access with Tailscale

To use the WebUI from your other devices (laptop, phone), start it with Tailscale. On Windows, double-click
`run_webui_tailscale.bat` or run `run_webui.bat 7870 tailscale`. On Linux, run `./run_webui.sh 7870 127.0.0.1 tailscale`.
`python webui.py --tailscale` and `RULZ_TAILSCALE=1` do the same.

It runs `tailscale serve`, so the WebUI stays bound to `127.0.0.1`. Tailscale publishes it over HTTPS at
`https://<machine>.<tailnet>.ts.net:7870`, and only devices logged in to your tailnet can open it. The URL is printed
on start. It uses its own HTTPS port, so an existing `tailscale serve` on the default port (ComfyUI, for example) is
not touched. When the WebUI stops (Ctrl+C, closing its console window, logging off), the address switches to a short
"RulzTrainer is not running" notice instead of a 502 error, and the next start brings the WebUI back on it.
Only a hard kill (Task Manager, a crash) leaves the 502. `tailscale serve --https=7870 off` removes the address for good.

Requirements: Tailscale installed and logged in on this machine, and HTTPS certificates turned on for the tailnet
(admin console > DNS > HTTPS Certificates). Browsers that connect remotely get no native *Browse* buttons, because
those dialogs would open on this machine's screen. Type server paths instead, or upload datasets on the Dataset tab.

### CLI

```bash
python -m rulztrainer.train --config configs/example.yaml --set steps=1000 --set job_name=cat_lora
```

## Running on vast.ai

Everything needed is in `scripts/vast/` and the repo is public, so a template only has to point at it.

**Create the template** (once), either with the CLI:

```bash
pip install vastai && vastai set api-key <your key>
bash scripts/vast/create_template.sh
```

or in the web editor (Templates -> Create) with these settings:

| field | value |
|---|---|
| Image path:tag | `vastai/pytorch:cuda-12.8.1-auto` (CUDA 12.8 build: RTX 30 / 40 / 50 series hosts) |
| Launch mode | Jupyter + SSH, direct connections |
| On-start script | `/opt/instance-tools/bin/entrypoint.sh` |
| Ports | `1111`, `7870`, `8080`, `22` |
| Environment variables | `PROVISIONING_SCRIPT=https://raw.githubusercontent.com/speedyrulz/RulzTrainer/main/scripts/vast/provision.sh`<br>`PORTAL_CONFIG=localhost:1111:11111:/:Instance Portal\|localhost:7870:17870:/:RulzTrainer\|localhost:8080:18080:/:Jupyter\|localhost:8080:8080:/terminals/1:Jupyter Terminal`<br>`OPEN_BUTTON_PORT=1111` |
| Disk space | 120 GB (the base model download is about 30 GB; datasets, caches and outputs come on top) |

Put `HF_TOKEN` (needed for the gated MOSS-Image repos) into your **account** environment variables
(cloud.vast.ai -> Account -> Environment Variables), never into the template: account variables are injected into
every instance and land in `/etc/environment`, where the WebUI and the training subprocess read them.

**Rent an instance** from the template (one 16 GB+ GPU is enough; two GPUs enable the multi-GPU modes). On the first
boot `provision.sh` clones the repo into `/workspace/RulzTrainer`, installs the requirements into the image's
`/venv/main` (its torch is reused) and registers the WebUI as a supervisor service; this takes 3-5 minutes and
shows in the Instance Portal's log viewer. Then click *Open* on the instance: the Instance Portal lists
**RulzTrainer** (the WebUI, proxied by Caddy on port 7870 with the portal's token / basic auth, so it is never open
to the internet unauthenticated), Jupyter (file browser + terminal for uploads and downloads) and the tunnel links.

Datasets: upload on the WebUI's Dataset tab (drag and drop images + `.txt` captions) or into
`/workspace/RulzTrainer/datasets/<name>` with Jupyter / `scp` / Syncthing. Results are written to
`/workspace/RulzTrainer/output/<job>/`; download them from Jupyter before destroying the instance (the disk is gone
with it; stopping keeps it). `models/` on the instance disk caches the downloaded base model between runs.

**Auto-download your own model files from a Hugging Face Storage Bucket.** Add
`RULZ_MODEL_BUCKET=speedyrulz/MOSS-Image-bucket` to the template's environment variables (web editor: Environment
Variables; CLI: `MODEL_BUCKET=speedyrulz/MOSS-Image-bucket bash scripts/vast/create_template.sh`). After the
install, `provision.sh` syncs the bucket into `/workspace/RulzTrainer/models/bucket/` in the background, keeping
its folder layout, so the WebUI is usable (dataset uploads) while about 18 GB come down. Progress is in
`models/bucket_download.log`; `models/bucket/.download_complete` appears when every file is there. Then point the
single-file fields at the files, for example:

| WebUI field | path on the instance |
|---|---|
| Diffusion model | `models/bucket/diffusion_models/moss_image_base_int8_convrot.safetensors` |
| Text encoder | `models/bucket/text_encoders/qwen3.5_9b_int8_convrot.safetensors` |
| VAE | `models/bucket/vae/flux2-vae.safetensors` |
| Turbo LoRA (samples) | `models/bucket/MOSS-Image-Base-8-steps-dmd.safetensors` |

Rerun `bash scripts/vast/download_models.sh` (with `RULZ_MODEL_BUCKET` set) to fetch files you add to the bucket
later; complete files are skipped. `RULZ_MODEL_INCLUDE="vae/* text_encoders/*"` limits the download to matching
files and `RULZ_MODEL_DIR` changes the destination. A public bucket needs no token; a private one uses `HF_TOKEN`
from your account environment variables. The script upgrades `huggingface_hub` on the instance if its version
has no bucket support.

Service control on the instance: `supervisorctl status|restart rulztrainer`; its log is
`/var/log/portal/rulztrainer.log`. `RULZ_REPO` / `RULZ_BRANCH` environment variables point the provisioning at a
fork or branch. `Dockerfile` builds an image with the install baked in (`FROM vastai/pytorch:cuda-12.8.1-auto`,
same supervisor wiring) for faster instance starts: build, push and use it as the template image; the provisioning
variable is then unnecessary.

## How it trains

- **Caching first.** On the first run the Flux2 VAE encodes every image into normalized 128-channel
  latents (2x2 patchified, batch-norm normalized exactly as the reference pipeline does), and the
  Qwen3.5-9B text encoder embeds every caption through the chat template. Both are cached under
  `<dataset>/.rulz_cache`, then the encoders are unloaded. Training only holds the DiT and the LoRA.
- **Timesteps.** `logit_normal` (mean/std), `sigmoid` (scale) or `uniform`, with an optional `timestep_shift`.
- **Flow matching loss.** `x_t = (1 - sigma) x0 + sigma * noise`, target `noise - x0`, timestep = `sigma * 1000`,
  sigmas snapped to the model's 1000-step training grid. Default sampling is logit-normal; `timestep_shift`
  is available.
- **LoRA** on the DiT attention and FFN projections (`q, k, v, o, ffn_up_gate, ffn_down`) in all 42 blocks, or
  only the blocks you pick (`lora_blocks`, e.g. `21-41`, `10-31,40`, `even`; the UI has a per-block toggle grid
  with presets). Blocks without LoRA get no backward pass if they sit before the first trained block, so
  training only later blocks is also faster per step. The exported file simply omits the untrained blocks.
  Optionally also the two text-projection blocks inside the DiT. The Qwen text encoder stays frozen, as the
  authors recommend.
- **Samples** are generated with the reference Euler schedule (shift 5, CFG 3, negative-branch layer-22 skip).

## Using two GPUs

`multi_gpu: data_parallel` with `devices: cuda:0,cuda:1` loads a full model copy on each GPU; every step each
copy trains its own images (effective batch = batch_size x GPUs), the LoRA gradients are averaged onto the
primary, the optimizer steps once and the LoRA weights are broadcast back. This is the mode that speeds
training up (roughly 1.7-2x throughput with two cards; the faster card waits for the slower one each step).
It needs no torch.distributed and works on Windows; each GPU must fit the whole model, so pair it with
`transformer_quant: int8_convrot` or `nvfp4` on 16 GB cards, or bf16 at modest resolutions.

`multi_gpu: model_split` places consecutive block ranges on different GPUs instead. It halves the memory per
GPU (full bf16 at high resolution becomes possible) but is not faster, since the GPUs take turns. The split is
auto-balanced by estimated cost (blocks before the first trained block are forward-only and cheap; every later
block also runs backward), or set `split_plan: "cuda:0=0-25;cuda:1=26-41"` to give the faster card more.

### Splitting side jobs across the GPUs (`parallel_jobs`, on by default)

With either multi-GPU mode, the work around training is split across the GPUs too. Each GPU has its own worker
that pulls the next job from a shared queue, so the faster card simply does more of them:

- **Samples and probes**: every sample prompt and every probe render (face, body, each seed) is a job. Two probes
  means one per GPU. In data parallel each GPU renders on its own training copy. In model split the training
  model spans both GPUs, so images split only when a separate sampling model is set (it gets a copy per GPU).
  Scoring stays on the main thread after the renders.
- **Latent caching**: a VAE copy per GPU, each encoding its share of the images (and their flips).
- **Text caching**: a second text encoder copy is loaded on the other GPU when the encoder sits wholly on one
  GPU (a quantized Qwen3.5-9B, about 10 GB) and timing shows the copy saves more than it costs to load. A bf16
  encoder (19 GB) is spread over both GPUs and stays a single copy.
- **SAM 3 region masks** (`region_weights` with `mediapipe+sam3`): a SAM 3 copy per GPU, each kept resident for
  the whole batch instead of moving to the GPU and back for every image.

Face detection (InsightFace) and MediaPipe run on the CPU and are not split. With `compile_model`, the first
sampling round of each image size runs one GPU after the other while the graphs compile; later rounds run in
parallel. Turn the switch off ("Split samples, probes and caching across GPUs") to keep all side jobs on the
primary GPU.

Measured with data parallel on an RTX 5060 Ti + RTX 4060 Ti (8 images, 512 px, 8 sampling steps):

| Side job | One GPU | Split across both |
|---|---|---|
| 2 samples | 28.6 s | 14.7 s |
| Face + body probe (render and score) | 35.4 s | 17.7 s |
| Latent cache, 8 images + flips | 5.6 s | 1.9 s |
| SAM 3 region masks, 8 images | 16 s (was 28 s before SAM 3 stayed resident) | 9 s |

Text caching is only split when it pays: the first few prompts are timed on the loaded encoder, and a second copy
is loaded only if it saves more time than loading it takes (about 13 s for the int8 Qwen3.5-9B). Images rendered on
different GPU models differ very slightly in their pixels (different kernels); the same seed gives the same picture.

## Model sources and quantization

**Hugging Face / diffusers folder**: pick Base or SFT, or any local folder with `transformer/`, `text_encoder/`,
`tokenizer/`, `vae/`.

**Single safetensors files (ComfyUI)**: point the UI at the three files you already use in ComfyUI: the diffusion
model (`diffusion_models/`), the Qwen3.5-9B text encoder (`text_encoders/`) and the Flux2 VAE (`vae/`). Fused-qkv
("repacked") files are split back automatically. The tokenizer is downloaded once (a few MB) if not given.
Any file left empty is taken from the Hugging Face model instead, so you can mix (for example a ComfyUI DiT
file with the HF text encoder). For decoding samples use the VAE that matches MOSS-Image (the FLUX.2 klein
"small" VAE, `flux2-vae_small.safetensors` in ComfyUI); the encoder is identical across Flux2 VAEs, so caching
latents works with either.

Accepted weight formats: bf16 / fp16 / fp32, fp8 (upcast), ComfyUI `int8_tensorwise` with or without ConvRot
(Hadamard rotation), and ComfyUI `nvfp4`. Those two quantized formats are decoded exactly as comfy_kitchen does
(verified against the HF weights: re-quantizing them with RulzTrainer reproduces the ComfyUI file bit for bit).

*Diffusion model quant* / *Text encoder quant*:

| value | effect |
|---|---|
| `auto` | keep the file's format (bf16 stays bf16, int8/nvfp4 files stay quantized) |
| `none` | dequantize everything to bf16 |
| `int8`, `int8_convrot` | quantize eligible Linear layers on load (about half the VRAM) |
| `nvfp4` | quantize on load to FP4 with FP8 block scales (about a quarter of the VRAM) |

Only the frozen base weights are quantized. The LoRA trains in fp32. Exported LoRAs are bf16 by default
(`lora_save_dtype`).

### int8 compute: W8A8 (`int8_matmul`, `int8_backward`)

int8 and int8_convrot layers do not dequantize their weight any more. With `int8_matmul: true` (the default)
each Linear runs **W8A8**: the activations are quantized per token to int8 (symmetric, fp32 scale per row), the
int8 activations are multiplied with the stored int8 weight by `torch._int_mm` (int32 accumulate), and the
result is rescaled by the activation scale and the per-output-row weight scale in the compute dtype. For
int8_convrot files the activations are first rotated with the same grouped Hadamard (group 256) the weight was
stored with: x W^T = (xH)(WH)^T, so no inverse rotation of the weight is needed, and the rotation spreads
activation outliers over the group, which is what makes per-token int8 accurate on these models.

The two elementwise passes around the GEMM (per-token quantization of the activations, int32 -> bf16 rescale by
both scales) are single fused triton kernels when `triton` imports (`triton-windows` on Windows, in
`requirements.txt`; triton ships with the Linux torch wheels). For int8_convrot the Hadamard rotation of the
activations is folded into the quantization kernel: H_256 = H_16 (x) H_16, so each 256-group of a token row is
rotated as H_16 X H_16 with two tiny tensor-core dots (fp32 accumulate, the intermediate carried as a bf16
hi + lo pair), the row amax is taken over the rotated values and they are quantized straight from registers;
the rotated bf16 tensor is never written. Without triton the eager PyTorch versions run (a bf16 matmul for the
rotation, then two and three memory passes); the choice is made once at import and a kernel failure at run time
logs one warning and switches to eager. `RULZ_W8A8_TRITON=0` forces the eager path, e.g. for the before/after
bench.

The previous behaviour (`int8_matmul: false`) dequantized the weight to bf16 in every forward *and* every
backward, three times per step with gradient checkpointing, which made int8 training about 3x slower than
bf16. Measured with `scripts/bench_int8.py` (4096 tokens, int8_convrot, 50 iterations; "eager" is the same
W8A8 path with `RULZ_W8A8_TRITON=0`), forward / forward+backward ms:

| layer (K -> N) | path | RTX 5060 Ti | RTX 4060 Ti |
|---|---|---|---|
| 3072 -> 8192 (ffn up) | bf16 weight | 4.1 / 8.3 | 4.8 / 8.9 |
| | W8A8 fused, bf16 backward (default) | **1.9** / 6.7 | **2.3** / 7.2 |
| | W8A8 fused, int8 backward | 1.9 / 4.7 | 2.3 / 5.6 |
| | W8A8 fused, Hadamard as a separate matmul (previous) | 2.0 / 6.7 | 2.4 / 7.3 |
| | W8A8 eager, bf16 backward | 2.9 / 7.5 | 3.5 / 8.3 |
| | W8A8 eager, int8 backward | 2.9 / 6.6 | 3.5 / 8.2 |
| | dequantize per call (old) | 6.5 / 12.9 | 7.3 / 14.6 |
| | `torch._int_mm` alone | 1.2 | 1.3 |
| 8192 -> 3072 (ffn down) | bf16 weight | 4.2 / 8.3 | 4.7 / 8.9 |
| | W8A8 fused, bf16 backward | **1.7** / 6.5 | **2.0** / 7.2 |
| | W8A8 fused, int8 backward | 1.7 / 4.6 | 2.0 / 5.5 |
| | W8A8 fused, Hadamard as a separate matmul (previous) | 2.1 / 7.1 | 2.5 / 7.7 |
| | W8A8 eager, bf16 backward | 3.2 / 8.3 | 4.0 / 9.2 |
| | dequantize per call (old) | 6.4 / 12.9 | 7.3 / 14.6 |
| 3072 -> 3072 (q/k/v/o) | bf16 weight | 1.5 / 3.1 | 1.8 / 3.3 |
| | W8A8 fused, bf16 backward | **0.8** / 2.6 | **1.0** / 3.0 |
| | W8A8 fused, int8 backward | 0.8 / 2.0 | 1.0 / 2.6 |
| | W8A8 fused, Hadamard as a separate matmul (previous) | 0.9 / 2.8 | 1.1 / 3.1 |
| | W8A8 eager, bf16 backward | 1.5 / 3.3 | 1.9 / 3.8 |
| | dequantize per call (old) | 2.4 / 4.8 | 2.7 / 5.5 |

With the fused kernels the forward is 2-2.4x faster than bf16 on every MOSS layer shape (the eager W8A8
path was 1.4x on the ffn layers and only bf16 speed on the 3072 -> 3072 ones). Folding the Hadamard into the
quantization kernel took the rotate + quantize step from 0.31 / 0.93 ms (K = 3072 / 8192, 4060 Ti) to
0.13 / 0.44 ms; the remaining gap to the bare int8 GEMM is the int32 -> bf16 rescale of the output plus that
one read of x. With checkpointing the forward runs twice per step, so a step is bf16 speed or better at half
the VRAM. Forward error versus the dequantized weight is under 1% (the activation quantization), the same
order as the weight quantization itself. The fused quantizer and rescale round exactly like the eager ones
(the rescale even avoids the eager path's second bf16 rounding); the fused rotation keeps the rotated values
in fp32 where the separate bf16 matmul rounds them, so its int8 values are within 1 LSB of the two-step path
and closer to the exact rotation (about 5% of the two-step values are 1 LSB off an fp64 reference, about
0.002% of the fused ones).

`int8_backward` chooses how grad_x is computed:

| value | effect |
|---|---|
| `bf16` (default) | exact: the int8 weight is widened to bf16 once per backward and grad_x = grad_out @ W (no activation or gradient quantization; convrot applies the inverse Hadamard to grad_x) |
| `int8` | the weight scale is folded into grad_out, grad_out is quantized per token to int8 and multiplied with the int8 weight; about 1% gradient error, saves the bf16 weight copy and is slightly faster |

Fallbacks: nvfp4 layers always dequantize (there is no fp4 matmul here); CPU tensors and shapes the int8
kernel rejects (in/out features not multiples of 8) use the dequantize path silently; a kernel error at run
time logs one warning and falls back for that call. Rows are padded to 17 when a call has 16 tokens or
fewer, because `torch._int_mm` refuses smaller M. In the WebUI these are *int8 layers compute* and *int8
backward* in the model section.

## Training resolutions and resolution step-up

*Training resolutions* is a list of megapixel values (`resolutions: [0.25, 1.0, 4.0]` in YAML; the WebUI offers
0.25 to 4 MP in 0.25 steps, 1 MP = 1024x1024). Every image is bucketed to each selected area with its aspect
ratio preserved (sides multiples of 16), and latents are cached once per image and resolution, so a second run at
the same resolutions starts immediately. `min_bucket_side` / `max_bucket_side` now rescale the bucket uniformly
before clamping, so a 2:3 portrait at 0.25 MP keeps its aspect ratio instead of being cropped to the minimum
side; raise `max_bucket_side` above 2048 if you want 4 MP buckets for wide images. The old single `resolution`
(a side length) still works when `resolutions` is empty.

With several resolutions selected and *Resolution step-up* off, all of them are mixed in every epoch (one item per
image and resolution, so an epoch is N times longer). Per-image loss watch, curriculum weights and exclusions are
keyed by image, so they apply to every resolution of that image.

**Resolution step-up** (`resolution_step_up: true`) trains the resolutions one after another, lowest first, and
lets the likeness probes decide when to move on:

* every probe (`probe_every` steps) yields the smoothed combined face/body score; the first probe of a stage is
  that stage's baseline and a later probe counts as an improvement when it beats the stage's best by more than
  the noise band;
* after `step_up_patience` probes in a row without improvement (default 1) the batcher switches to the next
  resolution; a new stage starts with a fresh baseline, so a dip right after the switch does not count against it;
* at the highest resolution, `step_up_stop_patience` probes without improvement (default 1) stop training; the
  final LoRA is saved as usual and `<job>_best.safetensors` still tracks the best score over the whole run.

`steps` stays the hard upper limit. In this mode the plain early stop (`likeness_patience`) is not used; the
learning-rate control, curriculum and best-checkpoint actions keep working across stages. The mode needs probes
(`probe_every > 0` with face and/or body references); the trainer refuses to start otherwise. The current
resolution and stage are logged, shown in the monitor line and written to `likeness.jsonl` (`stage`,
`stage_best`, `stage_since_improve`) and to training states, so a resumed run continues at the same stage.

## Training states

`save_state: true` writes `output/<job>/state_XXXXXX/` folders (fp32 LoRA, optimizer, LR scheduler and RNG
states) every `save_state_every` steps (0 = with each LoRA save), keeps a rolling `keep_states` of them
(0 = all), and always writes the final state when training ends or is stopped. Set `resume_state` to one of
those folders to continue exactly where it stopped. `resume_from` loads only LoRA weights from any exported file.

## Samples and monitoring

`sample_seed` is the seed of the first prompt; with `sample_walk_seed` each further prompt uses seed + 1.
`sample_before_training` renders a sample set at step 0 for comparison. The log reports each prompt and the
Monitor page shows the sampling step. The Monitor page also has per-GPU VRAM/utilization bars, an adjustable
loss-smoothing slider and an auto-scrolling log. `compile_model` runs the DiT blocks through torch.compile
(skipped with a warning if triton is not installed; `triton-windows` from requirements.txt provides it on Windows).
No C++ compiler is needed: the blocks are compiled with Dynamo's `specialize_float` so the LoRA scale and
multiplier stay Python constants instead of becoming CPU tensors (which would make Inductor emit C++ and fail with
"InvalidCxxCompiler: cl is not found" on a machine without MSVC on PATH), and the fused W8A8 kernels run eagerly
behind a graph break, so int8 layers keep their fused triton path inside compiled blocks.

## Identity similarity for character LoRAs (face + body)

Pick reference images on the Dataset tab (*Face ref* / *Body ref* toggles) or list them as `face_refs` /
`body_refs`. After every sampling pass each sample is scored and the results go to `face_scores.jsonl`, the
Monitor chart and the sample captions:

- **Face**: the largest face in the sample is embedded with InsightFace (ArcFace, `buffalo_l`) and compared by
  cosine similarity to every reference face; the best match is reported as a percent (cosine x 100). For
  ArcFace, 50%+ generally means the same person; the references' own pairwise similarity is shown as the
  realistic ceiling.
- **Body**: MediaPipe pose landmarks give scale-free proportion ratios (shoulder/hip width, torso/leg length,
  upper/lower leg, upper arm/forearm, head/shoulder width, leg/height...). With `body_backend: mediapipe+sam3`,
  a SAM 3 person mask adds silhouette widths at the shoulder, waist, hip and thigh levels (normalised by body
  height) plus waist/hip and shoulder/waist, which captures build rather than only bone proportions. A median
  profile over the body references is the target; a sample scores by how close its ratios are
  (20% deviation on a ratio ~ 61%). Silhouette features are only used when the skeleton confirms a full-body
  view, and samples without a usable pose get no body score. Body references should be full-body, upright,
  fairly neutral poses, and the same goes for the sample prompts you want body-scored.
- **Combined**: `face_weight` x face + `body_weight` x body (defaults 0.7 / 0.3; whichever is available).

Model downloads (the InsightFace pack, the MediaPipe pose model and SAM 3) happen on first use with a progress
bar in the UI (reference checks) and in the status line during training.

Requirements: `insightface` + `onnxruntime-gpu` (face), `mediapipe` (body); `facebook/sam3` is a gated
Hugging Face model (accept the license, then `hf auth login`), or point `sam3_model` at a local
transformers-format folder. SAM 3 (3.4 GB) is kept in system RAM and moved to `body_device` only while scoring.

## Per-image loss watch and per-image learning rate

`loss_watch: true` records every image's loss normalised by the expected loss at its noise level (40 sigma
buckets, kernel-smoothed) and smooths the residual per image. At each epoch boundary every image gets a
verdict relative to the cohort median: *easy*, *healthy*, *suspect* (hard but improving), *stuck* (hard and not
improving for two epochs) or *exhausted* (learned, flat). `per_image_lr: true` acts on the verdicts the way
Fizgig's loss watch does: stuck images are throttled x0.5, then x0.25, then x0.125 (recovering one rung per
improving epoch), exhausted ones ease to x0.6, the healthy cohort gets `loss_watch_healthy_mult`. With batch
size 1 and no gradient accumulation this is a true per-image learning rate (the optimizer lr is scaled for that
step); otherwise it becomes a per-sample loss weight. `loss_watch_exclude_after: N` benches an image that stays
at the lowest throttle for N more epochs; exclusions are remembered in `<dataset>/.rulz_cache/excluded.json`
and cleared by editing the image's caption. Output: `per_image_loss.jsonl` (every observation),
`per_image_loss.json` (the table), a Monitor table and verdict badges on the Dataset tab. Not used with
`automagic3`, whose step normalisation would absorb the per-image scaling. Verdicts need a few epochs to mean
anything, so this suits typical character sets trained for thousands of steps.

## Identity training: what each step teaches (character LoRAs)

Four levers change the training objective and data for likeness, all off by default:

- **Region-weighted loss** (`region_weights`): at cache time the largest face (InsightFace) and the person
  silhouette (SAM 3 with `body_backend: mediapipe+sam3`, else a MediaPipe landmark hull) are mapped onto the
  16-px latent cells of each bucketed image and turned into a weight map (`face_loss_weight` 3 / `body_loss_weight`
  1.5 / `background_loss_weight` 0.3, `face_margin`, `region_feather_cells`), normalised to mean 1 and stored as
  `<stem>_regions.safetensors`. The per-cell squared error is multiplied by it. `region_prior_weight` adds an
  optional prior-preservation term on background cells toward the base model's prediction (extra forward).
- **Timestep emphasis**: `timestep_sampling: identity` (logit-normal mean -0.5, median sigma ~0.38),
  `content` (sigma = u^`content_power`, low-noise heavy) or `style` (mirror), plus `sigma_min` / `sigma_max`.
- **Automatic crops** (`crop_augment: face | face+bust`): a square face crop (`crop_face_scale` x face size) and a
  bust crop (`crop_bust_scale`) per image with a face, with captions from `crop_face_caption` / `crop_bust_caption`
  (`{caption}`, `{raw}`, `{trigger}`), skipped when the image is already a close-up or the crop would be tiny.
  Crops are ordinary items (own latents, captions, region maps, loss-watch verdicts) and count as face-heavy for
  the curriculum; `crop_augment_weight` scales how often they are drawn.
- **LoRA+** (`lora_plus_ratio`, 16 = paper): the `lora_up` matrices train at lr x ratio for faster convergence
  (AdamW-style optimizers; ignored by automagic3). Resuming a state requires the same ratio.

The face analysis cache is `<dataset>/.rulz_cache/faces.json`; the Train tab has preview buttons for region
maps and crops.

Measured on a 5-image synthetic character set (120 steps, probes every 20 steps, same seeds, face likeness EMA
at step 100; baseline 74.2): region-weighted loss 76.0, LoRA+ ratio 4 74.7, `identity` timesteps 73.9, auto crops
65.5 (the only crops available were 300 px and had to be upscaled), everything combined with LoRA+ 16: 63.1
(unstable, loss 40% higher). The **Identity training preset** (a switch at the top of the Train tab, `--preset identity` on the CLI, or
`preset: identity` recorded in the YAML) applies the whole recommendation in one go and restores your previous
settings when switched off. Suggested recipe: `region_weights: true`, `lora_plus_ratio: 4` with `adamw8bit`,
crops only from high-resolution photos (`crop_augment_min_px: 0` means crops are never upscaled), timesteps at
`logit_normal`; treat `identity` timesteps and the curriculum as options to A/B with the probes.

### Experimental: direct identity loss

`id_loss_weight > 0` (default 0 = off; the applied weight is capped at 0.5) adds a face-recognition term to the
diffusion loss. On every `id_loss_every`-th training sample with `sigma <= id_loss_sigma_max` (0.35) and a cached
face at least `id_loss_min_face_cells` (4) latent cells wide, the trainer forms the predicted clean latent
`x0 = noisy - sigma * pred`, crops the face cells plus `id_loss_margin_cells` (2) of context, decodes only that
crop with gradients (`decode_latents_grad`), aligns it to the 5-point face template with a differentiable affine
warp built from the cached landmarks, embeds it, and adds `weight * (1 - cos(embedding, reference))`. The
reference is one random face reference per step (`id_loss_random_ref`) or their mean embedding.

- Backends (`id_loss_backend`): `arcface` ports InsightFace's `buffalo_l/w600k_r50.onnx` to torch with
  `onnx2torch` (`pip install onnx2torch`; a self-check against the ONNX scorer on a reference face must reach
  cosine > 0.99, else the loss is disabled). This is the very network the face probe scores with, so **the probe
  metric becomes the training target** and can climb without a real likeness gain. `facenet` uses a vendored
  InceptionResnetV1 with VGGFace2 weights (downloaded once to `models/facenet/`) and keeps the probe an
  independent check.
- Needs `face_refs` (the reference identity) and builds the face cells / landmarks cache even without
  `region_weights` (person masks are skipped then). Runs on the primary GPU only (data parallel: the other
  replicas train on the diffusion loss alone).
- Guard: with probes on, if the smoothed face score falls more than 2 x `likeness_noise_band` below its best for
  two probes in a row, the controller vetoes the identity loss for the rest of the run (logged, `id_loss_vetoed`
  in the status).
- The Monitor loss chart shows the identity loss (smoothed, right axis); the step log reports its mean, sample
  count, seconds per sample and the peak VRAM.

Cost and effect, measured on an 8-image synthetic character set (512 px, base int8, rank 16, lr 2e-4, 80 steps,
one 16 GB card, arcface, weight 0.2; 22 of 80 steps had an eligible sample): the ArcFace port matched ONNX at
cosine 0.9998; the decoder runs as a bf16 copy with gradient checkpointing (an 18 x 18-cell face crop costs
~0.6 GB for forward + backward instead of 3.5 GB in fp32); peak VRAM 8.0 GB vs 6.05 GB without the loss (+2 GB
including the extra VAE and face network); 0.86 vs 0.80 s/it (+7%). The identity loss fell from 0.86 (steps
10-25) to 0.61 (27-43) to 0.47 (51-79). A control with the same draws and a negligible weight (1e-6, measured
only) fell 0.88 / 0.68 / 0.54, so most of the drop is ordinary learning and the loss adds roughly 0.07 on top in
this short run. Treat it as an experiment to A/B against the probes (with the facenet backend for an independent
score), not a default.

## Likeness probes and controller (character LoRAs)

With face/body references set, `probe_every: N` renders a fixed-seed headshot (`probe_face_prompt`) and/or
full-body image (`probe_body_prompt`) every N steps at `probe_size` with the turbo LoRA (cheap enough to do
often), scores them, and records `likeness.jsonl` plus a Monitor chart. Scores are smoothed with `probe_ema`
and only changes beyond `likeness_noise_band` count. The controller then, each individually switchable:

- keeps `<job>_best.safetensors` at the best smoothed combined score (`likeness_track_best`) and stops after
  `likeness_patience` probes without improvement;
- hill-climbs a bounded learning-rate multiplier (`likeness_lr_control`, `likeness_lr_min/max/step`): up while
  the smoothed score rises, down when it falls. It works with AdamW-style optimizers (through the scheduler)
  and with Automagic 3 (by scaling its adaptive lr tensors), which keeps doing its own per-step adaptation;
- applies a face/body curriculum (`likeness_curriculum`): when the face score lags the body score, close-up
  dataset images (largest face >= 3% of the image, classified once with the face detector) are sampled more
  often and the timestep distribution shifts toward lower noise where facial detail forms; the reverse when the
  body lags. Strength 0..1.

The scores never enter the loss; they steer data, lr and stopping. Use the same reference set and prompts for a
whole run so the series is comparable.

## Sampling with a different model

`sample_model_path` (a `.safetensors` file, a diffusers folder or a repo id) plus `sample_model_quant` use another
checkpoint for sample images only, for example training on Base while previewing on SFT. The current LoRA weights
are copied onto it before each sampling pass, the turbo LoRA (if any) is applied there, and the training model is
moved to system RAM for the duration (so RAM must hold both models; quantize the sampling model if tight).

## Turbo LoRA for samples

`turbo_lora` + `turbo_lora_strength` attach a ComfyUI-format LoRA (for example an 8-step distill) that is applied
only while generating sample images, never during training steps. Set `sample_steps` to match. `sample_lora` +
`sample_lora_strength` stack a second sampling-only LoRA (for example a style LoRA) alongside the turbo one.

## Dataset page

The WebUI's Dataset tab lets you upload images and `.txt` captions into a folder, shows them in a grid, and
writes caption edits back to the `.txt` files as you type.

## FIBO 2.0 (Bria): training on dev, using the LoRA on Edit Turbo

RulzTrainer also trains LoRAs for [briaai/Fibo-2.0-dev](https://huggingface.co/briaai/Fibo-2.0-dev), Bria's 8B
single-stream DiT (44 blocks, Qwen3-VL-4B text encoder, the same Flux2 VAE as MOSS-Image). Pick **FIBO-2.0-dev** as
the model, or any diffusers folder / repo id whose transformer is a `BriaFibo2Transformer2DModel` (the architecture is
detected from the model; `model_arch: auto`). The repo is gated: accept its terms on Hugging Face and `hf auth login`.

- **Train on dev, use on Edit Turbo.** [Fibo-2.0-edit-turbo](https://huggingface.co/briaai/Fibo-2.0-edit-turbo) is
  distilled from dev with a byte-identical transformer config, so a LoRA trained on dev loads onto it unchanged. Bria
  says Turbo itself is not meant for fine-tuning.
- **Captions** are plain `.txt` sentences or structured VGL `.json` captions next to each image (the `.json` wins). A
  JSON caption is cleaned the way the model was trained (empty values and score keys dropped, compact form), and the
  trigger word goes into its `short_description`. Caption dropout trains on FIBO's null caption.
- **Memory.** The bf16 transformer is 16 GB, so `transformer_quant: auto` quantizes it to int8 (W8A8) on GPUs under
  22 GB; set `none` for bf16 on a larger card. The text encoder (8.9 GB bf16) is loaded for caching only and unloaded
  afterwards (`text_encoder_quant: int8` halves it). Measured on a 16 GB RTX 4060 Ti with int8 + W8A8, rank 16 on
  the six block groups, gradient checkpointing, batch 1:

  | Resolution | Peak training memory | Step |
  |---|---|---|
  | 512 px | 8.9 GB | 2.6 s |
  | 768 px | 9.7 GB | 5.7 s |
  | 1024 px (1 MP) | 11.0 GB | 10.9 s |

  Loading and quantizing the transformer takes about 30 s; a 512 px preview of 12 steps with CFG takes about 10 s.
- **LoRA targets.** The six usual groups map onto FIBO's `to_q/k/v`, `to_out`, `w1/w3` and `w2`; FIBO adds
  `cross_attn` (the five text-injection blocks), `adaln`, `refiner`, `perceiver` and `embed`. Bria's own trainer adapts
  every Linear layer (all of these); the default here is the six block groups.
- **Timesteps.** `timestep_sampling: reference` uses Bria's recipe (sigma ~ sigmoid(N(shift, 1)) with the shift set by
  the image size, 10% uniform); the other modes work as for MOSS. Previews use the pipeline's schedule (size-shifted
  sigmas, CFG as a second pass with the null caption); Bria samples dev at 50 steps / CFG 5, 20-30 steps is fine for
  previews. The MOSS-only sample shift and skip layer are ignored.
- **Output format.** FIBO LoRAs are saved in the diffusers / PEFT format (`transformer.<module>.lora_A/B.weight` plus
  the adapter config in the file metadata), since ComfyUI has no FIBO 2.0 support yet:

  ```python
  import torch
  from diffusers import BriaFibo2Pipeline  # diffusers main (PR #14991)
  pipe = BriaFibo2Pipeline.from_pretrained("briaai/Fibo-2.0-edit-turbo", dtype=torch.bfloat16).to("cuda")
  pipe.transformer.load_lora_adapter("output/<job>", weight_name="<job>.safetensors")  # the pipeline has no load_lora_weights
  edited = pipe(prompt=vgl(scene), image=source, num_inference_steps=4, guidance_scale=1.0).images[0]
  ```

- **Not yet:** edit-pair training (source + target + instruction), the `text_render` text images, single-file
  (ComfyUI) checkpoints, and a separate turbo LoRA for previews (Bria ships none; use fewer steps instead).

**Loading the LoRA in diffusers.** `BriaFibo2Pipeline` has no `load_lora_weights`; load through the transformer as
above. On a 16 GB card the bf16 transformer needs offloading (`pipe.enable_sequential_cpu_offload()`): load the LoRA
*before* enabling it. Loaded afterwards, the adapter weights miss the offload hooks and the output is noise or black
(diffusers' pipeline-level loader works around this; the transformer-level one does not).

**Verified** (12 photos of one person, 512 px, rank 16, lr 2e-4, `reference` timesteps, int8, one RTX 4060 Ti):

| | Face likeness (ArcFace vs 3 reference photos) |
|---|---|
| Probes during training, steps 0 / 50 / 100 / 150 / 200 / 250 | 13.5 / 36 / 46 / 58 / 58 / 61 % |
| diffusers `BriaFibo2Pipeline` on **dev**, text-to-image, base vs LoRA | 15 % vs **65 %** |
| diffusers on **Edit Turbo**, re-pose edit of another person's photo, base vs LoRA | 12 % vs 22 % |
| diffusers on **Edit Turbo**, "make the person smile" edit, base vs LoRA | 9 % vs 14 % |

The LoRA file loads into diffusers' transformer without warnings, and with the same LoRA the diffusers model and
RulzTrainer's produce bit-identical outputs. Note what an edit does, though: Edit Turbo conditions on the source
image, so a text-to-image LoRA shifts an edit toward the trained identity only partly; it does not replace the face
in a photo of someone else. Replacing identities in edits needs edit-pair training (not supported yet). Generation
with the LoRA (dev, or Turbo given no source image) carries the full identity.

## Krea 2: training on Raw, using the LoRA on Turbo

RulzTrainer trains LoRAs for [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw), Krea's 12B single-stream DiT
(28 blocks with grouped-query attention, a Qwen3-VL-4B text encoder whose twelve tapped layers are fused per token, and
the 16-channel Qwen-Image VAE). Pick **Krea-2-Raw** as the model, or any diffusers folder / repo id whose transformer
is a `Krea2Transformer2DModel` (detected from the model). The repo is gated: accept the Krea 2 Community License on
Hugging Face and `hf auth login`. The download skips the repo's 26 GB single-file copy and sample images (about 35 GB).

- **Train on Raw, use on Turbo.** Krea publishes Raw as the checkpoint to train on and uses LoRAs trained on it with
  [Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo), the 8-step distill (its own
  [Krea-2 LoRAs](https://huggingface.co/collections/krea/krea-2-loras) work that way). Set `sample_model_path:
  krea/Krea-2-Turbo` with `sample_steps: 8` and `sample_cfg: 1` to preview (and probe) the LoRA as it will be used;
  Turbo is gated separately, so accept its license too. Sampling Raw needs 28-52 steps with guidance.
- **Guidance.** `sample_cfg` uses the same convention for every model (1 = off). Krea's `guidance_scale` g is
  `sample_cfg` g + 1 here: Krea's 3.5 for Raw is 4.5.
- **Memory.** The transformer is 25 GB in bf16, 13.5 GB in int8, about 8 GB in nvfp4. `transformer_quant: auto`
  counts the memory the model will live in, all GPUs of a model split together: bf16 from 40 GB, int8 (W8A8) from
  22 GB, else nvfp4. One 16 GB card therefore trains in nvfp4 (int8 plus the VAE already fill 14.2 GB, with no room
  left for training activations at 1 MP); two 16 GB cards with `multi_gpu: model_split` train in int8, the model
  loaded on the CPU and its blocks placed directly on both cards. Split in int8 over an RTX 5060 Ti + RTX 4060 Ti:
  10.0 s per step at 1 MP, peaking at 11.5 / 10.5 GB, with int8's smaller error (weights only, under 1%).
- **Text features** follow `Krea2Pipeline`: the prompt in Krea's system/user template, twelve Qwen3-VL hidden layers
  per token. The pipeline pads every prompt to 512 tokens; padding is never attended to and text tokens have no
  position in the DiT, so the cache stores only the real tokens (a few MB per caption) with the same result.
- **Latents** come from the Qwen-Image VAE (16 channels, 8x), normalized with its per-channel mean/std and packed 2x2
  into 64 channels, so they are cached under their own file names (`_qwenvae`); region maps are shared with the
  other models (16-pixel cells either way).
- **LoRA targets**: the usual groups map onto `attn.to_q/k/v`, `attn.to_out.0`, `ff.gate/up`, `ff.down`; Krea 2
  adds `attn_gate` (the sigmoid output gate of every attention), `text_fusion` and `embed`. Krea's own LoRAs adapt all
  of them (rank 32).
- **Export** is the diffusers / PEFT format with the same module names as Krea's LoRAs
  (`transformer.transformer_blocks.<n>.attn.to_q.lora_A.weight`), with the adapter config (alpha) in the metadata.
  Loaders that ignore the metadata assume alpha = rank, so keep alpha equal to rank (the default) for those.
- **Timesteps**: `timestep_sampling: reference` is a logit-normal shifted like the sampler (Krea has not published its
  training distribution); the default `logit_normal` works as for the other models.

```python
from diffusers import Krea2Pipeline  # needs peft for load_lora_weights
pipe = Krea2Pipeline.from_pretrained("krea/Krea-2-Turbo", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("output/<job>/<job>.safetensors")
image = pipe("<trigger>, ...", num_inference_steps=8, guidance_scale=0.0).images[0]
```

Verified against diffusers: `tests/test_krea2.py` checks the port against `Krea2Transformer2DModel` (tiny random
weights; also the compact text layout, the latent packing, the schedule against `FlowMatchEulerDiscreteScheduler`, and
an exported LoRA merged into diffusers' weights). With the real Raw weights in bf16 the port matches diffusers' model to
bf16 rounding (cosine 0.99999 over a 512x512 prediction), and the compact prompt features are as close to the
pipeline's (cosine 0.99989) as the pipeline is to itself at another padding length.

## Using the LoRA in ComfyUI

Recent ComfyUI builds support MOSS-Image natively (`comfy/ldm/moss_image`, CLIP type `moss_image`), so no
plugin is required. Copy the trained `.safetensors` into `ComfyUI/models/loras/` and add a standard
**Load LoRA** node after *Load Diffusion Model*. Both the unfused checkpoints and ComfyUI's repacked
fused-qkv files work: ComfyUI maps the `proj.q/k/v` LoRA keys onto slices of the fused weight.

The file uses the generic ComfyUI key format (the same one ComfyUI's own MOSS-Image LoRAs use):

```
diffusion_model.layers.<n>.proj.<q|k|v|o|ffn_up_gate|ffn_down>.lora_down.weight
diffusion_model.layers.<n>.proj.<...>.lora_up.weight
diffusion_model.layers.<n>.proj.<...>.alpha
```

These names are identical in the diffusers checkpoint and the ComfyUI plugin model, so no key conversion is needed.

## VRAM guidance (16 GB cards)

| Setting | Recommendation |
|---|---|
| Gradient checkpointing | on |
| Batch size | 1 (use gradient accumulation instead) |
| Optimizer | `adamw8bit`, or `automagic3` (self-tuning lr, ported from ai-toolkit) |
| Resolution | 1024 |
| Text encoder device | `auto` (spreads over all GPUs + CPU during caching only) |
| Max caption tokens | 512 to 1024 unless you train on very long prompts |

The DiT in bf16 is about 10 GB. The text encoder is about 18 GB in bf16, so it is loaded only for caching
with `device_map="auto"`; pick `8bit` / `4bit` quantization if you want it on a single GPU.

## Layout

```
rulztrainer/
  config.py        TrainConfig dataclass + YAML
  dataset.py       scanning, bucketing, latent/text caching, batching
  lora.py          LoRA injection, ComfyUI-format export/import, sampling-only extra LoRAs
  quant.py         int8 / int8_convrot / nvfp4 weight quantization matching comfy_kitchen, QuantLinear
  sampler.py       flow-matching Euler sampler (reference schedule)
  trainer.py       training loop, saving, sampling, status.json for the UI
  train.py         CLI entry point
  moss/model.py    MOSS-Image DiT (vendored, Apache-2.0, OpenMOSS)
  moss/loading.py  download + load (HF layout or single files), quant dispatch, encode helpers
  moss/single_file.py  ComfyUI single-file loaders (dequant, unfuse qkv, text-only Qwen3.5)
  archs.py         the model families (MOSS-Image, FIBO 2.0, Krea 2) behind one interface
  fibo/            Bria FIBO 2.0: DiT port, loading, VGL prompts, sampler
  krea2/           Krea 2: DiT port, loading (Qwen-Image VAE, Qwen3-VL layer stack), sampler
webui.py           NiceGUI web UI (runs training as a subprocess)
configs/           example configs
tests/             smoke tests (no weights needed): python -m pytest tests
install.bat / run_webui.bat   Windows install and start;  install.sh / run_webui.sh   the same for Linux
scripts/vast/      vast.ai: provision.sh (PROVISIONING_SCRIPT), supervisor wrapper + conf, create_template.sh
Dockerfile         optional vast.ai image with the install baked in
```

## Adding another model

Model families are `Arch` classes in `rulztrainer/archs.py` (MOSS-Image, FIBO 2.0 and Krea 2 today). An Arch provides
the loading functions (transformer, VAE, text encoder), `encode_prompt` / `pad_embeds`, `encode_images` /
`decode_latents` for its VAE, a sampler, the LoRA module names for the target groups, the export format, and its
reference timestep recipe. A family with a VAE of its own sets `latent_tag`, so its latents are cached under their own
file names. The transformer must take the trainer's convention, `model(latents, sigma * 1000, embeds, mask)` with the
normalized 2x2-patchified latents [B, C, H/16, W/16], and expose `.layers` (the blocks) for block selection, compile
and the model split, plus `split_tail_modules()` (what runs after the last block); `rulztrainer/krea2/` is the most
recent template, and `tests/test_krea2.py` checks a port against the diffusers model with tiny random weights.
