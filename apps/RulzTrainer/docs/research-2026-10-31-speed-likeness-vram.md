# RulzTrainer research: speed, character likeness and VRAM

2026-10-31. Scope: cutting-edge, rarely used and new ideas to make RulzTrainer LoRA training faster, reach better character likeness, and use less VRAM without losing speed or quality. Extra depth on Krea 2, Qwen-Image 2.x, Flux 3 Klein, MOSS-Image and MiniMax H3. Ideas that only pay off on bigger rented vast.ai cards are included and marked as such.

## How this was made, and how to read it

- **Sweep.** 15 research scouts covered speed, optimizers, LoRA variants, likeness objectives, the reward stage, data, text encoders, VRAM, systems, and each of the five models. They returned 161 distinct findings.
- **Verification.** Every finding was checked twice: once against its cited sources (do they exist, say what is claimed, and transfer to LoRA on a frozen quantized DiT at batch 1), and once against the RulzTrainer code (is it already there, how does it interact with W8A8/nvfp4, compile, keep blocks, block swap, data_parallel, races and probes). Near-duplicates from different scouts were merged into 83 groups first. Several verifiers ran small benchmarks or offline measurements on the user's machine and on Krea 2's real weights; those numbers are marked as measured.
- **Gap sweep.** A completeness critic found four uncovered areas (evaluation blind spots, statistics of races, deployment compatibility, data attribution); scouts covered them and each new finding was verified the same way.
- **Result.** 75 of 83 groups and all 21 gap findings survived; 8 groups and 9 individual members were dropped (appendix A). Almost every surviving claim came back "weakened": the idea is real, but its headline number was measured on LLMs, pretraining, UNets, big batches or datacenter GPUs. The sections use the verifiers' corrected numbers. **Unmeasured** means nobody has measured it for this workload.
- **Verdicts.** *Do it* (cheap and safe), *Race it* (worth an A/B with the probes), *Prototype* (bigger build, promising), *Watch* (paper only or not ready), *Skip-ish* (kept but marginal). Confidence is the verifiers' (low < 0.4, medium 0.4-0.65, high > 0.65).
- Workload assumed throughout: character LoRAs, 0.5-1 MP, batch 1, rank 8-16, 1000-3000 steps; local RTX 5060 Ti 16 GB + RTX 4060 Ti 16 GB on Windows; Krea 2 nvfp4 ~9 s/step, MOSS int8 ~5 s/step, Qwen 2.1 int8 ~2.5 s/step at 0.5 MP.

## Executive summary

The biggest risk to both speed and likeness work is not a missing trick but **decision noise**: probes use a few seeds and a turbo sampler, races drop racers on single readings, and one recognizer (ArcFace) drives best-checkpoint, early stop, races and the reward stage. Fixing that first makes every later race cheaper and more trustworthy. After that, the largest per-step speed levers are **training fewer blocks** (identity-block windows plus not backpropagating through blocks that carry no LoRA) and **dropping tokens in the middle blocks** (TREAD), each worth an estimated 10-50% per step but unmeasured on DiT LoRAs. Optimizer, schedule and LoRA-variant ideas mostly land inside probe noise on this workload. For new models, **Flux 3 Klein** is the strongest addition (estimated about twice as fast per step as Krea 2 at 1 MP, ~5-7 days to port, mostly reused code); **MiniMax H3** is portable (~2-3 weeks) but runs best on 32 GB+ vast.ai cards; its licence excludes the US, EU, UK and South Korea without MiniMax's approval, so check it first, and run a cheap pre-port gate (an ai-toolkit H3 LoRA vs your Krea 2 LoRA, scored with your probes) before building anything.

### Top 10

| # | What | Verdict | Expected gain | Conf. | Cost | Section |
|---|---|---|---|---|---|---|
| 1 | Trustworthy probes and races: measure probe noise, paired race decisions on shared seeds, a held-out "gold" face recognizer (FaceNet), an SSCD copy check | Do it | fewer wrong drops and wrong best checkpoints; guards against reward hacking (unmeasured) | high/med | small each, 0 s/step | [Evaluation, probes and races](#evaluation-probes-and-races) |
| 2 | Distilled-base fix: `turbo_lora: auto` stacks the Raw-to-Turbo LoRA on a distilled Turbo base (comfy_files.py:42-43, trainer.py:2146-2175); then race training on Krea 2 Turbo with the Ostris training adapter, and re-measure the Qwen 2.1 adapter on/off | Do it (fix) / Race it | probes become true deployment renders; likeness unmeasured | med | fix: a few lines; races: no code | [Krea 2](#krea-2), [Likeness from data, teachers, references and Turbo transfer](#likeness-from-data-teachers-references-and-turbo-transfer) |
| 3 | Train fewer blocks, and skip the backward through LoRA-free blocks: Qwen 2.1 `lora_blocks 10-14` race, a static mid-block window, then suffix pass-through | Race it (no code first) | est. 2.5 -> 1.3-1.9 s/step on Qwen 2.1; 20-30% wall clock Krea/Qwen; likeness unmeasured | med/low | 0 -> ~1 day | [Qwen-Image 2.x](#qwen-image-2x), [VRAM](#vram-without-losing-speed-or-quality), [LoRA parameterization](#fewer-steps-to-likeness-lora-parameterization-initialization-timestep-and-noise-sampling) |
| 4 | No-code speed checks: cuDNN attention on the 4060 Ti; fp4 backward on Krea 2 nvfp4; persistent compile cache on vast.ai | Do it / Race it | ~6-8% on Ada at 1 MP (kernel bench); 8.73 -> 7.93 s/step Krea fp4 backward (measured); ~160 s per fresh rental | med | none to 0.5 day | [Speed per step](#speed-per-step), [Systems and two GPUs](#systems-and-two-gpus) |
| 5 | TREAD token routing (skip ~half the image tokens through the middle blocks) | Prototype | 10-25% step time (unmeasured; may be 0 on some configs) | med | 3-5 days | [Speed per step](#speed-per-step) |
| 6 | Probes off the critical path: render probes asynchronously on the idle second GPU, and calibrate the turbo probe engine per adapter | Prototype / Do it | 11-16% wall time (measured on Qwen 2.1 int8); 50-75% per probe render | med | medium | [Evaluation](#evaluation-probes-and-races), [Speed per step](#speed-per-step) |
| 7 | Add Flux 3 Klein | Prototype | est. 4-5 s/step nvfp4 1 MP on the 5060 Ti vs ~9 s Krea 2 (unmeasured) | high | ~5-7 days | [Flux 3 Klein](#flux-3-klein) |
| 8 | Fuse the LoRA up-projection into the W8A8/NVFP4 rescale epilogue | Prototype | 3-9% per step, likeness-neutral | med | 3-4 days | [Speed per step](#speed-per-step) |
| 9 | Measure the nvfp4-vs-int8 likeness gap, and check the LoRA after ComfyUI's merge-and-requantize | Race it / Do it | decides whether recovery adapters and format ramps are worth building; a few ArcFace points may be lost on nvfp4 deploy (unmeasured) | low/med | race: no code | [VRAM](#vram-without-losing-speed-or-quality), [Deployment](#deployment-compatibility-and-data-attribution) |
| 10 | Reward-stage upgrades: high-sigma moves, flat-group skip, per-scorer advantages; then a pathwise ArcFace reward through the last turbo steps | Race it / Prototype | -1/4 to -1/3 backward at cfg > 1; est. +0.02-0.06 ArcFace over the regular-stage best (partly metric gaming) | med | small / medium | [Likeness objectives and the reward stage](#likeness-objectives-and-the-reward-stage) |

**On rented vast.ai cards** the picture changes: on 48 GB+ cards the biggest lever is simply keeping every block (checkpointing off), which Fizgig measured at 13-27% per step on A100 80 GB, A40 and RTX PRO 6000, and RulzTrainer's 8% memory margin leaves 7.7-11 GB unused on 96-141 GB cards. A 96 GB card is also the place to settle, once, how much likeness the int8 and nvfp4 bases cost against bf16. Batch > 1 still does not pay (tok/s stays flat), and NCCL data_parallel is not worth building for character LoRAs; on multi-GPU boxes run one race configuration per GPU. Per-card settings are in [Bigger hardware on vast.ai](#bigger-hardware-on-vastai).

Close behind: pair-aware Muon for MOSS (S2a), a warmup-stable-decay schedule with a cooldown when probes stop a run (S2a), stratified timestep sampling (S2b), smaller kept-block activations (S5), the MiniMax H3 port on 32 GB+ cards (M5), and the per-card vast.ai profiles (S9).

### Things the verifiers found in RulzTrainer itself

- **Latent bug:** with a distilled Turbo base, `turbo_lora: auto` still stacks the Comfy Raw-to-Turbo LoRA (comfy_files.py:42-43, trainer.py:2146-2175). The same guard is needed for MiniMax H3 and a distilled Flux 3 Klein.
- **Unsourced figures:** the Fizgig "2-7 points" text-fusion claim (README.md:1612-1613, config.py:24-26) and the Qwen 2.1 training-adapter likeness figures 77/56/76 (config.py:55-57, README.md:1663-1666) could not be traced to any public source. Re-check or re-measure them.
- **Your MOSS config differs from the defaults** in two places nothing in the literature covers: `int8_backward: int8` (default bf16) and the `moss-8step-v2` turbo LoRA at 0.7. The MOSS section proposes a parity race for exactly these.
- **Already there (rejected as duplicates):** Krea 2 per-bucket shift with Krea's mu formula and the Turbo pin, the Qwen 2.1 "texture collapse" levers, the flip policy for faces, per-token HALO-1 int8 backward, prefix-free cuDNN masking.

## Experiment queue

Races and benches that need **no new code**, in suggested order. Use 2 seeds per arm unless stated, and the existing probe noise band as the bar.

| # | Name | Arms (diff vs current defaults) | Metric | Stop / adopt rule |
|---|---|---|---|---|
| 1 | `ada_cudnn` (bench) | 4060 Ti; `attention_backend: torch` vs `cudnn`; Krea 2, MOSS int8, Qwen 2.1 int8 at 0.5 and 1 MP; scripts/bench.py 20 steps | median s/step at equal loss | adopt cuDNN as the Ada auto if >= 5% faster |
| 2 | `probe_noise` (bench) | re-probe one finished LoRA (e.g. krea2-AZav-rulz) 16 seeds x current probe prompts, plus the trigger-form A/B: "{trigger}, close-up portrait photo..." vs preset "close-up portrait photo of {trigger}..." | ArcFace / DINO mean and spread | sets the real noise band; race trigger form only if the gap >= 0.01 |
| 3 | `turbo_probe_check` | re-probe the checkpoints of 2-3 finished MOSS runs at full quality (no turbo, 30 steps, CFG 3) vs `moss-8step-v2` at 0.7; same for a Krea run | Spearman of probe scores across checkpoints | keep turbo probes if rho >= 0.9 and the best checkpoint matches |
| 4 | `qwen_fast_identity` | Qwen 2.1 int8, 0.5 MP, rank 8, lr 2e-4, adapter auto, 2000 steps; `lora_blocks` "" vs "10-14" vs "8-16" | best combined likeness, s/step, skin speckle at full sampling | adopt if within 2 points of all-blocks, >= 1.25x faster, no speckle |
| 5 | `qwen_adapter` (separate runs) | `training_adapter` fizgig vs "" vs fizgig at strength 0.5 | peak ArcFace, step to 50% of peak, DINO body | 3 probes without gain or 2000 steps |
| 6 | `krea_turbo_base` (separate runs) | A: Raw + `turbo_lora: auto`; B: `model_path` Krea-2-Turbo + Ostris turbo training adapter + `turbo_lora: ""`; nvfp4, rank 16, 3000 steps | peak ArcFace on Turbo 8 steps CFG 1; step to 90% of peak | B wins if +2 points or 20% sooner on 2 datasets |
| 7 | `fp4_backward` (separate runs) | Krea 2 nvfp4, rank 16, 2000 steps; `nvfp4_backward` bf16 vs fp4 | best/final ArcFace + DINO, s/step | fp4 becomes the Blackwell default if within the bf16 seed spread |
| 8 | `nvfp4_gap` (separate runs) | Krea 2 `transformer_quant: nvfp4` on one card vs `int8` + `model_split` (or bf16 on a 32 GB+ vast card); AZav50, 1500-2000 steps | best ArcFace + DINO, final checkpoints re-scored on the int8 base | if int8 does not lead by more than the seed spread, drop recovery-adapter and format-ramp work |
| 9 | `moss_convrot_parity` (separate runs) | MOSS 0.5 MP rank 16 1500 steps: bf16 base vs int8_convrot `int8_backward: bf16` vs int8_convrot `int8_backward: int8` (your setting) | ArcFace/DINO every 250 steps, smoothed loss | parity if final ArcFace within ~3 points and loss within 2% |
| 10 | `moss_shift3` | `timestep_shift` 1.0 vs 3.0 (diffusion-pipe's MOSS example) | ArcFace + DINO every 250 steps | adopt only if it beats shift 1 by more than one-seed noise on both seeds |
| 11 | `idloss_window` | Qwen 2.1 int8, `id_loss_backend: facenet`; `id_loss_weight` 0 vs 0.2 with `id_loss_sigma_max` 0.35 vs 0.2 with 0.7 | ArcFace face, DINO body must not drop | winner must beat the others by > 2x noise band on 2 seeds |
| 12 | `schedule_and_soup` | `lr_scheduler` constant vs cosine vs linear (Qwen 2.1 int8 1500 steps); then `save_every 100` + soup stage over the last 3/5/10 checkpoints vs `_best` | ArcFace + body at the final step | keep a decay or a soup only if > 2x noise band on most runs |
| 13 | `lowres_rank_check` | same 4-racer race at the low bucket only vs normal 0.5-1 MP | Kendall tau of racer order | build low-res race rounds only if tau >= 0.6 in 3 of 4 races |
| 14 | `precision_gate` (vast.ai RTX PRO 6000 96 GB) | Krea 2 `transformer_quant` none vs int8_convrot vs nvfp4; 1 MP; all blocks kept; 2 seeds; 1500-2000 steps | ArcFace + DINO, re-scored on bf16 and int8 bases | if bf16 and nvfp4 best ArcFace are within the seed spread, train races in nvfp4 and stop the recovery work |
| 15 | `h100_int8_bwd` (vast.ai H100) | int8_convrot; `int8_backward` bf16 vs int8; all blocks kept; 1500 steps | probe curve + s/step | adopt int8 backward on Hopper if within seed spread and >= 10% faster |

Once code exists, the next races are: suffix pass-through, TREAD `tread_keep` 1.0 vs 0.6, the paired race-drop rule, the reward-stage changes, stratified timesteps, and Flux 3 Klein's block/shift races.

## Research ideas nobody has tried

Hypotheses with no published test on this kind of workload; each is a candidate for a RulzTrainer experiment or a write-up of its own.

- **Probe statistics:** paired tests on shared probe seeds (common random numbers) cut wrong race drops; the noise of face and body scores on DiT renders has never been measured.
- **Cooldown on demand:** when the probes or a race stop a run, a short (1-sqrt) learning-rate cooldown from that step beats ending at full lr.
- **Pair-aware Muon on a diffusion LoRA,** starting with MOSS, whose base was pretrained with Muon.
- **Suffix pass-through:** exact forward with an identity gradient through untrained suffix blocks keeps likeness within a point while cutting step time 1.5-2x.
- **Low resolution as a race fidelity rung** preserves racers' likeness ranking.
- **Pathwise identity reward** through the last turbo steps of a flow DiT image LoRA (shown only for video and SD adapters).
- **Self-training at high sigma only:** training selftrain renders only above a sigma floor avoids guidance inflation from synthetic images.
- **Training-time null-space constraint** against Turbo and style LoRAs improves stacked likeness.
- **Per-bin probe gaps** tell the user which pose, expression or lighting to shoot next.
- **fp32 residual stream for MOSS** training (bf16 rounding in its massive channels may degrade LoRA deltas).
- **Frequency-aware caption dropout** on sentence captions and LLM text encoders stops identity parking in hair or ethnicity words.
- **Flux 3 Klein:** a Base-trained LoRA transfers to the 16-step flow-map checkpoint (stated nowhere); a flow-map-consistent distillation of Base+LoRA carries identity better than applying the Base LoRA at 0.75.
- **Self-reference identity tokens** (a clean reference face in the reference stream at t = 0) raise trigger-only likeness of a single-person LoRA.

Each section lists its own paper-only ideas in more detail.

## Contents

Themes: [Speed per step](#speed-per-step) · [Fewer steps to likeness: optimizers and learning-rate schedules](#fewer-steps-to-likeness-optimizers-and-learning-rate-schedules) · [Fewer steps to likeness: LoRA parameterization, initialization, timestep and noise sampling](#fewer-steps-to-likeness-lora-parameterization-initialization-timestep-and-noise-sampling) · [Likeness objectives and the reward stage](#likeness-objectives-and-the-reward-stage) · [Likeness from data, teachers, references and Turbo transfer](#likeness-from-data-teachers-references-and-turbo-transfer) · [Text side](#text-side) · [VRAM without losing speed or quality](#vram-without-losing-speed-or-quality) · [Systems and two GPUs](#systems-and-two-gpus) · [Evaluation, probes and races](#evaluation-probes-and-races) · [Deployment compatibility and data attribution](#deployment-compatibility-and-data-attribution) · [Bigger hardware on vast.ai](#bigger-hardware-on-vastai)

Models: [Krea 2](#krea-2) · [Qwen-Image 2.x](#qwen-image-2x) · [Flux 3 Klein](#flux-3-klein) · [MOSS-Image](#moss-image) · [MiniMax H3](#minimax-h3)

Appendices: [A. Rejected ideas](#appendix-a-rejected-ideas) · [B. Sources](#appendix-b-sources)

## Speed per step

This section covers ideas that cut the wall-clock time of one training step (or of the renders around it: probes, samples, reward rollouts) on the 5060 Ti / 4060 Ti pair, without giving up likeness. Verdict: two ideas move the step by double digits or close to it. Token routing (TREAD) is the largest untouched lever (10-25%, unmeasured on these models, with a likeness risk). Fusing the LoRA branch into the W8A8/NVFP4 epilogue is the safest (5-8%, partly measured, likeness-neutral). INT8 attention and cuDNN-on-Ada go after the attention share, which the README measures at 42-50% of compiled GPU time at 1 MP. The rest are small, already mostly in place, or quality items with no speed gain. One member (HALO-2 int8 backward) is refuted.

### TREAD token routing, with SPRINT's knobs (P21 + P22)

**Prototype. Expected gain: 10-25% lower step time at a 0.4-0.5 drop over the middle blocks (unmeasured on these models; may be zero or negative on some LoRA configs). Confidence: medium (routing) / low (SPRINT knobs). Cost: ~3-5 days for Krea 2 + Qwen 2.1 + MOSS + FIBO.**

**What it is.** At block `start`, a subset of image tokens is gathered out of the sequence (text tokens always kept). Blocks `start..end` run on the shorter sequence. At block `end` the processed rows are scattered back into the saved full residual, and the routed tokens carry their block-`start` value unchanged. The loss stays over all tokens. Nothing new is trained, and routing is off at inference. The saving is linear in FFN/projections and quadratic in attention, in forward, recompute and backward. SPRINT (P22) adds three design knobs to the same feature: structured selection (one token per 2x2 group instead of random holes), dense first/last blocks, and a short full-token tail.

**Evidence.**
- [TREAD paper (arXiv 2501.04765)](https://arxiv.org/html/2501.04765): 3.03 it/s vs DiT 1.86 under identical conditions (4x A100 80GB, global batch 256, ImageNet-256). The "1.63x" is 3.03/1.86 computed by the scout, not a stated result. Unguided FID 3.93 at 680 epochs vs DiT 9.62 at 1400. Ablation: 0.75 selection rate is worse than no routing (FID 49.29 vs 43.47 on B/2). The guided setting ends with a 75K-iteration unrouted fine-tune at batch 1024. From scratch only; LoRA and pretrained adaptation are not studied. The 14x convergence claim does not transfer.
- [SimpleTuner TREAD.md](https://github.com/bghira/SimpleTuner/blob/main/documentation/TREAD.md): `tread_config.routes[{selection_ratio, start_layer_idx, end_layer_idx}]`, negative indices allowed. Presets: balanced 0.5 @ 2->-2, high-res 0.6 @ 2->-3, LoRA example 0.4 @ 3->-4, max-speed 0.7 @ 1->-1 ("massive loss spike"). "Typically 20-40%", no GPU, model or batch named. LoRA/LoKr see a larger initial loss spike, and "some LoRA configurations show slight slowdowns". Works with int8/int4/NF4 bases. Masked-loss tokens are force-kept (lower speedup). L1-norm ranking mixed with randomness. The doc is internally inconsistent on ratio semantics: it calls 0.5 balanced but also says >0.35 "will definitely affect convergence". Cite that as "docs are inconsistent", not as a threshold.
- [SimpleTuner FLUX2 quickstart](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/FLUX2.md): routes 0.5 @ 2->-2 and 0.5 @ 4->-4, "Expected speedup: 20-40% depending on configuration". The int8-quanto and gradient checkpointing on that page are general memory options, not part of the TREAD example.
- [SimpleTuner README](https://github.com/bghira/SimpleTuner): Krea2, Qwen Image and MiniMax H3 are supported. TREAD for Krea2 came in PR #2800 (June 2026), and Qwen Image needed a fix PR (#2459, Jan 2026). Neither PR reports benchmarks. MiniMax H3 TREAD is "not supported" only in sparse-attention mode. Flux.2 Klein sits inside the Flux.2 entry. MOSS-Image and FIBO have no prior art.
- [Fizgig MINIMAX_H3.md](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/MINIMAX_H3.md) (scout variant, not separately verified): TREAD is on for H3 LoRA runs.
- [SPRINT (arXiv 2510.21986)](https://arxiv.org/html/2510.21986): first 2 and last 2 blocks dense, the middle on 25% of tokens (one random per 2x2 group). Drop-ratio ablation (SiT-B/2 FID): 0% 54.1, 25% 43.2, 50% 32.3, 75% 27.5, 87.5% 50.2. Structured 27.5 vs random 30.1. 20K fine-tune steps recover >94% of the 200K gain. 8x A100 bf16: 5.2 vs 2.5 it/s (256 px) and 2.01 vs 0.79 it/s (512 px). Batch-32 VRAM 19.6 vs 29.6 GB and 37.9 vs 77.7 GB. All of it is from-scratch class-conditional SiT with a trained channel-concat fusion and a [MASK] filler, which a frozen base does not have. The "ICLR'26" venue and a code repository were not confirmed.
- [SR-DiT (arXiv 2512.12386)](https://arxiv.org/html/2512.12386v1): adding SPRINT routing at 0.75 cut FID 10.56 -> 4.58 at 400K iterations (8x H200, ~10 h), on a REG + INVAE SiT-B/1 baseline. From scratch.
- Not transfer: every number above. The 34-51% VRAM figure is batch 32 with full backprop. At batch 1 with checkpointing, the verifiers expect 0.3-1.5 GB.

**RulzTrainer today.** Nothing routes tokens (grep finds no tread/token drop/sprint). Every block loop runs the full [text + H*W] sequence: krea2/model.py:328-338, qwen21/model.py:337-348 (block-causal attention at :117-131), moss/model.py:218-228 (per-sample loop, image first, text last), fibo/model.py:408. Savings today are by block (lora_blocks: no backward before the first trained block, README:235-237; checkpointing.py) and by precision (quant.py). regions.py:215-244 already holds per-item face_cells and a background prior on the latent grid. Delta: token-level sparsity inside the trained window.

**Fit and risks.**
- W8A8 / nvfp4: the per-token quantizers and `torch._int_mm` / `_scaled_mm` are row-count agnostic, so the saving multiplies with the quantized kernels.
- torch.compile (trainer.py:1481, `dynamic=True`; compiling.py limits 64/4096): keep the kept count a per-bucket constant (rounded to 64). That adds at most one extra shape per bucket. The gather/scatter sits between `_run` calls, outside the compiled blocks.
- Keep blocks: routed blocks store 0.4-0.6x the activations, so `_calibrate_checkpointing` must run with routing on. It will then keep 1-3 more blocks. `choose_kept_blocks` assumes a uniform per-block cost (checkpointing.py:52); it still works, just suboptimally.
- Block swap: weights only, unaffected. model_split: the gather/scatter follows the existing `.to(bdev)` moves (krea2/model.py:331-335). data_parallel threads: one index set per replica.
- Alignment taps (alignment.py:82, reshaped at :200-203) and the Qwen attention tap (qwen21/model.py:127-129) assume the full grid. Restrict them to blocks outside the window, or scatter-fill first.
- Attention localization, id loss and region-weighted loss need face cells present: force-keep cells with region weight > 1.
- Per model: Qwen 2.1 must gather x, the key_mask columns and the [B, L, 3] positions (qwen21/model.py:283-294). Its text stream stays exact (text attends text only), so Qwen is the cleanest target. Krea shares cos/sin [L, ...] (krea2/model.py:321), fine at batch 1, and its sigmoid gate follows x. Krea's joint attention means its text tokens also see a sparser image. MOSS is the easiest (gather `video[:, keep]` and `rope[:, keep]`).
- Train/inference mismatch is carried only by the LoRA. Expect SimpleTuner's initial loss spike. Watch automagic's lr controller reacting to it (consider freezing lr adaptation for the first ~100 routed steps).
- The region-prior base forward (trainer.py:988-990), DOP/isolation, probes, samples and turbo renders must run unrouted.
- Export: no new parameters, no change. Windows: plain gather/index_select/scatter, no NCCL, no new triton kernel.
- Where lora_blocks already starts deep (e.g. 21-41 on MOSS), the gain shrinks.
- 2x2 grid selection needs even latent dims (32-px buckets). Blind 2x2 sampling drops most of a small face's cells, which is why a `face` mode built on face_cells fits the likeness workload better. SPRINT's 75% is unsupported for a frozen base. Skip its concat fusion and [MASK] token.

**Implementation path.**
1. New rulztrainer/routing.py: builds per-bucket keep indices on the latent grid, with modes `random | l1 | grid2x2 | face`. `face` keeps face_cells dense and drops background first. The force-keep set is region weight > 1.
2. config.py: `tread_keep` (float, 0 = off), `tread_start_block` (default 3), `tread_end_block` (default -4), `tread_tail_share` (default 0.1; SPRINT's 2.5% is too short for 1000-3000-step LoRAs, the verifiers suggest 10-20%), `tread_select`, `tread_dense_ends` (2), `tread_keep_regions` (true). Add all of them to race.py RACE_KEYS.
3. trainer.py: around the main forward (:975), set `rmodel._rz_tread = TreadPlan(...)` while `cur_step < (1 - tail_share) * total`. Clear it for the region-prior forward, probes, samples and DOP. Calibration runs with it on.
4. In each arch forward (krea2/model.py:328, qwen21/model.py:337, moss/model.py:218, fibo/model.py:408): gather at `i == start` and scatter at `i == end`, as listed under Fit.
5. README section, example.yaml, wiki.

**How to test.**
- Race 1: Krea 2 nvfp4, 1 MP, rank 16, 1500 steps, `tread_keep` 1.0 vs 0.6 (start 3, end -4, tail 0.1, regions kept), probes every 100. Run sequentially on the same GPU for clean timing.
- Plot ArcFace and DINO probe curves against seconds, not steps. Check the loss watch for the first-200-step spike.
- Accept only if likeness at equal wall-clock is at or above the unrouted run within probe noise.
- Then MOSS int8 and Qwen 2.1 at 0.5 MP with the same pair. Then, on Qwen 2.1 first: `random` vs `grid2x2` vs `face` at 0.5 drop. Try 0.5 vs 0.6 keep, or 75% drop, only if the first race shows no likeness loss.
- Verifier step-time estimates at keep 0.6 (unmeasured): Krea 2 nvfp4 9 s -> ~7-7.5 s, MOSS int8 5 s -> ~4-4.3 s at 1 MP, Qwen 2.1 0.5 MP 2.5 s -> ~2.1-2.3 s.
- Caveat: the verifiers could not search for community reports of TREAD hurting LoRA likeness. Do a quick search before building.

### LoRA up-projection in the W8A8/NVFP4 epilogue (G3: findings 33, 145)

**Prototype. Expected gain: ~5-8% per step for Krea-sized int8 at 1 MP (partly measured: 1.07 s/step ceiling on the trained linears, 4060 Ti); 4-9% for Krea 2 nvfp4 (unmeasured); 3-7% MOSS/Qwen int8 (unmeasured). Likeness-neutral. Confidence: medium. Cost: ~3-4 days.**

**What it is.** Today every trained QuantLinear runs the quantized GEMM, a rescale that writes y[M,N] in bf16, then two more bf16 GEMMs for the LoRA and a full-size add. The LoRA branch is memory-bound at rank 8-16, so its cost is the extra [M,N] pass and the launches, not the FLOPs. The fix computes S = xA [M,r] with cuBLAS and adds `scale * S @ Bᵀ` inside the rescale epilogue, so y is written once. In backward, gS and grad_B share one read of dY.

**Evidence.**
- [LoRAFusion (arXiv 2510.00206)](https://arxiv.org/html/2510.00206v1), EuroSys '26: materialize only S, fuse the base GEMM with alpha*S*B, fuse grad_S with grad_B. 1.27x average / 1.39x max kernel speedup on H100 (layer-wise 1.21x / 1.30x). DRAM traffic -34 to -37%. Applies to dequantize-then-bf16 QLoRA bases. It warns that fusing dequantization is often slower at large token counts. The 1.47x end-to-end is multi-adapter LLM scheduling and does not transfer.
- [SVDQuant / Nunchaku (arXiv 2411.05007)](https://arxiv.org/abs/2411.05007): an unfused low-rank branch adds about 50% (57% at rank 32) to a 4-bit DiT GEMM. Fusing it cuts that to 5-10%.
- [FourTune (arXiv 2607.05711)](https://arxiv.org/abs/2607.05711): W4A4G4 LoRA on FLUX.1-dev with fused kernels at 2.27x bf16 LoRA throughput. That mixes 4-bit compute with fusion, and the hardware was not stated.
- Verifier measurement, RTX 4060 Ti, Krea 2 block shapes, int8_convrot, rank 16, compiled, M=4200 (1 MP), 28 blocks with checkpointed step accounting:
  - 4.38 s base linears without LoRA, 5.71 s with today's compiled LoRA (5.90 s eager), 4.64 s ideal fused. That is a 1.23x headroom, matching LoRAFusion.
  - The LoRA branch adds ~30% to trained-linear time. GQA k/v layers gain 50-90%.
  - 0.5 MP was too noisy to quote (shared GPU). Re-measure on idle GPUs.
- [LyCORIS 4.x kernels](https://raw.githubusercontent.com/KohakuBlueleaf/LyCORIS/main/docs/kernels/benchmarks.md) ([Change.md](https://raw.githubusercontent.com/KohakuBlueleaf/LyCORIS/main/Change.md)): RTX 4090 fp16 LoRA 1.46x fwd+bwd vs eager, 1.43x vs compile, VRAM 1.18x lower. The benchmark appears to time the adapter path only. The LoKr "2-2.5x" on a 5090 is 1.3x on one of three shapes. Labelled "early experimental", weeks old. It cannot fuse into Rulz's quant kernels, so it would recover only ~2-4% of a step.
- Refuted part: [FALQON (arXiv 2510.24061)](https://arxiv.org/html/2510.24061). Its row-append trick and frozen A rely on FP8 GEMMs and a frozen SVD factor of the quantization error. Quantizing A to int8/FP4 changes the LoRA math. Drop it.

**RulzTrainer today.** LoRALinear.forward (lora.py:381-399) computes `base(x) + _added(x)*multiplier`. `_added` (lora.py:352-357) casts fp32 lora_down/up to bf16 and runs two F.linear calls. QuantLinear (quant.py:1014-1020) dispatches to:
- `_W8A8LinearFn` / `w8a8_forward` (quant.py:759-769): triton act quant, `torch._int_mm`, `_rescale_kernel` (quant.py:396-407, one program per row), bias add.
- `_NVFP4LinearFn` / `nvfp4_forward` (quant.py:876-886): `_scaled_mm`, then an eager full-size `y[:, :N] * (a_g*scale2)` pass and a bias add.

Every launch wrapper is `@torch.compiler.disable`, so inductor never fuses the LoRA into the GEMM. Nothing fuses the adapter today.

**Fit and risks.**
- int8 / int8_convrot: the epilogue fits `_rescale`, re-tiled to [BM,BN] tiles for tl.dot, rank padded to 16. S can come from the un-rotated x, or from rotated x with A pre-rotated (the Hadamard is orthogonal).
- nvfp4: the cuBLAS epilogue is not reachable. A Triton post-kernel does slice, per-tensor scale, bias and LoRA in one pass. That also removes today's eager scale pass, a win even without LoRA.
- Compile: sits behind the existing graph break, so no new break.
- Keep blocks: the saved tensors (x, S) are unchanged, so calibration is unchanged.
- Block swap streams qdata as before. model_split and data_parallel launch per device via `_on_device`.
- A rank change in a race costs one cached triton recompile.
- Probes and samples get the faster forward too. Export: parameters unchanged.
- Fallback to the current path for DoRA, LoKr, OFT, extras/turbo LoRAs, train_down, and dropout > 0 unless S is taken from dropout(x).
- Main risk: tl.dot correctness and speed on sm_120 under triton-windows. Test it.

**Implementation path.**
1. quant.py: add a 2D-tiled `_rescale_lora_kernel` (int32 tile -> row/col scales + bias + `scale * tl.dot(S[BM,R], Bt[R,BN])` in fp32 -> one bf16 store). Add an `_nvfp4_epilogue_kernel` replacing the eager scale/bias pass.
2. `_QuantLoRAFn(autograd.Function)`:
   - Forward: `S = x2d @ A.bf16ᵀ`, the quantized GEMM, the fused epilogue; save x and S.
   - Backward: `gS = scale*(g@B)`, `grad_B = scale*(gᵀ@S)`, `grad_A = gSᵀ@x2d`, then the existing base backward followed by `grad_x.addmm_(gS, A)`. In int8-backward mode, fold gS@A into the rescale epilogue.
   - Later: fuse grad_S with grad_B into one read of dY.
3. lora.py: take the fused path when kind == lora, not DoRA, dropout is Identity, and the base uses W8A8/native FP4. Fold the multiplier into the scale.
4. config `lora_fused: auto|on|off` (auto = on for W8A8/W4A4), WebUI advanced, README, wiki.
5. Cheap follow-ups noted by the verifier (unmeasured): a horizontally fused [3r, K] down GEMM for q/k/v, which share x on Krea.

**How to test.**
- Unit test (tests/test_quant.py): fused vs unfused outputs and grads for x, A, B (relative error < 1e-2, cosine > 0.9999) on Krea / Qwen 2.1 / MOSS shapes. Ranks 8/16/32, int8 / int8_convrot / nvfp4, M in {17, 2150, 4200}, compiled and eager.
- Microbench: extend scratchpad/lora_overhead_bench2.py, on an idle GPU only.
- Whole step: scripts/bench.py, Krea 2 nvfp4 on the 5060 Ti and MOSS/Qwen 2.1 int8 on the 4060 Ti, 0.5 and 1 MP, compile on, median of steps 21-45. Alternate off/on back to back. Keep it if the step is at least 3% faster, loss matches within ~1e-3 per step, and peak memory is not higher.
- Likeness sanity: fused vs unfused, run sequentially on the same GPU (the cards differ in speed), 1500 steps, probes every 250. Pass if final ArcFace/DINO are within ~±0.01. Abort the fused arm if any probe falls more than 0.02 below the unfused arm at the same step.

### INT8 attention for the training step (G2: finding 32)

**Prototype. Expected gain: forward-only INT8 (exact bf16 backward) ~3-8%; full SageBwd-style fwd+bwd ~6-10% at 0.5 MP and ~10-16% at 1 MP if the kernel reaches 1.3-1.5x over cuDNN (all unmeasured). Confidence: medium. Cost: ~2-3 days for the forward-only race; 1-2 weeks for an in-house backward.**

**What it is.** Run the attention matmuls themselves in INT8. Q and K are quantized per block, with K smoothed by subtracting its token-mean. P is quantized per token in forward PV, and V per block. The backward recomputes QKᵀ and runs PᵀdO, dS·K and dSᵀQ in INT8, keeping only dP = dO·Vᵀ in FP16. INT8 tensor cores on GeForce run at 4x the bf16-with-fp32-accumulate rate.

**Evidence.**
- [SageAttention3 / SageBwd (arXiv 2505.11594v3)](https://arxiv.org/html/2505.11594v3), NeurIPS 2025 Spotlight. RTX 4090:
  - Kernels: forward ~2x over FA2, backward 1.2-1.6x, fwd+bwd "up to 1.67x". The sequence length of the peak is not stated.
  - End-to-end on a 1B Llama: 2.1 -> 1.9 s at 8K (1.10x) and 6.0 -> 5.2 s at 16K (1.15x).
  - Keeping dP in FP16 raises dQ cosine from 97.47% to 99.77%.
  - Fine-tuning at parity (Qwen2.5-3B MMLU 0.653 vs 0.640 bf16). Pretraining converges more slowly. LLMs only.
- [Abstract v2](https://arxiv.org/abs/2505.11594v2): "lossless performance in fine-tuning tasks".
- Follow-up "SageBwd: A Trainable Low-bit Attention" (arXiv 2603.02170): dS is the dominant error (cosine 0.979). At small tokens per step SageBwd matches full precision. No end-to-end speedup, no code.
- [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention/tree/main): forward-only kernels. Issue #317 (backward support) has been open since Nov 2025 with no reply. lora-rescripts keeps its SageBwd mode disabled until official code exists. The sd-scripts maintainer says Sage "cant use in training".
- [NVIDIA Ada whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf): RTX 4090 INT8 660.6 TOPS vs 165.2 TFLOPS bf16/fp32-acc.
- Not transfer: the 1.67x baseline is FA2 on Ada. On the 5060 Ti RulzTrainer runs cuDNN, measured 15-22% faster per step than PyTorch's pick (README.md:526-532). No image-DiT or likeness results exist anywhere.
- In favour: all four DiTs have q/k RMSNorm (krea2/model.py:92-93, qwen21/model.py:114-115, moss/model.py:71, fibo/model.py:170-171), head dim 128, and batch 1 with 2-4.4k tokens is the low tokens-per-step regime where the follow-up finds parity.

**RulzTrainer today.** attention.py:24-80 only sets the SDPA priority order (config.py:209, trainer.py:714, webui.py:5366). There are no own attention kernels. README.md:1818-1819 (commit 7054bdb) measured attention at 42-50% of GPU time in compiled 1 MP steps on the 5060 Ti, with GEMMs at 31-38%. The verifier scales this to ~25-30% at 0.5 MP. Every family calls the single `attention.sdpa()` (krea2/model.py:109, moss/model.py:63, qwen21/model.py:130-131, fibo/model.py:187). The big calls are maskless at batch 1. Krea 2's GQA is already expanded before SDPA (krea2/model.py:103-105), so the kernel needs no GQA support.

**Fit and risks.**
- Compile: wrap the kernel as a `torch.library.custom_op` with `register_autograd`, or it adds a graph break per block.
- Keep blocks: works under non-reentrant checkpoint. Saving int8 Q/K slightly cuts the memory of kept blocks.
- Block swap and model_split: no interaction.
- data_parallel and side-job threads: choose the kernel per call, never as a global toggle. attention.py's docstring records a process-global attention state breaking Krea's fp32 VAE in another thread.
- Samples and probes stay on cuDNN.
- Windows: triton-windows 3.8 with torch 2.14.1. Int8 tl.dot (mma s8) is untested in this repo. SageAttention's CUDA wheels would need to match torch 2.14/cu130; the pure-Triton variant avoids that.
- FP4 attention (SageAttention3) has no backward, so Blackwell gets nothing beyond INT8.

**Implementation path.**
1. Add `attention_backend: int8` (attention.py BACKENDS, config.py:209, webui.py:5366) as a custom_op.
   - Forward: vendor SageAttention's Apache-2.0 Triton INT8-QK kernels (`sageattention/triton/attn_qk_int8_per_block.py`, `quant_per_block.py`), returning the LSE.
   - Backward: call the exact bf16 aten backward (`_scaled_dot_product_cudnn_attention_backward` on Blackwell, flash/efficient elsewhere) on the saved q, k, v, o and lse.
2. `sdpa()` routes a call there only when grad is enabled, on CUDA, in bf16/fp16, with no mask and no is_causal, and head_dim 64 or 128. Everything else falls through, including the Qwen causal text call, samples, probes and VAEs.
3. One switch covers Krea 2, MOSS, the Qwen 2.1 image call and maskless FIBO. Flux 3 Klein and MiniMax H3 inherit it through `sdpa()`.
4. Only if step 1 wins: write the SageBwd backward in Triton, validated against bf16 gradients.
5. Tests: O, dQ, dK, dV cosine vs SDPA at Krea and MOSS shapes.

**How to test.**
1. Numerical gate: q, k, v and dO captured from one Krea 2 block and one MOSS block at 0.5 and 1 MP, at sigma 0.2/0.5/0.8. Require O cosine > 0.999 and dQ/dK/dV cosine > 0.99.
2. Speed gate: scripts/bench.py, 20 steps, median of steps 6-20, two runs each. Krea 2 nvfp4, MOSS int8_convrot and Qwen 2.1 int8, on both cards. Check with `RULZ_PROFILE_STEP` that the attention share drops. Pass at least 5% lower s/step at 0.5 MP.
3. Quality race: cuDNN vs int8, 1500-2500 steps, 2 seeds, one Krea 2 and one Qwen 2.1 character, probes every 100-250.
4. Adopt as opt-in if the best ArcFace is within 0.01 of baseline on both seeds and wall-clock to the baseline's best is at least 5% shorter. Drop it if the face score falls more than 0.02 below, the loss diverges, or the speed gain is under 5%.

### cuDNN on Ada, and GQA-native SDPA for Krea 2 (G22: findings 89, 153)

**Do it. Expected gain: ~6-8% step time whenever the 4060 Ti trains at 1 MP (data_parallel, model_split, racer on GPU 1), less at 0.5 MP (estimate from a kernel micro-benchmark; end-to-end unmeasured). enable_gqa: ~0% speed, 0.1-0.5 GB less in kept blocks. Confidence: medium. Cost: ~0.5 day.**

**What it is.** Two cheap attention changes. First, `attention_backend: auto` resolves to cuDNN only on Blackwell (attention.py:43-48); on the 4060 Ti it uses PyTorch's pick, which is the memory-efficient kernel. Second, Krea 2's 48q/12kv heads are expanded with `repeat_interleave` before SDPA. Passing `enable_gqa=True` on cuDNN removes the 4x K/V copies saved for backward.

**Evidence.**
- Verifier micro-benchmark on this machine (GPUs partly busy, medians of 7 interleaved rounds), fwd+bwd, 48q/12kv, d128, bf16:
  - cuDNN vs memory-efficient: 4060 Ti 48.5 vs 75.5 ms (0.5 MP) and 164 vs 263 ms (1 MP); 5060 Ti 26 vs 60 ms and 42 vs 102 ms. That is 35-38% faster on the attention kernel.
  - repeat vs gqa: 5060 Ti 26.0 vs 26.1 ms (S=2248) and 42.3 vs 42.0 ms (S=4296); 4060 Ti ~1-2%, within noise. Peak extra memory per call (S=2304): 276 vs 233 MB.
  - cuDNN + enable_gqa gradients, with and without a key-padding mask, match repeat and math (relative error at most 8e-3).
  - With the memory-efficient backend first, enable_gqa raises "No available kernel". Under PyTorch's default order it falls to math: about 4 GB per call and 6-10x slower.
- [diffusers transformer_krea2.py](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/transformers/transformer_krea2.py) expands K/V. [Krea's mmdit.py](https://raw.githubusercontent.com/krea-ai/krea-2/main/mmdit.py) uses cuDNN + gqa and pads to 256. [Config](https://huggingface.co/CalamitousFelicitousness/Krea-2-Base-Diffusers/raw/main/transformer/config.json): 48/12 heads, text-fusion 20/20 (no GQA there).
- [Comfy-Org/Krea-2 discussion #6](https://huggingface.co/Comfy-Org/Krea-2/discussions/6): attention 41% of DiT time, but in one community inference run with no resolution or GPU stated. The scout's "1024px" attribution is wrong.
- [PyTorch 2.15.0](https://github.com/pytorch/pytorch/releases/tag/v2.15.0) (28 Oct) adds Windows flash SDPA for Ampere+. Its 11.5 -> 0.49 ms GQA figure is against the old math fallback, not cuDNN. [2.15 SDPA docs](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html) list GQA on flash, cuDNN, math and memory-efficient. [FlexAttention blog](https://pytorch.org/blog/flexattention/): 90%/85% of FA2 on A100.
- Version note: the G22 verifier labelled the local torch as 2.9.1, but the project venv holds torch 2.14.1+cu130 (checked). The flash conclusion holds either way, since Windows flash arrives in 2.15. Re-check the memory-efficient enable_gqa rejection on 2.14 before relying on the guard.

**RulzTrainer today.** sdpa() has no enable_gqa argument (attention.py:77-80). krea2/model.py:103-109 repeats K/V. Two of finding 89's three changes already exist: krea2/model.py:311-316 passes mask=None when nothing is padded and never pads to 256. Qwen 2.1's key_mask is None when unpadded (qwen21/model.py:322), so FlexAttention has nothing to skip at batch 1. README.md:517-540 says Ada was not measured.

**Fit and risks.**
- Gate enable_gqa per call on the resolved backend being cuDNN, or the 4060 Ti under `torch` falls to math (~4 GB per call, OOM risk on 16 GB).
- The alignment tap (alignment.py:92-98) expects matching head counts. Expand k only while the tap is active.
- Compile: shapes unchanged, low risk (untested). W8A8/nvfp4, block swap, probes, races, export: unaffected.
- Per-call gating is thread-safe under data_parallel and model_split.
- A torch 2.15 upgrade for Windows flash means re-validating the int8/NVFP4 triton kernels and `_scaled_mm`. Not now.

**Implementation path.**
1. attention.resolve: `auto` also picks cuDNN on capability 8.9. Better: a 20-step per-GPU micro-benchmark (cuDNN vs PyTorch's pick on the first bucket), cached in .rulz_cache.
2. `sdpa(..., enable_gqa=False)` plus `gqa_ok()` (true only for cuDNN, or a flash/efficient-GQA capability check).
3. krea2/model.py Attention.forward: skip repeat_interleave when gqa_ok(). Keep repeat as the fallback.
4. README "Attention kernel" section and wiki.
5. Drop FlexAttention for Qwen 2.1 and the torch 2.15 move for now.

**How to test.**
- Ada: scripts/bench.py 20 steps on the 4060 Ti for Krea 2, MOSS int8 and Qwen 2.1 int8 at 0.5 and 1 MP, `attention_backend: torch` vs `cudnn`. Make cuDNN the Ada default if it wins at least 5% s/step at equal loss and memory. Then a short race of the two backends on GPU 1; probe trajectories should match within noise.
- GQA: Krea 2 nvfp4 1 MP, 20 steps x2 per arm, calibrated keep blocks. Pass if loss matches within 0.0007 per step, peak VRAM is lower or keep blocks higher, and s/step is not worse. Check that attention localization still works.
- Guard unit test: enable_gqa is never passed when the backend resolves to `torch` without flash.

### Raceable backward modes, then MS-EDEN for the FP4 backward (G6: findings 36, 37)

**Race it. Expected gain: up to -9% s/step on Krea 2 nvfp4 on the 5060 Ti if the existing fp4 backward holds likeness (measured 8.73 -> 7.93 s); fp4_eden adds no speed, only lower grad_x error (~17.5% -> ~14%, simulated). Confidence: low. Cost: ~0.5 day for the race keys; ~3 days for fp4_eden.**

**What it is.** `nvfp4_backward: fp4` already runs the frozen-layer grad_x GEMM at FP4 rate, with per-element stochastic rounding of grad_out. Most configs keep the exact bf16 backward. Quartet II's MS-EDEN applies a randomized Hadamard to 128-entry chunks of grad_out, rounds to nearest, and puts the unbiasing correction into the E4M3 group scales by stochastic rounding. That gives a lower-variance unbiased estimator. The first step needs no new kernel: make the backward modes raceable and find out whether plain fp4 already holds likeness.

**Evidence.**
- [Quartet II (arXiv 2601.22813)](https://arxiv.org/html/2601.22813): MSE on N(0,1), x1e-3: SR 23.5, MS-EDEN 9.4, RTN 9.0. Val BPB gap to BF16 at 560M 1.56% (NVIDIA) / 1.51% (4/6) / 1.26%; at 1.9B 1.92 / 1.76 / 1.44. RTX 5090 linear-layer speedup >4x; 1.1B end-to-end "185% of baseline". LLM pretraining from scratch; the loss gaps do not transfer.
- [NVIDIA NVFP4 pretraining (arXiv 2509.25149)](https://arxiv.org/html/2509.25149v2): SR on Dgrad/Wgrad gradients only; SR on forward tensors is detrimental. ~16% of layers kept in BF16. Contrary evidence: in its 1.2B ablation, "transforming Fprop or Dgrad inputs degrades model quality", and both members rotate the Dgrad GEMM.
- [RTX Blackwell whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf): RTX 5090 FP4 1676 dense vs 209.5 BF16/fp32-acc (8x).
- RulzTrainer's own numbers: README.md:491-515 gives 17.5% grad_x error (cosine 0.985), -9% s/step on Krea 2 (8.73 -> 7.93 s), +0.2 GB peak. Whole-model LoRA grad cosine vs bf16 at sigma 0.2/0.5/0.8: 0.99 / 0.93-0.96 / 0.97-0.98. The measured FP4 backward is only 1.5x faster per layer (17.6 -> 11.4 ms), because the on-the-fly W^T requant and the grad quantization dominate.
- Verifier simulation (scratchpad/sim/eden_sim.py): grad_x error ~17.5% (SR, reproduces the README) -> ~14% (EDEN). The floor is ~9.3% from the RTN requant of W^T.
- **Refuted, finding 37 (HALO-2 int8_halo):** [HALO (arXiv 2501.02625)](https://arxiv.org/html/2501.02625v3)'s INT8 collapse (GSM8K 4.5) is a per-tensor-scale, no-SR failure. RulzTrainer's int8 backward is already per-token and already HALO-1 (quant.py:772-786, ~1% error per README.md:440-446). Simulated per-token int8 grad_x error (RTN / +SR / token-axis H / N-axis H): Gaussian 0.85 / 1.20 / 0.85 / 0.85%; channel outliers 4.7 / 6.6 / 4.7 / 0.83%; Student-t3 3.9 / 5.5 / 1.3 / 0.88%. SR always hurts in int8, the token-axis Hadamard helps only i.i.d. heavy tails, and the two extra passes eat the speed gain. Drop it.

**RulzTrainer today.**
- quant.py:42 has `NVFP4_BACKWARDS = ("bf16", "fp4")`. quant.py:891-902 dequantizes, transposes and RTN-requantizes W^T per row, then SR-quantizes grad_out (kernel at quant.py:600-652, seed counter `_SR_CALLS` at :888).
- int8_backward: int8 is already set in configs/default.yaml:22, identity.yaml:22, my_job.yaml:25 and others. Qwen 2.1 uses bf16, and my_job.yaml:27 keeps `nvfp4_backward: bf16`.
- RACE_KEYS (race.py:49-78) omits int8_backward and nvfp4_backward, so no race can A/B them today.

**Fit and risks.**
- fp4_eden fits better than the scout said. W^T is already requantized every backward, so the random-sign rotation along N folds into that requant, with no stored copy.
- The scout's "+0.5 byte/param rotated copy at load" and "reuse the ConvRot weight" are wrong. grad_x contracts along N (ConvRot rotates along K), and the signs must be fresh per call (`_SR_CALLS` seed) to stay unbiased.
- Use 256-entry chunks (H16 x H16, the trick in `_rot_quant_act_kernel`, quant.py:473). All Krea 2 N dims divide 256.
- Compile, keep blocks and block swap are unaffected. Under model_split / data_parallel, layers on the 4060 Ti (no FP4) use bf16. data_parallel threads share the GIL-atomic seed counter.
- Per-block override (bf16 backward in the keep-blocks region) is easy, since the mode is a per-QuantLinear attribute.
- Applies only to nvfp4 layers on Blackwell (Krea 2). Nothing for the int8 MOSS / Qwen 2.1 runs.
- Side idea, larger and independent (estimate, unmeasured): a fused dequantize -> transpose -> requantize kernel for W^T. Today that is 3-4 passes over ~200 MB of bf16 per large Krea layer.

**Implementation path.**
1. race.py: add `int8_backward` and `nvfp4_backward` to RACE_KEYS. `swap_in` calls `Q.configure_int8` / `Q.configure_nvfp4` with the row's values.
2. Only if plain fp4 loses: add `nvfp4_backward: fp4_eden` (NVFP4_BACKWARDS, config.py:208, webui.py:5354). A ROTATE+EDEN variant of `_nvfp4_quant_act_kernel` (per-call Philox signs along N, H16xH16 rotation per 256-chunk, e2m1 RTN, c = sum(x²)/sum(x·Q(x)) per chunk, SR on each 16-group's e4m3 scale times c). Requantize W^T with the same signs (RTN). N % 256 != 0 falls back to fp4.
3. Optional: bf16 backward for the keep-blocks region.

**How to test.**
- Units (fp4_cuda-gated): fp4_eden grad_x error on Krea shapes (expect ~0.14 vs 0.175). The mean of 64 seeded calls approaches exact as 1/sqrt(n); watch heavy-tailed inputs. Repeat the README whole-model grad cosine at sigma 0.2/0.5/0.8.
- Race: Krea 2 nvfp4, rank 16, 0.5-1 MP, one 5060 Ti, 2000 steps, probes every 100, race_drop_lowest off. Racers: bf16, fp4, fp4_eden, plus a bf16 row with its own seed for seed noise.
- Stop rule:
  - If fp4 lands within the bf16 seed spread, make fp4 the wizard default on Blackwell and skip EDEN.
  - Otherwise adopt fp4_eden only if it is within the spread and at least 7% faster than bf16.
  - Drop any mode that sits more than one seed spread below bf16 at two consecutive probes after step 1000.
- For int8: hook grad_out on a few MOSS / Qwen 2.1 layers and measure int8 vs exact grad_x error. Close the item if it is under ~2%.

### Compile cache persistence, custom ops for the quantized Linears, fused eager kernels (G13: findings 46, 152, 157)

**Do it (cache persistence on vast.ai); Skip-ish (custom ops, fused eager kernels). Expected gain: cache saves the ~160 s cold vs 36 s warm compile on a fresh vast.ai box; custom ops 1-4% per step (unmeasured); fused eager kernels 5-12% on block swap / finetune / race_parallel paths only (unmeasured). Confidence: low. Cost: ~0.5 day (cache); ~3 days (custom ops).**

**What it is.** The quantized-Linear triton wrappers are `@torch.compiler.disable`, so each block compiles with graph breaks. Three ideas sit on top of that:
- Register the quantizers and GEMM paths as `torch.library.custom_op` (opaque, no break) or `triton_op` (traced into, epilogue fusion).
- Hand-fuse (1+w)-RMSNorm + modulation and SwiGLU for the paths that run without compile.
- Persist inductor artifacts with Mega-Cache.

**Evidence.**
- [triton_op recipe](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html) and [torch.library 2.15](https://docs.pytorch.org/docs/2.15/library.html): triton_op since 2.6 is traced into, `register_autograd` is preferred, and autotune is limited under compile.
- [Regional compilation recipe](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html): 9.66 s -> 0.35 s on 64 repeated layers. That is a toy cold-start figure.
- [PyTorch 2.12 notes](https://github.com/pytorch/pytorch/releases/tag/v2.12.0): user-defined Triton kernel unary epilogue fusion.
- [Mega-Cache tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html): portable with the same torch, triton and GPU; no timings.
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) ([modulated_rms_norm.py](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/src/liger_kernel/ops/modulated_rms_norm.py), [swiglu.py](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/src/liger_kernel/ops/swiglu.py)): ~20% throughput and ~60% memory on LLaMA 3-8B, 8x A100, FSDP, dominated by the fused cross-entropy. Does not transfer. No stated Windows support. `offset` defaults to 0; Krea's zero-centred norm needs offset=1 and rounds slightly differently.
- RulzTrainer's own compile gains: MOSS int8 6.61 -> 4.90 s, Krea 2 nvfp4 8.77 -> 7.74 s on the 5060 Ti.
- Contrary, RulzTrainer's own profile: README.md:1818-1820 puts attention at 42-50%, GEMMs at 31-38% and fused pointwise at ~10%, and concludes custom ops "could gain a few percent at most". README.md:1797-1820 counts 19-29 graphs per model with 0 recompiles, not "28-42 separate compiles". The 25-36 s warm-up is ~0.5-1% of a 1000-3000-step run.

**RulzTrainer today.** Blocks compile per instance with `dynamic=True` (trainer.py:1473-1484). Wrappers are disabled at quant.py:79, 430, 445, 519, 567 and 653, and the forward goes through autograd.Functions taking a non-tensor QuantWeight (quant.py:802, 909, 952). Compile is off for race_parallel, finetune and block_swap (trainer.py:876-881, blockswap.py:22, finetune.py:32, race_parallel.py:14). The reasons are threads tracing at once, swapped modules and streamed weights, not graph breaks, so custom ops would not lift those exclusions. quant.py:427-429 records a torch 2.14 dynamic-shape trace failure for triton launches, which makes triton_op riskier than custom_op.

**Fit and risks.**
- custom_op needs QuantWeight flattened to tensors plus ints and bools. The NVFP4 SR seed must be an explicit int argument.
- Keep blocks: possibly one more kept block at 1 MP (the fp32 norm copy and silu output), unmeasured. The LoRA path saves Linear inputs anyway.
- Fused eager kernels are rarely used by this user: Krea 2 int8 fits at 9.17 GB without block swap, and finetune and race_parallel are opt-in. Port in-repo rather than depend on Liger (Windows).

**Implementation path.**
1. `torch.compiler.save_cache_artifacts` at run end and `load_cache_artifacts` at start, stored at `<model_dir>/.rulz_cache/compile/<torch>-<triton>-<sm>.bin` and shipped in scripts/vast.
2. Behind `RULZ_QUANT_OPS=1`: `rulz::w8a8_linear` and `rulz::nvfp4_linear` custom_ops with `register_fake` and `register_autograd` reusing `w8a8_backward_x` / `nvfp4_backward_fp4`, plus a `torch.library.opcheck` test. Do not touch the compile exclusions.
3. Only if the eager paths become common: rulztrainer/kernels.py with fused (1+w)-RMSNorm + modulation and SwiGLU, Krea first, then MOSS `_situ` and the Qwen tanh gates.

**How to test.** scripts/bench.py on the 5060 Ti at 1 MP, 45 steps, Krea 2 nvfp4 and MOSS int8_convrot, `RULZ_QUANT_OPS=0` vs `1`. Record median s/step over steps 21-45, warm-up seconds, `compiling.report()` graphs and breaks, peak VRAM, and the calibrated keep count. Losses must match to bf16 noise; run one 1000-step likeness check per arm. Keep a change only if it gives at least 3% median step time or at least one extra kept block on both models. Otherwise leave README.md:1818-1820 as the record. For the cache: time a cold vast.ai start with and without the shipped artifact.

### Four-over-Six NVFP4 block scaling (G5: finding 35)

**Race it (weight side; activation side opt-in). Expected gain: no speed gain (weight side free; activation side ~1-3% slower, unmeasured); NVFP4 weight error -8% (measured offline), W4A4 output error -5 to -8%; likeness change likely within probe noise (unmeasured). Confidence: medium. Cost: ~1-2 days.**

**What it is.** NVFP4 maps each 16-value block's absmax to E2M1's max of 6. Four Over Six quantizes each block twice, with absmax mapped to 6 and to 4, and keeps the lower-MSE version. The tensor scale uses 256 instead of 448 so the max blocks can also pick 4. The result stays bit-compatible with Blackwell tensor cores, because the choice lives in the ordinary E4M3 scale. It is a quality lever for the nvfp4 speed path, not a speed lever.

**Evidence.**
- [Four Over Six (arXiv 2512.02010v3)](https://arxiv.org/html/2512.02010v3), MIT Han Lab + NVIDIA, code at mit-han-lab/fouroversix (builds for sm_120, has a FLUX.1-dev example, no diffusion quality numbers).
  - AWQ / SmoothQuant: 19.9% / 5.3% closer to BF16. GPTQ gets worse (+34.6% gap).
  - Pretraining (30B-A3B, 1T tokens, 384 B200): loss 22.3% closer to BF16. Does not transfer.
  - Quantizer overhead under 15%.
  - Plain RTN, the closest analog to RulzTrainer, is small and mixed: 10 of 12 entries improve, but Qwen3-1.7B WikiText goes 23.06 -> 23.60 and Qwen3-32B C4 27.54 -> 27.67.
- [Quartet II](https://arxiv.org/html/2601.22813) uses 4/6 for forward weights and activations. Gaps vs BF16: 560M 1.56 / 1.51 / 1.26%.
- [NVIDIA NVFP4](https://arxiv.org/html/2509.25149v2): forward stays RTN.
- TransformerEngine PR #2972 (merged 22 May 2026) adds `NVTE_NVFP4_4OVER6`. Reviewers said 256 helps "only slightly", and gradient quantizers were left out.
- Verifier offline check on real Krea 2 Raw weights (blocks 10-11, CPU):
  - Weight error 0.0943-0.0950 -> 0.0869-0.0871 (-7.6 to -8.4%); 42-45% of blocks pick 4. 448 -> 256 alone costs nothing.
  - ff.up W4A4 output error: Gaussian 0.134 -> 0.123, lognormal 0.127 -> 0.120, 16 outlier channels 0.121 -> 0.114.
  - A wider 11-scale weight search gives -11.3%.
- Scale of the prize: README 20-step Krea loss is W4A16 0.0713 vs W4A4 0.0724. Quantization noise is ~1.5% of the loss.

**RulzTrainer today.** Every path uses absmax -> 6 and a 448*6 tensor scale: quantize_nvfp4 (quant.py:165, :168), the fused act kernel (:613, g at :658), the eager fallback (:858) and the FP4 backward (:897, :899). Weight installs go through quantize_weight (:235, :1043), split_rows (:227) and finetune.py:255, so one change covers all of them. The user's Krea 2 runs auto-quantize HF bf16 to nvfp4 (output/am4check-B3-automagic3-nvfp4/train.log).

**Fit and risks.**
- Tensor-core format unchanged, so `_scaled_mm`, the swizzle, block swap, model_split, data_parallel and the wrappers are untouched.
- The act quantizer is already compile-disabled. Checkpointing runs the quantizer twice per step.
- Default must stay "6": the README claims bit-for-bit comfy_kitchen parity, and tests assert kernel == reference.
- Do not apply 4/6 to the SR gradient: picking the lower-error SR draw breaks `test_nvfp4_stochastic_rounding_is_unbiased`. TE also excludes gradients.
- On the 4060 Ti (W4A16) only the weight side matters. MOSS / Qwen 2.1 int8 runs gain nothing.
- Related speed note from the verifier: each Krea block quantizes the same normalized input separately for to_q/k/v/gate and for ff up/gate. Quantizing a shared input once would save more quantizer time than 4/6 costs. Unmeasured.

**Implementation path.**
1. quant.py: `nvfp4_scale_rule: 6 | 4over6 | search` (search for weights only). For 4over6 and search, set `per_tensor_scale = amax/(256*6)` and pick the per-block lowest SSE.
2. Extend `set_nvfp4_default` / `configure_nvfp4` so all installers, `sample_model_quant` and `text_encoder_quant` follow it.
3. Add a `FOUR_OVER_SIX` constexpr to `_nvfp4_quant_act_kernel` (reuse `_round_e4m3`, `_e2m1_value`).
4. Use 4/6 for the W^T RTN requant in the FP4 backward. Leave the SR grad_out on the standard rule.
5. Config next to nvfp4_backward, wired at trainer.py:709/741/786, plus WebUI, README and wiki.

**How to test.**
- Units: kernel equals the eager reference; per-block SSE is at or below standard; the default still matches the ComfyUI file bit for bit; SR unbiasedness still passes.
- Offline on real activations at sigma 0.2/0.5/0.8: expect 5-8% lower W4A4 error. Stop if it is under 3%.
- Race sequentially on the 5060 Ti: Krea 2 Raw, AZav50, rank 16, 1500-2000 steps, 2 seeds per racer.
  - Racers: A standard, B 4over6, C search weights + 4over6 activations, optional D int8_convrot via model_split.
  - Use the same probe engine for all, so base quantization does not confound the score.
- Make B or C the default only if its best face score beats A by more than A's seed spread, at no more than 3% s/step cost. Otherwise keep only the free weight-side search.

### Cheaper probe engines and probe-bias calibration (G60: findings 103, 111)

**Do it (calibration + per-adapter sigma lists). Expected gain: 0-5% wall-clock on a 1000-3000-step character run (probes already run on the side GPU); 50-75% less probe render time per probe at 4 or 2 steps (NFE arithmetic, unmeasured); larger in races, self-training and the reward stage. Confidence: medium. Cost: ~1-2 days.**

**What it is.** Few-step adapters make each probe render cheaper: lightx2v's 4-step Lightning, linoyts' 2-step DMD2, PrunaAI's 5/6-step LoRAs, nynxz's 153 MB SVD extraction of the Turbo LoRA. For Flux 3 Klein, the cheaper engine is the 16-step distilled flow-map checkpoint. Few-step engines can understate likeness for a base-trained LoRA. So the useful part is a per-run calibration: render the first and last probe sets also on the plain model at full settings, and log the bias and its drift.

**Evidence.**
- [lightx2v/Qwen-Image-2.1-Lightning](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning): 4 steps, sigmas [1.0, 0.75, 0.5, 0.25], true_cfg 1, dynamic shift (base 0.5, max 0.9), FP32-latents warning. No release date and no licence on the card.
- [linoyts/Qwen-Image-2.1-DMD2-2steps](https://huggingface.co/linoyts/Qwen-Image-2.1-DMD2-2steps): rank 32 attention-only, sigmas [1.0, 0.5], shift 1, 2000 steps on one H100, needs a diffusers dev branch.
- [nynxz/Qwen-Image-2.1](https://huggingface.co/nynxz/Qwen-Image-2.1): 153 MB vs 913 MB, rank 9-75 (mean 26.5), exact diffs for img_in, proj_out and the timestep embedder. PSNR 20.4 dB to the official LoRA; 16.6 dB to full Turbo.
- [PrunaAI/P-Q-Image-2.1-Comfy](https://huggingface.co/PrunaAI/P-Q-Image-2.1-Comfy): 5/6/8-step sigma lists, "up to 6.3x faster", no quality numbers.
- [Fizgig v6.5.3](https://github.com/shootthesound/Fizgig/releases/tag/v6.5.3): turbo previews "sometimes add body horror"; it defaulted to 25 steps plain. [musubi-tuner #1163](https://github.com/kohya-ss/musubi-tuner/issues/1163): kohya says a base-trained LoRA "is not expected to work well on the turbo model".
- Flux 3 Klein: [BFL README](https://github.com/black-forest-labs/flux-3-image-klein) gives distilled 16 steps without CFG, ~6x faster (~3 s vs 12 s on H200 at 1 MP). See also [sampling.py](https://raw.githubusercontent.com/black-forest-labs/flux-3-image-klein/main/src/flux3/sampling.py) (target sigma via `timesteps_ref`), [ComfyUI flux3](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/ldm/flux3/model.py) (`ref_time_in`), [HF blog](https://huggingface.co/blog/flux-3) (1.93 s, 23.8 GiB on H100) and [SGLang](https://docs.sglang.io/cookbook/diffusion/FLUX/FLUX.3) (guidance must be 1). No source says whether a base LoRA transfers. The 5.8 / 5.3 GB file sizes are unverified. Datacenter timings do not transfer.

**RulzTrainer today.**
- config.py:43-61 ARCH_TURBO uses the rank-178 LoRA at 8 steps for Qwen. `turbo_lora` (config.py:319-321) already takes any path.
- qwen21/sampler.py:31-40 interpolates the 8 Turbo sigmas to any step count. With 4 steps that gives ~1.0/0.93/0.85/0.41 instead of Lightning's schedule, and DMD2's [1, .5] is not reachable.
- lora.py (~:854) applies `.diff` only to norms, so nynxz's three linear diffs are dropped.
- Probes and samples share one engine (trainer.py:2147-2157, :3285). Nothing calibrates against full-quality renders; README.md:1425 advises judging by hand, and the MOSS turbo LoRA is already documented to understate likeness.
- The untracked turbo_lora_check.png suggests a manual Krea check was done once.

**Fit and risks.** Sampling-only extras sit on LoRALinear / QuantLinear, so W8A8, nvfp4, compile and block swap are untouched. A 153 MB file saves ~0.75 GB on the side GPU. Races share one probe engine, so a constant bias cancels; drift that interacts with the character LoRA mis-ranks configs and shifts early stop, which is the case for calibrating. The reward stage reuses probe_steps (trainer.py:2778). A 2-step render may compress ArcFace scores. No Windows issues.

**Implementation path.**
1. Per ARCH_TURBO entry: `sigmas` and `shift_mode` (turbo / dynamic / none). Add `probe_turbo_lora` and `probe_sigmas`, defaulting to the sample settings.
2. qwen21/sampler.qwen21_sigmas accepts an explicit list.
3. lora.attach_extra_lora applies linear `.diff` as a sampling-only offset.
4. `probe_calibrate: true`: at the first and last probe, also render at ARCH_FULL_SAMPLE settings and log delta_face / delta_body to likeness.jsonl. Warn on drift. Works for every family.
5. For Flux 3 Klein when added: a flow-map sampler behind `sample_model_path`, enabled only if the calibration gap is under threshold.

**How to test.** One Qwen 2.1 character run, 0.5 MP, rank 16, 2000 steps, checkpoints every 250. Re-probe each checkpoint offline with plain 25 steps (reference), Turbo 8, nynxz-153 at 8 (after the diff fix), Lightning 4 and DMD2 2. Metrics: Spearman rank correlation with the reference, bias and drift, best-checkpoint and early-stop step, s/image on the 4060 Ti. Adopt an engine if correlation is at least 0.9, the best checkpoint is within one of the reference's, and drift is under 0.02 ArcFace. Then a 2-config race to confirm the same winner. For Flux 3: same protocol with base 50 steps + CFG as the reference.

### Quantized attention for no-grad renders (G8: finding 39)

**Skip-ish. Expected gain: training step 0; whole probe ~3-6% faster; whole character run well under 1%; 1 MP samples and self-training renders ~7-20% of their render time (estimates from a measured attention share). Confidence: medium. Cost: ~1 day plus a Windows wheel problem.**

**What it is.** Use SageAttention2 (INT8 QK, FP8 or FP16 PV, forward only) for samples, probes and self-training renders. These forward passes have no backward, so the quantized kernel is allowed.

**Evidence.**
- [SageAttention2++ (arXiv 2505.21136)](https://arxiv.org/abs/2505.21136): the 3.9x figure is the INT4-QK "4+8" variant on an RTX 4090. The INT8+FP8 "8+8" kernel is ~3.0x over FA2. Flux ImageReward drops 0.912 -> 0.902 with 8+8. The FP16 accumulator overflows unless the P/V ranges are narrowed.
- [SageAttention3](https://arxiv.org/html/2505.11594v3): 1038 TOPS on an RTX 5090, inference only. End-to-end speedups are for long-sequence video.
- [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention): recommends SA2 for precision-sensitive use. No official Windows wheels.
- Contrary: [Comfy-Org/Qwen-Image discussion 2](https://huggingface.co/Comfy-Org/Qwen-Image/discussions/2) and the [KJNodes docs](https://comfyui-wiki.com/en/custom-nodes/ComfyUI-KJNodes/nodes/pathch-sage-attention-kj) report black or noisy Qwen-Image output with FP8-PV and advise the slower fp16-PV kernel. SA3 is reported to crash on Windows.
- [Nunchaku](https://github.com/nunchaku-ai/nunchaku) supports none of Krea 2, Qwen-Image 2.x, MOSS or FIBO. Drop that sub-proposal.
- Verifier microbenchmark (4060 Ti, single block, bf16), attention share of block matmul time: Krea 2 at a 512 px probe (1324 tokens) 9%; Qwen 2.1 (1224 tokens) 12-13%; Krea 2 at 1 MP 10-13%; Qwen 2.1 at 1 MP 28-30%.

**RulzTrainer today.** attention.py:25-26 offers only SDPA backends, and tests/test_attention.py:41-42 asserts that `set_backend('sage')` raises. Samples and probes share the training model's blocks and backend. Sageattention is not installed in the venv.

**Fit and risks.**
- Gating on `torch.is_grad_enabled()` is wrong. The region-prior base (trainer.py:988-989), the DOP target (:2470-2471) and the reward KL reference (:2849-2855) are no-grad training targets. Routing them through quantized attention would teach the LoRA the quantization error. Use an explicit thread-local render context instead (side jobs run in per-GPU threads, multigpu.py:52, 457).
- Leave reward rollouts out: an on-policy ratio is recomputed in bf16 (trainer.py:2847, 2859).
- Use SA2 on both cards, never SA3, so probes stay comparable across the 5060 Ti / 4060 Ti split and in race_parallel. Store the setting in the probe history and refuse to compare across settings, the step-0 floor probe included.
- Fall back to SDPA on masks.

**Implementation path.** A `render_attention()` context in attention.py and a `render_attention: off | sage | sage_fp16pv` option (fp16-PV by default for Qwen 2.1). Enter it only in the sampler, probe renders and self-training renders. Linux/vast.ai first; on Windows only with a wheel built for torch 2.14 cu130.

**How to test.** Re-probe five checkpoints each from one Krea 2 nvfp4 run and one Qwen 2.1 int8 run, with off vs sage at identical seeds (probe_seeds 4, 512 px, 8 steps, probe_cfg 1.0). Adopt only if all of these hold:
- probe time drops by at least 5%;
- the mean absolute score difference is under 0.5 points and the maximum is under likeness_noise_band (1.0);
- the best-checkpoint pick is the same and no ranks flip;
- there are no black or NaN renders.

### FlatQuant learned transforms for the quantized base (G10: finding 42)

**Watch. Expected gain: int8 ~0; nvfp4 W4A4 at most recovers toward W4A16 (loss 0.0724 vs 0.0713, ~1.5%), at an estimated 3-10% added step time (unmeasured). Confidence: low. Cost: ~1-2 weeks; zero-code ceiling test first.**

**What it is.** FlatQuant learns, per linear layer, an invertible Kronecker affine transform plus per-channel scaling and clipping, calibrated on block-output MSE. The inverse is folded into the weight, and the transform runs fused with the activation quantizer. It would replace the fixed Hadamard-256 of int8_convrot, or add a transform to nvfp4.

**Evidence.**
- [FlatQuant (arXiv 2410.09426v4)](https://arxiv.org/html/2410.09426v4), ICML 2025: W4A4 LLaMA-3-8B zero-shot FP16 73.23 / QuaRot 61.34 / SpinQuant 66.98 / FlatQuant 71.23. LLaMA-3-70B WikiText-2 perplexity 2.86 / 55.44 / 3.78. Calibration takes ~0.9 h (not GPU-hours). The 2.61% FLOPs, 3.41 MB and 0.07x vs 0.26x overhead are LLaMA-2-7B on an RTX 3090. All INT4 LLM results; [its repo](https://github.com/ruikangliu/FlatQuant) has no FP4 or diffusion results.
- [ConvRot (arXiv 2512.03673)](https://arxiv.org/abs/2512.03673): fixed group-wise Hadamard for W4A4 DiTs.
- Contrary: [MR-GPTQ (arXiv 2509.23202)](https://arxiv.org/html/2509.23202), ICLR 2026. NVFP4's 16-value groups neutralize outlier mitigation. Llama-3.1-8B W4A4 NVFP4 recovery: RTN 94.67%, RTN+Hadamard 93.82%, QuaRot 93.80%, SpinQuant 93.40%, SmoothQuant 95.90%, GPTQ 95.92%. Rotations hurt.
- [KroQuant (arXiv 2607.21446)](https://arxiv.org/html/2607.21446) already learns Kronecker transforms for DiTs (MXFP4). Learned vs fixed block-Hadamard-32 on PixArt-Sigma: LPIPS 0.347 vs 0.382, PSNR 16.5 vs 15.7 dB; mixed on FLUX.1-schnell. [DiRotQ (arXiv 2605.16732)](https://arxiv.org/abs/2605.16732) uses calibrated rotations. The scout's "nobody has done this for DiTs" is false.

**RulzTrainer today.** int8_convrot's Hadamard is folded into the weight (quant.py:97-111, 1062-1068), applied in the fused kernel (quant.py:473-512, 712-728) and inverted in backward (quant.py:772-785). nvfp4 has no transform (quant.py:847-889). There is no calibration PTQ anywhere. int8 forward error is already under 1% (README.md:432-433). The near-lossless option already exists: Krea 2 int8 model_split at 10.0 s/step vs nvfp4 9.2 s/step (README.md:1590-1596).

**Fit and risks.** Full-row transforms do not match the per-256 or per-16 kernels. Calibration needs a bf16 reference and hours per base model, and it yields a RulzTrainer-only format (use a sidecar file). Quantization is shared by all racers (race.py:15-18), so comparisons need separate runs. Block swap moves the transform buffers. Export is unaffected.

**Implementation path.** Nothing until the ceiling test shows a gap. Then the cheaper NVFP4-aligned variant first: a per-channel scale, a clip, and optionally a learned 16x16 block transform fused into `_nvfp4_quant_act_kernel` (quant.py:600). Fit it with a `scripts/calibrate_quant.py` on cached latents, write a `<model>.rulzquant.safetensors` sidecar, and select it with `transformer_quant: nvfp4_flat`. Try GPTQ-style weight calibration, and making `nvfp4_matmul` raceable, before full FlatQuant.

**How to test.** Zero-code ceiling test, as separate runs with the same seed, images and probe seeds: Krea 2, one character, 0.5-1 MP, rank 16, 2000 steps, probes every 100, 2 seeds each.
- (A) nvfp4 W4A4.
- (B) nvfp4 with `nvfp4_matmul: false` (W4A16).
- (C) int8_convrot with `multi_gpu: model_split`.

Stop if A-C is within seed spread (under ~0.01 ArcFace). If B is about equal to A, an activation transform is pointless. Otherwise build nvfp4_flat and accept it only if it closes at least half the A-C gap at no more than 5% step overhead over A.

## Fewer steps to likeness: optimizers and learning-rate schedules

This section covers changes on the optimizer side that could reach the likeness plateau in fewer steps: the learning-rate schedule and cooldown, factor-aware LoRA updates (Riemannian preconditioning, pair-aware Muon), self-tuning learning rates (Prodigy), per-matrix learning rates, update masks, slow momentum, spike handling and checkpoint averaging. The overall verdict is sober. No item has a measured likeness gain on any flow-matching DiT, and most evidence comes from LLM pretraining. All of these items touch only the fp32 LoRA factors and run after backward, outside the compiled blocks, so none of them conflicts with W8A8, nvfp4, compile, keep blocks, block swap or data_parallel. The cheapest useful steps are races that need no code: decay versus constant lr, warmup length, and averaging a run's own checkpoints through the soup. After those, two small builds are worth a race: a cooldown when the probes stop a run, and r x r gradient preconditioning. Pair-aware Muon is the one bigger build worth prototyping. Automagic 4 (now committed, `rulztrainer/optimizers/automagic4.py`) already covers part of what Prodigy and the Cautious mask would add, as noted in each item below.

### Cooldown at the end: warmup-stable-decay schedule and cooldown when the probes stop

**Race it. Expected gain: 0-3 combined-probe points on top of lora_ema, possibly zero (unmeasured on diffusion LoRA). Confidence: medium (0.48). Cost: ~1 day; the constant-vs-cosine precursor race needs no code.**

**What it is.** The schedule holds a constant lr after warmup and then decays over the last 10-20% of steps with f(n) = 1 - sqrt(progress), down to about 0.01x. The decay can start at any step, so a run whose length is decided on the fly (probe plateau, race winner) can still end with a proper decay. In RulzTrainer the decay would start when the probes trigger an early stop. REX (eta0 (1-p)/(1-p/2)) is the fixed-length alternative.

**Evidence.**
- [Hägele et al., NeurIPS 2024](https://arxiv.org/html/2405.18392v3): a (1-sqrt) cooldown beats linear, and the benefit plateaus at about 20% of steps. On 1B/100B tokens, "Cosine to 10%" scores 46.26 and "1-Sqrt 20%" scores 46.23 aggregate, which the paper calls comparable. The cooldown is slightly less lr-sensitive. SWA over a constant lr helps but does not reach the cooldown level, and EMA did worse than SWA. Setting: LLM pretraining at 33M-8B.
- [Dremov et al., TMLR 2025](https://arxiv.org/abs/2508.01483): the cooldown shape is a bias-variance trade-off, and higher beta2 in the cooldown helps. Base betas were (0.9, 0.95) and the cooldown always went to zero, on a 210M LLM. With model averaging, the high-variance mirror-cosine shape beat sqrt. This matters because RulzTrainer averages its LoRA through the EMA.
- [REX, MLSys 2022](https://ar5iv.labs.arxiv.org/html/2107.04197): REX was top-1 in 73% and top-3 in 95% of low-budget settings, but by small margins (RN20-CIFAR10 at 1%: 27.94 vs cosine 28.49, error bars overlap). There are no diffusion experiments.
- [Semenov et al.](https://arxiv.org/html/2509.01440v1): decaying to 0.01x beats 0.1x. [Wen et al.](https://arxiv.org/abs/2509.02046): optimizer rankings can flip during lr decay. Both are LLM pretraining.
- [Schaipp](https://arxiv.org/html/2510.19376) is a 23M DDPM U-Net on Navier-Stokes fields, not an image model. There, wsd gave the lowest loss but less stable sample quality, sqrt gave a higher loss but more stable samples, and wsd's best peak lr was about half of cosine's, not higher as the scout said. Loss and sample quality can disagree.
- Trainers: sd-scripts ships warmup_stable_decay; [musubi-tuner](https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/docs/advanced_config.md) and OneTrainer ship REX. No trainer issue or document reports a quality result for diffusion LoRA either way.
- Does not transfer: every number is from pretraining from scratch. In a 20-image run a cooldown mainly lowers the training loss, part of which is memorization. An ICLR 2026 paper (WSO) reports that decayed checkpoints sit in sharper minima. [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) used a constant lr with no cooldown.

**RulzTrainer today.** `build_scheduler` (trainer.py:302-319) offers constant, cosine and linear against a fixed `cfg.steps`. The default is `lr_scheduler: constant` (config.py:290), and the wizard never sets a schedule, so every run ends at full lr with only `lora_ema: 0.98` (config.py:262) as smoothing. A cosine or linear run that stops early on `likeness_patience` decays toward the wrong total. A probe stop breaks the loop at once (trainer.py:1246-1250). `likeness_lr_control` (likeness.py:290-299, `apply_lr_ratio` at 337) is a bounded probe-driven multiplier between 0.5x and 2x, never a decay to 0.01x. Automagic 3/4 get a flat lambda (trainer.py:305-306). Automagic 4's zero-centred vote already anneals "as the signal is consumed", which is a soft self-decay, not a cooldown.

**Fit and risks.**
- No conflict with W8A8, nvfp4, compile, keep blocks, block swap or model_split, because the lr is a Python float in param_groups.
- data_parallel keeps one optimizer, so no change is needed. race_parallel steps each lane's own scheduler (race_parallel.py:254), so the cooldown state must be kept per variant.
- Resume: LambdaLR saves a callable object's `__dict__`, so the schedule must become a small callable class that holds `cooldown_start`.
- Fine-tune windows rebuild the scheduler with `offset=step` (trainer.py:1331), so the cooldown start must live on the trainer.
- Automagic raises its lr back up each step, so a cooldown there must lower `group["max_lr"]` along the curve.
- Drop the beta2-to-0.999 part. From the current 0.99 its half-life is about 700 steps, longer than a cooldown of max(50, 0.15*s), so it would hardly act.
- Skip scoring every racer after its own cooled copy. That costs rounds x racers x ~15% of steps, and racers are already compared on EMA weights.
- Cost per stop: +15% of the stop step. A stop at step 1500 adds 225 steps: about 34 min on Krea 2 nvfp4 (9 s/step), about 19 min on MOSS int8 and about 9 min on Qwen 2.1 int8.

**Implementation path.**
- config.py: let `lr_scheduler` accept `wsd` and `rex`, and add `cooldown_fraction: 0.2`, `lr_min_ratio: 0.01` and `cooldown_on_stop: false`. Consider a `cooldown_shape: sqrt|mirror_cosine|linear07` option, because the EMA is on by default and Dremov found mirror cosine better under averaging.
- trainer.py: turn the `fn` of `build_scheduler` into `Schedule(cfg, offset, cooldown_start=None)` with a `start_cooldown(step, n)` method on every shape.
- At trainer.py:1246, when `cooldown_on_stop` is on and the run is not a race: replace the break by `cooldown_end = step + max(50, int(0.15*step))`. Ignore further stop actions until then, update the status total, and let the existing final probe and `_best` rule keep the better of the EMA-at-stop file and the cooled file. Apply the same hook to the resolution step-up stop.
- Races: wsd per racer only.
- webui.py select, a README paragraph next to lora_ema, and tests in the style of tests/test_likeness.py (schedule values, a cooldown started mid-run, a state_dict round trip).

**How to test.**
- No code first: on Qwen 2.1 int8, rank 8, lr 2e-4, 1500 steps, `probe_every: 100`, `probe_seeds: 4`, `race_drop_lowest: false`, race `lr_scheduler` constant vs cosine vs linear. Judge on the step-1500 probe from likeness.jsonl, not on the champion's best score at any step, because a decay pays off in the last 20%.
- After the code: constant vs wsd vs cosine in the same race. Then compare `cooldown_on_stop` with a plain stop on Krea 2 nvfp4 with `likeness_patience: 3`, same seed, comparing `_best` against the cooled final.
- Stop rule: keep wsd only if it beats constant by more than `likeness_noise_band` (1.0) on two seeds.

### Riemannian r x r preconditioning of the LoRA factor gradients

**Race it. Expected gain: 0 to +1 probe point, plus a wider usable lr band. Gains in steps-to-likeness are unproven, and there is a real risk of a neutral-to-negative result. Confidence: low (0.39). Cost: ~60 lines plus plumbing, about half a day.**

**What it is.** A LoRA's product up@down does not change under (down, up) -> (R down, up R^-1), but Adam's per-coordinate steps do change. Zhang and Pilanci fix this by preconditioning the raw factor gradients before the optimizer, with g_down <- (up^T up + dI)^-1 g_down and g_up <- g_up (down down^T + dI)^-1. These are two r x r solves per module and need no new optimizer state. In theory the same lr is then correct for both factors, which would replace the LoRA+ ratio guess.

**Evidence.**
- [Zhang and Pilanci, ICML 2024](https://ar5iv.labs.arxiv.org/html/2402.02347) (code: [pilancilab](https://github.com/pilancilab/Riemannian_Preconditioned_LoRA)): GPT-2 E2E BLEU 68.9 -> 69.6, 4-bit Mistral-7B GLUE 88.28 -> 89.01. The SD1.5 results are qualitative only, with no CLIP, DINO or face metric. The "lr robustness" claim sweeps the text-encoder lr with the U-Net lr fixed at 1e-4: AdamW gave a usable result only at a text-encoder lr of 1e-6, about 50x lower, not 100x. The repo's scaled-gradient code has an unanswered issue (#3) about dimension errors, so write the method from Eq. 1.
- Contrary, and the one quantitative diffusion test: [AdaPreLoRA](https://arxiv.org/html/2605.08734), Table 6, Mix-of-Show character LoRAs on SD with grid-searched settings, ranks Scaled AdamW last. CLIP 24.51 / FID 34.18 vs plain AdamW 30.68 / 27.80 at scale 1.0, and 24.21 / 48.23 vs 31.47 / 34.15 at scale 0.7. AdaPreLoRA itself beats AdamW by only 0.67 FID and lists DiT and quantized backbones as untested.
- Independent LLM replications show no gain: [Riemannion](https://arxiv.org/html/2507.12142v2) has RPrecAdamW at 86.8 vs Adam 87.1. The [LoRA-RITE](https://arxiv.org/html/2410.20625) tables show ScaledAdam at 54.38 vs 54.33 (Gemma-2B) and 67.52 vs 68.71 (Gemma-7B), 3-5% slower. "A Riemannian Geometry for LoRA" (2026) puts it below AdamW on all five GPT-2 E2E metrics.
- Relatives, not recommended: LoRA-RITE loses to Adam by 2.6 points in Riemannion and costs +7-8% step time. [BaLoRA](https://arxiv.org/html/2605.31484v1) gains at most 0.002 loss at r=8. [LoRA-Pro](https://arxiv.org/html/2407.18242v3) costs +4 GB and +10-12% time.
- What does transfer: the mechanism is backbone-agnostic (the paper ran it on 4-bit Mistral) and almost free. Learning-rate robustness is the one property that independent sources report consistently.

**RulzTrainer today.** The only factor-balancing lever is `lora_plus_ratio` (config.py:369, lora.py:556-568). The identity preset sets it to 4 (config.py:96), and build_optimizer collapses it for Automagic (trainer.py:271-278). The README's own measurement on a 5-image set, face EMA at step 100 (README.md:762-764), gives baseline 74.2 and LoRA+ 4 at 74.7. With LoRA+ 16 the run was unstable. That instability is the regime the preconditioner targets. Nothing touches B^T B or A A^T. Automagic 4's sensitivity balance is a per-element cap, still diagonal, so the two are complementary. One correction to the scout: automagic3 does not adapt a per-parameter lr. It nudges one lr per group from pooled sign votes, so it does not duplicate the preconditioner.

**Fit and risks.**
- Insertion point: after `self.dp.reduce_grads()` (trainer.py:1131), before `clip_grad_norm_` (1133) and `optimizer.step()` (1138), so the clip sees what Adam will see. It is stateless, so it composes with adamw, adamw8bit (bnb reads p.grad), Automagic 3/4, lora_ema, DoRA, resume, per-image lr and likeness_lr_control.
- No interaction with W8A8, nvfp4 or compile (it runs eagerly after the compiled backward), or with keep blocks and block swap. model_split: batch the solves per (device, rank). data_parallel: run it once on the primary.
- Races: the helper must accept both LoRALinear and the race master views (race.py:531-543). Skip it when `self.ft` is set, for kind lokr or oft, and in the reward stage.
- Init: lora_up starts at zero (lora.py:286-296), so (up^T up + dI)^-1 is about 1/d early. That pollutes Adam's v history and starves the down factor. Normalise the Gram matrix to unit mean eigenvalue (divide by tr/r + d) so only the direction mixing remains and max_grad_norm keeps its meaning.
- Export can see the factors drift to different scales. Log max|A| / max|B| at save time.
- Time: about 10-30 ms per step against 2.5-9 s steps. Memory: kilobytes.
- Leave BaLoRA out. Its per-step SVD re-factorisation breaks lora_ema, which averages each factor separately (ema.py:21-24), and the Automagic sign history.

**Implementation path.**
- New `rulztrainer/optimizers/lora_precond.py` with `precondition_lora_grads(pairs, delta, normalize=True)`. It groups pairs by (shape, device) and runs batched `torch.linalg.cholesky_ex` plus `cholesky_solve` in fp32, with an `inv_ex` fallback.
- config.py: `lora_precondition: none|scaled` and `lora_precondition_delta: 1e-2`, a fraction of the mean Gram diagonal with a 1e-6 floor. Add both to RACE_KEYS "LR & optimizer" (race.py:50) and history.py's key list.
- trainer.py: build pairs beside `params` (plain run and per race variant), call the helper before the clip, and log on resume when the value differs.
- Do not force `lora_plus_ratio` to 1. Let the race decide.
- Tests: an invariance test (A, B and R A, B R^-1 give the same first-order delta W), finite grads with up == 0, and the normalisation bound.

**How to test.** Race on Qwen 2.1 int8 at 0.5 MP, same seed, `race_drop_lowest: false`, probes every 50 steps:
- (a) adamw8bit 1e-4 with LoRA+ 4 (the identity preset)
- (b) adamw8bit 1e-4, preconditioned, LoRA+ 0
- (c) adamw8bit 4e-4, plain
- (d) adamw8bit 4e-4, preconditioned

Read the peak ArcFace face and body scores, the step of the peak, and whether (c) collapses while (d) holds. The high-lr pair decides the only claim with hard support. Then rerun the winning pair on Krea 2 nvfp4 and MOSS int8, and race automagic4 with and without the preconditioner. Stop rule: a result inside the seed-twin spread is no gain. Keep it only for robustness if it gives less than +1 point at equal lr.

### Pair-aware (gauge-invariant) Muon for LoRA, MOSS first

**Prototype. Expected gain: 10-30% fewer steps to 95% of the best smoothed combined score and 0-3 final points (the code-fit verifier's estimate), with a real chance of a null or negative result on AdamW-pretrained bases. Unmeasured for any image LoRA. Confidence: medium (0.45) for the pair-aware form; low (0.3) for per-factor Muon on MOSS. Cost: ~200 lines written from the paper's equations, 2-3 days.**

**What it is.** The idea is to apply Muon's spectral steepest descent to the rank-r product up@down instead of orthogonalising each factor separately. Per-factor Muon, which OneTrainer ships, is not invariant to rescaling the factors.
- sMuon builds a rank-2r surrogate of the full gradient from g_up and g_down. It takes its matrix sign with a 2r x 2r Newton-Schulz, then solves for least-squares factor steps, with momentum 0.9 and an RMS-matched lr of 0.2*sqrt(d_in*d_out/r).
- LoRA-Muon whitens each factor by the other factor's Gram matrix.
- Riemannion retracts with a truncated SVD and is the most expensive of the three.
- MOSS-Image was pretrained with hybrid Muon-AdamW, which makes MOSS the first model to try.

**Evidence.**
- [sMuon, Anson et al., Aug 2026](https://arxiv.org/html/2608.14492v1): ReLoRA (162M, FineWeb, rank 64) final val loss is AdamW 3.83, per-factor Muon 3.70, LoRA-Muon 3.68, Riemannian 3.66, sMuon 3.66. Optimizer step on an A100 at r=16: AdamW 0.66 ms, sMuon 8.7 ms (0.62%). In LoRA SFT on AdamW-pretrained bases sMuon is at parity or slightly behind: Llama-3.2-3B about 62.9 vs AdamW 63.6, Qwen2.5-3B about 73.8 vs 74.2. Clear wins appear only on Muon-pretrained Moonlight. There is no code. The paper initialises down=0 with a semi-orthonormal up, the opposite of RulzTrainer.
- [Riemannion](https://arxiv.org/html/2507.12142v2): Llama-3-8B rank-16 commonsense 88.1 vs Adam 87.1, per-factor Muon 83.0. This is the only diffusion experiment: SD2 DreamBooth, ranks 4/8/16, DINO + CLIP, where it reports better concept preservation across lrs and faster convergence at low rank. The result is figures only, with no table.
- [LoRA-Muon, EleutherAI](https://arxiv.org/html/2606.12921v1): a gauge-invariance proof, and rank 32 reaches 1.776 vs dense Muon 1.789 on TinyShakespeare.
- [Sincari et al.](https://arxiv.org/html/2609.24678v1): LoRA rank 8 on T5, IncLoRA+Muon 81.24 vs +AdamW 75.83, backward transfer -9.70 -> -2.73. Muon spreads updates over about 7 effective singular directions, AdamW over 1.4-1.8. Text only, and only 0.24 points over the best AdamW config.
- [MOSS-Image report](https://arxiv.org/html/2610.23374): hybrid Muon-AdamW, both at lr 5e-5, Muon momentum 0.95, 5 Newton-Schulz steps. In a 1B ablation at 256 px and lr 1e-4, loss 0.41 came at 257K vs 385K steps (-33.2%) and 0.403 at 939K vs 1317K (-28.6%). This is pretraining from scratch and does not transfer. [KohakuMossTrainer](https://raw.githubusercontent.com/KohakuBlueleaf/KohakuMossTrainer/main/docs/concepts/training.md) documents Muon only for full fine-tunes: qkv split in 3, ffn_up_gate in 2, update scale 0.2*sqrt(max(rows, cols)). [Kromoss](https://huggingface.co/lodestones/Kromoss) has no AdamW control.
- Contrary: "Can Muon Fine-tune Adam-Pretrained Models?" (ICML 2026) finds optimizer mismatch hurts full fine-tunes, while LoRA narrows the gap. OneTrainer PR #1314, a pair-aware attempt, was closed without measurements. The [lilting.ch write-up](https://lilting.ch/en/articles/muon-image-lora-feasibility) found no measured image-LoRA results. LoRA-RITE (gauge-invariant) lost to Adam by 2.6 points, so invariance alone does not guarantee a win.
- Dense diffusion: [Schaipp](https://arxiv.org/html/2510.19376) found Muon 18% lower final loss than tuned AdamW on a 23M U-Net trained from scratch, at a 1.45x step cost.

**RulzTrainer today.** There is no Muon anywhere. `build_optimizer` (trainer.py:259-292) has adamw, adamw8bit, adafactor, automagic3 and automagic4. The plumbing is in place: fp32 lora_down [r,in] and lora_up [out,r] with scale = alpha/rank (lora.py:228), down and up groups in `lora_param_groups` (lora.py:556), and race master views that expose `.lora_down.weight` and `.lora_up.weight` (race.py:531-541). Automagic 4 already stacks same-shape tensors into buckets, and the same pattern makes per-pair work cheap.

**Fit and risks.**
- No conflict with W8A8, nvfp4, compile, keep blocks, block swap or model_split: the step runs after backward (trainer.py:1129-1138) as fp32 torch matmuls, with no triton and no NCCL, so it is fine on Windows.
- State must sit in `opt.state[p]` (momentum only), so move/offload (race.py:625-648), relay `follow_leader` (race.py:677-697) and save/resume keep working.
- `apply_lr_ratio` and the per-image lr stay meaningful because the step is linear in lr. `lora_plus_ratio` becomes meaningless: collapse it and warn, as Automagic does. Grad clipping becomes nearly a no-op.
- DoRA's dora_scale and the LoKr/OFT parts go to an auxiliary AdamW group. Krea's attn_gate and text_fusion and Qwen's txt_in are LoRALinear pairs and are handled as pairs.
- Init: up=0 in RulzTrainer (lora.py:286-296), so d_down=0 on step one. Use jitter (1e-4 * tr/r). A per-optimizer init swap would break the race's "same init for the same shape" rule (race.py:507-514).
- MOSS: the fused `proj.qkv` up factor (out 3*dim) must be orthogonalised in 3 chunks and ffn_up_gate in 2 (export already splits by name, lora.py:804-806).
- VRAM: fp32 momentum is 4 B/param, about 100-200 MB at rank 8-16 on Krea 2. That is half of adamw but double adamw8bit, so there is no saving against the default.
- Time: about 300 pairs in a Python loop costs 40-120 ms per step (about 1% of a 9 s Krea step, 2-4% of a 2.5 s Qwen step), under 1% once batched with bmm.
- The reward stage builds its own AdamW (trainer.py:2766). Fine-tune windows should fall back to adafactor.

**Implementation path.**
- New `rulztrainer/optimizers/lora_muon.py` with `LoRAMuon(torch.optim.Optimizer)`. A "lora_pairs" group carries `pairs=[(i_down, i_up, scale)]`; an "aux" group is stepped by an embedded AdamW.
- Options: `lora_muon_momentum: 0.9`, `lora_muon_rms_match: true`, `lora_muon_plus_norm: none|row|col` (the [Muon+](https://arxiv.org/html/2602.21545v1) normalisation, -0.4 to -2.0 perplexity on LLMs), and a per-param `split` hint for MOSS qkv/ffn_up_gate.
- lora.py: `lora_muon_param_groups(injected, lr)` that works on wrappers and race views alike.
- config.py: `lora_muon` in OPTIMIZERS (config.py:22), the build_optimizer branch, and webui selects (they read OPTIMIZERS).
- tests/test_lora_muon.py in the style of tests/test_automagic.py: tiny MOSS, 8 steps, a state round trip, a follow_leader copy, and gauge invariance (scale down by c and up by 1/c, then assert the change to up@down is the same).
- Do not ship per-factor Muon, the scout's G69 sketch. Riemannion has it at 83.0 vs Adam 87.1, and pairing is the point.

**How to test.**
- Race on MOSS int8 first, then Krea 2 nvfp4 and Qwen 2.1 int8, at rank 16, 0.5 MP, 1500 steps, probes every 100, same seed, drop-lowest off, rows:
  - adamw8bit 1e-4
  - lora_muon 1e-4 (RMS-matched)
  - lora_muon 2e-4
- Read the steps to 95% of the best smoothed combined score, the final best score with face and body separately, and probe images with varied prompts for base drift.
- Repeat as a 4-seed relay.
- Stop rule (from the MOSS verifier): adopt only if lora_muon reaches the adamw8bit peak at least 15% sooner, or beats the peak by more than the seed spread, on both seeds, with no worse prompt adherence. If it ties on MOSS, skip Krea and Qwen.

### Warmup length and the LoRA+ default (no code), shape-aware lr (small)

**Do it (the warmup race). Expected gain: a few percent earlier convergence from dropping warmup; the static shape multipliers 0-1 point (possibly 10-20% fewer steps if the LLM LoRA+ speedup transfers, unproven). Confidence: low (0.33). Cost: no code for the warmup race; about half a day for `lr_scaling: shape`.**

**What it is.** LoRA's up factor starts at zero, so the effective lr on the product is low early. [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) calls this an implicit schedule, which argues against an extra warmup. LoRA+ sets lr_up > lr_down. A static per-projection multiplier would follow the matrix shape. Krea 2's GQA gives k/v 6144->1536 against q/o 6144->6144 (krea2/model.py), and the FFN is 6144->16384->6144.

**Evidence.**
- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/): optimal LoRA lr about 10x full fine-tuning, roughly rank-independent (the optimum shifts by less than 2x from rank 4 to 512). The "15x for runs under ~100 steps" figure is labelled anecdotal and refers to whole runs, not the first 100 steps. Setting: Llama/Qwen SL and RL.
- [LoRA+](https://arxiv.org/abs/2402.12354): 1-2% better and up to 2x faster on LLMs. The best ratio varies: about 2^4 on RoBERTa, 2^1-2^2 on Llama-7b. It does not help easy tasks, and the authors advise lowering lr_A about 2x with a large ratio. [musubi](https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/docs/advanced_config.md) says to start around 4, without data.
- [rsLoRA](https://arxiv.org/abs/2312.03732): helps only at ranks of 32-128 and up, at a fixed lr of 5e-5. Low ranks were not tested.
- Spectral targets ([Advanced_Optimizers](https://raw.githubusercontent.com/koratahiu/Advanced_Optimizers/main/docs/advanced_features.md)): no benchmarks. The scout's "multiply lora_up by sqrt(d_out/d_in)" does not implement these targets. The static Adam equivalent is lr proportional to 1/sqrt(d_in) on both factors. [Muon is Scalable](https://arxiv.org/abs/2502.16982) rescales toward Adam's shape-independent RMS. It names GQA kv heads and wide MLPs as cases where equal-spectral-norm updates go wrong.

**RulzTrainer today.** LoRA+ is `lora_plus_ratio` (config.py:369), and the identity preset uses 4 (config.py:96). The README measurement (README.md:762-764) has LoRA+ 4 at 74.7 vs baseline 74.2, inside the probe band, with LoRA+ 16 unstable. So "start at 4, not 16" is already the measured default. `warmup_steps: 50` (config.py:291) ramps the lr from about 0. use_rslora is hard-coded False in the PEFT template (lora.py:99), and scale = alpha/rank (lora.py:228). Automagic 4's sensitivity balance (`min(1, reference/sensitivity)` per element) is already a sensitivity-aware step cap for the automagic path, so shape multipliers would only matter for the adamw family. Automagic collapses the LoRA groups anyway (trainer.py:271-278).

**Fit and risks.**
- Optimizer-side only, so W8A8, nvfp4, compile, keep blocks, block swap and data_parallel are untouched.
- A new grouping must be named in the resume check (trainer.py:1495-1501) and added to RACE_KEYS.
- Do not split q/k from v/o under automagic3. Its docstring pools Q and K because coupled tensors with separate lrs diverge.
- Computed multipliers sqrt(d_out/d_in):
  - Krea 2: q/o/attn_gate 1.0, k/v 0.5, ff gate/up 1.63, ff down 0.61.
  - Qwen 2.1: attention 1.0, gate/proj 1.73, out 0.58.
  - MOSS: attention 1.0, ffn_up_gate 2.31, ffn_down 0.61.
  This 0.5-2.3x spread is smaller than LoRA+ 4.
- Skip rsLoRA. At a fixed rank it equals setting `lora_alpha = alpha*sqrt(rank)`, which every race row can already do. At rank 16 that means 4x the effective lr, the regime already found unstable.
- Skip the 1.5x early boost. It has no source and it fights the warmup. If it were ever built it would belong in the scheduler lambda, not in `apply_lr_ratio`, which compounds with the probe lr control.

**Implementation path.** Only if the race below shows something:
- `lora_param_groups(..., shape=True)`: m = clamp(sqrt(out/in), 0.5, 4) multiplied into the up lr, bucketed into at most about 6 named groups (`lora_up_x1.63`).
- `lr_scaling: none|shape` in config.py, ignored with a warning under automagic.
- RACE_KEYS, and a test beside `test_lora_plus_groups_and_resume_check` (tests/test_identity.py:153). The tiny MOSS model's ffn_up_gate gives exactly 2x.
- A README sentence that `lora_alpha = rank*sqrt(rank)` reproduces rsLoRA.
- Side check: the full fine-tune path trains at the same `learning_rate` as a LoRA (trainer.py:1323-1330). By the 10x rule the wizard should propose about lr/10 for finetune.

**How to test.**
- No code: on Qwen 2.1 int8, identity preset, 1500 steps, probes every 100, same seed, race `warmup_steps` 0 / 50 / 200.
- With code: baseline (ratio 0), LoRA+ 4, LoRA+ 4 + shape, and shape only, as seed twins.
- Stop rule: drop shape scaling if it does not beat LoRA+ 4 by more than the band on two seeds.

### Average the run's own checkpoints (PMA) through the soup

**Race it (zero code). Expected gain: +0-1 probe point at the end of a constant-lr run, or a steadier final file. Confidence: low (0.3). Cost: one afternoon of races; about 1 day if automated.**

**What it is.** Average the last N step checkpoints of one run, uniformly (SMA) or with linearly rising weights (WMA), on the materialised deltas. The PMA paper's claim is that averaging constant-lr checkpoints matches an annealed run, so averaging could stand in for an lr decay.

**Evidence.**
- [PMA, NeurIPS 2025](https://arxiv.org/html/2505.12082v3): HumanEval 31.1->36.6 (1.3B/13B MoE), 54.3->61.6 (10B/100B), 50.6->57.9 and GSM8K 85.9->91.3 (dense 70B). N=3 is about 1 point below N=15, and N=10 is the compromise. Gains shrink once the lr is annealed. All of this is LLM pretraining with merge intervals of 4-80B tokens.
- [Krea 2 report](https://www.krea.ai/blog/krea-2-technical-report): Krea used PMA because it "achieves performance comparable to EMA" while avoiding EMA's memory cost on a full 12B model, not because it scored better. That reason does not apply to a LoRA, whose EMA costs about 0.2 GB. Krea saw "no significant differences between merging methods" and diminishing returns in SFT.
- [Fizgig KREA2.md](https://github.com/shootthesound/Fizgig/blob/master/docs/KREA2.md): EMA 0.98 lifts late-epoch likeness on Krea 2 and MiniMax H3, while wider windows (0.99, 0.995) lagged. That supports the EMA RulzTrainer already has, not PMA.

**RulzTrainer today.** `lora_ema: 0.98` is on by default (config.py:262, ema.py), so every saved checkpoint is already an average over about 50 images. The seed soup (soup.py, README soup section) averages materialised deltas (up@down*scale), re-factors by SVD and refuses OFT and DoRA. It takes any file list, so `soup_files` (config.py:422) with a run's last N step checkpoints is PMA-SMA today, and `scripts/soup.py --weights 1,2,3,4` gives WMA. With steps 0 the probes score each file, the plain average and strength variants (trainer.py:3126-3145). The default constant lr is the regime where PMA's argument applies.

**Fit and risks.**
- Works on files only, so no interaction with W8A8, nvfp4, compile, keep blocks, block swap or data_parallel.
- `save_every: 250` (config.py:305) is too coarse. N=4 would span 1000 steps and mix in undertrained files, the same failure Fizgig saw with EMA 0.995. Use save_every 50-100 and `keep_checkpoints: 0`.
- The window must not cross a resolution step-up, an lr-control change, a curriculum phase, a look warm-up or a relay handoff.
- LoKr needs a high soup rank.
- Skip PMA-init (restart from the average of 3 saves on a spike) unless the loss watch shows real divergence. Nothing rolls back on spikes today, and spikes are rarely fatal at lr 1e-4.

**Implementation path.** Only after the race below shows a gain:
- `checkpoint_average: N` and `checkpoint_average_weights: uniform|linear` that call the soup averaging at the end of a run on the last N post-event checkpoints, save `<job>_avgN.safetensors`, and probe it at strength 1.0 and 1.15 beside `_best`.
- A WebUI field in Saving & samples with a hint to lower save_every.

**How to test.**
- Train Krea 2 nvfp4 and Qwen 2.1 int8 at rank 16, 2000 steps, constant lr, lora_ema 0.98, `save_every: 100`. Run the soup stage with `soup_files` set to the last 3, 5 and 10 checkpoints, plus a WMA file.
- Score against `_best` with `probe_seeds` of at least 4, ArcFace and DINOv2 body. Use 3 subjects x 2 seeds.
- Stop rule: adopt only if the average beats `_best` by more than 2x `likeness_noise_band` on at least 4 of 6 runs without a strength boost.
- Secondary: rerun with lora_ema 0 to see whether the two are redundant.

### Prodigy (prodigy-plus-schedule-free) as an lr finder

**Skip-ish. Expected gain: race efficiency only. One ~300-step run per new architecture could replace a 3-5 point lr sweep, if d0 needs no tuning, which DiT reports dispute. Likeness at best equals tuned adamw8bit. Confidence: low (0.38). Cost: ~1 day plus an optional dependency.**

**What it is.** Prodigy estimates the distance to the solution, d_t, from sum <g_k, x0 - x_k> and uses d_t as Adam's lr. d grows geometrically until the correlation stops, and it never decreases. [prodigy-plus-schedule-free 2.0.1](https://pypi.org/project/prodigy-plus-schedule-free/) adds factored second moments, StableAdamW (do not combine it with grad clipping), split_groups, prodigy_steps, d_limiter and optional Schedule-Free. The author marks it experimental.

**Evidence.**
- [Prodigy](https://arxiv.org/abs/2306.06101): "close to hand-tuned Adam" on vision and NLP benchmarks.
- [Schaipp](https://arxiv.org/html/2510.19376) (23M fluid-flow U-Net): Prodigy roughly matches AdamW's second-best tuned lr. ScheduleFree leads early, ends slightly worse and gives lower sample quality unless a cooldown is added. [Semenov](https://arxiv.org/html/2509.01440v1): Prodigy is weak at small batch.
- DiT reports from the trackers: Flux-dev with a frozen base "immediately collapses" at lr=1 and works only at 2e-5, where d stays constant (konstmish/prodigy #39). SD3.5 Medium under-learned until d0=1e-5 (#27). A Flux fine-tune did not converge at the default d0 (P+SF #22). "No tuning" becomes "tune d0".
- Refuted member: per-factor D as a learned LoRA+ ratio. In the author's tracker (#23, #31), one group's d ran away to about 1e13 with LoRA+ two-group setups, and he recommends `split_groups_mean=True`. OneTrainer and SimpleTuner ship it that way.
- VRAM is about neutral, or worse, against adamw8bit. prodigy_steps frees only the ~9%-sized s/p0 slices.
- [diffusion-pipe](https://raw.githubusercontent.com/tdrussell/diffusion-pipe/main/examples/main_example.toml) and the README measurements both point to fixed-lr adamw8bit as the best final result.

**RulzTrainer today.** There is no Prodigy, D-Adaptation or Schedule-Free, and none is in the venv. The role is now mostly covered by Automagic 4 (automagic4.py, README "Self-tuning optimizers"). Its zero-centred vote holds the lr on noise, climbs on a trend and "walks a too-high start down". On Krea 2 rank 16 it went 1.7e-4 -> 6.0e-4 at step 70 -> 3.2e-4 at 160 at 1.6 s/step, against 2.2 s for Automagic 3. The wizard already carries per-model recipes (webui.py:3097-3102). What Prodigy would add is a d value to read out and seed adamw8bit racers with, for models that have no recipe (Flux 3 Klein, MiniMax H3).

**Fit and risks.**
- No conflict with W8A8, nvfp4, compile, keep blocks, block swap or model_split.
- Skip the clip at trainer.py:1133 for this optimizer.
- With Schedule-Free on: `optimizer.eval()` must happen inside `_averaged()` before `dp.sync_weights` (trainer.py:1404-1415), and lora_ema must be 0, because the EMA would shadow the fast iterate.
- The relay `follow_leader` does not copy group-level d.
- Exclude it from finetune windows, which rebuild the optimizer every window. Do not expose `get_avg_learning_rate`, which `apply_lr_ratio` treats as Automagic.
- State is not resumable across P+SF versions.

**Implementation path.** Only if Automagic 4 does not find a usable lr on a new model:
- Optional `optimizer: prodigy` with `prodigy_d_coef: 1.0`, `prodigy_steps_frac: 0.25`, `prodigy_schedulefree: false`.
- `split_groups=True, split_groups_mean=True, d_limiter=True, use_stableadamw=True`, warmup 0, `d0` exposed with a 1e-5 hint for DiTs.
- Report d per group in status.

**How to test.** On Qwen 2.1 int8, race adamw8bit 1e-4 + LoRA+ 4 vs adamw8bit 2e-4 vs plain prodigy vs automagic4, same seeds. Compare the face/body EMA at 500 and 1000 steps and the step at which each row reaches 95% of the best final score, and log d per group against the winning fixed lr. Stop rule: drop it if automagic4 ties prodigy, since automagic4 needs no extra plumbing.

### Spike-aware second moment (diagnostic first)

**Do it (the diagnostic, about an hour); Watch the optimizer change. Expected gain: lower probe-to-probe variance on heterogeneous sets, 0-1 point, no higher ceiling. Confidence: low (0.25). Cost: diagnostic about an hour; the spike guard about a day.**

**What it is.** Kourkoutas-beta lowers beta2 per layer while a "sunspike" (the pooled grad norm over its EMA) is present, so a spike passes through v quickly. SPAM clips coordinates where g^2 > theta*v and periodically resets momentum.

**Evidence.**
- [Kourkoutas-beta](https://arxiv.org/abs/2508.12996): -38% bits-per-char vs Adam beta2=0.95 and -58% vs 0.999 on small-enwik8. Those Adam baselines were partly diverging (1.639 vs 2.637 vs 3.906; medians give about 31%). Setting: a 6-layer char-LM on Apple MLX at batch 4, about 6% slower. Small-batch adapter SFT is listed as future work. There is a single author and the reference code is MLX-only.
- [SPAM](https://arxiv.org/abs/2501.06842): theta 5000, resets every 500-1000 steps with a 150-300 step re-warmup, sized for runs of 10k steps or more. Full-parameter LLaMA2-7B scores 66.7 vs Adam 63.4, and vision gains are 0.15-0.3 points. The same group later says SPAM "struggles to stabilize gradient norms".
- Corrections: pytorch_optimizer ships SPAM but not Kourkoutas, which is adv_optm-only ([adam_settings.md](https://raw.githubusercontent.com/koratahiu/Advanced_Optimizers/main/docs/adam_settings.md)). bnb does read `group["betas"]` per group per step, so per-block beta2 works with adamw8bit by splitting param groups. No per-group optimizer instances are needed.

**RulzTrainer today.** Betas are hard-coded at (0.9, 0.99) for adamw and adamw8bit (trainer.py:263, 267). `clip_grad_norm_` at trainer.py:1133 (max_grad_norm 1.0, config.py:292) discards its return value, so the pre-clip norm is never logged. The loss watch (losswatch.py) scales the step lr but does not touch v. Automagic 3/4 have no first moment, and their beta2 of 0.999 remembers a spike for about 1000 steps.

**Fit and risks.**
- In flow matching the grad norm swings 2-4x with the sampled sigma. An un-normalised sunspike would dip beta2 every few steps and turn the scheme into "Adam with beta2 about 0.92". Normalise by a per-sigma-bucket EMA, reusing the loss-watch bucket idea.
- The 1.0 clip already bounds what enters v on a spike step.
- Bias correction uses the current beta2, so gate the scheme with a warmup of at least 150 steps.
- No conflict with W8A8, nvfp4, compile, keep blocks, block swap or model_split, since per-block norms are local.
- The relay copies only per-param state, so keep the EMA scalars in the state of each group's first param.

**Implementation path.**
1. Diagnostic: keep the return of `clip_grad_norm_` and log `grad_norm` (pre-clip) and `clip_active` in the status payload and in per_image_loss.jsonl, which already records sigma.
2. Only if real rare spikes remain after sigma normalisation: add `spike_guard: bool`, `spike_threshold: 3.0`, `beta2_min: 0.88` and `spike_warmup_steps: 150`, with per-block groups via `lora_param_groups_by`, a block-level clip, and RACE_KEYS. Skip the SPAM momentum reset for runs under 5k steps. Skip the cooldown beta2 piece.

**How to test.** Stage 0 is one normal run with the diagnostic. If the norm ratio is explained by sigma, or the clip fires on most steps, stop there. Stage 1 races adamw8bit vs + block clip vs + sigma-normalised beta2 dip, with automagic4 as a control, on a deliberately heterogeneous set with flips and auto crops on, 3 seed rows. Compare the smoothed face/body curves and their std.

### Cautious update mask and Cautious Weight Decay

**Skip-ish. Expected gain: at most a few percent fewer steps, likely inside probe noise; CWD numerically moot at weight_decay 0.01. Confidence: low (0.32). Cost: 60-80 lines, about half a day.**

**What it is.** The Cautious mask zeroes the coordinates where the proposed update u disagrees in sign with the current gradient (m = (u*g > 0)) and rescales the rest by 1/mean(m). Cautious Weight Decay (CWD) decays only where sign(p) == sign(u).

**Evidence.**
- [Cautious Optimizers, ICLR 2026](https://arxiv.org/html/2411.16085v4): FineWeb-Edu perplexity 15.07 -> 14.92 at 520M, Mini-ImageNet ViT 72.11 -> 73.52. About 3% lower throughput, and the mask keeps about 0.55 of coordinates at large batch. No diffusion and no LoRA.
- [CWD, ICLR 2026](https://arxiv.org/abs/2510.12402): OLMo-1B val loss 2.65 -> 2.56, ViT-S/16 78.84 -> 79.45. LoRA fine-tuning on Mistral-7B gives +0.35 MMLU and +0.75 AGIEval, with the inner-product variant sometimes below baseline.
- Contrary: [Grams](https://arxiv.org/abs/2412.17107) reports C-Adam below Adam on PEFT (16.62 vs 17.80). [Wen et al.](https://arxiv.org/abs/2509.02046) find Cautious slightly worse than AdamW at 130M at some budgets. koratahiu/OneTrainer deprecated the cautious mask. A prodigy-plus issue (#44) reports loss explosions on diffusion. [HeavyBall](https://github.com/HomebrewML/HeavyBall) ships `caution=True`. CWD has one qualitative positive Flux2 Klein report.

**RulzTrainer today.** Nothing like it exists. Refuted member: the scout's pairing with automagic3. Automagic 3 and 4 have no first moment: the update is rsqrt(v)*grad (automagic3.py:445-470), and in v4 it is multiplied by the balance. The mask (u*g > 0) is therefore identically true there and does nothing. Only torch `adamw` has the momentum the mask needs, and bnb AdamW8bit's fused kernel cannot be masked. CWD fails at RulzTrainer's magnitudes: lr 1e-4 x wd 0.01 shrinks the weights 0.3% over 3000 steps, and up starts at zero.

**Fit and risks.** An fp32 path costs +135-270 MB of VRAM over adamw8bit at rank 8-16 on Krea 2, which is tight only on Krea 2 nvfp4 at 1 MP. There is no compile, quantization or Windows issue. Keep `cautious`/`cautious_wd` as flags of one class so the relay's same-type check (race.py:692) still copies state.

**Implementation path.** `rulztrainer/optimizers/cautious_adamw.py`: `CautiousAdamW(torch.optim.AdamW)` keeping the step/exp_avg/exp_avg_sq keys, with clamped `m.mean()` (min 1e-3), registered as `adamw_c`. Do not touch automagic3 or automagic4.

**How to test.** On Qwen 2.1 int8, race adamw vs adamw_c (cautious on, cautious_wd off), same seed, 1500 steps, probes every 50, drop-lowest. The metric is the step at which the face EMA reaches the baseline's final score. Race cautious_wd only together with weight_decay 0.1-0.5. Stop rule: if the gap is inside the band, drop it.

### AdEMAMix 8-bit (slow-EMA momentum)

**Skip-ish. Expected gain: within the probe noise band (0 to +1-2 points), with a risk of slower early progress. Confidence: low (0.23). Cost: ~25 lines (bnb class already installed).**

**What it is.** AdEMAMix adds a second, slow EMA of the gradients (beta3 0.999-0.9999, weighted by alpha 4-10) to Adam, with alpha and beta3 warmed up over the run.

**Evidence.**
- [AdEMAMix, ICLR 2025](https://arxiv.org/html/2409.03137v2): a 1.3B model on 101B tokens matches AdamW on 197B. The paper says it is weaker in short runs, where beta3 0.999 wins only at 32k-64k iterations. ViT-86M on ImageNet-1k shows no test gain when data is small relative to capacity. "Forgetting" there means loss on previously seen pretraining batches.
- [Semenov](https://arxiv.org/html/2509.01440v1): best overall at 124M-720M, but it trails at short horizons with small batches.
- [Simplified-AdEMAMix](https://arxiv.org/abs/2502.02431): a 150M preliminary result only. Its large-batch match used alpha=0.
- Trainer adoption ([SimpleTuner](https://raw.githubusercontent.com/bghira/SimpleTuner/main/documentation/OPTIONS.md), ai-toolkit, OneTrainer, [pytorch_optimizer](https://github.com/kozistr/pytorch_optimizer)) comes with no published results. OneTrainer's adv-optimizer author replaced Simplified-AdEMAMix with Nesterov momentum.
- Refuted member: the anti-forgetting rationale. The slow EMA protects past training batches, not the base model's behaviour. Isolation (DOP), the anchor and the reward KL do that job. The "long reward stage" target does not exist either: the reward stage uses its own AdamW (trainer.py:2766) for tens of steps.

**RulzTrainer today.** Not present. bitsandbytes 0.49.2 is installed and ships `AdEMAMix8bit`. lora_ema 0.98 already smooths the iterate.

**Fit and risks.**
- The relay restarts state across optimizer types (race.py:692), which empties the slow EMA, so race with drop-lowest only.
- Exclude it from finetune.
- `t_alpha = t_beta3 = steps` conflicts with probe early stops.
- bnb issue #2075: the 32-bit classes are broken, so use `AdEMAMix(optim_bits=32)` or `AdEMAMix8bit`.
- VRAM +50-150 MB (8-bit).

**How to test.** Cheaper first: race adamw8bit with betas (0.95, 0.995) against (0.9, 0.99). This needs a `adam_betas` option, since the betas are hard-coded today. If that moves the probes, race ademamix8bit (beta3 0.999, alpha 4, warmup) vs ademamix8bit (no warmup, alpha 2) on Qwen 2.1 int8, rank 16, 2000 steps, two seeds. Stop rule: a single-seed gap under about 2 points is noise.

## Fewer steps to likeness: LoRA parameterization, initialization, timestep and noise sampling

This section covers levers that change where a LoRA starts (gradient-informed init, warm start from a prior), what shape it has (HRA, T-LoRA, DeLoRA, SingLoRA, ABBA, FouRA, modulation targets), which blocks it trains when (progressive block window), and how sigma and noise are drawn and weighted (stratified timesteps, x0/eps loss weights, nearest-noise selection, contrastive flow matching). The verifiers checked all 14 items against the papers and the RulzTrainer code. None has a measurement on a flow-matching DiT, on faces, or at batch 1. Overall verdict: no item here is a proven step saver. The worthwhile ones are cheap races. A static mid-block window, which needs no code, is the only lever with a computable wall-clock gain (20-30% on Krea 2 / Qwen 2.1). Stratified timesteps are a free variance cut. Gradient init and an HRA adapter kind are the two builds worth one A/B each. The parameterization variants (DeLoRA, SingLoRA, ABBA, FouRA) and the noise tricks (KNN noise, contrastive FM) are expected to land inside probe noise.

Throughput figures used below come from the verifiers: Krea 2 nvfp4 about 9 s/step, MOSS int8 about 5 s/step, Qwen 2.1 int8 about 2.5 s/step at 0.5 MP.

### Progressive (and static mid) LoRA block window

**Race it (the static mid window needs no code; the schedule is about 1 day). Expected gain: 20-30% wall-clock per run on Krea 2 / Qwen 2.1, about 15% on MOSS, if steps-to-likeness does not grow (unmeasured). Confidence: low. Cost: zero for the static race; config, a window helper and re-calibration at stage boundaries for the schedule.**

**What it is.** Train the LoRA on a contiguous window of blocks for the first part of the run and widen it toward the full configured set at later stages. Blocks before the first trained block run forward-only: autograd stops there and they need no stored activations. AutoProg-Zero is the cited precedent. It used 4 stages and scored candidate sub-networks with NTK condition number, ZiCo and runtime.

**Evidence**
- [AutoProg-Zero, arXiv 2410.00350](https://arxiv.org/html/2410.00350v1) reports 0.39x runtime for full fine-tuning of DiT-XL/2 (FID 12.19 vs 21.05 Flowers, 18.43 vs 25.31 ArtBench; 8x A800, batch 256, 240K steps) and SD1.5 (one TITAN RTX), and 0.35x for DreamBooth SD1.4 (DINO 0.874 vs 0.849).
- Does not transfer. The released code ([AutoProg-Zero repo](https://github.com/changlin31/AutoProg-Zero)) unfreezes an interleaved stride of blocks (stage 1 = 0, 4, 8, ...), so backward runs through the whole network in every stage. It also lowers input resolution per stage (64 px at stage 1 for 256 px training), so much of the 0.39x comes from fewer tokens, not from freezing. The "frozen prefix stops backward" mechanism is RulzTrainer's own, not the paper's. The paper never combines the schedule with LoRA, and its LoRA baselines are badly tuned (FID 164/81). It is a single-version arXiv preprint with a text-overlap admin note, and the repo has a substring bug in block selection.
- Refuted member: the NTK/ZiCo proxy. It costs forward/backward passes that take hundreds of batch-1 steps to pay back. Drop it.

**RulzTrainer today.** `lora_blocks` is static (config.py:239). The backward skip is implicit, and checkpointing.py:35 builds keep candidates from min(blocks). `backprop_blocks()` (lora.py:451) returns None (no skip) when any LoRA sits before the blocks. Krea 2's default targets include `text_fusion` (archs.py:355), so on the main model the skip is off today. `LoRALinear.trainable` already turns wrappers off (lora.py:382; the race uses it at race.py:579-581), so no re-injection is needed. A race can give each row its own `lora_blocks` (trainer.py:1701), but backprop runs from the union's minimum (trainer.py:1719). A race therefore compares the likeness of windows, not their speed.

**Per-step model (code-fit verifier).** With checkpointing, a forward-only block costs about 1 forward, a trained block about 3, so a K-of-N window costs about N + 3K.
- Krea 2 (N=28), K=10: 0.52x DiT compute, about 5-5.5 s/step, but only with `text_fusion` / `attn_gate` held out until the last stage. That gives up the 2-7 point text-fusion lift early in the run.
- Qwen 2.1 (N=32), window 10-27: 0.67x, about 1.7 s/step.
- MOSS (N=42), window 21-41: 0.75x, about 3.8 s/step.
- With about 60% of steps in narrowed stages, that is about 20-30% wall-clock saved on Krea/Qwen and about 15% on MOSS. That is nowhere near 2.5x.

**Fit and risks**
- Identity depth: Fizgig found identity in Qwen 2.1 blocks 10-14 (README.md:1657), and `id_align_layer` auto picks a third of the depth (config.py:399). A deepest-K-first window starts in the wrong place. Start around the identity depth.
- torch.compile guards on `trainable`, so each boundary recompiles the affected blocks. The first steps after a boundary are slow.
- keep blocks: candidates change at each boundary, so `_setup_checkpointing` / `_calibrate_checkpointing` (trainer.py:1532-1560) must re-run. That costs a few measured passes and can fall back after an OOM.
- block swap: complementary. The swapped prefix is the forward-only prefix.
- model_split: the split is planned once from `trained=backprop` (trainer.py:804), so a widening window unbalances it mid-run. Accept the imbalance or plan for the final stage.
- data_parallel: replicas flip together. W8A8/nvfp4 forward kernels are unaffected. Export is unchanged because every wrapper exists from step 0. Resume state must carry the stage.
- VRAM only goes down in early stages.

**Implementation path**
- config.py: `lora_block_schedule: str = ""`, e.g. `"0.0:10-27, 0.6:0-27"` (fraction of steps: window). Validate that each stage is a subset of the final `lora_blocks`.
- lora.py: `set_block_window(injected, blocks)`, reusing the race's swap_in logic. On Krea, hold `text_fusion` / `attn_gate` / `embed` off until the final stage so `backprop_blocks()` stays non-None. Recompute `backprop` per stage.
- trainer.py: check boundaries in the step loop next to the resolution step-up (trainer.py:3339). At a boundary, flip the flags, re-run `_setup_checkpointing`, log the window, and save the stage in the resume state. Leave model_split's plan alone in v1.
- Later, optional: widen at the next probe once the probe plateaus (patience) instead of at fixed fractions.

**How to test**
- Step 1, no code. Three separate runs on Qwen 2.1 int8, 0.5 MP, same seed, equal wall-clock budget: `lora_blocks` all vs static `10-27`. On Krea 2, run static `8-27` without `text_fusion`. Compare the ArcFace face and DINO body probe curves at equal minutes.
- Step 2, after the schedule exists: race the static window against the schedule (`10-27` then all at 60%) at equal steps, `probe_every 50`.
- Stop rule: if the schedule does not beat all-blocks at equal minutes on the ArcFace curve, drop it and have the wizard recommend static mid windows instead.

### Stratified timestep sampling (per epoch or per image)

**Do it (opt-in, default off until raced). Expected gain: 0 s/step. Smoother loss and probe curves, and maybe a 0-2% earlier best-probe step (inside probe noise). Confidence: medium. Cost: about half a day (a ~40-line module plus a `u` argument to `sample_sigmas`).**

**What it is.** Replace iid uniform variates with stratified ones: u_k = (k + rand)/K, shuffled per epoch, then mapped through each sampler's inverse CDF (`Phi^-1(u)` via erfinv for logit_normal / sigmoid / logsnr / identity, u directly for uniform / content / style). The sampler shapes are kept exactly. Only the per-epoch coverage jitter goes away. A per-image scope walks each image through the strata over its successive visits, so no image spends ten epochs at high noise only.

**Evidence**
- [musubi-tuner advanced_config.md](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/advanced_config.md): `--num_timestep_buckets` builds a per-epoch pool with an equal count per bucket. The docs label it "experimental", suggest a value between 4 and 10 as a starting point, and say community testing is needed. No speed or quality numbers.
- [musubi-tuner minimax_h3_advanced.md](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/minimax_h3_advanced.md): H3's base time is stratified under the same flag before shift 12 (video) / 3 (audio) is applied.
- Background only: the [adaptive Beta sampler (CVPR 2025)](https://arxiv.org/html/2411.09998) gets CIFAR-10 FID 3.99 vs 8.16 at 0.2M iterations, from scratch, at about 1.5x cost per iteration. SpeeD (4.08) and Min-SNR (4.40) come close. It does not apply here.
- Correction: with 24 iid draws over 8 buckets, the expected number of empty buckets per epoch is 8 x (7/8)^24 ≈ 0.3, not the scout's 2-3. Over a 40-120 epoch run every octant gets about 125 draws ±10. Stratification removes epoch-to-epoch jitter. It does not change the converged LoRA.
- Refuted member: the multi-sigma-per-latent and antithetic (u, 1-u) extensions. Repeating a latent K times is batch K: K-fold time and activations, a new batch shape for compiled blocks, and an invalid batch-1 keep-blocks calibration. A second real image beats the same image at a second sigma. Do not build it.

**RulzTrainer today.** `sample_sigmas` (trainer.py:215-248) draws iid `torch.rand` / `torch.randn` per call on the CPU generator. It is called from the main loop (trainer.py:1078), the DOP pass (trainer.py:2448) and race lanes (race_parallel.py:218). `clean_share` (clean_share.py:103-140) only solves a logit offset per epoch. There is no stratification, epoch pool or erfinv anywhere. The epoch structure already exists: `BucketBatcher.epoch()` (dataset.py:384-419) and `_epoch_images` (trainer.py:2471-2478).

**Fit and risks**
- The change is CPU-side only. W8A8/nvfp4, compile (sigma is a runtime tensor), keep blocks, block swap, model_split and export are untouched. The 1000-grid snap stays the last line.
- `clean_share` composes automatically, because its offset is applied after the variate (trainer.py:224-228, 234-238). The scout's suggested quantile split is unnecessary.
- `reference` mode has no quantile: keep it iid.
- Key the pool per configuration and per batcher, like `_clean_shares` (trainer.py:2488). Each race lane has its own generator and batcher. data_parallel draws in the main thread, so one pool per config is enough.
- The per-image loss watch and `per_image_lr` benefit from less sigma noise.

**Implementation path**
- config.py near line 292: `timestep_strata: int = 0` (0 = iid, >=2 stratifies; call it strata, since "buckets" already means resolution buckets) and `timestep_strata_scope: "epoch" | "image"`.
- trainer.py: give `sample_sigmas` an optional `u` argument. It replaces `torch.rand(n)` with u, and `torch.randn(n)` with `sqrt(2) * erfinv(2u - 1)`.
- New `rulztrainer/strata.py` (`SigmaPool`): K strata x ceil(N/K) draws per epoch, shuffled with the run's generator. The image scope keeps a per-item counter over a permuted cycle of K.
- Wire it at trainer.py:1078 (and optionally 2448) and race_parallel.py:218. Add both keys to the race's "Timesteps & loss" tuple (race.py:56) and to webui.py next to clean_share (~5846).
- Test: equal counts per stratum, and logit_normal empirical quantiles that match iid.

**How to test.** Qwen 2.1 int8, 0.5 MP, a 20-30 image character set, three lanes with `timestep_strata` 0 / 8 (epoch) / 8 (image), same seed, logit_normal, 2-3 seeds per lane. Judge the ArcFace probe's step-to-best and its variance across seeds, plus the loss-watch noise. If the win sits inside seed variance, keep it as a free opt-in and do not sell it as a likeness lever. Only make it a wizard default for sets under ~100 images if the curves separate.

### Gradient-informed LoRA init (LoRA-One / Riemannion LOI)

**Race it (stage 0 needs no new code; the full feature is about 120 lines, 1 day). Expected gain: net wall-clock about 0 ±3%. Possibly 10-25% fewer steps to the first probe threshold, with 0-2 face points at the end of the run (unmeasured on any DiT). Confidence: low. Cost: 0 for stage 0, about 1 day for stage 1.**

**What it is.** Initialize down/up from the top-r singular subspace of the first full-weight gradient, so the LoRA starts inside the directions the loss moves instead of a random Kaiming subspace with up = 0. LoRA-One proves that GD-trained factors align with that subspace. Riemannion's LOI takes a small negative step along it (alpha = -0.01/sqrt(r)) using a randomized SVD sketched through backprop.

**Evidence**
- [Riemannion / LOI, arXiv 2507.12142](https://arxiv.org/html/2507.12142v2), Llama-3.2-1B, rank 16: RiemannLoRA-LOI 73.4 vs LoRA-LOI 70.1±1.6, rsLoRA 70.0, LoRA 67.9. LOI takes 0.25% of wall-clock. In its own Adam init-only ablation (Table 7), LOI scores 74.3 vs standard init 74.1 (+0.2), and LoRA-GA scores 71.2. The headline gap mixes the Riemannian optimizer with an under-tuned baseline.
- The same paper ran SD2 UNet DreamBooth (400-600 steps). It needed alpha = -1/sqrt(rank) (100x) and 15 power steps, reported only qualitative "learns faster", and had no LOI-only ablation. No code link.
- [LoRA-One, arXiv 2502.01235](https://arxiv.org/abs/2502.01235) (ICML 2025 Oral, code github.com/YuanheZ/LoRA-One): T5-base GLUE avg 88.73 vs LoRA 82.72 (LoRA-Pro 88.44, so within 0.3). LLaMA-2-7B r=8: GSM8K 60.44 vs 59.26, MMLU 47.24 vs 45.73, HumanEval 28.66 vs 25.85. The README says 8-64 items suffice. No diffusion experiments.
- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/): B = 0 creates an implicit lr schedule, and short runs want about 15x the FullFT lr. The blog does not claim that an aligned init removes this; that was the scout's inference.
- Contrary: [T-LoRA](https://arxiv.org/html/2507.05964v2) Fig. 5 finds that PiSSA-like, high-singular-value aligned inits overfit and hurt text alignment in few-shot diffusion customization, which is RulzTrainer's regime. LoRA-GA lands below vanilla LoRA in three independent tables.
- [Approximate Muon with low-rank adapters](https://arxiv.org/html/2608.14492v1) fixes the zero-factor issue with jitter plus an orthonormal factor, not with an aligned init.

**RulzTrainer today.** `init_parts` (lora.py:283-298): kaiming down, zero up. The race re-derives the same init per module from a seeded generator (race.py:506-513). There is no `lora_init` option. `finetune.lowrank_factors` (finetune.py:386) wraps `svd_lowrank`. The partners the scout cites (`lora_precondition`, `lora_muon`) do not exist, so the init stands alone. Automagic 4 already special-cases the zero-up pathology.

**Key code insight (code-fit verifier).** With zero up, the ordinary first backward already is the sketch. grad(lora_up) = scale * G @ down^T, i.e. Y = G Omega with Omega = the kaiming down. Set up := Q = qr(Y) and down := 0, and the next pass gives grad(lora_down) = scale * Q^T G, the r x in matrix to SVD. So the whole init is 2-3 ordinary passes of K items through `work_list`, read off `.grad`:
- no hooks and no X/D capture;
- no per-model target maps (`text_fusion`, `txt_in` and the MOSS text projection join automatically);
- zero extra VRAM.
The hook-based dense-G variant from the second finding would need about 1.5 GB fp32 per Krea 2 block and 30-60 min of block-group passes on a 5060 Ti. Do not build that one.

**Fit and risks**
- W8A8/nvfp4: use the default bf16 backward. The gradient is the quantized model's, which is the model being trained.
- Compile: values change, not shapes, so no recompiles. keep blocks and block swap: plain passes need nothing.
- data_parallel: `dp.reduce_grads()` between passes averages the sketch over replicas, then `dp.sync_weights()`.
- EMA: run the init before `_make_ema` (trainer.py:831) or call `ema.reset()`.
- Race: compute once per shape per seed in `Variant.init_lora` to keep "same init for the same shape". Seed rows get different inits by construction.
- Export: down/up are written as-is. Only the LOI/LoRA-One form applies, because LoRA-GA's W -= BA is impossible on int8/nvfp4 weights.
- Not for LoKr, OFT, finetune or resume.
- Signal risk: at batch 1 the flow-matching residual at a random noise draw is mostly irreducible noise, so a K=12 gradient subspace may capture the noise draws, or background/pose/lighting, rather than the identity. Mitigate with K=16-32, face/bust crops, sigma 0.3-0.6 and region weights on. Log the singular-value decay of Q^T G per layer type: a flat spectrum means a random init in disguise.

**Implementation path (stage 1)**
- config.py: `lora_init: zero|gradient`, `lora_init_items: 16`, `lora_init_power_steps: 0`, `lora_init_scale: 0.01` (applied as 0.01/sqrt(r)), `lora_init_sigma: ""` (empty = the run's sampler, or a band such as `"0.3-0.6"`). LoRA kind only. Add to RACE_KEYS "LR & optimizer".
- lora.py: `gradient_init(injected, run_pass)` doing the Q <-> down exchange described above. Final factors: down = V^T, up = -(scale/sqrt(r)) Q U_B / lora_scale.
- trainer.py: call it after `_setup_checkpointing` (trainer.py:950) and before the loop, with `caption_dropout` 0 and a status stage "gradient init pass 1/2". Then zero the grads.
- Test: on a two-Linear toy model with exact G, the recovered factors span the top-r subspaces, and export/load round-trips.

**How to test**
- Stage 0, no new code: a short full fine-tune (`finetune_rotate_epochs` small, 1-2 cycles), then `finetune_extract_ranks [16]`, then a normal LoRA run with `resume_from` (trainer.py:805-807). Race it against zero init. This tests whether a gradient-shaped start helps likeness at all.
- Stage 1: a two-row race on Krea 2 nvfp4, rank 16, 12-20 photos, 1000 steps. Settings: `race_follow: false`, `race_drop_lowest: false`, same seed and shape, `probe_every 50`, `probe_seeds 4`. Rows: zero vs gradient (K=16, face crops, sigma 0.3-0.6). Repeat on a second seed. Qwen 2.1 int8 is the cheaper first target.
- Stop rule: if the smoothed scores at steps 100-300 do not separate by more than the probe spread on two seeds, drop the feature rather than tune it. Also race it against the free surrogates: `lora_plus_ratio` 4-16, or double lr for the first 100 steps.

### Householder reflection adapter (HRA kind)

**Race it. Expected gain: likeness unproven (parity to a small gain). Speed about LoRA (within ±3% on Krea 2 / MOSS, +2-4% on Qwen 2.1). VRAM equal to LoRA. Confidence: medium. Cost: a new adapter kind (lora.py, config, webui, export, tests); multi-day (est.).**

**What it is.** Multiply the input of the frozen linear by an orthogonal matrix H made of r Householder reflections: W' = W H. In compact WY form, H = I - U S^-1 U^T, with S = triu(U^T U) and its diagonal halved. So H - I has rank <= r, and the adapter exports exactly as a rank-r LoRA: down = U^T, up = -(W U) S^-T, alpha = r. It is a dense orthogonal map on any r-dim subspace, unlike RulzTrainer's block-diagonal Cayley OFT on the output.

**Evidence**
- [HOFT, arXiv 2505.16531](https://arxiv.org/html/2505.16531v2), SD1.5 DreamBooth (objects/pets; q/k/v/out only; 2005 steps, batch 1; 10x A40), DINO / CLIP-I / CLIP-T:

  | Method | DINO | CLIP-I | CLIP-T |
  |---|---|---|---|
  | LoRA | 0.613 | 0.765 | 0.237 |
  | OFT | 0.632 | 0.785 | 0.237 |
  | HRA (lambda=0) | 0.670 | 0.803 | 0.238 |
  | HOFT r=4 | 0.680 | 0.810 | 0.235 |

- The LoRA row is the 2023 OFT paper's single-V100 run with unspecified rank, carried forward unchanged and not re-tuned. Treat +0.067 DINO as an upper bound. HOFT's own BOFT rerun scored 0.545, below LoRA, which shows the benchmark's hyperparameter sensitivity.
- Corrections: Table 11's times (LoRA 2.9 h, HRA 35.8 h, HOFT 4.3 h; peak memory HOFT 25.5 GB = LoRA) belong to the SDXL qualitative runs (5x A40, r=16), not to SD1.5. HOFT is two-sided (Q_U W Q_V, delta rank <= 2r). SHOFT's diagonal scale makes its delta full rank, so it does not export as a LoRA. Skip SHOFT.
- [PEFT HRA](https://huggingface.co/docs/peft/package_reference/hra) already ships the CWY form (triangular solve) and requires even r for identity init. The "naive HRA 35.8 h" cost is stale. HRA paper: [arXiv 2405.17484](https://arxiv.org/abs/2405.17484). HRA used lr 7e-6 and HOFT 5e-4 on the same task, so the lr is unsettled.
- No DiT, flow-matching, quantized-base or face result exists anywhere. The verifier's maturity labels: HRA "library_available", HOFT "paper_only".

**RulzTrainer today.** KINDS = ("lora", "lokr", "oft") at lora.py:105. OFT is the block-diagonal Cayley rotation on the output (lora.py:141-149, 360-370). There is no Householder code.

**Fit and risks**
- The input-side rotation is applied to x in bf16 before `QuantLinear` (quant.py:1011-1017). Orthogonal maps preserve row norms, so the per-token amax for W8A8/nvfp4 barely moves; ConvRot's Hadamard already rotates the same tensor.
- `_W8A8LinearFn` saves no activation, so peak memory is about LoRA's. The transient x' is about 31 MB per Krea 2 layer (84 MB for ffn_down) and is freed per layer, so it is irrelevant to keep blocks.
- Compile: plain matmuls. Block swap, model_split and data_parallel go through `adapter_params()` generically. Kind is a race key and a SHAPE_KEY, so a relay cannot mix kinds.
- Export: needs `dense_weight` (lora.py:152-157) once per layer at save. The delta is pinned to the quantized W, which gives a small second-order error on a bf16 base: negligible for int8, small for nvfp4.
- Unlike OFT/LoKr/DoRA, the export is a plain LoRA that ComfyUI, diffusers/PEFT (FIBO) and soup can all read. Resume needs a `_hra.safetensors` sidecar with the native vectors.
- Exclude DoRA and `lora_plus_ratio`. Skip the orthogonality regularizer (lambda = 0 was best) and Gram-Schmidt.
- Possible bonus (unmeasured): orthogonal adapters keep norms and angles. That may help long runs and Qwen 2.1's collapse that needed the training adapter.

**Implementation path**
- lora.py: kind `hra`, part `hra_u [r, in]` initialized in identical pairs (even r) so H = I. Add `hra_apply(x, u)` using `solve_triangular` (exact; do not use HOFT's Neumann approximation), and the forward branch `out = base(hra_apply(x, hra_u))`.
- Extend `set_kind` / `adapter_shapes` / `init_parts` and `export_state_dict` (comfy and peft keys), plus `peft_adapter_metadata`.
- config.py: even-rank check; reject dora+hra.
- trainer.py: metadata maps (1424-1426, 1943-1944). webui.py: Adapter select (~4032).
- Tests: extend tests/test_oft_dora.py's ComfyUI-merge check to plain and int8_convrot bases.
- Later: an output-side variant for Krea 2 GQA `to_k` / `to_v` and `ffn_down`, where out < in.

**How to test.** Race rows `network_type` lora r16 / hra r16 / oft 32, same seed, 1500 steps, `probe_every 100`, `probe_seeds 4`, on Krea 2 nvfp4 and Qwen 2.1 int8 at 0.5 MP. Sweep the HRA lr separately (1e-5 to 5e-4). Judge the combined face+body curve, the step at which each row reaches the LoRA row's final score, the final score, and `||W'||_col` drift. If HRA ties or wins, add the output-side variant and a Qwen 2.1 run with the training adapter off. If not, drop it.

### Loss weights: eps-equivalent, x0-equivalent, soft-min-SNR

**Race it (the min_snr vs none part needs no code; the new weights are about 25 lines). Expected gain: probe differences of 0.01-0.02 (needs two seeds to call). Speed/VRAM none. Confidence: low. Cost: about 25 lines.**

**What it is.** For a velocity model, ||x0_hat - x0||^2 = sigma^2 ||v_hat - v||^2 and ||eps_hat - eps||^2 = (1-sigma)^2 ||v_hat - v||^2. The x0-equivalent weight is sigma^2 and the eps-equivalent weight is (1-sigma)^2. Soft-min-SNR uses w = snr/(snr + lambda), the smooth form of min-SNR.

**Evidence**
- [Training Flow Matching: weighting and parameterization (2026)](https://arxiv.org/html/2603.06454v1). Uniform t, from-scratch U-Nets, single seed. FIDs for x0 / velocity / eps / classic weights: CIFAR-10 (1000 ep) 10.29 / 3.90 / 3.71 / 116; CelebA-64 (300 ep) 18.83 / 4.45 / 3.97 / 133.9.
- The small-data reversal (10k images: denoiser 11.07 vs velocity 27.25; velocity already wins at 50k, 6.89 vs 8.98) is a parameterization result. A frozen v-model cannot reproduce it by reweighting (refuted as support for x0).
- [CSFlow](https://arxiv.org/html/2606.08833), a 7,500-step pixel-space PixelGen fine-tune: GenEval baseline 0.791, P2 0.803, MinSNR 0.787, CSFlow 0.810. No variance.
- [Smoothed-KL](https://arxiv.org/abs/2606.13433): final-FID parity with min-SNR. [SpeeD](https://arxiv.org/html/2405.17403v3): DDPM, from scratch, no rectified flow.
- None of these involves LoRA, latents, logit-normal sampling or few-shot data.

**RulzTrainer today.** `loss_weight()` (trainer.py:248-256) supports none, `sigma_sqrt` and `min_snr` = clamp(snr, 5)/(snr + 1). Above sigma 0.3, min_snr is already close to eps (sigma 0.5: 0.50 vs 0.49), and it additionally suppresses sigma < 0.3. So `eps` is a small delta, while `x0` is genuinely absent. `loss_weighting` is already a race key (race.py:58). The webui select hardcodes three values (webui.py:5809). The option is undocumented in the README.

**Fit and risks.** It is a per-sample scalar applied after the forward, so none of the compute paths are touched. The mean-1 normaliser mostly does not matter: Adam variants and Automagic 3/4 are invariant to a constant loss scale, and only `max_grad_norm` and weight decay see it. Under a shaped sampler, a multiplicative weight redoes the sampler's work with more gradient variance. x0 (sigma^2) suppresses exactly the low-sigma steps where `id_loss_sigma_max`, face detail and the identity sampler live, so expect it to hurt. If tried, pair it with `identity`, never `logsnr`.

**Implementation path.** In trainer.py `loss_weight`, add `eps` = (1-sigma)^2, `x0` = sigma^2 (both clamped at min 1e-3) and `soft_min_snr` = snr/(snr + lam) with `soft_min_snr_lambda: 0.2` (equivalent to gamma 5). Add an optional mean-1 normaliser from 4096 `sample_sigmas` draws. Then extend the config validation, the webui select and tooltip, add a README "Loss weighting" paragraph near README.md:206, and add a unit test at sigma 0.001 / 0.5 / 1.0.

**How to test.** Qwen 2.1 int8, 1500 steps, `probe_every 100`, logit_normal. Rows: none, eps, min_snr, and x0 + identity. Judge by the mean of the last three ArcFace probes plus DINO body. Require more than 0.02 over the default on two seeds before changing any recommendation. Keep `none` as the default either way.

### Warm start from a person prior (LoFA-inspired)

**Race it (mostly existing options). Expected gain: a plausible best case of a few hundred fewer steps to the first probe threshold (about 15-45 min on Krea nvfp4, 8-25 min on MOSS, 4-12 min on Qwen 2.1). Final likeness about 0. Confidence: low. Cost: small (init scale, a rank-adapting load, a prior-building helper, threshold logging).**

**What it is.** Start the LoRA from an average of the user's earlier person LoRAs for the same model, either as a scaled trainable init or as a frozen training adapter that is off at inference. This stops every run starting from B = 0.

**Evidence**
- [LoFA, arXiv 2512.08785](https://arxiv.org/html/2512.08785), SDXL identity: face similarity 0.548 / DINO 0.497 / CLIP-I 0.600 in 3.7 s vs DreamBooth 0.488 / 0.460 / 0.544 in 1 h (HyperDreamBooth 0.527 in 274 s). Wan2.1-1.3B: FVD 589.8 vs 609.5, but LoFA-pose scores 610.7.
- The hypernetwork was supervised by about 3,100 per-identity LoRAs. Fine-tuning from a predicted LoRA was not tested. Refuted member: the hypernetwork does not transfer to a single-user trainer; do not pursue it.
- [SimpleTuner Klein training assistant](https://huggingface.co/SimpleTuner/flux-3-klein-image-LoRA-experiments/raw/main/chapters/03-assistant.md): rank 64, 30k updates on 8x H100. It is used frozen at strength 1 during training and off at inference. No control run is reported, and the 30k checkpoint drifts in composition.
- Contrary: [Hayou et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d4387c37b3b06e55f86eccdb8cd1f829-Abstract.html) find that zero-B init beats non-zero init because it tolerates larger lr. [Continual personalization](https://arxiv.org/abs/2502.07739v1) reports interference when a new concept starts from earlier concept weights.

**RulzTrainer today (largely present).**
- `resume_from` loads any LoRA as the trainable init (config.py:241, trainer.py:805-807), with no scale, and raises on a rank mismatch (lora.py:731-732).
- `soup_files` + `make_soup` average files at a chosen `soup_rank` (soup.py:67/137).
- `training_adapter` accepts a path on every family (README.md:1648-1654). Only the `auto` catalog is limited to qwen_image21.

**Fit and risks.** The init is just different fp32 starting values, so it is neutral to every compute path. The risks:
- Non-zero A and B lower the stable lr (Hayou), which interacts with Automagic's lr adaptation and with LoRA+.
- `lora_ema` starts from the prior.
- Blended-identity bleed from the averaged faces.
- A frozen prior adapter makes the LoRA learn a residual against something that is absent at inference.

**Implementation path**
- `init_scale` applied after `load_lora_weights` (trainer.py:805-807), either as sqrt(scale) on both factors or as scale on up.
- On a rank mismatch, make `resume_from` SVD-truncate or pad the file's delta via `soup.delta_of` / `soup.cook` instead of raising.
- A helper that runs `make_soup(strength=1, rank=cfg.lora_rank)` over chosen earlier LoRAs and writes a prior file.
- Steps-to-threshold per row in race.py.
- Register SimpleTuner's assistant in `ARCH_TRAINING_ADAPTER` when Klein lands.

**How to test.** Krea 2 or Qwen 2.1, 0.5 MP, rank 8, 2000 steps, same seed. Build the prior from at least 4 earlier LoRAs of other subjects, with the same targets and blocks. Rows:
- (a) zero init;
- (b) `resume_from` = prior with `init_scale` 0.3 (needs the scale option);
- (c) `training_adapter` = prior (no code);
- (d) on Qwen 2.1, Fizgig's adapter as a reference.

Primary metric: steps until ArcFace reaches 90% of the baseline's final best. Bleed check: similarity of the probe renders to the prior subjects' faces. Keep it only if it reaches the threshold at least 20% sooner on 2 of 3 subjects, with best face no more than 0.01 below baseline and no rise in prior-subject similarity.

### T-LoRA: timestep-dependent rank mask

**Watch (race only once a text-alignment probe exists). Expected gain: likeness about 0 or slightly negative, 10-30% slower likeness convergence. The win is prompt following (CLIP-T +0.016-0.024), which no RulzTrainer probe scores. Confidence: medium. Cost: about half a day.**

**What it is.** W_t = W + B M_t A, with a 0/1 mask keeping the first r(sigma) = floor((r - r_min)(1 - sigma)) + r_min components. Noisy steps get fewer components, because that is where single-subject LoRAs memorize pose and background.

**Evidence**
- [T-LoRA, arXiv 2507.05964](https://arxiv.org/html/2507.05964v2) (AAAI 2026; [code, MIT](https://github.com/ControlGenAI/T-LoRA)).
  - SDXL r=64: DINO 0.802 / CLIP-I 0.900 / CLIP-T 0.256 vs LoRA 0.808 / 0.901 / 0.232.
  - FLUX.1-dev, CLIP-T vs LoRA: r=4 0.268 vs 0.263, r=16 0.280 vs 0.264, r=64 0.280 vs 0.247. CLIP-I is within ±0.02.
- Single seed, objects/pets only, no faces. T-LoRA was trained longer (SDXL 800 vs 500 steps, FLUX 500 vs 400), and the authors say it learns the concept more slowly. In the user study, raters preferred plain LoRA for concept preservation (60.7 vs 39.3). r_min below r/2 "often" loses the concept.
- The FLUX r=32 / r_min=1 / 1000-step recipe comes from the README and was not tested in the paper.
- The paper applies the mask at inference too, so a static ComfyUI export is not what was evaluated.
- Refuted member: Ortho-LoRA init. The paper skips it on MM-DiT ("LoRA adapters trained on MM-DiT are full-rank"), and its frozen term is also masked (Eq. 6), so the scout's always-on negative extra is wrong.

**RulzTrainer today.** No rank mask. `_added` (lora.py:352-358) applies the full rank, and `set_multiplier` (lora.py:579) is the only per-forward knob.

**Fit and risks**
- The mask is one [B,1,r] multiply outside `QuantLinear`.
- Compile: keep it a tensor attribute, not a float (specialize_float).
- Block swap: a plain attribute, not a registered buffer.
- A forward pre-hook reading sigma*1000 from args[1] covers training, DOP, samples, probes, reward and selftrain.
- At rank 8-16 with r_min = r/2, and with Krea 2 / Qwen 2.1 reference sigmas mostly above 0.7, most steps run near r_min.
- Export loses the schedule; the only fix is a split pair (`_low`, `_high`) plus ComfyUI hook keyframes. Soup's SVD scrambles component order.

**How to test (when it is raced).** Qwen 2.1 int8, 0.5 MP, rank 16, `timestep_sampling identity`, 1500 steps. Rows: off / `lora_min_rank` 8 / `lora_min_rank` 12. Probe the finalist with the mask on vs static all-ones to size the export loss. Add a SigLIP/CLIP text-alignment number before claiming the flexibility gain.

### Modulation, norm and table targets (DiffFit style)

**Skip-ish (one free race with existing targets). Expected gain: global colour/lighting drift of ±1-2 probe points at best, with class-prompt bleed. Not a likeness lever. Confidence: low. Cost: zero for the race; a small ParamDelta wrapper only if it wins.**

**What it is.** Train the per-channel scale/shift paths (adaLN modulation, per-block tables, norm scales) as an extra group. These see one conditioning vector per sample, so they are nearly free.

**Evidence**
- [DiffFit](https://ar5iv.labs.arxiv.org/html/2304.06648): class-conditional DiT-XL/2 at 256 px, average FID over 8 datasets 15.39 vs full fine-tune 16.59 (0.12% of parameters, 0.49x time). Gamma in blocks 1-14 helps (8.29 to 7.61 on Food101).
- Its LoRA-R16 baseline (FID 81.31) was q,v-only and untuned. The "LoRA did badly" argument should be dropped.
- [MiniMax H3 card](https://huggingface.co/MiniMaxAI/MiniMax-H3): about 13B of 33B parameters sit in AdaLN branches and are precomputable for inference.
- Verifier correction: in all four supported families the modulation input is the timestep embedding only, never text. A modulation delta cannot be conditioned on the trigger.
- Krea 2 detail (the second, unverified finding is consistent with the code check): one shared `time_mod_proj` plus a per-block `scale_shift_table [6, 6144]` (krea2/model.py:168, 255, 309).

**RulzTrainer today (partially present).** The shared modulation is already trainable via `embed` on Krea 2 (archs.py:350, 435) and Qwen 2.1 (archs.py:464, 563), and FIBO has `adaln` (archs.py:236). MOSS's `shared_block_modulation` is unreachable (archs.py:147). Nothing trains norm weights, tables or biases.
- Refuted member: the `bias` / `diff_b` group. Every block linear in MOSS, Krea 2, Qwen 2.1 and FIBO is bias=False, so ComfyUI has no bias key to patch.
- Qwen 2.1 block LayerNorms have no affine weights (qwen21/model.py:163-165).
- Automagic 4 measured modulation projections at 3x-135x the bulk's gradient power (README:673-676), so they need their own lr.

**How to test.** Race the current targets vs the same plus `embed` on Krea 2 and Qwen 2.1, and `+adaln` on FIBO. 1000-1500 steps, face+body probes every 100, a trigger-less class-prompt sample set for drift, Automagic 4 or a 10x lower lr for `other`. Only if this wins: add `embed` to MOSS (about 10 lines in archs.py:185-207), then a Krea 2 table/norm delta exported as ComfyUI-named `.diff` keys via the krea2/single_file.py rename table. Note for H3 (unsupported): README:714 records that its `adaln_proj` calls the raw weight, so a LoRA wrapper gets no gradient there.

### DeLoRA (normalized rank-1 components with a strength scalar)

**Skip-ish (an lr-robustness race option at most). Expected gain: no demonstrated likeness gain; possibly tolerates 4-8x lr. Confidence: low. Cost: small (a flag on kind lora, a lambda parameter, a param group).**

- Mechanism: Delta W = (lambda ||W|| / r) B Xi A, with Xi normalizing each rank-1 component, so direction and strength are decoupled.
- [DeLoRA, arXiv 2503.18225](https://arxiv.org/html/2503.18225v2), SD1.5 DreamBooth r16: tuned DINO 0.693 vs LoRA 0.688 / DoRA 0.689. That is inside DeLoRA's own seed std (±0.0056). At default settings it ties LoRA at 0.686. The lr-robustness claim (LoRA/DoRA degrade at 4x lr) is qualitative.
- Corrections: the image runs used a FIXED lambda = 1e-3 with lr 2e-2 (vs 6e-4 for LoRA), not a learnable lambda of 10-15. [PEFT DeLoRA](https://huggingface.co/docs/peft/package_reference/delora) zero-inits B and uses per-column norms, so the scout's PiSSA-style frozen negative extra is unnecessary (refuted member). PEFT's Llama benchmark has DeLoRA at 46.8% vs LoRA 47.8%, about 12% slower per step.
- RulzTrainer: nothing similar. Reuse `w_col_norm` (lora.py:277-281). Keep lambda a [1] tensor parameter for compile. Automagic's `automagic_max_lr` 1e-3 (config.py:278) would cap it, so the mult must also raise min/max.
- Race if wanted: Krea 2 or Qwen 2.1, LoRA lr 1e-4 vs DeLoRA lr 1e-3 vs DeLoRA lr 4e-3 vs DoRA. Keep it only if DeLoRA at 4e-3 reaches the LoRA row's likeness in materially fewer steps.

### ABBA (Hadamard product of two LoRA pairs)

**Skip-ish. Expected gain: -0.02 to +0.02 likeness vs a rank-16 LoRA (verifier's estimate), +2-4% step time. Confidence: low. Cost: a new adapter kind with LoHa export.**

- Mechanism: delta = s (B1 A1) o (B2 A2), with pair 1 from an SVD of W0, pair 2 LoRA-style, and s = alpha^2/sqrt(r1 r2). The Khatri-Rao rewrite makes it a rank-(r1 r2) LoRA on the fly.
- [ABBA, arXiv 2505.14238](https://arxiv.org/html/2505.14238v4) is LLM-only: Llama-3.2 1B/3B commonsense 75.03 / 84.08 at r=32 (r=16: 73.76 / 83.68) vs LoRA 71.49 / 81.30. It wants lr 1e-3 and alpha 16-24, and saturates or overfits above rank 32.
- The only diffusion datapoint, [WaveFT on SDXL](https://arxiv.org/html/2505.12532v3), is negative: LoHa DINO 0.424 vs LoRA 0.463 / LoKr 0.449, with LoHa at about 2x the budget. HiRA background: [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/48c368f105e8145b945227b73255635a-Abstract-Conference.html).
- The premise "more effective rank helps likeness" is contradicted by the user's own result: LoKr 8 lost to LoRA on time-to-likeness. Refuted member: "30-35% below HiRA" is a comparison with HiRA, not a memory saving over LoRA.
- If raced: lora16 vs abba [8,8] vs lokr8 on Qwen 2.1 int8, 1500 steps. Drop it if it does not beat LoRA there.

### SingLoRA (single symmetric factor)

**Skip-ish. Expected gain: about 0 over LoRA+ (which RulzTrainer has). VRAM saving 100-300 MB at rank 16-32 (irrelevant). Confidence: low. Cost: a new kind.**

- [SingLoRA, arXiv 2507.05566](https://arxiv.org/html/2507.05566), SD1.5 DreamBooth, q/k only, 400 steps, CLIP-I / DINO: SingLoRA r16 0.690 / 0.151 vs LoRA r8 0.677 / 0.143. But LoRA+ r8 scores 0.688 / 0.150 at the same budget, which is parity. SingLoRA r8 scores 0.677 / 0.141, below LoRA at equal rank.
- Refuted members: "accepted at ICLR 2026" (it is an ICLR 2026 DeLTa workshop poster), and "halves parameters AND improves quality" (mutually exclusive in the authors' own table). No official code.
- Fit issues: the per-step ramp fights `specialize_float` compile (about 30 recompiles at 3000 steps). The update is PSD-restricted on square layers. The verifier's variant (a zero-init signed diagonal s instead of the ramp) fixes both if anyone races it: lora r16 vs singlora r32 on Qwen 2.1 int8.

### Nearest-of-k noise selection (Improved Immiscible Diffusion)

**Skip-ish. Expected gain: 0 to a few percent fewer steps, undetectable without repeated seeds. Confidence: low. Cost: about 25 lines.**

- [Immiscible Diffusion](https://arxiv.org/html/2406.12303v2) (Hungarian matching): 1.2-3x fewer steps from scratch. SD1.4 ImageNet fine-tune FID 10.28 vs 11.45 at batch 512.
- [Improved Immiscible Diffusion](https://arxiv.org/html/2505.18521v1) (KNN, k=4 best for flow matching on CIFAR, 64 for SD): baselines need 2.5-4.5x more steps on CIFAR-10. The SD fine-tune result is figure-only. The paper says larger data dimension needs larger k. Code: [MIT repo](https://github.com/yhli123/Immiscible-Diffusion); ai-toolkit ships the per-sample variant.
- Dimension argument (both verifiers): the nearest of k=8 Gaussian draws is closer to x0 by about 2.7% at CIFAR's 3k dims, but only about 0.5% at 131k dims (Krea 2 / Qwen 2.1 at 0.5 MP) and 0.4% at 262k (MOSS). k=64 reaches about 1% while drifting from Gaussian.
- Refuted member: a k in {0, 4, 8} race space is a wasted arm.
- RulzTrainer draws iid noise at trainer.py:1079 on the CPU generator. The change is orthogonal to every compute path.
- If raced: `noise_knn_k` {0, 8, 32}, two seeds, no drops. Drop it unless k=32 beats k=0 by more than 2x `likeness_noise_band` on both seeds.

### FouRA (frequency-domain gated LoRA)

**Skip-ish. Expected gain: likely within probe noise. The exportable variant is a plain LoRA in a rotated basis. Confidence: low. Cost: medium.**

- [FouRA, arXiv 2406.08798](https://arxiv.org/html/2406.08798v1): style LoRAs on SD1.5 at rank 64. HPSv2 22.3 to 25.2 and LPIPS diversity 38.3 to 43.9 at alpha 1, with the adaptive gate active at inference. The exportable frozen-mask variant scores +0.4 HPSv2, within std. No identity experiment, no code.
- Refuted member: the speed claim (2.09 vs 1.87 it/s). It is unexplained, measured at batch 1, and folding F into A/B makes the cost identical to LoRA.
- Diversity is partly opposed to likeness. RulzTrainer's probes have no diversity metric. Region weights, DOP and caption dropout already address leakage.
- If raced: a gated LoRA (an r x r gate on h) rather than a DFT, plus a mean pairwise DINOv2 distance in likeness.py. Drop it unless the frozen-mask row beats LoRA beyond probe noise.

### Contrastive flow matching with negatives

**Skip-ish. Expected gain: analytically near zero or slightly negative for this workload ("9x fewer iterations" counts as 0x). Confidence: low (0.12). Cost: about 15 lines.**

- [Contrastive Flow Matching (ICCV 2025)](https://arxiv.org/html/2506.05350): lambda 0.05, up to 9x fewer iterations from scratch on ImageNet at batch 256-1024. The gain grows with batch size.
- [SR-DiT](https://arxiv.org/html/2512.12386v1) on a strong stack: 3.64 to 3.61 FID.
- Verifier analysis: an independent negative (another image, fresh noise) carries no information about x_t. In expectation the loss reduces to a 1/(1-lambda) velocity rescale (about 5.3% at 0.05) plus a shift away from the negative pool's mean latent. The LoRA bakes that bias in, and the user's no-CFG turbo samplers and ComfyUI exports cannot undo it: expect about 2-5% extra contrast or saturation.
- Refuted member: "negatives from LoRA Isolation class renders". No such latent cache exists (DOP reuses training latents; the strangers are composited PNG pairs), and class negatives would not separate the subject flow from the class flow anyway.
- If raced: the deterministic equivalent target (t + lambda m_bucket)/(1-lambda), `cfm_lambda` <= 0.02, on Qwen 2.1. Adopt only if it beats both baseline seed twins beyond their spread with no saturation drift.

## Likeness objectives and the reward stage

This section covers changes to the training objective (identity loss, id/body alignment, consistency and overfit regularizers, patch weights) and to the likeness reward stage (Flow-GRPO in `rulztrainer/reward.py`). Overall verdict: none of the ten items has a published likeness number for a LoRA on a flow DiT, and every speedup in the sources was measured with full-parameter RL on 8-GPU clusters with groups of 21-24. The best bets are the cheap fixes to the existing reward stage (high-sigma move selection, an SDE window, a flat-group skip and a detached unconditional branch) and two experiments that need no code at all: a wider identity-loss sigma window and a race that finally measures id_align. A pathwise ArcFace reward and an NFT-style reward update are the two larger builds worth prototyping. DCO, APT and noise-consistency regularization are marginal for likeness.

### Reward stage: high-sigma moves, SDE window, flat-group skip, per-scorer advantages

**Race it. Expected gain: unmeasured for likeness; saves the update half of a reward step on flat groups and about a quarter to a third of the backward when guidance > 1. Confidence: medium (0.55). Cost: small code change in reward.py and trainer._reward_stage.**

**What it is.** Six cheap changes to the existing Flow-GRPO stage, taken from Qwen's own RL report and from TempFlow-GRPO/MixGRPO: (1) train only the highest-sigma moves of the turbo trajectory instead of a random subset; (2) inject SDE noise only on the trained moves and run the rest as plain Euler (ODE), so rollouts stay close to the probe sampler; (3) multiply each trained move's loss by the TempFlow weight `sigma_j*sqrt(-dt_j)`, mean-normalized to 1; (4) skip the update when the group's reward range is below a threshold; (5) normalize face and body advantages separately, then combine with face/body weights; (6) when probe_cfg != 1, compute the unconditional branch under no_grad, as Qwen-Image-2.0-RL does. The rationale from the sources: high-noise steps set layout and identity, training on all steps invites reward hacking, flat groups waste updates, and one scorer should not dominate.

**Evidence**
- [Qwen-Image-2.0-RL report (arXiv 2606.27608)](https://arxiv.org/html/2606.27608v1): 40-step ODE rollouts; training on all timesteps reward-hacked within a few iterations, fixed by a timestep subset emphasising high-noise steps near t=1; prompts kept only when the intra-group reward range exceeds a threshold (value not given); an embedding-level face-identity scorer next to the VLM judge; per-reward normalized advantages with weights summing to 1; CFG in rollouts only, unconditional branch excluded from the objective. T2I Elo 1115 -> 1193, edit Elo 1256 -> 1349, Qwen-Image-Bench 55.23 -> 57.84. Full-parameter, at scale; G, KL, clip, LR and compute not reported.
- [TempFlow-GRPO (arXiv 2508.04324)](https://arxiv.org/html/2508.04324v4): per-step weight Norm(sigma_t sqrt(dt)), branching 4 seeds x 6; FLUX.1-dev HPSv3 reaches Flow-GRPO's 300-step result in 80 steps (3.75x); GenEval 0.95 in about 2000 vs about 5600 steps. GPU count not given.
- [MixGRPO (arXiv 2507.21802)](https://arxiv.org/abs/2507.21802): SDE plus GRPO loss only in a sliding window, ODE elsewhere; nearly 50% less training time than DanceGRPO. The 71% figure for MixGRPO-Flash may be relative to MixGRPO, not DanceGRPO.
- [Stepwise credit assignment (arXiv 2603.28718)](https://arxiv.org/html/2603.28718v1): Tweedie x0-hat per step with 5 Euler substeps, gain advantages, SDE eta 0.9; SD3.5-M on 8xA100, 10 steps, batch 16; GenEval 0.71 vs 0.68 at cfg 4.5 but equal (0.60) at cfg 1.0; each iteration 1.8-2.4x slower, 2-3x fewer iterations, 20-40% less wall-clock (3 of 4 settings in the main text). No venue on arXiv (the "CVPR 2026" label is unsupported). Flow-GRPO diverged on OCR after about 500 steps; Stepwise did not.
- [Reward Score Matching (arXiv 2604.17415)](https://arxiv.org/html/2604.17415v4): "about 5x less" compute than TempFlow-GRPO is an estimate read off a figure; its branching direction flips between case studies, so "concentrate on low-SNR" is not a general rule.
- Cautionary: GRPO-Guard (arXiv 2510.22319) fixes Flow-GRPO hacking by ratio normalization and reweighting across noise levels, not step selection. Flash-DMD (arXiv 2511.20549) reports "oil painting" reward hacking when RL runs on few-step distilled models, which is exactly the turbo setting here.
- Does not transfer: all iteration and wall-clock multipliers (full-parameter, multi-GPU, 10-40 step rollouts, GenEval/PickScore/HPS/Elo rewards).

**RulzTrainer today.** reward.py is a full Flow-GRPO: SDE step reward.py:68-76, Gaussian log-prob :79, KL :85, group advantages :90-96. `advantages()` returns None only when rewards are exactly equal. `sample_group` (:111-129) adds noise at every step but the last. `_reward_stage` (trainer.py:2752-2895) picks `reward_train_steps` moves at random among those with std > 0 (trainer.py:2836-2837; confirmed). `_reward_of` (trainer.py:2915-2951) merges face and body via `combine()` before normalizing; only `clip_part` (reward.py:55) caps each part. The CFG pass is differentiated through both branches (trainer.py:2846). Options at config.py:404-412. Delta: everything above is new.

**Fit and risks.** No change to W8A8/nvfp4 kernels, compile or block swap. The stage already releases kept blocks (trainer.py:2797) and runs on devices[0] alone under data_parallel. "High-noise steps" means the first 1-2 of the 4-8 turbo sigmas from `arch.sample_sigmas`, not the 1000-step grid. Export and probes unchanged. Risk: training only high-sigma moves may slow the reward climb; low risk of under-training texture since the main stage set it. Dropped part: per-step Tweedie scoring with 5 Euler substeps needs T x G full VAE decodes plus detection and about 80-160 extra DiT forwards per reward step; on Krea 2 nvfp4 (about 77 s per reward step today) that roughly doubles or triples the step, and a high-sigma turbo x0-hat face is blurry, so the ArcFace signal is noisy. Not recommended except as an experiment without substeps.

**Implementation path**
1. config.py next to `reward_train_steps`: `reward_step_pick: random|high` (default random until raced), `reward_sde_window: bool`, `reward_step_weight: none|tempflow`, `reward_min_range: float` (0 = off), `reward_split_advantages: bool`.
2. `_reward_stage` (trainer.py:2836-2837): with pick=high choose the largest-sigma moves before rendering; pass a per-step noise list into `RW.sample_group` so non-trained moves use noise 0.
3. In the update loop multiply each move's loss by `w_j = sigma_j*sqrt(-dt_j)`, normalized to mean 1.
4. `RW.velocity`: unconditional velocity under no_grad when probe_cfg != 1.
5. `advantages()` (reward.py:90): take `min_range`, return None below it, log the skip like "all alike".
6. Optional: `_reward_of` returns (face, body); normalize per part, combine with face/body weights.
7. README reward section, wiki, WebUI reward panel. Applies to MOSS, FIBO, Krea 2, Qwen 2.1 and later Flux 3 Klein unchanged.

**How to test.** From one trained character LoRA (`resume_from`, steps 0) on Qwen 2.1 int8, then Krea 2 nvfp4; same seeds, prompts, `reward_lr`, `reward_kl`, `reward_steps` 200. Arms: A current stage; B pick=high + SDE window; C = B + tempflow + min_range 0.05; D (optional) = C + split advantages. Probe face + body every 25 reward steps, wall-clock per reward step, share of skipped updates, and a sharpness / high-frequency energy statistic on probe renders vs the stage start plus an eye check. Keep B or C if its best probe is at least A's (within probe noise, the SD of repeated probes at the start) at no more wall-clock and no rise in sharpness; drop any arm that falls 2 probe-noise SDs below its start; promote only after a win on two characters.

### Pathwise ArcFace reward through the last K turbo steps

**Prototype. Expected gain: +0.02 to +0.06 ArcFace cosine on probes over the regular stage's best checkpoint (verifier estimate, unmeasured; part may be metric gaming). Confidence: medium (0.55). Cost: medium build (new reward mode); about 0.5-3 h compute per stage at 100-300 updates.**

**What it is.** Instead of ranking finished turbo renders (GRPO), backpropagate an identity reward through the last K steps of the few-step sampler on a fresh render: run T-K steps under no_grad, the last K with gradients (each step checkpointed), decode only the face crop, score against the reference centroid, and add a velocity-space KL w_t ||v_theta - v_ref||^2 over the same K steps. The gradient is first-order (pathwise): no likelihood ratios, no SDE, no group sampling. It optimizes what the LoRA actually generates rather than a noised training image.

**Evidence**
- [IPRO (arXiv 2510.14255, CVPR 2026)](https://arxiv.org/html/2510.14255v4), [poster](https://cvpr.thecvf.com/virtual/2026/poster/39047): Wan2.2 5B, Wan2.2 A14B (low-noise expert only), in-house 15B I2V; 8-step Lightning without CFG, gradients through the last K=4; lambda_reward 0.1, lambda_KL 1; Adam 2e-5, 100 steps, batch 64. FaceSim: base 0.578, DPO 0.628, SFT 0.639, GRPO 0.633, IPRO 0.694; per-model gains +0.12 to +0.22. Hacking rate: base 7%, without KL 58% (FaceSim 0.754), without the face pool 52% (0.739), IPRO 10%. KL/reward ratio 10 is the sweet spot (ratio 1 -> 52% hacking; ratio 100 -> 9% but FaceSim 0.637). K ablation 0.659 / 0.671 / 0.684 / 0.694 / 0.697 for K=1-5. Low-noise gradient window beats high-noise (0.694 vs 0.646). CurricularFace confirms (0.581 -> 0.699). Hardware and LoRA-vs-full not stated.
- [ID-Aligner (arXiv 2404.15449)](https://arxiv.org/html/2404.15449v1): ReFL one-step x0, detector + recognizer cosine reward, SD1.5/SDXL; IP-Adapter face similarity 0.739 -> 0.800 (SD1.5), 0.512 -> 0.619 (SDXL); CLIP-T 22.0 -> 20.6 and 24.5 -> 23.7; Dreamshaper +10.7%, RealVisXL +22.3%. LoRA results only qualitative.
- [Reward Score Matching (arXiv 2604.17415)](https://arxiv.org/html/2604.17415v4): unifies ReFL/DRaFT/Flow-GRPO and recommends terminal-image reward gradients. Correction: its "best paper" status could not be found (an oral at the ICML 2026 SPIGM workshop), and its low-SNR advice does not back IPRO's low-noise window.
- Cautionary: [DiffusionReward (arXiv 2505.17910)](https://arxiv.org/abs/2505.17910) reports reward hacking with ReFL on face models. RulzTrainer's own idloss experiment (README:795-800) found most of the identity-loss drop came from ordinary learning (about 0.07).
- Does not transfer: I2V identity-drift repair vs a T2I LoRA that already knows the identity; batch 64 vs batch 1-4; probable full fine-tune vs rank 8-16.

**RulzTrainer today.** GRPO-like stage (reward.py:1-24, README:1064-1106) with non-differentiable scorers via `_reward_of` (trainer.py:2915-2940), one GPU. The KL to a frozen stage-start copy (trainer.py:2770-2771, 2849-2858) is already velocity-space: `kl_to` is (mean - mean_ref)^2 / (2 std^2) (reward.py:85-87) and the SDE mean is affine in v. idloss.py already has a differentiable path (face cells of x0, `decode_latents_grad` as a bf16 checkpointed VAE copy, differentiable 5-point alignment, ArcFace in torch) and a centroid of all references (idloss.py:205, `pick_ref` 229-233 with `id_loss_random_ref=false`); the mean cosine over all references equals |mean r| x cos(e, centroid), so IPRO's pool reward is the centroid mode up to a constant. The reward stage's face scorer takes the best match (facesim.py:217; confirmed). Delta: backprop through K real sampler steps of a fresh render, scored on the generated face.

**Fit and risks.** Gradients already flow through frozen QuantLinear (int8 W8A8 with bf16/int8 backward, nvfp4 backward); pathwise also needs input-grads through every block and the embedder because x_t chains across steps. `lora_blocks` and keep blocks give no saving; the stage already releases kept activations (README:1105). No double backward, so compile is fine. With each of the K steps checkpointed on top of per-block checkpointing, peak is about one training step plus stored x_t plus about 2 GB for crop decode and ArcFace (README:797); K=2-4 at 512-768 px fits 16 GB on both cards. Block swap and model_split compatible; stage runs on the first GPU alone (trainer.py:2763), no NCCL. Face location: no-grad decode of the final latent, InsightFace detection, then the differentiable crop decode. Use the turbo sampler's own sigmas. Races already excluded (config.py:541). Export unchanged. Awkward: Krea 2's turbo is a separate model; the stage today renders Raw at 4 steps with CFG 5.5 (README:1094-1095), doubling cost; pathwise through Turbo needs that model loaded as the stage base. MOSS 8-step DMD LoRA and Qwen 2.1 Turbo LoRA fit cleanly; FIBO depends on its probe steps. The reward is the probe's own ArcFace, so an independent scorer (FaceNet backend in idloss.py) is mandatory. Prompt adherence can drop (ID-Aligner CLIP-T -0.8 to -1.4).

Estimated cost per pathwise update at batch 1 (unmeasured): Qwen 2.1 int8 about 10-15 s, MOSS int8 about 20-30 s, Krea nvfp4 about 40-60 s (double with Raw + CFG), versus the measured 77 s GRPO step on Krea.

**Implementation path**
1. config.py (404-412): `reward_mode: grpo|pathwise` (default grpo), `reward_k_steps` (default 2), `reward_id_weight` (default 0.1; `reward_kl` stays the KL weight, giving IPRO's 10:1).
2. reward.py `pathwise_step()`: no_grad turbo ODE for T-K steps; last K with grad, each in `torch.utils.checkpoint`; per step the reference velocity with stage-start weights via `_swap_weights`, add w_t ||v - v_ref||^2; score final x0.
3. Identity term: reuse `idloss.IdentityLoss` (crop, `decode_latents_grad`, align, embed vs centroid); skip images with no face.
4. `trainer._reward_stage`: branch on `reward_mode`, sharing EMA copy, ref params, AdamW at `reward_lr`, dropout off, probes, saves, one-GPU rule.
5. Side fix for GRPO too: `_reward_of` / FaceScorer option to use mean cosine over references instead of best match.
6. Models: MOSS and Qwen 2.1 first; Krea 2 only with Turbo as stage base (or Raw + CFG at double cost); FIBO at its probe steps; Flux 3 Klein later. README "Likeness reward fine-tuning" and wiki.

**How to test.** One race-winner LoRA per model as common start, same reward prompts, probe seeds and about 1 h wall-clock. Arms: GRPO today; GRPO with pool-mean face score; pathwise K=2; pathwise K=4 where VRAM allows; pathwise without KL (hacking control). Qwen 2.1 int8 first, then MOSS. Metrics: ArcFace probe, an independent face score (FaceNet/VGGFace2 or CurricularFace on the same probe images), body score, prompt adherence (CLIP-T or a VLM yes/no on clothing and setting), blind look at probe grids (skin texture, eyes, pose sameness). Stop on the existing probe early stop/patience, keep the best checkpoint. Adopt if pathwise beats GRPO and the start by >= +0.02 on the independent face metric on 2 of 3 seeds/subjects, with <= 0.01 CLIP-T (or 5% VLM adherence) drop and no visible hacking; reject if the ArcFace gain is not matched by the independent metric.

### Identity loss: smooth signal-fraction weight instead of the sigma 0.35 cutoff (plus optional negative triplet)

**Race it (the window half is testable today). Expected gain: a few hundredths of ArcFace/FaceNet similarity at best, unmeasured on DiT LoRAs; triplet likely zero to marginal. Confidence: medium (0.4). Cost: no code for the first race; small change in idloss.py for signal weighting; +0.2 s per eligible sample.**

**What it is.** Replace the hard `id_loss_sigma_max` cutoff with a per-sample weight equal to the signal fraction, so mid-noise steps (where face shape is decided) contribute in proportion to how face-like the one-step x0 is. The flow-matching equivalent of DDPM's sqrt(alpha_bar_t) is (1-s)/sqrt((1-s)^2+s^2), not (1-s) as the scout proposed. Optional: a triplet hinge max(0, cos_hardest_neg - cos_ref + margin) against a bank of other people's face embeddings, making the loss discriminative.

**Evidence**
- [ReF-LDM (arXiv 2412.05043)](https://arxiv.org/html/2412.05043v1): L_ID weighted by 0.1 x sqrt(alpha_bar_t). FFHQ-Ref-Severe: no ID loss IDS 0.52 / NIQE 4.56; unscaled 0.69 / 6.56 (quality collapse); scaled 0.65 / 4.38; indicator t<100 0.52 / 4.55 (no gain); t<500 0.61 / 4.44. Reference-based face restoration with an LQ-conditioned LDM (fine-tuned 150k iterations at batch 8 on 4x 3090; ablations 50k steps). Not LoRA, not T2I.
- [Face inpainting with identity-preserving LDMs (arXiv 2605.16696)](https://arxiv.org/html/2605.16696v1): cosine loss plus a batch triplet (random other sample as negative), SD1.5, 1x RTX A6000. Correction: the 0.62 -> 0.78 ArcFace (E-Mask) and 0.385 -> 0.611 CosFace compare plain SD1.5 with the full ID-ControlNet; the triplet and t_s are never ablated, so the loss's own contribution is unknown.
- [LoRACLR (arXiv 2412.09622)](https://arxiv.org/html/2412.09622): d_p^2 + max(0, 0.5 - d_n)^2, ArcFace 0.828 vs 0.799 (+0.029). Correction: distances are L2 between LoRA layer outputs during merging, not identity embeddings; not evidence for an ArcFace triplet.
- Contrary: PersonaMagic (arXiv 2412.15674) found the one-step x0 at large t too blurry for identity guidance in T2I and timestep weights did not fix it; PuLID (arXiv 2404.16022) computes its ID loss on a few-step Lightning render instead. ReF-LDM's high-noise x0 is face-like because the LQ image conditions it; a text-conditioned LoRA lacks that anchor.

**RulzTrainer today.** Direct identity loss with a hard cutoff: `eligible()` returns False when sigma > sigma_max (idloss.py:211-217, confirmed), loss = mean 1 - cos vs one random or the mean reference (:235-258). Options config.py:387-393 (`id_loss_weight` capped at 0.5, `id_loss_sigma_max` 0.35, backend arcface|facenet); used at trainer.py:1015-1022 on the primary replica; README:769-803. No triplet or negative bank (grep). A ready negative bank exists: LoRA Isolation's 32 generated strangers (isolation.py:163). The "DOP class renders" bank does not exist: DOP regresses to the base prediction without rendering. Useful fact: under SNR matching, sigma 0.35 equals DDPM t of about 124, close to ReF-LDM's t<100 indicator that gave no gain; their t<500 equals flow sigma of about 0.735.

**Fit and risks.** The loss sits outside the DiT blocks (separate bf16 VAE decoder with checkpointing, fp32 ArcFace/FaceNet), so W8A8, nvfp4, compile, keep blocks and block swap are untouched; model_split already moves pred to the crop's device (idloss.py:249); data_parallel keeps it on the primary replica; per-sample weight on the snapped sigma fits the 1000-step grid. Cost: README measured +0.06 s/it averaged at 22 of 80 eligible steps, about +0.2 s per eligible sample at 512 px and +2 GB peak; with most steps eligible about 8% at Qwen 2.5 s/step, under 10% at Krea 9 s/step or MOSS 5 s/step, no extra peak at batch 1. Keep a sigma floor (about 0.8) for decode time and because ArcFace on blurred x0 is noise. The arcface backend trains on the probe metric, so race with facenet. With cos(gen, ref) around 0.5-0.8 and cos to strangers around 0-0.2, a 0.3-margin hinge rarely fires. The proposed LoRACLR velocity margin on the DOP pair conflicts with isolation: dropped.

**Implementation path**
1. config.py: `id_loss_weighting: cutoff|signal` (default cutoff), `id_loss_signal_power` (1.0), `id_loss_sigma_floor` (0.8), `id_loss_triplet_weight` (0), `id_loss_triplet_margin` (0.2).
2. idloss.py: in signal mode `eligible()` uses the floor; `__call__` returns sum(w_i l_i)/sum(w_i) with w = ((1-s)/sqrt((1-s)^2+s^2))^p.
3. `build()`: negative bank from the isolation strangers' face crops plus other datasets' face_refs, cached as `.rulz_cache/id_negatives.pt`.
4. Race rebind (trainer.py about 1777-1783), race knob list (race.py:60-61), WebUI identity-loss panel, README:769. `race_parallel` rejects the identity loss (race.py:220-222), so use the sequential race.

**How to test.** Sequential race on Qwen 2.1 int8, 0.5 MP, rank 16, `id_loss_backend: facenet` so the ArcFace probe stays independent. A: `id_loss_weight 0`; B: cutoff 0.35, weight 0.2; C: cutoff 0.7 (testable today); D: signal, p=1, floor 0.8; E: D + triplet 0.1, margin 0.2. Metrics: ArcFace face probe (primary), DINO body appearance and proportions (must not fall more than `likeness_noise_band`), eye check for over-sharpened faces. Log eligible share, mean weight, and for E the hinge-active fraction (drop the triplet if under 10%). Race drop-lowest with usual patience up to 2000 steps; a winner must beat A and B by more than 2x `likeness_noise_band` at its best checkpoint on two seeds; repeat on Krea 2 nvfp4 before changing defaults.

### Stop schedule for id_align/body_align (HASTE), after measuring id_align at all

**Race it (stop half); Watch (attention-map distillation). Expected gain: unquantified for LoRA likeness; probably within probe noise at id_align_weight 0.05, possibly useful as a lever that makes 0.3-0.5 safe early. Confidence: low (0.3). Cost: stop schedule is a small trainer change; attention distillation is a large build at +2-5% step time and +0.2-0.6 GB.**

**What it is.** HASTE found that REPA-style representation alignment speeds early training but caps quality later, and that switching it off at a fixed step (or when the alignment gradient turns against the diffusion gradient) recovers quality. It also distills DINOv2 attention maps into early student blocks. For RulzTrainer the transferable part is a stop/fade schedule on `id_align`/`body_align`, possibly allowing a higher early weight.

**Evidence**
- [HASTE, "REPA Works Until It Doesn't" (arXiv 2505.16792)](https://arxiv.org/html/2505.16792v1): SiT-XL/2 + HASTE FID 8.39 at 50 epochs vs vanilla 8.61 at 1400 (28x); Table 3 at 100 epochs REPA 7.5 / ATTA 8.5 / REPA+ATTA 8.1 / with termination 5.3; Table 4 at 500K always-on 8.1, tau=400K 7.4, tau=250K 5.3. But at 400K iterations always-on wins (5.5 vs 7.3 with tau=250K), so termination only pays late and stopping early hurts for a long stretch. lambda_R = lambda_A = 0.5; gradient-cosine trigger "works nearly as well" as fixed tau, no threshold given. MS-COCO MM-DiT 150K: FID 15.42 / 10.40 / 9.63 without CFG, 6.35 / 4.95 / 4.55 with CFG (attention alignment only, no termination). A100/H100, batch 256, fp16. From-scratch only; venue not stated on the page; no code link found.
- [MaskAlign (arXiv 2606.08788)](https://arxiv.org/html/2606.08788v1): aligning a random 25% token subset cut iteration time 0.359 -> 0.317 s (11.6%) and FID 3.4 -> 2.8 at 400K; ImageNet/SiT only.
- Does not transfer: all FID deltas; the capacity-mismatch argument (a rank-16 LoRA on a mature frozen base never "models the full distribution"). REPA/HASTE use weight 0.5; RulzTrainer's 0.05 leaves little late drag to remove.

**RulzTrainer today.** `IdAligner` (alignment.py:185-230): mean of face/person tokens after one block -> 2-layer SiLU MLP -> ArcFace 512 / DINOv2-base CLS 768, 1-cosine. config.py:397-401 (`id_align`, `id_align_weight` 0.05, `id_align_layer` auto = depth/3, `body_align`; confirmed). `_alignment_losses` at trainer.py:2715-2745 applies a constant weight every step. README:809 says these losses are "not measured for likeness yet". No stop/fade, no patch or attention target. Attention taps exist: `AttnTap` (alignment.py:91-109) gets rotated q,k before SDPA on Krea 2 (krea2/model.py:106-108), Qwen 2.1 (qwen21/model.py:126-128), MOSS (moss/model.py:117); FIBO's tap is on injection cross-attention (fibo/model.py:182).

**Fit and risks.** The stop schedule is pure trainer logic: no effect on W8A8/nvfp4, compile, block swap, model_split, sigma grid or export (the projector is discarded). The parallel race trains without alignment losses (trainer.py:2591 warning), so use the sequential race. Gradient-cosine trigger: skip; it needs a separate backward through a third of the depth every N steps (~1-2% at N=50) and cross-replica grad reduction under data_parallel, for a decision the probe plateau already makes (`LikenessController.since_improve` / `stage_since_improve`, likeness.py:231-287). Attention distillation: tapped blocks run uncompiled (trainer.py:1479, 2614); softmax(QK^T) for 500-1000 person queries x ~2000 keys x 12 heads in fp32 is ~100-200 MB per layer per sample; DINOv2-B teacher attention cached is ~50 MB per image (1.5 GB for 30 photos) or online +~0.4 GB / ~10 ms; DINO's 14 px grid vs the latent grid needs resampling on both axes; the teacher sees the clean photo while the student at sigma near 1 sees noise; FIBO needs a new self-attention tap.

**Implementation path (stop half only)**
1. config.py: `align_stop: ""|"share:0.5"|"step:N"|"probe:N"` and `align_fade: 0.1` (linear ramp to zero over that share before the stop); validation near config.py:571.
2. trainer.py: compute `align_scale` in the step loop, pass into `_alignment_losses` (trainer.py:2715); at 0 set `use_align` False (trainer.py:970) so `Taps.record_hidden` stops and the aligner's AdamW stops stepping (trainer.py:1141); log the stop step; keep `attn_localize` independent.
3. README 804-836, wizard and race keys.
4. Build `AttnAligner` only if the race shows id_align moves probes at all.

**How to test.** Sequential race on Qwen 2.1 int8 (2.5 s/step, 2000 steps, probe every 100): {id_align off}, {0.05 constant}, {0.3 constant} (all possible today), then {0.3 with align_stop share:0.5} after the code. Read the final ArcFace probe, the best-step probe and the step at which each arm reaches 0.9 of its best; differences below ~0.02 cosine are noise. Repeat on Krea 2 nvfp4 only if Qwen shows signal.

### Likelihood-free reward update (DiffusionNFT)

**Prototype. Expected gain: about the same likeness as the Flow-GRPO stage in about 0.5-0.7x the wall time (verifier estimate, unmeasured); about 1.5-2x cheaper per update when guidance > 1; possibly 1.5-2x stage wall-clock with off-policy rendering on the second GPU. Confidence: medium (0.55). Cost: medium build (new reward algorithm branch).**

**What it is.** DiffusionNFT trains on the forward (noising) process with the ordinary flow-matching loss on the model's own clean renders. Each render's normalized reward r splits the loss into a positive and negative part via implicit policies v+ = (1-b) v_old + b v_theta and v- = (1+b) v_old - b v_theta: loss = r ||v+ - v_target||^2 + (1-r) ||v- - v_target||^2. No likelihoods, no SDE, no stored trajectories, CFG-free training, and renders can come from any sampler, including off-policy on another GPU.

**Evidence**
- [DiffusionNFT (arXiv 2509.16117, ICLR 2026)](https://arxiv.org/html/2509.16117v2): r = 0.5 + 0.5 clip(r_norm/Z, -1, 1), group-centred; SD3.5-Medium at 512 px, LoRA alpha 64 / rank 32, lr 3e-4, 48 groups x G=24 per epoch, 10 rollout steps in ablations, 40 in final runs; 3-25x wall-clock vs Flow-GRPO; GenEval 0.98 in about 1k iterations vs 0.95 after more than 5k with CFG. Removing the negative loss collapses reward almost at once; ODE collection beats SDE; fully on-policy (eta=0) is unstable; time weighting w(t)=1-t collapses. No hardware, GPU-hours, or few-step/distilled sampler.
- [GDRO (arXiv 2601.02036, CVPR 2026)](https://arxiv.org/abs/2601.02036): offline, sampler-independent, DPO-derived; 2x (OCR) to 3.7x (GenEval) faster than Flow-GRPO (not 3-25x). No code confirmed.
- [DFR (arXiv 2610.24897)](https://arxiv.org/html/2610.24897): RAM base plus MMD on SD3.5-M VAE latents (8 reference samples per prompt), FD or DreamSim repulsion; up to +29% diversity and -12% drift at matched reward, 3-9% overhead per step, 35.7-41.1x vs Flow-GRPO; GenEval/OCR only.
- Cautionary: Centered Reward Distillation (arXiv 2603.14128) adds an adaptive KL to NFT because plain NFT reward-hacks (white backgrounds); a Jan 2026 blog analysis argues unanchored online NFT drifts into peaked high-reward regions. Hacking risk is not "unchanged vs GRPO" without an anchor.
- Does not transfer: the speedups (large groups, prompt-following rewards with high in-group variance, many GPUs). ArcFace likeness within a 4-image group varies little.

**RulzTrainer today.** The stage is exactly Flow-GRPO (reward.py:68-129, trainer.py:2752-2895): each update renders G=4 images with the SDE sampler under the probe policy with the turbo LoRA on (trainer.py:2796), trains `reward_train_steps` moves per image through the guided velocity, plus a no-grad reference pass via `_swap_weights` (trainer.py:2840-2860). The ratio exp(lp - lp.detach()) makes it strictly on-policy; one GPU (others wait, trainer.py:2764). Self-training (selftrain.py:1-8, config.py:415-420) is already the positive-only (RAFT) cousin; it mixes in real photos and is capped, so the NFT positive-only collapse does not apply. Correction to the scout: the current stage's log-prob is a cheap Gaussian; its cost is in renders and forward/backward passes, which NFT also needs. NFT's real advantages here: ODE renders the scorers read cleanly, CFG-free training, off-policy rendering on the second GPU.

**Fit and risks.** NFT calls the same `model(x_t, t, emb, mask)` as the GRPO update, so W8A8, nvfp4, LoRALinear/QuantLinear, compile, block swap and model_split are untouched; export unchanged; peak equals the current stage. Main caveat: with `turbo_lora=auto` the policy is base + DMD-style turbo LoRA + training LoRA at 8 sigmas, and a distilled model's v_old at arbitrary t is not the marginal velocity of its samples, so NFT's optimum is approximate (group-centred r cancels the mismatch to first order). Draw training sigmas from the render schedule (snapped to the 1000-grid), not logit-normal, or the update partly un-distills turbo. Keep v_old as a lagged copy separate from the stage-start KL anchor, and keep `reward_kl`. Off-policy rendering fits the thread-based data_parallel copies (README:281-300) with a periodic LoRA copy to GPU 1; not possible under model_split. Consider `reward_group` 6-8 for NFT because a 4-image group gives a noisy r.

**Implementation path**
1. config.py: `reward_algo: grpo|nft` (default grpo), `nft_beta` (0.1-1.0, default 1.0), `nft_eta` (soft update of v_old, ramp 0 -> 0.5), `reward_timesteps` (K, default 4).
2. reward.py: `sample_group_ode` (noise_level 0, final latents only), `nft_rewards()`, `nft_loss(v_theta, v_old, x0, eps, r, beta)`.
3. `_reward_stage`: per image draw K sigmas from `sig[:-1]`, x_t = (1-s) x0 + s eps, target eps - x0; v_old under no_grad with lagged weights (second `_swap_weights` buffer); v_theta with grad, no guidance; add ||v_theta - v_ref||^2 to the stage-start copy; soft-update old after each optimizer step.
4. Optional: render group s+1 on devices[1] while devices[0] trains on group s via the side-job queue.
5. Later: `reward_diversity: mmd` on cached base-render latents per prompt (8 base renders per reward prompt, +3-9% per step).

**How to test.** From one character LoRA (`resume_from`, steps 0) on Krea 2 nvfp4 and Qwen 2.1 int8: `reward_steps 200`, grpo vs nft, `reward_lr` per method (nft about 3e-5 to 1e-4), same prompts and seeds, `probe_every 25`. Metric: combined probe likeness vs wall-clock seconds, plus drift as mean probe DINO distance to the start LoRA's renders, plus an eye check for plain backgrounds, frozen pose, oversaturated skin. NFT wins if it reaches GRPO's best probe in <= 0.7x wall time or >= 1 point higher at equal time with no larger drift. Stop on the existing early stop or 4 probes without improvement; repeat on two characters. Also run grpo vs nft at probe_cfg > 1 without the turbo LoRA to isolate the CFG-free saving.

### Reward-stage anti-hacking: identity ceiling and in-group diversity multiplier

**Race it. Expected gain: zero to small gains in visual robustness (stiff/copied faces, pose collapse, texture artifacts) at roughly equal likeness; about +5% stage time. Confidence: low (0.35). Cost: small change in reward.py/_reward_of.**

**What it is.** Two cheap guards against an ArcFace reward being gamed: (1) a ceiling on the identity part so renders at or above reference likeness stop competing on identity (DivRL's hinge reduced to the existing clip); (2) multiply positive advantages by 1 + w x (1 - max DINOv2 cosine to the other renders in the group), following GARDO, to counter mode collapse. Likeness itself is not expected to rise; what rises is how much of the ArcFace gain is real.

**Evidence**
- [GARDO (arXiv 2512.24138)](https://arxiv.org/html/2512.24138v1): SD3.5-M LoRA, 8xA800, about 49 s/step, G=24. GenEval task: GRPO beta=0 0.95 / HPSv3 6.73 / diversity 15.6; GARDO 0.95 / 9.27 / 24.95; also improves DiffusionNFT (HPSv3 12.17 vs 5.66). "GARDO w/o div" at beta=0.04 already reaches 8.61 / 19.98, so about half of the gap comes from the diversity term. Selective KL and reference reset are never ablated separately; baseline is unregularized GRPO.
- [DivRL (arXiv 2606.23950)](https://arxiv.org/html/2606.23950): Flux-Kontext LoRA (1.8% params), Flow-GRPO, group 21, 6 steps, beta 0.1; R = nSSM if VSM >= 0.5 else nSSM - 5(0.5 - VSM)^2; full 48x48 similarity grid caused high-frequency artifacts, fixed by 2x2 pooling; DINO 0.594 -> 0.600, CLIP-I 0.781 -> 0.786, VSM 0.605 -> 0.614; 8xA100, 2 x 24 h. ECCV venue unverified.
- [AuK (arXiv 2609.08936)](https://arxiv.org/html/2609.08936v1): speaker similarity standardized only over candidates with correct content; target KL 1e-3. Speech, not images.
- [IPRO (arXiv 2510.14255)](https://arxiv.org/html/2510.14255v4): hacking 58% without KL -> 10% full method, but FaceSim also falls 0.754 -> 0.694, so not "at equal FaceSim".
- [ReNFT (arXiv 2609.00061)](https://arxiv.org/abs/2609.00061): collapse is suppression, not deletion; DreamSim-Div +58.8% / +55.0% at 98.9% / 99.0% of reward.
- Adjacent: D2-Align (arXiv 2512.24146) reports Flow-GRPO collapses diversity worst, including identity-collapsed faces; DisCo (arXiv 2510.01399) found diversity rewards get hacked too (fewer people, grid artifacts).
- Does not transfer: every positive result uses groups of 21-24 or batch 64 on datacenter GPUs.

**RulzTrainer today.** Group-relative advantages (reward.py:90-96), KL to the stage-start LoRA with `reward_kl` 0.04 (trainer.py:2771, 2859-2860; config.py:411), per-part clipping -0.5..1.5 (reward.py:52-62), best-match face scoring over all references (facesim.py:207-217), probes with best checkpoint and early stop in the stage (trainer.py:2875-2885), likeness-filtered self-training (README:999). DINOv2-base already loaded by the body scorer (bodysim.py:434). No diversity term, gating, selective KL, resets, curriculum or DPO mode.

**Fit and risks.** All changes are in advantage computation and KL; no effect on quantized base, compile, block swap, sigma grid or export. About one ViT-B forward per render for the DINO embedding. Dropped as not fitting `reward_group` 4: top-k selective KL (k=0.1 selects nothing), gated z-scores over ~2 survivors, reward-gap curriculum (6 pairs). Dropped as too costly: HPSv3 quality gate (HPSv3 is Qwen2-VL-7B, not one ViT pass; it would need load/unload per group or the second GPU), offline angle-DPO (about 10k renders plus 500 DPO steps, overlaps self-training). A reference reset weakens the only anchor to the first-stage LoRA, and a 100-300 step stage barely reaches 100 steps.

**Implementation path**
1. `Trainer._reward_of` (trainer.py:2915) also returns face and body parts and the renders' DINOv2 CLS embeddings (body scorer's model).
2. `reward_id_ceiling` (default 0 = off; e.g. 1.0 on the probes' linear scale = reference level): part reward = min(value, ceiling), via `clip_part` (reward.py:55).
3. `reward_diversity` (default 0): after `RW.advantages`, positive advantages x (1 + w x (1 - max cosine to other renders)); recommend `reward_group` 6-8 when on.
4. Optional `reward_kl_reset` (steps, default 0).
5. Log in-group DINO spread and face-score spread per reward step.

**How to test.** Same first-stage LoRA, seed and 200 `reward_steps`, Qwen 2.1 int8 then Krea 2. A: current stage at `reward_group 6`; B: `reward_id_ceiling 1.0` + `reward_diversity 0.5` at group 6. Metrics: probe face and body every 25 steps; mean pairwise DINOv2 distance and ArcFace-to-reference spread over 16 extra renders of 4 unseen prompts at stage start and end; blind A/B for stiff or copied faces and texture artifacts. Accept B if final probe likeness is within 1 point of A and DINO spread is >= 15% higher (or blind preference > 60%); reject if likeness drops > 2 points or spread does not move.

### Patch-level reference-match weights for self-training renders (PatchDPO)

**Race it (selftrain mode only). Expected gain: unmeasured; near zero or slightly negative on real photos; a small artifact-rejection gain on self-training renders is plausible. Confidence: low (0.35). Cost: small to medium (cache-time map in regions.py, no trainer-loop change).**

**What it is.** Score every 16 px cell of an image by its max cosine to any DINOv2 patch of the subject's real reference photos and multiply that into the region weight map. On real photos this mostly reproduces semantic regions; the useful case is self-training renders, where cells with no match in any real photo (broken hands, smeared accessories, odd backgrounds) are down-weighted or moved into the prior mask instead of being learned as the subject.

**Evidence**
- [PatchDPO (arXiv 2412.03177, CVPR 2025)](https://arxiv.org/html/2412.03177v2): patch quality = max cosine to reference patches using layer-7 features of an ImageNet ViT-B after 10 epochs of rotation/flip self-supervision; generated patches weighted by quality, reference by 1 - quality (low-quality patches are down-weighted, not pushed away). IP-Adapter-Plus SDXL, only the projection trained, 8 GPUs, 30k steps, batch 4/GPU, lr 3e-5: DINO 0.692 -> 0.727, CLIP-I 0.826 -> 0.838, CLIP-T 0.281 -> 0.292.
- [NearID (arXiv 2604.01973)](https://arxiv.org/html/2604.01973): frozen SigLIP2-so400m + ~15M MAP head, 19,386 identities and 316,505 distractors, ~6.5 A100-80GB hours; SSR 30.74% -> 99.17%; DreamBench++ human Pearson 0.516 -> 0.545. Scoped to rigid objects, explicitly not faces; a global embedding, not a patch feature. Not usable here.
- Does not transfer: neither paper trains a LoRA, uses flow matching or targets human identity. PatchDPO's setting (generated images scored against references) maps onto self-training renders, not real photos.

**RulzTrainer today.** Per-16-px-cell weight maps with mean 1 built at cache time (regions.py:209-330), stored as `<stem>_regions.safetensors`; trainer multiplies per-cell squared error by them (trainer.py:981-982); `region_prior_weight` pulls masked cells toward the base prediction (trainer.py:984-993). Self-training renders get their own region maps in `<job>/selftrain/cache` (trainer.py:3084-3113). DINOv2-base loaded for body appearance (bodysim.py:434, 549), CLS only today; patch tokens are `last_hidden_state[:, 1:]`. Region measured +1.8 face-likeness EMA (76.0 vs 74.2) on the 5-image synthetic set, for reference.

**Fit and risks.** Computed entirely at cache time on the side GPU: no effect on W8A8/nvfp4, compile, keep blocks, block swap, model_split, data_parallel, sigma grid or export. About 0.4 GB VRAM for DINOv2-base at cache time; the N^2 x patches^2 similarity pass is trivial for 20-100 images. Races already get separate region maps per config (README:1234). Risks: on real photos the map correlates with InsightFace/SAM3 cells; recurring outfits across a photo set get high weight, binding clothing to the trigger (the SAM3 background cap fixes backgrounds, not clothing).

**Implementation path**
1. regions.py `build_match_map()`: DINOv2-base patch tokens from bodysim's AppearanceScorer, resampled to the bucket's 16 px cells; per cell max cosine to patches of the other real images (excluding itself and near-duplicates, 32 px thumbnail correlation > 0.95 as selftrain uses); normalize to [0, 1].
2. Blend: `weight *= floor + (1 - floor) * match^gamma`, renormalize to mean 1, save in the same `_regions.safetensors` (trainer.py:981 unchanged).
3. config.py: `region_match: off|selftrain|all` (default off), `region_match_floor` (0.3), `region_match_prior` (bool: low cells ORed into `RegionMap.prior`).
4. Wire into `cache_region_maps` for real items in 'all' mode and the selftrain add path (trainer.py ~3108); preview overlay via `render_overlay`; race variant field.
5. Skip PatchDPO's backbone tuning and the NearID head.

**How to test.** First, no training: Pearson correlation between match map and face/body region map over the dataset; above 0.9 means 'all' adds nothing, keep only 'selftrain'. Then race the identity preset (`region_weights` on, `selftrain` on, `selftrain_at 0.5`) vs + `region_match: selftrain` (and + `all` as a third racer), one config per GPU, Qwen 2.1 int8 at 0.5 MP, 2000 steps, probes every 100, 2-3 seeds, 2 characters. Keep only if final face EMA is >= +1.0 over baseline averaged across seeds, body EMA drops <= 0.5, and likeness does not dip in the 300 steps after the self-training round (disable the race's early drop in that window).

### Overfit meter from LoRA-vs-base loss per sigma bin (APT), diagnostic only

**Skip-ish (as a control lever); Do it only as a logged diagnostic. Expected gain: diagnostic value only; as a loss/sigma lever neutral to negative for ArcFace likeness. Confidence: low (0.3). Cost: small; zero extra compute when region_prior_weight > 0, else about +1 forward per step.**

**What it is.** APT measures overfitting per timestep bin as gamma_t = 1 - exp(-T x (EMA_t[L_base] - EMA_t[L_lora])) and uses it to scale the loss by (1 - gamma_t) and gate augmentation. Proposed here as a logged per-sigma-bin chart of how far the LoRA beats the frozen base on the same noised input, available for free inside the `region_prior_weight` pass.

**Evidence**
- [APT (arXiv 2507.02687, CVPR 2025)](https://arxiv.org/html/2507.02687), [CVF page](https://openaccess.thecvf.com/content/CVPR2025/html/Chae_APT_Adaptive_Personalized_Training_for_Diffusion_Models_with_Limited_Data_CVPR_2025_paper.html): 10 bins of 100 steps, EMA 0.1 (SD2.1 needed T/10, so the scale is hand-tuned per model); p_aug = clamp(gamma_t, 0, 0.8) with zoom-out and +/-15 deg rotation; L = (1-gamma_t) L_DM + 30 (L_mu + L_sigma) + 3e-4 L_attn; SDXL rank-32 LoRA on UNet and text encoder, batch 1, one A100. Full APT: FID 53.130 -> 41.967, precision 0.565 -> 0.669, recall 0.608 -> 0.738, DINOv2 0.681 -> 0.660, user study 56.1 / 22.8 / 21.1 (20 participants). The indicator alone: FID 53.1 -> 46.9, recall 0.608 -> 0.680, DINOv2 0.681 -> 0.670 (down). Subject fidelity drops at every stage; the gains are diversity and prompt adherence.
- [Augmentations for facial resemblance (arXiv 2505.03557)](https://arxiv.org/html/2505.03557v2): random rotation and horizontal flip hurt facial identity (RealVisXL, RTX 3090).
- The per-token Rho-1 "excess loss" extension cites no diffusion source.

**RulzTrainer today.** losswatch.py:63-135 keeps decayed per-sigma-bucket (40 buckets) loss expectations and per-image residuals, never against the base. `region_prior_weight` (trainer.py:984-993) already runs a no-grad LoRA-off forward on the same noisy input, sigma and trigger embeds, so base MSE is free there. Correction: DOP (`_dop_pass`, trainer.py:2458-2476) uses the class caption with fresh noise, so it gives no trigger-caption loss gap; "free with DOP" is wrong. Latents are cached, so APT's per-step zoom-out/rotation cannot run; `crop_augment` (dataset.py:204) only zooms in. `sample_sigmas` (trainer.py:215) has no per-bin hook.

**Fit and risks.** Reuses `set_multiplier(0)` and the compiled forward; no effect on quant, keep blocks, block swap or export. Bin EMAs need a lock under data_parallel/model_split threads. Not in finetune mode (config.py:662). Likeness risk: the bins where the LoRA beats the base most (mid sigma) are where likeness is learned; scaling them down fights the target. Probes, best checkpoint and early stop already catch overfitting on the real metric. Without region_prior: about +3 s/step on Krea nvfp4, +1.7 s on MOSS int8, +0.8 s on Qwen 2.1 int8, or less every k steps.

**Implementation path.** In work() inside the region_prior block (trainer.py ~989) compute base_per = (base_pred - target)^2 region-weighted like `per`; feed (sigma, per, base_per) to an `OverfitMeter` in losswatch.py (10 bins, EMA 0.1, lock). Options `overfit_meter: off|log|gate`, `overfit_temp` (1000, needs re-tuning for velocity MSE), `overfit_meter_every`. Gate mode: weights x max(1 - gamma_bin, `overfit_min_weight` 0.5); optional sigma reweight hook like `logit_shift`. Leave out augmentation and the per-token variant.

**How to test.** Step 1, meter only: one Krea 2 or Qwen 2.1 character LoRA with `region_prior_weight > 0` and `overfit_meter: log`; check whether gamma_t rises before the ArcFace probe peaks and declines. If it does not lead the probe, stop. Step 2 only if it leads: A region_prior on; B gate with loss scaling; C gate + sigma reweighting; 2000 steps, rank 16; best-checkpoint ArcFace must not drop more than 0.01, plus ArcFace and CLIP-T on 8 non-training prompts; drop if ArcFace is lower than A in 2 of 3 seeds or no diversity gain at equal likeness. Side note from the verifier: APT's representation stabilization (matching base mid-block statistics under the class prompt) drives most of its gain and could be a separate candidate.

### DCO implicit-reward loss and consistency guidance

**Skip-ish. Expected gain: at best a small prompt-fidelity / less-overfit gain at about equal likeness; possible likeness loss from saturation. Confidence: low (0.35). Cost: small code; +25-35% s/step (unmeasured) unless region_prior_weight is already on; guidance doubles DiT passes per render.**

**What it is.** DCO replaces the diffusion loss with -log sigmoid(-beta_t (err_theta - err_base)), whose gradient is the normal gradient times (1 - sigmoid(d_t)): updates are damped per sample and timestep where the LoRA has already moved far from the base. Consistency (reward) guidance at sampling adds omega_con (v_lora - v_base), which in the linear regime approximates LoRA strength 1 + omega.

**Evidence**
- [DCO (arXiv 2402.12004, NeurIPS 2024)](https://arxiv.org/html/2402.12004v2), repo github.com/kyungmnlee/dco ("reward guidance", `RGPipe`, `rg_scale`): constant beta_t = 1000 (ablation 500/1000/2000); omega_con 2-5; SDXL, subject LoRA rank 32, lr 5e-5 plus textual inversion at 5e-4. User study vs DreamBooth: 55.1 subject / 72.7 prompt / 70.6 quality; vs DB+prior preservation 81.9 / 58.0 / 63.0. Merge: DCO 0.462/0.651/0.773 vs DB 0.386/0.672/0.430. About 1.3x training time per iteration, about 2x sampling time with guidance; multi-subject blending fails for similar subjects.
- 55.1% subject fidelity vs DreamBooth is near parity; the gains are prompt fidelity and quality. SDXL eps-prediction only; beta was set for eps-MSE scale and must be re-tuned for velocity MSE. No flow-DiT follow-up found.

**RulzTrainer today.** The LoRA-off base prediction is already computed by `region_prior_weight` (trainer.py:984-993, config.py:363, README:726-727) and by DOP (trainer.py:2458-2478); `set_multiplier` toggling works over QuantLinear (W8A8, nvfp4), compiled blocks, model_split and data_parallel. No DCO loss and no LoRA-vs-base guidance (sampler.py:37-41 only does CFG; reward.py:100-106 likewise). Note: region_prior only runs when regions exist and `regions.prior.any()`; DCO needs the base forward on every sample.

**Fit and risks.** The extra no-grad forward stores no activations (keep blocks unaffected) but re-streams swapped blocks under block swap. Adam, adamw8bit and automagic3 are scale-invariant, so only the relative (1 - sigmoid(d)) weighting matters. The loss saturates once the LoRA beats the base on an image by about 1/beta, which can cap face learning where character LoRAs need to move furthest; it overlaps DOP + region_prior + lora_ema + probe early stop. Guidance on turbo probe samplers (MOSS 8-step DMD, Qwen Turbo LoRA, Krea Turbo) would overcook (trainer.py:2160-2174 already warns about guidance on distilled samplers) and would skew best-checkpoint, race and early-stop decisions away from strength 1; never use it in probes or the reward stage.

**Implementation path.** config.py: `loss_mode: mse|dco`, `dco_beta` (1000, race 300-3000). In work() (~980-993) reuse the region_prior base forward or run one; per = -logsigmoid(-dco_beta x (err_theta - err_base.detach())) before `loss_weight`/look/loss_watch; log sigmoid(d) per image into loss_watch as a memorization signal. Reject with finetune (as config.py:662). Optional `consistency_guidance` in sampler.py for user samples only, refused or warned on turbo samplers.

**How to test.** Qwen 2.1 int8, 2000 steps, rank 16: A mse; B mse + `region_prior_weight` (cost control); C dco beta 1000; D dco beta 300. Probe face and body plus prompt adherence on held-out probe prompts and s/step. Keep DCO only if best-checkpoint face is within 0.01 of A/B with better prompt adherence, or likeness improves, at <= 1.35x wall-clock; drop at race patience if likeness trails by > 0.02 for 3 consecutive probes. Guidance test: non-turbo sampler, omega_con {0,1,2,3} vs LoRA strength {1.0,1.5,2.0} at equal wall-clock; adopt only if guidance beats the strength sweep.

### Noise consistency regularization (multiplicative latent perturbation)

**Skip-ish. Expected gain: 0 to +0.01 on the face probe (verifier estimate), possibly negative; main benefit in the paper is background diversity. Confidence: low (0.3). Cost: one extra forward + backward per application (+50% step time every 2nd step, +25% every 4th).**

**What it is.** Perturb the clean subject latent multiplicatively, z' = z x eps_m with eps_m ~ N(1, 0.2^2), noise z and z' with the same noise and timestep, and penalize the difference of the LoRA's predictions, so the LoRA encodes the subject rather than pixel-exact latents. RulzTrainer already has the prior-consistency half (DOP, region_prior); only the subject-consistency half is new.

**Evidence**
- [Noise Consistency Regularization (arXiv 2506.06483, CVPRW 2025 SyntaGen)](https://arxiv.org/html/2506.06483v1), [CVF page](https://openaccess.thecvf.com/content/CVPR2025W/SyntaGen/html/Ni_Noise_Consistency_Regularization_for_Improved_Subject-Driven_Image_Synthesis_CVPRW_2025_paper.html), [MERL TR2025-073](https://merl.com/publications/docs/TR2025-073.pdf): lambda_cp = lambda_cs = 0.5; SD1.5 UNet LoRA rank 4, lr 5e-4, DreamBooth with 100 prior images; 30 DreamBooth subjects (9 live, 21 objects), 10 prompts x 4 images. DINO/CLIP-I/CLIP-T: DreamBooth 0.602/0.778/0.329, Ours 0.634/0.792/0.324, DB+DCO 0.638/0.795/0.309, Ours+DCO 0.652/0.811/0.310. About 2x training FLOPs and memory.
- The gain is L_cp + L_cs combined; the L_cs-alone sweep and additive-vs-multiplicative comparison are qualitative only. CLIP-T drops. Mostly objects, not face ID. Single paper, v1, no follow-up or DiT replication.

**RulzTrainer today.** DOP (isolation.py; `_dop_job`/`_dop_pass` trainer.py:2444-2477; `dop_weight` 2.0, `dop_every` 2 at config.py:246-247) and `region_prior_weight` (trainer.py:984-993) cover L_cp. No latent perturbation or prediction-consistency term (only flip, crop and caption augmentation).

**Fit and risks.** Key correction: the scout's plain MSE between the two predictions conflicts with flow matching. Target = noise - x0 (trainer.py:963), so the two true velocity targets differ by z' - z. Compare in eps space: eps_hat = z_t + (1-s) v_hat, equivalent to (1-s)^2 ||v_hat' - v_hat - (z - z')||^2, which also vanishes at high sigma. On per-channel normalized latents (Qwen/Krea 16 ch, MOSS 128 ch) z x (1 + 0.2n) is noise scaled by |z|, not a brightness change as on SD latents. Elementwise, so it commutes with 2x2 packing and works on cached latents for every family. Run as a separate backward like `_dop_pass` (stop-grad on the clean prediction) to avoid two graphs on 16 GB (keep blocks was calibrated for one); block swap doubles H2D for the pass; compile sees identical shapes. Draw eps_m in the main thread from the run's generator for data_parallel/model_split determinism. Per application about +9 s on Krea 2 nvfp4, +5 s on MOSS int8, +2.5 s on Qwen 2.1 int8.

**Implementation path.** config.py next to `dop_weight`: `latent_consistency_weight` (0 = off; try 0.5-1.0), `latent_consistency_std` (0.1-0.2), `latent_consistency_every` (2). `_lc_job` modeled on `_dop_job` (trainer.py:2444), `_lc_pass` modeled on `_dop_pass` (trainer.py:2458): clean prediction under no_grad, z'_t = (1-s)(z x eps_m) + s noise, LoRA forward with grad, d = (z'_t - z_t) + (1-s)(v' - v_clean.detach()), optionally x regions.weight, backward weight x mean(d^2)/n_total. Log `lc_values` like `dop_values`.

**How to test.** Unit test first: the eps-space term is zero when eps_m = 1 and does not flip the clean-loss gradient sign. Then race on Qwen 2.1 int8: A best character recipe (DOP on, region weights on); B + weight 0.5, std 0.2, every 2; C + weight 1.0, std 0.1, every 2; 3 characters, 2 seeds, 2000 steps at 0.5 MP. Probe face and body at equal wall-clock and equal steps, plus DINOv2 background-crop similarity of probe renders to the nearest training image. Adopt only if B or C beats A on face by >= +0.01 on 2 of 3 characters with no body drop over 0.01 at equal wall-clock; drop if the face score falls by more than the probe noise floor (about 0.01) on any seed or best checkpoints come > 20% later. Only then MOSS int8 and Krea 2 nvfp4 with every 4.

## Likeness from data, teachers, references and Turbo transfer

This section covers likeness levers that sit outside the loss on the dataset's own pixels. They come in four kinds: what the trained LoRA is sampled on (Turbo, distilled bases, training adapters, guidance loss, probe strength), where identity sits in the network (block profiler, text-fusion boost), teachers and references (privileged-teacher matching, self-reference tokens, personalised distillation, RefMods), and extra data (edit-model views, regularisation sets, near-duplicate weights, composites). Overall verdict: the cheap wins are about measuring and transferring correctly, not about new identity signals. Fix the distilled-base sampling bug, race the Krea 2 Turbo + Ostris adapter and the Qwen 2.1 `lora_blocks 10-14` preset, and add a probe-strength sweep. The guidance loss is required groundwork for MiniMax H3 and distilled Klein. The teacher and reference ideas are unmeasured for single-person LoRAs anywhere, and none of the data-side items has a measured face-identity gain on a DiT.

### Training adapters for distilled bases (Krea 2 Turbo, Qwen 2.1, H3)

**Race it. Expected gain: unmeasured likeness change; probes become true Turbo renders of the trained weights, and H3 becomes trainable at all. Confidence: medium (0.65). Cost: ~0.5-1 day (guards, metadata, race key).**

**What it is.** A training adapter is a frozen helper LoRA that "undistills" a distilled base. It stays on during training and is switched off for sampling, so the user LoRA learns only the concept delta while sampling runs on the real distilled model. Ostris made one for Krea 2 Turbo: a LoRA trained at lr 1e-5 on thousands of Turbo renders. ai-toolkit merges it at 1.0 for training and inverts it (multiplier -1) for sampling. For MiniMax H3, whose released checkpoints are all CFG-distilled, an adapter (or the guidance loss below) is mandatory.

**Evidence.**
- [ostris/krea2_turbo_training_adapter](https://huggingface.co/ostris/krea2_turbo_training_adapter): LoRA at lr 1e-5 on thousands of Turbo renders; it "slows the breakdown" of distillation; long runs still produce artifacts once it is removed. "Could yield better results than training on the raw model" is a hope; the card has no benchmarks.
- [ai-toolkit krea2.py](https://raw.githubusercontent.com/ostris/ai-toolkit/main/extensions_built_in/diffusion_models/krea2/krea2.py): merge_in at 1.0, then multiplier -1.0 / `invert_assistant_lora` for sampling.
- [HF Krea-2-Turbo discussion #10](https://huggingface.co/krea/Krea-2-Turbo/discussions/10): RTX 5070 12 GB, Windows 11, 36 images at 1024, rank 32, Turbo + adapter, 25.4 -> 22.3 s/step after `lokr_full_rank false`, 18h45m for 3000 steps; body "very good", face "acceptable" with some drift. Krea staff asked why he did not train Raw. There is no head-to-head comparison anywhere.
- [bongobongo2020 krea2-character-lora-trainer](https://github.com/bongobongo2020/krea2-character-lora-trainer): Turbo + adapter in training, removed at inference; strength 0.8, CFG 1, ER-SDE, 8 steps; drop to ~0.7 if overcooked.
- [musubi krea2.md](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md): the train-Raw route needs `--turbo_dit` (~1x DiT in CPU RAM, no block swap), or a rank-extracted `--turbo_lora`; merging into convrot int8 is rejected.
- [SimpleTuner Qwen-Image-2.1-training-assistant-v2](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v2): rank 32 on to_q/k/v/out, 1000 updates at batch 1, lr 1e-4, one H100, 338 synthetic + 331 CC12M + 331 e621 images, 512-2048 px; qualitative results only. [v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v3) (8x H100, 30k steps) says the downstream benefit "remains to be evaluated"; v1's card reports no downstream benefit in its paired test. [Fizgig v6.5.0](https://github.com/shootthesound/Fizgig/releases/tag/v6.5.0) says only "higher resolution, much sharper".
- The Qwen 2.1 figures in RulzTrainer's own config.py:55-57 and README (likeness 77 with Fizgig's adapter, 56 without, 76 with SimpleTuner's) have no locatable public source. Treat them as unverified.
- H3: [circlestone image adapter](https://huggingface.co/circlestone-labs/MiniMax-H3-Image-Training-Adapter) (620 MB, ~10k steps on ~10k images, "significantly better" than Ostris's for images, do not stack with guidance losses); [Ostris H3 adapters](https://huggingface.co/api/models/ostris/minimax_h3_training_adapter/tree/main) (v1 155 MB, v2/v3 310 MB, ref2va v1-v3, FastH3). [Fizgig MINIMAX_H3.md](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/MINIMAX_H3.md): with the Ostris adapter, 50% likeness seven epochs sooner and a higher peak; one anecdotal A/B with no GPU or dataset named. [musubi minimax_h3_1f.md](https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/docs/minimax_h3_1f.md): one-frame training without a guard degrades structure within ~50 steps.
- Transfer: all of these are LoRA on a frozen DiT at batch 1 on consumer cards, so the setting matches. The SimpleTuner adapters were built on H100s.

**RulzTrainer today.** The mechanism already exists and is the right design. `training_adapter` (config.py:324-326) takes 'auto', '' or any path on any family. `lora.attach_training_adapter` (lora.py:902-941) renames ComfyUI keys through `arch.lora_module_name`, so the Krea rename works (archs.py:362-367). It is stored as non-persistent buffers on LoRALinear (lora.py:238-241), added only while `extras_enabled` is False (lora.py:396-398), attached to every data_parallel replica (trainer.py:775, 792) and never saved. That is mathematically ai-toolkit's "merge 1.0, invert at sample", but without a merge, so int8 ConvRot and NVFP4 bases just work. A Turbo base already loads (krea2/loading.py:104 `is_distilled`, single_file.py:159 "turbo" in the file name) and samples with fixed mu 1.15 (krea2/sampler.py:37). The deltas:
- (a) `ARCH_TRAINING_ADAPTER` (config.py:58-61) has only qwen_image21, so 'auto' on krea2 downloads nothing.
- (b) A latent bug. On a distilled base, `turbo_lora: auto` still attaches Comfy-Org's Raw-to-Turbo rank-64 LoRA for samples and probes (comfy_files.py:42-43, `_attach_sampling_loras` trainer.py:2148-2157). It stacks on an already-distilled model and corrupts probe scores. Nothing keys off `model.is_distilled`.
- (c) Whether the Ostris file's keys and alpha parse to the right scale is unverified. The "matched N modules" check catches a total miss, not a scale error.
- (d) The adapter name is not written into the saved LoRA's metadata (save_lora calls at trainer.py:1381/1438/1956).
- (e) A race shares the training adapter across its configs (README.md:1244), so adapter on vs off cannot be raced inside one run.
- The scout's "drops a second 12B model" saving does not apply. RulzTrainer samples Raw with the rank-64 turbo LoRA by default, so the memory saving is that LoRA, well under 1 GB.

**Fit and risks.** W8A8 / NVFP4: no merge, no requantize; the branch's bf16 matmuls at rank 16-64 cost a few percent at most. Block swap and model_split: the buffers move with their module (lora.py:239; prefetch reset at lora.py:952-954). torch.compile: `extras_enabled` is a Python bool guard, as it is for Qwen 2.1 today. Exports are untouched. Windows: nothing new. Risk: distillation erodes on long or high-lr runs; the user's 1000-3000 steps are about what Ostris calls short. For H3, region_prior and DOP passes should run adapter-on so their base predictions are de-distilled too; that needs checking. Never stack with the guidance loss.

**Implementation path.**
1. config.py: per-family, per-variant `ARCH_TRAINING_ADAPTER` entries: krea2 distilled -> ostris/krea2_turbo_training_adapter, Raw -> none; H3 later -> circlestone for images, Ostris v3 matched to fl2va/ref2va.
2. comfy_files.planned_downloads (~lines 42-61): choose the entry from model_path or a "turbo" file name.
3. trainer.py: on `model.is_distilled`, resolve `turbo_lora: auto` to '', treat it like a turbo LoRA in `_check_sample_settings` (trainer.py:2159-2175), and default sample/probe steps to 8 at CFG 1. This guard is the "Do it" part and should land first.
4. Write `rulz_training_adapter` (file and strength) into save_lora metadata.
5. A unit test like tests/test_qwen21.py:209 for the Ostris file: matched-module count and scale = alpha/rank.
6. race.py: allow `training_adapter` and `training_adapter_strength` as per-config keys.
7. Leave the scout's "build our own assistant adapter" job (SimpleTuner recipe via selftrain.py's renderer, q/k/v/o, rank 32, lr 1e-4, 1-2k steps; ~2-3 h side job per family and resolution) until the races show adapters matter.

**How to test.**
- Krea 2: two runs, same dataset and seed, 0.5-1 MP, rank 16, 3000 steps, nvfp4 on the 5060 Ti. A: Raw with `turbo_lora: auto`. B: `model_path: krea/Krea-2-Turbo`, `training_adapter: <ostris file>`, `turbo_lora: ''` (essential). Optional C: Turbo, no adapter, to confirm the adapter prevents breakdown. Probe every 250 steps, ArcFace face + DINOv2 body, all on Turbo at 8 steps, CFG 1 (for A, also probe the final checkpoints with `sample_model_path: krea/Krea-2-Turbo`). Watch adapter-off renders for broken hands and lines, and for body score falling while face rises. Stop at 3000 steps or 3 probes without improvement. B wins only if its peak face likeness is 2+ points higher, or it reaches 90% of peak 20% sooner, on 2 datasets.
- Qwen 2.1: `training_adapter` Fizgig vs '' vs Fizgig at strength 0.5, 0.5 MP, rank 8, lr 2e-4. Compare peak ArcFace and the step to 50% of peak. This re-measures the unsourced 77-vs-56 claim before any own-adapter work.
- H3, once added: circlestone vs Ostris v3 vs none; "none" should collapse within about 50-200 steps if the claims hold.

### Guidance-loss target for CFG-distilled bases (Qwen 2.1, future H3 and distilled Klein)

**Race it (needs code). Expected gain: no likeness gain expected; at best it matches the training adapter on Qwen 2.1 without a third-party file, and it is a prerequisite for H3 and distilled Klein. Confidence: medium (0.6). Cost: ~1 day.**

**What it is.** Replace the flow target `noise - x0` with `u + g*(target - u)`, where `u` is a no-grad prediction of the same model, LoRA on, on the same noised input under an empty or single-space prompt. The LoRA then learns already-guided velocities, so a CFG-distilled base keeps sampling correctly without CFG. musubi skips the extra forward below sigma 0.15, where the guidance contribution collapses. SimpleTuner's softer "drift" variant adds `w*MSE(pred, adapter-off conditional pred)` at w 0.25-0.5 instead.

**Evidence.**
- [musubi minimax_h3_advanced.md](https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/docs/minimax_h3_advanced.md) and [minimax_h3.md](https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/docs/minimax_h3.md): `target = uncond + scale*(velocity - uncond)`, space-token probe, LoRA active, scale 3-4 ("4 more reliable for longer runs"), sigma_min 0.15, about +50% step time ungated, works on INT8 bases; "plain flow training is therefore not offered as a recipe". [minimax_h3_1f.md](https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/docs/minimax_h3_1f.md): drift within ~50 steps without a guard.
- [diffusion-pipe supported_models.md](https://raw.githubusercontent.com/tdrussell/diffusion-pipe/main/docs/supported_models.md): "Qwen-Image-2.1 is a CFG-distilled model"; training without a de-distillation adapter breaks the distillation; H3 uses cfg = 4, "either this or a training adapter (not both)". [minimax_h3_notes.md](https://raw.githubusercontent.com/tdrussell/diffusion-pipe/main/docs/minimax_h3_notes.md): "any standard training gradually undistills the model".
- [ai-toolkit config_modules.py](https://raw.githubusercontent.com/ostris/ai-toolkit/main/toolkit/config_modules.py) and [SDTrainer.py](https://raw.githubusercontent.com/ostris/ai-toolkit/main/extensions_built_in/sd_trainer/SDTrainer.py): `guidance_loss_target` 3.0, `guidance_loss_schedule: sigma` (g_eff = 1 + (g-1)*sigma, for guidance-distilled models without a guidance embedding), optional CFG-Zero rescale.
- [SimpleTuner MINIMAX_H3_DRIFT.md](https://raw.githubusercontent.com/bghira/SimpleTuner/main/documentation/distillation/MINIMAX_H3_DRIFT.md): drift MSE, weight 0.25-0.5, no benchmarks.
- No source measures likeness with or against this loss. Its value is preserving distillation, not adding likeness.
- Refuted member: the idea of using it on Krea 2 Raw so Raw-trained LoRAs land at strength 1.0 on Turbo. The cited Krea anecdote (5060 Ti, "90% likeness by 1750 steps", [discussion #10](https://huggingface.co/krea/Krea-2-Turbo/discussions/10)) used ai-toolkit's "Differential Guidance", which is nested inside the guidance-loss branch and on its own only rescales the gradient. Raw is undistilled, so baking guidance into a Raw LoRA risks guiding twice on Turbo.

**RulzTrainer today.** The flow target is always `target = noise - latents` (trainer.py:966). The only extra forward is region_prior, which runs with the LoRA off on the conditional embeds (trainer.py:984-993), so it cannot be reused for this pass, which needs the LoRA on and the empty embeds. `empty_text_path` exists (dataset.py:265-267) but nothing calls it; `arch.empty_prompt` is defined per family (archs.py:45, 252). Qwen 2.1 is protected only by the training adapter (config.py:54-61). Scope: Qwen 2.1 (sampled at CFG 1, config.py:32-34) is the right target. Krea 2 Raw (CFG 5.5) and MOSS (CFG 3) should keep it off.

**Fit and risks.** W8A8 / NVFP4: just another QuantLinear forward; musubi reports it on INT8. torch.compile: the empty prompt has a different text length, so expect one recompile per bucket unless the empty embeds are padded to a fixed length. Keep blocks: a no-grad pass stores no activations, but leave ~0.3-0.5 GB headroom for transient buffers. Block swap: every swapped block crosses H2D twice per step, so the cost is more than +25-50% there. data_parallel threads: each replica runs its own pass in `work()`, no NCCL. Exports unchanged. Too high g over-sharpens, too low undistills. It must be mutually exclusive with a training adapter.

**Implementation path.**
1. TrainConfig: `guidance_loss_scale` (0 = off), `guidance_loss_sigma_min` (0.15), `guidance_loss_every` (1, the dop_every pattern), `guidance_loss_schedule` ('constant' | 'sigma').
2. Validate exclusivity with a non-empty `training_adapter`; warn on families whose full-quality sampling uses CFG > 1 (MOSS, FIBO, Krea Raw).
3. Always write `empty_text_path(cfg)` at caching time; load it once per replica device.
4. In `work()` after line 966: for items above sigma_min, run `rmodel(noisy_in, sigma*1000, empty_embeds, empty_mask)` under no_grad with the LoRA on, then `target = u + g_eff*(target - u)`. Region weights, loss_weighting and region_prior stay applied afterwards. Log the gap RMS `||target - u||` as musubi does.
5. Expose for qwen_image21 as an alternative to `training_adapter: auto`, make it the default recipe for future minimax_h3 and distilled flux3-klein archs, and add it as a race knob. A 'drift' mode can come later.

**How to test.** Qwen 2.1 int8, usual character set, 0.5 MP, batch 1, rank 16, 2000 steps, same seed. Arms: A `training_adapter: auto`; B no adapter, scale 4, sigma_min 0.15; C no adapter, scale 3, sigma schedule; D no adapter, plain target. Probes every 250 steps through the Turbo LoRA at 8 steps, CFG 1: ArcFace face, DINOv2 body, a structure check (texture collapse, wobble) and s/step. B or C wins only if best face likeness is within 1 point of A or above, with no texture collapse by step 2000 and at most 1.5x step time. Otherwise keep the adapter as the Qwen default and ship the loss only as the H3/Klein recipe. Optional Krea sanity run (Raw, scale 3 vs plain, Turbo probes at 0.8/1.0/1.2): expect oversaturation and drop Krea exposure if it shows.

### Block identity map: profiler, Qwen 2.1 `lora_blocks 10-14`, per-block export

**Race it. Expected gain: about 1.5x per step on Qwen 2.1 at 0.25 MP per Fizgig (roughly 2.5 -> 1.6-1.9 s/step at 0.5 MP, unmeasured); likeness unmeasured. Confidence: medium (0.6). Cost: race needs no code; profiler ~1 day.**

**What it is.** Fizgig profiled Qwen 2.1 character LoRAs by switching off block groups of a finished LoRA under fixed seeds, and found almost all likeness in blocks 10-14. Its Fast Identity Mode trains only those blocks; the earlier blocks skip backward, which gives the speedup. The transferable parts are the ablation profiler, a race of the 10-14 preset, and an "identity x1 / look x0.5" per-block export.

**Evidence.**
- [Fizgig v6.8.3](https://github.com/shootthesound/Fizgig/releases/tag/v6.8.3): identity in blocks 10-14, switching them off removes almost all likeness; Fast Identity Mode about 1.5x per step; rank 8 preset at 0.25 MP, "the resolution it was tested at"; profiler modes Weights-only / Quick (block groups, one seed) / Thorough (two seeds plus leave-one-block-out); Repair Studio presets Identity only / Look only / Identity x0.5. No likeness numbers.
- [Fizgig issue #265](https://github.com/shootthesound/Fizgig/issues/265): a rank-8 Fast Identity LoRA (30 patches, blocks 10-14) showed speckled skin in ComfyUI while previews were clean; closed 29 Oct without a confirmed cause.
- [danmo-make PR #76](https://github.com/danmo-ai/danmo-make/pull/76): for the older 20B Qwen-Image the likeness preset moved from last 16 blocks / rank 16 / 512 px to all 60 / rank 32 / 1024 px. Block maps are model-specific.
- MOSS: [MATLOWAI DeBand](https://huggingface.co/MATLOWAI/MOSS-Image-SFT-DeBand) puts SFT banding in blocks 21-41 (main fix 37-41, extra 32-36; banding score SFT 2.04 vs Base 1.12) and [Fizgig SFT-Look LoRA](https://huggingface.co/ShootTheSound/Fizgig-MOSS-Image-SFT-Look-LoRA) needed rank 896 for 21-41 vs 128 for 0-20. Both measure SFT colour drift, not identity, and the MOSS default (21-34) already excludes 35-41.
- Does not transfer directly: Fizgig tested at 0.25 MP and rank 8; the user trains at 0.5-1 MP and rank 8-16.

**RulzTrainer today.** `lora_blocks` parses ranges (config.py:239, lora.py:406 `parse_blocks`), blocks before the first trained block get no backward (README.md:236), the race varies `lora_blocks` per racer (race.py:68), and the WebUI has a per-block grid (webui.py:5668-5726). The README cites Fizgig's 10-14 finding (README.md:1672), but the wizard never sets it. MOSS defaults to 21-34 (configs/default.yaml:45, identity.yaml:45). Missing: any tool that measures which blocks of a finished LoRA carry likeness, and a per-block rescaled export. Reusable: each LoRALinear has its own `multiplier`, and 0 short-circuits the forward (lora.py:232, 382); `_probe_with(slots, step, tag)` (trainer.py:3264) renders and scores with a tag.

**Fit and risks.** Only the LoRA branch is switched, so W8A8, NVFP4, block swap and model_split are untouched. data_parallel: set the multiplier on every replica and on `sample_injected`. torch.compile: flipping a float multiplier inside compiled blocks can recompile once per block code object, so run the ablation at end of training or on the uncompiled sample model. Qwen caveat: with `text_in` or `train_text_embedder` on, gradients flow back through blocks 0-9 and the speedup is gone. Export scaling of lora_up per block is exact for LoRA and LoKr, not for OFT or DoRA. Issue #265 speckle is an unexplained caution against the reduced-block mode.

**Implementation path.**
1. `rulztrainer/blockprofile.py`: `profile_blocks(trainer, injected, groups, seeds)` maps module names to block indices (the `lora_module_names` regex), zeroes one group at a time, calls `_probe_with(..., tag=f'ablate_{g}')`, and writes face/body drops to `probes/ablation.json` and LoRA metadata (`rulz_block_profile`).
2. Options `block_profile: off|end|best` and `block_profile_groups` (Qwen 0-4, 5-9, 10-14, 15-19, 20-25, 26-31; MOSS 0-10, 11-20, 21-27, 28-34, 35-41).
3. `lora.scale_blocks(path, {range: factor})` writing `_identity.safetensors` (identity groups x1, rest x0.5), LoRA/LoKr only.
4. Wizard: a "fast identity: lora_blocks 10-14" race candidate for Qwen 2.1 character runs, not a default. Update README and wiki.

**How to test.** Qwen 2.1 race, 0.5 MP, rank 8 at 2e-4, `training_adapter: auto`, same seed, 2000 steps, probes every 200 steps on 2 seeds. Racers: A `lora_blocks: ''`, B `'10-14'`, C `'8-16'`. Metrics: best combined face+body likeness, wall time to 95% of A's best, and a close-up skin check at full sampling (25 steps, CFG 1, no turbo) for speckle. Adopt B or C as the wizard preset only if within 2 points of A, at least 1.25x faster per step, and speckle-free. Profiler validation: on A's best LoRA, the top group's ablation should drop face likeness at least twice as much as any other group, and the identity-only export should keep at least 90% of the face score. MOSS: profile a 21-34 LoRA, then race 21-34 vs 21-31 vs the profiler's top two groups under the same rule. Cost of one profile: about 24 renders (6 groups x 2 probe kinds x 2 seeds), a few minutes.

### Probe-strength sweep and Raw-vs-Turbo transfer gap

**Do it (the diagnostic part). Expected gain: no likeness gain by itself; a recommended inference strength, a transfer-gap alarm, and possibly a different best checkpoint. Confidence: medium (0.45). Cost: ~0.5 day; 10-30 s of extra probe renders per checkpoint.**

**What it is.** LoRAs trained on Raw/Base are used on a distilled target, and the right strength there is often below 1.0. Probe the trained LoRA on the Turbo target at a few strengths and once on the raw model, log the gap, and store the best strength in the LoRA. The scout's three heavier members (CASA spectral repair, hybrid first-step sampling, SISO last-step optimisation) are not worth building yet.

**Evidence.**
- [BFL Klein docs](https://docs.bfl.ml/flux_3/flux3_image_klein): train on Base (50 steps, CFG 3.5); on Distilled use 16 steps, CFG 1, LoRA scale 0.75 (0.5-1.0 works); 0.75 "matched the Base + LoRA result most closely".
- [SimpleTuner Klein chapter 06](https://huggingface.co/SimpleTuner/flux-3-klein-image-LoRA-experiments/raw/main/chapters/06-base-to-distilled.md): all 344 tensors load, but concept activation is less reliable after transfer and seed-dependent (character reduced to a pouch or an ordinary person); at 0.5 outputs largely keep people.
- [BFL FLUX.2 klein LoRA blog](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora): train base-4B, infer distilled at 4 steps, "usually gives even better results" (authors' own tests).
- Community: Krea says Raw-trained LoRAs "express strongly on Turbo"; a community Krea 2 character card recommends 0.8 on Turbo. README.md:1410 already warns the MOSS DMD turbo LoRA can understate likeness.
- Members not to build now: [CASA](https://arxiv.org/html/2605.01929) restores Wan2.1 style LoRAs on distilled video students (Steamboat-Willie 1.3B: quality 1.27 -> 1.58, CSD 78.35 -> 81.49, ~5 s per LoRA on a 4090), but two of four 14B cells get worse (e.g. CSD 63.47 -> 62.56) and there are no image or identity experiments. [Hybrid first step](https://arxiv.org/html/2503.10637v2) (SDXL: FID 15.52 -> 10.79, DreamSim diversity 0.264 -> 0.350) is a UNet population metric and would confound fixed-seed probe curves. [SISO](https://arxiv.org/html/2503.16025v1) (DINO 0.33 vs 0.51 on FLUX Schnell) is per-image test-time optimisation and overlaps idloss.py / reward.py.

**RulzTrainer today.** Probes already render on the distilled target: the family turbo LoRA as a sampling-only extra (config.py:43-51 ARCH_TURBO, config.py:319-321, trainer.py:2149-2157), or a full Turbo via `sample_model_path` (trainer.py:2177-2195). The trained LoRA's probe strength is hard-wired to 1.0. A per-module multiplier with ComfyUI strength semantics exists (lora.py:232, 382-393, `set_multiplier` at lora.py:579). No transfer metric, no recommended-strength metadata.

**Fit and risks.** The multiplier scales only the LoRA branch, so it works over int8_convrot and NVFP4 QuantLinear. Under torch.compile a Python float may recompile once per distinct value; a 0-dim tensor multiplier avoids that. Run it at checkpoint probes only. If the race or best checkpoint ranks by max over strengths, the chosen checkpoint can change; that is intended but should be logged.

**Implementation path.** Add `probe_turbo_scales` (default [] or [0.75, 1.0]) and `probe_raw_check` next to `probe_*` (config.py:426-436). In `_run_probe` / `_probe_with` (trainer.py:3264, 3319), at checkpoint probes only, loop `lora_mod.set_multiplier(slot_injected, s)`, render, score, restore 1.0. For the raw check render once with the turbo extras off at `arch.default_sample_steps` / `default_sample_cfg`. Write `scale_s` and `transfer_gap = raw - max(turbo_s)` to likeness.jsonl and `recommended_strength` to the export metadata. When Klein is added, default its probe strength to 0.75. CASA (`rulztrainer/casa.py`, reusing `finetune.lowrank_factors`, finetune.py:388-396) only if the gap proves real.

**How to test.** Re-probe saved checkpoints of one finished run per family (krea2-AZav-rulz offline, one MOSS int8, one Qwen 2.1) at strengths {0.6, 0.75, 1.0} and raw full quality (20-28 steps, CFG 3), `probe_seeds >= 4`, same seeds. Report transfer_gap and whether the argmax checkpoint moves. If the gap is under ~0.02 ArcFace and the best strength is 1.0 on all three, keep only the diagnostic. If the gap is 0.03 or more (MOSS DMD is the likely case), run CASA offline and accept it only if it closes half the gap on 8 seeds without lowering raw likeness.

### Text-fusion boost on Krea 2 (post-hoc group strength)

**Race it. Expected gain: 0 to a few ArcFace points on Krea 2 (unmeasured; negative on overtrained LoRAs). Confidence: medium (0.45). Cost: ~0.5 day; 2-3 probe passes per LoRA.**

**What it is.** Fizgig's Repair Studio multiplies the LoRA deltas of Krea 2's four text-fusion blocks (2 layerwise + 2 refiner, which fuse the 12 tapped Qwen3-VL layers) by 2 or 3 while the 28 DiT blocks stay at 1.0, then bakes that into the file. The reading is that the text-fusion delta is under-trained at a shared lr and carries identity worth amplifying.

**Evidence.**
- [Fizgig KREA2.md](https://github.com/shootthesound/Fizgig/blob/master/docs/KREA2.md): "measured across several LoRAs, x3 lifted the detail meter and likeness on every one with the composition unchanged"; "an overtrained LoRA is the one case it works against". No numbers, no LoRA count, Fizgig's own meter. No independent report found.
- RulzTrainer's README.md:1612-1613 credits Fizgig with "lifted likeness 2-7 points"; the current KREA2.md has no such figure. Re-check or remove that line.
- [Krea 2 mmdit.py](https://raw.githubusercontent.com/krea-ai/krea-2/main/mmdit.py): TextFusionTransformer layout. [TheDivergentAI krea2-turbo-distill-lora](https://huggingface.co/TheDivergentAI/krea2-turbo-distill-lora): txtmlp/tmlp/tproj are the worst low-rank fits of the Turbo-Raw shift (residual 46.1% at r64).
- Member 150 (automatic multi-group coordinate descent) cites MOSS block 22 and H3 blocks 0-5, but those are training-target choices, not post-hoc rescales; about 6 groups x 5 scales of probe passes would overfit noisy probe seeds.

**RulzTrainer today.** Krea 2 trains `text_fusion` by default (config.py:29; README.md:1607-1616); the target covers layerwise + projector + refiner (archs.py:432-434). txtmlp/tmlp/tproj sit in `embed`, off by default, so the scout's "freeze projections for Turbo" is already the case. Per-type adaptive lr exists via `automagic_groups: by_type` (archs.py:121-133, 356; trainer.py:2054-2056). A probe-driven global strength search already exists in the seed soup (`soup_strengths` 1.0/1.15/1.3, config.py:422-424, trainer.py:3127-3210). Missing: a per-group multiplier at probe time and a per-group bake at save; `set_multiplier` sets all modules at once.

**Fit and risks.** text_fusion is outside `compile_blocks`, so changing its multiplier triggers no recompile. The QuantLinear base is untouched. Baking by scaling lora_up / lora_B is exact for LoRA and LoKr (scale w2); skip OFT and DoRA. Risk: best-checkpoint selection may already pick a checkpoint where the boost hurts. Krea 2 only; other families have a single-Linear text projection.

**Implementation path.** Option `group_strengths` (e.g. `text_fusion: 1, 2, 3`, Krea 2 only). Add `set_multiplier(injected, value, prefix=None)` in lora.py. Reuse the soup probe loop (trainer.py:3127-3210) after training or in a steps-0 rescore on a given file. Keep the winner only if it beats x1 by more than the probe seed spread; bake it into lora_up in `export_state_dict` / `save_lora` and record `group_strengths` in metadata. Groups per arch in archs.py (Krea: text_fusion, attn_gate, embed).

**How to test.** 3-5 finished Krea 2 character `_best` LoRAs plus one deliberately overtrained last step. Probe text_fusion x1/x2/x3 on the same prompts with `probe_seeds` 8+. Primary metric ArcFace; also DINO body and a composition guard (DINO of the x3 render vs the x1 render). Accept if x2 or x3 beats x1 by at least 1 ArcFace point (above seed std) on most LoRAs without the guard dropping. Then compare a baked x3 file to x1 in ComfyUI by eye. If it holds, race `automagic_groups: by_type` vs single to see whether a stronger text-side lr makes the boost unnecessary.

### Privileged-teacher matching from the reference pathway

**Prototype. Expected gain: unmeasured on any image model; identity supervision at every sigma without ArcFace or a VAE decode. Confidence: medium (0.4). Cost: several days (Qwen 2.1 edit-path port) plus about +60-120% step time (verifier's estimate).**

**What it is.** A text-only student (LoRA on, trigger caption) is trained to match the frozen base's own prediction made with privileged conditioning: other photos of the same subject fed through the model's reference path. The teacher already "knows" the person when shown them, so the target needs no face network and works at high sigma, where the decode-based identity loss cannot. musubi splits the loss into magnitude, direction and per-channel DC terms so the dataset palette is not learned as a style shift.

**Evidence.**
- [musubi minimax_h3_advanced.md](https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/docs/minimax_h3_advanced.md): one extra no-grad forward of the same transformer with the LoRA off; loss `(||p||-||t||)^2 + 2||p||||t||(1-cos)` with `||p||` detached (equal to MSE at default weights); mag_weight 0.25-0.5, dc_weight 0.3; about +50% step time; mutually exclusive with the guidance loss. subject_ref recipe "validated on a 20-image character set, one-frame, rank 16": lr 3e-4, warmup 50, ~500 steps (1e-3 random-walks, 1e-4 settles weak); residual plateaus around 300 steps and the best checkpoints sit at or just after the plateau.
- [musubi minimax_h3.md](https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/docs/minimax_h3.md): references must be other pictures of the same subject; keep teacher conditioning to sigma_max 1.0 "because identity is decided at the top of the range"; sigma_min 0.15. [minimax_h3_1f.md](https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/docs/minimax_h3_1f.md): one of three mandatory loss methods for one-frame training; the whole mode is experimental.
- [Flux 3 Klein card](https://huggingface.co/black-forest-labs/flux-3-image-klein): up to 10 reference images, LoRA training on the separate Base. [SimpleTuner FLUX3_IMAGE.md](https://raw.githubusercontent.com/bghira/SimpleTuner/main/documentation/quickstart/FLUX3_IMAGE.md): up to ten reference datasets per sample.
- Corrections: member 142's `sigma_max 0.75` is wrong; musubi warns 0.75 stops identity learning for subject_ref. Its "much lower teaching-band error" belongs to the `ref` teacher (which sees the clip itself), not subject_ref. "Krea 2 Edit" is not an official model, only [Ostris's community Krea2OstrisEdit](https://huggingface.co/ostris/Krea2OstrisEdit).
- No likeness numbers anywhere; the only validation is one 20-image one-frame set on a video+audio model. The ceiling is the base's own reference identity, unmeasured for Qwen 2.1, Klein or Krea community LoRAs.

**RulzTrainer today.** Nothing exists (no teacher or subject_ref code). The Qwen 2.1 edit path is not ported (qwen21/model.py:25-26; README.md:1655-1656). The direct identity loss runs only at sigma <= 0.35 (README.md:784-791, config.py:388) and trains on the probe's own ArcFace net, so it can game the probe; a velocity-space teacher keeps the probes independent. Reusable: the LoRA-off prior forward (trainer.py:984-993 via `set_multiplier(rinj, 0.0)`), the DOP pass with other embeddings (trainer.py:1028-1030), face/bust crops from crop_augment (config.py:377-384).

**Fit and risks.** W8A8/NVFP4: no-grad forward through the same QuantLinear wrappers; the Qwen training adapter stays on. torch.compile: longer sequences and the img_mask give new shapes, so keep reference count and reference bucket fixed. Keep blocks: budget the teacher's no-grad peak at about 2-3x the sequence length. Block swap: each swapped block moves twice per step. model_split/data_parallel: fine, the teacher runs on the same replica. Qwen 2.1 int8 is about 7.3 GB, so a no-grad forward over 0.5 MP plus 1-2 references fits 16 GB. Risks: copying reference lighting or pose; reference-mode artefacts. MOSS and FIBO have no reference path. H3 (INT8 DiT ~21 GB plus a 14.6-25 GB text encoder) is unrealistic on 16 GB as a first target.

**Implementation path.**
1. Port the Qwen 2.1 reference path into qwen21/model.py (condition latents prepended, img_mask, img_shapes, RoPE for condition images, t=0 modulation under causal_condition), verified against diffusers QwenImage21Pipeline as tests/test_krea2.py does for Krea.
2. qwen21/loading.py: a teacher caption with k reference images in the `<image1><|vision_start|><|image_pad|><|vision_end|>` template, cached as a separate .rulz_cache entry.
3. Options: `teacher_match`, `teacher_match_refs` (1-2), `teacher_match_sigma_max` 1.0, `teacher_match_sigma_min` 0.15, `teacher_match_mag_weight` 0.5, `teacher_match_dc_weight` 0.3, `teacher_match_replace_target`.
4. trainer.py next to region_prior: no-grad teacher forward with the LoRA off and references; below sigma_min run without references as the preservation anchor; apply the split loss, optionally times the region map.
5. Add the teacher's peak to checkpointing.py's calibration margin. Design the future Flux 3 Klein arch with this hook from the start; defer Krea 2.

**How to test.** First a teacher-only sanity check: the frozen Qwen 2.1 in reference mode must score clearly higher ArcFace to the references than text-only base renders; if not, stop. Then Qwen 2.1 int8, 0.5 MP, rank 16, same set and seed: A flow matching with region weights; B A + direct identity loss (facenet backend, so the ArcFace probe stays independent); C pure teacher_match (1 ref, sigma 0.15-1.0, mag 0.5, dc 0.3, lr 3e-4, 50 warmup); D teacher_match added at weight 0.5. Probe every 100 steps, trigger-only and varied scenes, plus pose/lighting variance. Run 1500 steps or to patience. C or D wins only if best face beats A by 0.02+ and matches B, within 1.6x A's wall time, with no body drop.

### Edit-model synthetic views from the real photos

**Race it (no code needed for the first test). Expected gain: at best a few points of held-out ArcFace on 3-6 image sets; nothing or negative on 20+ image sets. Confidence: low (0.35). Cost: zero code for the manual test; ~1-2 days for synthetic flag, weight and auto gate.**

**What it is.** Render new views of the subject (azimuth, elevation, lighting) from the real photos with an edit or multi-reference model, gate each render by identity against the real set, cap the synthetic share, and train on them as plain images. Unlike selftrain this works at step 0 and for 1-3 image datasets.

**Evidence.**
- [SynCD](https://arxiv.org/html/2502.01720v2): Masked Shared Attention sets, filters aesthetic >= 6 and mean pairwise DINOv2 >= 0.7; adding SynCD to IP-Adapter Plus raises MDINOv2-I 0.744 -> 0.766 (0.777 with MSA). Objects and animals only, no people; only the 12B variant is FLUX.
- [UNO](https://arxiv.org/html/2504.02160v1): FLUX.1 diptychs, DINOv2 + VLM chain-of-thought filter keeps 35.43%; rank-512 LoRA on 8x A100; DreamBench DINO 0.760 vs DreamBooth 0.668. No face metric.
- [arXiv 2505.03557](https://arxiv.org/html/2505.03557v2): 10 subjects, 3-15 images each, SDXL (RealVisXL), one RTX 3090; balance is "crucial" (not "only helps when balanced"), no concept above 25% of the dataset; FaceDistance discard of the top 15% for n >= 8.
- Generators: [CharacterSheet LoRAs](https://huggingface.co/Alissonerdx/CharacterSheet/blob/main/README.md) (Klein 9B and Krea 2; warns face, anatomy and clothing may differ between views); [fal Qwen-Image-Edit-2511 Multiple-Angles](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA) (96 poses = 8 azimuths x 4 elevations x 3 distances, Apache-2.0, no identity claim).
- Does not transfer: encoder-scale gains (rank 512 on 8x A100, 95K samples) and object DINO gains as a proxy for face identity. Practitioner workflows ([weirdwonderfulai](https://weirdwonderfulai.art/comfyui/qwen-image-edit-can-create-character-consistent-lora-dataset/), [runcomfy](https://www.runcomfy.com/trainer/ai-toolkit/qwen-image-edit-2511-likeness-custom-lora)) are unquantified and report face drift.
- Refuted member 63: on-policy DMD distillation from Qwen-Image-Edit ([CollectionLoRA](https://arxiv.org/html/2605.25378v2), effect LoRAs, 8x H800, no identity metric). The teacher's 16ch VAE cannot score Qwen 2.1's 64ch, Krea's packed 64ch or MOSS's latents, and a 20B teacher plus a fake-score LoRA does not fit 16 GB.

**RulzTrainer today.** `selftrain` (selftrain.py, README.md:995-1024, config.py:413-420) already caps its own renders at `selftrain_share` 0.2 with `selftrain_weight` 0.5, so a share cap exists. The look consistency filter (lookfilter.py:1-25, README.md:1452) is the per-image ArcFace/DINO gate (fence median - 1.5 IQR, face floor 0.25). Multiple datasets with repeats and trigger exist (`DatasetSpec`, config.py:133-141). Missing: a per-dataset synthetic flag and loss weight, an automatic gate against the real images, set-level DINO consistency, a generator.

**Fit and risks.** Synthetic items are ordinary cached latents: no interaction with W8A8, NVFP4, compile, keep blocks, block swap, data_parallel or exports. Qwen-Image-Edit-2511 is 20B and needs int8/NVFP4 plus offload, roughly 1-3 min per image at 1 MP on 16 GB, and cannot share a card with training. Klein 5.6B int8 ConvRot would take under 9 s on a 4070-class card once supported. Risk: the edit model's look bleeds in; gated renders can raise probe scores without raising real likeness (same scorers), so judge on held-out real photos.

**Implementation path.** Skip DMD. (1) `DatasetSpec` gets `synthetic: bool` and `weight: float`; run option `synthetic_share_max` 0.25, enforced in dataset.py via `selftrain.cap_count`; per-item weight applied where the selftrain weight is applied. (2) At load, gate synthetic datasets with `lookfilter.score_images` against the real images using `lookfilter.fence`; failures to `excluded_by_look/`, scores to `look_scores.json`. (3) Optional set-level check: mean pairwise DINO of views from one source photo. (4) Generation as an offline `scripts/augment_views.py` (Qwen-Image-Edit-2511 + Multiple-Angles on the second GPU). (5) Klein multi-reference later.

**How to test.** Today, with no code: make views in ComfyUI, run the look consistency filter, add the folder as a second dataset with low repeats. Two sets: 5 photos without profiles, and 20 photos; hold out 3-4 real photos (profile, back-lit, full body) as face_refs/body_refs. A: real only. B: real + gated synthetic (8 azimuths x 2 elevations, share 0.25, weight 0.5 once weights exist). C: B ungated. Adopt B only if it beats A by 2+ points of held-out face likeness on the small set on two seeds and loses no more than 1 point on the 20-photo set. If C is not clearly worse than B, the gate adds nothing.

### Near-duplicate down-weighting

**Skip-ish (cheap race on burst-heavy sets). Expected gain: zero on curated sets; possibly less background/pose copying on burst-heavy sets; ArcFace neutral to slightly negative. Confidence: low (0.35). Cost: ~0.5 day.**

**What it is.** Cluster near-identical photos (phone bursts) and sample each cluster with weight 1/|cluster|, so repeats do not multiply a burst into a memorisation canary.

**Evidence.**
- [arXiv 2605.10439](https://arxiv.org/html/2605.10439v2): 50 exact-copy canaries repeated 50 times in ~800-1,000 image sets, UNet LoRA on SD1.5/SDXL; dedup lowers SSCD (CelebA-HQ SD1.5 0.93 -> 0.53, SDXL 0.84 -> 0.56; Pokemon 0.76 -> 0.65). The dedup is an unspecified baseline and slightly lowers CLIP score.
- [arXiv 2310.02664](https://arxiv.org/html/2310.02664v2): SD on Imagenette, LoRA r=16 memorised 0.00%, r=1024 8.77% at 400 images; "plain" and "class" are fixed prompts, not trigger-only vs descriptive captions.
- Does not transfer: exact copies at ~70% of samples are far stronger than a few bursts in a 20-40 image set; nothing measured on DiTs, identity or ArcFace.

**RulzTrainer today.** `BucketBatcher.set_weights` already takes per-image weights (dataset.py:384-414) and `_refresh_batcher_weights` (trainer.py:2535-2552) multiplies curriculum, `crop_augment_weight` and loss-watch exclusions into it. A 32x32 thumbnail-correlation dedup (`DUP_MAX` 0.95) exists for selftrain only (selftrain.py:35, 117-125, 193). The scout's rank and caption warnings are not needed: the wizard recommends rank 16 (8 on Qwen 2.1) (webui.py:3108-3116), and trigger + class captions are deliberate for isolation (README.md:1326).

**Fit and risks.** Pure data-side sampling; no interaction with the compute stack. Key clusters by `it.image`, not `item.key`, so crops and multi-resolution items of one photo share a cluster. Sampling with replacement raises epoch-to-epoch variance slightly.

**Implementation path.** `dedup_weight` (bool) and `dedup_threshold` (0.95); a helper reusing `selftrain.thumb` (optionally `lookfilter.body_embedder`), cached in `.rulz_cache/dedup.json`; union-find clusters; multiply by 1/|cluster| in `_refresh_batcher_weights`; log "N near-duplicates in M clusters"; add to race keys (race.py:59-64).

**How to test.** Only on a set with at least 3 clusters of size >= 3. Race on vs off, Qwen 2.1 int8, 0.5 MP, 1500-2000 steps, probes every 200 steps on new-scene prompts, plus nearest-training-image background similarity (face masked). Keep if best ArcFace stays within -0.01 and background similarity drops 0.03+ over 2-3 seeds; drop if ArcFace falls more than 0.02.

### Regularisation set and background-aware captions

**Skip-ish. Expected gain: at or below one-seed probe noise on likeness; possible less background/outfit bleed. Confidence: low (0.3). Cost: ~1 day plus 10-60 min of renders per subject.**

**What it is.** The WACV 2025 "Data Perspective" recipe: subject captions end with an explicit background clause, plus a generated class-image set with attribute-pool captions (adjective dropout) split 20% same background / different subject, 60% different both, 20% stylised.

**Evidence.** [arXiv 2311.04315](https://arxiv.org/html/2311.04315v4) ([CVF](https://openaccess.thecvf.com/content/WACV2025/html/He_A_Data_Perspective_on_Enhanced_Identity_Preservation_for_Diffusion_Personalization_WACV_2025_paper.html)): DreamBench DINO 0.682 -> 0.704 / CLIP-I 0.808 -> 0.824 on SD full fine-tune; 0.744 / 0.842 on SDXL rank-32 LoRA with text encoder trained (no SDXL baseline); 100 reg images still prevent overfitting. Caveats: their DreamBooth baseline ran 1500 iterations vs 4000-8000; objects and pets, no faces; CLIP-T drops 0.301 -> 0.293. Contrary: [Furkan Gozukara's Flux comparison](https://dev.to/furkangozukara/compared-civitai-default-flux-lora-vs-my-best-flux-and-more-developments-5133) and [r/StableDiffusion threads](https://lr.us.psf.lt/r/StableDiffusion/comments/1dxocch/lora_regularization_images) find reg images neutral or harmful for characters. [Fizgig LORA_ISOLATION.md](https://github.com/shootthesound/Fizgig/blob/master/docs/LORA_ISOLATION.md) already gives DOP + stranger pairs (48/50 -> 0/50 strangers read as subject on Krea 2, ArcFace 0.77 -> 0.78, ~1.5x time). The inpainting step and the 1:0.25 ratio are the scout's own.

**RulzTrainer today.** LoRA Isolation (config.py:243-250, README.md:1300-1340) covers class preservation without images; `region_prior_weight` (trainer.py:986) pulls background to base; a trigger-less dataset already "trains as it is" (README.md:1340); selftrain has the render-and-register machinery. Missing: per-dataset loss weight, reg-prompt generator, caption variants. RulzTrainer has no captioner, so the background clause can only be a WebUI caption lint.

**Implementation path and test.** Add `loss_weight` and `role: subject|reg` to `DatasetSpec` (useful on its own, shared with the synthetic-views item); reg items skip DOP, isolation balancing, region maps and loss watch. A `regset.py` reusing selftrain's render loop (200 images, templated pools, 20% reusing background phrases from subject captions instead of inpainting), `reg_share` 0.25, `reg_weight` 0.5-1; 3 cached caption variants for dropout. Sequential race on Qwen 2.1 int8 (isolation does not run with race_parallel, config.py:607): A baseline, B isolation, C B + reg set, D C without the same-background 20%. Equal wall clock. Keep C/D only if face likeness is within 2-3 points of B and stranger leak, CLIP-T or background diversity improves beyond seed spread on both datasets.

### Self-reference identity tokens (Krea 2, MOSS, future Klein)

**Watch (Prototype on Krea 2 only if the cheap variant shows promise). Expected gain: trigger-only likeness may be zero or negative; the likely gain is a second product (stronger likeness when a face photo is supplied at inference). Confidence: low (0.3). Cost: several days of model-module work; about +20% wall time at a 50% reference share.**

**What it is.** Train the person LoRA with a second photo of the subject as clean t=0 reference tokens on their own RoPE index, with reference dropout so the trigger-only path still works. Klein is natively multi-reference; Krea 2 has Ostris's community edit layout; MOSS has a reserved RoPE r axis.

**Evidence.**
- Klein: [HF blog](https://huggingface.co/blog/flux-3) (5 per-stream + 28 joint blocks, 4-axis RoPE, each 1024^2 reference adds 4,096 tokens, 1.93 s vs 3.87 s on H100); [diffusers img2img script](https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_flux3_klein_img2img.py) (bn-normalised refs at l=-1, t=0, no ref dropout, rank 4, lr 1e-4); [NeMo recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/model-coverage/diffusion/black-forest-labs/flux3-image-klein.mdx) (T2I targets exclude the reference stream); [pup_prompts.json](https://raw.githubusercontent.com/black-forest-labs/flux-3-image-klein/main/src/flux3/pup_prompts.json).
- [Sparse Context](https://arxiv.org/html/2606.23682) (Klein-9B, 2x A100): ref-token keep 0.05-0.25 lifts CLIP-I 0.708 -> 0.754, DINO 0.412 -> 0.455. [CRAFT](https://arxiv.org/html/2608.14403) (4x B200, rank 64): XVerseBench single-ID 84.22 vs 66.29. Both are multi-subject zero-shot models on CLIP-I/DINO/XVerseBench; these numbers do not transfer to a rank 8-16 single-person LoRA scored with ArcFace.
- Krea 2: [ComfyUI-Wiki](https://comfyui-wiki.com/en/news/2026-07-04-ai-toolkit-krea2-edit-training) and [ai-toolkit krea2.py](https://raw.githubusercontent.com/ostris/ai-toolkit/main/extensions_built_in/diffusion_models/krea2/krea2.py): 384^2 VLM budget, clean t=0 latents, kv_cache asymmetric mask ("must be trained with kv_cache" for cached inference), ~1750 steps for an edit concept. [yijunwang2/krea2-reid](https://huggingface.co/yijunwang2/krea2-reid) (rank 32, 25.6K steps) is a generic reference-identity LoRA with qualitative results only and reported age/skin drift.
- MOSS: [tech report](https://arxiv.org/html/2610.23374) (r axis, r=0, reference conditioning is future work); [ComfyUI PR #17296](https://github.com/comfyanonymous/ComfyUI/pull/17296) (826 -> 457 ms/step with cached ref KV). The chinoll card does not state rank 64 or ConceptEdit-12M training; the base never saw r>0, the weakest prior.
- Contrary: Reddit reports on Klein 9B say reference-only identity "doesn't beat a good LoRA" and breaks with face angle.

**RulzTrainer today.** No model accepts reference tokens: krea2/model.py:286-291 builds a 3-axis grid for [text, image]; moss/model.py:143-156 hard-codes r=0. Reusable: face/bust crops (dataset.py:215-261) and the probes. Main cost: krea2/model.py:173-177 has one temb_mod for all tokens, so reference tokens need a second modulation per block (same for MOSS, moss/model.py:187-237); krea2/model.py:330-334 moves temb_mod across devices.

**Fit and risks.** Fixed 512^2 references (1024 tokens) to limit compile churn; recalibrate `checkpoint_keep_blocks` with a reference present; `taps.set_layout` (krea2/model.py:327, moss/model.py:229) must exclude reference tokens. Krea nvfp4 at 0.5 MP: ~9 s -> ~12-13 s/step on reference steps. Ostris's layout also feeds the reference through Qwen3-VL, which needs a cache entry per fixed (caption, reference) pair. Reference-conditioned use needs Ostris's or kijai's ComfyUI node with an exactly matching layout. The main unmeasured risk is the LoRA leaning on the reference and storing less identity in its weights.

**Implementation path.** Cheapest first experiment with no model change: render varied-pose images of the subject with krea2-reid on Krea 2 Turbo and feed them through selftrain.py. If a full build follows, start on Krea 2: `ref_latents` in krea2/model.py with second modulation from sigma=0 and an optional `self_reference_kv_cache` mask; `self_reference_share` (default 0) and `self_reference_size` (512) in dataset.py with fixed mismatched-image crops; trigger-only probe stays the score for best checkpoint, early stop and races. Design Klein's adapter with the image_cond stream and a "ref" LoRA target group.

**How to test.** Krea 2 Raw nvfp4, 0.5 MP, rank 16, 2000 steps: A baseline; B `self_reference_share 0.5`; optional C 0.25. Metric: trigger-only probe likeness every 100-200 steps and s/step; secondary with-reference likeness on a held-out photo and a pose-copy check (skeleton3d similarity to the reference, no higher than A's). Drop B if trigger-only likeness trails A by more than ~2 points at matched wall time after 1000 steps; keep only if within noise and the with-reference score adds 5+ points, on 2-3 subjects.

### Personalised few-step distillation (OPD) for MOSS probes

**Watch. Expected gain: none for in-training probes (chicken-and-egg); at most a single 8-12-step personalised file. Confidence: low (0.3). Cost: about 25-30 s per distill step on MOSS int8, a few hundred steps ~2-2.5 h, as long as the identity run.**

**What it is.** DiffSynth's OPD distills an 8-step LoRA by rolling out the student (CFG 1), picking a random step, and regressing one student step onto the frozen teacher's integration of that segment (CFG 3, skip layer 22). Putting the finished identity LoRA inside both teacher and student would give a turbo distilled around that identity.

**Evidence.** [train_opd_lora.py](https://raw.githubusercontent.com/modelscope/DiffSynth-Studio/main/examples/moss_image/model_training/special/opd/train_opd_lora.py) and [the .sh](https://raw.githubusercontent.com/modelscope/DiffSynth-Studio/main/examples/moss_image/model_training/special/opd/train_MOSS-Image-SFT-8steps_lora.sh): lr 1e-4, rank 64, 5 epochs, 50 teacher sigmas, student timesteps 1000...24; the `min_distill_idx` bug is real. [DiffSynth README](https://github.com/modelscope/DiffSynth-Studio) (MOSS added 2026-10-20). [lodestones Base DMD](https://huggingface.co/lodestones/MOSS-Image-Base-8-steps-dmd) (rank 128, work in progress). No quality table anywhere; [Direct Distill](https://diffsynth-studio-doc.readthedocs.io/en/latest/Training/Direct_Distill.html) is deprecated with no metrics. Nobody has distilled around an identity LoRA.

**RulzTrainer today.** Probes use lodestones' DMD at 12 steps as a sampling extra (lora.py:233-241, 949); config.py:37-38 and README.md:1424-1426 already note it "can understate likeness", with full-quality probing (`turbo_lora: ''`, 20 steps, CFG 3) as the remedy. The reward stage (trainer.py:2752-2880, reward.py:99-131) has most of the rollout and CFG machinery.

**How to test (no code, decides whether to build).** Re-probe 2-3 finished MOSS runs' checkpoints with (a) full quality (`turbo_lora: ''`, 30 steps, CFG 3, skip 22) and (b) the generic DMD at 12 steps. If the Spearman rank correlation of face and body likeness across checkpoints is >= 0.9 and the best checkpoint is the same, drop the idea for probes. Otherwise build `distill.py` (trainable turbo extras started from lodestones' weights, identity LoRA frozen) and keep it only if it closes 50% of the |a - b| gap within 300 steps and never scores below (b).

### Optimised reference conditioning (RefMods)

**Watch. Expected gain: about zero for the user's workflow until a reference path is supported end to end. Confidence: low (0.3). Cost: deferred.**

**What it is.** Textual inversion on the reference tokens of a reference-conditioned DiT: encode face crops as reference latents, make them the only trainable tensor, run ~200 steps of the flow loss on the stills with an L2 pull toward the plain encode.

**Evidence.** [Fizgig REFMOD_HOWDOI.md](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/REFMOD_HOWDOI.md): H3 only; level with the plain encode in-distribution; 28 vs 21 on one unseen look (clown make-up) on Fizgig's own scale; a ref2va mod run on fl2va scored 60 vs 65 for a matched mod; "a few minutes on a 32 GB card", also runs on 16 and 24 GB. Never compared with a LoRA. [ComfyUI-Wiki](https://comfyui-wiki.com/en/news/2026-09-07-minimax-h3-refmods) says "for maximum likeness and generalisation, LoRAs are still king"; the [RefMod Pilot](https://huggingface.co/no3us/minimax-h3-refmod-pilot) says video likeness is untested. Applying it to Qwen 2.1 or Krea 2 is the scout's own idea.

**RulzTrainer today.** Nothing similar. fibo/model.py:330-364 accepts `context_latents`, but trainer.py never passes them; Qwen 2.1's edit path is not ported (qwen21/model.py:25-26); MOSS and Krea 2 T2I have no reference input. The gradient must reach the input tokens, which defeats `lora_blocks` backward skipping. No ComfyUI or diffusers loader exists for such a file on FIBO or Qwen 2.1.

**Implementation path and test.** Defer. If FIBO Edit is wired end to end (context_latents through trainer.py and the fibo sampler), a `refmod.py` side job (refmod_steps 200, refmod_anchor, sigma window) can be raced: A LoRA only, B + plain encode, C + optimised RefMod, D RefMod during LoRA training; adopt only if C or D beats A by 0.02+ ArcFace on an out-of-distribution prompt set on two subjects at most 1.5x step time.

### Same-subject in-context composites

**Skip-ish. Expected gain: slightly negative to neutral on single-image ArcFace/DINO probes. Confidence: low (0.25). Cost: ~0.5 day.**

**What it is.** Two photos of the same person on one 2:1 canvas with a panel-labelled caption, so joint attention ties the panels together, optionally with the loss masked to one panel.

**Evidence.** [In-Context LoRA](https://arxiv.org/html/2410.23775) (FLUX.1-dev, one A100, 5,000 steps, batch 4, rank 16, 20-100 sets per task) shows cross-panel consistency, with no identity metric; it says SDEdit "tends to be unstable, often failing to preserve identity". [Fizgig LORA_ISOLATION.md](https://github.com/shootthesound/Fizgig/blob/master/docs/LORA_ISOLATION.md) pairs are subject + stranger, never the subject twice. Counter-argument: with the same person in both panels the model can lower loss by copying across panels, weakening the pressure to store identity in the LoRA (the one-panel mask makes that shortcut stronger), and each face gets half the pixels. Correction: Qwen 2.1 is block-causal only over text tokens; image tokens see everything (qwen21/model.py:20, 104).

**RulzTrainer today and test.** `isolation.build_pairs` (isolation.py:203-245), pair registration (trainer.py:2400-2422), per-item `loss_scale` (trainer.py:1007) and `alignment.panel_mask` (alignment.py:149-160) already exist. If raced: `composite_pairs` at ~15% share vs baseline crop_augment on Krea 2 nvfp4 and Qwen 2.1 int8, 2 seeds, patience 2; drop if best ArcFace is not 0.01 above baseline by step 1500 or the diptych rate (2+ faces or a seam in single-image probes) exceeds 2%.

## Text side

This section covers ideas that act on the caption and text-encoder side: how the trigger is tokenised and placed, how many caption variants each image gets, word-level caption dropout, learned trigger embeddings in the frozen Qwen encoders, token-masked LoRA, and feeding a reference crop through the VLM encoder. The overall verdict is modest. All six items are backed by real papers, but no source measures a likeness gain for a rank 8-16 character LoRA on a 5-12B flow-matching DiT. The two cheap items are worth doing: fix the trigger-form mismatch between training captions and probe prompts, and add a multi-variant caption loader. Everything else is a race at best, and two items (TARA masking, soft persona tokens) mostly do not transfer.

### Trigger form consistency and tokeniser lint

**Do it (probe A/B first, then a cheap code change). Expected gain: unmeasured; guess under +0.01 ArcFace, plus a fairer probe signal for best checkpoint, early stop and races. Confidence: low (0.35). Cost: minutes for the probe A/B, about half a day for the option and the wizard lint.**

**What it is.** Krea 2, Qwen-Image 2.1 and MOSS use causal Qwen encoders with BPE tokenisers. A "rare" trigger splits into frequent fragments, and the first fragment has a different id depending on whether it follows a space. Training captions put the trigger at caption start ('trigger, caption' right after 'user\n', bare first piece). Every preset probe and sample prompt puts it mid-sentence ('close-up portrait photo of {trigger}', space-prefixed first piece). So the probe asks for a token sequence the LoRA was never trained on, sharing only the later pieces. Under the causal mask the early pieces are encoded as in any other context, so identity can only sit in the last piece and the tokens after it.

**Evidence.**
- Local tokeniser measurements (verified by the verifier with the `tokenizers` library on `models/tokenizer__Qwen-Image-2.1`, vocab 151,669, and `models/tokenizer__MOSS-Image`, vocab 248,077):
  - 'ohwx' -> oh|wx; 'sks' -> s|ks; 'zwx' -> z|wx; 'b0b' 3 pieces; 'p3rs0n' 5 pieces.
  - The fused trigger 'gweltman' -> gw|elt|man on both tokenisers.
  - First id differs between bare and space-prefixed forms: Qwen 54133 vs 56572, MOSS 52315 vs 54677 (for 'gw'); Qwen 'oh' 2267 vs ' oh' 14019.
  - ' Olivia' is one token, 'Olivia' two; ' TOK' one token, 'TOK' two.
- [Follow the Flow, arXiv 2504.01137v2](https://arxiv.org/html/2504.01137v2): 89% of items have a representative token; in causal encoders (Sana/Gemma, SDXL-Turbo CLIP) it is typically the last token. Pruning non-representative tokens raised prompt accuracy 83.48% -> 87.02% on 339 prompts with FLUX-schnell. Does not transfer: that is inference-time pruning with a bidirectional T5, not LoRA training.
- [LI-DiT, arXiv 2406.11831](https://arxiv.org/html/2406.11831): decoder-only encoders do worse on attributes late in the prompt (1,000-prompt position benchmark). Its instruction prefix (+4.19 T2I avg) is already Krea 2's native system prompt (`krea2/loading.py:41`), so no lever there.
- [Lost in the Prompt Order, arXiv 2601.14152](https://arxiv.org/abs/2601.14152): >14 pp loss for decoder-only LLMs when needed information comes later. LLM multiple-choice QA, not text-to-image; weak support.
- No source measures trigger form vs LoRA likeness on any DiT. Community reports on Qwen-Image are split and unmeasured.

**RulzTrainer today.**
- `archs.add_trigger` (`rulztrainer/archs.py:104-107`) returns `f"{trigger}, {caption}"`.
- `PRESET_PROBE_PROMPTS` (`rulztrainer/config.py:107-111`): 'close-up portrait photo of {trigger}, ...' and 'full body photo of {trigger} ...'.
- Isolation's fuse() rewrites both training and probe prompts to the fused word (`trainer.py:2390-2434`), so a fused 'gweltman' is also trained bare and probed space-prefixed.
- `isolation.fusion_warning` (`isolation.py:141-157`, called from `trainer.py:618`) is a partial tokeniser lint, log only.
- `alignment.token_positions` (`alignment.py:267`) counts every overlapping piece equally for attention localization.
- Delta: a trigger-form option, matching probe forms, and the lint in the wizard.

**Fit and risks.** Pure data and prompt side. No interaction with W8A8, nvfp4, compile, keep blocks, block swap, model_split, data_parallel or export. Text caches unchanged in size ('both' doubles the entries per item, which is small). The real risk is only that the gap is inside seed noise and nothing changes. Real-name single tokens (' Olivia') carry demographic priors and can bleed.

**Implementation path.**
1. `caption_trigger_form: prefix | sentence | both` in `config.py`, implemented in `archs.add_trigger` and FIBO's `prompts.add_trigger`. 'sentence' writes 'a photo of {trigger}, {caption}' so the first piece gets the same leading space as in probes; 'both' alternates by `hash(path) % 2` with two text-cache entries.
2. Or in addition, `probe_trigger_form`, or simply make the preset probe prompts start with '{trigger}, close-up portrait photo...' to match the caption-start form.
3. Wizard: show the trigger's pieces and ids in three forms (bare at caption start, after ', ', mid-sentence) through `arch.get_tokenizer`, and surface `fusion_warning` there. Read-only.
4. Optional, marginal: per-piece weights from `alignment.token_positions` (last piece 1.0) for attn_localize only.
5. Skip cache piece pruning (the DiT was trained with every token present; gain unknown, possibly negative).
6. Update the README and wiki trigger section.

**How to test.**
- Step 1, no training: take the local krea2-AZav-rulz run and one Qwen 2.1 int8 LoRA. Run the face and body probes with '{trigger}, close-up portrait photo, neutral background, soft light' vs 'close-up portrait photo of {trigger}, ...', 16 seeds each, turbo sampler. Compare ArcFace mean and DINO body.
- Stop if the gap is under 0.005 ArcFace: ship only the wizard lint.
- Step 2, only if the gap is 0.01 or more: race `caption_trigger_form` prefix vs both, one config per GPU, same seed, 1500-2000 steps, rank 16. Probe in BOTH forms and rank by the mean ArcFace over the two at best checkpoint. Race patience 3 probe rounds; repeat with a second seed before changing the default.

**Refuted member: anchor phrase (finding 70).** Replacing or initialising the trigger with a demographic phrase ('korean man') has no LoRA likeness evidence. In the FAD paper the anchored 'Normal' DINO values equal the trigger-token values, Cross-Init, ID-EA and DTI are textual inversion only, and the data-perspective paper is objects and pets. Isolation already accepts a multi-word class (`isolation.py:96-134`), so a manual anchor is possible today. If anyone wants it: race class 'man' vs 'korean man' under LoRA Isolation and adopt only if ArcFace rises and stranger bleed against the bare anchor prompt does not.

### Caption variants per image (caption_mix) and optional captioner

**Race it. Expected gain: unmeasured; likely within probe noise unless existing captions are poor or mismatched, possible gain in prompt-length robustness. Confidence: medium (0.4). Cost: about 1 day for the loader; the captioner is a second phase of 1-2 days.**

**What it is.** Each image gets several cached captions (for example short, medium, long, tags), each carrying the trigger, and one is drawn per step by a `caption_mix` weight. This matches how the base models were trained: MOSS sampled short/medium/long recaptions uniformly, and Krea 2 trains mostly on long captions with short and medium exposure. It extends the current two-point choice (full caption or trigger-only dropout). An optional captioner writes the variants with the Qwen3-VL already on disk and can recaption images the loss watch flags as stuck.

**Evidence.**
- [MOSS-Image technical report, arXiv 2610.23374](https://arxiv.org/html/2610.23374) (confirmed in full): Gemma4 recaptions in three lengths chosen uniformly, generated caption used with p = 0.8 (original:recaption 20:80), mostly English, text tokens excluded from the loss, 4096-token final-stage limit.
- [MOSS USERGUIDE](https://raw.githubusercontent.com/OpenMOSS/MOSS-Image/main/USERGUIDE.md): post-training JSON schema `caption.long/medium/short` plus watermark/borders; captioners Gemma 4, Qwen 3.5 27B, Qwen 3.5 35B-A3B in thinking mode, temperature 1, top_p 0.95; tag order Quality > Score > Meta > Year > Character > Series > Artist > General.
- [KohakuMossTrainer first-lora guide](https://raw.githubusercontent.com/KohakuBlueleaf/KohakuMossTrainer/main/docs/guides/first-lora.md): 4352 tokens fits 4096 image tokens plus a caption up to 256 tokens at 1 MP.
- [Krea 2 technical report](https://www.krea.ai/blog/krea-2-technical-report): mostly long captions plus short/medium exposure (scout's 'prompt-length coverage' quote is a paraphrase). [Krea encoder.py](https://raw.githubusercontent.com/krea-ai/krea-2/main/encoder.py): fixed system prompt, 34 prefix tokens stripped, max_length 512.
- [Fizgig KREA2.md](https://github.com/shootthesound/Fizgig/blob/master/docs/KREA2.md): recaptions stuck images with Qwen3-VL-4B; 'the top flags were all caption/image mismatches'; 'faster likeness and a higher ceiling' with no numbers.
- Caption policy (finding 67, weakened): [FreeCus, arXiv 2507.15249](https://arxiv.org/html/2507.15249) DINO 0.709 concise / 0.700 detailed / 0.696 filtered / 0.682 detailed+filtered on FLUX.1-dev 512 px. Does not transfer: FreeCus is training-free (caption used at inference) and its filter keeps subject attributes and strips background, the opposite of the proposed identity-attribute filter. [Beyond Facial Consistency, arXiv 2607.25622](https://arxiv.org/html/2607.25622) is an encoder-conditioned model with no caption ablation. [krea2-lora-training](https://github.com/chengyansen-ai/krea2-lora-training) and [danmo-make PR #76](https://github.com/danmo-ai/danmo-make/pull/76) give 'describe only what changes' advice with no metrics. Scenario's guide says the opposite (detailed descriptions help Qwen-family character fidelity). The Qwen-Image system-prompt note in the finding applies to Qwen-Image 1, not 2.1.
- The RoPE-index-0 length-signal mechanism in finding 93 is speculation.

**RulzTrainer today.**
- `dataset.read_caption` (`dataset.py:120`) returns one caption; `arch.caption_file` (`archs.py:109`, FIBO .json at `archs.py:299`) picks the file; one `text_path` per item (`dataset.py:43, 178`).
- Collate draws only between `it.text_path` and the trigger-only dropout prompt (`dataset.py:432-440`); `caption_dropout` 0.05 and `caption_dropout_keep_trigger` at `config.py:228-229`.
- loss_watch flags stuck images and tells the user to edit the caption (README 650-662). No captioner, no `caption_mix`, no `caption_policy`.
- Krea's ComfyUI TE file includes `model.visual.*` (`krea2/single_file.py:18`); Qwen 2.1's TE file has no vision tower (`qwen21/single_file.py:10-13`); MOSS's Qwen3.5-9B is unchecked.

**Fit and risks.**
- Variants are more entries in the hash-keyed text cache plus a per-step draw. No effect on W8A8, nvfp4, keep blocks, block swap, model_split, sigma grid or export. No s/step change (Krea nvfp4 about 9 s, MOSS int8 about 5 s, Qwen 2.1 int8 about 2.5 s).
- More text-length variety: watch for recompiles if text length is not marked dynamic.
- data_parallel threads draw independently; fine.
- Attention localization finds trigger tokens per cached caption, so every variant must contain the trigger.
- LoRA Isolation has one `dop_caption`/`dop_text_path` per item (`dataset.py:51-52, 327`); crop templates (`{caption}`) need a rule for which variant they wrap.
- Very short captions can increase bleed into the class word; keep the trigger in all variants and run with isolation where bleed matters.
- An identity-attribute filter could strip attributes the user wants to steer and has no measured support: opt-in only.
- Captioner: a 4B VL in bf16 (about 9 GB) fits as a pre-training side job; plain transformers, no Windows issue.

**Implementation path.**
1. `read_caption` returns a list: `<stem>.txt` plus optional `<stem>.short.txt` / `.long.txt` / `.tags.txt`, or a MOSS-schema `<stem>.json` with `caption.{short,medium,long}` for MOSS (FIBO keeps its VGL JSON). `add_trigger` on each.
2. `Item.text_paths` list of (variant, weight); `text_path` stays the first variant.
3. Add every variant to the cache jobs (`dataset.py:326`).
4. Collate draws by a `caption_mix` dict (default uniform over variants present), keeping the trigger-only dropout draw.
5. DOP and crop captions stay on the primary variant at first.
6. Phase 2: `caption_policy: off | variants` side job building Qwen3-VL-4B from Krea's TE file (or an HF Qwen3-VL), writing `.rulz.*.txt` beside images, never overwriting user captions.
7. Optional: loss_watch 'stuck for 2 epochs' queues a recaption and cache refresh (Fizgig's rule).
8. Add `caption_mix` to the race's 'Data & augmentation' group (`race.py:70`). Document in README and wiki.

**How to test.**
- Race on one character dataset, same seed, steps and LR, one config per GPU, Qwen 2.1 int8 first, then MOSS: A current single caption + caption_dropout 0.05; B caption_mix uniform over long/medium/short; C concise context-only captions.
- Probe every 250 steps with two prompt sets: short ('{trigger}, portrait') and long (80-150 token enhanced prompts). ArcFace and DINOv2 body on each set, plus CLIP-T or VQA adherence on the long set.
- Stop at 2000 steps or race patience. Adopt B or C only if best-checkpoint ArcFace beats A by at least 0.02 on both prompt sets over 2 seeds with no adherence drop. Otherwise keep the loader as an opt-in data format (MOSS JSON variants) without changing defaults.

### Frequency-aware caption dropout (FAD / sFAD)

**Race it (after the two items above). Expected gain: unmeasured for this workload; plausibly +0 to +0.01 on the ArcFace/DINO probes, more if captions often name hair, ethnicity or beard next to the trigger. Confidence: low (0.35). Cost: about 1 day plus about 6x text-caching time and about 1 GB disk on Krea 2.**

**What it is.** Word-level caption dropout weighted by how often each word co-occurs with the trigger. For captions containing the trigger, r(w) is the fraction that contain word w; p_drop(w) = p_min + (p_max - p_min) * sigmoid(alpha * (r(w) - c)), with p(trigger) = 0. Each token is dropped with p_drop(w) * p_step(i), where p_step is 0 during a 10% warm-up and then rises (1 - exp(-beta * s)) in the scheduled variant sFAD. Words that always sit next to the trigger are removed most, so identity cannot park in them.

**Evidence.**
- [Koh et al., 'Let Triggers Control', arXiv 2603.27199](https://arxiv.org/abs/2603.27199) ([HTML](https://arxiv.org/html/2603.27199v1), [CVF supplemental](https://openaccess.thecvf.com/content/CVPR2026W/P13N/supplemental/Koh_Let_Triggers_Control_CVPRW_2026_supplemental.pdf)), CVPR 2026 P13N workshop, no code. Equations confirmed; p_min/p_max 0.35/1.0; c, alpha and beta not reported; Table 4's 0.1-0.8 step range contradicts Eq. 7 reaching 1.
- Setups: SD1.5/SDXL lr 5e-5, 1500 steps, one RTX 3090, batch 8 with gradient accumulation 64 (far from batch 1). FLUX/Qwen-Image: rank 32, lr 1e-4, 2 epochs, repeat 50, about 50 images per concept, tag-style captions.
- Main Table 1 (trigger setting), Qwen-Image: plain FAD is flat or negative (hsng 0.7347 -> 0.7297, reeves 0.6543 -> 0.6502, faker +0.005, mbst +0.004). sFAD rows for Qwen-Image share identical values with the FLUX rows (reeves 0.7015, mbst 0.7149 and others), which looks like a copy error.
- Table 3 (anchor setting, a known phrase like 'japanese man' replaces the trigger): Qwen-Image hsng 0.7347 -> 0.7840, mbst 0.6784 -> 0.6840, faker 0.6373 -> 0.6529, reeves 0.6543 -> 0.6576; mean about +0.019. The scout's +0.049 headline is one identity in this setting. FLUX baselines differ between Table 1 (0.6945) and Table 3 (0.5628).
- SD1.5 faker: FID 207.10 -> 195.86, InsightFace distance 28.06 -> 26.95, GPT-4.1 similarity 6.98 -> 7.70 (50 image pairs). SDXL pochacco CCIP 0.8996 -> 0.9295.
- Precedent: [He et al., arXiv 2311.04315](https://arxiv.org/html/2311.04315v4) randomly drops descriptor words, DINO 0.704 vs 0.682 on SD; random, not frequency-weighted.
- No follow-ups, reproductions or community reports found.

**RulzTrainer today.** Whole-caption dropout only: `load_batch` (`dataset.py:440`) swaps the caption for the dropout prompt with probability `caption_dropout`; that prompt is the trigger alone with `caption_dropout_keep_trigger` (`dataset.py:270-280`, `config.py:228-229`). `caption_dropout` is already raceable (`race.py:70`). Keep-trigger dropout, region loss, ID loss and attention localization already push the trigger to carry identity, so FAD overlaps with them.

**Fit and risks.**
- Captions are cached and the encoder unloaded, so FAD must be pre-cached variants. Masking rows of a cached embedding is invalid: Qwen3-VL and Qwen3.5 are causal, so later tokens still carry the dropped word. Variants must be real re-encodes.
- Disk: Krea 2 caches 12 x 2560 bf16 per token, about 61 KB; 6 variants x 40 tokens is about 15 MB per image, about 0.7 GB for 50 images. MOSS and Qwen 2.1 cache a single layer, much smaller.
- `_trigger_positions` (`trainer.py:2681-2708`) only maps `it.caption` and the dropout prompt; variant paths must be added or attention localization and region batches silently lose trigger positions.
- Exempt crop captions, self-train items and LoRA Isolation DOP captions and pairs. FIBO VGL JSON needs field-level dropping or an exemption. Dropping words from natural sentences breaks grammar (the paper used tags).
- No effect on W8A8, nvfp4, keep blocks, block swap, model_split, data_parallel, sigma grid or export. No new compile shape class.

**Implementation path.**
- `config.py`: `fad` (off), `fad_min` 0.35, `fad_max` 1.0, `fad_center` 0.5, `fad_slope` 10, `fad_beta` 3, `fad_warmup` 0.1, `fad_variants` 4-6.
- `dataset.py` scan: split captions into words or comma phrases (trigger kept), compute r(w) per dataset over captions with that dataset's trigger, sample K variants at increasing p_step with a seeded RNG, store `it.fad_text_paths`, cache through `text_jobs` (the multi-GPU `run_queue` handles them).
- `load_batch`: when not caption-dropped, map the scheduled p_step(i) to the nearest level, choosing stochastically between the two neighbours; pass the step fraction in.
- Add variant paths to `_trigger_positions`; add `fad` to `race.py:70`; README and wiki next to caption dropout. Krea 2, Qwen 2.1, MOSS; skip FIBO at first. Flux 3 Klein and MiniMax H3 would get it through the same text-cache path once added.

**How to test.**
- Race `fad` on vs off, Krea 2 nvfp4 and Qwen 2.1 int8, one config per GPU, 20-40 images with sentence captions that mention hair, ethnicity or clothing, rank 16, 0.5 MP, 2000 steps, 2 seeds per arm.
- Metrics: ArcFace face and DINO body on trigger-only prompts and on prompts that change hair or outfit; disentanglement prompt 'trigger with blonde hair' on a dark-haired set (face score stays high, hair changes). Report best-checkpoint probe and area under the probe curve.
- Adopt only if `fad` beats off by at least +0.01 ArcFace (or +0.01 DINO) averaged over both seeds on at least one model, with no prompt-following drop. If two models x two seeds are within ±0.005, drop it.
- If adopted, an anchor-phrase arm could be raced, since the paper's biggest number comes from that setting.

### Qwen-Image 2.1 native reference path (visual persona through Qwen3-VL)

**Prototype (Qwen 2.1 only, low priority). Expected gain: unmeasured; about 40-60% slower steps, and the LoRA then needs a reference crop at inference. Confidence: low (0.2). Cost: several days (vision-slot port in the DiT, caching, probes).**

**What it is.** Give the identity to the encoder as an image, not only as a trigger. The verifier found that Qwen-Image 2.1's base checkpoint is already a unified generate/edit model: the encoder puts '<|vision_start|><|image_pad|><|vision_end|>' before the prompt in the user turn and runs Qwen3-VL on the reference, and the DiT replaces each image_pad slot with 2x2 VAE latent tokens of the reference, modulated at t = 0 in the block-causal prefix. So the proposal for Qwen reduces to porting that native path and training the character LoRA with a reference crop present. Identity then flows through the text tokens that attended to the image inside the VLM plus the reference latents.

**Evidence.**
- [Squeezing Capacity from MLLMs for Subject-driven Generation, arXiv 2605.26111](https://arxiv.org/html/2605.26111): InternVL3-8B, layerwise attention pooling over 28 layers, LoRA rank 512 on FLUX.1-dev, about 400K pairs, 8 H100 at batch 16 per GPU, 25K + 10K steps. DreamBench DINO-I 0.7482 / CLIP-I 0.8443 is the 'MLLM + VAE' model; 'MLLM only' scores 0.6788 / 0.8228, below Qwen-Image (0.7317). All-layer vs last-layer pooling 0.752 vs 0.657. Stage 2 trains the MLLM too. No face metric. Does not transfer to a 20-image, batch 1, rank 8-16 LoRA.
- [Qwen-Image report, arXiv 2508.02324](https://arxiv.org/html/2508.02324): TI2I template puts vision tokens then text in the user turn.
- [Beyond Facial Consistency, arXiv 2607.25622](https://arxiv.org/html/2607.25622): 0.7875 ArcFace on Pexels-100 is the authors' own method; InstantCharacter (also FLUX) scores 0.15; Qwen-Image 0.464, conditioning not stated. So how well Qwen's native reference path keeps faces is not established.
- Code evidence (installed diffusers 0.41): `modular_pipelines/qwenimage21/encoders.py:34, 81-103` and `transformer_qwenimage21.py:925-939`.

**RulzTrainer today.** Text-only encoders: Krea drops 'visual.' (`krea2/single_file.py:195`) and builds a text-only model; Qwen 2.1 reuses it (`qwen21/loading.py:119-171`). `qwen21/model.py:25-26` states the edit path is not ported, and the port assumes [text | target] (`model.py:19-23`). No TI or soft-token mechanism exists.

**Fit and risks.**
- Caching needs full Qwen3-VL-8B with its vision tower (about 0.6B), int8 on 16 GB at cache time only.
- Sequence grows: a 448 px crop is 784 tokens vs about 1950 at 0.5 MP, about +40% tokens and +60-90% attention cost; Qwen's 2.5 s/step could become about 3.5-4 s.
- Block-causal mask and RoPE need the vision-slot layout. Fixed crop size for one compile shape; keep-block calibration redone for the longer sequence. W8A8/nvfp4, block swap, model_split, data_parallel unaffected.
- LoRA keys and export unchanged, but users need a Qwen 2.1 reference-image node in ComfyUI (unverified) or the diffusers modular pipeline. A reference-requiring LoRA is a different product.
- Krea 2: diffusers pipeline_krea2 has no image path; a 20-image LoRA would have to teach the whole channel. Likely no gain or a loss. Do nothing there.

**Implementation path.** `ref_condition: off | face_crop`. `qwen21/loading.py`: load the full VL at caching, build the prompt as diffusers' `get_qwenimage21_prompt_embeds` (vision template, left padding), cache an `image_pad_mask`. `dataset.py` / `.rulz_cache`: per item pick a reference crop of the same subject from a different image (highest ArcFace), fixed area (448 px), cache its VAE latents. `qwen21/model.py`: accept the mask and condition latents, expand to 2x2 tokens, t = 0 modulation, RoPE and block-causal mask as in `transformer_qwenimage21.py:925-975`, loss on target tokens only. Sampler and probes pass a fixed crop. Skip the 'seed TI from pooled image state' fallback.

**How to test.** Qwen 2.1 int8, 0.5 MP, rank 16, 2000 steps, same data and seed. Race A text-only baseline; B `ref_condition=face_crop` probed with a fixed held-out crop; C base model with the same crop and no LoRA. ArcFace and DINOv2 body, a pose-diversity/copy-paste check, wall clock per step. Race patience 3 probe rounds. Keep B only if it beats A's best ArcFace by at least 0.03 at equal wall clock and beats C; also probe B without the reference to check the trigger-only path did not regress.

### Directional textual inversion pre-stage

**Watch. Expected gain: likeness neutral or slightly negative; prompt adherence and editability possibly better (not measured by current probes). Confidence: low (0.3). Cost: about 2-3 days, plus a 300-500 step warm-up per run (Krea 2 ~45-75 min, MOSS ~25-40 min, Qwen 2.1 ~12-20 min) and a sidecar embedding problem in ComfyUI.**

**What it is.** Before the LoRA, learn the trigger's input-embedding rows in the frozen Qwen encoder with Directional TI: fixed norm m* (mean vocabulary row norm), direction-only optimisation with Riemannian SGD, and a von Mises-Fisher pull (kappa 1e-4) toward an anchor/class embedding. Then cache captions with the learned rows, unload the encoder and train the LoRA as usual. The fixed norm addresses TI's norm inflation in pre-norm encoders.

**Evidence.**
- [DTI, arXiv 2512.13672](https://arxiv.org/html/2512.13672v1): SANA 1.5 1.6B (Gemma encoder) DINO / SigLIP-text TI 0.480/0.621 vs DTI 0.479/0.744; 4.8B 0.446/0.646 vs 0.452/0.757. One A6000, 500 steps, batch 4, bf16, lr 5e-3, about 30 min per subject on SANA-1.6B. User study text alignment 66.77 vs 10.83.
- Contrary inside the same paper: SDXL subject similarity 0.561 -> 0.450; with DCO + rank-4 LoRA (250 steps) image similarity 0.605 -> 0.568 while text alignment 0.456 -> 0.635. kappa = 0 gives higher image similarity (0.507 vs 0.450). The gain is text alignment, not identity.
- [Cross Initialization, arXiv 2312.15905](https://arxiv.org/abs/2312.15905): TI embeddings drift up to 100x; 5000 -> 320 steps (initialises from the encoder output, not strictly the class token).
- [diffusers advanced Flux README](https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/README_flux.md) and [HF blog](https://huggingface.co/blog/linoyts/new-advanced-flux-dreambooth-lora): pivotal tuning flags, `{lora}_emb.safetensors`, Prodigy 1.0 / rank 8 / 700 steps on a 40 GB A100. Face benefit reported for CLIP only; no T5/LLM-encoder face results.
- No face/ArcFace, Qwen-encoder or 5-12B DiT results anywhere.

**RulzTrainer today.** `encode_prompt` is no-grad and runs once: `krea2/loading.py:141-155` (returns `hidden[PREFIX_TOKENS:]`), `qwen21/loading.py:153-171` (drops the system turn), `moss/loading.py:213-222` (keeps the template). Trigger is a plain string (`archs.py:104-107`). Reusable: `text_encoder_quant`/`text_encoder_device` (`config.py:200`) and trigger positions (`trainer.py:2681`, `alignment.token_positions`), which allow overriding rows via `inputs_embeds` with no tokenizer change. LoRA Isolation/DOP already targets less bleed into trigger-free prompts.

**Fit and risks.**
- Full backward through the DiT to block 0: the lora_blocks backward skip must be off and keep-blocks calibration redone.
- W8A8 and nvfp4 pass input gradients via bf16 int8_backward; the stochastic-rounded FP4 backward is too noisy for one direction vector.
- VRAM: Krea nvfp4 DiT ~8 GB + Qwen3-VL-4B int8 ~4.5 GB + activations is borderline on 16 GB; Qwen 2.1 and MOSS need the encoder on the second GPU via `text_encoder_device` (plain device placement, fine on Windows).
- data_parallel: sum the embedding gradient on the main thread.
- Cache keys need an embedding hash; relay-from-winner would mix caches from different encoders. Samples and probes need the learned rows.
- Export is the show-stopper: the LoRA stays compatible but identity partly lives in a sidecar; ComfyUI support for embeddings in Qwen-family encoders (and Krea's 12-layer tap) is unverified. A final LoRA-only phase on plain captions fixes export but largely undoes the point.

**Implementation path.** `rulztrainer/textinv.py` (m*, unit direction per trigger, RSGD, vMF). `encode_prompt_grad` in krea2/qwen21/moss `loading.py` overriding rows at trigger positions, encoder with grad and per-layer checkpointing, same output slicing. Stage before caption caching in `trainer.py` (DiT frozen, region-weighted flow loss). Cache with an embedding hash, save `{run}_emb.safetensors`. Options `ti_steps` (0 = off), `ti_lr`, `ti_kappa`, `ti_anchor`, `ti_distill_steps`. Qwen 2.1 first, then MOSS, then Krea.

**How to test.** Qwen 2.1 int8, 0.5 MP, rank 16, 2000 steps, 2 subjects x 2 seeds. A plain trigger; B DTI 300 steps (kappa 1e-4, anchor 'woman'/'man') then LoRA; optional C = B + 200-step distill to plain caches. ArcFace and DINO body at equal LoRA steps and at equal wall clock; a SigLIP text-alignment probe on about 8 contextual prompts; a bleed probe on trigger-free class prompts; B also probed without the embedding. Drop if best ArcFace is 0.01 or more below A on both subjects, or SigLIP gains less than 0.01. Adopt as an option only if likeness is within ±0.01 and alignment or bleed improves on both subjects.

**Refuted member: soft persona tokens (finding 71).** K learned tokens in the system or user turn with a VQA warm-up. [UniCTokens](https://arxiv.org/html/2505.14671) face similarity is worse than DreamBooth (0.334 vs 0.436); [PromptGuard](https://arxiv.org/html/2501.03544v3) is safety steering; [Krea-2-Engineer](https://huggingface.co/BennyDaBall/Krea-2-Engineer-V1) makes no likeness claim; [LI-DiT](https://arxiv.org/html/2406.11831)'s instruction gain needs a multimodal-tuned LLM and the finding misattributes that prompt to Qwen 2.1 (whose SYSTEM_PROMPT is 'Comprehend and analyze the provided prompt.', `qwen21/loading.py:41`). Do not race unless DTI wins.

### Token-aware LoRA (TARA)

**Skip-ish. Expected gain: about zero for single-character likeness; possible small adherence gain only with RulzTrainer's own masked sampler, unknown or negative in ComfyUI. Confidence: low (0.2). Cost: about 1 day.**

**What it is.** TARA masks each LoRA's output in the UNet cross-attention K/V paths so it acts only at the rare token's position (Token Focus Masking), and adds a Token Alignment Loss pulling the trigger's key toward the class token's key. The DiT analogue would gate the LoRA delta on text positions to the trigger pieces. But in single-stream DiTs the same q/k/v/o weights act on image tokens, where identity is written, and the mask leaves those on.

**Evidence.**
- [TARA, arXiv 2508.08812](https://arxiv.org/html/2508.08812v1), [code (MIT)](https://github.com/YuqiPeng77/TARA): SD1.5/SDXL, rank 8, lr 1e-5, batch 1, frozen text encoder, one RTX A6000. After merging a second LoRA, DINO: TARA 0.489 -> 0.485 vs DB-LoRA -0.107, Custom Diffusion -0.322, Mix-of-Show -0.009 (15 min merge vs under 1 s). Multi-concept CLIP-I 0.672 / DINO 0.384 / CLIP-T 0.284 vs DB-LoRA 0.603 / 0.277 / 0.295. TFM alone DINO 0.277 -> 0.374; TAL alone lowers CLIP-T to 0.266. No DiT support.
- [Follow the Flow](https://arxiv.org/html/2504.01137v2) supports masking to the last piece in causal encoders.
- All gains are UNet multi-concept composition or merging; no single-identity, DiT or flow-matching result, and no DiT follow-up.

**RulzTrainer today.** `LoRALinear.forward` (`lora.py:381`) applies the delta at every position. Trigger positions exist (`trainer.py:2681`, `alignment.py:267`) but are built only when attention localization is on (`trainer.py:2622`). Layouts: text first for Krea 2 (`krea2/model.py:319`) and Qwen 2.1 (`qwen21/model.py:331`, with causal text-only text queries at 129-131, so a text-only delta acts like a deep per-layer TI); image first for MOSS (`moss/model.py:223`); FIBO reads the caption only in injection blocks. Isolation/DOP already targets bleed.

**Fit and risks.** Export mismatch is the show-stopper: ComfyUI and diffusers apply the delta to every token, so non-trigger text tokens get a delta never trained there, and probes would measure a model the user never gets. Mask must be a tensor attribute to avoid graph breaks; no new recompiles. No effect on W8A8/nvfp4 (fp32 LoRA branch only), keep blocks, block swap, model_split, data_parallel. TAL taps must run uncompiled like attention-localization taps (`trainer.py:2620`). Wrong for style datasets.

**Implementation path.** `lora_text_mask: off | trigger | trigger_last_piece`, default off and not in the wizard. Lift the `if attn` guard at `trainer.py:2622`. Per step a [1,S,1] mask (1 on image positions and trigger pieces), set as `LoRALinear.token_mask` and multiplied into `_added(x)`. All-zero text mask for the DOP pass. Sampler and probes apply the mask and also probe unmasked. Optional `tal_weight`. Krea 2, Qwen 2.1, MOSS; skip FIBO. A stricter variant that also masks image positions is a deep per-layer TI, leak-free but with far less capacity; only a race arm.

**How to test.** Qwen 2.1 int8, 0.5 MP, rank 16, 2000 steps, one dataset, same seed: A plain LoRA; B `lora_text_mask=trigger`; C B + TAL weight 1; D B + LoRA Isolation vs the existing isolation run. ArcFace and DINOv2 body, isolation stranger test, CLIP-T or VQA adherence, each scored masked and unmasked. Drop any arm whose unmasked ArcFace falls more than 0.02 below A or whose unmasked stranger leakage is not lower than A's. Keep only if B or C beats A on leakage or adherence at equal likeness, unmasked, on at least two datasets.

## VRAM without losing speed or quality

This section covers ways to cut VRAM, or to turn spare VRAM, RAM and PCIe into speed, without hurting likeness: smaller kept activations, finer checkpoint policies, activation offload, faster or more accurate 4-bit bases, mixed formats, adapter-side savings, fine-tune window memory and block swap. Overall verdict: on the two 16 GB cards most memory levers are small, and on 80-141 GB rentals the main lever is simply keeping every block (checkpointing off, 13-27% per step in Fizgig's A100/A40/RTX PRO 6000 measurements), so the memory items matter most on 24-32 GB cards such as the RTX 5090; each subsection ends with an "On bigger vast.ai cards" note. RulzTrainer's own measurements (README.md:1735-1758) show that removing recompute for the blocks that fit gains only 0-4%, because the W8A8/nvfp4 forward is cheap next to the exact bf16 backward. The best bets are an exact bf16 norm-copy fix plus a gated fp8 kept-activation trial, a W4A8 forward for nvfp4 bases on the 4060 Ti, a no-code nvfp4-vs-int8 likeness race that decides whether any recovery branch is worth building, and a suffix pass-through for mid-range `lora_blocks`; everything else is marginal or applies only to fine-tune mode or rented cards.

### Smaller kept blocks: bf16 norm copies, then fp8 kept activations

**Do it (exact part), then race fp8. Expected gain: ~15% smaller kept blocks at zero error; fp8 adds an estimated 3-5% faster step on MOSS int8 1 MP, 0-2% on Krea 2 nvfp4 1 MP, perhaps 3-5% on Krea at 0.5 MP (all unmeasured end to end). Confidence: medium (0.4). Cost: exact part ~0.5 day; fp8 pack hooks a few days.**

**What it is**: Kept blocks (`checkpoint_keep_blocks`) save every activation in bf16, and Krea's RMSNorm also saves an fp32 copy of its bf16 input. Step one replaces that with a small autograd Function that saves the bf16 input plus the fp32 rstd: bit-identical, about 15% fewer bytes per kept block. Step two wraps kept blocks in `torch.autograd.graph.saved_tensors_hooks` that store large non-parameter tensors as E4M3 with 1x16 bf16 group scales (the COAT recipe) and dequantize on unpack. The calibrator then keeps more blocks at the same budget, so fewer blocks recompute.

**Evidence**:
- [COAT, ICLR 2025](https://arxiv.org/html/2410.19313v3): 1x16 FP8 groups for non-linear layers, per-tensor FP8 for linear inputs, bf16 gradients; 1.65x per-layer activation memory (Table 7). The 1.54x end-to-end figure (7B on 4 GPUs, 55.1 to 35.6 GB) is mostly FP8 optimizer states (13.1 to 3.2 GB); activations alone drop about 1.5x (25.8 to 16.9 GB) because attention stays bf16. Fine-tune 55.5 vs 55.7 bf16, VLM 62.51 vs 62.80. The 2.25x speed is batch 1 to 2 on H100s and does not apply at batch 1. Per-group scales on linears cost about 40% GEMM throughput.
- [PACT](https://arxiv.org/html/2610.25048) (Oct 2026, RTX 5080 16 GB, PyTorch 2.11, CUDA 13.2, Llama-3.2-1B/3B, Qwen2.5-1.5B): retained activations about 12.3U to 6.1U per layer, peak -47 to -51% at bs4/seq1024, 1.7-1.9x step, accuracy -0.5 to -2.6 points. Adapters are on Q/V only, which is what lets it drop the o-proj and down-proj inputs; the speed comes mostly from NVFP4 GEMMs against a bf16-LoRA baseline, and no ablation isolates the storage change.
- [Jetfire](https://arxiv.org/html/2403.12422): INT8 data flow, 32-element blocks; 1.49x activation memory (GPT-2 24 layers, bs1), 1.42x per-layer speed at hidden 4096 on an RTX 4090. Its own tables show DeiT -0.12/-0.13 and GPT-2 -0.32/-0.40, slightly worse than the text claims.
- [NVFP4 pretraining](https://arxiv.org/html/2509.25149v2) App. E.5: activations are less sensitive to forward/backward quantization inconsistency than weights (relevant to computing dA from a quantized x; says nothing about stored activations). [GACT](https://arxiv.org/abs/2206.11357): up to 8.1x (abstract only).
- Verifier's local test (torch 2.9.1, RTX 4060 Ti, quarter-width Krea-shaped block scaled x4, 2560 tokens, LoRA r16 on all 8 linears), saved bytes per token per kept block, eager / compiled: bf16 283 / 230 KB; bf16 norm copies 239 / 194 KB with bit-identical gradients; FP8 E4M3 1x16 groups 127 / 95 KB (2.2-2.4x less) with 2.5% relative error in the LoRA-A gradient; per-row FP8 111 / 82 KB with 3.2%. At 1 MP Krea (about 4600 tokens) a kept block is about 1.0 GB compiled bf16 against about 0.42 GB fp8.
- Does not transfer: every source is an LLM, ViT or GPT-2; no DiT or LoRA-on-DiT result exists. An open PyTorch issue ("[aotd] Support saved tensors hooks in aot_autograd") says hooks run eagerly after a compiled forward, so the peak inside a compiled region is not cut.
- Contrary: README.md:1735-1758 measured MOSS int8 1 MP, 6 of 42 kept, -4%; Krea 2 nvfp4 1 MP, 1 of 28 kept, no change.

**RulzTrainer today**: whole-block keep/recompute in checkpointing.py:22-60, read in krea2/model.py:278-283, qwen21/model.py:278, moss/model.py:190, fibo/model.py:324. `_calibrate_checkpointing` (trainer.py:1582-1630) measures the real allocated slope per extra kept block, so a smaller footprint is picked up with no calibrator change. `_W8A8LinearFn` / `_NVFP4LinearFn` (quant.py:802-950) save only the quantized weight; the per-token int8 x from `w8a8_forward` (quant.py:759-769) is discarded. `LoRALinear._added` (lora.py:352-358) saves the bf16 x. Krea's RMSNorm (krea2/model.py:58-59) runs `F.rms_norm(x.float(), ...)`, which saves the fp32 copy. No `saved_tensors_hooks` anywhere. Delta: the norm Function, an fp8 pack/unpack module and one option.

**Fit and risks**:
- Unpack must rebuild with `empty_strided`: a wrong stride fails `assert_size_stride` in the compiled backward and silently corrupts the SDPA backward in eager (relative error 14 in the verifier's test).
- The pack hook must skip parameters and buffers, including the int8/fp4 qdata saved by `_save_qw`, and must deduplicate (one h feeds 4 LoRAs).
- Hooks do fire inside torch.compile'd blocks. Keep 4-D SDPA tensors bf16 behind a switch (COAT and Jetfire keep attention bf16).
- Block swap: swapped blocks are never kept, so no interaction. model_split: per-GPU calibration still works. data_parallel threads: the hooks context is thread-local and entered per block, fine.
- float8_e4m3fn casts work on Ada and Blackwell (checked on the 4060 Ti); pack/unpack can be plain torch, triton-windows optional. Export, sigma grid, probes and races unaffected.
- The fp8 part is lossy (2.5-3% relative error in kept-block LoRA gradients on random data); likeness effect unmeasured.
- Defer the PACT-style reuse of the GEMM's int8/NVFP4 x: `w8a8_forward` quantizes the shared input again for each of q/k/v/gate and gate/up, so saving each linear's own copy uses more memory than the one shared bf16 tensor unless deduplicated; the convrot copy is in the Hadamard-rotated basis (needs dA = dA_rot H); NVFP4 block scales are swizzled; DoRA and `lora_dropout > 0` break it. Linear inputs are only about 30% of a block's saved bytes and o/down inputs cannot be released with RulzTrainer's targets.
- Side finding: a per-tensor cache of the quantized activation would skip about 4 of 6 quantize kernels per block in forward and recompute, about 1%, cheap.

**Implementation path**:
1. RMSNorm autograd Function in krea2/model.py:51-59 and the matching norms in qwen21, moss and fibo: save bf16 input plus fp32 rstd.
2. `kept_activation_dtype: bf16|fp8` in config.py next to `checkpoint_keep_blocks` (line 202) and the WebUI memory section.
3. New rulztrainer/actpack.py (pack: tensors of at least about 1 MB to E4M3 with 1x16 bf16 scales, dedupe by storage ptr/offset/shape/stride/_version, record strides; unpack: `empty_strided`).
4. In each family's runner (krea2 `_run` 278, qwen21 278, moss `_run_layer` 190, fibo 324): when `block.keep_activations` and fp8, call the block inside the hooks context.
5. Leave `_calibrate_checkpointing` as is; log the new "GB per extra block". Flux 3 Klein and MiniMax H3 would inherit it through their own `_run`.

**On bigger vast.ai cards**: matters most on 24-32 GB cards (RTX 5090), where only part of the blocks can be kept. By the verifier's per-block figure (Krea 1 MP: about 1.0 GB compiled bf16 vs about 0.42 GB fp8), all 28 Krea blocks would need roughly 28 GB in bf16 vs roughly 12 GB in fp8 (estimate), so fp8 could move a 32 GB card from partial keeping toward checkpointing off, whose upper bound Fizgig measured at 13-27% per step (A100 80GB, A40, RTX PRO 6000, 0.25-2 MP). On 80-141 GB cards all trained blocks already fit and the speed gain is zero; the exact bf16 norm fix still frees headroom for larger resolution or batch. Unmeasured on any rental card.

**How to test**:
1. Unit test on one Krea and one MOSS block, bf16 vs fp8, eager and compiled: saved bytes per token, LoRA gradients within tolerance (the norm change must be exact), strides survive SDPA backward.
2. Bench 45 steps, compile on, median of steps 21-45: Krea 2 nvfp4 on the 5060 Ti at 0.5 and 1 MP, MOSS int8 1 MP, Qwen 2.1 int8 0.5 MP; record the calibration line and s/step for bf16, bf16-norm-only and fp8. Stop the fp8 work if it is under 2% faster than bf16-norm-only on at least two models.
3. If it passes, race `kept_activation_dtype` bf16 vs fp8 on the same card (rotate GPUs between seeds), 1500-2000 steps, probes every 100-200 steps; metric: best smoothed ArcFace face + DINO body and wall time to reach the bf16 best.
4. Accept if fp8's best is within the seed-to-seed spread (2 seeds each) and at least 3% faster in wall time; reject if likeness falls outside that spread at any probe after warm-up.

### W4A8 forward for nvfp4 bases on the 4060 Ti

**Prototype. Expected gain: Krea 2 single-card steps on the 4060 Ti maybe 15-25% faster (estimate, unmeasured); nothing on the 5060 Ti. Confidence: medium (0.45). Cost: one triton dequant-to-int8 kernel plus a new autograd Function, a few days.**

**What it is**: The 4060 Ti has no FP4 tensor cores, so an nvfp4 base runs a fused dequantize to bf16 and a bf16 GEMM (W4A16). A W4A8 forward dequantizes the nvfp4 weight to a transient per-row int8 weight plus scale, then reuses the existing per-token int8 activation quantizer, `torch._int_mm` and the fused rescale. The backward stays the exact bf16 dequantize, so only the forward (run twice under checkpointing) speeds up.

**Evidence**:
- README.md:400-420: fused nvfp4 dequantize runs at 17.2 ms against 16.7 ms for bf16 on a layer, while W8A8 is about 2x faster than bf16. Verifier estimate: a Krea ffn forward from about 17 ms to about 3-5 ms.
- README.md:466-474: nvfp4 forward error 9.5% weight-only (Ada) and 13.4% W4A4 (Blackwell), under 1% for int8. The int8 requantize adds about 1% on top of 9.5%.
- [QServe](https://arxiv.org/abs/2405.04532): naive INT4 dequant adds 20-90% overhead; W4A8 serving 1.2x (A100) / 1.4x (L40S) over TensorRT-LLM on Llama-3-8B. LLM serving throughput at large batch, inference only.
- [SVDQuant](https://arxiv.org/abs/2411.05007): 3.0x over W4A16 on a 16 GB laptop 4090, 3.1x on a 5090 with NVFP4, inference only, against a slow W4A16 baseline.
- [torchao quant_api](https://raw.githubusercontent.com/pytorch/ao/main/torchao/quantization/quant_api.py): no CUDA int8 x int4 training kernel; the intx path targets ExecuTorch/CPU. The scout's sm89 assert note is misattributed (it sits on `Float8DynamicActivationFloat8WeightConfig`).

**RulzTrainer today**: the scout's premise "the Ada floor is int8" is false. The fused nvfp4 dequantize (quant.py:180-191) runs on GPUs without FP4 cores (README.md:462-464); auto picks nvfp4 for Krea 2 below 22 GB (krea2/loading.py:39,84) and Qwen 2.1 below 11 GB (qwen21/loading.py:39,81); Krea 2 is 13.5 GB in int8 and about 8 GB in nvfp4 (README.md:1590). QuantLinear.forward picks W8A8, then native FP4, then dequantize (quant.py:966-1015). Delta: a `_W4A8LinearFn` path on Ada.

**Fit and risks**:
- Transient int8 weight per layer is about 100 MB for a Krea ffn; fits.
- A true in-register int4 unpack GEMM is optional and risky on triton-windows; at 4096 tokens the GEMM, not the unpack, dominates.
- int4/uint3 storage formats add little over nvfp4, which RulzTrainer already reads and writes in ComfyUI's format: skip them.
- Where it matters: Krea 2 single-card on the 4060 Ti, races with one config per GPU, Qwen 2.1 below 11 GB, vast rentals without FP4. The int8-quality alternative already exists: model_split at 10.0 s/step vs nvfp4 9.2 s/step at 1 MP (README.md:1590-1596).
- Compile, data_parallel, export and probes are unaffected; forward error rises slightly (about 1%).

**Implementation path**: in quant.py add a triton kernel that dequantizes nvfp4 to per-row int8 plus scale, and `_W4A8LinearFn` that runs `rotate_quantize_act_int8` / `_int_mm` / `_rescale` (quant.py:683-830) with the existing exact bf16 backward. Select it in QuantLinear.forward for fmt nvfp4 when FP4 cores are missing and `int8_matmul` is on. Expose in the WebUI model section and README.

**On bigger vast.ai cards**: not needed. Blackwell rentals (RTX 5090, RTX PRO 6000, B200) run nvfp4 natively as W4A4, and on cards of 22 GB or more Krea 2 trains in int8 or bf16 (G7 verifier), so A100/H100/H200 never need a 4-bit base for the current models. It would matter only on a non-Blackwell rental that cannot hold an int8 base, a case nothing in the evidence requires.

**How to test**: a bench_int8.py-style layer bench (Krea shapes, 4113 tokens) plus 20-step whole-step timing on the 4060 Ti, W4A8 vs fused dequantize. Adopt if steps are at least 15% faster and the forward error is at most 10.5%. Then one sanity race on the 4060 Ti, same seed, to check the probe curve stays within seed spread of the dequantize path.

### nvfp4 likeness gap race, then a frozen recovery branch

**Race it (the gap race needs no code); Prototype the recovery branch only if a gap shows. Expected gain: unmeasured; the ceiling is the nvfp4-vs-int8 likeness gap, which RulzTrainer has never measured; a data-free SVD init gives none. Confidence: low (~0.3 across members). Cost: race 0 code; recovery adapter a new module plus a one-off 10-30 min fit per base and turbo model.**

**What it is**: When the frozen base is nvfp4, the LoRA learns against a base with about 9.5-13% forward error, while the user infers on int8/bf16. A recovery branch is a frozen low-rank BA next to each quantized linear, fitted so Qx + BAx matches Wx on real activations (CLoQ's closed form: top-r SVD of R Sigma^1/2 mapped back with Sigma^-1/2, or ostris-style layer-wise distillation). It stays on in every forward and is never exported. Merges P19 (CLoQ compensator), G7 (quant-residual init, recovery adapter), the ARA half of G15 and the residual half of G51.

**Evidence**:
- [CLoQ, TMLR 2025](https://arxiv.org/html/2501.18475v2): activation-weighted two-SVD closed form; Llama2-7B init 0.7 h / 9 GB vs LoftQ 0.6 h / 14 GB; WikiText-2 ppl 4-bit QLoRA 5.70 / LoftQ 5.24 / CLoQ 5.25; 2-bit LoftQ 7.85 / CLoQ 6.51; GSM8K 4-bit 40.6 vs 35.0; 128 x 2048-token calibration, robust 32-256. The GitHub repo has no code yet (README only), so this is paper-only.
- [LoftQ](https://arxiv.org/html/2310.08659v4): LLaMA-2-7B NF4 rank 64 ppl 5.70 to 5.24 (full-precision LoRA 5.08); 4-bit GLUE unchanged (MNLI 89.9 vs 89.9) and GSM8K 35.1 vs 35.0. [ApiQ](https://arxiv.org/html/2402.05147v3): LoftQ 35.0 vs QLoRA 35.1 on 4-bit GSM8K; gains grow only at lower bits.
- [SVDQuant](https://arxiv.org/abs/2411.05007) on FLUX.1 NVFP4 W4A4: Image Reward 0.928 to 0.937 (bf16 0.953), PSNR 20.3 to 21.4; INT4 0.908 to 0.935. Post-training only; its branch is an SVD of smoothed weights, not of the residual. [Nunchaku Krea-2-Turbo W4A4](https://huggingface.co/ModelsLab/Krea-2-Turbo-W4A4-Nunchaku): even with a rank-32 branch LPIPS 0.276 / PSNR 17.7 at 1024 px; error rises about 3x first to last block.
- [ostris ARA](https://huggingface.co/ostris/accuracy_recovery_adapters): rank-16 bf16 side-chain, 3-bit Qwen-Image the stated sweet spot, 1 MP LoRA training on 24 GB; no numbers, scripts "coming soon". [Recover-LoRA](https://arxiv.org/html/2606.04238v1): Qwen3-4B INT2 g32 gate/up, rank 32 alpha 64, KL on 10k synthetic samples, 80-95% recovery on 9 of 12 benchmarks, mixed at 3-bit; LLM only.
- [FALQON](https://arxiv.org/html/2510.24061): FP8-error SVD init with frozen A is a speed trick (commonsense 0.632 vs QLoRA 0.644), LLaMA only. [OneTrainer wiki](https://github.com/Nerogar/OneTrainer/wiki/Quantization): SVD rank 16 does not fully remove artifacts on Qwen-Image-2512.
- Verifier's local measurement (Krea-2-Raw bf16, blocks 2/5, to_q/to_v/to_out/ff.up/ff.down, RulzTrainer's quantizers, CPU): relative weight error nvfp4 0.093-0.095, int8_convrot 0.009, int8 file to nvfp4 0.094-0.095. Residual energy in the top 16/32 singular values 0.7-2.4% / 1.4-4.7%, against 0.26-1.0% / 0.5-2.1% for white noise. LoftQ T=1 init moves nvfp4 error 0.0949 to 0.0943 (r16) and 0.0938 (r32); an SVDQuant-style branch with requantized remainder gives 0.084-0.092.
- README.md:479-486: native FP4 W4A4 raised loss 0.0713 to 0.0724 vs the dequantize path; there is no measurement that an nvfp4 base lowers ArcFace likeness at all.
- Refuted member (G7, finding 38): the data-free quant-residual LoRA init (LoftQ / FALQON style, `lora_init: quant_residual`) is a measured weight-space no-op on nvfp4 and int8_convrot; do not build it. The LoftQ-into-trainable-LoRA variant in G51 is also skipped (exported delta becomes rank 2r; EMA, soup, relay, probe baselines all shift).

**RulzTrainer today**: LoRA init is kaiming/zero (lora.py:283-305); `loftq_config` at lora.py:77 is only PEFT metadata. QuantLinear holds only qdata/scale/scale2/bias (quant.py:966-1030); `quantize_module_linears` (quant.py:1038) discards W. The frozen-helper plumbing exists: `train_down`/`train_up` buffers (lora.py:236-241), applied only when extras are off (lora.py:394-398), loaded by `attach_training_adapter` (lora.py:902-940), attached to replicas (trainer.py:775, 792), streamed by block swap (blockswap.py:34-43). The Krea int8-file-to-nvfp4 load (krea2/single_file.py:148-152, moss/single_file.py:87) dequantizes and requantizes and is the natural fit hook; the int8 file is a good enough teacher (0.0954 vs 0.0949).

**Fit and risks**:
- Must stay on for probes, samples, reward and self-train renders, unlike Fizgig's training adapter; needs its own always-on buffer pair (or live in QuantLinear.forward). Not in `_added`: multiplier 0 (region prior base prediction, lora.py:382) skips it.
- The Krea Turbo probe model needs its own ARA. A weight-side branch removes only the systematic part of W4A4 activation-quantization noise, and only if fitted through the real nvfp4 forward.
- Target mismatch: a LoRA trained on Q+comp then used on the same nvfp4 file in ComfyUI may get worse; it helps only when the LoRA runs on int8/bf16/fp8 bases.
- Memory: rank 32 on every block Linear about 0.2-0.4 GB on Krea 2 (about 0.1 GB at r_c=32 per P19's estimate). Speed: a few % (1-3% of a 9 s Krea 2 nvfp4 step). Do not fuse with the fp32 LoRA via a per-layer cat under compile.
- No gain expected for int8_convrot (0.9% error, white spectrum): MOSS, Qwen 2.1 and FIBO at int8, and vast cards of 22 GB or more, are out of scope. Calibration on Windows: plain torch.linalg on CUDA, about 1.5 GB fp32 per block of second moments on CPU.
- Cheaper fix for the probe mismatch: re-score the best checkpoints on the int8 base after training.

**Implementation path** (only after the race shows a gap): new rulztrainer/recovery.py runs 32-64 forwards on cached latents at reference sigmas, hooks one block at a time, accumulates Sigma = E[x x^T] per input (shared for q/k/v/gate and ff.up/gate), computes R = W_int8_dequant - W_nvfp4_dequant, fits the damped activation-aware rank-r correction, optionally refines with a few hundred Adam steps against the real W4A4 output, saves `ara_<model>_<quant>_r<rank>.safetensors` in models/. Options `recovery_adapter: auto|<path>|""`, `recovery_rank: 16`. New always-on buffers (fix_down/fix_up) added in LoRALinear.forward in training and sampling, excluded from `export_state_dict`, soup and relay, attached to replicas and the turbo probe model. Applies to krea2 when `transformer_quant` resolves to nvfp4, later MiniMax H3.

**On bigger vast.ai cards**: not needed for current models, since on cards of 22 GB or more Krea 2 trains in int8 or bf16 (verifier) and the nvfp4 error never enters. The one remaining case is choosing nvfp4 W4A4 for speed on a Blackwell rental (5090, PRO 6000, B200); then the Step 1 gap race, run on that card, decides whether a recovery branch is worth it.

**How to test**:
- Step 0, offline: share of activation-weighted output-error energy captured by rank 16/32 (top-r sigma^2 of R Sigma^1/2 over its norm), plus ||Wx - Qx|| before/after on held-out activations. Stop if under about 20%.
- Step 1, no code: race Krea 2 nvfp4 (one card) against int8 model_split, same dataset (e.g. AZav50), 2 seeds, rank 16, 1500-2000 steps, ArcFace face + DINO body on the same turbo sampler; re-score both final checkpoints on the int8 base. Stop if int8's best ArcFace is not above nvfp4's by more than the seed-to-seed spread.
- Step 2: three-way race nvfp4 / nvfp4+ARA / int8 split; metric best-checkpoint ArcFace and area under the probe curve, re-scored on int8; also render on the plain nvfp4 base. Keep the ARA only if it closes at least 50% of the gap in both seeds at under 5% step-time cost; default it off if it wins on int8 and loses on nvfp4.

### Suffix pass-through for mid-range lora_blocks

**Race it. Expected gain: about 1.5-2x faster per step with a mid-range `lora_blocks` (Qwen 2.1 `10-14`: about 2.5 s to 1.3-1.7 s/step, estimate); 0 with the default all-block LoRA; VRAM about 0. Confidence: low (0.3). Cost: one option plus a block-loop helper, about a day.**

**What it is**: When `lora_blocks` stops before the end, the blocks after the last trained block still run forward, checkpoint recompute and backward, only to pass the gradient through. A pass-through runs them once under no_grad and rejoins with `h = h + (f(h) - h).detach()`: the forward value stays exact, the gradient through the suffix becomes the identity (straight-through). This is the only transferable piece of DiT-BlockSkip, and an online variant of it (the paper uses an offline base-model residual, approximate in both forward and gradient).

**Evidence**:
- [DiT-BlockSkip, CVPR 2026](https://arxiv.org/html/2603.20755v1) ([poster](https://cvpr.thecvf.com/virtual/2026/poster/36823), Qualcomm AI Research / KAIST): FLUX LoRA 35.99 GiB to 20.78 / 15.61 / 10.42 GiB at 30/40/50% skip; 41.67 training TFLOPs vs 9.96 + 9.94 precompute; SANA 8.35 to 3.10 GiB, 56.00 s to 41.75 s per 100 iterations; DINO 0.7324 vs 0.7194 / 0.7171 / 0.6963. One A100, batch 1 with grad accumulation 4, 500 iterations, rank 128 (FLUX). No code found.
- Confounds: the baseline trains at 512^2 and the method at 256^2 (256^2 alone costs DINO 0.7164 vs 0.7324; patch sampling alone gives 30.48 GiB at 0.7253); most of the saving is bf16 weights (baseline 22.84 GiB params + 1.34 optimizer + 11.82 activations vs 15.54 + 0.88 + 4.37 at 30%); precompute memory excluded; FLUX baseline has no checkpointing.
- Warnings for suffix skipping: naive skip without residuals gives DINO about 0.43; skipping the last 50% of blocks gives 0.4808.
- Refuted parts of the scout's design: prefix residual caching gives nothing here (fresh noise and sigma each step; the prefix already runs forward-only under `lora_blocks`); DINO-based block selection is redundant (Fizgig's identity blocks and the per-block grid already pick the range); sigma-scheduled 256^2 crops do not fit cached multi-resolution latents and contradict the 0.5-1 MP recipe.

**RulzTrainer today**: `lora_blocks` (config.py:239, parsed lora.py:406, used trainer.py:747) makes blocks before the first trained block forward-only (README.md:235-237), except when Krea's text_fusion is trained (README.md:1615). Fizgig's Qwen 2.1 recipe trains blocks 10-14 of 32 (README.md:1657), so blocks 15-31 do forward + recompute + backward (about 3-4 forward units each). `crop_augment` face+bust (config.py:376-385) is not tied to sigma. Delta: the pass-through.

**Fit and risks**:
- The suffix gradient is biased (straight-through); likeness effect unmeasured, and the paper's late-block results warn of a real cost.
- W8A8/nvfp4: cheaper, the no-grad suffix needs no backward kernels. Compile: compiled blocks are reused under no_grad. Exclude suffix blocks from checkpointing and from keep-block calibration. Block swap unchanged (weights still needed). model_split, data_parallel unchanged.
- Qwen 2.1 and Krea run joint text and image streams: rejoin both. Final norm and proj_out stay differentiable. Sampling, probes and export use the full model, unaffected. No Windows issue.
- Flux 3 Klein / MiniMax H3: the same helper applies in their block loops.

**Implementation path**: `suffix_passthrough: bool = False` in config.py, active only when `lora_blocks` leaves untrained blocks after the last trained one and no LoRA targets sit after the blocks. A shared helper in checkpointing.py (or each family's block loop) runs those blocks under `torch.no_grad` on detached inputs and rejoins per stream. Add the key to the race keys (race.py:68), a README line and a wizard note that it only helps with mid-range `lora_blocks`.

**On bigger vast.ai cards**: still applies, because it saves compute, not memory. With checkpointing off each suffix block costs forward plus backward instead of forward, recompute and backward, so the saving per suffix block shrinks by the recompute it no longer replaces, but the pass-through still reduces it to one no-grad forward; unmeasured. Under data_parallel it applies per replica.

**How to test**: Qwen 2.1 int8 0.5 MP, Fizgig recipe (rank 8, lr 2e-4, EMA 0.98, `lora_blocks: 10-14`, training adapter on), same data and seed, 2000 steps; race on vs off, one config per GPU, ArcFace face + DINOv2 body every 250 steps; metrics: best likeness vs steps and wall-clock, s/step. Drop it if the face probe trails by more than 2 points at equal step from step 1000 on; accept if at equal wall time its best is at least the baseline's and the final is within 1 point. Repeat once on Krea 2 nvfp4 with `lora_blocks` around 8-16 of 28.

### Selective activation checkpointing (after quant-linear custom ops)

**Prototype. Expected gain: 0-1.5% per step on the 16 GB cards, maybe 3-8% on 24-48 GB vast.ai cards (estimates, unmeasured); exact, no likeness change. Confidence: medium (0.6). Cost: the custom-op conversion of the quantized linears is the bulk; the policy itself is small.**

**What it is**: A third per-block state between keep and full recompute: `checkpoint(..., use_reentrant=False, context_fn=partial(create_selective_checkpoint_contexts, policy))` with a policy that MUST_SAVEs GEMM / quantized-linear / SDPA outputs and PREFER_RECOMPUTEs the pointwise chain (norms, modulation, SiLU*up, gates, RoPE). The calibrator would fill the budget with K kept blocks, then M selective blocks, then full recompute.

**Evidence**:
- [PyTorch 2.15 checkpoint docs](https://docs.pytorch.org/docs/2.15/checkpoint.html): `create_selective_checkpoint_contexts`, MUST/PREFER SAVE/RECOMPUTE/CPU_OFFLOAD, `context_fn` only with `use_reentrant=False`, early_stop default True, "Use MUST_* over PREFER_*" so torch.compile does not override. The local venv has torch 2.14.1 with the same API (venv/Lib/site-packages/torch/utils/checkpoint.py:1702).
- [Korthikanti et al.](https://arxiv.org/abs/2205.05198): 5x activation memory, >90% less recompute overhead, MFU 42.1% to 54.2% (29% faster), 530B GPT on 2240 A100s, combined with sequence parallelism. Its savings come from recomputing the unfused attention core, which fused SDPA/cuDNN attention already avoids: does not transfer.
- [PyTorch blog, Mar 2025](https://pytorch.org/blog/activation-checkpointing-techniques): the same "save mm/bmm/addmm/SDPA/_scaled_mm, PREFER_RECOMPUTE else" policy gives about 50% memory reduction on a transformer; no consumer-GPU speed numbers.
- torch 2.14's SAC sees each torch.compile region as one opaque op (`inductor_compiled_code`, gh-175258), so policy granularity is lost inside compiled fragments.
- RulzTrainer's own arithmetic: MOSS int8 1 MP kept 6 of 42 blocks and saved 0.27 s of 6.70 s (about 0.045 s per block). If a selective block costs about half a kept block and still recomputes about 1/3 of its forward, the same budget buys 12 selective blocks at about 0.03 s each: 0.36 s vs 0.27 s, about +1.3% on MOSS, about 0% on Krea 2 nvfp4.

**RulzTrainer today**: keep/recompute only: checkpointing.py:22-28 (`set_kept_blocks`), :52-61 (`choose_kept_blocks`); calibration trainer.py:1538-1625; all runners call `checkpoint(block, *args, use_reentrant=False)` without a context_fn (krea2/model.py:278-283, qwen21/model.py:276-279, moss/model.py:188-191, fibo/model.py:322-325). `_W8A8LinearFn` (quant.py:802) and `_NVFP4LinearFn` (:909) are custom autograd.Functions whose triton kernels are invisible to TorchDispatchMode. Blocks compile per block with graph breaks at every quantized linear (compiling.py:3-7). Swapped blocks must stay checkpointed (trainer.py:1544-1546).

**Fit and risks**:
- For int8 the only visible GEMM is `aten._int_mm` with an int32 output twice the bf16 bytes; the triton quantizer reruns in recompute even if the GEMM is cached. Hence the prerequisite: register the quantized linears as torch.library custom ops (`rulz::w8a8_linear`, `rulz::nvfp4_linear`, with register_autograd and register_fake). That also removes the compile graph breaks, worth doing on its own.
- The PyTorch 2.15 release notes say effectful ops with PREFER_RECOMPUTE are now saved and MUST_RECOMPUTE on an effectful op raises under compile: relevant once the linears are custom ops.
- 16 GB ceiling: Krea 2 nvfp4 keeps 1 of 28 and MOSS int8 6 of 42 (README:1748-1752); selective blocks still cost about half a kept block, so at most about 2x as many fit. No memory reduction below full checkpointing.
- model_split / data_parallel threads: dispatch modes are thread-local and SAC handles backward-thread recompute; untested with the DP workers plus compile. Compile under a dispatch mode is the riskiest part on Windows. Exact: no export or probe impact.
- Related idea from the G17 check (verifier estimate, unmeasured): keep a block's FFN activations and recompute only its attention (`keep_activations="ffn"`), about 1.15 forward saved per byte vs 1.0 for whole-block keeping.

**Implementation path**: (1) custom ops in quant.py replacing the bare autograd.Function call in QuantLinear.forward; (2) `rulz_policy` in checkpointing.py returning MUST_SAVE for those ops, `aten.mm/addmm/bmm/_scaled_mm` and `_scaled_dot_product_*_attention` / `_cudnn_attention`, PREFER_RECOMPUTE otherwise; `block.checkpoint_mode` in {keep, selective, full}; (3) pass the context_fn in each family's `_run`; (4) a third calibration measurement (3 blocks selective) and a greedy fill by seconds saved per byte; (5) `checkpoint_selective: auto|off`, default off, documented next to README line 1735.

**On bigger vast.ai cards**: this is where it pays: the verifier estimates 3-8% on 24-48 GB cards where only part of the blocks fit kept. On 80-141 GB cards with every block kept, or checkpointing off, it does nothing. The custom-op prerequisite removes the compile graph breaks at every quantized linear (compiling.py:3-7), which helps on every card.

**How to test**: unit test that LoRA grads after one step match across full / keep / selective (int8_convrot and nvfp4). Bench 20 steps at 1 MP and 0.5 MP on each card separately for MOSS int8, Krea 2 nvfp4, Krea 2 int8 split and Qwen 2.1 int8, compile on and off: median s/step (steps 6-20), peak reserved, `num_alloc_retries`, the K/M log line; rerun on a 24-48 GB vast card. Ship as auto only if s/step improves at least 3% on one 16 GB config or 5% on a larger card with no new OOM or alloc retries. One sanity race (two racers, same seed, 300 steps): probe curves must coincide.

### Activation offload to pinned RAM

**Prototype (5060 Ti and 24-32 GB rentals, low priority). Expected gain: plausibly 5-15% per step if the copies hide (MOSS int8 about 5 s to 4.4-4.7 s, Krea 2 nvfp4 about 9 s to 8-8.5 s on the 5060 Ti); near 0 or negative on the 4060 Ti at 1 MP; the VRAM-only variant buys about one more kept block (0-1%). All unmeasured. Confidence: low (0.35). Cost: medium (new offload module, stream scheduling, calibration).**

**What it is**: Blocks that do not fit the keep budget could, instead of recomputing, copy their saved tensors D2H to pinned RAM on a side stream during the forward and prefetch them back during the backward (torchtune OffloadActivations pattern, or PyTorch's MUST_CPU_OFFLOAD policy). Bit-exact.

**Evidence**:
- [torchtune _activation_offloading.py](https://raw.githubusercontent.com/pytorch/torchtune/main/torchtune/training/_activation_offloading.py): saved_tensors_hooks, 1024 B minimum, stash of 5, skips Parameters/Buffers, warns above 60% RAM, no figures. The [torchtune tutorial](https://meta-pytorch.org/torchtune/main/tutorials/memory_optimizations.html) recommends offload only together with checkpointing and says it may decrease speed.
- [PyTorch 2.15 autograd docs](https://docs.pytorch.org/docs/2.15/autograd.html) (`save_on_cpu`), [checkpoint docs](https://docs.pytorch.org/docs/2.15/checkpoint.html) (MUST_CPU_OFFLOAD), [2.15 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.15.0).
- Every trainer implementation offloads checkpointed tensors and still recomputes, a VRAM lever, not a speed lever: [OneTrainer RamOffloading.md](https://raw.githubusercontent.com/Nerogar/OneTrainer/master/docs/RamOffloading.md) (no numbers), [musubi](https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/docs/hunyuan_video.md) `--gradient_checkpointing_cpu_offload`, [SimpleTuner](https://raw.githubusercontent.com/bghira/SimpleTuner/main/documentation/OPTIONS.md) unsloth backends, [diffusion-pipe](https://raw.githubusercontent.com/tdrussell/diffusion-pipe/main/docs/minimax_h3_notes.md) "unsloth". No source measures offload replacing recompute.
- [Fizgig NEWS v8.9.5](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/NEWS.md): 13-27% per step with checkpointing off (2-21% last blocks only), measured on A100 80GB, A40 and RTX PRO 6000 at 0.25-2 MP: an upper bound only.
- Sizes from RulzTrainer calibration logs: 0.25-0.29 GB per extra Krea 2 block at 1046 tokens (about 1.1 GB per block at 1 MP, roughly 30 GB per step each way) and 0.40 GB per Qwen 2.1 block at 2355 tokens; the scouts underestimated 3-4x. Per block about 45 ms on the 5060 Ti (PCIe 5.0 x8) and 90 ms on the 4060 Ti (PCIe 4.0 x8) against a ~70 ms nvfp4 block forward; per step about 1.2 s per direction on the 5060 Ti (hides behind a ~2 s forward) and 2.5 s on the 4060 Ti (does not hide).
- Refuted member (G12, finding 123, KohakuMoss larger micro-batch under block swap): budgets misread (8192 vs 4096 tokens), tok/s flat (5050 vs 4975) as the budget grows; irrelevant at batch 1.

**RulzTrainer today**: checkpointing.py (lines 1-67) has only keep/recompute; blockswap.py moves weights H2D only; no `save_on_cpu`, `saved_tensors_hooks` or activation offload anywhere. README.md:1752-1757 measures -4% (MOSS, 6 of 42 kept) and 0-2% (Krea, 7 of 28) from removing recompute on 16 GB cards.

**Fit and risks**:
- A torchtune-style hook skips only nn.Parameter/Buffer; RulzTrainer's int8/fp4 qdata saved by `_save_qw` (quant.py:789-799) are plain tensors, so a naive hook copies every block's weights D2H. Exclude by data_ptr.
- Never offload a block_swap block (a non-recomputed swapped block would pin its weights through the saved qw). Block swap already excludes compile and data_parallel.
- Compile: test whether eager hooks around a compiled block catch AOTAutograd's saved tensors; otherwise use a context_fn with MUST_CPU_OFFLOAD (all four runners already use `use_reentrant=False`).
- RAM: the machine has 96 GB; 30 GB pinned per GPU fits on one GPU, tight for two plus block swap plus caches. Windows has no expandable segments and large cudaHostAlloc pools are slow: allocate once and reuse per bucket. data_parallel doubles host traffic.
- Probes and samples run without grad, so hooks never fire; no export or sigma-grid impact.

**Implementation path**: new rulztrainer/offload.py (saved_tensors_hooks modelled on torchtune; exclude param/buffer data_ptrs; 1 MB minimum; per-shape pinned buffers reused across steps with an LRU cap like SimpleTuner's max_buckets; dedicated D2H/H2D stream with events; prefetch block i-1 when block i's backward begins). A third flag `block.offload_activations` in checkpointing.py; runners (krea2/model.py:281, qwen21:278, moss:190, fibo:324) run flagged blocks without checkpoint() inside the context. Extend `_calibrate_checkpointing` (trainer.py:1574) to time M = 0/4/8/all offload blocks on the largest bucket and keep the fastest; disable on a GPU whose D2H rate is too slow. Option `activation_offload: auto|off|N`, 5060 Ti first.

**On bigger vast.ai cards**: on 80-141 GB cards turn checkpointing off instead; Fizgig's 13-27% upper bound was measured on exactly that class (A100 80GB, A40, RTX PRO 6000) and offload has no role. On 24-32 GB cards it is a possible bridge to no-recompute; Linux helps because train.py:16-18 sets expandable_segments there, easing the pinned-pool fragmentation concern. Rental PCIe rates are not measured in the evidence, and multi-GPU data_parallel adds host traffic per GPU.

**How to test**: scripts/bench.py on the 5060 Ti at 1 MP, 45 steps, compiled, median of steps 21-45, off vs all, Krea 2 nvfp4 and MOSS int8; repeat on the 4060 Ti and in data_parallel. Loss must be bit-identical over 20 steps. Only if the 5060 Ti gains at least 5% with no VRAM growth, run a 1500-step race (same seed) and check probe curves overlay and wall-clock to the best checkpoint drops. Drop it for a model and card when the median gain is under 3% or the profiler shows GPU idle gaps on the copy stream. Cheapest first trial: Qwen 2.1 / MOSS at 0.5 MP on the 5060 Ti.

### Per-block format ramp for the nvfp4 base

**Race it (after an offline error check). Expected gain: unmeasured for training; costs about 0.9 GB for the last 4 of 28 Krea blocks in int8 and a little speed. Confidence: low (0.35). Cost: small to medium (callable quant filter plus re-calibration).**

**What it is**: Instead of one format for every block, give the first or last N blocks a more accurate format (int8_convrot) while the rest stay nvfp4, on the theory that late Krea blocks carry the largest activation outliers. The module-name half of the scout's idea (bf16 islands for modulation, text fusion, first/last) already exists.

**Evidence**:
- [ai-toolkit krea2.py](https://raw.githubusercontent.com/ostris/ai-toolkit/main/extensions_built_in/diffusion_models/krea2/krea2.py) keeps first, tmlp*, tproj*, txtmlp*, txtfusion.projector and last* in full precision; [musubi krea2_utils.py](https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/src/musubi_tuner/krea2/krea2_utils.py) targets `blocks.` and excludes `mod.`, `norm`, `txtfusion`.
- [Comfy-Org/Krea-2 discussion #6](https://huggingface.co/Comfy-Org/Krea-2/discussions/6): Kijai says not all Krea layers can use NVFP4 matmuls because of quality loss; a mixed BF16/MXFP8/NVFP4 build with blocks 0-2 and 24-26 shielded ran 1.35 s/it vs 2.21 s/it for the Comfy NVFP4 file (the scout misquoted this as "vs 17 s"); time split GEMM 50 / attention 41 / norm 8.8%; int8 convrot LPIPS 0.066 / IR -0.010 vs fp8 scaled 0.120 / -0.024 (GPU not named).
- [Nunchaku Krea-2-Turbo W4A4](https://huggingface.co/ModelsLab/Krea-2-Turbo-W4A4-Nunchaku): error rises about 3x first to last block (no per-block numbers); LPIPS 0.276 / PSNR 17.7 even with a rank-32 branch.
- [EntroPack MOSS card](https://huggingface.co/DiffSynth-Studio/MOSS-Image-EntroPack): 3.5-bit (2.24 GB DiT), sensitive layers at 5.0 bpp, INT8 level W8A8 ConvRot with sensitive linears at 8.5 bits, LoRA scripts, no PSNR. It is entropy-coded storage dequantized for compute, not W4A4 compute, and its named sensitive layers (embeddings, time embeddings, modulation, final) are already bf16 in RulzTrainer. [entropack GitHub](https://github.com/modelscope/entropack): Z-Image 4 bpp 7.18% relative L2, 544.7 vs 505.7 ms on H20; sensitivity method undisclosed.
- Misattributed: the KohakuMoss massive-channels page never discusses quantization. [OneTrainer PR #1568](https://github.com/Nerogar/OneTrainer/pull/1568): "int8 makes some noise" is real; the INT_W8A8 preset claim is not on the page.
- All fidelity numbers are inference-only; none measures LoRA likeness on a quantized base.

**RulzTrainer today**: QUANT_FILTER quantizes only main blocks (krea2/loading.py:35-36, qwen21/loading.py:34-36, moss/single_file.py:115-117, fibo/loading.py:30-32); `min_dim=1024` skips small Linears (quant.py:247). `install_state` (moss/single_file.py:64-90) keeps each tensor's own format under auto and QuantLinear is per-layer, so a mixed model already runs. Missing: per-layer format assignment (`transformer_quant` config.py:199 and `quantize_module_linears` quant.py:1035 take one target plus a boolean filter).

**Fit and risks**: int8 W8A8 runs native on the 5060 Ti next to nvfp4; on the 4060 Ti nvfp4 dequantizes, so mixing only trades memory. Compile: a few extra recompiles, well under the limit of 64 (compiling.py:22). Block swap and model_split move tensors generically (unverified for mixed formats); per-block byte budgets and keep-block calibration must re-run. MOSS and Qwen 2.1 already fit at int8 on 16 GB, so the VRAM saving there has no use; a speed gain from moving most blocks to nvfp4 with sensitive writers int8 is unmeasured (verifier guess: 10-20% of the GEMM half of the step).

**Implementation path**: let `install_state` and `quantize_module_linears` accept a callable name -> format|None; per-family block-index parsers (`transformer_blocks.N.` for Krea/Qwen 2.1, `layers.N.` for MOSS); options `quant_tail_blocks` (0), `quant_head_blocks` (0), `quant_edge_format` (int8_convrot); re-run keep-block and block-swap budgets on real per-block sizes. The optional frozen residual branch (`quant_residual_rank`) is covered by the recovery subsection above.

**On bigger vast.ai cards**: no use as a VRAM measure (Krea 2 fits in int8 or bf16 at 22 GB and up). On Blackwell rentals (5090, PRO 6000, B200) the same per-layer format hook could serve speed instead: nvfp4 for most blocks with the sensitive ones int8 (verifier guess for MOSS/Qwen 2.1 on Blackwell: 10-20% of the GEMM half of the step, unmeasured; Kijai reported Krea NVFP4 about 15% faster than FP8 at inference on a 5090 after the re-upload).

**How to test**: Stage 0 offline on 20 cached Krea 2 batches at sigmas 0.2/0.5/0.8: relative L2 error of the final velocity and of each block's output vs bf16, for all-nvfp4, nvfp4 + tail 4 int8_convrot, nvfp4 + head 3 / tail 3, nvfp4 + rank-32 residual; proceed only for a variant that cuts final velocity error by at least 25%. Stage 1 race on the 5060 Ti, same character set at 0.5 MP, rank 16, 1500 steps, probes every 250 (ArcFace primary, DINOv2 body); drop any variant above 15.5 GB or more than 10% slower; keep only if best ArcFace beats all-nvfp4 by at least 0.01 on two seeds or reaches its best 20% sooner.

### Fine-tune mode window memory: Adafactor step in backward, CPU offload, CPU EMA shadow

**Prototype (fine-tune mode only). Expected gain: an in-backward Adafactor step takes 12 to 8 B/param (with anchor), about 1.5x larger windows (MOSS 0.25 MP about 10 to 6-7 windows); full CPU offload of master, grad and state leaves about 4 B/param on GPU, about 3x larger windows (Krea 2 1 MP maybe 12 to 4-5); LoRA runs gain 0.1-0.4 GB from a CPU EMA shadow. Step-time cost unmeasured here. Confidence: medium (0.6). Cost: medium.**

**What it is**: In fine-tune mode the fp32 window weight and its gradient dominate (8 of 10 B/param with adafactor). Running each tensor's Adafactor step inside `register_post_accumulate_grad_hook` frees its gradient at once; a further option keeps a bf16 GPU weight and moves the fp32 master and state to pinned CPU with a worker-thread step. For LoRA runs, a pinned CPU EMA shadow takes the fp32 copy off the GPU.

**Evidence**:
- [torchao optim README](https://github.com/pytorch/ao/tree/main/torchao/optim) (RTX 4070 Ti SUPER 16 GB): ViT-giant 1.1B eager bf16 with checkpointing, batch 32: none 9.82 GB / 1.27 s/it, offload 5.24 / 1.52, offload_gradients 4.01 / 1.53, DeepSpeed untuned 6.85 / 3.13. ViT-H 630M compiled batch 8: fused AdamW 12.23 GB / 41.9 imgs/s, AdamW8bit 8.33 / 42.5, AdamWFp8 8.33 / 43.2, AdamW4bit 7.72 / 42.4. Caveats: no LR scheduler, no grad clipping, offload_gradients incompatible with accumulation, RAM about 2x model + states, the CPU step is often the bottleneck.
- [torchtune tutorial](https://meta-pytorch.org/torchtune/main/tutorials/memory_optimizations.html): optimizer-in-backward removes the gradient buffer, needs accumulation 1, little effect on LoRA; puts CPU-offload RAM at about 4x model size.
- [OneTrainer RamOffloading.md](https://github.com/Nerogar/OneTrainer/blob/master/docs/RamOffloading.md): fused back pass matters only for full fine-tunes; it does not mention EMA, so the "OneTrainer pattern" for a CPU EMA is unsupported.
- [Fizgig FINETUNE.md](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/FINETUNE.md): H3 one matrix type across all 50 blocks, 4/5/9 windows on 32/24/16 GB; optimizer-in-backward saves 5.2 GB; H3 peaks 8.8-12.3 GB on 16 GB at about 1.5x step time, 19.1-21.5 GB on 24 GB; RAM 23-38 GB; lr 3e-5 for H3 ("1e-4 destroys"), 1e-5 for Krea 2 / Qwen 2.1; Krea 2 0.25 MP 8 windows about 5.1 s/step, 1 MP 12 windows 11.4 GB about 9.8 s/step. [MINIMAX_H3.md](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/MINIMAX_H3.md): full fine-tune needs the pruned int8 file.
- All measured gains are full fine-tunes; none carries over to rank 8-16 LoRA.

**RulzTrainer today**: member 139's novelty claim is wrong: finetune.py already packs per-component windows across all blocks (finetune.py:10-12, 97-165), uses Adafactor and documents Fizgig's learning rates (finetune.py:33-35). `BYTES_PER_PARAM` (finetune.py:54): adafactor 10, adamw8bit 12, adamw 18, automagic3 18, automagic4 12; anchor +2 (trainer.py:1297). MOSS int8 0.25 MP on the 4060 Ti: 10 windows, 13.1 GB, about 1.4 s/step (README.md:1374). Optimizer rebuilt per window (trainer.py:1321-1336). automagic3 already has a fused in-backward mode (automagic3.py:83-88, 231) that trainer.py:283 forces off. EMA shadow lives on the weight's device (ema.py:20).

**Fit and risks**: an in-backward step breaks global `clip_grad_norm_` (trainer.py:1133; use per-tensor RMS clipping), the OOM retry that re-runs backward (trainer.py:1125-1128; raise instead), accumulation (must be 1), per-image lr (write `step_lr_mult` into the groups before backward), the anchor (`pull_to_base`, finetune.py:315, must run in the hook) and the nan-skip. Compile, data_parallel and races are already excluded from fine-tune (finetune.py:31); W8A8/nvfp4 trunks untouched (windows are TunedLinear in bf16). Offload: 1-3 GB over PCIe per step (about 0.2-0.4 s on the x8 links), a CPU Adafactor over 1-2 B params likely 1-3 s, RAM +4-12 GB on top of 15-36 GB of deltas and base copies, tight for Krea 2 on 64 GB. Prefer a native implementation over a torchao dependency on Windows.

**Implementation path**: `finetune_step_in_backward` (default off, accumulation 1 only): in `_finetune_window` build per-parameter Adafactor, hook clips by RMS, applies lr x `step_lr_mult`, steps, pulls to base, sets grad None; skip clip/step/pull in the loop; adafactor at 6 B/param in this mode. Optional `finetune_offload` (bf16 GPU weight, fp32 pinned master + state, D2H grad on a side stream, worker-thread step, H2D before next forward; budget 2 + anchor). For LoRA, `ema_device: cpu|auto` in config.py and ema.py. Skip torchao 4-bit/FP8 states (adamw8bit covers most of it) and reg images at lr x0.2 (no reg-image dataset type; isolation is barred from fine-tune).

**On bigger vast.ai cards**: windows grow with VRAM (Fizgig H3: 9 windows on 16 GB, 5 on 24 GB, 4 on 32 GB; a naive Krea 2 full fine-tune needs about 78 GB per Fizgig), so on 96-141 GB cards (RTX PRO 6000, H200) few windows, possibly one, cover the model and CPU offload loses its point, while the in-backward step still drops the 4 B/param gradient. Fine-tune mode excludes data_parallel today (finetune.py:31), so multi-GPU NCCL would need new work.

**How to test**: numerical check on one MOSS window, clipping off: in-backward deltas within 1e-6 max abs of the normal step. Then MOSS int8 0.25 MP on the 4060 Ti: (A) current, (B) step-in-backward, (C) offload; log windows per cycle, peak VRAM, s/step, CPU-step time, RAM peak. Quality at equal wall-clock (e.g. 2 h or 3 cycles of A), extract at rank 64 per cycle, ArcFace + DINOv2 probes; accept B/C if best probe at equal wall-clock is at least A's minus about 0.005 ArcFace with no OOM and under 25% s/step overhead. Repeat on Krea 2 nvfp4 1 MP on the 5060 Ti. For `ema_device: cpu`: identical probe scores, VRAM drops by the shadow, s/step up under 2%.

### Frozen down-projection (LoRA-FA / train only B)

**Race it (only for configs at the OOM edge). Expected gain: VRAM at peak, rank 16: Krea 2 nvfp4 1 MP about -0.3 GB activations and -0.15 GB A grads (about 3% of the 14 GB peak); MOSS int8 1 MP about -1.0 GB; Qwen 2.1 int8 0.5 MP about -1.1 GB; optimizer saving vanishes if the rank is doubled. Speed 0% directly, 0-1% via extra kept blocks. Likeness: likely a drop at rank 8-16. Confidence: low (0.35-0.4). Cost: about 60 lines.**

**What it is**: With A (lora_down) frozen, grad(B) needs only the r-dim code A x, so autograd keeps [tokens, r] instead of [tokens, d_in] for the adapter branch, and adapter grads and optimizer state halve at the same rank. PaCA is the extreme form (A = one-hot column selection).

**Evidence**:
- [PaCA, ICLR 2025](https://arxiv.org/abs/2503.01905) ([code](https://github.com/WooSunghyeon/paca), MIT, LLM-only PEFT fork): abstract -22% time, -16% memory, +16% throughput on A100/Gaudi2; body gives 19% time saving, per-model memory 23 to 20 GB, 27 to 23, 40 to 35 (12-15%), bf16 LLaMA at batch 8-16 x 512-768, no checkpointing. At equal rank 8 it is slightly below LoRA (50.4 vs 50.6, 56.8 vs 57.0); quantized QPaCA falls to 8% time saving.
- [Asymmetry in LoRA, ICML 2024](https://arxiv.org/html/2402.16842): RoBERTa B-only r16 87.5 vs LoRA r8 87.2 (gain mostly CoLA; B-only r8 loses on MNLI, MRPC, RTE). Table 4's "B-only r=64 at 0.12%" is almost certainly a typo (same parameter count as LoRA r32); the genuinely half-parameter row is B-only r32 45.36 vs 44.76, with Social Sciences 51.81 to 46.88. No memory, speed or convergence data.
- [LoRA-FA](https://arxiv.org/abs/2308.03303) is the original frozen-A method (LLM-only). kohya sd-scripts `networks/lora_fa.py` (PR #755, Aug 2023) recorded 3-7% speedup on SDXL, a freezing bug, VRAM "not much lower" after the fix, no quality data, no later uptake.
- Corrections: Krea 2 d_in is 6144 (krea2/model.py:229-235), not 3072; the "~1/96 per adapted linear" is per adapter, not per block (q/k/v share one input; SDPA and norms save their own tensors), so frozen A removes about 20% of a block's activations.

**RulzTrainer today**: lora.py:212-213 creates both factors, lora.py:230-231 sets requires_grad on all; `_added` (lora.py:349-352) saves x for grad(A). QuantLinear saves only qdata/scales (quant.py:789-793, 802-808, 915-918, 952-957), so the LoRA down is the only holder of x. README.md:1746-1754: Krea 2 nvfp4 keeps 1 of 28 blocks at a 14 GB peak at 1 MP, and the whole keep feature gains 0-4%.

**Fit and risks**: no interaction with int8 ConvRot / nvfp4 (adapter outside the base, unrotated basis) or compile. DoRA (lora.py:374-383) saves x anyway: refuse or warn. `lora_param_groups` (lora.py:555-568) filters one list; LoRA+ `plus_ratio` (line 564) becomes moot; `mirror_lora` (lora.py:533-551) copies A; EMA (ema.py:27-28) and soup (soup.py:9-11) are fine. Likeness risk: a fixed random 16-dim projection of a 4096-6144-dim input has JL distortion about 25% at r=16. Refuted members: the "select" (PaCA) mode is too narrow for identity (16 of 6144 features), and the proposed MiniMax H3 "select by default at <=16 GB" has no diffusion support.

**Implementation path**: `lora_down_mode: train|frozen` (default train) in config.py, LoRA only; LoRALinear sets `lora_down.weight.requires_grad_(mode == "train")` after init/set_kind/load, keeps kaiming init, applies dropout after the down when frozen; param-group builders filter `requires_grad`; add to RACE_KEYS "LoRA rank & layers" (race.py:49-81). Test file in the test_oft_dora.py style: grad None on A, saved-tensor bytes via `saved_tensors_hooks`, export/load round trip, ComfyUI merge equality.

**On bigger vast.ai cards**: no value for the supported models: VRAM is not the limit there and the likeness risk stays. The only conceivable case is a future 33B MiniMax H3 at its memory edge, for which the verifiers found no diffusion evidence.

**How to test**: race train r16 vs frozen r32 (and frozen r16) on Qwen 2.1 int8 0.5 MP, one character set, 1000 steps, probes every 100; compare ArcFace / DINO curves and the step each reaches its plateau; read the calibration line and the per-step peak (trainer.py:1187/1206). Only if frozen r32 matches the likeness curve, try Krea 2 1 MP with identity loss on to see whether the freed ~0.5 GB lets it fit. Do not advertise a speed gain.

### bf16 adapter weights with stochastic rounding

**Skip-ish. Expected gain: about 100-300 MB VRAM (1-2%), speed under 1%. Confidence: medium (0.65). Cost: small (dtype option, optimizer restriction, data_parallel SR seeding).**

**What it is**: Store lora_down/lora_up in bf16 and let the optimizer write back with stochastic rounding (fp32 temporary plus mantissa dither), removing the 4-byte master copy, half the grad bytes and optionally half the EMA shadow.

**Evidence**:
- [Ozkara, Yu, Park, AISTATS 2025](https://arxiv.org/html/2502.20566v1): bf16+SR AdamW, no fp32 master; ppl 14.07 vs 14.45 (350M), 10.05 vs 10.11 (6.7B); 1.07-1.54x vs O1 mixed precision but only 1.03-1.06x vs O2; memory -21% / -30% vs O1, only 11-13% at 1.3B/2.7B; A100-40GB; SR-tuned lr 2-4x default; stagnation warning when the quantization gap dominates. LLM pretraining only, no fine-tuning, LoRA or diffusion. Nothing about throughput transfers to a 10-30M-parameter adapter on a quantized base.
- Community evidence is anecdotal: OneTrainer users compare bf16+SR against plain bf16 LoRA, not fp32.

**RulzTrainer today**: the SR write-back already exists in automagic3 (docstring lines 92-97, `_sr_truncate` line 269, `_stochastic_round` 284, `_stochastic_copy_` 348, bf16 branch 562-569) and automagic4. What blocks it is lora.py:214-215 forcing fp32. EMA shadow is fp32 (ema.py:95); README says the LoRA trains in fp32 (README.md:376) and EMA costs one fp32 copy (README.md:591). Export already defaults to bf16 (lora.py:584, 634).

**Fit and risks**: no interaction with W8A8/nvfp4, block swap, model_split; one recompile for the dtype change. data_parallel: `_sr_truncate` uses each device's RNG, so replicas drift unless SR noise is seeded per step or rank 0 broadcasts. torch adamw would round-to-nearest and lose updates; bitsandbytes adamw8bit has no SR. At lr about 1e-4 and weights 1e-3 to 1e-2, SR mostly adds noise; stagnation appears late at low lr or after probe-driven lr cuts; the paper's "raise lr 2-4x" conflicts with probe lr control. Keep LoKr, OFT and DoRA in fp32.

**Implementation path**: `adapter_dtype: fp32|bf16` (default fp32); cast in LoRALinear.__init__ only for kind lora without DoRA; allow only with automagic3/4 in `build_optimizer` (around trainer.py:259) or a small SRAdamW using `_stochastic_copy_`; keep the EMA shadow fp32 (a bf16 shadow at decay 0.98 needs SR too); fix data_parallel SR seeding; update README lines 376 and 591 and the wiki.

**On bigger vast.ai cards**: no value: 100-300 MB is noise on 32-141 GB cards, and under multi-GPU data_parallel the per-device SR noise makes replica drift a bigger concern, not a smaller one.

**How to test**: race on Qwen 2.1 int8 0.5 MP, rank 16, automagic3, 2000 steps: fp32 vs bf16+SR vs bf16 round-to-nearest (negative control), 2 seeds, probes every 250, peak VRAM, s/step, fraction of adapter params unchanged per step late in training; repeat on MOSS int8 with cosine decay to low lr. Adopt only if best face score is within the fp32 seed spread and the saving is at least 150 MB or one more kept block; drop if the unchanged fraction climbs above about 50%.

### Block swap flat buffers and pinned-register option

**Skip-ish (the one-line pinned allocator option: Do it). Expected gain: about 0 on the user's 16 GB defaults, which never swap; perhaps a few hundred MB less reserved slack on Windows in swapped runs and seconds off startup pinning (unmeasured). Confidence: low (0.3). Cost: one line for the allocator option; a slot refactor is medium and must handle the backward.**

**What it is**: Pack each swapped block's frozen tensors into one pinned uint8 buffer, copy it in one transfer into a preallocated GPU slot and rebind views, instead of one async copy per tensor into fresh allocations; plus `pinned_use_cuda_host_register` / `pinned_num_register_threads` to speed up initial pinning.

**Evidence**:
- [RamTorch](https://github.com/lodestone-rock/RamTorch): GPU slot window, loader thread, Belady eviction; its 0.4% overhead is a pipeline simulator result, not a measured run, no hardware named. [OneTrainer RamOffloading.md](https://github.com/Nerogar/OneTrainer/blob/master/docs/RamOffloading.md): ring-buffer allocator for layer weights, no numbers.
- [PyTorch 2.15 CUDA notes](https://docs.pytorch.org/docs/2.15/notes/cuda.html): pinned allocator options (8 register threads suggested), no platform restriction stated. [GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html) is Linux-only; `cudaMemcpyBatchAsync` is not exposed by PyTorch.
- README measurement: Krea int8 on the 4060 Ti with 14 of 28 blocks swapped ran 2.6 s/step vs 2.2-2.8 s unswapped; transfers already hide. Per-tensor launch overhead is well under 1 ms per block.

**RulzTrainer today**: blockswap.py:1-178 already has pinned copies (79-81), a side stream (61, 126-129), events with max_inflight (113-116, 157-162), next-block prefetch (132-137, 152-154), record_stream (144-146); `empty_cache` only at setup and reoffload (88-89, 108-109), not per block as the scout claimed. train.py:16-18 sets only `expandable_segments`, and only on Linux. Block swap is single-GPU (config.py:589-598) and turns compile off (trainer.py:880-881); the wizard (webui.py:2920-2945) recommends it only when model GB + 4.5 GB exceeds VRAM.

**Fit and risks**: show-stopper in the proposed two-slot design: QuantLinear saves qdata and scale for the backward (quant.py:789-793), so freeing a slot on the forward post-hook overwrites weights the recompute's backward still reads, giving silently wrong gradients. A slot design needs 3 slots and reuse gated on the block's backward completing (tensor hook on the input grad plus an event). Mixed dtypes need aligned byte views.

**Implementation path**: first, in train.py append `pinned_use_cuda_host_register:True,pinned_num_register_threads:8` to `PYTORCH_ALLOC_CONF` when `block_swap > 0` and no allocator config is set, on all platforms (verify Windows accepts the keys). Only if wanted: 3-slot packed buffers and `block_swap_prefetch: 1|2`, with gradient-equality tests in tests/test_blockswap.py.

**On bigger vast.ai cards**: nothing to gain for current models, which do not swap at 32 GB and up (Krea 2 int8 is 13.5 GB). GPUDirect Storage is Linux-only and so available on vast, but the scout notes weights fit in RAM there anyway.

**How to test**: Krea 2 int8 on each card at 0.5 and 1 MP, block_swap 14 and 4, 200 steps: mean s/step (steps 20-200), max reserved/allocated, kept-block count, startup pinning time. Gradients and LoRA weights after 50 steps must match within bf16 noise. Adopt the slot refactor only if s/step improves at least 2% or peak reserved drops at least 300 MB; otherwise keep only the allocator option.

### Sequence-chunked feed-forward (Mini-Sequence Transformer)

**Skip-ish. Expected gain: likely a net slowdown in kept blocks; nested chunk recompute in checkpointed blocks lowers the transient peak about 0.3-0.5 GB (Krea 2 1 MP) at roughly +15-25% step time; best case one more kept block, 1-3%. Confidence: low (0.25). Cost: about 40 lines.**

**What it is**: Split the FFN along the sequence into mini-sequences, each under its own checkpoint, so the [S, I] intermediates exist for one chunk at a time. Exact for per-token FFNs.

**Evidence**:
- [Mini-Sequence Transformer](https://arxiv.org/html/2407.15892v4): Llama3-8B 4k on one A100-80GB, about 67 GB to 47 GB (Table 11 says 48); 3271.42 vs 3194.90 TFLOPS (about 2.3% lower, batch 2, 8k); MLP memory at 80k tokens 14.72 to 11.66 GB (M=8). Most of the headline comes from the LM head (128k vocabulary, 59.92 to 9.12 GB); the MLP-only reduction is 20.8%. A DiT has no LM head. The paper also has internal inconsistencies (47 vs 48 GB, 12x vs 14x for Mistral, MLP width 13824 vs 14336).

**RulzTrainer today**: FFNs compute full intermediates in one shot (krea2/model.py:121-122, qwen21/model.py:145-146, moss/model.py:119-120). The scout's claim that quantizer copies are saved is wrong: quant.py saves only weights via `_save_qw`. A kept Krea block's FFN saves about 4 x S x I x 2 bytes, about 550 MB at 1 MP (55-60% of the block), and the FFN is about 67% of the block forward, so per-chunk recompute in a kept block nearly cancels the reason to keep it. Chunking without recompute is free in the no_grad extra forwards but those are not where the peak sits.

**On bigger vast.ai cards**: still marginal. With checkpointing off, per-chunk recompute adds back the recompute the big card was meant to avoid; chunking would help only to fit a larger resolution or batch without checkpointing, which nothing in the evidence measures (MsT's own setting, 80 GB A100 at 4k-80k tokens, gains mostly from an LM head a DiT lacks).

**How to test** (only if pursued): on the 5060 Ti, Krea 2 nvfp4 1 MP and Qwen 2.1 int8 0.5 MP, `ffn_chunks` 1 vs 4 with and without per-chunk recompute, 50 steps compiled: peak VRAM, auto kept blocks, s/step; exactness check under 1e-3 relative in bf16 (verify nvfp4's per-tensor global activation scale per chunk); race only if s/step improves. The better follow-up is the keep-FFN / recompute-attention policy noted in the selective checkpointing subsection.

### APOLLO optimizer states for fine-tune windows

**Skip-ish. Expected gain: no window gain over the current fine-tune default (adafactor, 10 B/param); about 17% smaller windows only vs adamw8bit; 1.6-1.7x only vs adamw/automagic3, which nobody uses for the fine-tune here; step time under 2% change. Confidence: low (0.25-0.4). Cost: about 150 lines in-house (CC-BY-NC code cannot be vendored).**

**What it is**: APOLLO keeps Adam moments on a random low-rank projection of the gradient and applies a channel-wise (or tensor-wise for APOLLO-Mini) scale to the raw gradient: SGD-like memory with Adam-like scaling, no SVD.

**Evidence**:
- [APOLLO project page](https://zhuhanqing.github.io/APOLLO), [arXiv 2412.05270](https://arxiv.org/abs/2412.05270), MLSys 2025: LLaMA-7B C4 pretraining optimizer memory 8-bit Adam 13 GB, 8-bit GaLore 4.9 GB, APOLLO 1.6 GB, Mini 0.0 GB; ppl at 150K steps 14.61 / 14.65 / 13.02 / 13.09; up to 3x throughput via 4x batch (8xA100 on the page, 4xA100 in the README).
- [GitHub](https://github.com/zhuhanqing/APOLLO): defaults rank 256 / scale 1 / channel and rank 1 / scale 128 / tensor (Mini); the MMLU 65.03 vs GaLore 64.96 run used rank 128 / scale 32, not defaults; README says scale needs tuning; open issue #14 (scheduler warmup doubled), #5 (FSDP explosion, closed), #3 (bitsandbytes pin). LLM trainers only; no diffusion use.

**RulzTrainer today**: the premise (16 B/param baseline, no low-state optimizer) is wrong: `BYTES_PER_PARAM` (finetune.py:54) budgets adafactor 10 and automagic4 12; Adafactor is the documented fine-tune optimizer (README.md:1370-1371) and the WebUI sets it automatically (webui.py:3095, 5583); MOSS int8 stays at 10 windows (README.md:1373). The fine-tune runs lr 1e-5 with anchor decay (README.md:1360-1366), far from APOLLO's tuned LLM lr. The real lever left is the fp32 weight + grad (8 of 10 B), which APOLLO does not touch (see the fine-tune subsection).

**On bigger vast.ai cards**: if a large-card fine-tune switches to adamw or automagic3 for quality, APOLLO would give about 1.6-1.7x larger windows than those (source skeptic's arithmetic); against adafactor still nothing. Its 3x throughput came from 4x larger batches on A100s in LLM pretraining and does not carry over to batch-1 character training.

**How to test** (only as an optimizer-quality experiment): register `apollo` / `apollo_mini` (config.py:22, trainer.py:259, finetune.py:54 at 10 B), confirm identical window plans in the launch log, then race adafactor vs apollo vs automagic4 in fine-tune mode on MOSS int8, lr 1e-5, anchor 0.01, two cycles, judged by the likeness probes.

## Systems and two GPUs

This section covers the plumbing around the training step: compile caching, CPU-GPU sync points, CUDA graphs, the GIL cost of driving two GPUs from threads, and a pipeline schedule for `model_split`. The overall verdict is modest. RulzTrainer's steps are 2.5-9 s and GPU-bound at 0.5-1 MP, batch 1 and gradient accumulation 1, so none of these items speeds up a typical character run by more than a few percent. The one clear win is carrying the compile cache to fresh vast.ai instances. Three items (G78, G79, G81) share the same first step, which costs nothing: run `RULZ_PROFILE_STEP` and read the per-GPU idle time that `profiling.py` already reports before writing any code.

### Portable compile cache for vast.ai (Mega-Cache)

**Do it. Expected gain: roughly 1-2 minutes of cold compile saved per fresh vast.ai instance, per model and resolution (extrapolated from the Windows 160 s cold vs 36 s warm measurement; Linux unmeasured); about 0 per step. Confidence: high (0.75). Cost: ~0.5 day.**

**What it is.** `torch.compiler.save_cache_artifacts()` serialises the PGO, AOTAutograd, Inductor, Triton and autotune caches into bytes; `load_cache_artifacts()` repopulates them on another machine with the same torch and triton versions and the same GPU. A fresh rental then skips Triton recompilation, but not dynamo tracing, so warm-up falls toward the warm figure rather than to zero. The scout bundled four other Inductor levers with this; the verifier tested or checked each and only the cache survives (see the refuted members below).

**Evidence**
- [PyTorch compile caching tutorial (Mega-Cache)](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html): artifacts populate all modular caches and are validated against torch/triton version and GPU. No timings given.
- [PyTorch 2.13.0 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.13.0): `keep_static_cubin_raw` so restored caches on another machine avoid recompilation. The verifier notes it defaults to off (env `TORCHINDUCTOR_KEEP_STATIC_CUBIN_RAW`).
- [torch/_inductor/config.py (main)](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/_inductor/config.py): `fx_graph_cache` and `autotune_local_cache` are on by default, so a second run on the same box already hits; the gap is only fresh machines.
- RulzTrainer's own measurement, README.md:1820: Krea 2 nvfp4 warm-up 36 s, 160 s with a cold inductor cache (Windows, 5060 Ti). That is the only number behind the gain estimate.

**Refuted or dropped members of this item**
- `TORCHINDUCTOR_COMPILE_THREADS>1` on Windows: tested by the verifier on this box (torch 2.14.1+cu130, triton 3.8.0, 12 CPUs). It fails with "pass_fds not supported on Windows"; with `TORCHINDUCTOR_WORKER_START=spawn` it fails with BrokenProcessPool. `TORCH_COMPILE_WORKER_MODE` does not exist in 2.14.1, and the 2.15 thread mode is documented "for free-threaded Python". The warm 25-36 s is mostly dynamo/AOT tracing that workers do not touch anyway. [pytorch#125235](https://www.github.com/pytorch/pytorch/issues/125235) uses `COMPILE_THREADS=1` as a hang workaround.
- `coordinate_descent_tuning`: tunes only Inductor's pointwise/reduction kernels, about 10% of GPU time per the README profile (attention 42-50%, matmuls 31-38%, README.md:1834). Expect under 1% per step, with a longer cold warm-up that races cannot amortise.
- Static shapes plus text padding: counterproductive. `dynamic=True` already gives 0 recompiles over 15 caption lengths (README.md:1822-1825). Static would multiply graphs by buckets x text lengths x train/sample/guidance/region-prior variants x GPUs, and race rank swaps add more.
- `shape_padding` with text padded to multiples of 64: acts only on GEMMs inside Inductor graphs, which here are just the rank 8-16 LoRA matmuls. Negligible.
- Minor source corrections: `max_autotune_gemm_backends` defaults to "ATEN,TRITON,CPP", not the long CUTLASS/NVGEMM list.

**RulzTrainer today.** `compiling.py` raises dynamo limits and leaves Inductor at defaults; `trainer.py:1483` compiles each block with `block.compile(backend="inductor", dynamic=True)`. `compiling.report()` already logs compile stats after 5 steps. `scripts/vast/provision.sh` and `rulztrainer.sh` set no `TORCHINDUCTOR_CACHE_DIR` or `TRITON_CACHE_DIR`, so the cache sits in container `/tmp` and is lost on every new instance.

**Fit and risks**
- No effect on W8A8/NVFP4 numerics, the LoRA/Quant wrappers, the sigma grid or export formats.
- A mismatched GPU or torch/triton version is rejected, which is harmless but means one artifact per (GPU, torch, triton).
- vast.ai is Linux, where parallel compile workers already work, so the cold Linux figure may be below 160 s. Unmeasured.
- Windows gains nothing beyond the existing on-disk cache.

**Implementation path**
1. `compiling.py`: `load_artifacts(path)` and `save_artifacts(path)` around `torch.compiler.load_cache_artifacts` / `save_cache_artifacts`, in try/except so a mismatch is skipped silently.
2. Key the file as `<cache_root>/compile/<torch>-<triton>-<gpu name>-sm<cap>.bin`.
3. `trainer.compile()` (trainer.py:1474): load before the `block.compile` loop; save where `compiling.report()` runs after 5 steps, and again at the end so sampling and probe shapes are included.
4. Config option `compile_cache` (default auto: `<models dir>/.rulz_compile`).
5. `scripts/vast/provision.sh` and `rulztrainer.sh`: export `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` under `/workspace`, so a restarted instance stays warm even without the artifact. Optionally sync the `.bin` with the models bucket.
6. Optional: a `RULZ_COMPILE_CDT=1` measurement switch for coordinate descent, in the style of `RULZ_COMPILE_NESTED`.

**How to test.** Speed only, no likeness race; losses must match to bf16 noise.
- On a vast instance or WSL Linux, `scripts/bench.py` for Krea 2 nvfp4 and MOSS int8 at 1 MP, 20 steps. Run A: empty cache dir, no artifact, save it. Run B: fresh empty cache dir, artifact loaded.
- Metric: wall time to the end of step 5 plus the "s compiling" line from `compiling.report`.
- Keep if B is at least 60 s faster, or within 10 s of a fully warm run, with identical step losses.
- Coordinate descent (optional): 45-step benches, median s/step over steps 21-45, warm cache; keep only at >= 2% gain and < 60 s extra cold warm-up.
- Windows compile threads: retry only after installing torch 2.15, with `TORCH_COMPILE_WORKER_MODE=thread` and `TORCHINDUCTOR_COMPILE_THREADS=6` on a cold cache; stop at the first crash or hang.

### Remove per-step CPU-GPU syncs

**Race it (as a bench, not a likeness race), after a free profile. Expected gain: about 0.2-2% per step at gradient accumulation 1 (unmeasured; scales with gradient accumulation). Confidence: medium (0.55). Cost: ~1 day.**

**What it is.** Each `.item()`, `.cpu()` or `torch.cuda.synchronize()` drains the GPU queue; the GPU then idles while Python does the between-step work and enqueues the next kernels. Keeping the loss, per-image loss and sigma on the device, copying them non-blocking into a small pinned buffer after `optimizer.step()` is enqueued, and reading them one step later behind a CUDA event, lets the CPU run ahead. Timings move to CUDA event pairs. The scout's second half, a pinned prefetch thread for the latent/text cache, adds about nothing for this workload.

**Evidence**
- [PyTorch 2.15 CUDA semantics notes](https://docs.pytorch.org/docs/2.15/notes/cuda.html): asynchronous execution; host-to-GPU copies are faster from pinned memory; `non_blocking` overlaps copies with compute; `.item()` is prohibited under graph capture. No numbers.
- PyTorch's [pin_memory / non_blocking tutorial](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html) (contrary): pinning before a copy "is usually slower than just executing the transfer", pinning blocks the host thread, and few or small tensors gain little. This is the batch-1 case: a few MB of latent and text embeds per item, already in the OS cache.
- RulzTrainer's own related measurement (README.md:268): drawing all of a step's batches first in data_parallel drops the syncs between micro-batches, and Krea 2 with `gradient_accumulation: 4` trains at 6.26 s per image against 6.37 at 1 per step, about 1.7%.
- The CUDA-graph prerequisite argument does not transfer: graph capture of training is impractical here (see the CUDA graphs item).

**RulzTrainer today.** Nothing like this exists. Sync points in the step (verifier-checked):
- `trainer.py:1001`: `per`/`sigma` `.cpu().tolist()` before backward when `loss_watch` is on.
- `trainer.py:1021`: `torch.cuda.synchronize(rdev)` to time the identity loss.
- `trainer.py:1022/1025`: `report.item()` on every micro-batch.
- `trainer.py:1038`: `work_list` times each batch by wall clock and relies on that `.item()` for `dp_balance`.
- `multigpu.py:82`: `reduce_grads` ends with a synchronize per device; `blockswap.py:116` throttles with `event.synchronize`.
- `dataset.py:439-441`: `load_file()` per item per step, pageable `.to(rdev)`.

After the one sync per step the GPU waits through `clip_grad`, the optimizer enqueue, EMA and scheduler updates, logging and `status.update` (trainer.py:1210-1214, which writes a JSON with 2000-entry loss and step-time lists), `load_batch`, `sample_sigmas`, CPU `randn` and the H2D copy. The verifier puts that gap at a plausible 10-60 ms: about 0.2-0.7% of Krea 2 nvfp4 (~9 s/step), 0.3-1% of MOSS int8 (~5 s/step), 0.5-2% of Qwen 2.1 int8 (~2.5 s/step at 0.5 MP).

**Fit and risks**
- `dp_balance` depends on the wall-clock-plus-`.item()` timing; it must move to CUDA events or its split will be wrong.
- `losswatch.multiplier()` comes from stored per-key stats (losswatch.py:210), so observing one step late is benign; `true_per_image_lr` reads `results[0][1][0]` in the same step and needs the same treatment.
- The OOM retry path must discard a failed step's deferred readback.
- torch.compile, int8 ConvRot W8A8 and NVFP4 kernels, the sigma grid and export formats are untouched; the syncs sit outside the compiled blocks.
- With block swap or data_parallel the gain shrinks further, because their own syncs remain.
- Windows: plain Python threading and pinned memory, no NCCL. Pinned RAM is a few MB, not the scout's "few hundred MB".

**Implementation path**
1. `trainer.py` `work()`: return `report`, `per.detach()`, `sigma.detach()` as device tensors.
2. After `optimizer.step()` is enqueued, stack the step's loss/per/sigma (and `id_val`), `copy_(non_blocking=True)` into a pinned CPU buffer, record an event.
3. Next step, after its forward is enqueued, synchronize that event and feed `loss_watch.observe()`, `accum_loss`, the log line and `status.update` with the previous step's values; flush after the last step.
4. Replace the id-loss `time.time()` + synchronize (trainer.py:1019-1022) and the `work_list` wall-clock timing (1036-1040) with `torch.cuda.Event(enable_timing=True)` pairs read a step later.
5. Keep `sample_sigmas` and `randn` on the main thread so runs stay reproducible.
6. Skip the pinned prefetch thread; at most `load_batch()` the next step in a background thread without pinning.
7. No user option; at most a `RULZ_SYNC_FREE=0` escape hatch. Applies to all four families.

**How to test**
- Before coding: `RULZ_PROFILE_STEP=100` on Qwen 2.1 int8 at 0.5 MP and read the per-GPU idle time. If the idle share is under 1%, drop the item.
- After: `scripts/bench.py`, 200 steps, before vs after, same seed, `loss_watch` off, on Qwen 2.1 int8 0.5 MP, MOSS int8 and Krea 2 nvfp4 on the 5060 Ti alone, then MOSS in data_parallel on both cards. Metric: median s/step over steps 20-200.
- Correctness: with `loss_watch` off the loss curves match step for step to bf16 noise. With `loss_watch` and `per_image_lr` on, one 1500-step A/B with ArcFace face and DINO body probes within probe noise.
- Stop rule: keep only if median s/step improves >= 1% on at least one model, no model gets slower, and `dp_balance` converges to the same split ratio.

### Two GPUs from one process: GIL, free-threading, NCCL

**Do it (as a documented decision plus a small profiling extension; no build). Expected gain: no speedup; it heads off a dead end. A Linux NCCL process mode is worth about 0-5% at 0.5-1 MP on 2 GPUs (unmeasured). Confidence: high (0.8). Cost: ~2 hours.**

**What it is.** RulzTrainer drives both GPUs from threads of one Python process, so both threads take turns at the GIL while dispatching kernels. The three ways out are free-threaded Python, one process per GPU, or less Python work per step. The status check says the first two are closed on the user's setup in 2026, so the answer is to keep threads and only measure.

**Evidence**
- [triton#8491](https://github.com/triton-lang/triton/issues/8491): cp314t wheels exist, but libtriton does not declare GIL compatibility, so importing it re-enables the GIL. Open, no milestone, last activity 2026-05-18. The "no priority, no support" quote is from a contributor (Qubitium), not a maintainer; maintainer Jokeren said the GIL=0 autotune fix (PR #8437, unmerged as of Aug 2026) is not planned near term.
- [PyTorch 2.13.0 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.13.0): 3.13t builds stopped, users told to move to 3.14t. [PyTorch releases](https://github.com/pytorch/pytorch/releases), 2.10: 3.14t "experimentally supported".
- [pytorch#130249](https://github.com/pytorch/pytorch/issues/130249): Triton "will result in segfaults with backend="inductor"" on 3.13t (June 2025).
- [py-free-threading tracker](https://py-free-threading.github.io/tracking/): torch from 2.6.0, triton not listed.
- [triton-windows README](https://github.com/woct0rdho/triton-windows): free-threaded builds seem to work with MSVC/GCC/Clang but not with the bundled TinyCC.
- [torch.distributed 2.15 docs](https://docs.pytorch.org/docs/2.15/distributed.html): Windows supports all backends but NCCL, Windows is prototype, Gloo is slower than NCCL for GPUs.
- [PyTorch Windows FAQ](https://docs.pytorch.org/docs/2.15/notes/windows.html): CUDA IPC is not supported on Windows.
- Unverified member: "torch 2.15 added no-GIL compile workers" has no source; the 2.15 worker mode is documented only for free-threaded Python.

**RulzTrainer today.** `multigpu.DataParallelLoRA` runs one thread per GPU through `ThreadPoolExecutor` (multigpu.py:46-52, 91-102), exchanges only LoRA grads and weights device to device (multigpu.py:57-80), uses no torch.distributed (README.md:242-249), and goes serial while torch.compile may trace (multigpu.py:95-99, 410-428). `profiling.StepProfile` (profiling.py:66-100, trainer.py:1052) already reports per-GPU idle time. Measured two-GPU scaling on the 5060 Ti + 4060 Ti at 1 MP (README.md:244-272):
- MOSS int8_convrot compiled: 4.79 s per 2 images (2.40 per image) against 4.90 s/step alone, about 2x.
- Krea 2 nvfp4: alone the 5060 Ti does 8.5 s per image, but in data_parallel the two GPUs measured 14.2 and 12.4 s per batch. The README attributes this to the 5060 Ti's short FP4 kernels waiting on the other thread's interpreter turn. This is the one place where GIL cost is visible, and it is nvfp4-specific.

**Fit and risks**
- A Windows process-per-GPU mode would all-reduce through Gloo on CPU tensors with host round trips: strictly worse than today's device-to-device copy.
- Free-threaded 3.14t breaks triton-windows (TinyCC) and therefore the int8 ConvRot and NVFP4 triton kernels; Inductor's autotuner is not thread-safe under GIL=0.
- A Linux NCCL mode would duplicate the `.rulz_cache` text/VAE caching and break the WebUI's in-process job state, `race_parallel` one config per GPU (race.py:217), side jobs on the second GPU, `dp_balance`, `SpeedMeter`, and in-process probes and samples. A large refactor for 4-8 GPU rentals only.
- No interaction with quantization, the sigma grid, keep blocks, block swap or export formats.

**Implementation path**
1. README note under "Using two GPUs": free-threaded Python is unsupported until Triton declares GIL compatibility (triton#8491, PR #8437); Windows stays on threads (no NCCL, no CUDA IPC).
2. `profiling.StepProfile`: next to the existing idle time, report per worker thread the CPU time in `cudaLaunchKernel` runtime events, so a data_parallel profile shows whether idle time is launch-bound.
3. Do not build `multi_gpu: processes` now. If a 4+ GPU profile ever shows idle above ~15% that grows with GPU count: Linux only, spawn per rank, all_reduce one flat fp32 bucket of LoRA grads, rank 0 keeps probes, samples, race control and the WebUI.

**How to test.** `RULZ_PROFILE_STEP=20` in data_parallel on the 5060 Ti + 4060 Ti for MOSS int8_convrot (compiled and eager), Krea 2 nvfp4 and Qwen 2.1 int8, at 0.5 MP and 1 MP; compare each GPU's idle share against the same GPU alone. The extra idle share is the upper bound on GIL cost. Repeat on a 4-GPU vast.ai box. Stop rule: under 5% extra at 2 GPUs and under 10% at 4 GPUs closes the topic; prototype the NCCL mode only above 15% and growing with GPU count, then bench it against threads on s/image over 200 steps, same seed, losses within bf16 noise.

### CUDA graphs for probe and sample renders

**Watch. Expected gain: training steps under 3% (estimate), offset by capture cost; turbo probe and sample renders maybe 3-10% per render, about 1% or less end to end. All unmeasured. Confidence: low (0.25). Cost: ~2-3 days for the sampler-only version.**

**What it is.** A CUDA graph replays a captured kernel sequence with "greatly reduced CPU overhead". The scout proposed capturing the whole training micro-step at low resolution, with text padded to a few bucket lengths to bound the number of graphs. The verifier found the premise wrong (steps are 2.5-9 s, not 0.5-1 s) and several hard capture blockers in the training step. The only plausible target is one denoising step of the inference sampler for probes and samples.

**Evidence**
- [PyTorch 2.15 CUDA semantics notes](https://docs.pytorch.org/docs/2.15/notes/cuda.html): reduced CPU overhead; no CPU syncs and no dynamic shapes in capture; pools shareable only when graphs replay in capture order. Omitted by the scout: only one capture per process at a time, and no non-captured CUDA work may run on any thread while a capture is in progress.
- [torch/_inductor/config.py](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/_inductor/config.py): `triton.cudagraphs` off unless `TORCHINDUCTOR_CUDAGRAPHS=1`; `cudagraph_skip_dynamic_graphs=False` re-records a graph per unique shape set; `cudagraph_dynamic_shape_warn_limit=8`.
- [2.12.0](https://github.com/pytorch/pytorch/releases/tag/v2.12.0): `torch.accelerator.Graph`, `torch.cond` capture via CUDA 12.4+ conditional nodes, with no Inductor support yet. [2.15.0](https://github.com/pytorch/pytorch/releases/tag/v2.15.0): per-op cuBLAS workspaces fix replay of graphs captured separately on one stream; captured gradient accumulation belongs to FSDP2/pipelining and does not apply. [2.11.0](https://github.com/pytorch/pytorch/releases/tag/v2.11.0): cudagraph partitioning supports unsafe symints.
- No source measures a gain for DiT or LoRA training; documented wins are small launch-bound models and LLM decode. PyTorch forum reports cite overwritten-output errors, NaNs only under reduce-overhead, OOMs, and rough edges with activation checkpointing plus graphs.

**RulzTrainer today.** No CUDA graph use anywhere (grep finds none). Blocks compile with `dynamic=True` in default mode (trainer.py:1473-1488). Text length floats per batch (krea2/loading.py:175 pads to the batch's longest caption).

**Fit and risks (why training-step capture is out)**
- Loss path is Python per item: loss_watch `.tolist()`, per-item multipliers, `.item()` syncs, identity loss with data-dependent face crops and `torch.cuda.synchronize`, region_prior and DOP extra forwards (trainer.py:990-1030).
- NVFP4 stochastic rounding passes a Python-int seed from `itertools.count` (quant.py:888-899); a captured graph would replay the same seed every step and silently bias the FP4 backward. int8 W8A8 kernels and `torch._int_mm` are capturable.
- Block swap host-syncs on events and owns a side stream (blockswap.py:113-128).
- Checkpoint recompute inside a captured backward is fragile; graph pools raise peak memory on cards already at a 13.1 GB peak (README.md:1692).
- data_parallel, model_split and side jobs all run CUDA work from other threads in the same process; each capture needs a process-wide pause.
- Race rank swaps invalidate graphs; each quantized Linear is a graph break, so reduce-overhead per block yields many small partitions.
- Windows: triton-windows plus cudagraph trees is not exercised anywhere.

**Implementation path (only if the profile justifies it)**
1. Measure first: CUDA-event vs wall-clock GPU-busy fraction for one training step and one probe render in krea2/qwen21/moss `sampler.sample`, at 512 px and 0.5 MP, with and without the second GPU's side jobs. Proceed only if idle exceeds ~10%.
2. `sample_cuda_graphs: off|auto` in config.py. Under auto and `block_swap: 0`, capture one denoising step per (width, height, cfg, padded text length) with static latent, sigma and embed buffers; pad prompts through a new `pad_to` argument on `pad_embeds`; replay for every step.
3. Hold a process-wide lock during capture and fall back to eager on any capture error.
4. Training capture stays out until the step is free of `.item()`/`.cpu()`/`.tolist()`, NVFP4 rounding takes its seed from a device tensor, and block swap is off.

**How to test.** Krea 2 nvfp4 and Qwen 2.1 int8 at 0.5 MP, rank 8, probes on; record per-step time, per-render time and GPU-busy fraction. If built, A/B `sample_cuda_graphs` off vs auto, 300 steps, probes every 50, identical seeds. Reject if the median probe render gains < 5%, total wall time gains < 1%, peak VRAM rises > 300 MB, or fixed-seed latents differ by more than 1e-3 max abs in bf16 (ArcFace and DINO scores must match).

### Micro-batch pipeline for model_split

**Skip-ish. Expected gain: zero for the current recipe (batch 1, gradient accumulation 1, quantized models that fit one card); about 1.1-1.4x per image over sequential model_split at gradient accumulation 4 on bf16 models (unmeasured). Confidence: low (0.3). Cost: ~1 week.**

**What it is.** model_split puts contiguous block ranges on each GPU, and the GPUs take turns. A GPipe or 1F1B schedule splits each optimizer step into M micro-batches and overlaps stage 0's forward of micro-batch 2 with stage 1's forward of micro-batch 1, and likewise in backward. Only block-boundary activations cross PCIe. The bubble fraction is (stages - 1)/(M + stages - 1), so with M = 1 there is nothing to overlap.

**Evidence**
- [DeepSpeed pipeline tutorial](https://www.deepspeed.ai/tutorials/pipeline/): micro-batches from gradient accumulation, local grad accumulation per stage, partition by parameters/type/uniform; performance "strongly relies on load balance". No speedup figure.
- [GPipe, arXiv 1811.06965](https://arxiv.org/abs/1811.06965): "almost linear speedup", on a 557M AmoebaNet and a 6B Transformer trained from scratch with large batches and many micro-batches. Does not transfer to batch-1 LoRA.
- [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe): pipeline_stages for Krea 2, MiniMax H3, Qwen-Image 2.1 and MOSS-Image; no speed numbers; its config says `gradient_accumulation_steps` is the number of micro-batches and low values leave GPUs idle; native Windows is "difficult or impossible".
- [RamTorch](https://github.com/lodestone-rock/RamTorch): ~50% vs ~92-98% utilisation for 1f1b vs staggered_1b1f on a 10.9 GB model, with no hardware named and likely from its schedule simulator; the 0.4% overhead is explicitly simulated.

**RulzTrainer today.** model_split exists (multigpu.py:8-12, README.md:276-279), auto-balanced by a fixed cost model (forward-only blocks 1, trained blocks 4; multigpu.py:319-341) or a manual `split_plan` (config.py:194). Activations hop inside each family's forward loop (krea2/model.py:322-343, qwen21/model.py:333+). The step runs one `work_list` in one thread (trainer.py ~1112). Side jobs (samples, probes, caching, SAM 3) already use the second GPU (README.md:281-295), which cuts the idle time further.

**Fit and risks**
- Show-stoppers for the user: batch 1 with gradient accumulation 1 (config.py:100, 273); every quantized model trained (Krea nvfp4, MOSS/Qwen int8) fits one 16 GB card, so data_parallel covers throughput; a threaded pipeline in one process meets the same GIL limit that kept Krea 2 nvfp4 data_parallel at 14.2/12.4 s per batch vs 8.5 s alone (README.md:270-274).
- Gradient accumulation 4 changes the step semantics and lr the likeness recipes were tuned at.
- keep blocks must be recalibrated or set to 0 (activations multiply with micro-batches in flight); torch.compile needs the data_parallel serialise-until-settled logic (trainer.py:1086-1110); block swap is already rejected with model_split (config.py:592).
- The 4060 Ti dequantizes NVFP4, so stage balancing needs measured block time, not the cost model.
- Alignment taps, identity loss, DOP, region weights and per-image loss watch assume one sequential forward per item.
- Only relevant for bf16 Krea/Qwen/H3 on 2x16 GB; big rented GPUs usually fit the whole model, where data_parallel wins.

**Implementation path (if ever)**
1. Refactor each family's forward (krea2/model.py ~328, qwen21/model.py ~333, moss/model.py) into prologue / `run_blocks(lo, hi, state)` / epilogue.
2. `multigpu.py`: a PipelineSplit runner, one thread per stage, queues for boundary activations and gradients, GPipe schedule (all forwards, then backwards in reverse); each stage detaches its input with `requires_grad` and calls `torch.autograd.backward(out, grad_from_next)`.
3. Option `split_pipeline: off|auto`, active for model_split with gradient accumulation >= 2; sequential path kept for M = 1, sampling, probes and the reward stage.
4. Force `checkpoint_keep_blocks: 0` under the pipeline; reuse the compile serialisation; replace the cost model with a short timed probe per block per GPU; disable alignment taps, identity loss and DOP at first. MiniMax H3 would inherit it by exposing `run_blocks`.

**How to test.** Bench bf16 Krea 2 or Qwen 2.1 at 0.5 MP on the 5060 Ti + 4060 Ti with `gradient_accumulation: 4`: sequential split vs pipelined split vs int8 data_parallel with `dp_balance: auto`, seconds per image over 50 steps after warm-up, peak VRAM per GPU; check gradients match within bf16 tolerance on a fixed seed. Only if pipelined is >= 1.3x faster per image than sequential and not slower than data_parallel, race bf16 pipelined split at GA 4 against int8 data_parallel at batch 1 / GA 1 with equal images seen, ArcFace face and DINO body probes every 250 images, stop after 3 probes without improvement or at 3000 images. Drop the feature below 1.2x per image.

## Evaluation, probes and races

This section covers how RulzTrainer measures likeness (probes), how it turns those scores into decisions (best checkpoint, early stop, lr control, race drops, relay leaders, selftrain filters, the reward stage), and how cross-run memory could cut race cost. None of these items trains a better LoRA directly; they change which checkpoint or race configuration survives, and how much wall time the probes take. The strongest finding is a gap in the statistics: probes average 1-2 renders, throw the per-render scores away, and the race drops on a point estimate, so nobody knows whether today's drops are inside render noise. The cheap, sound steps are logging per-render scores with a measured noise model, a held-out "gold" face recognizer (FaceNet is already vendored), and an SSCD copy gate for selftrain; the heavier ideas (VLM adherence judge, async probes on GPU 1, low-resolution race rounds, PBT, transfer BO) are either unmeasured or need data the user does not have yet.

All verifiers checked the current code. The race/likeness edits that were uncommitted when the brief was written are now in HEAD (0d1c98a: automagic4 keys added to RACE_KEYS; 83af986: race_patience_rising). Nothing below proposes what is already there.

### Measured probe noise and paired race decisions

**Do it (step 1-3), race the drop rule. Expected gain: unmeasured; fewer wrong drops when racers sit within about 10 normalized points, plus an immediate answer to "are my race drops inside noise". Confidence: high (0.7). Cost: small for logging and a fixed-N paired test, zero step time; adaptive extra renders are a bigger change.**

**What it is.** Keep every (prompt, seed) probe score instead of the mean, measure per-render sigma and the correlation rho between racers on shared seeds, and drop or switch the relay leader only when a paired one-sided test says a racer is worse than the round leader. Racers already share probe seeds, prompts and step counts, so they are common random numbers: the variance of a paired difference is 2*sigma^2*(1-rho), which is smaller than for unpaired means when rho > 0. A one-way ICC (between-racer variance over between plus within) tells whether the race metric discriminates at all at a given step. This item also absorbs the cheap part of the "Pareto selection" finding below: a margin test on drop_lowest.

**Evidence.**
- [Optuna WilcoxonPruner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.WilcoxonPruner.html): pairs each trial's per-instance results with the best trial by instance id, p_threshold default 0.1, n_startup_steps 2, experimental since v3.6.0; warns that repeated looks inflate false positives and that a Pocock-corrected p is needed.
- [Isaac Lab PBT guide](https://isaac-sim.github.io/IsaacLab/main/source/features/population_based_training.html): cutoffs at mean +/- max(threshold_std*sigma, threshold_abs); if no member crosses them, nobody is replaced. Warns that noisy objectives replace useful policies and that identical low scores can mean an uninformative metric (the ICC check).
- [Bax, Sarkar, Shtoff 2024](https://arxiv.org/abs/2408.00908): requiring success at several decision points eases the strict Bonferroni p-values of early stopping. It is about A/B bucket tests; mapping it onto race_patience is an inference.
- [irace](https://search.r-project.org/CRAN/refmans/irace/html/irace-package.html): iterated F-race; the opened page does not itself list the elimination tests.
- Sample size: n = (z_{1-a} + z_{1-b})^2 * 2*sigma^2*(1-rho) / delta^2. With an assumed sigma of 6 normalized points and rho 0.3 (one-sided a = 0.1, power 0.8), that is about 25 renders per racer to resolve a 3-point gap, 9 for 5 points and 4 for 8 points (verifier checked the arithmetic). Sigma and rho are assumptions; nobody has measured them for face embeddings on diffusion renders.
- Limits: paired Wilcoxon cannot reject at alpha 0.1 with 3 pairs (minimum one-sided p is 0.125) and is coarse at 4-6, so use a paired t-test or bootstrap at small n. [Klein et al.](https://arxiv.org/html/2409.02086v2) warn that pairing only helps when draws stay aligned; [Sharma](https://arxiv.org/abs/2512.24145) formalizes paired seeds as common random numbers in RL. RulzTrainer's own likeness.py:93-96 records that two probe seeds' gap flipped sign across steps on a real run, so rho across steps is weak; rho across racers at the same step is plausibly high but unmeasured.

**RulzTrainer today.**
- config.py:430 probe_seeds 1 (the wizard preset sets 2, config.py:101), probe_floor_seeds 4 for step 0 only (config.py:431), one probe_face_prompt and one probe_body_prompt string (config.py:428-429), likeness_noise_band 1.0 (config.py:437).
- trainer.py:3264-3318 `_probe_with` scores each render, then keeps only `face = sum(face_scores)/len(face_scores)` (trainer.py:3307). Per-render scores never reach likeness.jsonl.
- race.py:394-399: drop_lowest removes `min(self.alive, key=key)` on the raw round score with no margin; decide() (race.py:407-418) takes the max best score; the relay leader is the max score (race.py:384-386). Patience drops use since_improve, which the noise band guards (race.py:358).
- Delta: per-render score storage, a measured sigma/rho/ICC, a paired test and a margin in record() and the relay leader switch.

**Fit and risks.** Pure evaluation statistics: no interaction with W8A8, nvfp4, compile, keep blocks, block swap or exports. scipy is already a dependency (regions.py). Probe renders already spread over GPUs through `_render_round`, so extra renders scale with GPU count. The hard part is adaptive "add 4 more seeds for an undecided pair": racers train in turn and parked racers live on the CPU (weights, optimizer, EMA), so re-probing needs swap_in / render / swap_out, with care for lora_ema views. A fixed-N paired test at round end avoids that. Probe EMA smoothing (probe_ema) and the per-step paired test answer different questions; test on the per-step paired differences.

**Implementation path.**
1. In `trainer._probe_with`, return per-job scores (kind, prompt index, seed, score) and store them as `face_scores` / `body_scores` in ProbeRecord (likeness.py:113), normalized with `controller.norm`.
2. Accept optional lists `probe_face_prompts` / `probe_body_prompts` next to the existing string options (prompt-level variance needs 3-4 face and 2-3 body prompts).
3. New `rulztrainer/likeness_stats.py`: paired difference per (prompt, seed), paired t-test or bootstrap one-sided p, Holm over racers, one-way ICC. Unit tests in tests/test_race.py with synthetic scores.
4. `RaceSchedule.record` takes per-racer score vectors. New `race_confidence` (0 = today's behaviour, 0.9 suggested): drop the lowest only if its Holm-corrected paired p < 1 - race_confidence, else log "too close to call" and keep everyone. Without vectors, fall back to `race_drop_margin` (default = likeness_noise_band): drop only if the lowest is more than the margin below the next lowest. Apply the same rule to the relay leader switch and to decide().
5. `race_min_icc` (about 0.2): skip the first drop when the metric is not discriminative yet.
6. Optional later: adaptive renders capped by `race_max_probe_renders` (16-24), via swap_in of parked racers.
7. Show sigma, ICC and "too close to call" in the WebUI race panel; README and wiki.

**How to test.**
1. Measure first: on a finished run (krea2-AZav-rulz, plus a Qwen 2.1 and a MOSS run) render the best LoRA and a mid-run checkpoint at 16 seeds x 3 face prompts and 16 seeds x 2 body prompts. Compute sigma per kind, rho between the two checkpoints on shared seeds, and the ICC. Go only if sigma*sqrt(2(1-rho)) is comparable to the gaps the race actually dropped on (read from race history in likeness.jsonl).
2. Offline replay: re-render each racer's per-round checkpoint of a past 3-4 racer race with 8 seeds, compare "drop min" vs the paired rule on how often the eventual winner (best 16-seed final score) was dropped.
3. Live: the same 4-config race twice on seed twins, race_confidence 0 vs 0.9. Metrics: picked LoRA's normalized combined score on 16 held-out seeds, total probe time, wrong drops (a dropped racer whose 16-seed score beats the pick). Stop rule: equal step budget; adopt if the winner's 16-seed score beats baseline by more than its bootstrap CI, or if wrong drops fall with no loss.

### Held-out "gold" face recognizer (FaceNet first, AdaFace optional)

**Do it. Expected gain: unmeasured; lower probe noise from two recognizers (maybe 0-5% wall time through fewer seeds or shorter patience) and the only automatic hacking check for runs that optimize ArcFace. Confidence: medium (0.6). Cost: small with FaceNet (already vendored, no new download); medium for AdaFace (vendor IR-100, in-house export).**

**What it is.** Score probes with ArcFace plus a second recognizer and log the disagreement. Name one recognizer "gold": no loss, alignment or reward ever optimizes it, and best checkpoint, early stop and race ranking use it whenever ArcFace is being optimized (reward stage, id_loss_backend arcface, id_align). This is the proxy/gold split from RLHF applied to face identity.

**Evidence.**
- [StyleID (arXiv 2604.21689)](https://arxiv.org/html/2604.21689): human same/different agreement (Cross-ID) AdaFace TPR 0.7835 / AUROC 0.9498 vs ArcFace 0.7649 / 0.9418; CLIP 0.256, SigLIP2 0.174. ArcFace pose consistency 0.8309 on real vs 0.4389 on stylized images. Every pair is real vs stylized, so the AdaFace edge (+1.9 TPR points) is small and not shown for photoreal LoRA renders.
- [WithAnyone (arXiv 2510.14975)](https://arxiv.org/html/2510.14975v1), App. F.1: averages ArcFace, FaceNet and AdaFace because one model may bias the identity metric.
- [AdaFace](https://github.com/mk-minchul/AdaFace): MIT, R18/R50/R100 weights including R100 WebFace12M, 112x112 BGR input, mean/std 0.5, MTCNN alignment. Weights are Google Drive checkpoints; no published ONNX export.
- [LVFace](https://github.com/bytedance/LVFace): ICCV 2025 Highlight, ONNX and .pt, LVFace-B IJB-C TAR@1e-4 97.70; non-commercial research only (opt-in download at most).
- [Benchmarking FR without real faces (arXiv 2607.14932)](https://arxiv.org/html/2607.14932): 24 models on 12 synthetic sets, ranking consistency Spearman 0.87, no human judgements.
- Refuted inside this item: the scout's IPRO "CurricularFace cross-check 0.581 -> 0.699" is not in IPRO ([arXiv 2510.14255](https://arxiv.org/html/2510.14255v1)); its only alternative encoder is CLIP. IPRO's 10% vs 52-58% hacking compares ablations (no KL, no FSM) with the full method, not proxy vs gold. Note the G32 verifier reported a CurricularFace check in IPRO v4 App. H (Wan 2.2 A14B, Arc 0.578 -> 0.694, Cur 0.581 -> 0.699); the two verifiers read different versions, so treat the second-recognizer agreement as weakly supported at best.

**RulzTrainer today.**
- One InsightFace buffalo_l ArcFace judges everything: probes, best checkpoint, early stop, lr control, race drops, selftrain filter (facesim.py:105 FaceScorer; trainer.py:2932-3069, 3300, 3449).
- The reward stage optimizes that same ArcFace (reward.py:9-10); its guards are KL, small lr and probes with the same ArcFace (README.md:1117).
- id_loss_backend arcface makes "the probe metric the training target" (README.md:797); facenet exists only as a loss backend (config.py:390, idloss.py:138-148, rulztrainer/facenet.py). id_align projects into ArcFace space (alignment.py:42).
- Delta: a second recognizer at probe time, a stored per-recognizer floor and ceiling, and a selection rule.

**Fit and risks.** Runs only in probes, reward scoring and selftrain filters, on CPU (onnxruntime) or the side GPU (R100 about 250 MB). No interaction with the training compute path; no Windows blocker. Pitfalls: AdaFace wants BGR while InsightFace's recognizer wants RGB; InsightFace kps + norm_crop give the arcface_dst template, close to but not identical to MTCNN alignment; scores must be normalized per recognizer with their own floor and reference_ceiling before averaging.

**Implementation path.**
1. facesim.py: an Embedder hook in FaceScorer that reuses the detected faces (.kps), runs norm_crop at 112 or 160 and embeds with a second backend.
2. Backend `facenet`: reuse `idloss.facenet_embedder` in no-grad mode.
3. Optional backend `adaface`: vendor the MIT IR-100 like facenet.py, load the WebFace12M checkpoint, BGR and mean/std 0.5.
4. config `face_gold_backend: none | facenet | adaface`, default facenet when reward_steps > 0 or id_loss_backend == arcface; validate it differs from id_loss_backend.
5. LikenessController (likeness.py:138, update at :231) stores `face_gold` and the ArcFace-gold spread per ProbeRecord, with the gold score's own floor and ceiling.
6. When gold exists, best checkpoint, early stop and race ranking use it (combined with body as usual); lr control and curriculum keep ArcFace; the reward stage keeps optimizing ArcFace only.
7. WebUI Monitor curve, README, wiki.

**How to test.**
1. Offline: re-score saved probe images (krea2-AZav-rulz, past race and reward-stage jobs) with ArcFace, FaceNet and AdaFace. Report rank correlation across checkpoints and seed-to-seed std.
2. In reward-stage and id_loss=arcface runs, look for ArcFace rising while FaceNet/AdaFace go flat or fall (the hacking signal). A blind human pick among the top-3 by each recognizer settles which selector to trust.
3. Race on one character, fixed seeds: A selects and early-stops on ArcFace, B on gold; Qwen 2.1 int8, 1500-3000 steps plus about 100 reward steps. Metric: held-out human preference on 8 fixed prompts, the unused third recognizer as tie-breaker.
4. Stop after 2-3 characters. Make gold the default if it is never worse and wins on any character, or if ArcFace and gold diverge beyond the noise band in reward runs.

### SSCD copy check for selftrain and probes

**Do it (selftrain gate plus logged copy scores). Expected gain: unmeasured; decision quality, mainly stopping selftrain from feeding near-copies of training photos back in. Confidence: medium (0.55). Cost: small; milliseconds per image, descriptors cached once per dataset.**

**What it is.** For each probe and selftrain render, compute the max SSCD similarity to the training images, on the full image and on the body window crop (the outfit view). SSCD is the standard replication detector in diffusion memorization work. Read the score against the training photos' own median best cross-match, as reference_ceiling does for likeness, because a single-outfit photo shoot makes outfit copying expected.

**Evidence.**
- [SSCD repo](https://github.com/facebookresearch/sscd-copy-detection): 7 TorchScript models, 512-d (1024 for disc_large), resize 288 short edge or 320 square; MIT code, no separate weight licence stated. For sscd_disc_mixup, cosine > 0.75 marks a copy at 90% precision on DISC.
- [Somepalli et al. (arXiv 2305.20086)](https://arxiv.org/abs/2305.20086): text conditioning and caption specificity drive replication about as much as duplication; caption randomization mitigates. Uses SSCD with dot product > 0.5 as a likely partial copy and reports the 95th percentile. Setting: SD 2.1 UNet full fine-tune, batch 16, 256 px, 10k-100k images.
- Verifier correction: 0.75 is the DISC operating point; the diffusion-replication threshold is 0.5. [arXiv 2509.25705](https://arxiv.org/abs/2509.25705) reports Pearson 0.7148 between latent convergence and SSCD (preprint; the "ICLR 2026" label is unverified). The scout's CVPR 2025 and WACV 2025 (0.47 vs 0.30) numbers are unverified.
- Small-data support: LoRA on SD2 with 10 images shows heavy training-set reproduction, falling by about 50 images (arXiv 2502.09665, from the verifier's search). No source measures SSCD on DiT LoRAs or calibrates it for person photos.
- [Vendi score (arXiv 2210.02410)](https://arxiv.org/html/2210.02410) is real, but see below.

**RulzTrainer today.**
- likeness.py:28-31 and :62 (MIN_ROOM_SHARE, added because "a few points of copied outfit read 100") admit DINOv2 body saturates on copied outfit and setting. bodysim.py:598 takes the best cosine to any single reference, which rewards copying one photo.
- selftrain.py:35 / 193: a near-duplicate check on 32x32 thumbnail correlation (DUP_MAX 0.95), only among the round's own kept renders, never against training photos.
- Nothing computes SSCD, Vendi or a pose-copy distance.

**Fit and risks.** Runs on decoded renders in the side-job slot (about 100-200 MB VRAM); no interaction with training kernels; torch.jit.load works on Windows; weights come from dl.fbaipublicfiles.com (downloads.py entry). Dropped members: Vendi over 1-2 probe seeds is noise (usable only in selftrain rounds, selftrain_images 32, config.py:416); a pose-copy distance cannot reuse skeleton3d/SMPL-X, which are pose-independent by design (bodyshape.py:1-18) and would need cached MediaPipe joint angles; a grey-filled background view feeds SSCD out-of-distribution inputs (WINDOW_GREY, bodysim.py:442), so skip it; the race has no Pareto machinery (race.py:340 ranks one scalar).

**Implementation path.**
1. New `rulztrainer/copycheck.py`: load sscd_disc_mixup, 320 square, ImageNet norm; cache L2-normalized descriptors in `<dataset>/.rulz_cache/sscd.npy` for the full image and the `body_window()` crop.
2. ProbeRecord gets `copy_full`, `copy_outfit` and their dataset baselines (median best cross-match); plot in the WebUI history, logged only.
3. selftrain.py:193: a new "copies a photo" reject reason when copy_full > max(copy_threshold = 0.5, baseline + margin).
4. Config `copy_check: bool`, `copy_max: float`; optionally best-checkpoint skips probes above copy_max.
5. Defer Vendi (selftrain only) and pose-copy.

**How to test.**
1. Offline on krea2-AZav-rulz: score all saved probes with copy_full / copy_outfit; check they rise late on runs that visibly copy an outfit or background and stay near baseline on good runs.
2. Hand-label about 30 probe and selftrain renders as copy / not copy; pick the threshold that gives at least 80% precision.
3. Race selftrain on with copy_check off vs on, 2 racers, same seeds, 2000 steps, a 12-20 photo dataset. Metrics: normalized face/body likeness on held-out prompts (new outfit and scene), share of selftrain renders rejected.
4. Keep the gate if held-out likeness is not lower beyond the noise band and fewer copies are fed back; drop it if it rejects under 5% of renders on typical datasets.

### Prompt-adherence probes with a Qwen3-VL judge

**Prototype. Expected gain: unmeasured; occasionally a different best checkpoint or race winner with better editability at nearly the same likeness. Confidence: medium (0.6). Cost: medium (new judge module, edit prompts in prepare_cache); about 20-60 s per edit probe on Krea nvfp4 or MOSS, 15-30 s on Qwen 2.1 (estimate).**

**What it is.** Add 3-6 compositional "edit" probe prompts (outfit + setting + action, e.g. "{trigger} wearing a red raincoat walking on a beach at night"). Score each render with ArcFace (identity in context) and with VQAScore: the P("Yes") of a Qwen3-VL asked whether the image shows the prompt without the trigger. Add one over-fit question ("Is the face stiff or pasted-on?"). Today nothing measures whether the trigger still composes with new contexts, so a late, over-fit checkpoint can win on the headshot alone.

**Evidence.**
- [MLLMs as T2I evaluators (arXiv 2505.00759v3)](https://arxiv.org/html/2505.00759v3): VQAScore on TIFA160, Spearman/Kendall 69.5/51.2 for Qwen3-VL-8B, 74.6/56.3 for Qwen3-VL-32B, 70.7/52.9 for fine-tuned CLIP-FlanT5. Only the 32B beats CLIP-FlanT5.
- [DEPICT (arXiv 2610.03617)](https://arxiv.org/html/2610.03617): Qwen3-VL-4B VQAScore SRCC 0.628 GenAI-Bench, 0.658 TIFA160; DEPICT 0.691 at about 12.9 passes per pair. The 0.099 s vs 0.492 s latency was measured on one B300 with vLLM batching (caption names Qwen3.5-27B); it does not transfer to an unbatched judge on a 16 GB card. Code not released.
- [DreamBench++ (arXiv 2406.16855v2)](https://arxiv.org/html/2406.16855v2): GPT-4o judge reaches 98.17% of human consistency on prompt following, 83.31% on concept preservation (DINO 50.72%); ranks by CP x PF and calls CP/PF an over-fit sign (their interpretation, not validated). Its Table 4 has CLIP-T at 61.48%, so CLIP-T is a weak fallback despite the text.
- [IPRO (arXiv 2510.14255)](https://arxiv.org/html/2510.14255v1): a yes/no "over-optimized face" question gives the hacking rate, but the judge is Gemini 2.5 Pro on video; an 8B open model on stills is unmeasured. The scout's "Phota" VLM-judge claim for this item could not be found.

**RulzTrainer today.**
- Probes render only probe_face_prompt and probe_body_prompt (trainer.py:3275-3289; config.py:428-429; PRESET_PROBE_PROMPTS config.py:107-111), scored by ArcFace and body scorers (trainer.py:3297-3306). ProbeRecord (likeness.py:113-135) has no adherence field.
- reward.default_prompts (reward.py:35-40) and selftrain render new scenes, but their filters check face detection, sharpness and likeness (selftrain.py:15-20), never prompt following.
- Judge weights already on disk: Qwen 2.1's TE is Qwen3-VL-8B, vision tower and lm_head only dropped at load (qwen21/single_file.py:10-13); Krea's TE is Qwen3-VL-4B with model.visual.* in the file (krea2/single_file.py:18, TE_DROP :195) and a tied lm_head (krea2/qwen3vl_4b_text_config.json:24); FIBO loads a full Qwen3VLForConditionalGeneration (fibo/loading.py:77-82). MOSS's Qwen3.5-9B drops visual./lm_head (moss/single_file.py:265). Whether Krea's 4B is still instruction-following is unverified; a stock Qwen3-VL-8B-Instruct is the safe default.

**Fit and risks.** The judge runs after rendering in the scoring thread (scorers are main-thread only, trainer.py:3296); it never touches the DiT's W8A8/nvfp4 kernels, compile, keep blocks or block swap. Its LM linears can reuse the int8 W8A8 path as fibo/loading.py does (triton-windows works); SDPA attention, no flash-attn, no NCCL. Memory: Qwen3-VL-8B int8 with vision is about 9-10 GB and cannot stay resident beside a side-GPU DiT replica (Krea nvfp4 about 7 GB, Qwen 2.1 int8 about 8 GB); load it after `_free_for_rendering` or keep it in pinned RAM. Krea's 4B int8 (about 5 GB) fits more easily. The real cost is rendering: 4 prompts x 2 seeds = 8 extra turbo renders per edit probe. Judge noise (Spearman about 0.63-0.75 per image) means 6-8 images per probe for a stable mean. VQAScore saturates on easy prompts; normalize against the step-0 base render. Keep the judge out of every gradient path.

**Implementation path.**
1. config.py: `probe_edit_prompts` (an "identity" preset of 4 prompts), `probe_edit_every` (a multiple of probe_every; every 4th probe keeps overhead under about 5% of a 1000-3000 step run), `judge_model` (auto: the run's own Qwen3-VL TE file for Qwen 2.1, Krea, FIBO; else stock Qwen3-VL-8B-Instruct). Add the prompts to prepare_cache (trainer.py:580-582), since the TE is unloaded after caching.
2. New `rulztrainer/judge.py`: build Qwen3VLForConditionalGeneration from the TE file keeping visual.* and lm_head, LM linears through quant.py; P("Yes") from the " Yes"/" No" logits.
3. `trainer._probe_with`: edit RenderJobs (kind "edit") on edit-probe steps, scored by ArcFace and the judge.
4. ProbeRecord: `adherence`, `adherence_norm` (floor = step-0 base render), `overfit_rate`.
5. LikenessController vetoes save_best when adherence_norm drops more than a band below its best; adherence as an optional race-score term.
6. Reuse the judge as a selftrain gate and in reward-stage logging.

**How to test.**
1. Offline on krea2-AZav-rulz: render 4 edit prompts x 2 seeds per saved checkpoint, hand-label follows-the-prompt yes/no. Require judge AUC >= 0.8 and check adherence falls in late checkpoints while ArcFace still rises. If adherence is flat across checkpoints, stop.
2. Race best-by-likeness vs best-by-likeness with adherence veto, 2 seeds each, Krea and Qwen 2.1, 2000 steps.
3. Compare the picks on 12 new edit prompts x 4 seeds (normalized ArcFace, judge adherence) and a blind side-by-side.
4. Accept at >= 5 normalized adherence points for <= 2 points of face likeness lost; reject if the veto never changes the pick across 4 runs.

### Asynchronous probes on an idle second GPU

**Prototype. Expected gain: about 11% wall time at probe_every 40 and 16% at probe_every 25, measured on Qwen 2.1 int8; Krea with a separate Turbo sampler maybe 20-40% (unmeasured). Only where GPU 1 would otherwise idle. Confidence: medium (0.5). Cost: medium-large (render worker thread, lagged decisions, snapshot-based best saves).**

**What it is.** Snapshot the LoRA (and EMA) to CPU and hand it to a persistent render worker on GPU 1 that holds its own frozen base, VAE, turbo sampler and scorers. Training continues on GPU 0; the controller and race act on the score one probe interval later. In a turn-based race, racer B trains while racer A's probe renders.

**Evidence.**
- Measured locally (output/qwen21-test/train.log): Qwen 2.1 int8 at about 2.5 s/step; a default probe (face + body, 8 turbo steps) costs 12.2-12.4 s; the step-0 probe 42 s. That is about 11% of wall time at probe_every 40 and 16% at 25.
- [Jaderberg et al. 2017](https://arxiv.org/abs/1711.09846) and the [Isaac Lab PBT guide](https://isaac-sim.github.io/IsaacLab/main/source/features/population_based_training.html) describe asynchronous population members and the interval / checkpoint I/O trade-off; they are about population search, not moving one run's evaluation off its GPU.
- Closer evidence from the verifier: [HF "Async GRPO with LoRA across HF Jobs"](https://huggingface.co/blog/asyncgrpo-lora-hfjobs) syncs only the adapter (a few MB) to inference replicas, tolerates up to 4 versions of staleness (mean 1.5-2.0), median weight sync 6-8.5 s. LLM RL, not image LoRA. No diffusion LoRA trainer (ai-toolkit, OneTrainer, kohya) is known to do this.

**RulzTrainer today.** side_devices() (trainer.py:426-431) returns both GPUs only with multi_gpu on and parallel_jobs (README.md:281); probes then split across GPUs but stay synchronous. `_run_probe` (trainer.py:3319-3325) runs frozen_adapters, `_sampling_slots`, `_probe_with`, controller.update inline. With sample_model_path, `_sampling_slots` (trainer.py:2199-2245) moves the training model to CPU and back on every probe. No persistent render worker, no overlap.

**Fit and risks.**
- Where it applies: multi_gpu off with a second CUDA device (block_swap forces multi_gpu off, config.py:592), or a turn-based single-GPU race. With data_parallel or model_split both GPUs already train; data_parallel gives about 1.6-1.9x throughput, far more than this, so for typical 2x16 GB runs this is a niche win.
- Windows: torch.multiprocessing cannot share CUDA tensors via IPC on Windows; the scout's spawn-process design is wrong. Use a daemon thread like multigpu.py's GPU workers, applying the CPU state dict with `mirror_lora` (lora.py:529).
- torch.compile: two threads tracing at once crash dynamo (trainer.py:2266-2268); keep the render copy eager. W8A8/nvfp4 triton kernels run eager anyway; triton-windows cache races between two threads are unmeasured.
- Memory on the render GPU: MOSS int8 (about 5 GB) and Qwen 2.1 int8 (about 7-8 GB) fit with VAE, ArcFace and DINO; Krea fits only as nvfp4 (about 7 GB), so the 5060 Ti must be the render GPU; Krea int8 (about 12-13 GB) is too tight on the 4060 Ti.
- Decisions lag one interval: early stop trains up to probe_every extra steps; best saves must write the scored snapshot, not live weights; snapshot the EMA inside `_averaged()`. Step 0, final, reward-stage and relay round-end probes stay synchronous.

**Implementation path.** `probe_async: off | auto | on` (auto = multi_gpu off and devices[1] exists). New `rulztrainer/probeworker.py`: daemon thread pinned to that device, builds the sampling model uncompiled (sample_model_path or training base), its own VAE (`_vae_on`), FaceScorer and BodyScorer, pulls (step, racer, CPU LoRA/EMA state, tag) jobs and reuses `_probe_with` job building and scoring. trainer.py: async run_probe only snapshots and enqueues; `_drain_probe_results()` at each step boundary runs the existing controller.update, best-save and actions. This is also what would make 16+ renders per decision (the paired-test item above) affordable.

**How to test.** Qwen 2.1 int8 and MOSS int8, 1500 steps, probe_every 25, probe_seeds 2, same seed: (a) multi_gpu off, sync probes; (b) multi_gpu off, probe_async on GPU 1; (c) data_parallel with split sync probes. Metrics: wall clock to final step and to best step, best normalized combined likeness, step of the best (lag check). Run (b) twice and confirm scores match (a) within likeness_noise_band. Also a 3-config turn-based race with and without async. Keep async if (b) cuts wall time >= 8% vs (a) with best likeness inside the band; document it only for setups where (c) cannot run (block_swap, Krea int8).

### Face probe statistic: held-out pool, copy-paste score and pool-mean reward

**Race it (the reward-stage pool mean); log the rest. Expected gain: unmeasured; no speed or VRAM cost. Confidence: medium (0.45). Cost: small.**

**What it is.** Three changes to how a face render is scored against references: (a) a held-out reference pool never trained on, (b) WithAnyone's copy-paste score M_CP = (theta_gt - theta_gr) / max(theta_tr, eps), theta = arccos(cosine), from +1 (sits on the reference) to -1 (sits on the unseen photo), and (c) for the reward stage, a pool-mean or top-k-mean ArcFace reward instead of the max, following IPRO's Facial Scoring Mechanism.

**Evidence.**
- [WithAnyone (arXiv 2510.14975)](https://arxiv.org/html/2510.14975v1), ICLR 2026: Table 1a InstantID Sim(GT) 0.464 / Sim(Ref) 0.734 / CP 0.337; UMO 0.458 / 0.732 / 0.359; WithAnyone 0.460 / 0.578 / 0.144. Refuted premise: Sim(GT) tracks Sim(Ref) across all 15 methods and the copier InstantID has the best Sim(GT), so held-out similarity alone does not demote copiers; only CP separates them. Also t there is the prompt's target group photo, not a free held-out portrait.
- [IPRO (arXiv 2510.14255)](https://arxiv.org/html/2510.14255v1), CVPR 2026, Table 4 on Wan2.2: base 0.578 FaceSim / 7% hacking; without KL 0.754 / 58%; without FSM 0.739 / 52%; full 0.694 / 10% (Gemini 2.5 Pro judge). Image-to-video with differentiable reward backprop; RulzTrainer's reward stage is GRPO with a non-differentiable scorer and already has KL.
- [Expression-diverse references (arXiv 2610.11023)](https://arxiv.org/html/2610.11023v1): single-reference ArcFace falls with expression intensity (Spearman -0.662 on 70 Lightstage photos) and argues for max over a diverse gallery, which cuts against replacing max for probes.
- [Phota (arXiv 2609.04151v2)](https://arxiv.org/html/2609.04151v2): normalized-mean identity anchor and match rate R(tau), separation strongest above 0.6.

**RulzTrainer today.** face_refs (config.py:328) accept arbitrary paths, but there is no second pool. FaceScorer.score takes the max cosine (facesim.py:207-217). The scale's 100 is the median best match among refs (likeness.py:72-89), capped at NORM_MAX 100 because a copying probe reads higher (likeness.py:65; README.md:963). The reward stage's `_reward_of` uses the same max scorer (trainer.py:2915-2935), clipped to PART_MIN..PART_MAX.

**Fit and risks.** CPU post-processing of existing embeddings; no training-stack interaction. Problems: 10-30 photo datasets from few sessions make a 2-3 photo hold-out cost 10-20% of the data and make CP meaningless if same-session; with 1-2 probe seeds, CP (an angle difference over theta_tr of about 0.8-0.9 rad) is noisy and a hard `likeness_cp_max` gate could flip decisions on noise; a held-out ceiling would move every existing calibration; races need one split per race, not per racer.

**Implementation path.** (1) reward: `reward_face_stat: max | mean | topk` (default max) in `trainer._reward_of`. (2) facesim: optional `face_holdout_refs` (folder, or int auto-split in dataset.py, one split per race); score() returns best_train, mean_topk_train, sim_holdout, cp; keep max as the default likeness. (3) cp and sim_holdout in ProbeRecord and the UI as diagnostics. (4) opt-in `likeness_score_source: train | holdout` and a soft `likeness_cp_penalty` (weight x max(0, cp - margin)), not a hard gate. README Likeness section and wiki.

**How to test.** Reward stage: race reward_face_stat max vs mean from the same start LoRA and seeds, 200-400 reward steps; metrics held-out ArcFace on varied-expression prompts, expression diversity across seeds (std of a landmark-based expression score), and an eye or VLM check for stiff copied faces; keep mean if held-out likeness is within noise and diversity is higher. Diagnostic: on 2-3 datasets with 3 genuinely different-session photos held out, Krea nvfp4 and Qwen 2.1 int8 to 3000 steps, probe_every 100, probe_seeds 4; log sim_train, sim_holdout and CP; check whether CP rises late while sim_train climbs and whether argmax checkpoints differ; adopt holdout/CP selection only if it wins a blind comparison (plus FaceNet/AdaFace vs the held-out photos on 16 varied prompts) on most datasets.

### Impostor-margin face probe against the strangers gallery

**Race it (offline first). Expected gain: unmeasured; probably small for faces, possibly more for bodies; only changes which checkpoint or racer is kept. Confidence: medium (0.4). Cost: small (32-100 extra embeddings per probe, under 1 s on CPU for ArcFace).**

**What it is.** Score each probe face not only by its best cosine to the references but by the margin over a gallery of look-alikes: margin = best_ref_cos - mean of the top-3 cosines to the gallery. The gallery is the 32 LoRA Isolation strangers already on disk, other datasets' references in multi-dataset runs, and the step-0 base-model probe renders (the real look-alikes). This catches runs drifting toward a generic, attractive face, which raise cosine to the strangers and base renders as well as to the references.

**Evidence.**
- [Finer-Personalization Rank (arXiv 2512.19026)](https://arxiv.org/html/2512.19026): gallery of 10 subjects x about 10 images; CLIP similarity puts every method at about 80% while gallery mAP falls sharply; DreamBooth (SD3-Medium) preserves identity best; copy-paste scores almost perfectly. Birds, cars, animals only; no faces, no human study; the ethics statement discourages use on human biometrics.
- [NearID (arXiv 2604.01973)](https://arxiv.org/html/2604.01973): frozen SigLIP2-so400m-384 plus a ~15M MAP head; 19,386 identities, 316,505 matched-context distractors; SSR 30.74 -> 99.17% (CLIP 10.31, DINOv2 20.43, Qwen3-VL-30B 49.73); DreamBench++ Pearson 0.516 -> 0.545; ~6.5 A100-80GB hours. Targets rigid objects, scopes out faces. Weights public (HF Aleksandar/nearid-siglip2).
- Refuted claim: "ArcFace cosine over-rewards drift toward the class mean" has no supporting source; ArcFace is a discriminative face recognizer, unlike CLIP/DINO. Pure rank-1/mAP against unrelated strangers saturates at 1.0 within a few hundred steps (good likeness 0.5-0.7 cosine, strangers 0.1-0.25), so it cannot rank race configs; use the continuous margin.

**RulzTrainer today.** Scores are already rescaled so 0 = the step-0 base render (a same-prompt, same-seed "stranger", at least probe_floor_seeds 4) and 100 = a typical reference (likeness.py:19-35, 54-60; README.md:938-960; trainer.py:3150-3155). LoRA Isolation keeps 32 pinned strangers, 8 ethnicities, men and women (isolation.py:40, 163-178). Races, early stop, best checkpoint, lr control and selftrain all read the controller's normalized scores, so a new metric plugs in at one point.

**Fit and risks.** No training compute interaction; InsightFace ONNX on Windows. SigLIP2/NearID for bodies needs about 0.9 GB fp16 on the side GPU or a few seconds per crop on CPU. Skip: Arc2Face look-alike synthesis (an SD1.5-sized extra pipeline), per-character NearID head training (overfits on 10-30 photos, hours of GPU, outside the paper's scope), rank-1/mAP as a race criterion.

**Implementation path.** facesim.FaceScorer.set_gallery(paths) embedding isolation.strangers(), other datasets' refs and the cached step-0 renders; score() also returns impostor (top-3 mean), margin and rank; `likeness_metric: cosine | margin` (default cosine) with margin normalized by floor = step-0 margin and ceiling = the references' leave-one-out margin; rank-1/mAP logged only. Optional `body_backend: nearid` in bodysim.py on the body window, same margin path. Optional FaceNet agreement term in reward.py (overlaps the gold-recognizer item above; build it there).

**How to test.** Offline: re-score saved probes (krea2-AZav-rulz, earlier race jobs) with cosine and margin; Kendall tau against a blind human ranking of 10-20 checkpoints per character; check margin flags checkpoints whose stranger/base cosine rises with reference cosine. Then race likeness_metric cosine vs margin on 2-3 characters (Qwen 2.1 int8 0.5 MP, then Krea nvfp4), same configs and seeds, 2000 steps, probes every 100; blind A/B of the two picks on 8 held-out prompts, FaceNet face cosine and body DINOv2 with neutral outfit/background as secondary measures. Keep margin as default only if the user prefers its pick on at least 2 of 3 characters and its FaceNet score is not lower. For NearID body: keep only if it agrees better with human judgement than DINOv2 when outfit and background are copied.

### Run-history recommender for the wizard and race rows

**Do it (small scope). Expected gain: unmeasured; once 5-10 runs per model exist, a first race could shrink from about 4-6 racers to 2-3, roughly 1-3 single-run equivalents saved per new character (2.5-7.5 h on Krea nvfp4 at about 9 s/step), if winners repeat across characters. Confidence: medium (0.45). Cost: small (more fields in the existing history JSON, a kNN suggester, a wizard button).**

**What it is.** Store dataset meta-features, the racers' actual settings and the winner with each run, then let the wizard suggest race rows from the user's most similar past runs on the same model and quant. Transfer BO (RGPE, PriorBand, FT-PFN context) is deferred until some model has more than about 20 runs.

**Evidence.**
- [Feurer et al., RGPE (arXiv 1802.02219)](https://arxiv.org/abs/1802.02219): hyperparameter-free GP ensemble with a worst-case bound against plain BO, "substantially" less optimization time (no figures in the abstract).
- [NePS PriorBand](https://automl.github.io/neps/latest/reference/search_algorithms/multifidelity_prior): mixes prior, incumbent and random sampling per rung, shifting away from a bad prior.
- [automl/QTT](https://github.com/automl/QTT): BSD-3, meta-learned fine-tuning HPO; pretrained optimizers cover image classification only.
- [FT-PFN (arXiv 2404.16795v3)](https://arxiv.org/html/2404.16795v3): zero-shot curve surrogate; its context is the current task's partial curves; cross-task transfer from old runs is untested.
- [Lourie, He, Cho (arXiv 2510.02721)](https://arxiv.org/abs/2510.02721), COLM 2025: confidence intervals for the best achievable score from random search.
- All evidence comes from HPO benchmarks with hundreds to thousands of mostly deterministic evaluations. Here each evaluation is an hours-long run with a noisy metric, and one user produces tens of runs over 4 models.

**RulzTrainer today.** Refuted premise: cross-run memory already exists. history.py:1-35 and :224-262 write output/_history/<run>.json with arch, run id, state, steps, a CONFIG_KEYS subset (history.py:27), per-racer probe series under probes.racers, probe_raw scores and probe images; the WebUI lists, overlays and rescores them (webui.py:4828-4888, 4994; README.md:1123). Missing: dataset meta-features (the reference ceiling is computed in likeness.py but not stored), race_configs (config.py:458, not in CONFIG_KEYS, so racers' settings are lost), GPU, quant and wall-time fields, the code version, and any recommender (wizard_recommendations at webui.py:2970 is rule-based). output/_history holds 3 runs locally.

**Fit and risks.** Pure bookkeeping, no GPU-side interaction. Configs drift with code versions (probe scales changed at v2.9.0, already handled by rescoring), so store the RulzTrainer version. Model and quant must be part of the match (Krea nvfp4 vs int8 differ in speed and scores). vast.ai runs write history on the rented box, so the vast scripts must copy it back. No face embeddings in meta-features. The user's own finding that LoRA beats LoKr everywhere hints that wins are mostly model-dependent; if so, a per-model "best past rows" list gets most of the value without meta-features. Flux 3 Klein and MiniMax H3 start from wizard defaults.

**Implementation path.** history.py: add race_configs, transformer_quant, network_alpha and the rest of RACE_KEYS (race.py:49) to CONFIG_KEYS; build_record adds winner (name plus merged settings), version, gpus, wall_s and a data block (image count, bucket histogram, median face-area fraction, caption length, reference ceiling/floor). trainer.py computes the meta-features once after caching. `history.suggest(arch, quant, meta, k=3)` ranks by z-scored meta-feature distance weighted by best combined score and returns winner and runner-up rows. WebUI: "Suggest race from my past runs" next to the wizard recommendations, hidden under 3 matching runs. vast scripts copy output/_history/<run>.json back. A SQLite DB is unnecessary at this size.

**How to test.** Offline replay: hide each stored race, let suggest() pick from the rest, and count how often the actual winner (or a row within likeness_noise_band of it) is in the top-2, against the wizard defaults. Live, 2-3 new characters per model: A = default wizard race with 4-6 rows and drop_lowest, B = the 2-3 suggested rows. Success: B within 1 point of A's best normalized score using at least 30% fewer race steps on 2 of 3 characters. Stop: B more than 2 points below A on 2 characters means winners do not transfer; keep only a per-model best-rows list.

### Low-resolution early race rounds with a rank check

**Prototype (measure the rank correlation first; that needs no code). Expected gain: 20-40% of the race phase's wall clock, about 10-20% of a 3000-step run (estimate); per-step speedup 1.5-2.5x at 1 MP base and 1.3-1.7x at 0.5 MP, not the scout's 4x. Confidence: low (0.35). Cost: medium (relax a validate() rule, a run-level batcher switch, smoothing reset).**

**What it is.** Run the warm-up and first race rounds with every racer on the lowest-resolution bucket (0.25 MP), then switch the shared batcher to the run's normal resolutions for the rest. A within-run Kendall tau between the racers' order at the last low-res probe and the first full-res probe tells whether the cheap rounds can be trusted. FT-PFN curve extrapolation is dropped.

**Evidence.**
- [Egele et al. (arXiv 2307.15422)](https://arxiv.org/abs/2307.15422): one-epoch-then-top-K matches multi-fidelity HPO with about an order of magnitude less compute on classic benchmarks, which the authors themselves call possibly too easy.
- [Zhang and Duh, AMTA 2024](https://aclanthology.org/2024.amta-research.12.pdf): successive halving's early-final correlation "may not always be true", generally holds for NMT under appropriate setups (2,469 models trained from scratch).
- [FT-PFN / ifBO (arXiv 2404.16795v3)](https://arxiv.org/html/2404.16795v3), [automl/ifBO](https://github.com/automl/ifBO): 14.69M params, up to 10 hyperparameters, 0.23 s vs 17.7 s (DPL) at 400 points; its prior assumes improving, convex, converging curves, a poor match for likeness curves with lr-control spikes, step-up dips and a reward stage; requires Python 3.11; the "60 MB" weight size is unverified.
- [Language-guided BO for LoRA (arXiv 2602.11171)](https://arxiv.org/abs/2602.11171), ICML 2026: data-subset proxy with strong correlation, more than 20% over standard hyperparameters, LLM LoRA only.
- No source tests image resolution as a fidelity for a diffusion LoRA or for likeness metrics.

**RulzTrainer today.** The race already does successive elimination (race.py:258-276 estimate_steps; RaceSchedule at race.py:278), with patience drops and race_patience_rising for slow starters (config.py:461-465). Resolution step-up exists (config.py:219; `_enter_stage` at trainer.py ~2328) with per-resolution latent caches, but race.py:194-195 refuses race + resolution_step_up, so the scout's "promote via step-up" path does not work as written. Resolutions are shared by all racers (not a RACE_KEY), which helps: a rung is a run-level switch that keeps the race fair. Probes render at fixed probe_size 512 (config.py:433), so scores are comparable across rungs but may miss high-resolution face detail.

**Fit and risks.** Only a batcher rebuild: W8A8, nvfp4, block swap, the sigma grid and exports are unaffected; compile recompiles once per new token count, as step-up already does. checkpoint_keep_blocks is calibrated for the largest bucket and is conservative at 0.25 MP, losing part of the speedup unless recalibrated per stage. Probe cost per round is fixed and dilutes the gain. Main risk: settings whose effect grows with resolution (id_loss with id_loss_min_face_cells skipping small faces, region weights, face_margin, face crops) may rank differently at 0.25 MP.

**Implementation path.** `race_low_res_rounds` (int, default 0): for the first K rounds every racer trains on `_enter_stage(0)` with items_for_side, then the shared batcher switches to the normal set. Relax race.validate (race.py:194-195) only for this mode. Reset each racer's smoothing baseline at the switch and do not drop anyone in the first full-res round. Log within-run Kendall tau to race.json and history; warn below 0.5. WebUI field on the Race rules page with the estimate_steps cost. Leave FT-PFN out (at most an offline script on output/_history).

**How to test.** Ground truth: Krea nvfp4 and Qwen 2.1 int8, two datasets, 4 rows (lr 1e-4 vs 2e-4, rank 8 vs 16, id_loss on, region weights on), race_drop_lowest false, full 2000 steps at the normal 0.5-1 MP. Then the same rows with race_low_res_rounds 2 at 0.25 MP, drop-lowest on, warm-up 1, probe_seeds >= 2. Adopt (and offer it in the wizard) if tau >= 0.6 in at least 3 of 4 races, the winner is within the noise band every time, and race wall clock drops by at least 25%. If an id_loss or region-weight row is last at low res but first at full res, mark those keys full-fidelity only. Before writing code, the rank question can be answered with two plain races of the same rows, one with resolutions limited to the low bucket and one at normal resolution, both race_drop_lowest false.

### Calibrating face/body weights from the user's own blind picks

**Skip-ish (keep the A/B panel idea; the margin drop is folded into the paired-test item above). Expected gain: unmeasured; a convenience gain on the final pick, zero step time. Confidence: medium (0.4). Cost: medium (WebUI panel, a small preference fit).**

**What it is.** A WebUI panel shows blinded, randomly ordered pairs of existing probe renders (same prompt and seed) from two checkpoints (best, best_face, best_body, last) and records which one looks more like the subject. A logistic Bradley-Terry fit P(A>B) = sigmoid(w . (axes_A - axes_B)) with an L2 prior toward the config weights then suggests a face_weight. The user knows the subject, which is the familiar-rater setting.

**Evidence.**
- [Phota (arXiv 2609.04151v2)](https://arxiv.org/html/2609.04151v2): about 1000 blinded pairwise comparisons over 10 consenting subjects, judged only by people who know them; Elo with 95% bootstrap CIs; "embedding similarity does not necessarily correspond to perceived likeness"; identity, VLM adherence and VLM quality kept separate.
- [DreamBench++](https://arxiv.org/html/2406.16855v2): CP x PF leaderboard.
- [Prediction-Powered Ranking (arXiv 2402.17826)](https://arxiv.org/abs/2402.17826), NeurIPS 2024: few human plus many judge comparisons give rank-sets with coverage; the guarantee is asymptotic and the experiments used thousands of LMSYS human comparisons. With 20-50 picks, rank-sets will be too wide to drive elimination.
- [Finer-Personalization Rank](https://arxiv.org/html/2512.19026): no human correlation, no adherence axis.

**RulzTrainer today.** likeness.combine (likeness.py:47) with face_weight 0.7 / body_weight 0.3 (config.py:340-341) feeds one smoothed score into best, patience, lr control and step-up (likeness.py:247, 268-300); separate best_face/best_body checkpoints exist (config.py:444). webui.py:4567-4590 already rescales saved runs with a common face_weight, webui.py:6092 exposes the weights. No pairwise store, no fit. Three of the four proposed Pareto axes (adherence, copy, quality) do not exist, so with only face and body the fit reduces to one face/body ratio.

**Fit and risks.** No compute or VRAM cost, no model-specific code. Races run unattended for hours, so mid-race picks are impractical; only weights stored from earlier runs could steer a race. Neighbouring checkpoints are often near-indistinguishable, so many picks are coin flips.

**Implementation path.** webui.py "Pick" tab writing `<job>/human_pairs.jsonl` and `output/_human_pairs.jsonl` with each candidate's axis vector; new `rulztrainer/preference.py` fitting the logistic model, reporting per-axis agreement, refusing to suggest under 20 picks; a "use learned weights" button for new runs. Postpone Pareto fronts and PPR rank-sets until adherence and copy scorers exist.

**How to test.** Collect 40 blinded picks over 3-4 past runs, hold out 10, compare held-out pairwise accuracy of learned weights vs 0.7/0.3. Keep the feature only if it beats the default by 10 points or more and the learned face_weight differs from 0.7 by more than 0.1.

### Exploit-only-losers relay with mutation (small-population PBT)

**Skip-ish (race the exploit-only-losers change once the paired test exists). Expected gain: unmeasured; at most about the probe noise band, more often nothing. Confidence: low (0.35). Cost: medium; same compute as the relay (on Krea nvfp4 at about 9 s/step, 4 x 2000 steps is about 20 h).**

**What it is.** In the relay, copy the winner's LoRA and optimizer only into racers that are clearly worse (below max(likeness_noise_band, k x std across racers), Isaac style), leave the second-best alone, and perturb the copied racers' non-shape floats by x0.8 or x1.25 with probability 0.25 (learning_rate or automagic_max_lr, timestep_shift, logit_mean/std, clean_share, lora_dropout, caption_dropout, id_loss_weight, dop_weight). The race then reports a discovered per-round schedule rather than a static winning row.

**Evidence.**
- [Jaderberg et al. 2017](https://arxiv.org/abs/1711.09846): exploit/explore yields hyperparameter schedules (RL, MT, GANs).
- [Isaac Lab PBT](https://isaac-sim.github.io/IsaacLab/main/source/features/population_based_training.html): only underperformers copy a random leader; mutation_rate 0.25, change range [1.1, 2.0]; cutoffs mu +/- max(threshold_std*sigma, threshold_abs), defaults 0.1 / 0.025.
- [MF-PBT (arXiv 2506.03225)](https://arxiv.org/html/2506.03225v1): 32 agents in 4 sub-populations (delta 1/10/25/50), Humanoid IQM at 1B steps 23793 vs PBT 16171 vs random search 17713, on 2x A100 40 GB; notes PBT propagates seed luck. The RLC 2025 venue is unverified.
- [Anyscale PB2 blog](https://anyscale.com/blog/population-based-bandits): population 4 vs 8 claim is qualitative (a plot, no numbers).
- Contrary: [FIRE-PBT (arXiv 2109.13800)](https://arxiv.org/pdf/2109.13800.pdf) shows plain PBT at 71.7 vs hand-tuned 76.6 on ImageNet from greediness. With 2-4 racers and about 8-20 exploit events (probe_every 100-250), PB2's GP over about 10 dimensions and MF sub-populations do not carry over; the RL +47% / +34% numbers do not transfer.

**RulzTrainer today.** race_follow: RaceSchedule.record names the max-score leader (race.py:384-386); Trainer._race_follow (trainer.py:1848-1854) calls follow_leader (race.py:677-713) for every other racer, copying LoRA, optimizer state (same type), EMA and controller; every racer keeps fixed settings; no mutation. SHAPE_KEYS (race.py:81) already bans shape rows in a relay. validate() rejects race_follow with race_parallel (race.py:205-206), so "4 racers in two waves on 2 GPUs" needs new work. likeness_lr_control already gives a single-agent probe-driven lr schedule.

**Fit and risks.** Only LoRA state moves, so W8A8/nvfp4 and memory settings are unaffected; parked racers live in system RAM. A mutated lora_dropout swaps the dropout module and may cost a recompile. learning_rate must be pushed into optimizer param_groups and scheduler base_lrs; under automagic3/4 mutate automagic_max_lr instead. REGION_KEYS (race.py:82) are baked into cached region maps (race.py:723-730), so exclude them. Timestep keys change the loss scale, fine because ranking is on probes. The winner yaml must become a per-round schedule.

**Implementation path.** `race_pbt: bool` alongside race_follow; record() computes losers; `_race_follow` calls follow_leader only for losers; new `race.mutate(v, rng, keys, bounds)`; per-round overrides in schedule.history for write_report and `_winner.yaml`. Skip PB2 and MF sub-populations.

**How to test.** Qwen 2.1 int8 then MOSS, 2-3 characters, equal compute: (A) current relay with 4 seed rows, (B) pbt with the same rows plus mutation, optionally (C) one run with likeness_lr_control at 4x probe seeds; 4 racers x 2000 steps, probe_every 200. Re-probe each arm's final and `_best` LoRA offline with 16+ fresh seeds (in-race scores are max-selection biased). Adopt only if B beats A by more than likeness_noise_band on 2 of 3 characters; drop the idea if within noise on the first two.

## Deployment compatibility and data attribution

This section covers two gaps found in the completeness critic's sweep. The first is whether the likeness RulzTrainer measures during training holds up when the LoRA is actually used: merged and requantized by ComfyUI, run on GGUF bases, stacked with turbo and style LoRAs, or run at a strength other than 1. The second is which training images help or hurt likeness. None of these items raises likeness per step. The useful pieces are four:

- a ComfyUI merge-and-requantize deploy check for nvfp4 and fp8;
- an alone-versus-stacked probe pass;
- a probe-backed recommended strength;
- a selftrain sigma floor, which is a few lines of code and can be raced.

Per-image attribution (gradient, DataInf, unlearning) has never been tested on ranking photos of one person, so each method has to pass a stability gate before any UI work. The accuracy-recovery adapter and the null-space compose guard are measured or predicted to do little.

### 1. ComfyUI merge-and-requantize deploy check

**Do it (the weight-space check and one comfy_merge probe column only). Expected gain: none directly, since this is a diagnostic. A rough signal-to-noise estimate of about 5-12 suggests nvfp4 may lose a few ArcFace points and int8/fp8 a negligible amount (unmeasured). Confidence: medium (0.6). Cost: small build; the weight-space check runs in minutes, and one extra turbo probe set takes about 1-2 min on Krea 2 nvfp4 at 8 steps.**

**What it is.** When ComfyUI fully loads a quantized file, it does not keep the LoRA as a side branch. It dequantizes the weight, adds the delta in lora_compute_dtype, and requantizes the merged weight with a recalculated scale and stochastic rounding seeded from `string_to_seed(key)`. RulzTrainer trains and probes x·Q(W) + x·ΔW instead. Per element, a LoRA delta of about 1-5% of |w| is smaller than the quantization step of fp8 e4m3, of int8 per-group absmax, and especially of nvfp4 (e2m1 per 16-element block). Stochastic rounding keeps the delta in expectation but adds noise per weight. In a matmul the coherent signal grows with fan-in and the noise only with its square root, so the gap may be small for int8/fp8 and larger for nvfp4. Only a measurement can say.

**Evidence**
- [ComfyUI ops.py](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/ops.py): `set_weight` calls `requantize_from_float(weight, scale="recalculate", stochastic_rounding=seed)`, and `post_cast` seeds it with `string_to_seed(seed_key)`. The verifier found the requantization is conditional:
  - The fast quantized matmul runs only when weight_function and bias_function are empty.
  - `cast_bias_weight` patches in float and never requantizes.
  - The lowvram `post_cast` requantizes only when `want_requant` is set and no regular weight_function exists.
  - So in some lowvram or offload states ComfyUI applies the LoRA in float, exactly as RulzTrainer does.
  - There is no layout named "int8_convrot". The format is int8_tensorwise with a convrot flag; convrot_w4a4 is a separate format.
- [ComfyUI model_patcher.py](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/model_patcher.py): the merge happens in lora_compute_dtype. Plain fp8 weights then go through `comfy.float.stochastic_rounding(..., seed=string_to_seed(key))`.
- [city96 ComfyUI-GGUF ops.py](https://raw.githubusercontent.com/city96/ComfyUI-GGUF/main/ops.py): dequantizes and runs `calculate_weight` on every forward and never patches the packed tensor. GGUF therefore behaves like RulzTrainer's side branch; only the base differs.
- [SVDQuant](https://arxiv.org/abs/2411.05007): LoRAs run in a high-precision branch with no requantization. The paper reports 3.5x memory savings, 3.0x over W4A16 on a laptop 4090 (INT4) and 3.1x on an RTX 5090 (NVFP4). The [Nunchaku README](https://github.com/nunchaku-tech/nunchaku) lists FLUX.1 (including Krea-dev, which is not Krea 2), Qwen-Image v1/Edit, Z-Image and SANA. None of the five target models is on the list.
- Deployment builds that exist:
  - [unsloth/Qwen-Image-2.1-GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF): Q2_K-Q8_0 plus F16, with a Qwen3-VL-8B UD-Q4_K_XL text encoder recommended.
  - [molbal/krea2-gguf](https://huggingface.co/molbal/krea2-gguf): Q4_0-Q8_0, non-K quants only, described as "very close to BF16" from Q5_0 up. It needs molbal's ComfyUI-GGUF fork.
  - [molbal/MiniMax-H3-GGUF](https://huggingface.co/molbal/MiniMax-H3-GGUF): Q4_0/Q5_0/Q8_0/Q8_CR/U16G. The FL2VA build was pruned to FP8 first.
  - [Gluttony10 INT8-ConvRot](https://huggingface.co/Gluttony10/MiniMax-H3-INT8-CONVROT): an H3 bundle with turbo LoRAs.
  - [Flux 3 Klein](https://github.com/black-forest-labs/flux-3-image-klein): fp8r and nvfp4 official, int8 in the -comfy repo, and Base is the checkpoint meant for LoRA training.
  - None of these model cards says anything about LoRA behaviour.
- No source measures how a LoRA transfers across quantization for image DiTs. The nearest data point is [CoA-LoRA](https://arxiv.org/html/2509.25214v4): a shared adapter loses 5.06% on SST-2 against per-configuration QLoRA. That is an NLU setting (RoBERTa-Large GLUE), and the magnitude does not transfer.

**RulzTrainer today**
- `quant.py:1062 quant_weight_from_comfy` loads ComfyUI int8_tensorwise+convrot and nvfp4 files in ComfyUI's own layout. Before the LoRA is applied, the training base is therefore already the deployment base.
- `sample_model_path` / `sample_model_quant` (config.py:470-471, trainer.py:2178-2195) can probe on a different base, but they keep the LoRA as a side branch and cannot load GGUF.
- `delta_weight` (lora.py:330) already returns the fp32 delta in the form ComfyUI merges.
- `quantize_int8` (quant.py:104) and `quantize_nvfp4` (quant.py:157) round to nearest; neither has a stochastic-rounding mode.
- fp8_scaled files are upcast to bf16. Krea 2 fp8_scaled users therefore train on the upcast weights, while ComfyUI keeps fp8 and merges with stochastic rounding. That is a real gap.
- Delta: stochastic-rounding quantizers, a ComfyUI merge emulation, an fp8 e4m3 path, and a deploy-check command.

**Fit and risks**
- **Model copy:** the merge must run on a separate model copy (the sample model on the side GPU), never on the training QuantLinear. The merged copy has no LoRA branch, so run it eager or give it its own compile. With the probe as a side job, block swap, model_split and data_parallel need no changes.
- **Memory:** a second Krea 2 copy fits on the 4060 Ti next to ArcFace and DINO at nvfp4 (about 7 GB) or Q4 GGUF (about 7.7 GB). A 12 GB int8 copy plus the probe models is tight on 16 GB.
- **Hardware:** matching ComfyUI's nvfp4 compute needs FP4 kernels, i.e. the 5060 Ti. On the 4060 Ti the dequantized-matmul fallback changes activation quantization.
- **Parity:** bit-exact parity is unrealistic, since it depends on comfy_kitchen internals and the device RNG stream. Statistical equivalence is enough.
- **Windows:** no issue; the code is pure torch, and the gguf package dequantizes in numpy.
- **Do not build the full matrix:**
  - The GGUF column is a base-mismatch question that sample_model_path already half answers. It would need a loader and a key map per family, and Krea GGUF needs the fork's architecture.
  - The Nunchaku column would emulate a runtime that does not exist for these models.
  - A quantized text-encoder column conflicts with the cached trigger embeddings, and transformers' GGUF support for qwen3-vl is uncertain.
  - Build 1-2 ComfyUI columns, not 3x7.

**Implementation path**
1. `quant.py`: add stochastic-rounding variants of `quantize_int8` and `quantize_nvfp4`, port ComfyUI's `string_to_seed`, add `comfy_merge_requant(qw, delta, key)` (dequantize, add delta, recalculate scale, stochastic round with a seeded `torch.Generator`), and add an fp8 e4m3 path for fp8_scaled sources.
2. Add a "deployment check" CLI command and WebUI button that takes a finished LoRA and its base file. It first reports the per-layer relative error ||Q_sr(W+ΔW) − Q(W) − ΔW|| / ||ΔW||, then builds a merged copy of the sample model, runs the turbo probe set, and writes `deploy.jsonl` beside `likeness.jsonl`.
3. Add the option `likeness.deploy_check: none | comfy_merge`, which runs once on the final or best checkpoint, not at every probe.
4. Later, if wanted, add GGUF through `gguf.quants.dequantize` into a bf16 base for sample_model_path, with per-family key maps.
5. Targets: Krea 2 (nvfp4, int8_convrot, fp8_scaled), Qwen 2.1 and MOSS (int8_convrot). Plan it into the Flux 3 Klein (fp8r, nvfp4, comfy int8) and H3 (int8 convrot, GGUF Q8_CR) loaders from the start. H3's pruned FP8 builds take different LoRAs, which is a separate compatibility issue.
6. README and wiki.

**How to test**
- **LoRAs:** one finished LoRA per case: Krea 2 trained on nvfp4, Krea 2 trained on fp8_scaled (upcast), and Qwen 2.1 and MOSS trained on int8_convrot.
- **Comparison:** side-branch probe against comfy_merge probe, on the same seeds and prompts, turbo at 8 steps, paired by seed. Metric: mean ArcFace face similarity plus DINO body.
- **Validation:** check the emulation once against a real ComfyUI render of the same seed with the LoRA loaded on the quantized file. ArcFace should agree within noise; pixel parity is not needed.
- **Stop rule:** if the paired drop is under 1 ArcFace point (3 seeds x 4 prompts) for int8 and fp8, report "safe" and keep the check off. If nvfp4 drops more than 2-3 points, turn the deploy check on by default for nvfp4 and look at remedies: recommend an int8 deploy file, or add a merge-aware fine-tune step.

### 2. Alone-versus-stacked probe pass for races

**Race it (first as a check that the two rankings agree). Expected gain: selection only. It helps only if racers rank differently alone and stacked, which no source shows for rank, alpha, target or EMA choices on a DiT. Confidence: medium (0.45). Cost: probe time rises about 1.5-3x per round, roughly +30-90 s per round on Krea 2 nvfp4 with an 8-step turbo. Each new strength value also costs one recompile under compile.**

**What it is.** The race and best-checkpoint logic score one likeness stream: the subject LoRA with the turbo LoRA, plus a style LoRA if `sample_lora` is set. They never score alone and stacked in the same run. The proposal adds a second probe pass each round, with a reference style LoRA on and the subject at 0.8 and 1.0. It logs `deploy_drop = alone − stacked` and lets the race rank on a blend.

**Evidence**
- [TARA](https://arxiv.org/html/2508.08812): single-concept DINO before and after merging. On SD1.5, DB-LoRA drops from 0.440 to 0.333 (−0.107) and TARA from 0.489 to 0.485 (−0.004). On SDXL, DB-LoRA drops 0.179 and TARA 0.005. One RTX A6000. In this paper "merging" means adding a second subject LoRA on a UNet, not a style LoRA on a DiT, so the magnitudes do not transfer.
- [Orthogonal MC Dropout](https://arxiv.org/html/2510.03262): orthogonality between LoRAs brings little benefit, and direct merge gives the best image quality. The claim that composability "must be measured on renders" is the scout's inference from this, not the paper's statement.
- [Nunchaku PR #754](https://github.com/nunchaku-tech/nunchaku/pull/754) (closed unmerged) documents bugs in the SVDQuant engine, not general stacking behaviour:
  - composed LoRA strengths are pre-baked;
  - Lightning LoRAs with per-block rank differences (128 vs 192) break CPU offload;
  - AMPLIFICATION_FACTOR was 2.0, later 1.0;
  - a tester reported noise with two composed LoRAs.
- Per the verifier, community reports say a style LoRA at 0.8 flattens character features, with some disagreement for large DiTs such as Flux.

**RulzTrainer today**
- `sample_lora` / `sample_lora_strength` (config.py:322-323, README.md:1428) already add a sampling-only second LoRA.
- `_attach_sampling_loras` (trainer.py:2146-2157) attaches the turbo LoRA and sample_lora as frozen extras, to every replica (trainer.py:850-852) or to the sample model (trainer.py:2192).
- `_probe_with` (trainer.py:3264-3271) turns the extras on, so probes already score the stacked view whenever sample_lora is set.
- `set_multiplier` (lora.py:579) gives ComfyUI-style subject strength.
- The race ranks on `controller.combined_ema` (race_parallel.py:344-351).
- Delta:
  - an origin tag on each extras entry, because `set_extras_enabled` flips turbo and style together today;
  - a second pass per round;
  - its own step-0 zero for the stacked stream, since a style LoRA shifts the "stranger" baseline;
  - `deploy_*` fields in `likeness.jsonl`;
  - a `race_score` blend.

**Fit and risks**
- **Quantization:** extras run unmerged as a separate floating-point matmul, so they work unchanged on int8_convrot W8A8 and NVFP4.
- **torch.compile:** compile runs with `specialize_float=True` (compiling.py:87-89). Each new multiplier value and each extras on/off state becomes a guard and costs one recompile per block graph. That is a one-time cost, but a real one on Krea's 28 blocks.
- **Block swap / model_split:** fine; they carry extras as buffers.
- **data_parallel:** every replica holds the extras. A turbo LoRA of rank 64-270 plus a style LoRA in bf16 is a few hundred MB per replica.
- **Score bias:** a style LoRA lowers ArcFace and DINO for every racer (anime or painterly faces score lower). Only the per-round ranking means anything, not the absolute stacked score.
- **Setup:** each model needs its own reference style LoRA.
- **Validation:** use held-out style LoRAs, or the race overfits to the one reference style.
- **Dropped part:** the "deployment-format requantized base" part has no evidence behind it. Item 1 covers it separately.

**Implementation path**
- Reuse `sample_lora` as the reference style LoRA. Add the options `deploy_probe: bool`, `deploy_probe_strengths: "0.8,1.0"` and `race_score: alone | min | blend:0.5`.
- `lora.py`: tag extras by origin (turbo / sample) and add `set_extras_enabled(model, on, origins=...)`, so the normal pass runs turbo only.
- `trainer._probe_with`: add a second pass with the sample LoRA on and `set_multiplier(subject, s)` for each strength, then restore 1.0.
- `likeness.py`: record `deploy_face`, `deploy_body` and `deploy_drop` against their own step-0 zero. Keep the controller (best checkpoint, stop, lr) on the alone stream, and give the stacked stream a separate EMA.
- `race_parallel.py:344-351`: return the blended score selected by `race_score`.
- This covers krea2, qwen_image21, moss_image and fibo through the shared probe path, and future families once they implement `extra_lora_converter`.

**How to test**
- **Step 1, does the arm carry information:** a 4-config race (rank 8 vs 16, lora_ema on/off, or default vs extended targets) on one character with `deploy_probe` on and `race_score: alone`, logging both streams. Compute the Spearman rho between the alone and stacked rankings each round. If the mean rho is 0.9 or higher across 2 datasets and 2 models (Qwen 2.1 int8 first at about 2.5 s/step, then Krea 2), stop and do not ship the blend.
- **Step 2, does it pick better configs:** compare the alone winner with the min winner on 2-3 other style LoRAs not used in the race, at 0.8 with the subject at 1.0, over 16 seeds, scoring ArcFace face and DINO body. Adopt `race_score: min` only if it beats the alone winner on held-out styles by more than the noise band while losing less than the noise band alone. Otherwise keep `deploy_drop` only as a logged diagnostic.

### 3. Probe-backed recommended strength in the file's metadata

**Prototype (the strength sweep only). Expected gain: better usability, not a better LoRA; unmeasured. Confidence: medium (0.45). Cost: about 20-30 probe sets, roughly 10-30 min on a side GPU for Krea 2 nvfp4. A no-turbo stack at 20+ steps costs about 3x per point.**

**What it is.** After the final save, sweep the subject strength from 0.6 to 1.3 under turbo and under turbo + sample_lora, fit a quadratic peak, and write `rulz_recommended_strength_turbo` / `_stacked` (plus the stack file names) into the safetensors metadata and the run summary. The score needs a penalty for quality or prompt-adherence drift, because likeness alone rises almost monotonically with strength and would always pick the top value.

**Status of the other members of the original pack**
- **Activation-whitened dynamic rank:** marginal; kept only as an option.
  - [kohya resize_lora.py](https://raw.githubusercontent.com/kohya-ss/sd-scripts/main/networks/resize_lora.py) offers sv_ratio, sv_cumulative and sv_fro, with `new_alpha = (alpha/old_rank)·new_rank` and metadata set to "Dynamic".
  - [SVD-LLM](https://arxiv.org/abs/2403.07378) whitening maps singular values to output loss, but it was shown for LLM weight compression and the abstract gives no numbers.
  - [KJNodes LoraReduceRank](https://comfyui-wiki.com/ko/custom-nodes/ComfyUI-KJNodes/nodes/lora-reduce-rank) already gives ComfyUI users kohya-style dynamic resize, plus sv_knee.
  - At rank 8-16 the file is tens of MB. The claim that "sv_fro 0.9-0.95 halves the rank" is unsourced.
- **Nunchaku/diffusers key variant:** refuted.
  - Krea 2 and Qwen 2.1 already export PEFT/diffusers keys with rank_pattern and alpha_pattern (lora.py:620-632; archs.py: peft for FIBO at 228, Krea 2 at 344, Qwen 2.1 at 458; comfy for MOSS near 145).
  - [Nunchaku](https://github.com/nunchaku-tech/nunchaku) supports none of the five target models (multi-LoRA since v0.2.0; native ComfyUI LoRA nodes in v1.2.0, 2026-01-12).
  - The Qwen LoRA PR #754 was closed unmerged.
- **Per-block strength tables:** dropped. With an 8-step probe noise floor of about 1 point, choosing 28-42 per-block values would overfit the probes.

**Evidence and transfer.** README ~1424 records Fizgig's finding that turbo LoRAs can understate likeness, so a strength picked under turbo may not carry over to full-step sampling. No source measures how much likeness a character LoRA loses to dynamic resizing.

**RulzTrainer today**
- Strength probing exists only for soups: `soup_strengths` 1.0/1.15/1.3 (config.py:424; trainer.py:3143-3200).
- The turbo and turbo+style stacks already exist through `turbo_lora`, `sample_lora` and `sample_lora_strength` (config.py:319-321).
- `finetune.py:386 lowrank_factors` and `extract_file` (finetune.py:421) do uniform-rank SVD.
- Nothing in the code matches "recommended strength", sv_fro, whiten or nunchaku.

**Fit and risks**
- **Placement:** post-training only, on the side GPU. Training is untouched.
- **Compile:** under compile, each strength value is a recompile (see item 2), so run the sweep eager.
- **Whitening statistics, if resize is built:**
  - Run the hooks uncompiled to avoid graph breaks.
  - The LoRA branch sees the unrotated bf16 activations, so W8A8 and NVFP4 do not distort the statistics.
  - Accumulate the r x r moment of A·x, because σ(ΔW·C^½) = σ(B·(A C Aᵀ)^½). A full covariance for an FFN input of about 12k would be roughly 600 MB per layer.
  - Draw the calibration sigmas from the training distribution.
- **Windows:** no NCCL or triton requirements.

**Implementation path**
- `rulztrainer/export_deploy.py` with `strength_sweep(trainer, lora_path, strengths, stacks)`, reusing `trainer._soup_probe` / `_probe_with` (trainer.py:3217 / 3264).
- New options: `deploy_sweep: off | on`, `deploy_sweep_strengths: "0.6..1.3"`, `deploy_sweep_full_steps` (adds the costly no-turbo stack).
- Score: the combined probe score minus a penalty for strength-driven drift (a prompt-adherence or base-similarity term on the probe renders).
- Optional: `resize(lora, target_fro, whiten=True)`, saved through `save_lora` with per-layer alpha (rank_pattern is already handled).
- Applies to all four families. For Flux 3 Klein and H3 the only export work is setting `Arch.lora_format`.

**How to test**
- **Strength sweep:** 2-3 finished character runs per model (Krea 2, Qwen 2.1, MOSS). Sweep 0.6-1.3 in 0.1 steps under turbo and under full-step sampling, with `probe_seeds` at 4 or more.
  - Check whether the turbo peak matches the full-step peak within 0.1.
  - If the likeness-only curve is monotone, the penalty term is needed.
  - Keep the feature only if the recommended strength beats 1.0 by at least 1 probe point on held-out seeds at full-step sampling.
- **Resize:** race the original LoRA against plain-SVD dynamic resize and whitened dynamic resize at matched parameter count, scoring ArcFace face and DINO body over 4 seeds. Keep whitening only if it retains at least 0.5 points more than plain SVD. If both stay within noise of the original at rank 8-16, resize is pointless for this workload.

### 4. Selftrain sigma floor

**Race it. Expected gain: between zero and a small positive, and only for runs with selftrain on; unmeasured. Confidence: low (0.3). Cost: a few lines of code; free at run time.**

**What it is.** ReGain shows that personalising on model-generated images lowers subject fidelity, because classifier-free guidance becomes inflated: the angle between the conditional and unconditional predictions widens. The proposal remaps each selftrain item's sigma to `selftrain_sigma_min + (1 − selftrain_sigma_min)·sigma`, so these items train only at higher noise and their invented fine detail is less likely to be learned. The sigma split is the scout's own inference and the paper does not test it. The paper's own band ratios also weaken the argument: the inflation is broad-band (low 4.70, mid 4.95, high 5.76), not confined to high frequencies.

**Evidence**
- [ReGain](https://arxiv.org/html/2609.38680v1) (UIUC, 30 Sep 2026). Fidelity, real-trained vs synthetic-trained:

  | Setting | Real-trained | Synthetic-trained |
  |---|---|---|
  | SD1.5 DreamBooth-LoRA, DINO | 0.630 | 0.540 |
  | SD3.5 Medium LoRA, DINOv2 | 0.650 | 0.567 |
  | SD3.5 Medium LoRA, DINO | 0.682 | 0.586 |
  | SDXL LoRA, DINO | 0.558 | 0.477 |

  - The guidance norm is larger in 29 of 30 subjects, by 48% ± 22%. The growth comes from the angle; the norms of the two predictions themselves do not grow.
  - The sampling-time fix recovers 51-64% of the gap (SD1.5 DreamBooth), about 60% (SD1.5 LoRA), 31-40% (SDXL LoRA) and 20-46% (SD3.5 LoRA). Sampling costs about 1.1x (5.4 s vs 4.9 s), and calibration takes about 12 min per subject on an A4000.
- What does not transfer:
  - The synthetic sets were pure synthetic, 5 images each, rendered at CFG 7-7.5 by an already personalized model.
  - Mixed real+synthetic sets, distilled models, 5-12B single-stream DiTs and ArcFace metrics were all untested.

**RulzTrainer today**
- Selftrain items carry `category "selftrain"` (trainer.py:2543) and `loss_scale = selftrain_weight` (trainer.py:3110), which the loss applies at trainer.py:1007-1009.
- The share cap is `selftrain.py:146 cap_count`; the options are at config.py:413-420.
- Sigma is drawn per batch at trainer.py:1078 (also race_parallel.py:218, and trainer.py:2454 for DOP) and snapped to the 1000-step grid (trainer.py:215-245). No per-source sigma range exists.
- Selftrain renders come from `_selftrain_render` (trainer.py:3033), often a turbo model at CFG 1, where the inflation mechanism may differ (unmeasured).
- Discrepancy: the README says selftrain is "Not in a race", but race.py:75 lists `selftrain_weight` as raceable.

**Fit and risks**
- **Compatibility:** at batch 1 this is a per-item remap that stays on the 1000-step grid. It does not recompile (sigma is a tensor input) and leaves the RNG stream unchanged. It is neutral to W8A8, NVFP4, keep blocks, block swap, model_split and data_parallel. Larger batches would need a per-row resample.
- **Risk:** the floor may throw away the detail selftrain is meant to add.
- **"Restored" source:** that half depends on item 8, which does not exist yet.
- **Skipped:** the ReGain sampling correction. Probes run turbo at CFG 1, where there is no conditional/unconditional pair to rescale.

**Implementation path**
- Add `selftrain_sigma_min` to config.py next to the selftrain options (default 0.0 = off, validated to 0-0.95).
- Apply it after `sample_sigmas` at trainer.py:1078 and race_parallel.py:218 to items with category "selftrain", then re-snap to the grid.
- Add it to race.py's raceable keys next to `selftrain_weight` (race.py:75).
- Optional diagnostic:
  - At the end of each selftrain round, compare the mean conditional/unconditional difference norm and angle (FFT-banded) for a few real items against the kept renders, with the LoRA on vs off via `set_multiplier` (as region_prior_weight does at trainer.py:986-989).
  - Run it only when `sample_cfg > 1` and the sampler is not turbo. It costs about 4 forwards per item per sigma, about 1-3 min per selftrain round.
  - Log to `selftrain.json` and warn above +20%.
- README selftrain section and wiki.

**How to test**
- **Arms:** Qwen 2.1 int8 and Krea 2 nvfp4, relay race from the same seed, selftrain on with `selftrain_at: 0.5,0.75`. Compare `selftrain_sigma_min` 0 vs 0.4 vs 0.6, plus a no-selftrain control.
- **Metric:** the probes' smoothed combined likeness (ArcFace face plus DINOv2 body) on the base sampler at full CFG, with the turbo probe as a secondary check. Log the guidance-norm ratio between real and selftrain items.
- **Stop rule:** the race's standard patience. Count it as a gain only if the 0.4 or 0.6 arm beats the 0 arm by more than the probe noise (about 1-2 points across 2 seeds) at both the final and the best checkpoint. If no selftrain arm beats the control, the question is moot.

### 5. Coverage report: per-bin probes and dataset bin counts

**Prototype (counts plus per-bin probes; no gradient attribution). Expected gain: nothing for the current LoRA. It gives advice for the next dataset; unmeasured. Confidence: low (0.35). Cost: about 9-12 extra probe renders x probe_seeds, once at the best checkpoint, a few minutes per run.**

**What it is.** Bin the dataset and a set of extra probe prompts by attribute:

- yaw: frontal, 3/4, profile;
- expression: neutral, smile, open mouth;
- lighting: soft, hard side, low light;
- framing: face, bust, full body.

A bin with low probe likeness and few dataset images becomes a "shoot more of this" hint. The original proposal also annotated each bin with attribution scores. Drop that part: no attribution code exists, and the papers' version costs tens of GPU-hours. Annotate with loss_watch verdicts and look-filter outliers instead.

**Evidence**
- [Concept-TRAK](https://arxiv.org/html/2507.06547v2): introduces a contrastive base / positive / negative prompt utility (for example "woman" / "old woman" / "young woman"). The cited v2 page has no CelebA-HQ results; only [v3](https://arxiv.org/abs/2507.06547v3) adds the CelebA-HQ and Synthetic benchmarks. The full method cost 52.4 H100 GPU-hours on SD1.4.
- [Learning to Weight Parameters for TDA](https://arxiv.org/html/2506.05647): learns separate subject, style and background weights. Subject Recall@10 rises from 19.4% to 50.6%, but only on the validation split of a synthetic SB-Pokemon set (D-TRAK, SD with LoRA), not on faces.
- Neither paper produces coverage advice. ArcFace verification drops on profiles (the CFP-FP and CPLFW benchmarks show the frontal-vs-profile gap), and turbo samplers may not follow pose or expression prompts.

**RulzTrainer today**
- Probes use one face prompt and one body prompt (config.py:428-429; `PRESET_PROBE_PROMPTS` at config.py:107-111; README.md:858).
- The dataset is split only into face-heavy vs body/other by face area (likeness.py:357-389, logged at trainer.py:2573-2577).
- `reference_ceiling` (likeness.py:72-89) returns None with fewer than 2 refs (likeness.py:81), so most hard bins will be uncalibrated.
- InsightFace loads only detection and recognition (facesim.py:118-119), so it gives no pose. Yaw has to come from the 5-point kps already cached in regions.py:94-123.
- The only MediaPipe model present is Pose (bodysim.py:23-25); there is no FaceLandmarker for blendshapes.

**Fit and risks**
- Off the training hot path, so W8A8, NVFP4, compile, keep blocks, block swap, model_split and data_parallel are unaffected. Probes run in the side-job slot, and VRAM is unchanged.
- Per-bin likeness will be noisy at 1-2 seeds and biased by ArcFace's pose sensitivity, so keep the report advisory and never auto-exclude.
- Lighting estimated from 2D is crude.

**Implementation path**
1. New `coverage.py` (or code in likeness.py):
   - yaw bins from the cached kps (nose offset from the eye midpoint, divided by eye distance);
   - framing bins from `face_area_fractions`;
   - lighting bins from the face-crop luminance mean and left/right gradient;
   - optional expression bins behind a lazily downloaded MediaPipe FaceLandmarker (smile and jawOpen blendshapes).
   - Cache the result in `.rulz_cache/coverage.json`.
2. Add the option `probe_bins: off | final`. With `final`, render the bin prompt suffixes at the best checkpoint on the existing probe path.
3. Measure each render's actual yaw and expression again before assigning it to a bin.
4. Normalize by a per-bin `reference_ceiling` when the bin has 2 or more face_refs. Otherwise use the global ceiling and mark the bin "uncalibrated".
5. Write `coverage.json` to the job folder and show a WebUI table: bin, dataset count, probe likeness, hint, plus the loss_watch and look-filter flags that fall in that bin.

**How to test.** This is a validity test, not a race.
- Build ablated copies of a balanced 40-image set: no profiles, no smiles, no full-body.
- Train each with the same seed (Qwen 2.1 int8, 1500 steps).
- Pass if the report ranks the removed bin as the top gap in 3 of 3 ablations, and the balanced run flags no bin far below the others.
- Also run the balanced set at 3 probe seeds. If the seed-to-seed spread per bin is larger than the drop the ablation causes, fall back to count-only hints.

### 6. Per-image likeness attribution (in-run TracIn and post-hoc DataInf)

**Watch. If built at all, build the in-run version as a diagnostic column, and only after the stability gate below. Expected gain: diagnostic. The verifier's guess for the effect of acting on the scores is 0 to +0.01 ArcFace on sets with a few subtly bad photos, and noise on clean sets. Confidence: low (0.35 in-run, 0.3 DataInf).**

**Cost:**
- **In-run:** about 1-2% of step time at probe_every 200 with 2-4 refs (about 20-35 s per probe event on Krea nvfp4), plus 100-300 MB of VRAM.
- **DataInf** (40 images x 4 sigmas = 160 batch-1 passes): about 24 min on Krea nvfp4, 13 min on MOSS int8, 7 min on Qwen 2.1 at 0.5 MP.

**What it is.** Both methods score each training image by its first-order effect on a likeness utility, not on the loss.

- **In-run version (TracIn / In-Run Data Shapley style).** At each probe event, compute a utility gradient g_u over the LoRA parameters: the id_loss ArcFace (or FaceNet) cosine on a one-step predicted x0 of a reference image, at sigma 0.2-0.4. Then, each step, add lr_t · ⟨grad / (sqrt(v)+eps), g_u⟩ to that image's score.
- **Post-hoc version (DataInf).** After the run, collect per-image LoRA gradients at fixed sigmas and noise seeds, then apply DataInf's per-layer Sherman-Morrison closed form with the utility gradient as the validation vector.

**Evidence**
- [Data Shapley in One Training Run](https://arxiv.org/abs/2406.11011) (ICLR 2025 Outstanding Paper runner-up): "negligible additional runtime" applies only to its fastest variant, and its case studies are pretraining.
- [In-Run Data Shapley for Adam](https://arxiv.org/abs/2602.00329): the abstract's "about 95% throughput" and "R about 0.11" do not match the paper's own v4 body, which reports:
  - Adam-Ghost keeps 43.9% of AdamW throughput (112 vs 256 samples/s; GPT-2 Small, batch 16, one A100);
  - R = 0.0579 between SGD and Adam values;
  - R = 0.8434 vs 0.9992 against exact local Shapley, which is a local metric, not retraining.
  - The settings are LLMs and text classifiers only.
- [Concept-TRAK](https://arxiv.org/html/2507.06547v2) (ICLR 2026): the jump from 0.04 to 0.955 Recall@10 comes from four stacked changes. The reward-on-predicted-x0 utility alone gives 0.261. Setting: SD1.4 textual inversion on H100s.
- [DataInf](https://arxiv.org/html/2310.00902v3) (ICLR 2024):
  - per-layer closed form with damping 0.1 x mean squared gradient norm / d_l;
  - SD1.5 LoRA tests at r=1 (styles) and r=4 (31 DreamBooth subjects, 3 images each) on 4x V100;
  - timing for RoBERTa on one A40: 13 s vs 70 s (LiSSA) vs 11,279 s (exact);
  - error is O(d_l²). A rank-16 LoRA on 3072-wide layers has d_l around 1e5, versus about 1e3-1e4 for SD1.5 r=4.
- [Learning to Weight Parameters](https://arxiv.org/html/2506.05647): shown on D-TRAK, not combined with DataInf, and for between-subject retrieval.
- Contrary evidence:
  - [Do Influence Functions Work on LLMs?](https://arxiv.org/abs/2409.19998) finds DataInf, LiSSA and Hessian-free influence "consistently perform poorly" on Llama2-7B and Mistral-7B.
  - [Mlodozeniec et al.](https://arxiv.org/pdf/2410.13850) show that diffusion attribution depends heavily on the choice of measurement function.
- No paper shows that attribution-guided curation raises identity metrics for a person LoRA. Every result is between-subject retrieval; ranking photos of the same person is untested.

**RulzTrainer today**
- `losswatch.py:63-249` gives loss-residual verdicts, and `per_image_lr` throttles stuck images (README.md:651-662).
- `lookfilter.py` already catches wrong-person and off-look photos by ArcFace/DINO.
- `idloss.py:235-257` is the differentiable path (predicted_x0, decode, align, embed, cosine).
- No attribution, influence or TRAK code exists.
- `batch_size` 1 and `gradient_accumulation` 1 are the defaults (config.py:100, 272). The gradient between `clip_grad_norm_` and `optimizer.step()` (trainer.py:1133-1138) is therefore exactly one image's gradient, and no ghost trick is needed.
- automagic3 keeps factored second moments (automagic3.py:403-410), so a sqrt(v) preconditioner is available. The scout's per-parameter-lr workaround is unnecessary.

**Fit and risks**
- **Engineering fit:**
  - g_u covers the fp32 LoRA parameters only, through the same quantized forward id_loss already backpropagates through, so W8A8 and NVFP4 are fine.
  - The dot product runs outside the compiled blocks, so compile is unaffected.
  - Block swap and model_split are fine (idloss.py:247).
- **data_parallel:** the dot product has to happen per replica before `dp.reduce_grads()` (trainer.py:1129).
- **Races:** each racer needs its own g_u.
- **DataInf pass:** force bf16-exact backward (int8_backward=bf16, NVFP4 stochastic-rounded backward off).
- **Statistical risks:**
  - face_refs are dataset images (config.py:328), so they show large self-influence; exclude them or use a held-out set.
  - g_u refreshes only every 100-250 steps, which makes this TracIn-CP with stale checkpoints.
  - Each image appears only 20-150 times per run, so the per-image sums are noisy.
  - Goodhart: when id_loss and the utility both use ArcFace, the scores favour images that game ArcFace. Use FaceNet (`id_loss_backend=facenet`) for the utility.

**Implementation path**
- `rulztrainer/attribution.py`:
  - `Attribution(cfg, params, optimizer, id_loss)` builds g_u at probe events from 2-4 refs at sigmas in [0.2, 0.4], reusing `IdentityLoss.__call__` (idloss.py:235).
  - `observe(item_key, lr_t)` runs between `clip_grad_norm_` and `optimizer.step()`, reading v from AdamW `exp_avg_sq` or from automagic3's row/col factors.
- Options: `likeness_attribution: bool` (requires batch 1, grad accumulation 1, face_refs and probe_every > 0), plus `attribution_refs` and `attribution_sigmas`.
- Write `attribution.json` beside `per_image_loss.json` and show a signed column on the Dataset tab. Do not act on the scores automatically in v1.
- DataInf, if pursued, is a post-run "Analyse dataset" button on the side device: `datainf_scores(...)`, with gradients in pinned CPU fp16 (2-6 GB of host RAM for 40-60 images) and options `attribution_utility: renders | id_loss`. Skip learned layer weights in v1.

**How to test**
1. **Stability gate:** run the same Qwen 2.1 int8 config with 2 seeds and compute the Spearman rho of the per-image scores. If rho < 0.4 (in-run) or about 0.5 (DataInf, across noise-seed sets), stop.
2. **Causal check:** race full set vs set minus the bottom 10-15% (refs excluded) vs set minus the same count at random vs set minus the look-filter + loss-watch picks. Use 3 seeds and 1500-2000 steps. Score best-checkpoint ArcFace on held-out reference photos plus DINO body.
3. **Keep rule:** keep the feature only if removing the bottom-k beats random removal by more than the seed std (about 0.01-0.02 ArcFace) on 2 of 3 seeds, on two model families, and also beats the look-filter removal. If more than 70% of the bottom-k overlaps the look-filter outliers, the feature adds nothing.

### 7. Leave-one-cluster-out by bounded unlearning

**Watch. Expected gain: diagnostic only; any gain needs a retrain without the flagged cluster; unmeasured. Confidence: low (0.3). Cost: about 3-5 min per cluster on Qwen 2.1, 6-10 min on MOSS and 15-20 min on Krea (5060 Ti), so 1.5-2.5 h of GPU time for 6-8 Krea clusters before any retrain.**

**What it is.** Cluster the dataset by pose bins, DINO k-means or folder/session. Starting from the best LoRA, run about 80 steps of descent on the remaining images while ascending, with a bounded term, on one cluster. Then probe. The per-cluster change in probe likeness approximates leaving that cluster out.

**Evidence**
- [GUDA](https://arxiv.org/abs/2601.22651) (ICML 2026): approximates leave-one-group-out retraining. The 100x speedup holds only on CIFAR-10. On SD1.5 UnlearnCanvas styles it took 8h54m vs 46h08m (about 5-6x), with fine-tuning from a shared checkpoint as the oracle. Its objective is a redirecting forget loss plus a preservation loss, not gradient ascent with a null-loss clamp, and it evaluates styles, not identities.
- [MUCS](https://arxiv.org/html/2605.17938) ([NeurIPS 2026 poster](https://neurips.cc/virtual/2026/poster/153796)):
  - bounded ascent with λ 0.2, clamped at the null loss, stopping at 95% of null (about 109 steps);
  - consistent noise is the largest ablation (about −19% AUC);
  - CIFAR-10 AUC 0.897 vs DAS 0.786 vs D-TRAK 0.702, and 1.5-2.9x faster.
  - Two corrections: MUCS unlearns a generated item, not a training cluster; and its null loss comes from a randomly initialized model, not the base at LoRA scale 0.
  - Setting: a 12-block pixel-space DiT-B trained from scratch at 64x64. No LoRA.

**RulzTrainer today**
- The race compares configurations and never ablates data (race.py:1-35). `variant_items` (race.py:733) is a precedent for per-variant item subsets.
- `race_parallel` runs both GPUs in lockstep, without compile or id_loss (race_parallel.py:13-14), so it cannot host a background side job.
- Relay does not run with race_parallel (README.md:1196).
- Acting on a result requires either the losswatch multipliers (losswatch.py:210-212, only with `per_image_lr`) or exclusion.

**Fit and risks**
- **Signal:** flow-matching loss on a LoRA fine-tune is dominated by irreducible noise. The gap between LoRA loss and null loss is a few percent at most, so a "95% of null" stop would fire immediately or flicker. It needs fixed (sigma, noise) pairs on the 1000-step grid, averaged over many pairs.
- **Confound:** every cluster is the same person. Ascent on one cluster also damages the shared identity.
- **Probe noise:** "a few points" (README.md:1197), so small clusters need `probe_seeds` of 4 or more.
- **Memory:** a second Krea copy on the 4060 Ti must be int8 (no FP4 there), which is tight.
- **Compatibility:** quantization and exports are unaffected. `model_split` must be off.
- **Free baseline:** the look consistency filter already catches the obvious off-identity clusters without any training.

**Implementation path (if pursued).** A post-run, single-GPU tool, not a race mode:
- `rulztrainer/ablation.py` saves and restores the LoRA and optimizer state with race.swap_out-style helpers.
- Per cluster: loss = mean(rest) − λ·min(mean(cluster), null), with null precomputed at LoRA scale 0 on fixed pairs; then probe with the baseline's seeds.
- Options: `ablation_clusters: pose | look | dino | folder`, `ablation_k`, `ablation_steps` (80), `ablation_lambda` (0.2).
- Output: `<job>_ablation.json` and a WebUI table with delta_face and delta_body against a two-seed-set noise floor.

**How to test**
1. Inject a poison cluster (4 photos of a look-alike) and check that it ranks first by delta_face, above the noise floor.
2. Compare against real leave-one-cluster-out continuations: train the best LoRA a further probe_every x 4 steps on the rest vs on all images, for 2-3 clusters. Stop if the Spearman rho with the unlearning deltas is below about 0.5.
3. Race the ablation-guided dataset against the original and against the look-filter fence. Keep the tool only if it beats both by more than the probe noise on two subjects and two models.

### 8. Identity-gated restored twins

**Skip-ish (at most, the triage-and-gate import path). Expected gain: unmeasured; small and possibly negative for typical 15-60-photo sets. Confidence: low (0.35). Cost: triage and gate take well under 1 s per image on CPU. An in-process Qwen 2.1 int8 restorer would take an estimated 15-60 s per image, unmeasured, after porting the edit path.**

**What it is.** Triage weak photos (JPEG quality from the quantization tables, ArcFace or AdaFace embedding norm, face box size). The user restores them with an external tool. Accept a restored copy only if two judges agree: ArcFace cos(restored, original) ≥ 0.75, the cosine to the face_refs centroid does not drop by more than 0.01, and the FaceNet cosine passes its own threshold. Faces under 112 px are rejected. Accepted copies are added as lower-weight twins at loss_scale 0.5, with the original kept.

**Evidence**
- Restorer LoRAs:
  - [Qwen-Image-2.1 Restore LoRA](https://huggingface.co/ausboss/Qwen-Image-2.1-Restore-LoRA): face match 0.821 to 0.905 and SSIM 0.578 to 0.766 at 8 steps (on 8 test pictures). Other sets: 29 pictures 0.79 to 0.88; 57 held-out pairs SSIM 0.544 to 0.762. One RTX 5090; SeedVR2 took about 3 s at 2 MP. All of this is against clean sources of synthetically damaged pairs. Faces of about 100 px may come back as a different person. Qwen Research License.
  - [FLUX 3 Klein Restore LoRA](https://huggingface.co/ausboss/FLUX-3-Image-Klein-Restore-LoRA): ArcFace 0.773 vs 0.753 on 8 faces, one seed. Shadows came back as stubble.
- Reference-guided restoration:
  - [Reference-guided restoration (arXiv 2505.21905)](https://arxiv.org/html/2505.21905v1): IDS on severe FFHQ-Ref is CodeFormer 0.370, DiffBIR 0.356, RestorerID 0.411, reference-guided 0.609.
  - [FaceMe](https://arxiv.org/html/2501.05177): IDS 0.647 vs CodeFormer 0.546; no timing and no code.
- Other sources:
  - [SeedVR2 docs](https://docs.comfy.org/tutorials/utility/seedvr2.md): one-step "conservative upscaling", INT8 convrot files, Apache 2.0. No VRAM, timing or identity figures. INT8 can produce all-black output with FlashAttention.
  - [AdaFace](https://arxiv.org/pdf/2204.00964): the 0.5235 correlation between feature norm and quality appears in the patent text and is unconfirmed in the paper.
- Contrary evidence in this repo: README.md:762-764 records auto crops upscaled from 300 px scoring 65.5 against a 74.2 baseline. Upscaled or invented detail hurt likeness on this trainer.

**RulzTrainer today**
- Look filter: lookfilter.py, README.md:1466-1500.
- `FaceScorer` (facesim.py:105).
- A FaceNet embedder ready to be the second judge (idloss.py:138, config.py:390).
- Selftrain twins with loss_scale and provenance (trainer.py:3082-3110), per-image sampling weights (dataset.py:398-407), and a Laplacian blur filter in selftrain.
- No restorer exists. The Qwen 2.1 edit path is not ported (qwen21/model.py:25-26), and SeedVR2 and Klein are unsupported.

**Fit and risks**
- **Placement:** runs before latent caching, so training compute paths and exports are untouched.
- **Detail:** the restorers' 2 MP output is mostly downsampled away at 0.5-1 MP training size.
- **Gate weakness:** ArcFace cos(restored, original) is unreliable on exactly the degraded faces that need restoring, so the centroid test must be mandatory.
- **Licences:** the Qwen and FLUX licences restrict commercial use. An in-process restorer would default to SeedVR2.
- **Races:** the twin set has to be fixed before a race so racers stay comparable.

**Implementation path (only if a race justifies it)**
- `rulztrainer/restore.py` with triage written to `.rulz_cache/restore_triage.json` and a two-judge gate.
- Twins go through the selftrain twin path, with options `restore_twins`, `restore_min_face_px` (112), `restore_gate_cos` (0.75) and `restore_weight` (0.5), plus a dataset-tab panel showing before/after scores.
- Restored copies go in `<dataset>/restored/` and are made with an external tool.

**How to test.** Race on AZav50 or a similar set: originals only vs originals + gated twins at 0.5 vs weak photos moved out by the look filter. Score probe face EMA, body score and a FaceNet check, and inspect by eye for invented freckles, stubble and pores. Adopt twins only if they beat both other arms by more than about 1 point on 2 seeds with no visible texture artifacts.

### 9. Accuracy-recovery adapter for the nvfp4 training base

**Watch. Run the no-code gate first. Expected gain: about zero on int8 bases. For Krea 2 nvfp4 deployed on bf16/GGUF there might be a small gain, and only with a distilled or SVDQuant-style adapter; unmeasured. Confidence: low (0.25). Cost: a rank-32 side branch adds about 1-3% step time and about 0.1-0.3 GB on a 12B model; a distilled build takes minutes to tens of minutes per model.**

**What it is.** A frozen low-rank adapter that cancels the training base's quantization error, kept on during training and off at sampling. With it on, the subject LoRA cannot learn that error.

**Evidence**
- [ostris accuracy_recovery_adapters](https://huggingface.co/ostris/accuracy_recovery_adapters): per-layer distilled bf16 side LoRAs. 3-bit Qwen-Image with rank 16 is called the "sweet spot" and enables 1 MP training on 24 GB. The card gives no quality metrics, and plain 3-bit without the adapter is reportedly broken. This targets a regime (torchao uint3) that RulzTrainer does not use.
- Verifier's measurement on Krea-2-Raw block 10 with `quant.quantize_weight`:
  - Relative weight error: int8_convrot 0.89-0.94%, nvfp4 9.4-9.5%.
  - The residual's spectrum is nearly white: rank 16/32/128 capture 1-3%, 1.6-5.5% and 6-19% of its energy. The closed-form residual-SVD adapter cuts nvfp4 error only from about 9.45% to 9.35%.
  - An SVDQuant-style split (keep rank r of W in bf16, requantize the rest) does better: 9.4% to 8.4-9.2% at r32, and to 7.2-8.8% at r128.
  - Activation whitening was not tested.
- Citation corrections:
  - [SVDQuant](https://arxiv.org/abs/2411.05007) takes the low-rank part of W before quantizing. It does not take an SVD of the quantization residual.
  - The [CoA-LoRA](https://arxiv.org/html/2509.25214v4) numbers (SST-2 −5.06%, +1.74 to 8.89% over QLoRA) are RoBERTa-Large GLUE averaged over bit-widths, not LLaMA-2-13B, and the results are mixed.
  - QLoRA practice and LoftQ suggest little harm at 4-8 bits. No report exists of an image LoRA that "learned the quantization error".

**RulzTrainer today**
- `attach_training_adapter` (lora.py:902-936) loads frozen `train_down`/`train_up` buffers and concatenates ranks with a second adapter (lora.py:931), so an adapter stacks with Fizgig's for free. It is off in samples and probes (lora.py:394-398).
- A bf16 teacher exists only when the base is quantized on load (`quantize_module_linears`, quant.py:1035; Krea at krea2/loading.py:130). It does not exist for pre-quantized ComfyUI files (moss/single_file.py:45) or for int8-to-nvfp4 conversion.

**Fit and risks**
- **Compatibility:** plain matmuls beside QuantLinear, so it works with W8A8, NVFP4, block swap, model_split and compile. Nothing is exported.
- **Activation error:** in NVFP4 W4A4, much of the error may come from activation quantization, which a weight-residual adapter cannot see.
- **`base_precision_mix`:** refuted for this hardware. Krea 2 int8 needs about 14.2 GB with VAE and does not fit next to it, and mixed formats break dp_balance.

**Implementation path (only if the gate shows a gap)**
- `quant.build_recovery_adapter(model, fmt, rank, mode)`:
  - mode `svdquant`: quantize W − L and put L in the training-adapter buffers;
  - mode `distill`: per-layer output matching against bf16 on hooked activations from 8 cached samples.
- Options: `quant_recovery_adapter: off | svdquant | distill`, `quant_recovery_rank: 32`. Only for nvfp4 bases quantized on load from bf16.

**How to test**
- **Step 0 (no code):** train the same character config on bf16 vs nvfp4 for 1500 steps (MOSS at 0.5 MP fits bf16; Krea 2 on a rented 24-48 GB card). Probe both LoRAs on a bf16 base and on the nvfp4 base, over 2 seeds. If the gap on the bf16 base is within the seed spread, stop.
- **Step 1, only if there is a gap:** race nvfp4 vs nvfp4 + adapter (svdquant r32, then distill r32) on the bf16 probe. Adopt if the adapter recovers at least half of the gap at no more than 3% step-time cost over 2 seeds.

### 10. Null-space compose guard against turbo and style LoRAs

**Skip-ish (a probe-time check at most). Expected gain: near zero to negative for subject + turbo; unmeasured on any DiT. Confidence: low (0.3). Cost: one extra probe render set for the probe-time check; under 1% of step time if ever trained.**

**What it is.** NP-LoRA projects a content LoRA's input side away from a style LoRA's top right singular vectors V_k, with the soft form ΔW_c(I − μ/(1+μ)·V_kV_kᵀ). The proposal applies the same projection during training, against the turbo LoRA and a PCA bank of style LoRAs.

**Evidence**
- [NP-LoRA](https://arxiv.org/html/2511.11051): μ 0.5, k equal to the LoRA rank. On SDXL rank 8 (32 pairs), S_arith/S_harm are 0.55/0.50 vs direct merge 0.51/0.42 and DuoLoRA 0.53/0.46. RTX 3090; merging takes 13.44 s vs over 10 min for ZipLoRA. The FLUX appendix compares only against K-LoRA (0.56/0.54 vs 0.55/0.50). The method is training-free and measures content-style balance, not identity.
- Evidence against:
  - [Orthogonal MC Dropout](https://arxiv.org/html/2510.03262): direct merge was preferred 81.3% of the time and was best on all 5 IQA metrics (MUSIQ 62.81 vs 59.69).
  - [Universal Weight Subspace](https://arxiv.org/html/2512.05117v2): subject LoRAs share the common directions too (CLIP 19.83 vs 19.73, a noise-level gap), so projecting them out likely removes useful directions.
- Other methods:
  - [QR-LoRA](https://arxiv.org/html/2507.04599v2) composes only with other QR-LoRAs (SDXL/SD3/FLUX.1-dev, r=64, about 20 min on an A800).
  - [SSR-Merge](https://arxiv.org/abs/2606.10617) and SDO (arXiv 2608.13820) are merge-time alternatives; their abstracts give no numbers.
  - Prior training-time work exists: Orthogonal Adaptation (arXiv 2312.02432).
- A turbo/DMD LoRA reshapes the whole velocity field and is not a "style" LoRA.

**RulzTrainer today**
- Nothing comparable exists; OFT (lora.py:27) is the only orthogonality feature.
- Each wrapper already holds the turbo LoRA's (down, up, scale) as frozen extras (lora.py:243, `attach_extra_lora` lora.py:828-900, `_attach_sampling_loras` trainer.py:2146).
- Probes already render with the turbo LoRA stacked, so best-checkpoint selection already accounts for most stacking interference.
- README.md:1425 notes that MOSS's DMD LoRA "can understate likeness". That is the one symptom this item might touch, but it may equally be a sampler or step-count effect.

**Fit and risks**
- **Compatibility:** fp32 lora_down, outside the quantized paths and outside the graph. Only applies to network_type lora.
- **Flaw:** the per-step soft projection (mechanism b) converges to a hard projection over 1000-3000 steps. Use μ/T per step, or apply it once.
- **Likeness risk:** projecting out the turbo LoRA's top-k input directions (k = 64-178) removes the highest-energy activation directions and likely costs likeness.

**Implementation path (probe-time only)**
- `rulztrainer/neighbours.py: np_project(injected, mu, k, source="turbo")` builds V per module from the extras (QR of downᵀ, then SVD of up·R).
- Option: `compose_guard: off | probe | export`. Mode `probe` scores the raw and projected LoRA on the sample copy. Mode `export` writes `<name>_np.safetensors` for a user-chosen style LoRA.

**How to test.** On saved LoRAs of 3 characters (Qwen 2.1 int8, Krea 2 nvfp4), probe with and without the projection at μ ∈ {0.25, 0.5, 1}, on turbo plus a 20-28-step full-quality control. Drop the idea if no μ beats seed noise on 2 of 3 characters, or if full-quality likeness drops. Only then consider a training penalty λ ∈ {1e-3, 1e-2} in a relay race.

## Bigger hardware on vast.ai

The verifiers judged every idea against the local pair: RTX 5060 Ti 16 GB (sm_120) and RTX 4060 Ti 16 GB (sm_89), on Windows, at batch 1. This section goes back over the items that were dropped or downgraded only because of 16 GB, GeForce tensor-core ratios, batch 1, Windows (no NCCL, triton-windows, no flash build) or the mismatched pair. It asks what changes on rented RTX 5090 32 GB, RTX PRO 6000 Blackwell 96 GB, A100 80 GB, H100/H200 80-141 GB, B200 and multi-GPU NVLink nodes.

Overall verdict: big cards change what RulzTrainer should *spend memory on*, not which methods it needs. Most 16 GB memory tricks become irrelevant at 48 GB and up. Four things become worth doing:
- Recompute goes away. Fizgig measured 13-27% per step with checkpointing off, and RulzTrainer's auto keep already gets there.
- The frozen base can be bf16 or int8 instead of nvfp4. This makes possible the precision race that nobody has run yet.
- The attention backend should be picked per architecture.
- Multi-GPU boxes should run racers as separate processes, so each racer can compile and none waits on the GIL.

Bigger batches do not pay. The only measured data point, KohakuMoss on a 5090, shows larger micro-batches losing throughput once nothing is streamed. FP8 compute adds no forward speed over the existing int8 W8A8 on any of these cards. At most it is a backward-pass option on Hopper and B200.

### What changes on big cards

| Idea | 16 GB local verdict | 32 GB RTX 5090 (sm_120) | 96 GB RTX PRO 6000 (sm_120) | 80 GB Hopper (H100/H200; A100 sm_80 noted) | B200 (sm_100) | Multi-GPU NVLink/NCCL |
|---|---|---|---|---|---|---|
| Precision race: bf16 vs int8 vs nvfp4 base (X2-1, S5 gap race) | Race needs model_split int8; bf16 Krea does not fit | Krea int8 fits (auto picks int8 from 22 GB) | bf16 Krea fits: the clean reference run | bf16 fits; no FP4 cores, so nvfp4 dequantizes | all three formats native | one format per racer process |
| No recompute (checkpointing off / all blocks kept) | Krea keeps 1 of 28 blocks, MOSS 6 of 42 | partial keep; SAC worth 3-8% (S5) | all blocks kept at 1 MP; most at 2 MP | all kept at 1 MP | all kept | per GPU |
| Selective checkpointing tier (G11) | 0-1.5% | 3-8% (estimate) | ~0 (everything fits) | ~0 | ~0 | - |
| Activation offload (G12), block swap tweaks (G18), fp8 kept activations (G9), MsT (G17), CARE-LoRA (G14) | marginal or dropped | marginal | unnecessary | unnecessary | unnecessary | - |
| FP8 compute path (G4) | dropped: FP8/fp32-acc is half the INT8 rate on GeForce | same GeForce ratio; row-wise `_scaled_mm` unsupported on sm_120 | same as 5090 (ratio not verified for the PRO part) | FP8 = INT8 peak (1979 dense), so forward gains 0 over W8A8; FP8 dgrad is the only use | FP8 = INT8 (4500 dense); MXFP8 dgrad measured 1.27-1.60x per GEMM on B300 | - |
| NVGEMM block-scaled GEMMs (G82) | dropped | Watch | Watch | n/a (no FP4) | Watch: sm_100 is CUTLASS's main target | - |
| nvfp4 W4A4 native | the 5060 Ti's only Krea base | 2x the INT8 peak; costs 13.4% forward error | as 5090 | not available (dequantize fallback) | native (code gate >= 10; untested on sm_100) | - |
| Attention backend (G22, G2) | cuDNN on Blackwell; cuDNN on Ada recommended (S1) | cuDNN (auto) | cuDNN (auto) | auto gives flash; cuDNN or FA3 worth a bench | cuDNN (auto); FA4 via FlexAttention optional | - |
| INT8 training attention (SageBwd, G2) | Prototype | Prototype (same 4x INT8:bf16 ratio) | as 5090 | ceiling halves (INT8 is 2x bf16, not 4x) | ceiling halves | - |
| Full fine-tune windows (G16) | 10 windows (MOSS 0.25 MP) | fewer windows | MOSS in 1 window; Krea in about 2 | MOSS in 1; Krea 2-3 | Krea in 1 window (180 GB) | not supported (no FSDP) |
| Per-block format ramp / base_precision_mix (G51, G15) | Krea int8 does not fit 16 GB | moot: int8 fits | moot | moot | moot | - |
| Recovery adapter for nvfp4 (X2-1) | Watch | only if nvfp4 is chosen for speed | as 5090 | n/a | only if nvfp4 is chosen | - |
| Batch > 1, length bucketing (G21), bigger micro-batch (G12/123) | batch 1 | throughput flat or worse (KohakuMoss) | unmeasured | unmeasured | unmeasured | data_parallel multiplies images per step |
| 1-2 MP training | 0.5-1 MP | 1 MP | 1-2 MP | 1-2 MP | 1-2 MP | - |
| Compile cache (G13, G80) | 160 s cold vs 36 s warm (Windows) | Do it (S6) | Do it | Do it | Do it | per process |
| Allocator settings (G20) | Windows: nothing to set | Linux already sets expandable_segments | same | same | same | same |
| Micro-batch pipeline for model_split (G77) | Skip-ish | model fits one card | same | same | same | only for H3 bf16 or a full fine-tune across cards |
| Two GPUs in threads vs processes (G81) | threads (no NCCL on Windows) | - | - | - | - | process-per-racer race; NCCL data_parallel only after a profile |
| MiniMax H3 (M5) | block swap, 20-30 s/step est. | pruned int8, no swap | pruned bf16 | bf16 or int8 | NVFP4 or bf16 | race_parallel |

### Run the precision race on a rented card first

**Race it. Expected gain: unmeasured. This race decides whether the quantized training base costs likeness at all, and so whether the recovery-adapter and format-ramp work (S5, S8) is ever needed. Confidence: medium that one race settles it; low that a gap exists. Cost: no code; one rented 96 GB card for about 2 x 2 runs.**

**What it is.** Train the same Krea 2 character config three ways: on a bf16 base (no quantization error), on int8 (W8A8, under 1% error) and on nvfp4 (W4A4, 13.4% forward error). Then score all three LoRAs on the base the user deploys on. The 16 GB pair cannot run the bf16 arm for Krea 2. A 96 GB card can run all three on the same card, at the same resolution, with no model_split and no block swap. That removes the confounders the 16 GB version of this race has.

**Evidence**
- RulzTrainer's own layer measurement on the 5060 Ti: Krea 2 ffn (6144 -> 16384, 4113 tokens), forward error 9.5% fused dequantize vs 13.4% native FP4, at 17.2 vs 4.0 ms. Whole-step average loss 0.0713 vs 0.0724 (README.md:466-486). int8 forward error is under 1% (README.md:432-433).
- The verifier measured on Krea-2-Raw weights: relative weight error 0.89-0.94% for int8_convrot and 9.4-9.5% for nvfp4 (S8 X2-1).
- Inference-only proxies point both ways. int8 convrot LPIPS 0.066 vs fp8 scaled 0.120 in the Comfy Krea 2 thread, which names no GPU. [Nunchaku Krea-2-Turbo W4A4](https://huggingface.co/ModelsLab/Krea-2-Turbo-W4A4-Nunchaku) has LPIPS 0.276 even with a rank-32 branch. Against that, QLoRA practice suggests little harm at 4-8 bits (X2-1 verifier). No source measures LoRA likeness trained on a quantized base.
- [Fizgig MINIMAX_H3.md](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/MINIMAX_H3.md): int8 about 0.17% error, NF4 about 9.5%.

**RulzTrainer today.** `transformer_quant: auto` chooses by the memory the model lives in:
- Krea 2: bf16 from 40 GB, int8 from 22 GB, else nvfp4 (rulztrainer/krea2/loading.py:39, 84).
- Qwen 2.1: bf16 from 24 GB, int8_convrot from 11 GB (qwen21/loading.py:39, 81).
- FIBO: int8 below 22 GB (fibo/loading.py:33).

So a 5090 already trains Krea 2 in int8, and a PRO 6000 or an 80 GB card in bf16. Nobody has compared the arms.

**Fit and risks.**
- The probe must render on the deployment base, not the training base. Krea probes run on the Turbo sampling model (S8).
- On Blackwell rentals (5090, PRO 6000, B200), nvfp4 is the speed option: the FP4 peak is 2x INT8 (RTX 5090 1676 vs 838, whitepaper figures per the G4 finding; the G2 verifier notes the 5090 pair comes from secondary sources). If the race shows no gap, nvfp4 is the fastest race format on those cards. If it shows a gap, use int8 for races and bf16 for the final run.
- Seed spread must be measured (S7, "Measured probe noise").

**Implementation path.** No code. Afterwards, write the result into the `auto` thresholds and the wizard's vast profiles (below): for example, int8 instead of bf16 on 40 GB+ cards if int8 matches bf16 on likeness, since W8A8 is faster.

**How to test.**
- Card: one RTX PRO 6000 96 GB, so all formats are native.
- Model and data: Krea 2 Raw on the AZav character set, 1 MP, rank 16, 1500-2000 steps, 2 seeds per arm. Checkpointing auto (all blocks kept).
- Arms: `transformer_quant: none`, `int8_convrot`, `nvfp4`.
- Probe: ArcFace face plus DINO body every 250 steps. Re-score the final and best checkpoints on the bf16 base and on the int8 base.
- Stop rule: if |bf16 - nvfp4| best ArcFace is within the seed-to-seed spread, close the topic and default Blackwell races to nvfp4. Otherwise run S5's recovery branch on the same card.
- Record s/step per arm. That gives the speed side of the trade on the PRO 6000.

### No recompute: keep every block on 48 GB+ cards

**Do it (verify and loosen the margin). Expected gain: Fizgig measured 13-27% per step with checkpointing off (2-21% keeping only the last blocks) on A100 80GB, A40 and RTX PRO 6000 at 0.25-2 MP. RulzTrainer's auto keep should already reach this; the delta over today is about 0-5% from a smaller margin (unmeasured). Confidence: high that no-recompute is right on these cards. Cost: ~2 hours.**

**What it is.** With gradient checkpointing, each block's forward runs twice. On a card where every block's activations fit, keeping them all removes the recompute, which is exact. RulzTrainer's `checkpoint_keep_blocks: -1` measures one and three kept blocks on the largest bucket. It then keeps as many as fit, up to all of them (rulztrainer/checkpointing.py:52-61), within `memory_margin` = max(1.5 GB, 8% of the card) (checkpointing.py:64). On a 96 GB card that margin leaves 7.7 GB unused, and on a 141 GB H200 11.3 GB. That is several Krea blocks at 2 MP.

**Evidence**
- [Fizgig NEWS v8.9.5](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/NEWS.md): turning checkpointing off when the run fits was 13-27% faster per step on Krea 2, Qwen Image 2.1, MOSS-Image, Z-Image and MiniMax H3. Keeping the last blocks only was 2-21%. The S5 verifier confirmed it was measured on A100 80GB, A40 and RTX PRO 6000 at 0.25-2 MP.
- RulzTrainer on 16 GB: MOSS int8 at 1 MP kept 6 of 42 blocks, -4%; Krea 2 nvfp4 kept 1 block, about 0% (README.md:1765-1767).
- Activation sizes from RulzTrainer calibration logs (S5 G12 verifier): 0.25-0.29 GB per Krea 2 block at 1046 tokens, about 1.1 GB per block at 1 MP. Qwen 2.1: 0.40 GB per block at 2355 tokens; about 0.35 GB x 32 blocks, roughly 11 GB at 0.5 MP. Arithmetic from these numbers:
  - Krea 2 at 1 MP needs about 31 GB of kept activations plus 25 GB of bf16 weights. That fits 80/96 GB.
  - Krea 2 at 2 MP needs about 2x the activations. That fits 96 GB only with an int8 base.

**RulzTrainer today.** Calibration is in trainer.py:1574-1625. `gradient_checkpointing: false` exists in config.py:201. The OOM retry halves kept blocks (README.md:1758-1759). Nothing here is Windows-specific.

**Fit and risks.** The identity loss adds 1.5 GB of margin (checkpointing.py:64-67), which is correct. With all blocks kept, the peak shifts to the backward start. Probes and samples render on the same card between steps, so the margin must still cover a render. The calibration's two-point slope assumes equal per-block cost. With mixed resolutions (`resolutions: [1, 2]`), it calibrates on the largest bucket, which is the right worst case.

**Implementation path.**
1. In `memory_margin`, cap the proportional part, for example max(1.5 GB, min(8% of total, 4 GB)). This is my suggestion, not measured.
2. Log "all N trained blocks kept: recompute off" when k == n_candidates, so a rental run shows it.
3. In the wizard's vast profiles, leave `checkpoint_keep_blocks: -1`. Do not set `gradient_checkpointing: false`: auto falls back gracefully when a 2 MP bucket does not fit.

**How to test.** On an H100 80 GB or PRO 6000:
- Bench Krea 2 bf16 and Qwen 2.1 bf16 at 1 MP and 2 MP, 45 steps, compiled. Run `checkpoint_keep_blocks: 0` against `-1` against `-1` with the capped margin.
- Record the median s/step for steps 21-45, the kept count, peak reserved and `num_alloc_retries`.
- Keep the cap only if it adds blocks with no new alloc retries over a 500-step run with probes on.

### Attention backend per architecture: cuDNN or FA3 on Hopper

**Race it (as a bench). Expected gain: unmeasured. Attention was 42-50% of a compiled 1 MP step on the 5060 Ti (README.md:1834), and its share grows at 2 MP. PyTorch claims cuDNN attention is up to 75% faster than FlashAttention-2 on H100 at the kernel level. Confidence: medium. Cost: ~0.5 day for a per-GPU auto bench; ~1-2 days for an `fa3` backend.**

**What it is.** RulzTrainer sets one SDPA priority order per process. `auto` means cuDNN first when compute capability >= 10 (B200, RTX 50, RTX PRO 6000) and PyTorch's own pick elsewhere (rulztrainer/attention.py:32-48). On the local torch 2.14.1 venv, PyTorch's default global order is flash, efficient, math, cuDNN (checked with `torch._C._get_sdp_priority_order()`). So an H100 or A100 rental gets PyTorch's built-in flash kernel. I did not check whether PyTorch prefers cuDNN internally on sm_90. The options on Hopper:
- cuDNN first: an existing value, `attention_backend: cudnn`.
- FlashAttention-3 through the flash-attn package: a new backend.

**Evidence**
- [PyTorch 2.5 blog](https://pytorch.org/blog/pytorch2-5/): the cuDNN "Fused Flash Attention" backend "can provide up to 75% speed-up over FlashAttentionV2" on H100. The 2.5 blog says it was enabled by default on H100 or newer. The [2.5.1 notes](https://github.com/pytorch/pytorch/releases/tag/v2.5.1) list "Several issues related to CuDNN Attention", and the 2.14.1 default order now puts cuDNN last among the real kernels.
- [FlashAttention-3](https://arxiv.org/abs/2407.08608): 1.5-2.0x speedup on H100, up to 740 TFLOPs/s in FP16. The [flash-attention README](https://github.com/Dao-AILab/flash-attention) says FA3 needs H100/H800 and CUDA >= 12.3, with "FP16 / BF16 forward and backward, FP8 forward" only. FA4 (CuTeDSL) is "optimized for Hopper and Blackwell GPUs (e.g. H100, B200)", with no statement on sm_120. Linux is required; Windows "still requires more testing".
- [PyTorch 2.11 notes](https://github.com/pytorch/pytorch/releases/tag/v2.11.0): FlexAttention has an FA4 backend on Hopper and Blackwell (G22 verifier). But DiffSynth measured flex 1.5-3.8x slower per training step on a B200 for Qwen-Image 2.1 (M2 G21), so not FlexAttention.
- G22 local micro-benchmark (Windows, torch 2.9.1): cuDNN beat memory-efficient by 35-38% on the 4060 Ti. enable_gqa gave no speed on cuDNN. This is the same family of result on another architecture.

**RulzTrainer today.** attention.py:24-80 has no FA3 or FA4 path; every family calls `attention.sdpa()`. The B200 already gets cuDNN. The A100 gets PyTorch's flash, which is FA2-class, and FA3 does not run there.

**Fit and risks.**
- FA3 is BF16 forward and backward. It needs [B, S, H, D] layout and a contiguous pack, an extra transpose per call.
- Krea 2's GQA is already expanded (krea2/model.py:103-105).
- Qwen 2.1's text-causal pass would need `causal=True`.
- FP8 attention is forward-only. It fits probes and samples, not training (S1 G8).
- The priority order is process-wide, so it is thread-safe in data_parallel (G22 verifier).
- A pip dependency built for the image's torch and CUDA versions is the main cost.

**Implementation path.**
1. G22 step 4, with sm_90 added: `auto` runs a 20-step micro-benchmark per GPU (cuDNN vs PyTorch's pick, and `fa3` when importable) on the first bucket, and caches the winner in `.rulz_cache` keyed by GPU name and torch version.
2. Optionally `attention_backend: fa3`: in `sdpa()`, call `flash_attn_interface.flash_attn_func` for maskless bf16 calls on sm_90 and fall back to SDPA otherwise. Install it in `scripts/vast/provision.sh` only on sm_90 hosts.

**How to test.**
- Card: H100 80 GB. Run Krea 2 bf16 and Qwen 2.1 bf16 at 1 MP and 2 MP, 45 steps compiled, with backends torch, cudnn and fa3.
- Record the median s/step for steps 21-45. Losses must match within bf16 noise over 20 steps.
- Adopt the per-GPU auto if the winner beats `torch` by at least 5%.
- One 300-step sanity race with identical seeds: the probe curves must coincide.

### Quantized GEMMs on datacenter cards: int8 vs FP8 vs NVFP4 (G4, G82 revisited)

**Race it (`int8_backward`, no code); Watch (FP8 dgrad, NVGEMM). Expected gain:**
- **W8A8 int8 forward vs bf16 on H100/B200:** the GEMM ceiling is 2x (peak INT8 is 2x BF16), not the 4x of GeForce. End to end unmeasured.
- **FP8 forward over int8:** 0 (equal peak).
- **FP8 or MXFP8 dgrad:** 1.27-1.60x per backward GEMM vs bf16 (measured on B300), useful only if the int8 backward hurts likeness.

**Confidence: low. Cost: 0 for the race; ~3-5 days for an FP8 dgrad path; NVGEMM depends on the custom-op work in S1.**

**What it is.** G4 was dropped because on GeForce, FP8 with FP32 accumulate runs at half the INT8 rate, row-wise `_scaled_mm` is unsupported on sm_120, and fast accumulate gave no gain on the 4060 Ti. Datacenter parts list one FP8 rate equal to the INT8 rate and no accumulate halving. So FP8 matches W8A8 there but still does not beat it. The open question is only the backward: RulzTrainer's exact bf16 grad_x vs its existing `int8_backward: int8` vs an FP8 E5M2 or MXFP8 dgrad with more range.

**Evidence**
- [H100 SXM5](https://glennklockwood.com/garden/processors/H100), dense: BF16 989.4, FP8 1978.9, INT8 1978.9 TFLOPS; 80 GB HBM3. [B200](https://glennklockwood.com/garden/processors/b200), HGX variant, dense: BF16 2250, FP8 4500, INT8 4500, FP4 9000; 186 GB HBM3e, 8 TB/s.
- B300 INT8 is unverified: secondary sources conflict sharply, with some listing a cut INT8 rate (web search, no NVIDIA datasheet found). Do not assume W8A8 int8 is fast on a B300 rental. Bench it, or use nvfp4/bf16 there.
- GeForce ratios (G4 finding, NVIDIA Ada whitepaper; the RTX 5090 pair from secondary sources per the G2 verifier). RTX 4090: INT8 660.6, FP8/fp32-acc 330.3, BF16/fp32-acc 165.2. RTX 5090: INT8 838, FP8/fp32-acc 419, BF16/fp32-acc 209.5, FP4 1676.
- G4 verifier, measured: row-wise `_scaled_mm` unsupported on sm_120, so the 5090 and PRO 6000 inherit that. On H100 the row-wise recipe runs: [torchao](https://docs.pytorch.org/ao/main/workflows/training.html) measured Llama3-8B on 8x H100 at +25.03% tensorwise and +10.05% rowwise over bf16, and on 8x B200 at +25.4% tensorwise, +4.0% rowwise and MXFP8 +20.0%. The page says "For small shapes ... speedup < 1". These are LLM pretraining shapes; they do not transfer directly to batch-1 LoRA.
- [KohakuMoss B300 doc](https://raw.githubusercontent.com/KohakuBlueleaf/KohakuMossTrainer/main/docs/performance/training-b300.md): MXFP8 vs bf16 dgrad was q 1.27-1.29x, up+gate 1.38-1.42x and down 1.56-1.60x, at 16k/32k tokens. Kohaku estimates "10-20% off a merged DiT step"; G4 notes that figure is a projection, not a training measurement.
- NVGEMM (G82): torch 2.14 added CuTeDSL CUTLASS kernels to Inductor, with scaled and NVFP4 GEMMs ([2.14 notes](https://github.com/pytorch/pytorch/releases/tag/v2.14.0)). The verifier: datacenter Blackwell sm_100 is the main CUTLASS target, and RulzTrainer's rescale is plain torch ops after `_scaled_mm` (quant.py:880-884).

**RulzTrainer today.**
- W8A8 via `torch._int_mm` with the exact bf16 backward by default; `int8_backward: int8` is optional (config.py:204-205). On the 5060 Ti the int8 backward measured 6.7 -> 4.7 ms fwd+bwd on the ffn up layer (README.md:407-409).
- NVFP4 native when capability >= 10 (quant.py:836-844), so a B200 takes the native path, untested on sm_100. A100, H100 and H200 dequantize nvfp4, which is slow. Never pick nvfp4 there.

**Fit and risks.**
- `int8_backward: int8` changes gradients: per-token int8 on heavy-tailed grad_out, about 1% error per the G6 verifier. Any FP8 path must stay inside `_W8A8LinearFn` so that compile graph breaks, block swap and data_parallel behave as today.
- NVGEMM needs the quantized linears as traceable custom ops (S1, G13).

**Implementation path.** Nothing before the race. If int8 backward costs likeness on Hopper or B200, add `int8_backward: fp8` as a per-row E4M3 weight copy times per-token E5M2 grad_out through `torch._scaled_mm`, gated on capability 9.x/10.x. If a later torch on the rental exposes NVGEMM for `_scaled_mm`, re-test with `max_autotune_gemm_backends` including NVGEMM once the custom ops exist.

**How to test.**
- Card: H100 80 GB. Krea 2 int8_convrot at 1 MP, all blocks kept, 2 seeds, 1500 steps: `int8_backward: bf16` vs `int8`.
- Metrics: s/step and ArcFace/DINO probe curves.
- Stop rule: if the int8 backward is within seed spread and at least 10% faster, make it the H100/B200 profile default and drop the FP8 dgrad idea. Repeat the speed half on a B200.

### Full fine-tune in one window on 96-180 GB cards

**Prototype. Expected gain: rotation overhead goes away and each component trains every step instead of 1/N of steps; compile becomes possible (measured 12-26% per step on LoRA runs, README.md:1814-1822; unmeasured for a fine-tune). Likeness: unmeasured. Confidence: medium (0.5). Cost: ~1 day.**

**What it is.** The fine-tune by rotation packs per-component windows into what the card holds. Each window trains as fp32 TunedLinear for some epochs and is then swapped back (rulztrainer/finetune.py:1-40). On a card that holds every component, plan_windows returns one window, and the swapping and the compile exclusion have no reason left.

**Evidence and arithmetic**
- finetune.py:54 counts 10 B/param for adafactor and 12 for adamw8bit, plus 2 B anchor (ANCHOR_BYTES).
- finetune.py's docstring gives the RAM for the deltas at 2 B per trained weight: MOSS 10 GB, so about 5B trained weights; Krea 2 24 GB, so about 12B. One window therefore needs about 60 GB for MOSS and about 144 GB for Krea 2 at 12 B/param, plus the base. My arithmetic.
- With S5's in-backward Adafactor step (8 B/param with anchor), Krea 2 is about 96 GB.
- [Fizgig FINETUNE.md](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/FINETUNE.md) (G16 verifier): naive Krea 2 full FT about 78 GB under their accounting; H3 4/5/9 windows on 32/24/16 GB.
- Measured locally: MOSS int8 at 0.25 MP on the 4060 Ti uses 10 windows, 13.1 GB peak, about 1.4 s/step (README.md:1389).
- The verifier found no Windows CPUOffloadOptimizer data; offload is irrelevant at this size.

**RulzTrainer today.** The finetune excludes data_parallel, compile and races (finetune.py docstring; trainer.py:878-879). The optimizer is rebuilt per window (trainer.py:1321-1336).

**Fit and risks.**
- One window means TunedLinear layers stay in place. Compile can trace them once, but the window swap path must not run.
- B200 (180 GB): Krea 2 in one window. PRO 6000 96 GB: MOSS and probably Qwen 2.1 in one window, Krea in about 2. H100 80 GB: MOSS in one.
- A full fine-tune across NVLink GPUs would need FSDP/ZeRO, which RulzTrainer does not have. NVLink does not help today.

**Implementation path.** In trainer.py, when `len(self.ft.windows) == 1`, skip the swap-out at the window boundary and lift the `self.ft is not None` compile exclusion (trainer.py:878-879) for that case. Add a vast profile "fine-tune, one window" for B200 and PRO 6000. The existing `finetune_extract_ranks` export is unchanged.

**How to test.**
- Card: PRO 6000. MOSS int8, 0.5 MP, 1 window (compiled) vs a forced 4-window plan (eager).
- Run equal wall-clock (2 h), extract rank 64 at each cycle boundary, and probe with ArcFace and DINO.
- Adopt if the best probe at equal wall-clock beats the 4-window arm by more than seed spread.

### Multi-GPU boxes: racers as processes, not threads

**Prototype. Expected gain: each racer gets torch.compile back (measured -12% Krea 2 nvfp4, -26% MOSS int8 per step on one GPU, README.md:1814-1822) and no GIL contention. The contention was measured on the local pair: Krea 2 nvfp4 trains at 8.5 s per image on the 5060 Ti alone, but 14.2 and 12.4 s per batch with both GPUs in threads (README "Unequal GPUs"). Rental gain unmeasured. Confidence: medium (0.45). Cost: ~3-5 days.**

**What it is.** race_parallel trains one configuration per GPU in threads of one process, with compile off, because "the copies' threads would trace at once" (rulztrainer/race_parallel.py:14, trainer.py:876-877). On a 4-8 GPU rental, that gives up compile on every racer and puts 4-8 dispatch threads behind one GIL. Independent processes, one per GPU and one config each, need no NCCL: racers only exchange LoRA files at round boundaries. A controller reads each racer's probe scores, applies the race rule (drop lowest, patience, relay), and for a relay copies the winner's LoRA and optimizer state (`save_state`) to the others.

**Evidence**
- G81 verifier: free-threaded Python is not viable with Triton in 2026. A Linux NCCL process mode is worth about 0-5% at 2 GPUs and "could matter on 4-8 GPU rentals". Profile first: build an NCCL data_parallel only if idle share exceeds 15% and grows with GPU count (S6).
- data_parallel at 2 GPUs: MOSS int8 compiled 4.79 s per 2 images (README.md:255).
- M5 tiers: "Multi-GPU boxes: run a parallel race, one configuration per GPU, rather than data_parallel. The user's workload is batch 1."
- NVLink: data_parallel moves only LoRA grads and weights device to device (multigpu.py:57-80), megabytes per step, so NVLink versus PCIe does not matter for LoRA.

**RulzTrainer today.** `multi_gpu: data_parallel | model_split`, threads only (multigpu.py:1-12). `devices` takes any list (multigpu.py:26). train_queue.py runs jobs one after another. Nothing runs several training processes at once.

**Fit and risks.**
- Each process caches latents and text for itself (duplicate .rulz_cache work), unless the cache is built once before the racers start.
- Probe scorers load per process. RAM per process: about 2x the frozen weights while loading.
- The compile cache (S6) should be shared on disk.
- The race's paired-seed logic (S7) works the same, since the racers render identical seeds.
- Linux only is fine for rentals. Windows keeps threads.

**Implementation path.**
1. New `rulztrainer/race_procs.py`: spawn `python -m rulztrainer.train --set device=cuda:i --set race_member=...` per racer, with `CUDA_VISIBLE_DEVICES` pinned.
2. Rounds sync on files under output/<job>/round_k/ (probe JSON, LoRA, state).
3. The controller reuses race.py's decision functions.
4. Option `race_parallel: processes` (Linux; falls back to threads on Windows).

**How to test.**
- Box: 4x RTX 5090. Run 4 racers with 4 configs, Krea 2 nvfp4 at 1 MP, 3 rounds of 250 steps: threads (today) vs processes.
- Metrics: wall-clock per round, per-GPU idle share (`RULZ_PROFILE_STEP=20`), and probe curves at the same seed. They must coincide within bf16 noise, since the math is the same apart from compile.
- Adopt if wall-clock per round drops at least 15%.

### Batch size and resolution on big cards

**Skip-ish (batch > 1); Race it (2 MP vs 1 MP). Expected gain: batch > 1 is throughput-flat to negative on a 5090 (KohakuMoss, measured) and unmeasured elsewhere; no source measures likeness against training resolution for these models. Confidence: medium on batch, low on resolution. Cost: zero code for both races; length bucketing (G21) only if batch > 1 is ever adopted.**

**What it is.** Spare VRAM can buy:
- (a) no recompute, covered above;
- (b) a higher-precision base, covered above;
- (c) batch > 1;
- (d) higher resolution.

On the evidence, the memory should go to (a) and (b) first.

**Evidence**
- [KohakuMoss 5090 doc](https://raw.githubusercontent.com/KohakuBlueleaf/KohakuMossTrainer/main/docs/performance/training-5090.md), MOSS LoRA, all-resident, fixed 294,912-token step: 4608x64 gave 1.66 samples/s; 36864x8 gave 1.26. "Bigger micro-batches lose when nothing is streamed". The smallest micro-batch row also used partial checkpointing, a confounder.
- [KohakuMoss README](https://raw.githubusercontent.com/KohakuBlueleaf/KohakuMossTrainer/main/README.md): on the 5090, 16384 -> 32768-token budgets gave 1.26 -> 1.34 samples/s while tok/s stayed flat (5050 vs 4975). The G12 verifier: the "gain" is packing, not throughput. Member G12/123 stays refuted on big cards.
- G21 (M2): at batch > 1, mixed caption lengths need a key mask (Krea passes a padding mask, krea2/model.py:109; Qwen key_mask, qwen21/model.py:320-325). cuDNN accepts masks, and flash does not. Length bucketing gain is unmeasured.
- Resolution data points, all speed-only: DiffSynth Qwen-Image 2.1 LoRA at 2048x2048 on a single B200, segmented attention 27-28 s/step on torch 2.8 and about 15 s/step on 2.12 (M2 G21); diffusers Qwen-Image 2.1 inference on H100 at 56.5 GiB peak (M2 G21).
- The local recipes: Fizgig's Qwen 2.1 presets at 0.5 MP (S5); RulzTrainer's Krea default 1 MP.

**RulzTrainer today.**
- `batch_size: 1` default (config.py:272).
- The README recommends batch 1 with gradient accumulation on 16 GB (README.md:1741).
- `resolutions` accepts up to 16 MP (config.py:524-526).
- `resolution_step_up` exists (config.py:219-221).
- The lr, `lora_ema` decay "per image" and per-image loss watch are tuned per image at batch 1.

**Fit and risks.**
- Batch > 1 with gradient accumulation semantics changes images per step, so step counts in the wizard's recipes no longer mean the same thing.
- data_parallel already multiplies images per step by the GPU count (README.md:245).
- 2 MP doubles the per-block activation memory and roughly doubles the attention share of the step.

**Implementation path.** None for batch. For resolution, a wizard option "train at the deploy resolution" on 80 GB+ profiles, set to `resolutions: [1, 2]` with `resolution_step_up: true` (existing).

**How to test.**
- Card: PRO 6000. Krea 2 bf16, the same character, all blocks kept, 2 seeds. Arms: `resolutions: [1]` vs `[2]` vs `[1, 2]` with step-up, at equal wall-clock.
- Probe: ArcFace on close-up and full-body prompts rendered at 2 MP, plus DINO body.
- Stop rule: keep 2 MP only if full-body ArcFace beats 1 MP by more than the seed spread at equal wall-clock.
- Batch: skip unless someone wants a throughput bench. That bench is batch 1 vs 2 vs 4 at 1 MP on an H100, s/image only.

### Rental environment: template, torch pin, compile cache

**Do it. Expected gain: avoids wrong-architecture surprises and keeps the S6 compile cache valid; speed effect indirect. Confidence: high. Cost: ~0.5 day.**

**What it is.** Small fixes so big-card rentals run the paths described here:
- `scripts/vast/create_template.sh` searches `gpu_ram>=16 num_gpus=1` with `cuda_max_good>=12.8` and image `vastai/pytorch:cuda-12.8.1-auto`, and describes itself as a "MOSS-Image LoRA trainer" (create_template.sh:17-30).
- `provision.sh` runs `install.sh system`, which keeps the image's torch (install.sh:9-22; requirements `torch>=2.9`). The local venv runs torch 2.14.1+cu130 with triton-windows 3.8.
- The image's torch version is not shown on [Docker Hub](https://hub.docker.com/r/vastai/pytorch), which lists a newer `cuda-13.2.1-auto` tag.
- A different torch version breaks the portable compile cache (S6) and changes `_scaled_mm`, NVFP4 and cuDNN behaviour.

**Evidence.** S6 "Portable compile cache for vast.ai": 160 s cold vs 36 s warm (Windows). The cache needs the same torch, triton and GPU type. [vast.ai pricing](https://vast.ai/pricing), fetched 2026-10-31, lists "68+ GPU Types ... From RTX 3060 to B200" and "NVIDIA GB200s are now available". Prices did not render (the grid loads client-side), so none are quoted here.

**Implementation path.**
1. Add `TORCH_VERSION` to provision.sh: install the local version's wheel for the host's CUDA instead of `system` when the image differs.
2. Export TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR to /workspace (S6).
3. Add template variants by `SEARCH` (`gpu_name=RTX_5090`, `gpu_ram>=80`, `num_gpus>=4`).
4. Install flash-attn 3 only on sm_90 hosts, if the attention bench above adopts it.
5. Log the GPU's compute capability and the resolved quant, attention and kept-block choices at the top of train.log.

**How to test.** Provision a 5090, an H100 and a B200 instance. Run `scripts/bench.py` for Krea 2 at 1 MP for 20 steps and check the logged choices against the profile table below.

### Items that stay dropped on big cards

- **G14 CARE-LoRA.** It only trims memory in kept blocks. On 80 GB+ memory is not binding, and on a 5090 the kept-block lever is S5's SAC tier, which is exact.
- **G12/123, bigger micro-batch under block swap.** No block swap on big cards. KohakuMoss shows bigger micro-batches losing when nothing is streamed. Still refuted.
- **G9 fp8 kept activations, G17 MsT, G18 block-swap buffers, G19 BlockSkip.** All buy memory that big cards already have. G19's speed half (suffix pass-through for mid-range `lora_blocks`) is card-independent and lives in S5.
- **G20 allocator.** train.py:10-18 already sets `expandable_segments:True` on Linux, so every vast instance has it. The other two settings do nothing without a memory fraction (verifier).
- **G15/G51 base_precision_mix and the per-block ramp.** The premise was "Krea int8 does not fit 16 GB". From 22 GB, auto already picks int8. Moot unless nvfp4 is chosen for speed on Blackwell, and then the precision race decides.
- **G77 micro-batch pipeline for model_split.** On rentals the model fits one card, and the verifier says it "never beats data_parallel when the model fits one GPU". The only remaining case is MiniMax H3 bf16 (66 GB) or a full fine-tune split across two 48 GB cards. Those are better served by renting one 80-96 GB card.
- **G2 SageBwd INT8 training attention.** Its case rests on GeForce INT8 being 4x BF16-with-fp32-accumulate. On H100 and B200, INT8 is 2x BF16 and the BF16 attention baseline (cuDNN, FA3) is stronger, so the ceiling roughly halves. Keep it a 5090/PRO 6000 prototype per S1.
- **G21 FlexAttention.** Measured 1.5-3.8x slower per training step on a B200 (DiffSynth). Do not build it.
- **S2b P8 dense first-gradient init.** On 80 GB the dense G can be materialized without sketching, but the init-only gain is +0.2 to +2.2 points on LLMs. Card size does not change the verdict.
- **MiniMax H3.** M5 already has the tier table: pruned int8 with no swap from about 30 GB free, a normal run at 48 GB, pruned bf16 at 96 GB, NVFP4 native on Blackwell. Its first likeness race belongs on a 48-96 GB card.

### Recommended vast.ai profiles

These are what the wizard could offer as a "rented card" choice. Every s/step is unmeasured; the first rental of each class should run `scripts/bench.py` and the precision race.

- **RTX 5090 32 GB (sm_120)**
  - Quant: Krea 2 int8 (auto) for final runs; nvfp4 for races if the precision race shows no gap. MOSS and Qwen 2.1 int8_convrot.
  - Attention: auto (cuDNN).
  - Compile: on, with the persistent cache (S6).
  - Checkpointing: auto keep (partial at 1 MP for Krea); SAC tier when built (S5).
  - Batch and resolution: batch 1, 1 MP.
  - Multi-GPU: on 2-8x 5090 boxes, race processes (one config per GPU), else race_parallel threads.
- **RTX PRO 6000 Blackwell 96 GB (sm_120)**
  - Quant: bf16 for the reference and final run; nvfp4 or int8 for races per the precision race.
  - Attention: auto (cuDNN). Compile: on.
  - Checkpointing: auto keep, all blocks at 1 MP, with the capped margin.
  - Batch and resolution: batch 1; 1 MP, or `[1, 2]` with step-up if the 2 MP race wins.
  - Fine-tune: MOSS/Qwen 2.1 one window, Krea about 2 windows.
  - This is the card for the precision race and for H3's first race.
- **A100 80 GB (sm_80)**
  - Quant: bf16, or int8_convrot W8A8 for speed. Never nvfp4: it dequantizes.
  - Attention: PyTorch flash (auto). FA3 does not run on sm_80.
  - Compile: on. Checkpointing: auto keep (all at 1 MP).
  - Batch and resolution: batch 1, 1 MP.
- **H100 / H200 80-141 GB (sm_90)**
  - Quant: bf16 for final runs, int8_convrot for races. Race `int8_backward: int8`.
  - Attention: per-GPU bench of cuDNN vs flash vs FA3.
  - Compile: on. Checkpointing: auto keep, all blocks.
  - Batch and resolution: batch 1; 1-2 MP.
  - Fine-tune: MOSS one window; Krea 2-3 windows (H200 1-2).
- **B200 180 GB (sm_100)**
  - Quant: nvfp4 native for races (untested on sm_100: check the first run's log); bf16 for final runs.
  - Attention: auto (cuDNN).
  - Compile: on. Checkpointing: all blocks kept.
  - Batch and resolution: batch 1; 1-2 MP.
  - Fine-tune: Krea 2 in one window.
  - Bench int8 before trusting it on B300-class parts (INT8 rate unverified).
- **Multi-GPU NVLink nodes (4-8x H100/B200, or 5090 boxes)**
  - One configuration per GPU (race processes once built; race_parallel threads until then).
  - Not data_parallel for character LoRAs: images per step multiply.
  - An NCCL data_parallel only after S6's profile shows idle share above 15% growing with GPU count.
  - NVLink adds nothing for LoRA exchange.

## Krea 2

This section covers Krea 2 Raw/Turbo: what the community and the trainers converge on, which of it RulzTrainer already does, and the Krea-specific items from the theme sections. The overall verdict is that RulzTrainer's Krea defaults already match the calibrated community recipe (adamw8bit 1e-4, Krea targets with attn_gate and text_fusion, per-bucket shift, isolation on, EMA 0.98, Turbo probes at strength 1.0, early stop). The useful deltas are diagnostic or cheap: a no-trigger bleed probe, a post-hoc text_fusion scale search, a fix for a latent Turbo-base sampling bug, and a race of training on Turbo with Ostris' de-distillation adapter. None of them has a measured likeness gain on a Rulz-style run yet.

### Status in RulzTrainer and model facts

**Status.** Supported for training on Raw (the default) and on Turbo (a Turbo base loads as distilled through the HF model_index `is_distilled`, krea2/loading.py:104, or "turbo" in a ComfyUI file name, krea2/single_file.py:159). Samples and probes go through Comfy-Org's rank-64 Krea 2 Turbo LoRA (`turbo_lora: auto`) at strength 1.0, 8 steps, CFG 1 (config.py:39-46), with the distilled schedule pinned at mu 1.15 (krea2/sampler.py:15, 37). Measured in the README (Krea 2 section, README.md:1590 onward): one RTX 5060 Ti trains in nvfp4 at 9.2 s per step at 1 MP, peaking at 14 GB; int8 model_split over the 5060 Ti + 4060 Ti runs 10.0 s per step at 1 MP. cuDNN attention first cut Krea 2 nvfp4 from 9.40 / 8.56 s to 7.90 / 6.70 s per step on the 5060 Ti (README.md:517-540). There is no training adapter for Krea in `ARCH_TRAINING_ADAPTER` (config.py:58-61, Qwen 2.1 only).

**Model facts.** About 12.9B parameters (25 GB bf16, 13.5 GB int8, about 8 GB nvfp4). Single-stream DiT, 28 blocks, width 6144, 48 query heads over 12 K/V heads (GQA 4:1, head dim 128). The text side is a Qwen3-VL-4B tapped at 12 layers (2, 5, ..., 35) under a fixed system prompt asking for colour, shape, size, texture, quantity, text and spatial relations (34 prefix tokens stripped, 512 text positions; Krea's [encoder.py](https://raw.githubusercontent.com/krea-ai/krea-2/main/encoder.py)), collapsed by a text-fusion stage (2 layerwise blocks across the 12-layer axis, a Linear(12,1) projector, 2 refiner blocks; 20/20 heads, no RoPE, no modulation). Text tokens carry no position in the DiT (krea2/model.py:287; text-fusion blocks have no RoPE, model.py:126). Modulation is one shared `time_mod_proj` plus a per-block `scale_shift_table` [6, 6144] (krea2/model.py:168, 255, 309); its input is the timestep only. Block linears have no bias. VAE is the Qwen-Image VAE (16 channels, packed 2x2 to 64). Turbo is a TDM distill (guidance plus timestep) sampled at 8 steps, guidance 0, mu 1.15; Raw is sampled at 28 steps, Krea guidance 4.5 (cfg 5.5 in RulzTrainer's convention) ([diffusers Krea 2 docs](https://huggingface.co/docs/diffusers/api/pipelines/krea2)). Krea's stated intent: "LoRAs trained on RAW are designed to express strongly on Turbo" ([Krea 2 open-source page](https://www.krea.ai/krea-2-open-source)).

### Recommended recipe

Settings the evidence supports for a character LoRA on the user's hardware. Most are already the wizard default; the source is named for each.

- **Base and quant.** Train Raw; nvfp4 on the 5060 Ti alone, or int8 model_split over both cards (README Krea 2 section). Int8 ConvRot is faster than fp8 elsewhere too: [musubi-tuner krea2.md](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md) measured convrot_int8 5.3 s vs fp8_scaled 7.1 s per step on a 3090 at 1024, and 1.9 vs 2.0 s (bf16) on an RTX PRO 6000 Blackwell. [OneTrainer PR #1568](https://github.com/Nerogar/OneTrainer/pull/1568) notes int8 "makes some noise in the background sometimes but not too bad".
- **Rank 16, alpha 16** for characters (the wizard's pick). The community uses 32/32 (musubi: "reproduce the model authors' recommended default"; [MJSAA/hannahowo-krea2-lora](https://huggingface.co/MJSAA/hannahowo-krea2-lora)), SimpleTuner 64, Fizgig r8 adaptive / r32 standard / r16 style ([Fizgig KREA2.md](https://github.com/shootthesound/Fizgig/blob/master/docs/KREA2.md)). Fizgig's isolation test that fixed bleed ran r16 ([LORA_ISOLATION.md](https://github.com/shootthesound/Fizgig/blob/master/docs/LORA_ISOLATION.md)). The verifier judged r32 not warranted for rank 8-16 practice; race it only if curious (see RACES).
- **LoRA, not LoKr.** Fizgig: LoRA about 20% faster than LoKr; LoKr at factor <=8 and LR 5e-5 if used. Matches the user's own tests.
- **LR 1e-4, adamw8bit, constant** (wizard default; musubi's example command; hannahowo 1e-4 constant; SimpleTuner 1e-4 with Lion). Higher rates peak early: OneTrainer's 16 GB preset uses 3e-4 at batch 2 ([preset JSON](https://raw.githubusercontent.com/Nerogar/OneTrainer/master/training_presets/Krea%202/%23krea2%20LoRA%2016GB.json)), and at 3e-4 r64 the 200-step character checkpoint replaced the 1000-step one ([jayark/sammy-krea-v4-loras](https://huggingface.co/jayark/sammy-krea-v4-loras)). Fizgig's isolation recipe runs 2e-4 to 4e-4 adaptive.
- **Targets: core + attn_gate + text_fusion; leave `embed` off** (default, config.py:24-29). The txtmlp/tmlp/tproj layers inside `embed` are the worst-fit layers of the Turbo-minus-Raw delta (SVD residual 46.1% / 41.5% / 36.6% at rank 64 / 128 / 256, [TheDivergentAI/krea2-turbo-distill-lora](https://huggingface.co/TheDivergentAI/krea2-turbo-distill-lora)), so a Raw LoRA that touches them transfers worse to Turbo.
- **Timesteps: `timestep_sampling: reference`**, a logit-normal shifted per bucket by Krea's own mu formula (archs.py:441-449; 0.5 at 256 tokens to 1.15 at 6400).
  - The proposed "per-bucket Krea shift" item was rejected because it is already implemented: Krea `reference_sigmas` uses Krea's mu formula per batch and Turbo sampling pins mu 1.15.
- **Isolation on whenever there is a subject** (wizard, webui.py:3076). Fizgig on Krea 2: a normal LoRA made 48/50 strangers read as the subject, isolation 0/50, at ArcFace 0.77 vs 0.78 (r16, 0.25 MP, INT8 base, DOP every 2nd step at weight 2). The bleed failure ("all women take the subject's face", not seen on Qwen/Z-Image by the same user) is reported in [pmczip/Krea2Raw_Loras discussion #4](https://huggingface.co/pmczip/Krea2Raw_Loras/discussions/4).
- **lora_ema 0.98** (default). Fizgig: EMA "lifts late-epoch likeness and steadies epochs" on Krea 2.
- **Frequent probes plus patience; budget 1000-2000 steps and let early stop decide.** Step counts conflict: 200 steps best (musubi r64 at 3e-4, 25 images), 500-700 steps best with 3000 hopelessly overfit (instasd roundup, OneTrainer LoKr), about 90% likeness by step 1750 of 3000 (one 5060 Ti user in [Krea-2-Turbo discussion #10](https://huggingface.co/krea/Krea-2-Turbo/discussions/10); the "36 images" in the scout note belongs to a different user in that thread). SimpleTuner: "Krea2 can overfit small datasets quickly" ([KREA2 quickstart](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/KREA2.md)). The wizard's probe_every 25 (webui.py:3166) and likeness_patience 5 (config.py:102) fit this.
- **Resolution 0.5-1 MP.** One report says 512 px runs come out soft ([pmczip discussion #3](https://huggingface.co/pmczip/Krea2Raw_Loras/discussions/3)); 1500 steps took about 45 min at 15.1 GB on a 16 GB card with 64 GB RAM in that thread.
- **caption_dropout 0.05** with short, variable-only captions is the community face-LoRA practice ([chengyansen-ai/krea2-lora-training](https://github.com/chengyansen-ai/krea2-lora-training)); unmeasured.
- **attention_backend: cudnn on both cards** (auto already does this on the 5060 Ti; set it by hand for the 4060 Ti half of a model_split until auto covers Ada, see "Attention on Krea" below).
- **Turbo strength.** Probe at 1.0 (default). Community strengths spread 0.7-1.25 (bongobongo 0.7-0.8 for Turbo-trained LoRAs, hannahowo 0.9 at 8 steps CFG 1 mu 1.15, the 5060 Ti user 1.25); pick it with a strength sweep, not by guessing (next subsection).
- **Do not enable a guidance-loss target on Raw** (see "Not recommended for Krea" at the end).

### Bleed probe and final strength sweep

**Do it. Expected gain: diagnostic (turns the Krea bleed failure into a number races and best-checkpoint can use); likeness gain unmeasured and small. Confidence: medium. Cost: probe-side code only, 2-4 extra renders per probe plus a 50-render audit at the end.**

**What it is.** Two Krea-specific failure checks that RulzTrainer does not run. (a) A no-trigger bleed probe: render class-only prompts ("a photo of a woman") with the LoRA on, score each against the subject with ArcFace, and count renders above Fizgig's same-person cut. (b) A final strength sweep: probe the best checkpoint on Turbo at several strengths and report the best one, instead of the 0.7-1.25 guesswork. The scout's mechanism for why Krea bleeds more (the bidirectional text-fusion stage encodes the class word with the trigger in context regardless of caption order, so Fizgig's "describe the stranger first" trick, which relies on causal encoders, does not help) is plausible from the architecture but untested; the Qwen3-VL encoder itself is still causal and only the fusion stage mixes tokens.

**Evidence.**
- [Fizgig LORA_ISOLATION.md](https://github.com/shootthesound/Fizgig/blob/master/docs/LORA_ISOLATION.md): Krea 2, normal LoRA 48/50 strangers read as the subject, isolation 0/50; ArcFace 0.77 vs 0.78. 0.25 MP, INT8 base, r16, close to the user's setup. 50-render protocol.
- [pmczip discussion #4](https://huggingface.co/pmczip/Krea2Raw_Loras/discussions/4): the LoRA applies without its trigger and all women take the subject's face; "sometime affects men as well"; not seen on Qwen/Z-Image by the same user.
- [pmczip discussion #3](https://huggingface.co/pmczip/Krea2Raw_Loras/discussions/3): eye-distance drift in 3/4 views (one user).
- Strength spread 0.7-1.25 across [bongobongo2020/krea2-character-lora-trainer](https://github.com/bongobongo2020/krea2-character-lora-trainer) (Turbo-trained, a different setting), [hannahowo](https://huggingface.co/MJSAA/hannahowo-krea2-lora) and [Krea-2-Turbo discussion #10](https://huggingface.co/krea/Krea-2-Turbo/discussions/10). A LoKr card says Raw LoRAs need the Turbo LoRA at >=0.75.
- All are LoRA runs on consumer cards at batch 1-2, so the settings transfer; they are single-run anecdotes.

**RulzTrainer today.**
- Isolation (DOP + fused trigger + stranger pairs, the Fizgig protocol) exists in isolation.py:1-41 and is on by default for a subject (webui.py:3076).
- `_probe_with` (trainer.py:3264-3309) renders only the face and body prompts, both carrying the trigger. No probe renders a class-only prompt.
- `soup_strengths` already probes 1.0 / 1.15 / 1.3 (config.py:424), but only for soups (trainer.py:3127-3210). Normal runs get no strength sweep.
- Nothing uses face keypoints or yaw for likeness.

**Fit and risks.** Probe-side only: renders use the existing sampling slots (turbo LoRA or Turbo model), scoring runs on the CPU or side GPU. W8A8, nvfp4, compile, keep blocks, block swap, model_split and data_parallel are untouched; no Windows issue. Cost on the 5060 Ti with nvfp4 Turbo at 8 steps is roughly 5-10 s per render at 0.5-1 MP (unmeasured), so keep 2-4 class-only renders per probe and the full 50 for the final audit and race tie-breaks. The 3/4-view inter-ocular check is speculative: InsightFace keypoints are noisy past about 30 degrees of yaw and the drift report is one user's anecdote. Skip it unless the bleed probe or samples show 3/4 drift is common.

**Implementation path.**
1. trainer.py `_probe_with`: add a probe kind `bleed`. Options `probe_bleed_prompt` (default: probe_face_prompt with the trigger removed) and `probe_bleed_seeds` (default 2).
2. Score with the existing face scorer: mean raw ArcFace cosine to the subject plus the count above 0.45 (Fizgig's same-person cut). Write both to the probe record (likeness.py `append_record`) and the status board.
3. Use it as a guard, not a likeness term: flag a best checkpoint whose bleed exceeds a threshold; optional race penalty `race_bleed_weight` (default 0).
4. End-of-run "bleed audit": 50 class-only renders on the best checkpoint, reported in the job report.
5. Generalize the soup strength path into `final_strength_sweep` (same list format as `soup_strengths`), probing the best checkpoint at each strength and suggesting a Turbo strength.
6. All four families get it; turn the per-probe bleed check on by default for Krea 2 only.

**How to test.** Krea 2 Raw nvfp4 with Turbo probes, 30-40 photos of one person, 0.5 MP, rank 16, 1500 steps. Race isolation off vs on (existing lever), face probe and bleed probe every 100 steps, plus the 50-render audit at the end. Pass if the bleed count separates the two clearly (Fizgig: 48/50 vs 0/50) and isolation's face likeness is within 2 points of the plain run. If bleed with isolation off is already below 10/50 on the user's data, keep only the final audit. Strength sweep: on the best checkpoint compare face/body probes at 0.8 / 1.0 / 1.15 / 1.3 on Turbo; keep the sweep only if the best strength differs from 1.0 by more than 2 likeness points in at least 2 of 3 runs.

### Text-fusion scale search (Fizgig "Text fusion x3")

**Race it. Expected gain: 0 to a few ArcFace points on Krea 2 character LoRAs (unmeasured; may be smaller for LoRAs trained with by_type text-side lr). Confidence: medium. Cost: 2-3 probe passes (minutes), no training time.**

Full write-up in the theme section "Likeness from data, teachers, references and Turbo transfer". Krea-specific points:

**What it is.** Multiply the LoRA deltas of the four text-fusion blocks (2 layerwise + 2 refiner) by 2 or 3 after training, keep the blocks at 1.0, pick the scale with the probes, and bake it into the saved file so it works at strength 1.0.

**Evidence.**
- [Fizgig KREA2.md](https://github.com/shootthesound/Fizgig/blob/master/docs/KREA2.md): "measured across several LoRAs, x3 lifted the detail meter and likeness on every one with the composition unchanged"; an overtrained LoRA is the one case where it hurt. No numbers, no count of LoRAs, no meter scale; it does not say whether the projector is scaled. No independent report of x2/x3 helping was found.
- RulzTrainer's README.md:1612-1613 and the config.py:24-26 comment credit Fizgig with "lifted likeness 2-7 points". The current Fizgig doc shows no such figure: re-check the source or soften the wording.

**RulzTrainer today.** text_fusion is a default Krea target and covers layerwise_blocks + projector + refiner_blocks (archs.py:432-434); txtmlp/tmlp/tproj sit in `embed`, off by default. `automagic_groups: by_type` already gives text_fusion and txt_in their own adaptive rate (archs.py:121-133; trainer.py:2054-2056). LoRALinear has a per-module `multiplier` (lora.py:232, 393) but `set_multiplier` sets all modules at once (lora.py:579). The delta is a per-group multiplier at probe time and a per-group bake at save.

**Fit and risks.** text_fusion is outside `compile_blocks`, so changing its multiplier causes no recompile. The quantized base is untouched (the multiplier scales only the fp32 LoRA branch). Baking by scaling lora_up / lora_B is exact for LoRA and LoKr (scale w2); skip OFT and DoRA. Negative on overtrained or late checkpoints, which best-checkpoint selection may already avoid.

**Implementation path.** Option `group_strengths` (e.g. `"text_fusion: 1, 2, 3"`, Krea 2 only) reusing the soup probe machinery (trainer.py:3127-3210); `set_multiplier(injected, value, prefix=None)` in lora.py; groups defined per arch in archs.py (text_fusion, attn_gate, embed); bake in export_state_dict/save_lora and record `group_strengths` in metadata. Do not build a multi-group coordinate descent until the single-group result holds: about 6 groups x 5 scales costs tens of probe passes and will overfit probe seed noise.

**How to test.** 3-5 finished Krea 2 character `_best` LoRAs plus one deliberately overtrained one; probe text_fusion at x1, x2, x3 with probe_seeds 8+. Primary ArcFace face similarity, plus DINO body and a composition guard (DINO of the x3 render vs the x1 render). Accept only if x2 or x3 beats x1 by at least 1 ArcFace point, above the seed-to-seed std, on most LoRAs without the guard dropping. Then race automagic by_type against single to see whether a stronger text-side lr makes the post-hoc boost unnecessary.

### Training on Turbo with Ostris' de-distillation adapter, and the Turbo-base sampling bug

**Do it (bug fix); Race it (Turbo base). Expected gain: probes render the exact deployment model instead of Raw plus a rank-64 extract; under 1 GB sampling memory saved; likeness change vs Raw unmeasured. Confidence: medium. Cost: the race needs no code today; the fix is a few lines in trainer.py and comfy_files.py.**

Full write-up in the theme section "Likeness from data, teachers, references and Turbo transfer". Krea-specific points:

**What it is.** [ostris/krea2_turbo_training_adapter](https://huggingface.co/ostris/krea2_turbo_training_adapter) is a LoRA trained at lr 1e-5 on thousands of Turbo renders that loosens the distillation. ai-toolkit merges it into Turbo at 1.0 for training and inverts it (multiplier -1) for sampling, so the user LoRA is learned on Turbo's weights and sampled on true 8-step, CFG-0 Turbo. RulzTrainer's `training_adapter` is already the mathematically equivalent design: a live frozen branch, on for training and off for samples, probes, reward and self-training renders, never saved (lora.py:238-241, 396-398, 902-941). It works on int8 ConvRot and nvfp4 bases without a merge, so musubi's "merging into int8 is rejected" never arises.

**Evidence.**
- Ostris card: long runs still break the distillation; "could yield better results than training on the raw model" is a hope with no benchmark.
- [ai-toolkit krea2.py](https://raw.githubusercontent.com/ostris/ai-toolkit/main/extensions_built_in/diffusion_models/krea2/krea2.py): merge at 1.0, multiplier -1, `invert_assistant_lora`.
- [Krea-2-Turbo discussion #10](https://huggingface.co/krea/Krea-2-Turbo/discussions/10): RTX 5070 12 GB, Windows 11, 36 images at 1024, rank 32, Turbo + adapter, 25.4 to 22.3 s/step after `lokr_full_rank: false`, 18h45m for 3000 steps; body very good, face acceptable with some drift. Krea staff asked why he trained on Turbo instead of Raw. No head-to-head comparison exists anywhere.
- [bongobongo2020 trainer](https://github.com/bongobongo2020/krea2-character-lora-trainer): Turbo + adapter, inference at 0.8, CFG 1, 8 steps; characters "often overfit late", drop to about 0.7 if overcooked.

**RulzTrainer today and the bug.**
- A Turbo base already loads and samples at mu 1.15 (krea2/loading.py:104, single_file.py:159, sampler.py:37). `training_adapter` takes any path on any family (config.py:324-326) and is attached to every data_parallel replica (trainer.py:775, 792, 2126-2144).
- Latent bug: with a distilled base and `turbo_lora: auto`, comfy_files.py:42-43 still attaches Comfy-Org's Raw-to-Turbo rank-64 LoRA for samples and probes (`_attach_sampling_loras`, trainer.py:2146-2157). It stacks on an already-distilled model and corrupts probe scores. Nothing keys `turbo_lora` or `_check_sample_settings` (trainer.py:2159-2175) off `model.is_distilled`.
- `ARCH_TRAINING_ADAPTER` has no krea2 entry, so `auto` downloads nothing; the adapter name is not written into saved LoRA metadata; and a race shares the training adapter across configs (README.md:1244), so adapter on vs off cannot be raced inside one run.

**Fit and risks.** The adapter branch runs in bf16 beside the quantized GEMM, a few percent at rank 16-64. Block swap and model_split move the buffers with their module; the extras_enabled toggle is a Python bool guard under compile, as for Qwen 2.1. Same s/step as Raw (about 9 s nvfp4 at 1 MP) plus a few percent. The 1000-3000 steps the user runs are about what Ostris calls short; distillation breakdown on longer or high-lr runs is the main risk. Whether the Ostris file's keys and alpha parse correctly is unverified (ai-toolkit sets alpha = rank); the "matched N modules" log catches a total miss, not a scale error.

**Implementation path.**
1. trainer.py: on a distilled base, resolve `turbo_lora: auto` to '' and make `_check_sample_settings` treat `is_distilled` like a turbo LoRA (8 steps, CFG 1). This is the bug fix; do it regardless of the race.
2. config.py: per-family, per-variant `ARCH_TRAINING_ADAPTER`: krea2 Turbo gets ostris/krea2_turbo_training_adapter, Raw gets none; comfy_files.planned_downloads (around lines 42-61) picks by model path or "turbo" in the file name.
3. Write `rulz_training_adapter` (file and strength) into save_lora metadata.
4. Unit test the Ostris file's key and alpha parsing like tests/test_qwen21.py:209.
5. Allow `training_adapter` and `training_adapter_strength` as per-config race keys.

**How to test.** Two runs, same character data and seeds, 0.5-1 MP, rank 16, same lr, 3000 steps, nvfp4 on the 5060 Ti. A: Raw with `turbo_lora: auto`. B: `model_path: krea/Krea-2-Turbo`, `training_adapter: <Ostris file>`, `turbo_lora: ''` (the last setting is essential until the fix lands). Optional C: Turbo base, no adapter, to confirm the adapter prevents breakdown. Probes every 250 steps, ArcFace face and DINOv2 body, all rendered on Turbo at 8 steps CFG 1 with the same seeds (for A use `sample_model_path: krea/Krea-2-Turbo` at least on the final checkpoints). Check adapter-off renders for structural breakdown (hands, lines, DINO body falling while face rises). Stop at 3000 steps or 3 probes without improvement. B wins only if its peak face likeness is at least 2 points higher, or it reaches 90% of peak at least 20% sooner, on 2 datasets.

### Attention on Krea: enable_gqa and cuDNN on the 4060 Ti

**Do it (cuDNN on Ada); Skip-ish (enable_gqa). Expected gain: enable_gqa about 0% speed, 0.1-0.5 GB less retained K/V; cuDNN on the 4060 Ti an estimated 6-8% of step time at 1 MP when it trains. Confidence: medium. Cost: small change in attention.py and krea2/model.py.**

Full write-up in the theme section "Speed per step". Krea 2 is the only supported model with GQA (48q / 12kv); its text-fusion blocks are 20/20, so no GQA there.

**Evidence (local micro-benchmark by the verifier, torch 2.9.1+cu130 on Windows, GPUs partly busy, medians of 7).**
- 5060 Ti, cuDNN, 48q/12kv, d128, bf16, fwd+bwd: repeat 26.0 vs gqa 26.1 ms at S=2248; 42.3 vs 42.0 ms at S=4296. No speed gain.
- 4060 Ti, cuDNN: 48.5 vs 47.8 ms and 164.3 vs 161.6 ms, within noise.
- Peak extra memory per call at S=2304: 276 MB repeat vs 233 MB gqa. cuDNN + enable_gqa gradients match the repeat path and the math reference with and without a mask (relative error at most 8e-3 in bf16).
- The memory-efficient kernel rejects enable_gqa on 2.9.1; under PyTorch's default order the call falls to math, about 4 GB per call and 6-10x slower.
- Side result: cuDNN vs memory-efficient on the 4060 Ti, 48.5 vs 75.5 ms at 0.5 MP and 164 vs 263 ms at 1 MP; on the 5060 Ti 26 vs 60 ms and 42 vs 102 ms.
- The "attention is 41% of DiT time" figure ([Comfy-Org/Krea-2 discussion #6](https://huggingface.co/Comfy-Org/Krea-2/discussions/6)) is one community inference run with no resolution or GPU stated; the README puts attention at about a fifth of a 1 MP training step.

**RulzTrainer today.** krea2/model.py:103-109 expands K/V with repeat_interleave; the mask is already None when nothing is padded and the sequence is never padded to 256 (krea2/model.py:311-316), so the scout's mask-free and no-pad-256 ideas are already in. attention.py:43-48 puts cuDNN first only on Blackwell; README.md:517-540 says Ada was not measured.

**Fit and risks.** enable_gqa must be gated on the resolved backend being cuDNN (or a flash/efficient-GQA capability check), or the 4060 Ti under `auto` falls to math and OOMs. The alignment tap (alignment.py:92-98) expects matching head counts and must expand k itself while active. Windows flash needs torch 2.15, a large jump that would require re-validating triton-windows and the int8/nvfp4 kernels. For Krea the cuDNN-on-Ada change matters directly because the two-card Krea path is int8 model_split over the 5060 Ti + 4060 Ti.

**Implementation path.** attention.py: `enable_gqa` parameter in `sdpa()` and a `gqa_ok()` helper; krea2/model.py: skip repeat_interleave when `gqa_ok()`; extend `auto` to pick cuDNN on capability 8.9, or better a 20-step per-GPU micro-benchmark cached in .rulz_cache. Update README and wiki.

**How to test.** Krea 2 nvfp4, 1 MP, batch 1, 20 steps on the 5060 Ti, twice repeat vs twice gqa, same seed, calibrated keep blocks: pass if loss matches within 0.0007 per step, peak VRAM is lower or keep blocks higher, and s/step is not worse. Then 20 steps on the 4060 Ti (Krea int8 split, MOSS int8, Qwen 2.1 int8 at 0.5 and 1 MP), `attention_backend` torch vs cudnn; make cuDNN the Ada default if it wins at least 5% s/step at equal loss and memory.

### TREAD token routing on Krea

**Prototype. Expected gain: Krea 2 nvfp4 at 1 MP from about 9 s to about 7-7.5 s per step (15-22%, estimate, unmeasured), plus 0.5-1.5 GB of routed activations that calibration turns into more keep blocks. Likeness: no direct upside, mismatch risk. Confidence: medium. Cost: per-arch gather/scatter in the block loop, config and race plumbing.**

Full write-up in the theme section "Speed per step". Krea-specific points:

- SimpleTuner already ships TREAD for Krea 2 (merged PR #2800, June 2026, together with CREPA, LayerSync and block swap) but reports no benchmark for it; its docs claim "typically 20-40%" with no GPU named, warn of a larger initial loss spike with LoRA/LoKr and of occasional LoRA slowdowns, and are inconsistent about whether 0.5 is safe or >0.35 "definitely affects convergence" ([TREAD.md](https://github.com/bghira/SimpleTuner/blob/main/documentation/TREAD.md)).
- Krea is one of the easier ports: the block loop is krea2/model.py:328-338; cos/sin are shared [L, ...] (krea2/model.py:321), fine at batch 1 with one index set per step; the sigmoid attention gate is per token and follows x. model_split device moves (krea2/model.py:331-335) must carry the index set.
- Force-keep face/person cells (region weight > 1) so the identity loss, attention localization and region prior see the face. Route only the main training forward (trainer.py:975); region-prior, DOP/isolation, probes and samples run unrouted; keep an unrouted final 10%.
- Fix the kept count per bucket (round to 64) so compile sees one extra shape per bucket.
- First race: Krea 2 nvfp4, 1 MP, rank 16, 1500 steps, `tread_keep` 1.0 vs 0.6 (start 3, end -4, tail 0.1, regions kept), probes every 100 steps; compare ArcFace/DINO against wall-clock seconds, not steps; accept only if likeness at equal wall-clock is at least the unrouted run's within probe noise. Watch the per-image loss watch for the first-200-step spike and consider freezing automagic's lr adaptation for the first 100 routed steps.

### Per-block format ramp for nvfp4 Krea

**Watch (cheap offline check first). Expected gain: unmeasured for training; last 4 blocks in int8 cost about 0.9 GB, a rank-32 frozen residual about 0.3-0.4 GB and under 1% of step time. Confidence: low. Cost: offline error measurement first; the full feature is a quant-filter generalization.**

Full write-up in the theme section "VRAM without losing speed or quality". Krea-specific points:

- The "bf16 islands" half is already in: krea2/loading.py:35-36 quantizes only `transformer_blocks.*`, so first/last, tmlp/tproj/txtmlp and text fusion stay bf16, matching [ai-toolkit](https://raw.githubusercontent.com/ostris/ai-toolkit/main/extensions_built_in/diffusion_models/krea2/krea2.py) and [musubi krea2_utils.py](https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/src/musubi_tuner/krea2/krea2_utils.py) (txtfusion "small and delicate").
- Corrected numbers from [Comfy-Org/Krea-2 discussion #6](https://huggingface.co/Comfy-Org/Krea-2/discussions/6): Kijai says not all Krea layers can use NVFP4 matmuls without bad quality loss; a mixed BF16/MXFP8/NVFP4 build with blocks 0-2 and 24-26 shielded ran 1.35 s/it vs 2.21 s/it for the Comfy NVFP4 file (not "vs 17 s"); int8 convrot LPIPS 0.066 / ImageReward -0.010 vs fp8 scaled 0.120 / -0.024, GPU not named.
- Contrary: [Nunchaku's Krea 2 Turbo W4A4 card](https://huggingface.co/ModelsLab/Krea-2-Turbo-W4A4-Nunchaku) reports LPIPS 0.276 / PSNR 17.7 at 1024 even with a rank-32 SVDQuant branch, and calibration error rising about 3x from first to last block. All fidelity numbers are inference-only.
- Show-stopper for the residual branch: it needs bf16 weights, and the user's Krea source is often a ComfyUI nvfp4 or int8 file.
- Stage 0 test: on 20 cached Krea batches at sigmas 0.2 / 0.5 / 0.8, measure final-velocity relative L2 error vs bf16 for all-nvfp4, nvfp4 + tail 4 int8_convrot, nvfp4 + head 3 / tail 3, and nvfp4 + rank-32 residual. Proceed only for a variant that cuts the error by at least 25%; then race it against all-nvfp4 (0.5 MP, rank 16, 1500 steps, 2 seeds), dropping anything over 15.5 GB or more than 10% slower, and keep it only if best-probe ArcFace beats all-nvfp4 by at least 0.01 on both seeds or reaches the same best at least 20% sooner. Skip LoftQ-style init of the trainable LoRA.

### Modulation targets (`embed` and the scale_shift tables)

**Skip-ish (one zero-code race). Expected gain: likeness probably marginal, plus or minus 1-2 probe points at best with a bleed cost on the class prompt. Confidence: low. Cost: none for the race; a ParamDelta wrapper and ComfyUI-named diff export if it ever wins.**

Full write-up in the theme section "Fewer steps to likeness: LoRA parameterization, initialization, timestep and noise sampling". Krea-specific points:

- Krea has no per-block adaLN MLP: one shared `time_mod_proj` plus a learned per-block `scale_shift_table` (raw Parameters, never wrapped by any Krea trainer; the [Krea 2 technical report](https://www.krea.ai/blog/krea-2-technical-report) says this light modulation replaces a per-block MLP that "can take 20-30% of parameters in many MMDiTs"). The shared projection is already trainable through `embed` (archs.py:350, 435); the tables are not (inject_lora wraps linear-like modules only, lora.py:478-482). The tables plus norm weights are about 1.4M fp32 parameters.
- The modulation input is the timestep only, so any delta is a global, prompt-independent shift that also lands on the trigger-less class prompt: isolation/DOP and region_prior penalise it, and without them it is concept bleed. The README's Automagic 4 measurement (README.md:673-676) found modulation projections carry 3x-135x the bulk's gradient power on Krea 2, so they need their own rate.
- `diff_b` is a dead end: every Krea block linear is bias-free. A `.diff` on the tables loads in ComfyUI only under its own key names (rename table in krea2/single_file.py:43-51), not under the PEFT `transformer.` keys Krea exports today.
- Zero-code race: `lora_targets` [q, k, v, o, ffn_up_gate, ffn_down, attn_gate, text_fusion] vs the same plus `embed`, automagic4 (or a 10x lower lr for the 'other' group), 1000-1500 steps, face + body probes every 100 steps, plus a trigger-less class-prompt sample set to watch for drift. Note that `embed` also contains txtmlp/tmlp/tproj, the worst-fit Turbo-transfer layers, and that a LoRA before the blocks disables the `lora_blocks` backward skip (lora.py:451-463).

### Caption-length variants and recaption of mismatched images

**Skip-ish (race only if captions are poor). Expected gain: unmeasured, likely within probe noise unless the existing captions are mismatched. Confidence: medium. Cost: near zero per step; a dataset.py change, an optional one-time captioner pass of 1-3 minutes for 20-40 images.**

Full write-up in the theme section "Text side". Krea-specific points:

- The Krea report says it trains mostly on long captions but also exposes the model to short and medium prompts. RulzTrainer already wraps captions in Krea's system prompt (krea2/loading.py:41). The scout's "RoPE index 0 makes length a strong signal" mechanism is speculation.
- [Fizgig KREA2.md](https://github.com/shootthesound/Fizgig/blob/master/docs/KREA2.md): stuck images in character runs were almost all caption/image mismatches ("the top flags were all caption/image mismatches"); it recaptions them with Qwen3-VL-4B between epochs. No numbers. RulzTrainer's loss_watch already flags stuck images (README.md:650-662) but has no captioner.
- Krea's ComfyUI text-encoder file includes `model.visual.*` (krea2/single_file.py:18), so a Qwen3-VL-4B captioner can be built from the file already on disk (about 9 GB bf16, a pre-training side job that unloads before training). Qwen 2.1's file has no vision tower.
- If raced, probe with both short and long prompts, or the comparison favours whichever length the probe uses. The identity-attribute filter has no measured support.

### Trailing checkpoint average (PMA)

**Skip-ish (one zero-code soup run). Expected gain: +0-1 probe point at the end of a constant-lr run at best. Confidence: low. Cost: a few minutes of probing.**

Full write-up in the theme section "Fewer steps to likeness: optimizers and learning-rate schedules". Krea-specific point: Krea's pretraining used PMA checkpoint merging instead of EMA, but the [technical report](https://www.krea.ai/blog/krea-2-technical-report) says PMA performs comparably to EMA and was chosen to avoid EMA's memory cost on full 12B weights, which a LoRA (EMA about 0.2 GB) does not have. lora_ema 0.98 is already on, and soup_files with steps 0 can already average a run's own step checkpoints. Test: train with save_every 100, then soup the last 3 / 5 / 10 checkpoints (and a WMA variant via scripts/soup.py `--weights 1,2,...,N`) and probe against `_best`; adopt only if the average beats `_best` by more than 2x likeness_noise_band on at least 4 of 6 runs without a strength boost.

### Self-reference conditioning (Ostris Krea 2 edit mode)

**Watch. Expected gain: trigger-only likeness unmeasured, possibly zero or negative; the real product is stronger likeness when a face photo is supplied at inference. Confidence: low. Cost: several days of krea2/model.py work, about +20% wall time at a 50% reference share.**

Full write-up in the theme section "Likeness from data, teachers, references and Turbo transfer". Krea-specific points:

- ai-toolkit trains Krea 2 with clean reference latents at t=0 (`model_kwargs.edit`), the reference also passed through the Qwen3-VL vision tower at a 384^2 pixel budget, and an optional asymmetric kv_cache mask; inference needs the ComfyUI-Krea2-Ostris-Edit node ([ComfyUI-Wiki news](https://comfyui-wiki.com/en/news/2026-07-04-ai-toolkit-krea2-edit-training)). The "as fast as a dedicated edit model" claim is about edit concepts. [yijunwang2/krea2-reid](https://huggingface.co/yijunwang2/krea2-reid) (rank 32, 25.6K steps) shows a generic reference-identity LoRA works on Krea 2, qualitative results only, with age/skin drift and look-alike confusion.
- Engineering cost on Krea: one shared `temb_mod` for all tokens (krea2/model.py:173-177) means reference tokens at t=0 need a second modulation applied per segment in every block; the 3-axis ids (krea2/model.py:286-291) have no reference grid; keep blocks must recalibrate with a reference present; alignment taps must exclude reference tokens. Verifier's estimate: about 9 s to 12-13 s per reference step.
- Cheaper first experiment with no model change: render varied-pose images of the subject with krea2-reid on Turbo from a dataset photo and feed them through the likeness-filtered self-training (selftrain.py). Also unmeasured.

### Not recommended for Krea

- **Guidance-aware (CFG-extrapolated) training target on Raw** (G48 member 84): refuted. The "90% likeness by 1750 steps" anecdote used ai-toolkit's differential guidance, which is nested inside the guidance-loss branch and amounts to a gradient rescale; Raw is undistilled and sampled at CFG 5.5, so baking guidance into a Raw LoRA and applying it to Turbo likely guides twice and oversaturates. The guidance loss belongs to Qwen 2.1 and the future distilled H3/Klein archs (theme section "Likeness from data, teachers, references and Turbo transfer").
- **Rank 32 as the Krea character default**: against the user's rank 8-16 practice and Fizgig's own r16 isolation recipe; race only.
- **3/4-view inter-ocular landmark check**: one anecdote and noisy keypoints at high yaw.
- **Per-bucket Krea shift**: already implemented (archs.py:441-449, krea2/sampler.py), see the recipe.

## Qwen-Image 2.x

This section covers Qwen-Image 2.1 and its Turbo, the only 2.x weights that are open, and RulzTrainer's port of them. Overall verdict: the port is already close to what the evidence supports. No item here is a big speed or likeness lever for character T2I LoRAs. The useful work is cheap. Measure what already exists (training adapter on/off, probe engine bias, `lora_blocks 10-14`), run the texture-fix decoder check, and add a licence metadata key. Edit-pair training is the one larger build, and it adds a new kind of LoRA rather than improving likeness. FlexAttention is a measured dead end.

### Status in RulzTrainer and model facts

Qwen-Image 2.1 is supported for T2I LoRAs (`QwenImage21Arch`, rulztrainer/archs.py:452; rulztrainer/qwen21/; README "Qwen-Image 2.1", README.md:1651-1708). The model is a 7B single-stream DiT released 2026-09-20: 32 blocks, one shared modulation, and a Qwen3-VL-8B text encoder read at its last layer before the final norm. Its RGBA VAE is 16x with 64 channels, and the DiT reads it unpatched (diffusers `patch_size=1`), so each 16x16 pixel tile is one token and 1024^2 is 4096 tokens. Attention is block-causal: text is causal over the text, image tokens see everything, and text is modulated at t=0 (`causal_condition`, qwen21/model.py:312-317). The Turbo (2026-10-09) is a full 8-step, CFG 1 checkpoint with a saved schedule. RulzTrainer loads Comfy-Org's int8 ConvRot files: DiT 7.3 GB, text encoder 9.4 GB. On the RTX 5060 Ti at int8, 0.5 MP, rank 8, it measures about 2.5 s/step with torch.compile and a 13.1 GB peak (README.md:1692-1693). Samples and probes use Comfy-Org's rank-178 Turbo LoRA at 8 steps, CFG 1 (config.py:43-61). Edit LoRAs are not supported (qwen21/model.py:25-26). Lineage: no Qwen-Image 2.2 or 2.5 exists in any source opened. Qwen-Image 2.0 (MMDiT, technical report [arXiv 2605.10730](https://arxiv.org/html/2605.10730v1)) has no open weights; secondary sources describe it as API-only. The 20B [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) and Edit-2511 are Apache-2.0, but they are MMDiT models RulzTrainer does not support. 2.1 is under the non-commercial Qwen Research License.

### Recommended recipe

- **Resolution 0.5 MP.** Fizgig calls it "the sweet spot": quicker than 1 MP and sharper than 0.25 MP ([Fizgig QWEN_IMAGE.md](https://github.com/shootthesound/Fizgig/blob/master/docs/QWEN_IMAGE.md)). The wizard already sets it (webui.py:3042-3045). SimpleTuner's 512 px-only 24/32 GB presets are a memory choice and say nothing about likeness.
- **Rank 8 at 2e-4** (rank 16 at half the rate for big or mixed sets), EMA 0.98. This is Fizgig's recipe as the README records it (README.md:1670-1672). No source here re-measures it.
- **`training_adapter: auto` (Fizgig v2), but re-measure it.** No public source has the 77 vs 56 likeness figure in config.py:55-57 and README.md:1663-1666. SimpleTuner's v1 assistant card reports no downstream benefit. Race Fizgig vs `""` vs Fizgig at strength 0.5 (see "Cross-referenced items" below).
- **`lora_blocks`: all blocks by default.** Race `10-14` as a candidate, not a default: it is tested only at 0.25 MP and rank 8, and Fizgig issue #265 reports speckle (see "Cross-referenced items").
- **Probes: keep the rank-178 Turbo LoRA at 8 steps, CFG 1.** Do not point `turbo_lora` at Lightning-4 or DMD2-2 yet. qwen21/sampler.py:31-40 would stretch Turbo's own sigmas over them instead of using each adapter's schedule.
- **`batch_size: 1`.** At batch 1, `key_mask` is None and the image pass runs mask-free. Do not build a FlexAttention path (G21 below).
- **VAE: keep the 512 px tiled decode.** It matches the full decode at 54 dB PSNR with +1.7 GB instead of +6.6 GB, and an untiled 1 MP decode runs out of memory (README.md:1684-1685). Add the texture-fix decoder only after the encoder check in G58.
- **Base: `int8_convrot`.** With 0.9% weight error there is nothing for a recovery adapter or a quantization-residual init to fix (see "Cross-referenced items").
- **Timesteps: keep `reference`.** The rejected idea that "texture collapse is a low-noise failure" adds nothing: every lever it names already exists (the `reference` timestep mode equals musubi's `qwen_shift`, and `timestep_shift` composes on top of it), and musubi's `discrete_flow_shift` was inert under `qwen_shift`.
- **Licence:** write a licence key into the LoRA metadata, and note that the attribution duty also covers models trained on Qwen 2.1 renders, which includes self-training (G61 below).

### Texture-fix VAE decoder for probes and previews (G58)

**Do it, but only the checks; keep it off by default. Expected gain: previews and self-training renders without the skin grid. Likeness effect unmeasured, expected zero to small. Confidence: medium (0.45). Cost: a 5-minute encoder check plus an offline re-score; a `vae_decoder` option is a small change.**

**What it is.** The stock Qwen 2.1 VAE puts a checkerboard/diamond grid on decoded images. It shows most on light skin at high resolution, and a plain encode-decode round trip reproduces it, so it comes from the autoencoder, not the DiT. madebyollin fine-tuned the decoder twice. The first release fine-tuned about 7.5M parameters: the two highest-resolution decoder stages plus the head, for about 5k steps at lr 3e-5. An update then fine-tuned about 61k scale/bias/RMSNorm-gain parameters for about 60k steps, with a large-activation penalty. The card says the weights were rescaled by powers of two (decoder /512, encoder /8), so the encoder file is not byte-identical, even though SimpleTuner says latents stay compatible.

**Evidence**
- [madebyollin/texture-fix-vae-for-qwen-image-2.1](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1): COCO val2017 rFID 3.38 -> 2.07, PSNR 33.29 -> 32.73, LPIPS 0.0357 -> 0.0384. DIV2K 1024 PSNR 32.85 -> 32.35. fp16 max activation about 3.5M -> about 1.3k, and fp16 NaNs 75 -> 0 on 185 stress inputs. Faces were not specifically tested. Qwen Research licence. The card lists 0.3B params but no file size; the scout's 0.68 GB is unverified.
- [Qwen-Image-2.1 discussion #12](https://huggingface.co/Qwen/Qwen-Image-2.1/discussions/12): the grid is most visible on light skin when zoomed on faces, and the round trip reproduces it. There is no official reply.
- [SimpleTuner QWEN_IMAGE_2-1 quickstart](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/QWEN_IMAGE_2-1.md): the texture-fix VAE is the default for validation, and the guide says "existing 2.1 training latents remain compatible". It warns that tiled decoding can cause colour seams. Its 2048^2 decode dropped from 26.87 to 15.28 GiB on an H200 by skipping temporal feature caches, not by tiling.
- The scout missed a sibling for Krea 2: [texture-fix-vae-for-qwen-image-1-and-krea-2](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-1-and-krea-2) (2026-10-20, Apache-2.0, encoder unchanged, rFID 4.75 -> 1.74 at -1.2 dB PSNR). Also covered by the [comfyui-wiki note](https://comfyui-wiki.com/ja/news/2026-09-24-texture-fix-vae-qwen-image-2-1).
- What does not transfer: no source measures faces or any face-recognition metric, and the rFID is general image quality at 256^2. ArcFace uses 112 px aligned crops cut from faces of about 150-350 px at 0.5-1 MP, so the resize filters out most of a high-frequency grid.

**RulzTrainer today.** All Qwen 2.1 decoding goes through qwen21/loading.py:217-259 (`_decode`, `decode_latents`, `decode_latents_grad`), and `grad_vae` returns the same frozen bf16 VAE. Previews and probes call `decode_latents` (trainer.py:2827), and the identity loss calls `arch.decode_latents_grad` (trainer.py:3493-3496). Latents come from `encode_images` on the same object (dataset.py:291), and the cache file names do not include the VAE (dataset.py:175). `get_vae` (qwen21/loading.py:300-306) already loads any ComfyUI-layout VAE through `vae_file` with `model_source: files` (config.py:184). So the texture-fix file can be dropped in today, but it then replaces the encoder too. Krea 2 has the same path (krea2/loading.py:284-290). The scout's "prefer untiled decode" is refuted by RulzTrainer's own measurement: the 512 px tiles stay.

**Fit and risks.** The VAE sits outside the DiT, so W8A8, nvfp4, compile, keep blocks, block swap, model_split and data_parallel are unaffected. The per-GPU VAE clones (trainer.py:436-441) simply clone the fixed decoder, and export is unchanged. Every racer decodes through the same decoder, so race rankings and relay picks should barely move; only absolute likeness levels shift. If ComfyUI keeps the stock VAE, previews will look slightly different from the user's final renders. fp16 decode becomes possible but adds little, since bf16 already works. The 2.1 fix is non-commercial, like the base model.

**Implementation path.**
1. Add `vae_decoder` to config.py: `""` (the VAE itself, the default), `texture_fix`, or a path. It applies to `qwen_image21` and `krea2`.
2. In `get_vae` (qwen21/loading.py, krea2/loading.py), keep the stock VAE for `encode_images`. When `vae_decoder` is set, load the fixed VAE through `single_file.load_vae_file` and attach its `decoder` and `post_quant_conv` to the stock object, so `decode_latents`, `decode_latents_grad`, `grad_vae` and `trainer._vae_on` need no new plumbing.
3. Add the file to the planned downloads and the WebUI model panel, with a licence note for the 2.1 variant.
4. Document it near README.md:1442 and 1684, and update the wiki.

**How to test.**
1. Encoder check (5 minutes): encode 20 dataset images with the stock VAE and with the full texture-fix file. If the cosine is above 0.9999, `vae_file` alone is enough and the decoder split is unnecessary.
2. Offline probe offset: re-score every saved checkpoint of an existing Qwen 2.1 run, and of the local krea2-AZav-rulz run, with both decoders on the same seeds. Report the mean and SD of face, body and combined likeness, and the Spearman rho of checkpoint order. Stop if rho > 0.95 and the offset is under 1 likeness point: the change is cosmetic.
3. Only if step 2 shows a shift: race two configs that differ only in `vae_decoder`, with id loss and selftrain on, 2 seeds, 2000 steps, 0.5 MP. Score both with the stock decoder, and accept only a win of at least 1.5 points on both seeds.

### Edit-pair LoRA training (G57)

**Prototype, only if edit LoRAs are wanted. Expected gain: a new LoRA type (before/after edits, possibly a face-reference-to-target identity edit). Nothing for character T2I LoRAs. Per-step cost at 0.5+0.5 MP is about 1.7-2x the 2.5 s T2I step (unmeasured). Confidence: medium (0.4). Cost: substantial (data layer, a Qwen3-VL vision-slot encode, N-segment attention, sampler).**

**What it is.** Qwen 2.1 handles both T2I and editing in one model. Condition images take the vision slots the VLM reserves in the text stream. Each condition block is bidirectional within itself and modulated at t=0, and the target image is appended at the end. Training an edit LoRA means building `[text, cond_1..n, target]`, giving the prefix the t=0 row, attending per segment, and computing the loss on target rows only.

**Evidence**
- [diffusers PR #14804](https://github.com/huggingface/diffusers/pull/14804) (merged 2026-09-18): the text plus condition-image prefix is modulated from t=0, and its K/V are cached after step one. The prefill is an exact multi-pass: image blocks bidirectional, text causal. Training collapses each `<|image_pad|>` run to one token. A KV clone fix cut peak memory at 2048^2 on an H100 (bf16, batch 1, 20 steps) from 64.5 to 56.5 GiB.
- [transformer_qwenimage21.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage21.py): `_qwenimage21_prefix_segments` returns `(start, end, is_text)` runs. `kv_cache` has `extract` and `cached` modes and raises unless `causal_condition` is on. `build_token_metadata` returns `image_ids` and `target_token_mask`.
- [Fizgig QWEN_IMAGE.md](https://github.com/shootthesound/Fizgig/blob/master/docs/QWEN_IMAGE.md): edit LoRAs need about 40 pairs (at least 20), photos of at least 1 MP with the same crop, and up to 9 extra references. Presets: rank 8 adaptive 2e-4..4e-4 for 12 epochs; "Strong" is rank 16 at 1e-4..2e-4. A colour grade (40 pairs, 0.59 MP, rank 16) was learned in 6 epochs, about 8 min on an RTX 5090. A 0.5 MP edit LoRA matched previews best at 768 in ComfyUI.
- [linoyts DMD2 card](https://huggingface.co/linoyts/Qwen-Image-2.1-DMD2-2steps): claims an attention-only rank-32 adapter preserves the input image better than one that also adapts the FFN. This is a claim on a distillation card, not an ablation.
- [musubi qwen_image.md](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/qwen_image.md): control latents are cached, and edit training needs extra memory. `--resize_control_to_official_size` is an inference option, not a training one, and the page does not cover 2.1.
- Misattribution: the line "a LoRA changes text-token KV, so prefix caching was disabled" comes from a commit in the guoriyue/VRL fork, not from the diffusers PR.
- What does not transfer: Fizgig's numbers are for a style/grade task on a 5090. No source measures identity-preserving edit LoRAs.

**RulzTrainer today.** The gap is documented at qwen21/model.py:25-26 and README.md:1655-1656, and FIBO has the same gap (README.md:1566, 1587). Two building blocks exist: the t=0 modulation row (qwen21/model.py:312-317) and a two-segment attention, text causal plus image over all (qwen21/model.py:117-130). dataset.py has no paired dataset type. The caption cache is text-only (qwen21/loading.py:159-186), and Comfy's Qwen 2.1 text encoder file has no vision tower (qwen21/single_file.py:10-13), so a vision-slot encode needs the full Qwen3-VL-8B weights.

**Fit and risks.**
- int8 ConvRot and NVFP4 quantize per token, so a longer sequence does not affect them.
- torch.compile: variable segment counts and lengths change shapes. Keep the per-segment SDPA calls outside the compiled block, or mark them dynamic.
- Memory: 0.5+0.5 MP roughly doubles the image tokens (2048 to 4096+), so keep-blocks calibration must rerun. int8 at 0.5+0.5 MP should fit on 16 GB with checkpointing; 1+1 MP needs block swap. Block swap, model_split and data_parallel are unchanged. Export uses the same Qwen 2.1 keys.
- Refuted member detail: the paper-only cached-prefix trick (a target-row-only LoRA plus a per-pair K/V cache) claimed 4x attention savings. The verifiers corrected this to about 1.5x on attention and 2x on linears at a 1:1 prefix:target ratio. Most of that gain comes from skipping the prefix backward, which a `no_grad` prefix recompute also gets, without the cache (about 2 GB per pair, about 80 GB for 40 pairs).
- A target-row-masked LoRA mismatches inference, because ComfyUI and diffusers apply the LoRA to every row.
- Probes need a reference-image input path in qwen21/sampler.py.

**Implementation path.**
1. dataset.py: a `pairs` dataset type (`originals/` and `edited/` with matching names, optional `photo_N` extras), Fizgig's same-shape check, and a shared bucket for reference and target.
2. qwen21/loading.encode_prompt: a reference-image path through the Qwen3-VL processor with vision slots, collapsing each `<|image_pad|>` run to one token. Cache the reference VAE latents in .rulz_cache.
3. qwen21/model.py:296-330 and 117-130: generalize to N segments following diffusers' `build_token_metadata` and `_qwenimage21_prefix_segments`, with the t=0 row for prefix rows and a per-image RoPE frame axis. Compute the loss on target rows in trainer.py.
4. qwen21/sampler.py: a reference image for samples and probes, with an optional inference-only `extract`/`cached` prefix.
5. Optional, behind a `lora_rows: target` flag: LoRALinear multiplies the LoRA branch by a target-row mask, and the prefix runs under `no_grad`, with no K/V cache on disk. Ship it only if the A/B below passes.

**How to test.**
1. Correctness: the multi-segment forward must be bit-close to diffusers' `QwenImage21Transformer2DModel` on the same reference+target input, as the existing port test requires (tests/test_qwen21.py).
2. Reproduce a Fizgig-style grade: 40 pairs, rank 16, 0.5 MP. Measure s/step on the 5060 Ti and the mean Lab delta of held-out photos to their edited targets at epochs 6-12.
3. Identity edit: race a full-row LoRA against a target-row LoRA. Metric: ArcFace between the reference face and the edited output on held-out references, plus DINO body, s/step and peak VRAM. Probe every 100 steps with patience 3. Drop target-row if its best ArcFace is more than 0.02 below full-row, or if ComfyUI outputs differ visibly from the previews. Keep it only if it is at least 1.3x faster at equal likeness.

### Block-causal attention: keep segmented SDPA, no FlexAttention (G21)

**Do it: add one code comment and skip FlexAttention. Expected gain: zero at batch 1, where `key_mask` is already None. At batch > 1 with mixed caption lengths, a few percent of attention time at most (unmeasured). Confidence: high (0.75). Cost: minutes for the comment; optional length bucketing is a small sampler change.**

**What it is.** RulzTrainer's two SDPA calls (text causal, image over all keys) are the "segmented" form that both diffusers and DiffSynth settled on. No dense L x S mask is ever built. A `[B,1,1,L]` bool key-padding mask appears only when a batch mixes caption lengths. That case costs the flash backend and some bias handling, not memory.

**Evidence**
- [DiffSynth-Studio PR #1724](https://github.com/modelscope/DiffSynth-Studio/pull/1724) (B200, bf16, 2048^2): segmented attention ran at 27-28 s/step vs flex at about 43 s/step on torch 2.8, and about 15 vs about 57 s/step on torch 2.12. On torch 2.8, flex gradients had relative error up to 3.4e4 from the second sequence length on; on 2.9.1-2.12 the error was about 4.9e-3. The LoRA delta-W difference between flex and segmented was 0.55, against 0.51 run-to-run. PR #1732 then made segmented attention the default.
- [DiffSynth issue #1706](https://github.com/modelscope/DiffSynth-Studio/issues/1706): Qwen 2.1 LoRAs collapsed to noise on torch 2.8 (only lr < 1e-7 was stable). The cause was flex's backward. Closed 2026-10-09.
- [diffusers PR #14804](https://github.com/huggingface/diffusers/pull/14804) and the [qwenimage21 docs](https://huggingface.co/docs/diffusers/main/en/api/models/qwenimage21_transformer2d): SDPA is the default and flex is opt-in. Uncompiled flex falls back to a dense fp32 score matrix and cannot render 2048^2. Compiled flex took 31.8 s vs 32.3 s over 20 steps (1.5%; 3.8% with two condition images) after a 37 s compile. All of these are inference numbers.
- [PyTorch attention.cpp](https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/transformers/attention.cpp): a bool mask becomes a same-shape bias and nothing L x S is allocated. The flash call takes no mask, but the cuDNN branch does take `attn_mask`.
- [PyTorch FlexAttention blog](https://pytorch.org/blog/flexattention/): 90% forward and 85% backward of FA2 on an A100. These are 2024 numbers on LLM-style masks and may be stale.

**Refuted members.**
- Finding 54's main proposal, a FlexAttention BlockMask path, is contradicted by DiffSynth's training benchmarks.
- Finding 99's claim that a mask forces SDPA off cuDNN is wrong. With cuDNN first on the 5060 Ti, a masked call probably stays on cuDNN; only the 4060 Ti ("torch" pick) drops from flash to mem-efficient.

**RulzTrainer today.**
- The two sdpa calls are at qwen21/model.py:130-131, and `key_mask` is None when no caption is padded (qwen21/model.py:320-325).
- Captions are right-padded only in `pad_embeds` (qwen21/loading.py:193). BucketBatcher groups by (w, h) only (dataset.py:385-414).
- `batch_size` defaults to 1 (config.py:100, 272), and data_parallel runs batch 1 per copy.
- Other padded calls: DOP class-prompt batches (trainer.py:2453) and Krea 2 (krea2/model.py:109).

**Fit and risks.**
- Flex would need inductor/triton on triton-windows, 128-token padding and a recompile for every new `text_len + H*W`, which multi-resolution buckets and resolution step-up keep producing.
- The alignment taps read q/k before attention (qwen21/model.py:126-128).
- Length bucketing touches no compute path. `caption_dropout` swaps in the empty-prompt cache inside `load_batch` (dataset.py:441), which changes the token count after batching, so dropout must be decided per batch.

**Implementation path.**
1. Add a comment above the two sdpa calls citing DiffSynth #1724/#1732 and diffusers #14804, so flex is not retried.
2. Optional, at `batch_size > 1`: sub-group each (w, h) bucket in `BucketBatcher.epoch` by cached text token count, read from the safetensors header as `trainer._longest_text_path` does, and decide `caption_dropout` once per batch.
3. Log the share of steps with a mask. This applies to every family through the shared batcher.

**How to test.**
1. Unit test: forward with `key_mask=None` against a padded batch with an all-True mask; require a difference within bf16 tolerance.
2. Microbenchmark on both cards: Qwen 2.1 int8, 0.5 MP, batch 4, mixed vs bucketed caption lengths, 50 steps after warm-up, measuring median s/step and peak VRAM. Keep bucketing only if it gains at least 3% on one card with no VRAM increase.
3. Sanity race at batch 4: 1500 steps, probes every 250.

### Licence metadata and the outputs clause (G61)

**Do it. Expected gain: none for training. It adds machine-readable licence metadata and documents an attribution duty the README misses. Confidence: high (0.85). Cost: negligible.**

**What it is.** The [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/raw/main/LICENSE) (2026-09-20) has these terms:
- 1(i) and 2(a): non-commercial means research or evaluation only. 2(b): commercial use needs a separate licence (model-business@notice.qwencloud.com).
- 4(b): any distributed model trained or improved with the Materials **or their outputs** must display "Built with Qwen" or "Improved using Qwen".
- 5(a): you own your derivatives. Section 8: PRC law, Hangzhou courts.
- Outputs appear only in the disclaimers.

The outputs clause reaches RulzTrainer through self-training (selftrain.py adds the model's own renders), the reward stage, and any Qwen-rendered dataset, even when the LoRA being trained is for Krea 2 or MOSS.

**Evidence**
- [QwenLM/Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1) and the [Turbo card](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo): Turbo's saved 8-step schedule is not overridden by `num_inference_steps` alone. It runs at CFG 1 with prefix KV caching, and the distillation method is not stated.
- [Discussions index](https://huggingface.co/Qwen/Qwen-Image-2.1/discussions): #52 (19 replies) and #40 (10 replies) ask about commercial use of outputs, with no team answer. #39, #24 and #9 ask for Apache-2.0, #7 reports licence-email bounces, and #6 is "License renders this model useless" (19 replies).
- Corrections: #37 only asks whether SFT is supported and reports no collapse. The scout's claim that third-party adapters (Fizgig, SimpleTuner, Lightning, Pruna, DMD2) carry the QRL is unverified. The line "Outputs are not part of the licensed Materials" comes from an unverified X post quoted in #40, not from the licence.

**RulzTrainer today.** The notice is already in README.md:1694-1695 and in the WebUI (webui.py:2963-2964, 7735). The saved LoRA metadata has no licence key: trainer.py:1425-1431, 1371-1372, 1951 and around 3360/3375 write only `ss_*`, `modelspec.architecture` ('qwen-image-2.1', archs.py:455), `rulztrainer_config` and `ss_blocks`.

**Fit and risks.** No compute path is touched. ComfyUI and diffusers ignore safetensors metadata. Make no claim about who owns outputs.

**Implementation path.**
1. Add a `license = ""` class attribute next to `modelspec` (archs.py:39). Set it to "Qwen Research License Agreement (non-commercial); Built with Qwen" on `QwenImage21Arch`, and optionally the Krea 2 Community License string on Krea.
2. Move the base metadata dict into one helper used by every save site, and have it add `modelspec.license` whenever the string is non-empty.
3. Mention the outputs clause in README.md:1694 and in the self-training help text.
4. Update the Qwen-Image 2.1 wiki page.

**How to test.** Save a tiny `qwen_image21` LoRA through the trainer's save path and read it back with `safe_open().metadata()`. Assert that both strings are present for Qwen, that the key is absent for MOSS and FIBO, and that the file still loads in `QwenImage21Pipeline.load_lora_weights` and in ComfyUI.

### Token-budgeted face crops (G59)

**Skip-ish (not Qwen-specific). Expected gain: at most about 1-2 face-probe points, mostly at 0.25 MP and for mid-size faces; unmeasured. Confidence: low (0.3). Cost: small (dataset.py plus two options).**

**What it is.** The idea is to size face crops by the latent tokens the face spans rather than by pixels. Qwen 2.1 takes 1 token per 16x16 px, so a face at 15% of frame width spans about 10 tokens at 1 MP, 6-7 at 0.5 MP, and under 5 at 512^2.

**Evidence**
- [diffusers QwenImage21Transformer2DModel docs](https://huggingface.co/docs/diffusers/main/en/api/models/qwenimage21_transformer2d): `patch_size=1`, in_channels 64, 32 layers, 32 heads x 128.
- [Qwen-Image-VAE-2.0 report, arXiv 2605.13565](https://arxiv.org/html/2605.13565v1): f16c64 scores FFHQ PSNR/SSIM 39.14/0.9541, against 38.14/0.9574 for FLUX f8c16. SSIM is actually lower, and FFHQ uses close-ups. Nothing in the report measures small faces in wide frames, so "a token-density problem, not an autoencoder one" is inference.
- [Fizgig QWEN_IMAGE.md](https://github.com/shootthesound/Fizgig/blob/master/docs/QWEN_IMAGE.md): 0.5 MP is the sweet spot. Two scout claims are not in that doc: "faster time-to-likeness on an RTX 5090", and Fast Identity Mode being tested only at 0.25 MP (that one is in the v6.8.3 release notes instead, see G55).
- [danmo-make PR #76](https://github.com/danmo-ai/danmo-make/pull/76): "a face is about five tokens tall" at 512. The PR's "16x16 token grid at 512" is an arithmetic error (it is 32x32). No likeness numbers.

**RulzTrainer today.** Two of the scout's three deltas already exist.
- 0.5 MP is the wizard default for Qwen (webui.py:3042-3045).
- `crop_augment face|face+bust` (config.py:377-385, dataset.py:202-261) crops 2.5x the face box (dataset.py:230-234), which gives about 17.6 face tokens at 704^2, 25.6 at 1 MP and 12.8 at 512. It skips the crop when the face covers more than 0.15 of the image area, and whenever the crop would need upscaling (dataset.py:235-237).
- `resolution_step_up` already reacts to probe plateaus (config.py:219-221).

**Fit and risks.** The never-upscale rule is the binding limit, and README's own numbers show upscaled crops hurt (65.5 vs 74.2). Every supported family is 16 px per token, so nothing here is Qwen-only. There is no compute interaction.

**Implementation path.**
1. In dataset.py `augment_items`, compute `face_tokens` from the face box, the bucket side and the arch's patch_px (16 for all current families).
2. Add `crop_min_face_tokens` (default 0 = off) as the skip test, and an optional `crop_face_tokens` scale target that still passes the never-upscale check.
3. Log face tokens per item in the loss watch, and add both keys to race.py's raceable list.

**How to test.** On a Qwen 2.1 int8 set at 0.5 MP mixing close-ups and full-body shots, race three configs: A (current face+bust), B (`crop_min_face_tokens 20` plus `crop_face_tokens 20`) and C (`crop_augment none`). Run 1500-2000 steps with probes every 100-200 steps and drop-lowest with patience 2. Score close-up and full-body probes separately, then repeat at 0.25 MP. Adopt B only if it gains at least 1.5 face points on full-body probes in two seeds, with body likeness within 1 point.

### Cross-referenced items: what is specific to Qwen 2.1

Full write-ups live in the theme sections. Only the Qwen-specific parts are here.

- **Training adapter (G50, de-distillation training adapters).** **Race it. Confidence: medium (0.65).** Qwen 2.1 is the only family with an `ARCH_TRAINING_ADAPTER` entry (config.py:58-61). The 77/56/76 likeness figures have no public source. [SimpleTuner v2](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v2) (rank 32 on q/k/v/out, 1000 updates, lr 1e-4, one H100) reports only qualitative results, [v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v3) says the downstream benefit is "to be evaluated", and v1 showed none. Race `training_adapter` Fizgig vs `""` vs Fizgig at strength 0.5, at 0.5 MP, rank 8, lr 2e-4. Compare peak ArcFace and the step to 50% of peak. A race shares one adapter across its configs (README.md:1244), so this needs separate runs, or per-config `training_adapter` keys in race.py. Building an own adapter (about 1000 turbo renders, 1-2 h, plus 1-2k steps at about 2.5 s, about 1 h on the 5060 Ti) should wait for that result.
- **Probe engines (G60, cheaper probe engines).** **Do it: the calibration part. Confidence: medium (0.5).** Probes already render on the side GPU, so cheaper engines save only about 0-5% wall clock on a 1000-3000 step run. The savings are larger in races, selftrain and the reward stage. Qwen-specific blockers:
  - qwen21/sampler.py:31-40 interpolates the 8 Turbo sigmas to any step count. [Lightning-4](https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning) therefore renders at about 1.0/0.93/0.85/0.41 instead of its dynamically shifted [1, .75, .5, .25], and [DMD2-2](https://huggingface.co/linoyts/Qwen-Image-2.1-DMD2-2steps)'s [1, .5] at shift 1 cannot be reached.
  - lora.py (around line 854) applies `.diff` only to norms, so [nynxz's 153 MB file](https://huggingface.co/nynxz/Qwen-Image-2.1) loses its exact img_in, proj_out and timestep-embedder diffs.

  Add per-adapter `sigmas`/`shift_mode`, a separate `probe_turbo_lora`, and `probe_calibrate` (first and last probe also rendered plain at 25 steps, CFG 1). Adopt a cheaper engine only if its Spearman rho to the plain reference is at least 0.9, it picks a best checkpoint within one checkpoint of the reference's, and its bias drift stays under 0.02 ArcFace cosine.
- **Identity blocks 10-14 (G55, identity block maps and block profiler).** **Race it. Confidence: medium (0.6).** [Fizgig v6.8.3](https://github.com/shootthesound/Fizgig/releases/tag/v6.8.3) puts identity in blocks 10-14 and reports about 1.5x per step, tested only at 0.25 MP and rank 8, with no likeness numbers. At 0.5 MP int8, expect roughly 1.6-1.9 s/step instead of 2.5 (unmeasured). [Issue #265](https://github.com/shootthesound/Fizgig/issues/265) (speckled skin in ComfyUI) is unresolved. `text_in` or `train_text_embedder` sends gradient back through blocks 0-9 and cancels the speedup. Race all blocks vs `10-14` vs `8-16` at 0.5 MP, rank 8, 2e-4, 2000 steps. Adopt only if the best combined score is within 2 points of all-blocks, the step is at least 1.25x faster, and close-ups at 25 steps show no speckle. Profiler groups for Qwen: 0-4, 5-9, 10-14, 15-19, 20-25, 26-31.
- **Reward stage (G25, reward-stage credit assignment).** **Race it. Confidence: medium (0.55).** [Qwen-Image-2.0-RL](https://arxiv.org/html/2606.27608v1) comes from the same lineage. Training on all 40 rollout steps reward-hacked; the fixes were a high-noise timestep subset, filtering prompts by in-group reward range, and an embedding-level face-identity scorer. Its Elo gains (+78 T2I, +93 edit) come from full fine-tuning and do not transfer as numbers. Qwen 2.1 int8 is the fastest family for racing the arms (random moves vs highest-sigma moves plus an SDE window, plus the TempFlow weight and a min range). Detaching the unconditional branch saves nothing here, because Qwen turbo probes run at CFG 1. Per-step Tweedie scoring is not recommended on 16 GB.
- **Quantization-residual init and recovery adapters (G7).** **Skip for Qwen.** Qwen 2.1 runs `int8_convrot` with about 0.9% weight error and the same white residual spectrum, so neither a data-free LoftQ init (refuted: on Krea 2 weights it cut the NVFP4 error only from 0.0949 to 0.0938 at rank 32) nor an accuracy-recovery adapter has anything to fix. [Qwen discussion #32](https://huggingface.co/Qwen/Qwen-Image-2.1/discussions/32) shows NF4 at 10.6 GB vs fp8 at 16.6 GB with equal latency (19.2 vs 19.3 s), with quality not measured.
- **Captions (G35, caption policy and caption-length mixture).** **Race it (multi-variant loader only). Confidence: medium (0.4).** RulzTrainer's Qwen 2.1 encoder uses the system prompt "Comprehend and analyze the provided prompt."; the "describe colour, shape, size, texture" system prompt the scout cited belongs to Qwen-Image 1. Comfy's Qwen 2.1 text-encoder file has no vision tower (qwen21/single_file.py:10-13), so a captioner has to build Qwen3-VL-4B from Krea's file, which carries `model.visual.*` (krea2/single_file.py:18), or download one. Qwen 2.1 int8 is the suggested first race platform: single caption vs `caption_mix` long/medium/short vs concise context-only captions, probed with both short and long prompts.

## Flux 3 Klein

This section covers FLUX 3 Image Klein (Black Forest Labs, released 15 Oct 2026): what the model is, the settings the published trainers agree on, and a concrete plan for adding it as a fifth RulzTrainer family. Verdict: it is worth adding. It is a 5.6B model with the same 4096 tokens per MP as Krea 2, its text encoder and VAE are files RulzTrainer already loads, and a rank-16 LoRA at 1 MP has been measured at 9.1 GB in INT8. The port is mostly reuse. Three settings are correctness rules, not tuning: keep the padded 80-token text window, use the 6.93 shift, and train the Base checkpoint. Everything about likeness on this model is still unmeasured, by anyone.

### Status in RulzTrainer and model facts

**Status:** not supported. `ARCH_NAMES` holds only `moss_image, fibo2, krea2, qwen_image21` (rulztrainer/archs.py:29) and README.md:726 says "MiniMax H3 and FLUX Klein are trained in ai-toolkit, not here". Adding the family reverses that line, and the wiki needs the same update (per-feature rule).

**Model facts** (BFL dit.py, diffusers transformer_flux3.py and ComfyUI flux3/model.py agree; confirmed by the G62 and G63 verifiers). Two DiT checkpoints of 5.6B each (Hub metadata says 6B). `flux-3-image-klein-base` is undistilled, runs 50 steps with true CFG, and is the one meant for LoRA training. `flux-3-image-klein` is a step-distilled "flow map": 16 steps, no CFG, about 6x faster, with an extra `ref_time_in` target-timestep embedding. Hidden 3072, 24 heads x 128, MLP ratio 3 (SwiGLU `mlp_in` 3072->18432, `mlp_out` 9216->3072). Attention and MLP run in parallel from one modulated input and share one gate. LayerNorm without affine plus adaLN (shift, scale, gate) from a per-mode `Linear(3072->9216)`. RMS QK-norm, then 4-axis RoPE (t, h, w, l) at 32 dims each. Stage 1 is 5 blocks per stream with separate weights (`content_mode_blocks.image`, `content_mode_blocks.image_cond`, `txt_mode_blocks`). Stage 2 is 28 shared `single_blocks` over [text | image | reference]. Each token gets its own timestep: image tokens at t, text and reference tokens always at t=0. There is no guidance embedding and no attention mask anywhere. Text encoder: Qwen3-VL-4B-Instruct (vision tower unused), hidden states of layers 4, 8, ..., 32 (8 of 36) concatenated to 20,480 per token, then `txt_in Linear(20480->3072)`. VAE: the FLUX.2 AE (32 ch at /8, 2x2-packed to 128 ch at /16, normalized with a non-affine BatchNorm2d(128, eps 1e-4)). Licence: FLUX Kommunity License v1.0 (non-commercial free, commercial outputs under US$5M revenue). The HF repos are gated, but the Comfy repack lists its files openly. BFL's internal eval puts it +91 Elo T2I and +70 editing above FLUX.2 klein 9B with 38% fewer parameters. Nobody has verified that independently.

### Recommended recipe

These are the settings the evidence supports for a character LoRA on the user's hardware, once the family exists. None of them has been tuned for likeness by any source.

- **Train Base, not the distilled checkpoint.** BFL recommends Base for LoRA training. ai-toolkit refuses `ref_time_in` keys. NeMo, diffusers and Fizgig train Base. ([BFL README](https://github.com/black-forest-labs/flux-3-image-klein), [ai-toolkit flux3/src/transformer.py](https://raw.githubusercontent.com/ostris/ai-toolkit/main/extensions_built_in/diffusion_models/flux3/src/transformer.py)). SimpleTuner's photo-aesthetics runs did fine-tune the distilled model with an assistant LoRA, so this is a strong preference, not a hard rule.
- **lr 1e-4, rank 16, batch 1, constant or cosine.** diffusers/HF blog: rank 16 / alpha 16, lr 1e-4 constant, 1500 steps, 1024 px, batch 1 ([HF blog](https://huggingface.co/blog/flux-3)). NeMo: AdamW lr 1e-4, cosine, 500 warmup, wd 1e-4, rank 4 ([NeMo yaml](https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/examples/diffusion/finetune/flux3_t2i_flow_lora.yaml)). SimpleTuner: rank 32, lr 1e-4, REPA on ([FLUX3_IMAGE.md](https://raw.githubusercontent.com/bghira/SimpleTuner/main/documentation/quickstart/FLUX3_IMAGE.md)). BFL docs: rank 32 / alpha 32, AdamW 1e-4, 4,000 steps, about 4.5 h for 20 images at ~1 MP ([BFL docs](https://docs.bfl.ml/flux_3/flux3_image_klein)). The verifier notes this last one is a style recipe. The HF blog swept ranks 8/16/32 and Prodigy without naming a winner. Start at rank 16, lr 1e-4, adamw8bit (Fizgig's preset optimizer).
- **Timesteps: `timestep_sampling: uniform`, `timestep_shift: 6.93`**, i.e. sigma = 6.93u / (1 + 5.93u). This is the density diffusers (`FlowMatchEulerDiscreteScheduler(shift=6.93, use_dynamic_shifting=False)`, weighting 'none'), NeMo and ai-toolkit all train with. It has no resolution dependence ([diffusers train script](https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_flux3_klein.py)). It already works today through trainer.py:240-241. See the shift subsection.
- **Text: cache the full padded 80-multiple window, pads included.** There is no text mask, and the DiT attends to the pad tokens' hidden states ([ai-toolkit text_encoder.py](https://raw.githubusercontent.com/ostris/ai-toolkit/main/extensions_built_in/diffusion_models/flux3/src/text_encoder.py): "do not strip padding").
- **Targets: q, k, v, o, ffn_up_gate, ffn_down on the 5 image-mode and 28 single blocks, plus the 5 text-mode blocks.** NeMo and ai-toolkit both train `txt_mode_blocks`. Keep the reference stream (`image_cond`) off for text-to-image: NeMo excludes it explicitly.
- **Quant: int8_convrot on the 4060 Ti, int8_convrot or nvfp4 on the 5060 Ti; no block swap needed.** Fizgig measured rank 16 at 1 MP on an RTX 5090: bf16 12.7 GB, INT8 9.1 GB, NF4 6.2 GB peak; full fine-tune 14.0 GB ([Fizgig FLUX3.md](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/FLUX3.md)). Train from the Comfy bf16 or int8-convrot file. Do not train on BFL's fp8r/nvfp4 variants as they ship: NeMo marks them inference-only and not validated for training. See the file-format notes below.
- **Samples and probes: Base at 50 steps, CFG 4.0** (BFL README/pipeline; the HF card, diffusers modular, ai-toolkit and SGLang use 3.5). The negative is the encoded empty prompt. RulzTrainer's CFG convention (uncond + cfg * (cond - uncond)) is the same as BFL's `v_u + g (v_c - v_u)`, so use the number as it is.
- **Captions:** BFL says short prompts give generic output. At minimum, upsample the probe and sample prompts. Long training captions need a race (see below).
- **torch.compile on:** Fizgig measured "Compile Blocks" at about 1.2x.

### Adding it to RulzTrainer

**Verdict:** Prototype. Expected gain: model access, not speed. It is the strongest small open model by BFL's own unverified Elo, at roughly half Krea 2's weights with the same tokens per MP. Confidence: high (0.85 G62, 0.75 P26). Cost: about 5-7 working days for text-to-image v1 (my estimate, not from a source).

**Architecture facts that decide the design** (from the model facts above):
- 33 blocks on the image path (5 image-mode + 28 single), 5 text-mode blocks on the text path, and 5 reference-mode blocks that never run in text-to-image training.
- Text and reference tokens are modulated at t=0. The text stream's stage-1 output is therefore a pure function of the caption. ComfyUI's fast path exploits this at inference.
- With no attention mask and no padding mask, every block at batch 1 has a fixed shape for a given (image bucket, text window).
- The text window grows in steps of 80 tokens: 80, 160, 240 and so on, up to 4096 (BFL), 8192 (diffusers modular, ai-toolkit) or 512 (default in the diffusers LoRA script).
- Image RoPE ids are `(0, row/2, col/2, 0)` (image_rope_scale 0.5), text ids use l = index, and reference k sits at l = -(k+1).

**Text encoder: reuse Krea 2's, change four things.** The Comfy repack ships `text_encoders/qwen3vl_4b_{bf16 8.9 GB, int8_convrot 4.9 GB}`. That is the same file name and model Krea 2 and FIBO already load (config.py COMFY_TEXT_ENCODER_FILES, `krea2/single_file.py:221 load_text_encoder_file`, `krea2/qwen3vl_4b_text_config.json`). `krea2/loading.py:142-154 encode_prompt` already runs the encoder with explicit position ids and stacks `hidden_states` at `SELECT_LAYERS` (krea2/loading.py:47). A `flux3/loading.py` copies that path and changes:
1. `SELECT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)`. Return `[L80, 8, 2560]`. The model's `txt_in` flattens this to 20,480.
2. Use the chat template as a bare user turn plus `add_generation_prompt`, with no system prompt. Drop Krea's `PROMPT_PREFIX` / `PREFIX_TOKENS` stripping (krea2/loading.py:41-44).
3. Right-pad to a multiple of 80 with pad id 151643, give pads position id 1 (BFL: `cumsum(mask) - 1`, pads masked_fill to 1), cap at 4096, and keep every pad row in the cache.
4. `prompt_tokens` returns the real-token span, so trigger lookup and attention localization still find the trigger.

**VAE: reuse unchanged.** Flux 3 uses `vae/flux2-vae.safetensors` (0.3 GB). FIBO already maps to that same file (config.py:87-88 COMFY_VAE_FILES['fibo2']). `moss/single_file.py:221 load_vae_file` reads both the diffusers and the original BFL key layout and checks for the `bn.*` latent norm. The BatchNorm normalization is `moss/loading.py:161-176`, and the base `Arch.encode_images` / `decode_latents` (archs.py:71-89) already call it, so the new Arch inherits them. Decode with the full flux2-vae, not MOSS's "small" decoder (README.md:335). The encoder is identical across Flux2 VAEs (README.md:335) and `latent_tag` defaults to `""` (archs.py:50), so MOSS and FIBO latent caches can be reused for the same images and buckets.

**What else is reused, and from where:**
- **Arch contract and registry:** subclass `Arch` (archs.py:36-138) as `Flux3KleinArch`, register `"flux3_klein"` in `_ARCHS` (archs.py:580) and `ARCH_NAMES`. Add detection to `arch_from_transformer_file` (archs.py:595) on the key `early_stream_modulations.image.lin.weight`, and to `detect_arch`'s class map with `Flux3Transformer2DModel`.
- **Krea2Arch as the template** (archs.py:340-449): the `GROUPS` / `EMBED` / extras pattern in `lora_module_names`, `text_prefixes` for the text-side stage, and `compile_blocks`. Krea's text-fusion stage is the precedent for running blocks outside the main list every step through `self._run` (krea2/model.py:278-283, 317) while still being a LoRA target group.
- **QwenImage21Arch** (archs.py:452-577): the `text_in` / `embed` extras split, `extra_lora_converter` (archs.py:524-541) as the pattern for loading LoRAs saved in a fused layout, and the `is_distilled` / `uses_distilled_schedule` pattern (qwen21/sampler.py:46). The distilled flow-map checkpoint as `sample_model_path` needs the same flag.
- **Block machinery:** `checkpointing.set_kept_blocks` and `trained_block_indices` (checkpointing.py:22-42), block swap (`model.layers[i]`, blockswap.py:55,74) and model_split all iterate `model.layers`. Expose `layers` as a property, as Krea does (krea2/model.py:265-268): `content_mode_blocks.image` (indices 0-4) + `single_blocks` (5-32). The 5 image-mode blocks carry 5/33 of the image-token work, so they belong in the list for swap, keep blocks and compile. `txt_mode_blocks` run outside, like `text_fusion`. Skip the `image_cond` stage entirely when there are no references; do not copy ai-toolkit's 2-token dummy pass.
- **Single-file loading and quant:** `moss/single_file.py` `read_state` (lines 25-56), `strip_prefix` and `install_state` (lines 64-108), and fused-qkv splitting as in `unfuse_qkv` (moss/single_file.py:134-140, using `QuantWeight.split_rows`, quant.py:223). Krea's `load_transformer_file` (krea2/single_file.py:131-166) is the template for quant `auto` resolving by memory and for the `is_distilled` flag.
- **Sampler:** `sampler.moss_sigmas(steps, 6.93)` nearly reproduces BFL's `alpha t / (1 + (alpha-1) t)` grid (G63 verifier). The base sampler is MOSS-like Euler with CFG but without MOSS's skip-layer uncond.
- **Caption dropout:** already encodes the real empty prompt or the trigger alone (dataset.py:265-280), which gets its own 80-token window. That is correct for Flux 3 as long as batches do not mix windows.
- **Training adapter, likeness probes, races, export:** unchanged machinery (lora.py live frozen branch, likeness.py, race.py, `lora_format = "comfy"`).

**What is new:**
- `rulztrainer/flux3/model.py`: port BFL dit.py with BFL module names (`content_mode_blocks.image.N.q_proj|k_proj|v_proj|attn_out|mlp_in|mlp_out`, `txt_mode_blocks.N.*`, `single_blocks.N.*`, `emb_in.image`, `txt_in`, `time_in`, `early_stream_modulations.*`, `final_layer.image`). Also: the parallel attention+SwiGLU block with one gate, 4-axis RoPE ids, per-token modulation (a second `time_in` embedding at t=0 for text, computed once per forward), and slicing text off before `final_layer.image`. The forward keeps the trainer's call `model(latents, sigma*1000, embeds, mask)` (trainer.py:975, 1596, 2460) and ignores the mask.
- `rulztrainer/flux3/loading.py`: the encoder changes above, `pad_embeds` that returns the stored window unchanged with an all-true mask, and the VAE glue.
- `rulztrainer/flux3/single_file.py`: load the Comfy bf16 / int8-convrot / nvfp4 files, strip `model.diffusion_model.` / `diffusion_model.`, split fused `qkv` (torch.chunk dim 0, or `split_rows` for quantized rows), drop `.input_scale` keys (read_state does not skip them today, so they would show up as unexpected keys), and keep `emb_in*`, `txt_in`, `time_in*`, `*modulations*` and `final_layer*` out of quantization (ai-toolkit's exclusion list, the same idea as `DIT_QUANT_FILTER`). Refuse `ref_time_in.*` for training; accept it for `sample_model_path`.
- `rulztrainer/flux3/sampler.py`: base Euler with CFG against the encoded empty prompt, plus a flow-map sampler for the distilled checkpoint (see "Distilled checkpoint as probe engine").
- `Flux3KleinArch`: `n_blocks = 33`, `lora_format = "comfy"`, `GROUPS = {"q": ("q_proj",), "k": ("k_proj",), "v": ("v_proj",), "o": ("attn_out",), "ffn_up_gate": ("mlp_in",), "ffn_down": ("mlp_out",)}`, extras `text_blocks` (txt_mode_blocks, on by default), `embed` (emb_in.image, txt_in, final_layer.image), `adaln` (modulations) and `ref` (image_cond blocks + emb_in.image_cond, off). Also `reference_sigmas = 6.93u/(1+5.93u)`, `sample_sigmas = moss_sigmas(steps, 6.93)`, `empty_prompt = ""`, `default_negative = ""`, and `latent_tag = ""`.
- config.py: `ARCH_DEFAULT_TARGETS["flux3_klein"] = _CORE_TARGETS + ("text_blocks",)`, `ARCH_FULL_SAMPLE["flux3_klein"] = (50, 4.0)`, `COMFY_REPOS["flux3_klein"] = "black-forest-labs/flux-3-image-klein-comfy"`, `COMFY_DIFFUSION_FILES` for base and distilled, `COMFY_DIFFUSION_QUANTS = ("int8_convrot", "nvfp4", "bf16")`, `COMFY_TEXT_ENCODER_FILES` = the krea2 entry, and `COMFY_VAE_FILES` = the fibo2 entry. Plus a wizard recipe, README and wiki.

**File formats and key names:**
- **Comfy repack** (`black-forest-labs/flux-3-image-klein-comfy`): `diffusion_models/flux3-image-klein{,-base}-{bf16 11.1 GB, int8-convrot 5.8 GB, nvfp4 5.3 GB}.safetensors`. The int8-convrot file is the same ConvRot format RulzTrainer's int8 path reads, but with fused `.qkv.weight`, which ComfyUI and ai-toolkit split into q/k/v on load.
- **Vendor repo `variants/fp8r` (rowwise FP8) and `variants/nvfp4`:** inference-only per NeMo, and their key layout was not readable (gated). Do not add an FP8 compute path for them. The FP8 compute-path group (G4) was refuted on the 4060 Ti: RulzTrainer's INT8 path is as fast or faster, and row-wise `_scaled_mm` is unsupported on sm_120. Dequantize them and requantize into RulzTrainer's own int8_convrot / nvfp4 instead. Most of this already exists: `read_state` dequantizes ComfyUI scaled fp8 with a per-tensor or per-row `weight_scale` to a plain tensor (moss/single_file.py:25-56), and `install_state` re-quantizes to the requested format, converting a QuantWeight whose format differs by dequantize-then-quantize (moss/single_file.py:84-88). The work left is a key adapter if BFL's fp8r names differ from `weight` + `weight_scale`, plus dropping `.input_scale`. The simpler route is to use the Comfy bf16 or int8-convrot file and never touch the vendor variants.
- **LoRA export:** keep BFL names so `lora_format: comfy` writes `diffusion_model.single_blocks.N.q_proj.lora_down/up`. ComfyUI's lora.py maps `diffusion_model.*` and `lora_unet_*`, and ai-toolkit saves the same names. Do not emit diffusers PEFT names (`transformer.single_transformer_blocks.N.attn.to_q`, `ff.linear_in`, ...): ComfyUI's lora_convert.py has no diffusers-to-Comfy converter for Flux 3 ([lora_convert.py](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/lora_convert.py)). Fizgig notes saved LoRAs need ComfyUI >= 2026-10-15.

**Fit with the engine:**
- **W8A8 / NVFP4:** QuantLinear (quant.py:966) wraps each Linear, so the blocks quantize like Krea's. P26's "one activation quant for both attention and MLP" needs a new fused multi-output QuantLinear. Defer it: the quantizer is a minor share of a 4096-token step.
- **NVFP4 on the 4060 Ti** dequantizes on the fly (quant.py:13-15), so use int8_convrot there.
- **Keep blocks / block swap:** at about 6-7 GB of int8 weights, Klein should fit 16 GB at 0.5-1 MP without block swap, unlike Krea 2. Fizgig's 9.1 GB INT8 at 1 MP supports this. Recalibrate `checkpoint_keep_blocks` for the 33-block list.
- **torch.compile:** the text window varies in 80-token steps, which multiplies the dynamo shape set with image buckets (compiling.py:59 warns about the 8-shape fallback). Cap the window set, bucket captions by window, or mark the text length dynamic.
- **data_parallel / model_split:** nothing new. These only need the ordered `layers` list. No NCCL.
- **Sigma grid:** the 6.93 shift snapped to the 1000 grid puts about 87% of training steps above sigma 0.5, so clean_share and the identity timestep modes move the density much further from native than on Krea. Race them; do not copy Krea defaults.
- **Probes:** Base probes cost 50 steps x 2 passes (CFG), 100 DiT passes against Krea Turbo's 8. They run on the side GPU, but they will be slower than Krea probes until the distilled engine passes its transfer gate.

**Speed and memory on the user's cards (unmeasured):** the verifiers estimate about 4-5 s/step in nvfp4 at 1 MP on the 5060 Ti and about 2.5-3 s at 0.5 MP, against Krea 2's ~9 s/step. A second estimate puts it at 3-5 s/step at 0.5 MP int8 on the 4060 Ti (MOSS-like: 5.6B, 33 image-path blocks vs MOSS 5B / 42). Long upsampler-style captions add text tokens to all 28 joint blocks. Memory references: HF blog H100 after load bf16 18.80 GiB, fp8r 14.35, nvfp4 13.4 (inference). Fizgig 5090 LoRA training numbers are in the recipe above.

**Risks:**
- The model expects long upsampler captions, which lengthen the joint sequence.
- Base probes are slow.
- No identity block map exists (Fizgig: "not charted").
- The licence is Kommunity, and the HF repos are gated.
- BFL's own `generate()` defaults to alpha 16.0 while every downstream uses 6.93. Check this once.
- Reference-image training (multi-ref prompts, sequence-axis RoPE, bbox JSON) is new dataset semantics. Keep it out of v1.

**Implementation order:**
1. `flux3/model.py` plus a parity test against diffusers `Flux3Transformer2DModel` on fixed inputs. Max abs error within bf16 noise. Check the int8-convrot load against bf16 at cosine > 0.999 per block output.
2. `flux3/loading.py` encoder plus a parity test: 3 prompts against ComfyUI or diffusers encodings, cosine > 0.9999 including the pad rows.
3. `flux3/single_file.py`, Arch, config tables, base sampler. Render one seed at alpha 6.93 and at 16 and confirm 6.93 matches reference outputs.
4. Export: reload a saved LoRA in ComfyUI (>= 2026-10-15) and confirm no missing keys.
5. Wizard, README (replace README.md:726), wiki.
6. Then the first races below, then the flow-map probe engine.

**How to test (first races, once it loads):** use the same 20-40 image character set at 0.5 MP, rank 16, lr 1e-4, adamw8bit, 2000 steps, probes every 100 steps.
- Race Flux 3 Base int8 against Krea 2 Raw nvfp4 and Qwen 2.1 int8. Metric: ArcFace face plus DINO body probe likeness, steps to reach the Krea run's best probe score, and s/step.
- Within Flux 3, race `lora_blocks` over {all 0-32, single only 5-32, image-mode + single 0-13 = 0-18, single 14-27 = 19-32}, with and without `text_blocks`. With `lora_blocks` 5-32, `trained_block_indices` skips the backward pass through the image-mode blocks.
- Stop with the existing race patience (drop a racer after 3 probes without improving). Declare a winner only if its best face score beats the next one by more than the probe's seed-to-seed spread.

### Pad-attended text window (keep pads, never compact)

**Do it (part of the port). Expected gain: correctness, not speed; avoids an unmeasured silent likeness loss. Confidence: high (0.8). Cost: included in the loader work.**

**What it is:** Flux 3 right-pads every prompt to a multiple of 80 tokens with pad id 151643, and the DiT attends to every pad token's Qwen3-VL hidden state. There is no text mask. The pad states are not zeros: they depend on the caption, because the encoder is causal. They are part of the context the model was trained on, so stripping them changes the model's function.

**Evidence:**
- [BFL text_encoder.py](https://raw.githubusercontent.com/black-forest-labs/flux-3-image-klein/main/src/flux3/text_encoder.py): OUTPUT_LAYERS (4..32), PAD_MULTIPLE 80, MAX_LENGTH 4096, right padding, pads at position 1, user turn plus add_generation_prompt, and no mask returned to the DiT.
- [ai-toolkit text_encoder.py](https://raw.githubusercontent.com/ostris/ai-toolkit/main/extensions_built_in/diffusion_models/flux3/src/text_encoder.py): "trained with the pad tokens in-context", "do not strip padding". [ai-toolkit flux3.py](https://raw.githubusercontent.com/ostris/ai-toolkit/main/extensions_built_in/diffusion_models/flux3/flux3.py): "attends to real pad-token states, not zeros", caches (L, 20480).
- [diffusers train script](https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_flux3_klein.py): TEXT_PAD_MULTIPLE 80, caption dropout uses the empty prompt encoded at the same token window.
- [ComfyUI text_encoders/flux.py](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/text_encoders/flux.py): pads to a multiple of 80 with 151643, at RoPE position 1.
- This is model-intrinsic, so it transfers fully to batch-1 LoRA training.

**RulzTrainer today:** Krea's loader compacts to real tokens and zero-pads with a mask (krea2/loading.py:175-186). The krea2/loading.py:10-15 docstring explains the pipeline layout, and Krea's forward masks the padding (krea2/model.py:312-316), which is what justifies compacting there. Qwen does the same (qwen21/loading.py:194). Caption dropout already loads a really encoded empty-or-trigger prompt (dataset.py:265-280), not zeros, so that half of the scout's claim is already in place.

**Fit and risks:** the cache costs 40 KB per token ([L, 8, 2560] bf16). A 240-480 token window is 9.8-19.7 MB per caption, about 0.6 GB for 30 images: fine on disk and in RAM. At batch 1 no cross-sample padding arises. At batch > 1 with different windows, rows cannot be zero-padded. Either bucket by window or re-encode at the longer window, as diffusers and ai-toolkit do. Do not build the scout's "shared pad block" fallback: pad states depend on the preceding tokens, so they cannot be shared across captions. Never mix caches from other families into this one.

**Implementation path:** `flux3/loading.encode_prompt` returns the full window. `Flux3KleinArch.pad_embeds` returns it unchanged with an all-true mask and raises (or groups by window) when a batch mixes windows. Use the encoded `""` both for caption dropout and as the CFG negative in samples and probes.

**How to test:** after the parity check, race the full-window cache against a deliberately compacted cache. Use about 20 images, rank 16, 1500 steps, Base nvfp4 or int8, batch 1, with ArcFace plus DINO probes every 250 steps on upsampled probe prompts. Expect the compacted run to be worse. Keep the padded window even if it is not, because it matches inference.

### Shift 6.93 as the reference timestep recipe

**Do it. Expected gain: correctness; the likeness effect of getting it wrong is unmeasured. Confidence: high (0.8). Cost: about 5 lines plus one check render.**

**What it is:** every Flux 3 implementation shifts with t' = alpha t / (1 + (alpha - 1) t), alpha = 6.93. Training (diffusers, NeMo) samples u ~ U(0,1) and maps it through the same function, so the training density matches the sampler's. With 50 sampling steps, 44 are above sigma 0.5. In training, P(sigma < 0.5) = 1/7.93 = 12.6%. There is no resolution-dependent shift.

**Evidence:**
- [HF blog](https://huggingface.co/blog/flux-3): "time shift of 6.93, so 44 of 50 steps occur above sigma = 0.5".
- [diffusers modular before_denoise.py](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/modular_pipelines/flux3/before_denoise.py): constant 6.93, no mu.
- [ComfyUI supported_models.py](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/supported_models.py): Flux3 shift 1.93586, "# mu = ln(6.93), the reference timestep shift".
- [ai-toolkit pipeline.py](https://raw.githubusercontent.com/ostris/ai-toolkit/main/extensions_built_in/diffusion_models/flux3/src/pipeline.py): DEFAULT_ALPHA 6.93.
- [BFL sampling.py](https://raw.githubusercontent.com/black-forest-labs/flux-3-image-klein/main/src/flux3/sampling.py): the formula.
- [BFL pipeline.py](https://raw.githubusercontent.com/black-forest-labs/flux-3-image-klein/main/src/flux3/pipeline.py): `generate(alpha=16.0)` default. This discrepancy is real and unexplained; every downstream implementation uses 6.93.
- Minor contrary point: a NeMo comment says dynamic shifting applies per resolution at inference, while diffusers modular says the shift is fixed.
- Correction carried from G65: ComfyUI's FLUX.2 Klein value 2.02 is also a mu (shift factor exp(2.02) = 7.56). Flux 3's shift is therefore slightly lower than FLUX.2's, not "3.4x higher". Do not carry diffusion-pipe's FLUX.2 "shift 3" over either.

**RulzTrainer today:** `timestep_shift` already applies exactly this form after sampling (trainer.py:240-241) and snaps to the 1000 grid. `timestep_sampling: uniform` + `timestep_shift: 6.93` reproduces the recipe now. `Arch.reference_sigmas` (archs.py:136) is the hook for a named family recipe.

**Fit and risks:** no compute cost. The density leaves only about 12.6% of steps below sigma 0.5, so clean_share and the identity mode (mean -0.5) move this model further from its native density than they move Krea's. That makes them race candidates, not defaults.

**Implementation path:** `Flux3KleinArch.reference_sigmas`: `u = rand(n); return 6.93*u / (1 + 5.93*u)`. `sample_sigmas = moss_sigmas(steps, 6.93)`. On the first run, add a one-off render of the same seed at alpha 6.93 and 16 to compare against the HF Space before trusting probe baselines.

**How to test:** a 3-arm race with 3 seeds each and the existing race (drop lowest at each probe, patience 2):
- `timestep_sampling: reference` (6.93 shifted-uniform)
- `logit_normal` + `timestep_shift: 3`
- the identity mode

Metric: ArcFace face plus DINO body probe likeness.

### Captions in the prompt-upsampler dialect

**Race it. Expected gain: unmeasured for likeness; upsampled probe prompts make probe scores representative of real use. Confidence: medium (member weakened). Cost: probe-prompt upsampling ~0.5 day; a captioning side job ~1-2 days (estimates).**

**What it is:** BFL says the model "should not be used" without its prompt upsampler, because the upsampler's captions are what it was trained on, and short prompts give generic output. The dialect (pup_prompts.json) is one prose paragraph of 100-350 words. It opens with medium and style, then scene, lighting and camera, then every in-focus person with an observable profile (ethnicity, gender, age range, skin tone, hair, clothing). An optional JSON array adds `{id, bbox [y0,x0,y1,x1] on 0-1000, desc}` rows, cited in the caption as `<person_1>`. The i2i variant is 50-150 words.

**Evidence:**
- [BFL README](https://github.com/black-forest-labs/flux-3-image-klein): trained on upsampler captions; `--no-pup` is noticeably weaker.
- [pup_prompts.json](https://raw.githubusercontent.com/black-forest-labs/flux-3-image-klein/main/src/flux3/pup_prompts.json): the dialect as above.
- [Distilled card](https://huggingface.co/black-forest-labs/flux-3-image-klein): "Do not use the model without a prompt upsampler".
- The upsampler is a 9B Qwen3.5 fine-tune (Apache 2.0), shipped as Comfy bf16 18.8 GB / int8_convrot 9.8 GB / w4a8 6.7 GB ([comfy repo](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)).
- Nothing measures the effect on a character LoRA. The bbox-tag idea is untested anywhere.

**RulzTrainer today:** caption .txt files, trigger insertion as `trigger, caption` (archs.py:104-107), and face/body boxes from InsightFace/SAM3 for region weighting. There is no auto-captioner, and no W4A8 loader for the 6.7 GB upsampler file.

**Fit and risks:** long profiles that state skin tone, hair and ethnicity can describe away the identity the LoRA should learn. Captions of 240-480 tokens enlarge the text cache (above) and the text part of the joint sequence. The int8 upsampler (9.8 GB) fits a 16 GB side GPU. Qwen3-VL-4B, which is loaded anyway for text, plus BFL's system prompt is the cheaper captioner.

**Implementation path:**
1. First and cheapest: upsample `sample_prompts` and the probe prompts once per run and cache them.
2. Optional side job `flux3_caption`: Qwen3-VL-4B with the image and the pup t2i system prompt, writing .txt captions.
3. `Flux3KleinArch.add_trigger` inserts the trigger into the first person profile ("a woman" -> "a woman, <trigger>,") instead of prefixing it.
4. Leave bbox tags out until a race supports them.

**How to test:** a 3-arm caption race:
- short `trigger, caption`
- upsampled long captions with the trigger in the profile slot
- long captions with the person's attributes stripped except the trigger

Probe all arms with the same upsampled prompts. A winner needs face likeness ahead by more than 0.01 cosine, averaged over the last 3 probes, in 2 of 3 seeds. Otherwise keep the simplest option.

### Distilled checkpoint as probe engine (cross-ref: "Cheaper probe engines", G60)

**Race it (as a gate, after support lands). Expected gain: ~6x fewer DiT passes per probe image (16 vs 100) if transfer holds; transfer unverified. Confidence: medium (0.5). Cost: flow-map sampler ~1 day (estimate).**

Flux 3's fast variant is a separate flow-map checkpoint: the same layout plus `ref_time_in`, called with the current and the target timestep, `x += (t_next - t) * f(x, t, t_next)`, at 16 steps with no CFG. SGLang rejects guidance != 1 and non-empty negatives. Timings are datacenter-only: ~3 s vs 12 s per 1 MP image on H200 (BFL README), and 1.93 s per 1024 px distilled on H100 (HF blog). BFL docs say a Base LoRA runs on Distilled without conversion, at 16 steps, CFG 1, LoRA scale 0.75 (0.5-1.0 work). Community reports agree, secondhand. No source measures whether likeness survives the transfer. kohya's comment on Qwen 2.1 ("a LoRA trained on the base model is not expected to work well on the turbo model") argues for a gate.

Model-specific plumbing: `sample_model_path` / `_get_sample_model` (trainer.py:846-867, 2179-2196) already loads a second checkpoint with mirrored LoRA slots. The 5.8 GB int8 file fits the side GPU. Add `is_distilled` from `ref_time_in.in_layer.weight` (as krea2/single_file.py:159 does by file name), a `sample_flow_map` path in `flux3/sampler.py` that passes the target sigma, and a scale for the trained LoRA at sampling time (no such option exists; `turbo_lora_strength` and `sample_lora_strength`, config.py:321-323, scale other LoRAs). The new option would default to 0.75 for Klein distilled. The text-stage cache below must not be used with the flow map: its target time modulates the early blocks.

**How to test:** render the probe set per checkpoint with Base at 50 steps CFG 4 (reference) and with Distilled + LoRA at 16 steps at scales 0.75 and 1.0. Adopt the distilled engine only if Spearman rank correlation with the reference is at least 0.9 across checkpoints, the best-checkpoint pick is within one checkpoint of the reference's, and the bias drift is under 0.02 ArcFace cosine. The probe-calibration mechanism is in the theme section.

### Text-stage cache (opt-in)

**Skip-ish (later opt-in). Expected gain: about 1% step time, about 190 MB of VRAM, text caches about 6.7x smaller; it costs the ability to train the text blocks. Confidence: medium. Cost: ~1 day (estimate).**

**What it is:** text tokens go through `txt_in` (20480->3072) and 5 `txt_mode_blocks` that see nothing but text at constant t=0. Their output is therefore a pure function of the caption. Cache `[L80, 3072]` after the text stage (6 KB/token) instead of `[L80, 8, 2560]` (40 KB/token): a 480-token window drops from 19.7 MB to 2.9 MB. `txt_in` (63M params, 126 MB bf16) and the 5 text blocks never need to be on the GPU, and the t=0 modulation vector is computed once per run.

**Evidence:** [ComfyUI model_base.py](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/model_base.py)'s Flux3 fast path precomputes `txt_states` once, because "the t = 0 early blocks depend on the conditioning alone", and excludes flow_map models. [BFL dit.py](https://raw.githubusercontent.com/black-forest-labs/flux-3-image-klein/main/src/flux3/dit.py) has `txt_in Linear(20480, 3072, bias=False)` and the 5 text mode blocks. Against it: both ai-toolkit (the whole transformer) and [NeMo](https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/examples/diffusion/finetune/flux3_t2i_flow_lora.yaml) (`txt_mode_blocks` + `context_embedder`) train these blocks with LoRA.

**RulzTrainer today:** text-encoder outputs are cached and the encoder is unloaded. Krea's default trains its text-side stage (`text_fusion`, config.py:24-29). Fizgig measured text fusion as under-trained, and scaling it x3 lifted likeness 2-7 points on several Krea 2 LoRAs. That is a reason to expect Klein's text blocks to matter too.

**Fit and risks:** this caching is incompatible with the `text_blocks` and `embed` (txt_in) target groups and with the distilled flow-map model. Switching target groups invalidates the cache. Text tokens are about 80-160 of ~4200 at 1024 px, so caching the text stage saves well under 2% of step FLOPs.

**Implementation path:** a cache tag `txt_stage`, used only when neither `text_blocks` nor `embed` is targeted. At caching time, run `txt_in` + `txt_mode_blocks` (bf16, frozen) with the t=0 modulation and store `[L80, 3072]`. Cache the empty prompt's stage output too, for dropout and CFG.

**How to test:** only after the default (text blocks trained) exists. Race `text_blocks` on vs off plus `txt_stage` cache. Adopt the cache only if likeness with frozen text blocks is within seed noise, and log s/step and peak VRAM.

### Distillation guards and training adapters (cross-ref: "Guidance-loss / CFG-augmented training target", G48; "De-distillation training adapters", G50)

**Skip-ish for Base. Required only if anyone trains the distilled checkpoint. Confidence: medium (0.6 / 0.65).**

Base is undistilled and sampled with true CFG, so the guidance loss should stay off (G48 verifier: it applies only to distilled bases), and no training adapter is needed. If the distilled checkpoint is ever trained, it needs one of two guards. The first is a guidance-loss target (uncond + g*(v - uncond), g 3-4, sigma_min 0.15, about +50% step time ungated per musubi), which the G48 plan would make the default for a future distilled flux3 arch. The second is a frozen assistant LoRA through RulzTrainer's existing live `training_adapter` branch (lora.py:238-241, 396-398), as SimpleTuner does: assistant at strength 1.0 for training, 0.0 for validation. Never combine the two. Recommendation stays: train Base.

Refuted member inside G48: member 84's "CFG-extrapolated target so Raw-trained LoRAs land on Turbo" (Krea anecdote misattributed; it risks guiding twice). Do not port it to Klein Base to Distilled.

### Self-reference identity through the reference stream (cross-ref: "Self-reference identity conditioning", G39)

**Watch. Expected gain: unmeasured for trigger-only likeness, possibly zero or negative; the likely gain is a second product (likeness from a supplied face photo). Confidence: low (0.3). Cost: several days on top of v1.**

Klein is natively multi-reference: up to 10 references enter as clean t=0 tokens through `emb_in.image_cond` and `content_mode_blocks.image_cond` at RoPE l = -1, -2, .... The joint blocks attend to them. The diffusers img2img LoRA script and ai-toolkit already pack references this way (bn-normalized, t=0, l=-1, sides snapped to 16, `match_target_res`). Klein is the model where this is most in-distribution: the pup i2i rules say transferred people keep "identity, face and accessories". Cost: a 512x512 face reference adds 1,024 tokens (+25% sequence at a 1024 px target). Each full 1024x1024 reference adds 4,096 tokens and roughly doubles generation time (3.87 s vs 1.93 s, H100). The quantified gains in the theme section (Sparse Context, CRAFT) come from multi-subject zero-shot models on datacenter GPUs and do not transfer. The theme section recommends Krea 2 first. For Klein, v1 should only reserve the `ref` target group and keep `image_cond` skipped. If it is built later: `self_reference_share` 0.5 with mismatched-image face crops (dataset.py:215-261), with the trigger-only probe as the score.

### Flow-map-consistent LoRA on the distilled checkpoint

**Watch. Expected gain: unmeasured, probably marginal for character LoRAs. Confidence: low (0.2). Cost: a new post-training stage, ~3-4x a training step per update, after Klein support.**

**What it is:** a second stage that distills the Base+LoRA teacher (with CFG) into a new LoRA on the distilled flow-map checkpoint. Sample consecutive (t, s) from the 16-step grid, integrate the teacher with k Euler substeps, and regress the student's one jump `x_t + (s - t) f_student(x_t, t, s)` onto the teacher's endpoint. The goal is a LoRA for the deployed 16-step model with no base-to-distilled transfer loss.

**Evidence:**
- [Flow Map Matching, arXiv 2406.07507](https://arxiv.org/abs/2406.07507): two-time flow maps, distillation from a velocity field. Sample quality comparable at 10-20x less generation time on CIFAR-10 / ImageNet-32.
- [Align Your Flow, arXiv 2506.14603](https://arxiv.org/abs/2506.14603): a LoRA on FLUX.1 dev distilled to a 4-step flow map without adversarial losses, 10k steps, about 32 A100 GPU-hours. So LoRA-constrained flow-map distillation has been done, but as step distillation, not identity transfer.
- [MeanFlow, arXiv 2505.13447](https://arxiv.org/abs/2505.13447): 1-NFE FID 3.43 on ImageNet 256 from scratch. Irrelevant to a LoRA.
- [BFL dit.py](https://raw.githubusercontent.com/black-forest-labs/flux-3-image-klein/main/src/flux3/dit.py): `timesteps_ref` is required by, and only by, the flow-map DiT.
- The diffusion-pipe "will not work well" quote the scout cited is about FLUX.2 Klein, not Flux 3.

**RulzTrainer today:** no Klein, no flow-map code. The reusable parts are the `_reward_stage` post-training pattern (trainer.py:2751ff, run on one GPU, trainer.py:2762) and the second transformer via `sample_model_path` (trainer.py:2181).

**Fit and risks:** the main selling points mostly exist without it. CFG is already baked into the distilled model, and Base LoRAs reportedly load onto it at scale ~0.75. Per student step: k=4 substeps x 2 (CFG) = 8 teacher forwards + 1 student forward/backward. Two resident DiTs need one model per GPU, or int8/NVFP4 plus swap. `ref_time_in` changes the compiled graph signature.

**Implementation path:** only after the probe-engine gate shows a measurable transfer gap. Add a `flowmap_steps` stage modelled on `_reward_stage`, with options `flowmap_substeps`, `flowmap_lr` and `flowmap_rank`, and save `<job>_flowmap.safetensors`.

**How to test:** a 3-arm comparison on the distilled 16-step model with the same seeds:
- (a) base LoRA at scale 1.0
- (b) base LoRA at scale 0.75
- (c) the flow-map LoRA after 500 stage steps

Reference ceiling: the base LoRA on Base at 50 steps, CFG 4. Adopt (c) only if it beats max(a, b) by at least +0.02 normalized face score over 2 identities. Drop the feature if (b) is within noise of the ceiling.

### FLUX.2 Klein 4B/9B

**Skip-ish. Expected gain: no likeness gain over supported families; only value is an Apache-2.0 4B deliverable. Confidence: medium (0.65). Cost: a second new arch (double-stream blocks), after Flux 3.**

FLUX.2 [klein] is a FLUX.1-style double+single stream DiT: 4B = 5 double + 20 single blocks with Qwen3-4B (taps 9/18/27 = 7680), 9B = 8 double + 24 single with Qwen3-8B (12288). It has no guidance embedding, uses the same FLUX.2 VAE, pads text to a minimum of 512 tokens, and uses a chat template with an empty think block. The 4B is Apache 2.0 and the 9B is non-commercial; both are gated. All recipes agree on training the base at lr 1e-4, rank 16-32 (musubi: adamw8bit, dim 32, blocks_to_swap max 13 / 16; SimpleTuner: rank 16, 10k steps, 12/8 GB for 4B and 22/14 GB for 9B in bf16/int8) ([SimpleTuner FLUX2](https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/FLUX2.md), [musubi flux_2.md](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/flux_2.md)). SimpleTuner's page does not state the base-only rule; BFL, musubi and diffusion-pipe do. BFL puts Flux 3 Klein above klein 9B (+91/+70 Elo, 38% fewer params). Corrections: FLUX.2's sampler uses BFL's empirical mu, about 2.02 at 4096 tokens and 50 steps. The 0.5-1.15 token-count mu is FLUX.1's rule copied in ai-toolkit. Klein 4B also pads to 512 without a mask, so the compact text cache would break it too.

What it would reuse from the Flux 3 work: the Flux2 VAE (moss/single_file.py:221-253), the comfy LoRA export, int8_convrot/NVFP4, and the base-only recipe. New: double-stream blocks exposed as one ordered `layers` list with the img/txt split inside the wrappers, a Qwen3 text-only loader, a 512-row padded cache, and an empirical-mu `reference_sigmas`. Defer unless the user wants Apache-2.0 4B LoRAs. If built, race klein-4B int8 against Qwen 2.1 int8 and Krea 2 nvfp4 at 0.5 MP, rank 16, 2000 steps. Keep it only within 0.02 ArcFace of the best racer while being faster or Apache-licensed.

## MOSS-Image

This section covers what is specific to MOSS-Image: its numerics (a residual stream that peaks near 62k), the open question of LoRA training on the int8_convrot file, the fused-qkv layout, and the MOSS angle on ideas written up in the theme sections. Verdict: no MOSS-only lever has a measured likeness or speed gain. The most useful work is cheap checking of the user's own default path (int8_convrot with `int8_backward: int8`, and the turbo LoRA used for probes) plus one timestep race that needs no code. An fp32 residual stream is a small race behind an offline gate; fused qkv is a sub-1% cleanup.

### Status in RulzTrainer and model facts

Supported in `rulztrainer/moss/` (`model.py`, `loading.py`, `single_file.py`). The user's own config, `configs/default.yaml`, trains MOSS-Image-Base from `moss_image_base_int8_convrot.safetensors` with `transformer_quant: int8_convrot` (line 18), `int8_matmul: true`, `int8_backward: int8` (line 22; the trainer default is `bf16`, `config.py:205`), resolution 1024 with buckets 512-2048, `caption_dropout: 0.05`, `max_token_length: 1024`, rank 16 / alpha 16 on q, k, v, o, ffn_up_gate, ffn_down, `lora_blocks: 21-34` (line 45), `compile_model: true`, adamw8bit at 1e-4 with weight decay 0.01, constant schedule, 50 warmup steps, `logit_normal` mean 0 / std 1 with `timestep_shift: 1.0`, `sample_skip_layer: 22`, and `turbo_lora: moss-8step-v2.safetensors` at strength 0.7 (lines 89-90). That turbo file is not the trainer's built-in default (lodestones/Kromoss `8-steps-dmd-lora`, 12 steps, CFG 1, shift 5, `config.py:44`), and none of the sources below say anything about it.

Model facts (from the sources cited below). MOSS-Image is a single-stream DiT with 42 joint blocks of width 3072. One `shared_block_modulation` Linear (dim to 6 x dim) serves all 42 layers, `ffn_up_gate` is fused, and the activation is SiTU. RoPE is 4D on axes (r, t, w, h), where r is a reference index fixed at 0 in the released model and the leading 32 of 128 head dims are not rotated. It uses the Flux2 VAE; the user's text encoder file is Qwen3.5-9B. It was pretrained with hybrid Muon-AdamW (lr 5e-5, wd 0.01, clip 1.0, warmup 2000, EMA 0.9999). Captions were Gemma4 recaptions in short, medium and long, chosen uniformly, used with probability 0.8, and text tokens were excluded from the loss. Base has no post-training. SFT draws faint vertical stripes that come from blocks 37-41, with a secondary source in 32-36. The joint residual stream peaks at 62,089 (Base) at block 26, 95% of fp16's maximum. Channel 1356 runs at 505x the median RMS, and 8 channels carry 96-99% of the stream energy through block 20. Repacked ComfyUI files fuse q/k/v into one `proj.qkv`, which RulzTrainer splits on load.

### Recommended recipe

1. **Train on Base, not SFT.** The OpenMOSS [USERGUIDE](https://raw.githubusercontent.com/OpenMOSS/MOSS-Image/main/USERGUIDE.md) says Base keeps its training information intact and makes LoRA training easier. [MOSS-Image-SFT-DeBand](https://huggingface.co/MATLOWAI/MOSS-Image-SFT-DeBand) shows SFT's banding (score 2.04 vs Base 1.12) comes from its late blocks. This is already the default. The only change is to add the banding to the SFT hint at `webui.py:2947-2950` (G67).
2. **Keep `lora_blocks: 21-34`.** DeBand and the Fizgig SFT-Look LoRA show where SFT changed colour and texture (blocks 21-41, with 51-61% of the SFT-minus-Base difference there). They do not show where identity lives. No MOSS identity block map has been published. Race 21-31 against 21-34 only after a block profile (G55 in *Likeness from data, teachers, references and Turbo transfer*).
3. **Keep turbo LoRAs unmerged and in fp32.** The [lodestones DMD card](https://huggingface.co/lodestones/MOSS-Image-Base-8-steps-dmd) says the distillation delta is smaller than bf16 rounding, and [Kromoss](https://huggingface.co/lodestones/Kromoss) says not to merge it into bf16 weights. This is already handled (`lora.py:828`, `trainer.py:2147-2157`), and no merge-to-bf16 export exists.
4. **Check the probe sampler before trusting it.** Fizgig found the generic DMD LoRA can understate likeness (`config.py:37-38`, `README.md:1424-1426`). The user probes with a different turbo file at 0.7. Run G68's Step 0 (below) with that file before relying on turbo probes for best checkpoint or race ranking.
5. **Race a higher training shift.** diffusion-pipe's MOSS example uses `logit_normal` with shift 3 ([supported_models.md](https://github.com/tdrussell/diffusion-pipe/blob/main/docs/supported_models.md)). KohakuMoss's `moss` timestep sampler (mu -1.09, s 0.95, cutoff 0.95 in its own convention) has a median sigma of about 0.78 ([training.md](https://raw.githubusercontent.com/KohakuBlueleaf/KohakuMossTrainer/main/docs/concepts/training.md)). RulzTrainer draws sigma = sigmoid(N x std + mean) and then applies the shift (`trainer.py:224, 240-241`). So `timestep_shift: 3` on `logit_normal(0, 1)` puts the median sigma at 0.75, against 0.5 in the user's config. The effect on LoRA likeness is unmeasured. It needs no code, because `timestep_shift` and `logit_mean` are race keys (`race.py:56`).
6. **Keep bf16 DiT compute and the bf16 islands.** Do not add an fp16 path. Plain fp16 overflows at about 2048x3072 / 2560^2, and KohakuMoss itself trains in bf16 (G66, G67). Embedders, modulation and the final layers already stay bf16 (`moss/single_file.py:115-117`). [EntroPack](https://huggingface.co/DiffSynth-Studio/MOSS-Image-EntroPack) names exactly those as MOSS's quantization-sensitive layers (G51).
7. **Keep int8_convrot, but run the parity check.** The only external report says convrot LoRA training diverges on MOSS. RulzTrainer's design differs on both suspected causes, but no bf16-vs-convrot A/B exists, and the user runs the non-default `int8_backward: int8` (G67).
8. **Keep the text encoder frozen** (USERGUIDE). `train_text_embedder: false` is already set. Keep captions within about 256 tokens at 1 MP, KohakuMoss's budget of 4352 tokens = 4096 image + 256 caption ([first-lora.md](https://raw.githubusercontent.com/KohakuBlueleaf/KohakuMossTrainer/main/docs/guides/first-lora.md)). MOSS is the model with the strongest case for a short/medium/long caption mixture (G35 in *Text side*).
9. **Full-quality samples: CFG 3, skip-layer guidance at 22.** The USERGUIDE says layers 22/24/26 all help. This is already `sample_skip_layer: 22` and `ARCH_FULL_SAMPLE` 20 steps at CFG 3.
10. **Keep adamw8bit at 1e-4.** Muon, the optimizer MOSS was pretrained with, is the one MOSS-specific optimizer race. It needs code (G69 in *Fewer steps to likeness: optimizers and learning-rate schedules*).

### int8_convrot training parity (and the other MOSS failure modes)

**Race it. Expected gain: none (validation, risk removal); size of any convrot handicap unmeasured. Confidence: low (0.3). Cost: 20-35 min per arm for the 200-400-step check; the 1500-step race is about 2 h per arm at ~5 s/step; a few lines of README/WebUI text.**

**What it is.** The scout listed four documented MOSS pitfalls. Three of them turn out to be already handled or moot here:
- int8-convrot LoRA training divergence: open.
- fp16 overflow at block 26: moot, because RulzTrainer has no fp16 DiT compute.
- turbo LoRAs merged into bf16: moot, because they are already applied unmerged.
- SFT banding: Base is already the default; only the WebUI wording is missing.

What remains is one open question: does training on the int8_convrot file match training on bf16 weights? diffusion-pipe says it does not, without giving a cause.

**Evidence.**
- [diffusion-pipe supported_models.md](https://github.com/tdrussell/diffusion-pipe/blob/main/docs/supported_models.md): the int8-convrot format "will NOT work right for training LoRAs", the loss "goes up sharply", "I don't know why; it only happens on this model", full bf16 is needed, and float8 is untested. This is one unexplained anecdote about diffusion-pipe's own loading and matmul path, with no numbers or loss curve. A web search found no second report. Community NF4 and FP8 MOSS training bases exist.
- [KBlueLeaf/MOSS-Image-FP16-fixed](https://huggingface.co/KBlueLeaf/MOSS-Image-FP16-fixed): plain fp16 at 2560^2 and 50 steps hits 65,504 at step 29 in block 26, then NaN and a black image. 1536^2 and 2048^2 single forwards do not overflow; 2048x3072 does. The margin is about 1.05x.
- [lodestones/MOSS-Image-Base-8-steps-dmd](https://huggingface.co/lodestones/MOSS-Image-Base-8-steps-dmd): the transformer must run fp32 with no autocast, and casting to bf16 discards most of the distillation. The LoRA route is shift 5, CFG 1, 4-8 steps, rank 128 / alpha 128 on q, k, v, o, ffn_up_gate and ffn_down.
- [lodestones/Kromoss](https://huggingface.co/lodestones/Kromoss): "Do not merge it into the bf16 weights". PartiPrompts CLIP is 28.13 for DMD2 at 8 steps, 27.90 for Kromoss at 50 steps and 27.19 for Base.
- [MATLOWAI/MOSS-Image-SFT-DeBand](https://huggingface.co/MATLOWAI/MOSS-Image-SFT-DeBand): SFT stripes build up across steps, and reverting layers 21-41 fixes them (main fix 37-41). Correction to the scout: the page shows that SFT-difference LoRAs reproduce the banding, not that LoRAs trained on SFT inherit it. A LoRA used on SFT bands anyway, because SFT does.

**RulzTrainer today.**
- int8 weights use per-output-row scales (`quant.py:6-7`), so the scout's proposed fix of per-row instead of per-tensor scales is already in place.
- The LoRA branch sees the unrotated input. `LoRALinear.forward` computes `base(x) + _added(x)` (`lora.py:381-399`), and only `QuantLinear` rotates internally. The hypothesis that the LoRA is fed rotated inputs does not apply.
- The default `int8_backward: bf16` is exact, with the inverse Hadamard applied to grad_x (`README.md:444`). The user's config sets `int8_backward: int8` (`configs/default.yaml:22`), which also quantizes grad_out. That is the less-validated path.
- MOSS training on the int8_convrot file demonstrably works: `README.md:937` reports 1,160 body probes from 32 MOSS-Image runs with likeness rising. The input does not say which `int8_backward` those runs used.
- Tests check only layer-level quantization error (`tests/test_quant.py`) and OFT/DoRA under int8_convrot (`tests/test_oft_dora.py:60`). No bf16-vs-convrot training A/B exists.
- Turbo LoRAs are applied unmerged as sampling-only extras (`lora.py:828`, `trainer.py:2147-2157`). The WebUI already steers users to Base (`webui.py:2947-2950`).

**Fit and risks.**
- A bf16 MOSS arm needs a 10.2 GB DiT; the text encoder is unloaded after caching. It fits a 16 GB card at 0.5 MP with checkpointing, or with block swap. The user's 1 MP resolution may need block swap.
- Block swap excludes torch.compile and data_parallel in RulzTrainer. That changes the arm's speed, not its loss, so compare loss and probes per step, not per wall-clock minute.
- No interaction with model_split or export formats, and nothing Windows-specific.
- Precision and quant settings are not race keys (`race.py:48-75`), so run the arms as separate jobs, one per GPU.

**Implementation path.**
1. Add `scripts/moss_quant_parity.py`, an offline benchmark rather than a pytest, because it needs the real weights. It loads `moss_image_base_bf16` and `moss_image_base_int8_convrot` through `rulztrainer/moss/single_file.py`.
2. Run three arms from the same cached latents and captions, with identical seeds, sigma draws and noise, for about 300 steps at rank 16:
   - bf16;
   - int8_convrot W8A8 with `int8_backward: bf16`;
   - int8_convrot with `int8_backward: int8`.
3. Log the per-step loss difference. At a few checkpoints, log the relative error of the W8A8 output against bf16 for `proj.o` and `proj.ffn_down` in blocks 18-32, where the stream peaks.
4. If the curves agree, write a measured MOSS parity statement into README.md's int8 section.
5. Add one sentence to the SFT hint at `webui.py:2949`: SFT draws faint vertical stripes from its late blocks, so train on Base and preview on SFT.
6. Do not add an fp16 guard; the DiT always computes in bf16.
7. Only if divergence shows up, add a MOSS-specific exemption in `quant.py` that falls back to a dequantized bf16 matmul for the stream writers.

**How to test.**
- Arms: on the user's character dataset, MOSS Base bf16 against MOSS Base int8_convrot (W8A8, `int8_backward: bf16`), plus a third arm with `int8_backward: int8`, the user's setting.
- Shared settings: same seed, dataset, rank 16 and lr, 0.5 MP, about 1500 steps.
- Probes every 250 steps with the same probe seeds and the same turbo sampler in every arm.
- Metrics: mean ArcFace face score and DINO body score per probe, plus the smoothed training loss.
- Parity: the final-probe ArcFace difference is within one-seed probe noise (about 3 points, per README's noise column), and the smoothed loss curves stay within 2% of each other with no upward drift in a convrot arm.
- Defect: a convrot arm's loss rises over the first 200 steps while bf16 falls, or its probe score trails by more than 2x the noise at two consecutive probes. In that case, run a further arm with `int8_matmul: false` (dequantized bf16 compute) to separate the W8A8 activations from the weights.
- Cheap first pass: the 200-400-step version of the script takes about 20-35 minutes per arm at about 5 s/step int8 (bf16 is somewhat slower). Stop there if all arms overlay.

### fp32 residual stream (the MOSS numerics fix)

**Race it, only after an offline gate passes. Expected gain: unmeasured for training, probably small to none for likeness; probe renders about 5-8x closer to fp32 in LPIPS. Confidence: low (0.25). Cost: a few lines in `moss/model.py` plus a short gate script; about +0.6 GB VRAM at 0.5 MP and 1-3% step time.**

**What it is.** MOSS's joint residual stream reaches about 62k, and a few channels carry almost all of its energy. At that magnitude bf16's 8-bit mantissa rounds in steps of 256, so block outputs and small LoRA deltas written into those channels are partly lost. Carrying the residual `x` in fp32 across blocks, while still casting the normed inputs of every Linear to bf16/int8, removes that rounding at no change to the weights or the LoRA space. The alternative, fp16 compute with a 1/16 fold of every stream writer, gets the same fidelity but touches the quant kernels, and the verifier rejects it.

**Evidence.**
- [KBlueLeaf/MOSS-Image-FP16-fixed](https://huggingface.co/KBlueLeaf/MOSS-Image-FP16-fixed):
  - The stream peaks at 62,089 (Base) / 61,828 (SFT) at block 26 of 42.
  - bf16 vs fp32 gives LPIPS 0.0501 / 0.0901 / 0.0935 (Base) and 0.0413 / 0.0656 / 0.0683 (SFT) across ComfyUI / diffusers / KohakuMoss, with PSNR 26.7-29.3 dB. fp16-fixed vs fp32 gives 0.0062 / 0.0113 / 0.0147 and 0.0042 / 0.0096 / 0.0122.
  - VAE decode PSNR is 57.9 dB in bf16 vs 77.3 dB in fp16.
  - The fold is c = 1/16 on video_embedder, text_embedder.proj, the refiner proj/ffn_down and the gate rows of shared_block_modulation; 3.2% of the folded values are subnormal.
  - The card itself says plain fp16 at 1024^2 is about as close to fp32 as the fixed weights (ComfyUI Base 0.0062 for both). So most of the gain is fp16's mantissa, not the fold.
  - Limits: inference only, 50-step CFG sampling at 1024^2, the author's own small benchmark (4 images).
- [KohakuMossTrainer numerics.md](https://raw.githubusercontent.com/KohakuBlueleaf/KohakuMossTrainer/main/docs/concepts/numerics.md):
  - Run-time modes are none / online / fp32_residual, with clamp addable to any of them ("weights" is an export script, not a run-time mode).
  - Probe maxima: joint stream 6.2e4, text stream 1.8e3, o 1.3e3, ffn_down 700, attention logits 600, ffn_up_gate 170.
  - time_embedding, shared_block_modulation and video_head_modulation are kept fp32.
  - No loss-scaler start value is given. The scout's "2^16, as KohakuMoss" is unsupported.
- [KohakuMossTrainer massive-channels.md](https://raw.githubusercontent.com/KohakuBlueleaf/KohakuMossTrainer/main/docs/concepts/massive-channels.md): channel 1356 at 505x the median RMS, channel 42 at 99x, about 30 more at 10-30x. The 8 largest channels hold 96-99% of the energy in blocks 1-20 (98.7% at block 10) and about half at block 35.
- Against a training benefit:
  - KohakuMoss trains in bf16 and uses fp16 with the residual fix only on cards without bf16 ([wiki](https://github.com/KohakuBlueleaf/KohakuMossTrainer/wiki)).
  - [LoRA-Forge](https://github.com/PineCookie/lora-forge/releases) 1.3.0's fp16 MOSS mode is framed as speed, not quality.
  - The ulp-256 rounding hits only the ~8 massive channels. Content channels are about 500x smaller and round at about 0.5-1, so identity deltas there are not swallowed.
  - No source measures LoRA likeness under bf16 vs fp32-residual or fp16 training.
- Refuted member: masking or penalizing LoRA writes into the massive channels (KohakuMoss concentration loss). The cited 0.21-0.23 vs 0.32 top-32 share comes from a LoKr finetune of the already healed model, with 0.32 as its starting value and no unpenalized run. The +1.7% loss figure is from a small non-MOSS DiT. The page says nothing about LoRAs. A rank-8 write of order 1-100 into a channel at about 6e4 barely moves the RMSNorm divisor. Do not build it unless the diagnostic in step 2 of "How to test" shows real LoRA energy in the top-8 channels.

**RulzTrainer today.**
- `rulztrainer/moss/model.py:223` builds `x = torch.cat((video, text), 1)` in the embedder weight dtype (bf16).
- `Block.forward` (lines 109-120) adds `o * gate_a` and `ffn_down * gate_f` into x in bf16.
- `RMSNorm` (lines 34-41) already computes in fp32 and casts back to `x.dtype`.
- The Linear inputs are already cast with `y.to(dtype)` before q/k/v and ffn_up_gate.
- Probes and samples already decode with an fp32 VAE (`moss/loading.py:76`). Only `grad_vae`, the id-loss decode (`loading.py:195-201`), is bf16, so the VAE's +19 dB point is mostly moot here.
- There is no fp16 path anywhere: MOSS loaders read every format into bf16 (`moss/single_file.py:25`), and the only fp16 option is `lora_save_dtype`.
- Delta: the dtype of x and of the per-sample modulation.

**Fit and risks.**
- int8 ConvRot W8A8 and NVFP4 quantize the post-norm `y`, so they are untouched; their bf16 outputs promote to fp32 on the add.
- torch.compile: one dtype-driven recompile, no new graph breaks.
- Keep blocks: block-boundary tensors double. At 0.5 MP that is about 2,250 tokens x 3,072 x 4 B = 28 MB x 42 blocks = 1.2 GB, against 0.6 GB in bf16, plus fp32 adds inside kept blocks. The calibration would keep 1-3 fewer blocks on 16 GB. The user trains at 1 MP, where the scout estimated about 1 GB extra at 4096 tokens.
- Block swap moves only weights, so it is unaffected. model_split moves x between GPUs at 2x the bytes, which is negligible. data_parallel threads are unaffected.
- Export is unchanged. The one side effect is a small train/inference numerics mismatch for users who run the LoRA in bf16 ComfyUI.
- The fp16 + 1/16 fold path is not worth it:
  - the fused triton W8A8 rescale and NVFP4 `_scaled_mm` emit bf16;
  - it needs a GradScaler with fp32 LoRA params under threads;
  - the fold changes ConvRot/NVFP4 scales on the embedders;
  - embedder, text-projection and modulation LoRAs must be unscaled on export;
  - "free fp16 tensor cores" is irrelevant when the Linears run int8 or FP4.

**Implementation path.**
1. Add `moss_residual: bf16 | fp32` (default bf16) to `rulztrainer/config.py`, the wizard and README.
2. In `rulztrainer/moss/model.py` forward, when the option is fp32:
   - `x = torch.cat((video, text), 1).float()` at line 223;
   - make `mod_i` / head modulation fp32;
   - do the same for the text-refiner blocks' residual.
   `Block.forward` needs no change.
3. Keep the cast to `final_norm_video`'s dtype as today.
4. Let `checkpointing.py`'s keep-blocks calibration re-run. Expose the flag to the sampler, so probes and samples use the same numerics.
5. MOSS-Image only. Krea 2 and Qwen 2.1 have no reported massive-stream overflow.

**How to test.**
1. Offline gate, a few minutes. Take an existing MOSS character LoRA (int8 base) and a few cached latents. At sigma 0.2 / 0.5 / 0.8, compute the LoRA delta `v_lora - v_base` under three settings: bf16 residual, fp32 residual, and full fp32 (no quant, fp32 Linears). Report the cosine and relative error of each delta against full fp32. If bf16's delta cosine is at least 0.99 and within 1% of the fp32-residual one, stop.
2. Massive-channel diagnostic, same LoRA. Measure the share of the o and ffn_down LoRA output energy that lands in the top-8 channels by stream RMS. If it is about 8/3072, drop the channel-mask idea for good.
3. Race only if step 1 fails its threshold: `moss_residual` bf16 vs fp32.
   - Same dataset, seed and int8 base; rank 16; 2000 steps at 0.5 MP; probes every 250 steps.
   - Render the probes for both arms with the fp32-residual sampler, so measurement bias is equal.
   - Metric: ArcFace face (primary) and DINOv2 body, mean of the best 3 probes.
   - 2 seeds per arm. Adopt only if fp32 wins on both seeds by more than probe noise (about 0.01 ArcFace).

### Fused qkv as one W8A8 GEMM

**Skip-ish (do it when the MOSS loader is next touched, or if MOSS nvfp4 is used). Expected gain: unmeasured; estimate 0.5-1% s/step for MOSS int8_convrot, bit-identical numerics. Confidence: low (0.35). Cost: moderate refactor (lora.py, single_file.py, model.py, archs.py; effort not estimated by the verifiers).**

**What it is.** RulzTrainer splits MOSS's fused `proj.qkv` into three Linears on load. Each joint block therefore rotates and quantizes the same normed activation three times and launches three narrower int8 GEMMs. Keeping one [3072 x 9216] QuantLinear and chunking its output does one quantization pass and one `_int_mm`. The LoRA modules keep their q/k/v names as delta-only branches added to the output slices.

**Evidence.**
- [ComfyUI comfy/ldm/moss_image/model.py](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/ldm/moss_image/model.py): the comment says repacked files fuse q/k/v for "one GEMM and one activation quantization instead of three". Confirmed nearly word for word.
- [ComfyUI PR #17095](https://github.com/comfyanonymous/ComfyUI/pull/17095) (kijai, merged 2026-10-16): fused qkv is detected from the file, and q/k/v LoRA keys map onto slices of the fused weight. It has no benchmark.
- [KohakuMossTrainer training.md](https://raw.githubusercontent.com/KohakuBlueleaf/KohakuMossTrainer/main/docs/concepts/training.md): Muon splits qkv into 3 chunks. It does not say whether the trainer keeps the GEMM fused; that was the scout's inference.
- Contrary: the [diffusers fast-diffusion tutorial](https://huggingface.co/docs/diffusers/v0.25.1/tutorials/fast_diffusion) measured `fuse_qkv_projections` on SDXL at 2.54 s to 2.52 s, under 1%. ComfyUI's reason is inference throughput. In training, the LoRA forward, the backward and the checkpoint recompute dominate, so the share saved is smaller.

**RulzTrainer today.**
- `unfuse_qkv` is always called on load (`moss/single_file.py:134-149`) and is documented at `README.md:330`.
- `Block` keeps separate q/k/v Linears (`moss/model.py:105-116`).
- Each QuantLinear runs its own rotate+quantize, `_int_mm` and rescale (`quant.py:758-768`). Each sits behind `@torch.compiler.disable` (`quant.py:430-653`), so each is a graph break.
- `QuantWeight.split_rows` dequantizes and requantizes nvfp4 slices on load (`quant.py:224-228`), which adds error to MOSS nvfp4 files.

**Fit and risks.**
- int8 / int8_convrot (the user's path): fusing is exact. Activation quantization is per token, weight scales are per row, the rotation runs along the input dim, and int32 accumulation is exact. The fused forward is bit-identical.
- Saved per block:
  - forward, and again in the checkpoint recompute: 2 rotate+quantize passes over [M, 3072], 2 rescales, 2 `_int_mm` launches;
  - backward: one grad_x GEMM instead of three, one inverse Hadamard instead of three;
  - 2 graph breaks.
- Cost: the exact backward briefly holds a [9216, 3072] bf16 weight copy, about 38 MB more transient peak per block.
- NVFP4: a fused weight removes the lossy per-slice requantization. That is the one real side benefit.
- DoRA scales the input per module (`dora_input_scale`, `lora.py:381-398`), so DoRA must keep the unfused path. LoRA, LoRA+ and OFT work per slice.
- Unaffected: block swap (one larger pinned tensor), model_split, data_parallel, keep blocks (it re-measures), export (keys stay `lora_unet_layers_N_proj_{q,k,v}`), Windows (no new kernel).

**Implementation path.**
1. `moss/single_file.py`: skip `unfuse_qkv` when the file has `proj.qkv`. For unfused files, concatenate the q/k/v rows into one QuantWeight, which is exact for int8. Add `fuse_qkv` (auto on for int8 / int8_convrot, off when `dora` is set).
2. `moss/model.py` Block: when fused, keep a frozen `proj["qkv"]` (3072 to 9216) and chunk its output.
3. `lora.py`: add a delta-only LoRALinear variant with no base, named `proj.q/k/v`, so targets, `lora_blocks`, LoRA+ groups and export keys are unchanged. Stacking the three A matrices into one [3r x 3072] GEMM is an optional second step.
4. `archs.MossArch`: map the q/k/v targets onto the fused block.
5. Qwen 2.1 int8 next, by concatenating to_q/k/v at load. Krea 2 nvfp4 last, if at all: each slice has its own global scale2.

**How to test.** This is a speed change with exact numerics, so no likeness race is needed.
1. Unit test: the int8_convrot fused forward matches the three-Linear forward bitwise, grad_x matches within bf16 summation tolerance, the LoRA state_dict keys are identical, and an exported file loads in the unfused path and in ComfyUI.
2. Timing on both cards: MOSS int8_convrot at 0.5 and 1 MP, batch 1, rank 16, compile on, `checkpoint_keep_blocks: auto`, block swap off and on. Run 200 steps after a 30-step warmup. Report median s/step, peak VRAM and the graph-break count.
3. Keep it on by default only if s/step improves by at least 1% on both cards and peak VRAM rises by at most 100 MB. Otherwise make it opt-in.
4. Sanity check: one 1000-step character LoRA whose probe curves overlap within seed noise.

### Cross-referenced items: the MOSS-specific part

These items are written up in the theme sections. Only the MOSS-specific settings, numbers and risks are given here.

- **Muon for LoRA (G69, *Fewer steps to likeness: optimizers and learning-rate schedules*).**
  - MOSS is the one supported model pretrained with Muon: hybrid Muon-AdamW, lr 5e-5, momentum 0.95, 5 Newton-Schulz steps, wd 0.01, clip 1.0 ([report](https://arxiv.org/html/2610.23374)). The optimizer-mismatch argument therefore mildly favours Muon here.
  - The 28.6-33.2% fewer-steps figure is from a 1B pretraining ablation at 256 px and lr 1e-4, and does not transfer. KohakuMoss and Kromoss (lr 9e-6, full FT) use Muon only for full finetunes.
  - MOSS-specific implementation: orthogonalize the fused qkv up factor in 3 chunks and ffn_up_gate's in 2, deriving the split from the module name as `lora.py:804` does for export.
  - Race: adamw8bit at 1e-4 against muon at 1e-4 and 2e-4 on MOSS int8, rank 16, 2 seeds. Adopt only if Muon reaches the adamw8bit peak at least 15% sooner. Needs code. Confidence low (0.3).
- **Block profiler (G55, *Likeness from data, teachers, references and Turbo transfer*).**
  - The MOSS evidence (DeBand, SFT-Look) maps SFT's colour and banding drift, not identity. Excluding 35-41 is already the default (21-34).
  - Suggested MOSS groups: 0-10, 11-20, 21-27, 28-34, 35-41. Profile an existing 21-34 LoRA, then race 21-34 against 21-31 and against the profiler's top two groups.
  - At about 5 s/step this is unmeasured. Watch for one recompile per block when multipliers flip inside compiled blocks.
- **Caption mixture (G35, *Text side*).**
  - MOSS's member is the best sourced of the group: Gemma4 short/medium/long chosen uniformly, generated captions with p = 0.8 (20:80), mostly English, text tokens excluded from the loss, a 4096-token final limit.
  - The USERGUIDE's post-training JSON schema is `caption.long/medium/short` plus watermark and borders, with captioners Gemma 4, Qwen 3.5 27B or Qwen 3.5 35B-A3B (temperature 1, top_p 0.95).
  - For MOSS, accept a `<stem>.json` in that schema as caption variants. Whether the user's Qwen3.5-9B text-encoder file has a vision tower usable for captioning is unchecked.
  - Likeness gain is unmeasured. Probe with both short and long prompts.
- **Activation offload (G12, *VRAM without losing speed or quality*).**
  - RulzTrainer's own data: keeping 6 of 42 MOSS blocks instead of recomputing gave only -4% (`README.md:1752-1757`). Removing all recompute is plausibly 5-15%, taking MOSS int8 from about 5 s to 4.4-4.7 s on the 5060 Ti if the copies hide. Unmeasured.
  - Pitfall: `quant.py` saves quantized weights through `save_for_backward`, so a naive pack hook would offload the weights too.
  - Refuted member: the KohakuMoss "larger micro-batch under weight streaming" gain is packing, not throughput (tok/s flat at 5050 vs 4975), and does not apply at batch 1.
- **Mixed-precision base (G51, *VRAM without losing speed or quality*).**
  - EntroPack's MOSS package keeps embeddings, time embeddings, modulation and final layers at higher bits. RulzTrainer already keeps those bf16 (`moss/single_file.py:115-117`).
  - EntroPack is entropy-coded storage that is dequantized for compute, so its 3.5 bpp does not carry over to FP4 matmuls.
  - The KohakuMoss page cited for MOSS quantization sensitivity never discusses quantization (misattributed).
  - MOSS already fits at int8 (about 5 GB DiT) on 16 GB, so the VRAM saving has no use. Speed from moving most blocks to nvfp4 on the 5060 Ti is unmeasured (probably 10-20% of the GEMM half of the step).
- **Reference-token identity (G39, *Likeness from data, teachers, references and Turbo transfer*).**
  - MOSS has a reserved RoPE r axis (`moss/model.py:143-156`, r hard-coded to 0); the report lists reference conditioning as future work.
  - [ComfyUI PR #17296](https://github.com/comfyanonymous/ComfyUI/pull/17296) feeds reference latents through the video embedder with their own RoPE index (826 to 457 ms/step with cached KV at 1024^2). chinoll's Moss-Image-Edit LoRA shows a LoRA can open the channel. The card states neither rank 64 nor ConceptEdit-12M as its training data.
  - Base never saw r > 0 tokens, so this is the weakest prior in the group: a per-subject LoRA must teach the channel from scratch.
  - Cost is about +20% wall time at a 50% reference share. Trigger-only likeness gain is unmeasured and may be negative. Krea 2 goes first.
- **Personalized few-step distillation (G68, *Likeness from data, teachers, references and Turbo transfer*).**
  - DiffSynth's MOSS OPD recipe is LoRA rank 64, lr 1e-4, 5 epochs, a teacher with 50 sigmas at CFG 3 skipping layer 22, and a student with 8 timesteps at CFG 1 ([script](https://raw.githubusercontent.com/modelscope/DiffSynth-Studio/main/examples/moss_image/model_training/special/opd/train_opd_lora.py)). No quality table exists.
  - It costs about 25-30 s per step on MOSS int8 at 0.5 MP, so a few hundred steps take about 2-2.5 h. It cannot fix in-training probes, because it needs the finished LoRA.
  - Do its Step 0 now, with no code: on 2-3 finished MOSS runs, re-probe the saved checkpoints with full quality (`turbo_lora: ""`, 30 steps, CFG 3, skip 22) and with the turbo sampler, and compute the Spearman correlation of ArcFace and DINO likeness across checkpoints. If Spearman is at least 0.9 and the best checkpoint is the same, keep turbo probes and drop the idea. For the user this must use the `moss-8step-v2` file at 0.7, the sampler actually in use.

## MiniMax H3

This section covers MiniMax H3: what the model is, the still-image character-LoRA recipe the other trainers have settled on, and a concrete plan for adding it to RulzTrainer, including whether a 33B video+audio DiT can be trained for stills on the 5060 Ti / 4060 Ti 16 GB pair and how fast. Verdict: a port is feasible and reuses most of RulzTrainer's machinery (Arch plug-in, ComfyUI int8_convrot reader and W8A8 kernels, block swap, the Qwen3-VL text path, the live training adapter). It does not fit one 16 GB card without streaming most blocks or the pruned NVFP4 file. On vast.ai cards from 32 GB (RTX 5090) up, the pruned int8 file runs with no swap. No source has compared H3 against Krea 2 or Qwen-Image 2.1 on still-image likeness. Before writing the port, check whether H3 beats Krea 2 on the probes. The README already sends H3 training to ai-toolkit (README.md:726-731), and nothing in the evidence argues for changing that before such a check exists.

### Status in RulzTrainer and model facts

**Status.** Not supported. `ARCH_NAMES` and `_ARCHS` list only moss_image, fibo2, krea2 and qwen_image21 (rulztrainer/archs.py:29, 580). README.md:726-731 says H3 and FLUX Klein are trained in ai-toolkit. It also records two H3 facts already: `adaln_proj` layers call their raw weight, so a LoRA there gets no gradient, and Fizgig's ablations put identity in blocks 20-49. Fizgig's H3 EMA A/B is the source of the `lora_ema` 0.98 default (README.md:594, ema.py:5).

**Model facts** (G71, checked byte-for-byte against the official config, diffusers and ComfyUI code by the verifier). H3 ([MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), released 2026-07-28/31) is a 33B dense single-stream video+audio DiT. About 13B of its parameters sit in per-block AdaLN branches that depend only on the timestep, so a LoRA cannot use them. The [transformer config](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/transformer/config.json) gives:
- 50 blocks plus 2 text token-refiner blocks; hidden 5376; 56 heads x 128 (inner 7168); SwiGLU FFN 14336; QK-RMSNorm; full MHA with no mask.
- Text, video, audio and condition rows packed into one sequence, with a 3-slot (video/text/audio) AdaLN modality table and 3-axis MM-RoPE.
- Text encoder: Qwen3-VL-32B hidden_states[50] before the final norm (50 of its 64 layers are kept), 5120 -> 5376.
- VAE: f16t4d24 (16x spatial, 4x temporal, 24 channels, per-channel mean/std, [config](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/vae/config.json)) with a 1x2x2 patch. One image token covers 32 px, so a 1 MP still is 32x32 = 1024 tokens.
- [Scheduler](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/schedulers/scheduling_minimax_h3.py): rectified-flow Euler with `x0 = x_t + sigma*v`. That is the opposite sign to RulzTrainer's target. Constant shift `s*sigma/(1+(s-1)*sigma)`, s = 12 for video and 3 for audio, with no sequence-length mu.
- Checkpoints: CFG-distilled FL2VA and Ref2VA. They were trained on 5-15 s clips at 24 fps and 768 px short edge. Pretraining included text-to-image ([blog](https://www.minimax.io/blog/minimax-h3)).

**Licence** ([LICENSE](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE), [licence Q&A](https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/docs/QA-about-License.md)). It excludes the US, EU, UK and South Korea unless you apply to MiniMax, and needs authorization above USD 20M yearly revenue. It also requires a NOTICE file and a visible "MiniMax H3" mark, and LoRAs inherit the terms. The user has to check this before any work.

**Correction to one input claim.** The scout wrote that H3's 1024 tokens per MP is the "same token density as Krea 2 / Qwen-Image". It is not. RulzTrainer's families all use a 16-pixel latent cell, so 1 MP is 4096 image tokens (rulztrainer/archs.py:4, 49; Qwen 2.1's sample_sigmas counts `(h/16)*(w/16)`, archs.py:519-523). H3 has a quarter of the image tokens at equal megapixels. This changes the speed estimate below. It also means H3's 0.25 MP default is only 16x16 tokens.

### Recommended recipe

These are the settings the evidence supports for a still-image character LoRA. Each line gives its source. Nothing here is measured on the user's data.

- **Base file:** Comfy-Org `minimax_h3_fl2va_pruned_int8_convrot` (20,970,379,616 B) ([Comfy-Org tree](https://huggingface.co/api/models/Comfy-Org/MiniMax-H3/tree/main/diffusion_models)). It is the default in ai-toolkit and diffusion-pipe. Fizgig measures the int8 base error at 0.17% ([Fizgig MINIMAX_H3.md](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/MINIMAX_H3.md)). On the 5060 Ti alone, the community pruned NVFP4 file (11.67 GiB, [coolthor](https://huggingface.co/coolthor/MiniMax-H3-pruned-NVFP4)) is the alternative.
- **Targets:** `blocks.N.attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2`. Exclude `adaln_proj` and `token_refiner` ([musubi minimax_h3.md](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/minimax_h3.md); README.md:729; Fizgig: training the refiner "softens output").
- **Blocks:** lora_blocks `20-49`, never 0-5. Fizgig says 0-5 "deform anatomy and pull the dataset's colour into the render", and 20-49 "reaches higher likeness sooner with quicker steps" than 6-49 (unquantified).
- **Rank:** 8 by default, 16 for larger sets (Fizgig). musubi's example uses 16/16; the fal realism LoRA (clips) uses 32.
- **Distillation guard (mandatory):** a frozen training adapter, either [circlestone image adapter](https://huggingface.co/circlestone-labs/MiniMax-H3-Image-Training-Adapter) (Fizgig's default for photos) or Ostris v3. Without a guard, one-frame training shows "wobbly lines and broken proportions" within about 50 steps ([musubi minimax_h3_1f.md](https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/docs/minimax_h3_1f.md)). Never combine the adapter with the guidance loss (circlestone card; [diffusion-pipe](https://raw.githubusercontent.com/tdrussell/diffusion-pipe/main/docs/supported_models.md): "either this or a training adapter (not both)").
- **Timesteps:** uniform base time through shift 12 (musubi, diffusers scheduler, SimpleTuner). For stills the right shift is open. diffusion-pipe's example uses image_shift 1, "or slightly higher", and says shift 8 helps detail ([diffusion-pipe toml](https://raw.githubusercontent.com/tdrussell/diffusion-pipe/main/examples/minimax_h3_example.toml)). Race it; see "Probing and sampling stills".
- **Optional focus band:** 0.4-0.8 base time with probability 0.5 (musubi `--h3_timestep_focus_prob`). The "~2x faster in the band" claim is one unquantified statement.
- **Optimizer:** automagic3 from 1e-6 (Fizgig's default), or fp32 adamw 1e-4, which Fizgig calls "the single biggest likeness gain on H3" (unquantified). musubi's example is adamw8bit 1e-4.
- **lora_ema 0.98:** already the default. Fizgig: 5 points above EMA-off on late epochs, with half the epoch-to-epoch spread.
- **Resolution:** start at 0.25 MP (Fizgig: ~4x cheaper per step than 1 MP), then step up to 0.5-1 MP with probes. fal found "resolution mattered more than rank or step count" across 16 configurations, judged by human review ([fal LoRA](https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA)).
- **Captions:** caption dropout 0.05 (Fizgig). For one-frame items, captions should say there is no sound (musubi _1f).
- **Samples and probes:** CFG 1, adapter off. Decode the duplicate latent and keep frame 0 (musubi, within 1-2 dB of a true two-latent decode). Turbo: larryvrh `turbo_v4_step600_ema` at 6 steps, strength 0.75 (Fizgig previews; unvalidated on 1-frame renders).
- **Full fine-tune, if ever:** lr 3e-5; "1e-4 destroys an H3 fine-tune" ([Fizgig FINETUNE.md](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/FINETUNE.md)).
- **Memory on 16 GB:** int8 with about 36 blocks streamed (Fizgig's ~15 GB tier), or model_split over both cards, or NVFP4 with few swaps on the 5060 Ti. musubi gives a 64 GB host-RAM floor ([musubi minimax_h3_advanced.md](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/minimax_h3_advanced.md)).

### Adding it to RulzTrainer (P27, G71)

**Prototype, gated on a pre-port likeness check. Expected gain: access to H3 inside the probe/race stack. Likeness versus Krea 2 / Qwen 2.1 is unmeasured, and step time is unmeasured (estimate below). Confidence: medium (0.6). Cost: ~2-3 weeks for a stills-only first version (my estimate; the verifier sized the core files at ~1000 lines, like rulztrainer/qwen21).**

**What it is.** A new `minimax_h3` family that trains one-frame image LoRAs on the pruned int8_convrot FL2VA file. The adapter is live (never merged), the text cache is Qwen3-VL-32B layer 50, the latents are 24-channel 1-frame, and the target is H3's flow target under a constant shift.

**Evidence.**
- Every architecture fact above comes from the [diffusers transformer](https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/transformers/transformer_minimax_h3.py), the config, the scheduler and [ComfyUI comfy/ldm/minimax/model.py](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/ldm/minimax/model.py), checked by the G71 verifier.
- Trainer support: musubi (experimental, with a one-frame mode), ai-toolkit (pruned int8_convrot + nvfp4 TE), diffusion-pipe (T2I/T2VA), Fizgig (LoRA/LoKr, experimental full FT) and SimpleTuner (which does have a quickstart). OneTrainer has none.
- The only published H3 training timings are musubi's: 27 -> 12.7 s/it (bf16 -> INT8, swap 48) and ~4 s/it (h2d-only, swap 32), on an unnamed GPU with video latents. Treat them as ratios, not 16 GB speeds.
- Fizgig's "likeness in a few hundred steps" has no measurement behind it. No source compares H3 stills with Krea 2 or Qwen 2.1.

**Local check.** The repo's own venv has diffusers 0.41.0 with `models/transformers/transformer_minimax_h3.py` (MiniMaxH3Transformer3DModel), `models/autoencoders/autoencoder_kl_minimax_h3.py` (AutoencoderKLMiniMaxH3, spatial tiling on by default) and `autoencoder_kl_minimax_h3_audio.py`. Neither the diffusers transformer nor its VAE has the pruned `adaln_t_table` curve mode; only ComfyUI's model.py does.

#### What is reused (file:line)

| Piece | Where | Use for H3 |
|---|---|---|
| Arch plug-in | rulztrainer/archs.py:36 (base), :452 (QwenImage21Arch as the template), :580 (`_ARCHS`), :595 (`arch_from_transformer_file`), :641 (`detect_arch` class map) | New `MiniMaxH3Arch` with `n_blocks 50`, `size_multiple 32`, `latent_tag "_h3vae"`, `lora_format "comfy"`, `compile_blocks` -> `model.layers`. Add a file signature (e.g. `token_refiner.blocks.0...` or `adaln_t_table`) and `"MiniMaxH3Transformer3DModel"` to the detect map. |
| ComfyUI quant reader | rulztrainer/moss/single_file.py:25-45 (`read_state`: `comfy_quant`, `weight_scale`, `weight_scale_2`), rulztrainer/quant.py:1058-1074 (`parse_comfy_quant`, `quant_weight_from_comfy`) | Reads int8_tensorwise+convrot (with `convrot_groupsize`) and nvfp4 directly. The H3 files use the same weight / weight_scale / comfy_quant triple. `w6a8` is rejected (quant.py:1074). |
| W8A8 / NVFP4 kernels | rulztrainer/quant.py (QuantLinear :966, `w8a8_forward` :759, `nvfp4_forward` :876), `can_quantize` :247 (min_dim 1024) | The block linears run W8A8 as for Krea/Qwen. The 9-wide pruned `adaln_proj` falls under min_dim and stays unquantized automatically. |
| Own NVFP4 quantization | rulztrainer/quant.py:235 `quantize_weight` | Quantize the 40 GB pruned bf16 file on load if the community NVFP4 file does not parse. This needs the bf16 file in RAM once. |
| Block swap | rulztrainer/blockswap.py:54 (`BlockSwap` streams `model.layers[i]` H2D-only from pinned RAM), trainer.py:2349-2361 (swaps blocks 0..k-1), config.py:589-599 (single GPU, gradient_checkpointing required, no finetune) | Works unchanged if the H3 model exposes `.layers`. The first k blocks are swapped, which matches lora_blocks 20-49 (see the speed section). |
| Forward-only prefix | rulztrainer/lora.py:451 `backprop_blocks`, checkpointing.py:35 `trained_block_indices` | With lora_blocks 20-49 and no LoRA before the blocks (refiner frozen), blocks 0-19 run forward-only. |
| Text encoder | rulztrainer/qwen21/loading.py:119 `load_text_encoder`, :154 `encode_prompt` (hooks the final norm's input = the state before the final norm), krea2/single_file.py:200 `build_text_model(state, quant, device, config)`, fibo/loading.py:93 `place_text_encoder` (auto: one GPU if it fits, else accelerate dispatch over GPUs and CPU), config.py:198-200 | Build Qwen3VLTextModel with a 32B text config and `num_hidden_layers 50` from Comfy's `qwen3vl_32b_minimax_h3` int8_convrot file (27.1 GB). Final norm becomes Identity, so the existing hook returns layer 50. `text_encoder_device: auto` spreads it over both 16 GB cards (and CPU) for the one-time caching job. |
| Timestep shift | trainer.py:240-241 (`s*sigma/(1+(s-1)*sigma)`, the same formula as H3's), :243-245 (1000-grid snap) | `timestep_shift 12` with uniform sampling reproduces musubi's base-time draw. The focus band is `sigma_min/max` plus a mixture in `reference_sigmas`. |
| Flow target | trainer.py:966 `target = noise - latents` | H3's forward returns -v (`x0 = x_t + sigma*v`), so the trainer target is unchanged. |
| Training adapter | config.py:58-61 `ARCH_TRAINING_ADAPTER`, lora.py:902 `attach_training_adapter`, lora.py:237, 394 (on while training, off while sampling, never saved) | Add a `"minimax_h3"` entry pointing at circlestone. It is a live branch, so it works on int8/NVFP4 without the merge that musubi needs. |
| Turbo / sample tables | config.py:34 `ARCH_FULL_SAMPLE`, :43 `ARCH_TURBO`, qwen21/sampler.py:20 `TURBO_SIGMAS` (pinned-sigma pattern) | `ARCH_TURBO["minimax_h3"]` = larryvrh turbo v4 step600 ema, 6 steps, cfg 1, strength 0.75; full sampling at cfg 1. |
| Caches, buckets, probes, race, EMA, export | `.rulz_cache`, size_multiple buckets, likeness probes, race.py, ema.py, comfy LoRA export | Unchanged. ComfyUI loads bare `diffusion_model.blocks.N.attn.qkv_proj.lora_A/B` H3 keys untranslated ([ComfyUI lora.py](https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/lora.py)), so export is simpler than FIBO's split. |

#### What is new

- **`rulztrainer/minimax_h3/model.py`** (modelled on krea2/model.py and qwen21/model.py). It needs:
  - 50 blocks plus the 2-block token refiner;
  - fused `qkv_proj` (one LoRA target group `qkv` instead of q/k/v);
  - fused `mlp.fc1` (gate+up, 2 x 14336);
  - 56x128 MHA with QK-RMSNorm through the existing SDPA path, and 3-axis MM-RoPE (16 freqs per axis, 96 of 128 dims; time axis in 1/40 s);
  - a packed-sequence builder: text rows, 1-frame video rows (24 ch, 1x2x2 patch -> 96 -> 5376 via `video_patch_proj` in fp32), and optionally the audio rows;
  - the 3-slot AdaLN index (row `t*3 + tag`; video 0, text 1, audio 2);
  - pruned curve mode: an fp32 `adaln_t_table` [1025, 8], interpolated at t = 1 - sigma and clamped, feeding the 9-wide `adaln_proj`, kept in fp32 and never LoRA-wrapped;
  - the fp32 `final_layer.video_out` head;
  - a return of -v.
- **`minimax_h3/loading.py` and `single_file.py`:** Comfy key layout (`blocks.N.attn.qkv_proj`, `adaln_proj.linear`, `token_refiner.blocks.N`, `condition_proj`, `video_patch_proj`, `final_layer.video_out`). They must detect pruned versus full checkpoints. For a full checkpoint, either refuse it or run musubi's load-time rank-8 SVD (`--prune_adaln`).
- **VAE wrapper:** wrap diffusers' AutoencoderKLMiniMaxH3 (already in the venv). Encode one frame to [24, 1, H/16, W/16] with per-channel mean/std. Decode by duplicating the latent on the time axis and keeping frame 0. The 36-layer attention decoder (~4.9 GB) loads only for samples and probes.
- **Audio rows:** musubi keeps a constant silent audio latent masked out of the loss; ai-toolkit drops audio rows for image datasets. The P27 maintainer says to keep them until measured, because dropping them changes the attention distribution. The parity test below should decide.
- **Region maps:** the latent cell is still 16 px (H/16), so the region-weighted loss, face cells and the likeness code keep working on the latent grid. Only the token grid is 32 px, so attention localisation and alignment taps must map 2x2 cells to one token, as Qwen 2.1's 32-px grid already does (archs.py:8-9).
- **Prompt template:** H3 feeds Qwen3-VL-32B "with special tokens". The exact template must be copied from the diffusers or ComfyUI code; it is not in the input.
- **Not needed for v1:** the guidance loss (use the adapter), TE layer streaming (accelerate dispatch over both cards is enough for one caching pass), the nvfp4_awq TE reader, w6a8, the Ref2VA reference path, TREAD, and audio/voice training.

#### File formats and key names

- **Transformer (Comfy-Org, per variant):**
  - bf16: 66.3 GB
  - int8_convrot: 34.0 GB
  - pruned_bf16: 40.2 GB
  - pruned_fp8_scaled: 21.0 GB
  - pruned_int8_convrot: 21.0 GB (19.53 GiB)
  - pruned_w6a8: 16.0 GB
- **Community NVFP4:** pruned NVFP4 at 11.67 GiB (coolthor). 200 linears packed 4-bit with E4M3 block scales and an FP32 global scale. Whether quant.py parses it is unverified.
- **Text encoder `qwen3vl_32b_minimax_h3`:** bf16 51.5 GB, int8_convrot 27.1 GB, nvfp4_awq 15.7 GB; 50 layers, final norm removed.
- **VAEs:** video VAE ~4.9 GB (fp16 or int8_convrot); audio VAE fp32 0.6 GB.
- **LoRA keys:** `diffusion_model.blocks.N.attn.qkv_proj.lora_A/lora_B.weight`, alpha = rank. ai-toolkit LoRAs on pruned bases may also carry `adaln_proj` lora_A of shape [rank, 8], which a RulzTrainer loader must tolerate or skip ([multimodalart/MiniMax-H3-Pruned](https://huggingface.co/multimodalart/MiniMax-H3-Pruned)).

#### Can it train stills on 16 GB at all, and how fast?

**Feasibility: yes, with streaming.**
- **Weights:** the pruned int8 transformer is 21 GB, about 0.42 GB per block. Fizgig's auto planner runs int8 with about 36 of 50 blocks streamed at ~15 GB free. musubi swaps up to 48 of 50.
- **RulzTrainer on one card:** block_swap 36 needs about 15 GB of pinned RAM for the transformer. Add ~16-27 GB for the text encoder during caching (CPU part of the dispatch). Musubi's 64 GB host-RAM floor is the realistic minimum.
- **Two cards:** model_split puts ~10.5 GB of blocks on each card with no streaming. It then occupies both GPUs, and the 4060 Ti forces int8 (no FP4).
- **NVFP4 pruned (11.67 GiB):** on the 5060 Ti alone it needs only a few swapped blocks.

**Speed: unmeasured in RulzTrainer.** These are the parts, from the input plus my arithmetic.
- **Compute.** Per token, H3's block linears hold about 19.3B weights. Per block: qkv 5376x21504, out 7168x5376, fc1 5376x28672 and fc2 14336x5376, about 385M, from the G71 config. Krea 2 has ~12B. H3 has 4x fewer image tokens at equal megapixels, so linear FLOPs per step at 1 MP come to roughly 19.3x1024 / (12x4096), about 0.4x Krea 2. That counts image tokens only and is my arithmetic; text length adds to both.
- **What slows H3 down.** It runs int8 W8A8, not FP4, on the 4060 Ti and by default on the 5060 Ti. Block swap turns torch.compile off ("block swap streams the blocks' weights in and out", trainer.py:880-881). The G71 and P27 verifiers assumed compile carries over; it does not with block swap.
- **Streaming.**
  - The verifiers put ~2x14 GB of H2D per step at ~1-1.5 s on the 5060 Ti (PCIe 5.0 x8) and ~2-3 s on the 4060 Ti (PCIe 4.0 x8), partly hidden behind compute.
  - With lora_blocks 20-49, swapped blocks 0-19 run forward-only and are not recomputed. That should cut the traffic to about (20 + 2x16) x 0.42 = ~22 GB per step for block_swap 36. This is my reading of checkpointing.py:35 and blockswap.py; confirm it with a profile.
- **Guard.** The adapter branch costs a few percent. The guidance loss instead adds ~50% compute plus a third streaming pass.
- **Estimate, unmeasured.**
  - 5060 Ti, int8, ~36 swapped blocks, adapter on: roughly 5-12 s/step at 0.5 MP and 9-18 s/step at 1 MP.
  - 4060 Ti: somewhat slower.
  - NVFP4 on the 5060 Ti with few swaps should be faster than int8.
  - A 2000-step run would take ~3-7 h at 0.5 MP on the 5060 Ti.
  - For comparison, the verifiers estimated 8-15 s at 0.5 MP and 15-30 s at 1 MP (2-4x Krea 2 nvfp4's ~9 s and 6-10x Qwen 2.1 int8's ~2.5 s). They assumed Krea-like token counts. My lower figure comes from the 32-px token, but the small sequences (300-1300 rows) may use the GPU less efficiently, so the first job after a port is a one-config timing run.
- **Probes** are also 33B forwards under swap at 6 Turbo steps, so probes and races cost several times what they cost on Krea 2 (G71 verifier).

#### Hardware tiers: local 16 GB and vast.ai

Larger cards remove the 16 GB problem. From about 30 GB free, the pruned int8 file runs with no block swap (Fizgig), so torch.compile and checkpoint_keep_blocks work again. From 48 GB, a normal run with no swap is possible (G71 verifier).

**Multi-GPU in RulzTrainer.** It runs in threads in one process and never uses torch.distributed (rulztrainer/multigpu.py:1-14), so NCCL on vast.ai changes nothing. There are two ways to use more GPUs:
- **data_parallel:** every GPU holds the whole model.
- **race_parallel:** one race configuration per GPU.
Both fit H3 only where one card holds the whole model; block_swap excludes both (config.py:589-594).

**Native FP4.** It needs compute capability 10.x or 12.x (quant.py:836-844): RTX 50, RTX PRO 6000 Blackwell and B200. A100, H100 and H200 run int8 W8A8 or bf16.

**Sources for the "no swap" figures** in the table: Fizgig's tier table, musubi's memory table and the Comfy-Org file sizes.

**Step times.** No source gives an H3 still-image training s/step for any named GPU, so every s/step below is unmeasured unless stated. The only named-GPU H3 timing is coolthor's RTX 5090 inference: 243 frames at 864x480, 10 steps, NVFP4 175 s / 26,914 MiB against int8 185 s / 28,581 MiB. musubi's 12.7 s/it (classic swap 48) and ~4 s/it (H2D-only, swap 32) are video training on an unnamed GPU. Per musubi, stills are "far cheaper than video steps"; per Fizgig, a 22-frame clip costs about 7x a still.

| Tier | Transformer | Block swap / split | Text encoder for caching | Expected s/step (stills) | Notes |
|---|---|---|---|---|---|
| 16 GB local (5060 Ti, 4060 Ti) | pruned int8_convrot 21 GB; or pruned NVFP4 11.67 GiB on the 5060 Ti only | block_swap ~36 (Fizgig's ~15 GB tier); NVFP4 with ~10 swapped (G74); or model_split over both cards, int8, no swap | int8_convrot 27.1 GB through `text_encoder_device: auto` (dispatch over both cards and CPU) | Unmeasured. My estimate for int8 + ~36 swapped on the 5060 Ti: ~5-12 s at 0.5 MP, ~9-18 s at 1 MP. Slower on the 4060 Ti. | torch.compile off under block swap (trainer.py:880-881). Pinned RAM ~15 GB; 64 GB host RAM floor (musubi). Start at 0.25 MP. |
| ~22-24 GB (Fizgig's middle tier; a 24 GB vast card) | pruned int8_convrot | ~14 blocks streamed (Fizgig) | as above, or nvfp4_awq 15.7 GB if a reader is added | Unmeasured | Still under block swap, so no compile and single-GPU only. |
| 32 GB (RTX 5090) | pruned int8_convrot with no swap (Fizgig: ~30 GB free -> int8 without swap up to 1 MP); or pruned NVFP4 with native FP4 and large headroom | none | int8_convrot 27.1 GB needs about 31.4 GB free under `place_text_encoder`'s one-card rule (size x 1.1 + 1.5 GiB, fibo/loading.py:102), which a 5090 just misses, so it dispatches with CPU spill (one-off); nvfp4_awq 15.7 GB fits once a reader exists | Unmeasured. NVFP4 vs int8 was ~5% faster with 1.6 GB less peak in coolthor's 5090 inference. | Compile and keep blocks on. 1 MP practical. Multi-5090 boxes: data_parallel or race_parallel with a full int8 (or NVFP4) copy per GPU. |
| 48-96 GB (48 GB cards, RTX PRO 6000 96 GB) | 48 GB: pruned int8 with no swap (G71 verifier: "a normal run"), or pruned bf16 40.2 GB with little headroom. 96 GB: pruned bf16 (no quantization error), or NVFP4/int8 for speed on the PRO 6000 (FP4-capable). | none | 96 GB: bf16 TE 51.5 GB fits for caching and is then unloaded; 48 GB: int8_convrot 27.1 GB | Unmeasured | This is the tier for the H3-vs-Krea 2 vs Qwen 2.1 likeness race in the plan. The guidance-loss alternative (+~50%) is affordable here. Full fine-tune is possible (lr 3e-5). |
| 80+ GB datacenter (A100/H100 80 GB, H200 141 GB, B200) | pruned bf16 40.2 GB or int8 W8A8 (A100/H100/H200 have no FP4); B200: NVFP4 native. Full bf16 66.3 GB fits on 80 GB+, but pruning does not affect attention/MLP LoRAs, so it is not needed. | none | bf16 51.5 GB | Unmeasured | Multi-GPU through data_parallel threads or race_parallel; NCCL is not used. Clip and voice LoRAs and full fine-tune belong here (clips cost ~7x to ~37x a still, Fizgig). Linux/triton for W8A8 is the standard path. |

**Which tier for what.**
- **First likeness verdict:** use the 48-96 GB tier. One card, no swap and compile on gives the cleanest s/step and probe curves.
- **Character stills:** the 5090 tier is enough.
- **16 GB local:** tune it only if H3 wins on the probes.
- **Multi-GPU boxes:** run a parallel race, one configuration per GPU, rather than data_parallel. The user's workload is batch 1.

#### Risks, implementation path and tests

**Fit and risks.**
- **block_swap exclusions:** it excludes model_split, data_parallel, race_parallel-style replicas and finetune (config.py:589-599), so a race on one card runs configurations one after another.
- **Distilled base:** turbo_lora "auto" must not stack a raw-to-turbo LoRA on a distilled base. That is the same latent bug the G50 verifier found for Krea Turbo (comfy_files.py:42-43).
- **Sign conventions:** the velocity sign and t = 1 - sigma are easy to invert silently, so the parity test comes first.
- **One-frame training** is "outside the released training distribution" (musubi). diffusion-pipe and Fizgig report that image-only training degrades motion and video knowledge; that is an author assertion with no data, and it is irrelevant for stills.
- **Windows:** no blocker. No NCCL is needed, and triton-windows already runs the W8A8 kernels.
- **Licence:** see the facts above.

**Implementation path.**
1. Pre-port gate, no trainer code: take an H3 character LoRA trained in ai-toolkit and a Krea 2 LoRA from RulzTrainer on the same dataset. Render the same probe prompts and seeds in ComfyUI and score both with RulzTrainer's face/body scorers. That needs a small script around facesim.py / bodysim.py; I did not check how their APIs take external images. Continue only if H3 is ahead.
2. `rulztrainer/minimax_h3/{model.py, loading.py, single_file.py, sampler.py}`, `MiniMaxH3Arch` in archs.py, and entries in `ARCH_TRAINING_ADAPTER`, `ARCH_TURBO`, `ARCH_FULL_SAMPLE`, `ARCH_DEFAULT_TARGETS` and the comfy_files download list.
3. TE caching through the existing Qwen3-VL builder with a 32B/50-layer config.
4. VAE wrapper around the venv's AutoencoderKLMiniMaxH3.
5. Wizard: a minimax_h3 preset per the recipe, a warning about RAM and swap, and a hard floor of block 6 on lora_blocks.

**How to test.**
1. Parity: a 1-frame forward on a cached latent must match ComfyUI's MiniMaxH3Model, or diffusers' transformer for the unpruned path, within int8 tolerance.
2. Pruned modulation against full: maximum relative error per block of 1e-3 or less across the 1000 grid sigmas.
3. Export: a RulzTrainer LoRA loads in ComfyUI's native H3 workflow and changes the output.
4. Race on vast.ai (48 GB: pruned int8, adapter, no swap): H3 against Krea 2 nvfp4 and Qwen 2.1 int8, same 30-40 image set, rank 16, 0.5 MP, 2000 steps, same seeds. Probe ArcFace face and DINOv2 body every 200 steps and log wall-clock.
5. Adopt H3 for stills only if its best probe beats the best of Krea 2 / Qwen 2.1 by at least 0.02 ArcFace at equal wall-clock. Otherwise keep H3 in ai-toolkit.
6. Only then run one 16 GB timing run (block_swap 36, int8; then nvfp4 on the 5060 Ti) to record s/step and peak VRAM.

### Pruned checkpoints and the 16 GB memory plan (G74)

**Do it (as part of the port). Expected gain: the pruned file is the prerequisite for 16 GB, at -26 GB bf16 / -13 GB int8 (66 -> 40 GB, 34 -> 21 GB). Confidence: high (0.75). Cost: included in the port.**

**What it is.** H3's 50 per-block AdaLN projections are Linear(2688 -> 96,768), about 13.04B parameters. They depend only on t. Comfy-Org's pruned files replace them with an fp32 `adaln_t_table` [1025, 8] interpolated at t = 1 - sigma, plus 9-wide projections, which leaves 43.6M parameters. musubi does the same at load time with a mean-centred rank-8 SVD plus a constant channel.

**Evidence.**
- [musubi advanced](https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/docs/minimax_h3_advanced.md): the table, -26 GB, TE streaming at 0.9/0.5/0.3 GB per layer (bf16/int8/nvfp4), classic swap 12.7 s/it against H2D-only ~4 s/it, and a 64 GB host-RAM floor.
- [coolthor NVFP4](https://huggingface.co/coolthor/MiniMax-H3-pruned-NVFP4): 13,039,369,728 -> 43,642,368 parameters. RTX 5090 inference at 864x480, 243 frames, 10 steps (not 4): NVFP4 175 s / 26,914 MiB against int8 185 s / 28,581 MiB. FP4 needs sm_120 for the speedup.
- Quality of the curve fit: dipjeb reports a rank-8 fit residual of 9e-5 ([dipjeb](https://huggingface.co/dipjeb/MiniMax-H3-Turbo-LoRA-Pruned)). multimodalart reports cosine 0.9958 against 0.9992 with a turbo LoRA ([multimodalart](https://huggingface.co/multimodalart/MiniMax-H3-Pruned)).
- Fizgig's tiers: ~30 GB int8 with no swap up to 1 MP; ~22 GB with ~14 blocks streamed; ~15 GB with ~36 streamed; 12 GB or less on 4-bit (HQQ 4.8% / NF4 9.5% base error). One-way swap is 6.4x faster than round-trip.

**RulzTrainer today.**
- ComfyUI nvfp4 files and FP8 block scales already load (quant.py:1058-1074).
- The per-family auto quant table exists (krea2/loading.py:39, 73-84).
- H2D-only block swap exists (blockswap.py:54).
- text_encoder_quant int8 / int8_convrot / nvfp4 exists (config.py:200).
- Missing: an H3 row in the auto table that also returns a block-swap count, and TE layer streaming (optional; accelerate dispatch covers caching on two cards).

**Fit and risks.**
- Pruned AdaLN is time-only, so LoRAs on attention and MLP are interchangeable between pruned and full bases.
- "No trainer LoRA-trains AdaLN, so all LoRAs are interchangeable" is false in general. ai-toolkit stores [rank, 8] `adaln_proj` LoRA tensors on pruned bases, and full-model turbo LoRAs fail to reshape their 51 `adaln_proj` patches on pruned bases (dipjeb).
- A turbo LoRA used for probes must therefore be checked for `adaln_proj` keys.
- The 1000-grid lines up with the 1025-row table by interpolation. That is not exact, and it does not need to be.

**Refuted inside this item.**
- Transferring AdaLN pruning to Krea 2, MOSS, FIBO or Qwen 2.1 gains nothing; the code shows their modulation is shared or small (krea2/model.py:255, 309; moss/model.py:177; qwen21/model.py:256; FIBO 44 x Linear(256 -> 4x3584), about 161M parameters, fibo/model.py:32, 202). Do not add an `adaln_prune` option for them.

**Implementation path.** In the H3 loader, read the curve mode directly. Add an H3 row to the auto table:
- 30 GB or more: int8, no swap.
- ~22 GB: int8, 14 swapped.
- ~15 GB: int8, 36 swapped, or nvfp4 with ~10 swapped on sm_120.
Default the bucket area to 0.25 MP and step it up with probes.

**How to test.**
1. Modulation check against the full model.
2. One-denoise-step cosine of 0.999 or more between pruned int8 and full bf16.
3. On the 5060 Ti: pruned int8 with 36 swapped against pruned nvfp4 with the minimum swap, both at 0.25 MP, rank 16, 2000 steps, probes every 250 steps. Pick nvfp4 only if its likeness is within 1 point and it is faster.

### De-distillation guard: live training adapter, not guidance loss (cross-ref G50, G48)

**Do it (mandatory for H3). Expected gain: prevents structural collapse within ~50 steps (musubi). Fizgig: 50% likeness seven epochs sooner and a higher peak with the Ostris adapter versus none (one A/B, no GPU or dataset named). Confidence: medium (0.65). Cost: one table entry plus key renaming, once the family exists.**

Full write-ups: theme items "De-distillation training adapters (Krea Turbo, H3, own adapters)" (G50) and "Guidance-loss / CFG-augmented training target" (G48). What is specific to H3:

- **Plain flow matching is not offered.** musubi's docs say plain flow training "is therefore not offered as a recipe"; one-frame drift appears within ~50 steps without a guard.
- **The three guards:** a training adapter (no extra forward), the guidance loss target = uncond + s(v - uncond) with s = 3-4 (4 "more reliable for longer runs"), sigma_min 0.15, ~1.5x per step), or teacher matching (+1 forward).
- **Adapters:**
  - circlestone: 620,284,872 B, rank 64, ~10k steps on 10k images, "significantly better" than Ostris for image-only training, and it biases the model toward images.
  - Ostris: v1 155 MB, v2/v3 310 MB, with ref2va and FastH3 variants ([Ostris tree](https://huggingface.co/api/models/ostris/minimax_h3_training_adapter/tree/main)).
  - Fizgig A/B: circlestone gave sharper, cleaner photo LoRAs; Ostris learned a video look about 3x faster.
- **RulzTrainer's training adapter is already the right design.** It is a live frozen branch (lora.py:237, 394, 902) that works on int8/NVFP4 without merging. musubi has to merge into a bf16 base and rejects int8; ai-toolkit keeps it live because a merge "would resample every int8 scale". In this design the guidance loss is the worse choice for H3: +50% compute plus a third block-swap transfer per step.
- **Extra forwards must keep the adapter on.** The region prior and DOP passes run with extras toggled, so check that their base predictions keep the adapter on. Otherwise they compare against the distilled base (G50 verifier).
- **Gaps from G50 that also apply here:** write the adapter name into the saved LoRA's metadata, and make training_adapter a per-config race key (README.md:1244 says races share it).
- **How to test:** after the port, race adapter circlestone against Ostris v3 against none against guidance loss 4 / sigma_min 0.15. 0.25 MP, rank 8, lora_blocks 20-49, 1000 steps, probes every 100 steps. "none" should collapse within ~50-200 steps if the claims hold. Pick the adapter with the best smoothed face probe; drop guidance loss unless it beats it by at least 2 points.

### Probing and sampling stills (G75)

**Do it (as part of the port). Expected gain: correct probe decodes and 3-5x fewer DiT evaluations per probe with Turbo at 6 steps. The shift for stills is open. Confidence: medium (0.55). Cost: ~1 day inside the H3 sampler.**

**What it is.** H3's VAE decodes a lone 1-frame latent poorly ([diffusion-pipe notes](https://raw.githubusercontent.com/tdrussell/diffusion-pipe/main/docs/minimax_h3_notes.md)). musubi duplicates the latent into a pseudo two-latent clip and keeps pixel frame 0, within about 1-2 dB of a true two-latent decode. The scheduler's constant shift of 12 was tuned for clips of ~30k tokens, against 256-1024 tokens for a still.

**Evidence.**
- Fizgig previews use 6 steps instead of 20, with Turbo at 75% on top of the training LoRA. 16 GB previews are capped at 768x640x22 frames.
- [awesome-minimax-h3-integration](https://github.com/MiniMax-AI/awesome-minimax-h3-integration): turbo v4 step600 ema (744 MiB, larryvrh) is the recommended default; "At 4 steps the audio track can degrade along with fast motion; 6–8 steps helps both".
- [Pika H3 Flex Hybrid](https://huggingface.co/hyukudan/Pika-H3-Flex-Hybrid-b30-49): six-step sigmas 1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0. This is exactly shift 12 on base times 1, .75, .5, .25, .1, .04.
- [FastH3](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-fastvideo) (DMD2, 8 steps, shift 10/3) needs VSA gate weights and sparse-attention kernels that do not exist on triton-windows.
- [PTD](https://alan-lanfeng.github.io/PTD/): 4/6-step LoRA preferred over the LightX2V DMD LoRA in 66% of votes, measured on 8xH100.

**Corrections.**
- The turbo LoRA is larryvrh's, not LightX2V's.
- Fizgig's 4-step loop belongs to Repair Studio, not the previews.
- A 5-frame probe render is unnecessary given musubi's duplicate decode.
- No source validates video turbo LoRAs on 1-frame renders. Like the MOSS DMD LoRA, it may understate likeness (config.py:38).

**RulzTrainer today.**
- turbo_lora / turbo_lora_strength / sample_lora_strength (config.py:319-323) and probe_steps (config.py:434) already exist.
- Pinned turbo sigmas exist for Qwen 2.1 (qwen21/sampler.py:20).
- A global timestep_shift on the 1000 grid exists (trainer.py:236-245).
- race_configs can race timestep_shift rows.
- No new options are needed.

**Implementation path.**
- H3 `decode_latents`: duplicate on T and keep frame 0.
- `sample_sigmas`: the constant-shift formula, with no mu.
- `reference_sigmas`: uniform base time through an image shift. Default it to ~1-2 per diffusion-pipe, documented as unvalidated.
- `ARCH_TURBO["minimax_h3"]`: larryvrh v4 step600 ema, 6 steps, 0.75, cfg 1.

**How to test.**
1. Decode check on 20 fixed renders: solo, duplicate and true two-latent decodes. Adopt the duplicate decode if it is within 1 likeness point of two-latent.
2. Turbo probe fidelity: Turbo at 6 steps with strength 0.75 and 1.0 against 30-step renders at checkpoints 0/500/1000/1500. Accept if Spearman rho is 0.8 or more and the best-checkpoint pick agrees.
3. Shift race for stills: timestep_shift 1/2/4/8/12, 1500 steps, probe_every 100, drop-lowest with patience 3. Confirm the winner on 2 seeds. Keep the diffusion-pipe default if the spread stays inside likeness_noise_band.

### H3 recipe settings: block map, optimizer, EMA, TREAD (G72, cross-ref P21)

**Do it (block map as the H3 preset); race the optimizer claim on Krea 2 / Qwen 2.1 now; skip TREAD for H3 stills. Expected gain: about 40% less backward compute from training 30 of 50 blocks. Likeness claims unquantified. Confidence: medium (0.6). Cost: preset only.**

- **Block map** ([Fizgig](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/MINIMAX_H3.md)):
  - 20-49 "reaches higher likeness sooner with quicker steps". 6-49 "is not a likeness upgrade" and "preserves less of the model's priors".
  - Blocks 0-5 are never trained; the refiner stays off; voice uses 38-48.
  - The [Pika hybrid](https://huggingface.co/hyukudan/Pika-H3-Flex-Hybrid-b30-49) splices FL2VA blocks 0-29 with Ref2VA blocks 30-49 and keeps multi-reference conditioning. That is consistent with late conditioning, but nobody tested it separately.
  - lora_blocks is pure config (config.py:239) and composes with checkpoint_keep_blocks and block swap.
  - Do not port the 20-49 map to other models by analogy. Fizgig's Qwen 2.1 map is 10-14 (README.md:1672).
  - The transferable part is a per-bank block profiler: zero a 5-block bank's LoRA scale, render a turbo probe, score it. Its full write-up belongs with the theme items, not here.
- **Optimizer and EMA:**
  - EMA 0.98 is banked as the default (README.md:583-597).
  - Fizgig's fp32 adamw "single biggest likeness gain on H3" is unquantified.
  - Plain adamw exists (trainer.py:262), and race.py:50-68 already varies optimizer and lora_ema. Race adamw 1e-4 against automagic3 on Krea 2 / Qwen 2.1 today, with no code.
- **Other H3 numbers:**
  - rank 8 / 50 epochs, or 16 / 60 epochs;
  - 0.25 MP;
  - caption dropout 0.05;
  - musubi example: rank 16, adamw8bit 1e-4, 16 epochs;
  - diffusion-pipe example: rank 32, adamw_optimi 1e-4, shift 8, image_shift 1, cfg 4;
  - [prithivMLmods face LoRA](https://huggingface.co/prithivMLmods/MiniMax-H3-Facial-Realism-CloseUp): rank 16, AdamW 1e-4, 2000 steps, checkpoints 1800-2000;
  - correction: the quote "1 is way too low for videos" is not in the diffusion-pipe toml.
- **TREAD for H3 stills: skip.**
  - Fizgig routes only clip steps (half the video tokens dropped at block 2, restored at block 47) and always runs stills and previews in full.
  - SimpleTuner reports a higher initial LoRA loss.
  - At 256-1024 image tokens per still, there is little attention to save.
  - The paper's 1.86 -> 3.03 it/s is from-scratch ImageNet DiT on 4x A100.
  - General treatment: theme item "TREAD token routing around the middle blocks during LoRA training" (P21).
- **How to test (H3, after the port):** race lora_blocks 20-49 against 6-49 at equal wall-clock, 1000 steps, probes every 100. Expect 20-49 to be faster per step; adopt 6-49 only if it leads by more than 2 smoothed points over 3 probes.

### Teacher matching from the reference pathway (cross-ref G73)

**Watch. Expected gain: unmeasured (musubi publishes no likeness numbers); +~50% step time, more with references. Confidence: medium (0.4). Cost: large (needs a reference path).**

Full write-up: theme item "Privileged-teacher distillation from the reference pathway" (G73). What is specific to H3:
- **What it is:** musubi's `subject_ref` teacher, the same transformer with the LoRA off and other photos of the subject as references. It is "the only teacher-matching mode available with --one_frame" and one of three mandatory guards for one-frame training.
- **Loss and gates:** (||p||-||t||)^2 + 2||p||||t||(1-cos), mag_weight 0.25-0.5, dc_weight 0.3. sigma_max stays at 1.0, because 0.75 "keeps the student from learning identity"; sigma_min 0.15.
- **Recipe:** lr 3e-4, 50 warmup, ~500 steps, rank 16, validated on one 20-image one-frame set.
- **Constraint:** it cannot be combined with the guidance loss.
- **Cost on 16 GB:** two block-swapped passes per step with a longer teacher sequence.
- **Order:** an H3 port should not start here. It becomes interesting only after the adapter route is measured, and on Qwen 2.1 or Flux 3 Klein first (see the theme item).

### RefMods: optimised reference conditioning (cross-ref G76)

**Watch. Expected gain: +7 points on Fizgig's scale (28 vs 21) on one unseen look, against a plain encode and never against a LoRA. Confidence: low (0.3). Cost: needs the Ref2VA path.**

Full write-up: theme item "Optimised reference conditioning (RefMods)" (G76).
- **What it is:** textual inversion on the reference latents of H3's Ref2VA/FL2VA path, 200 steps against the frozen model ([Fizgig REFMOD_HOWDOI.md](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/REFMOD_HOWDOI.md)).
- **Measured:** equal to the plain encode on in-distribution shots; 60 vs 65 when a ref2va mod runs on fl2va.
- **Tempering evidence:** comfyui-wiki says "for maximum likeness and generalisation, LoRAs are still king".
- **Format:** RefMods are an H3-only format for the ComfyUI-MiniMaxH3Mod pack.
- **For RulzTrainer:** irrelevant until a Ref2VA path exists, and the input gradient must flow through every block, which defeats lora_blocks.

### Full fine-tune of H3 (cross-ref G16)

**Skip-ish. Expected gain: none for character LoRAs. Confidence: medium (0.6). Cost: n/a.**

Full write-up: theme item "Full fine-tune memory: CPU optimizer, optimizer-in-backward, component windows" (G16).
- **Fizgig's H3 numbers:** component windows, Adafactor plus optimizer-in-backward (saving 5.2 GB), 8.8-12.3 GB peak on 16 GB, lr 3e-5, pruned int8 required.
- **RulzTrainer:** finetune.py already does component windows and Adafactor. But `block_swap` and `finetune` are mutually exclusive (config.py:595-596), and finetune excludes torch.compile. An H3 fine-tune on the user's box would therefore need model_split over both cards to hold the 21 GB base.
- **If ever:** cap lr at 3e-5.

### AdaLN / modulation targets (cross-ref P14)

**Skip-ish. Expected gain: none for identity. Confidence: low (0.3-0.35). Cost: n/a for H3.**

Full write-up: theme item "DiffFit-style modulation / bias / norm targets" (P14).
- **The 13B question:** H3's 13B AdaLN parameters are a time-plus-modality-only path, so a delta there cannot carry the trigger or the subject.
- **Gradient flow:** in ai-toolkit the raw-weight call means a LoRA gets no gradient (README.md:729).
- **Pruned bases:** the projection becomes 9-wide, so there is almost nothing left to adapt.
- **Recommendation:** keep `adaln_proj` excluded.

## Appendix A. Rejected ideas

Dropped by the verifiers. "Fit refuted" means the idea fails for this trainer and workload (already present, impossible, or pointless), not that the source is wrong. Ideas dropped only because of the local 16 GB cards are revisited in [Bigger hardware on vast.ai](#bigger-hardware-on-vastai).

| Idea | Why dropped | Verifier note (first sentences) |
|---|---|---|
| **G4** FP8 compute path (Ada and Blackwell), FP8 files, FP8 dgrad | sources weakened, fit refuted, already_in_rulz partially | Benchmarked on the 4060 Ti: RulzTrainer's INT8 W8A8 path is as fast as or faster than FP8, `use_fast_accum` gives no measurable gain, and row-wise `_scaled_mm` is unsupported on sm_120. Several cited numbers were misattributed (AD103 figures quoted as AD104; Kohaku's 10-20% is a B300 projection). Revisited for H100/B200 in [Bigger hardware on vast.ai](#bigger-hardware-on-vastai). |
| **G82** Inductor NVGEMM block-scaled GEMMs | sources weakened, fit refuted, already_in_rulz no | The quoted release-note text is genuine. Two small misattributions: "small-M and block-scaled" is in 2.14, not 2.15, and the profiling-configs env var is in config.py, not the 2.15 notes. |
| **G29** Latent-space identity reward head | sources weakened, fit refuted, already_in_rulz partially | Three of the finding's premises are wrong for this codebase: (a) The reward stage has no differentiable decode: renders are no_grad and the scorers are non-differentiable by design (reward.py:18-19, :110). (b) The README measures the identity-loss decode at about 0.6 GB and +7% s/it, not "several GB". |
| **G40** Flip and colour jitter policy for faces | sources weakened, fit refuted, already_in_rulz partially | Both citations are real and were reported faithfully, but the generation paper is qualitative and SDXL/UNet-based. The proposal mostly restates RulzTrainer's current defaults. |
| **G14** CARE-LoRA compressed LoRA-down activations | sources weakened, fit refuted, already_in_rulz no | Sources are real and the numbers are accurately quoted. The finding glosses three points: the SD3 baseline was LoRA r8, not r16; the setting had no checkpointing at batch 4; and the dA approximation is only above 0.6 cosine. |
| **G20** CUDA caching-allocator settings | sources weakened, fit refuted, already_in_rulz partially | Key correction: two of the three proposed settings do nothing in RulzTrainer's setup, which the allocator source confirms (garbage_collect_cached_blocks needs allowed_memory_maximum from set_per_process_memory_fraction; max_non_split_rounding only applies at or above max_split_size, which is unlimited by default). The finding also misreads train.py: expandable_segments is set on Linux only (train.py:16), so... |
| **G52** Krea 2 per-bucket shift | sources weakened, fit refuted, already_in_rulz yes | Both deltas the finding proposes are already in the code: the per-batch shift from the bucket's token count, using Krea's exact mu formula, and the Turbo pin at mu 1.15. The finding's premise that RulzTrainer's shift is a run-level setting is wrong for timestep_sampling: reference. |
| **G56** Qwen 2.1 texture collapse as a low-noise failure | sources refuted, fit refuted, already_in_rulz yes | Key code evidence comes from musubi-tuner main. - src/musubi_tuner/training/trainer_base.py: qwen_shift sets mu = get_lin_function(x1=256, y1=0.5, x2=8192, y2=0.9)((h//2)*(w//2)), then shift = math.exp(mu). |
| **G6/member 37** HALO-2-style left-hand Hadamard on grad_out (plus stochastic rounding) to make the INT8 frozen-layer backward safe by default | member refuted inside a kept group | HALO's INT8 collapse (GSM8K 4.5) comes from per-tensor scaling with no SR. RulzTrainer's int8 backward already scales per token, already matches HALO-1 (rotated weight plus inverse Hadamard), and already runs in most user configs at about 1% error. |
| **G7/member 38** Quantization-residual LoRA initialization (LoftQ / SVDQuant / FALQON) for int8 and especially NVFP4 frozen bases | member refuted inside a kept group | The sources are real, but all the evidence is from LLMs, and FALQON's SVD init is a speed trick whose accuracy is at or below QLoRA. I measured it on Krea 2's real weights (CPU, the repo's own quantizers): the NVFP4 residual is nearly white. |
| **G12/member 123** Activation offload to pinned host RAM to buy a bigger token budget under block swap (KohakuMoss: +10-18% samples/s from larger micro-batches when weights are streamed) | member refuted inside a kept group | The KohakuMoss numbers exist but were misread. The 'balanced' rows were measured at 8192 tokens and vram_8g at 4096, so '14-18% slower at the same 4096-token budget' compares two different budgets. |
| **G31/member 63** Teacher-driven pseudo-labels and on-policy DMD-style distillation: identity editors (Qwen-Image-Edit, Flux 3 Klein multi-reference, InfiniteYou/PuLID) as teachers for the character LoRA | member refuted inside a kept group | The sources are real: CollectionLoRA 2605.25378 with its table numbers exact, InfiniteYou, and the Comfy Klein post dated 2026-10-15. The on-policy DMD half does not transfer. |
| **G34/member 71** Learned soft 'persona' tokens in the encoder's system/user turn, cached at caption time, optionally warmed up with an understanding (VQA) loss on the VLM | member refuted inside a kept group | The sources are real, but the claim is a stretch for this workload. UniCTokens' face similarity is worse than DreamBooth (0.334 vs 0.436). |
| **G37/member 70** Anchor phrase as the class noun and as the trigger's initialiser: a specific, well-represented demographic phrase instead of (or fused with) a fresh rare token | member refuted inside a kept group | The sources exist, but none shows an anchor gain for LoRA likeness. In FAD Table 3 the anchored 'Normal' DINO values equal the trigger-token values of Table 1, so no gain is shown, and the 0.6373-0.7840 range mixes Normal and FAD rows. |
| **G48/member 84** Guidance-aware training target (CFG-extrapolated velocity) so Raw-trained LoRAs land faithfully on the guidance-distilled Turbo | member refuted inside a kept group | The ai-toolkit fields exist as described, but the Krea anecdote (HF Krea-2-Turbo discussion #10, jessikat29, Aug 22 2026, 5060 Ti 16 GB) describes 'Differential Guidance 3.0', not the guidance loss. In ai-toolkit's SDTrainer.py that block sits inside `if do_guidance_loss:` (line 866 now, line 820 in the Aug 2026 version), and on its own it is target = pred + s*(target - pred). |
| **G66/member 127** Massive-channel-aware adapters: mask or penalize LoRA writes into MOSS's 8 gain-control residual channels (KohakuMoss concentration loss, weight 0.01, k=32) | member refuted inside a kept group | The cited numbers are misattributed. The 0.21-0.23 vs 0.32 top-32 share comes from a LoKr finetune of the already HEALED model, and 0.32 is its starting value, not an unpenalized run (the page says there is no with/without comparison). |

## Appendix B. Sources

476 distinct links cited in the sections, grouped by site. Each was opened by at least one scout or verifier.

**arxiv.org**

- <https://arxiv.org/html/2501.04765>
- <https://arxiv.org/html/2510.21986>
- <https://arxiv.org/html/2512.12386v1>
- <https://arxiv.org/html/2510.00206v1>
- <https://arxiv.org/abs/2411.05007>
- <https://arxiv.org/abs/2607.05711>
- <https://arxiv.org/html/2510.24061>
- <https://arxiv.org/html/2505.11594v3>
- <https://arxiv.org/abs/2505.11594v2>
- <https://arxiv.org/html/2601.22813>
- <https://arxiv.org/html/2509.25149v2>
- <https://arxiv.org/html/2501.02625v3>
- <https://arxiv.org/html/2512.02010v3>
- <https://arxiv.org/abs/2505.21136>
- <https://arxiv.org/html/2410.09426v4>
- <https://arxiv.org/abs/2512.03673>
- <https://arxiv.org/html/2509.23202>
- <https://arxiv.org/html/2607.21446>
- <https://arxiv.org/abs/2605.16732>
- <https://arxiv.org/html/2405.18392v3>
- <https://arxiv.org/abs/2508.01483>
- <https://arxiv.org/html/2509.01440v1>
- <https://arxiv.org/abs/2509.02046>
- <https://arxiv.org/html/2510.19376>
- <https://arxiv.org/html/2605.08734>
- <https://arxiv.org/html/2507.12142v2>
- <https://arxiv.org/html/2410.20625>
- <https://arxiv.org/html/2605.31484v1>
- <https://arxiv.org/html/2407.18242v3>
- <https://arxiv.org/html/2608.14492v1>
- <https://arxiv.org/html/2606.12921v1>
- <https://arxiv.org/html/2609.24678v1>
- <https://arxiv.org/html/2610.23374>
- <https://arxiv.org/html/2602.21545v1>
- <https://arxiv.org/abs/2402.12354>
- <https://arxiv.org/abs/2312.03732>
- <https://arxiv.org/abs/2502.16982>
- <https://arxiv.org/html/2505.12082v3>
- <https://arxiv.org/abs/2306.06101>
- <https://arxiv.org/abs/2508.12996>
- <https://arxiv.org/abs/2501.06842>
- <https://arxiv.org/html/2411.16085v4>
- <https://arxiv.org/abs/2510.12402>
- <https://arxiv.org/abs/2412.17107>
- <https://arxiv.org/html/2409.03137v2>
- <https://arxiv.org/abs/2502.02431>
- <https://arxiv.org/html/2410.00350v1>
- <https://arxiv.org/html/2411.09998>
- <https://arxiv.org/abs/2502.01235>
- <https://arxiv.org/html/2507.05964v2>
- <https://arxiv.org/html/2505.16531v2>
- <https://arxiv.org/abs/2405.17484>
- <https://arxiv.org/html/2603.06454v1>
- <https://arxiv.org/html/2606.08833>
- <https://arxiv.org/abs/2606.13433>
- <https://arxiv.org/html/2405.17403v3>
- <https://arxiv.org/html/2512.08785>
- <https://arxiv.org/abs/2502.07739v1>
- <https://arxiv.org/html/2503.18225v2>
- <https://arxiv.org/html/2505.14238v4>
- <https://arxiv.org/html/2505.12532v3>
- <https://arxiv.org/html/2507.05566>
- <https://arxiv.org/html/2406.12303v2>
- <https://arxiv.org/html/2505.18521v1>
- <https://arxiv.org/html/2406.08798v1>
- <https://arxiv.org/html/2506.05350>
- <https://arxiv.org/html/2606.27608v1>
- <https://arxiv.org/html/2508.04324v4>
- <https://arxiv.org/abs/2507.21802>
- <https://arxiv.org/html/2603.28718v1>
- <https://arxiv.org/html/2604.17415v4>
- <https://arxiv.org/html/2510.14255v4>
- <https://arxiv.org/html/2404.15449v1>
- <https://arxiv.org/abs/2505.17910>
- <https://arxiv.org/html/2412.05043v1>
- <https://arxiv.org/html/2605.16696v1>
- <https://arxiv.org/html/2412.09622>
- <https://arxiv.org/html/2505.16792v1>
- <https://arxiv.org/html/2606.08788v1>
- <https://arxiv.org/html/2509.16117v2>
- <https://arxiv.org/abs/2601.02036>
- <https://arxiv.org/html/2610.24897>
- <https://arxiv.org/html/2512.24138v1>
- <https://arxiv.org/html/2606.23950>
- <https://arxiv.org/html/2609.08936v1>
- <https://arxiv.org/abs/2609.00061>
- <https://arxiv.org/html/2412.03177v2>
- <https://arxiv.org/html/2604.01973>
- <https://arxiv.org/html/2507.02687>
- <https://arxiv.org/html/2505.03557v2>
- <https://arxiv.org/html/2402.12004v2>
- <https://arxiv.org/html/2506.06483v1>
- <https://arxiv.org/html/2605.01929>
- <https://arxiv.org/html/2503.10637v2>
- <https://arxiv.org/html/2503.16025v1>
- <https://arxiv.org/html/2502.01720v2>
- <https://arxiv.org/html/2504.02160v1>
- <https://arxiv.org/html/2605.25378v2>
- <https://arxiv.org/html/2605.10439v2>
- <https://arxiv.org/html/2310.02664v2>
- <https://arxiv.org/html/2311.04315v4>
- <https://arxiv.org/html/2606.23682>
- <https://arxiv.org/html/2608.14403>
- <https://arxiv.org/html/2410.23775>
- <https://arxiv.org/html/2504.01137v2>
- <https://arxiv.org/html/2406.11831>
- <https://arxiv.org/abs/2601.14152>
- <https://arxiv.org/html/2507.15249>
- <https://arxiv.org/html/2607.25622>
- <https://arxiv.org/abs/2603.27199>
- <https://arxiv.org/html/2603.27199v1>
- <https://arxiv.org/html/2605.26111>
- <https://arxiv.org/html/2508.02324>
- <https://arxiv.org/html/2512.13672v1>
- <https://arxiv.org/abs/2312.15905>
- <https://arxiv.org/html/2505.14671>
- <https://arxiv.org/html/2501.03544v3>
- <https://arxiv.org/html/2508.08812v1>
- <https://arxiv.org/html/2410.19313v3>
- <https://arxiv.org/html/2610.25048>
- <https://arxiv.org/html/2403.12422>
- <https://arxiv.org/abs/2206.11357>
- <https://arxiv.org/abs/2405.04532>
- <https://arxiv.org/html/2501.18475v2>
- <https://arxiv.org/html/2310.08659v4>
- <https://arxiv.org/html/2402.05147v3>
- <https://arxiv.org/html/2606.04238v1>
- <https://arxiv.org/html/2603.20755v1>
- <https://arxiv.org/abs/2205.05198>
- <https://arxiv.org/abs/2503.01905>
- <https://arxiv.org/html/2402.16842>
- <https://arxiv.org/abs/2308.03303>
- <https://arxiv.org/html/2502.20566v1>
- <https://arxiv.org/html/2407.15892v4>
- <https://arxiv.org/abs/2412.05270>
- <https://arxiv.org/abs/1811.06965>
- <https://arxiv.org/abs/2408.00908>
- <https://arxiv.org/html/2409.02086v2>
- <https://arxiv.org/abs/2512.24145>
- <https://arxiv.org/html/2604.21689>
- <https://arxiv.org/html/2510.14975v1>
- <https://arxiv.org/html/2607.14932>
- <https://arxiv.org/html/2510.14255v1>
- <https://arxiv.org/abs/2305.20086>
- <https://arxiv.org/abs/2509.25705>
- <https://arxiv.org/html/2210.02410>
- <https://arxiv.org/html/2505.00759v3>
- <https://arxiv.org/html/2610.03617>
- <https://arxiv.org/html/2406.16855v2>
- <https://arxiv.org/abs/1711.09846>
- <https://arxiv.org/html/2610.11023v1>
- <https://arxiv.org/html/2609.04151v2>
- <https://arxiv.org/html/2512.19026>
- <https://arxiv.org/abs/1802.02219>
- <https://arxiv.org/html/2404.16795v3>
- <https://arxiv.org/abs/2510.02721>
- <https://arxiv.org/abs/2307.15422>
- <https://arxiv.org/abs/2602.11171>
- <https://arxiv.org/abs/2402.17826>
- <https://arxiv.org/html/2506.03225v1>
- <https://arxiv.org/pdf/2109.13800.pdf>
- <https://arxiv.org/html/2509.25214v4>
- <https://arxiv.org/html/2508.08812>
- <https://arxiv.org/html/2510.03262>
- <https://arxiv.org/abs/2403.07378>
- <https://arxiv.org/html/2609.38680v1>
- <https://arxiv.org/html/2507.06547v2>
- <https://arxiv.org/abs/2507.06547v3>
- <https://arxiv.org/html/2506.05647>
- <https://arxiv.org/abs/2406.11011>
- <https://arxiv.org/abs/2602.00329>
- <https://arxiv.org/html/2310.00902v3>
- <https://arxiv.org/abs/2409.19998>
- <https://arxiv.org/pdf/2410.13850>
- <https://arxiv.org/abs/2601.22651>
- <https://arxiv.org/html/2605.17938>
- <https://arxiv.org/html/2505.21905v1>
- <https://arxiv.org/html/2501.05177>
- <https://arxiv.org/pdf/2204.00964>
- <https://arxiv.org/html/2511.11051>
- <https://arxiv.org/html/2512.05117v2>
- <https://arxiv.org/html/2507.04599v2>
- <https://arxiv.org/abs/2606.10617>
- <https://arxiv.org/abs/2407.08608>
- <https://arxiv.org/html/2605.10730v1>
- <https://arxiv.org/html/2605.13565v1>
- <https://arxiv.org/abs/2406.07507>
- <https://arxiv.org/abs/2506.14603>
- <https://arxiv.org/abs/2505.13447>

**github.com**

- <https://github.com/bghira/SimpleTuner/blob/main/documentation/TREAD.md>
- <https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/FLUX2.md>
- <https://github.com/bghira/SimpleTuner>
- <https://github.com/thu-ml/SageAttention/tree/main>
- <https://github.com/pytorch/pytorch/releases/tag/v2.15.0>
- <https://github.com/pytorch/pytorch/releases/tag/v2.12.0>
- <https://github.com/linkedin/Liger-Kernel>
- <https://github.com/shootthesound/Fizgig/releases/tag/v6.5.3>
- <https://github.com/kohya-ss/musubi-tuner/issues/1163>
- <https://github.com/black-forest-labs/flux-3-image-klein>
- <https://github.com/thu-ml/SageAttention>
- <https://github.com/nunchaku-ai/nunchaku>
- <https://github.com/ruikangliu/FlatQuant>
- <https://github.com/pilancilab/Riemannian_Preconditioned_LoRA>
- <https://github.com/shootthesound/Fizgig/blob/master/docs/KREA2.md>
- <https://github.com/HomebrewML/HeavyBall>
- <https://github.com/kozistr/pytorch_optimizer>
- <https://github.com/changlin31/AutoProg-Zero>
- <https://github.com/kohya-ss/musubi-tuner/blob/main/docs/advanced_config.md>
- <https://github.com/kohya-ss/musubi-tuner/blob/main/docs/minimax_h3_advanced.md>
- <https://github.com/ControlGenAI/T-LoRA>
- <https://github.com/yhli123/Immiscible-Diffusion>
- <https://github.com/bongobongo2020/krea2-character-lora-trainer>
- <https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md>
- <https://github.com/shootthesound/Fizgig/releases/tag/v6.5.0>
- <https://github.com/shootthesound/Fizgig/releases/tag/v6.8.3>
- <https://github.com/shootthesound/Fizgig/issues/265>
- <https://github.com/danmo-ai/danmo-make/pull/76>
- <https://github.com/shootthesound/Fizgig/blob/master/docs/LORA_ISOLATION.md>
- <https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/model-coverage/diffusion/black-forest-labs/flux3-image-klein.mdx>
- <https://github.com/comfyanonymous/ComfyUI/pull/17296>
- <https://github.com/modelscope/DiffSynth-Studio>
- <https://github.com/chengyansen-ai/krea2-lora-training>
- <https://github.com/huggingface/diffusers/blob/main/examples/advanced_diffusion_training/README_flux.md>
- <https://github.com/YuqiPeng77/TARA>
- <https://github.com/Nerogar/OneTrainer/wiki/Quantization>
- <https://github.com/modelscope/entropack>
- <https://github.com/Nerogar/OneTrainer/pull/1568>
- <https://github.com/pytorch/ao/tree/main/torchao/optim>
- <https://github.com/Nerogar/OneTrainer/blob/master/docs/RamOffloading.md>
- <https://github.com/WooSunghyeon/paca>
- <https://github.com/lodestone-rock/RamTorch>
- <https://github.com/zhuhanqing/APOLLO>
- <https://github.com/pytorch/pytorch/releases/tag/v2.13.0>
- <https://www.github.com/pytorch/pytorch/issues/125235>
- <https://github.com/triton-lang/triton/issues/8491>
- <https://github.com/pytorch/pytorch/releases>
- <https://github.com/pytorch/pytorch/issues/130249>
- <https://github.com/woct0rdho/triton-windows>
- <https://github.com/pytorch/pytorch/releases/tag/v2.11.0>
- <https://github.com/tdrussell/diffusion-pipe>
- <https://github.com/mk-minchul/AdaFace>
- <https://github.com/bytedance/LVFace>
- <https://github.com/facebookresearch/sscd-copy-detection>
- <https://github.com/automl/QTT>
- <https://github.com/automl/ifBO>
- <https://github.com/nunchaku-tech/nunchaku>
- <https://github.com/nunchaku-tech/nunchaku/pull/754>
- <https://github.com/pytorch/pytorch/releases/tag/v2.5.1>
- <https://github.com/Dao-AILab/flash-attention>
- <https://github.com/pytorch/pytorch/releases/tag/v2.14.0>
- <https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/KREA2.md>
- <https://github.com/shootthesound/Fizgig/blob/master/docs/QWEN_IMAGE.md>
- <https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/QWEN_IMAGE_2-1.md>
- <https://github.com/huggingface/diffusers/pull/14804>
- <https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage21.py>
- <https://github.com/kohya-ss/musubi-tuner/blob/main/docs/qwen_image.md>
- <https://github.com/modelscope/DiffSynth-Studio/pull/1724>
- <https://github.com/modelscope/DiffSynth-Studio/issues/1706>
- <https://github.com/QwenLM/Qwen-Image-2.1>
- <https://github.com/kohya-ss/musubi-tuner/blob/main/docs/flux_2.md>
- <https://github.com/tdrussell/diffusion-pipe/blob/main/docs/supported_models.md>
- <https://github.com/KohakuBlueleaf/KohakuMossTrainer/wiki>
- <https://github.com/PineCookie/lora-forge/releases>
- <https://github.com/comfyanonymous/ComfyUI/pull/17095>
- <https://github.com/kohya-ss/musubi-tuner/blob/main/docs/minimax_h3.md>
- <https://github.com/MiniMax-AI/awesome-minimax-h3-integration>

**raw.githubusercontent.com**

- <https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/MINIMAX_H3.md>
- <https://raw.githubusercontent.com/KohakuBlueleaf/LyCORIS/main/docs/kernels/benchmarks.md>
- <https://raw.githubusercontent.com/KohakuBlueleaf/LyCORIS/main/Change.md>
- <https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/transformers/transformer_krea2.py>
- <https://raw.githubusercontent.com/krea-ai/krea-2/main/mmdit.py>
- <https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/src/liger_kernel/ops/modulated_rms_norm.py>
- <https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/src/liger_kernel/ops/swiglu.py>
- <https://raw.githubusercontent.com/black-forest-labs/flux-3-image-klein/main/src/flux3/sampling.py>
- <https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/ldm/flux3/model.py>
- <https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/docs/advanced_config.md>
- <https://raw.githubusercontent.com/KohakuBlueleaf/KohakuMossTrainer/main/docs/concepts/training.md>
- <https://raw.githubusercontent.com/koratahiu/Advanced_Optimizers/main/docs/advanced_features.md>
- <https://raw.githubusercontent.com/tdrussell/diffusion-pipe/main/examples/main_example.toml>
- <https://raw.githubusercontent.com/koratahiu/Advanced_Optimizers/main/docs/adam_settings.md>
- <https://raw.githubusercontent.com/bghira/SimpleTuner/main/documentation/OPTIONS.md>
- <https://raw.githubusercontent.com/ostris/ai-toolkit/main/extensions_built_in/diffusion_models/krea2/krea2.py>
- <https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/docs/minimax_h3_1f.md>
- <https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/docs/minimax_h3_advanced.md>
- <https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/docs/minimax_h3.md>
- <https://raw.githubusercontent.com/tdrussell/diffusion-pipe/main/docs/supported_models.md>
- <https://raw.githubusercontent.com/tdrussell/diffusion-pipe/main/docs/minimax_h3_notes.md>
- <https://raw.githubusercontent.com/ostris/ai-toolkit/main/toolkit/config_modules.py>
- <https://raw.githubusercontent.com/ostris/ai-toolkit/main/extensions_built_in/sd_trainer/SDTrainer.py>
- <https://raw.githubusercontent.com/bghira/SimpleTuner/main/documentation/distillation/MINIMAX_H3_DRIFT.md>
- <https://raw.githubusercontent.com/bghira/SimpleTuner/main/documentation/quickstart/FLUX3_IMAGE.md>
- <https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_flux3_klein_img2img.py>
- <https://raw.githubusercontent.com/black-forest-labs/flux-3-image-klein/main/src/flux3/pup_prompts.json>
- <https://raw.githubusercontent.com/modelscope/DiffSynth-Studio/main/examples/moss_image/model_training/special/opd/train_opd_lora.py>
- <https://raw.githubusercontent.com/modelscope/DiffSynth-Studio/main/examples/moss_image/model_training/special/opd/train_MOSS-Image-SFT-8steps_lora.sh>
- <https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/REFMOD_HOWDOI.md>
- <https://raw.githubusercontent.com/OpenMOSS/MOSS-Image/main/USERGUIDE.md>
- <https://raw.githubusercontent.com/KohakuBlueleaf/KohakuMossTrainer/main/docs/guides/first-lora.md>
- <https://raw.githubusercontent.com/krea-ai/krea-2/main/encoder.py>
- <https://raw.githubusercontent.com/pytorch/ao/main/torchao/quantization/quant_api.py>
- <https://raw.githubusercontent.com/pytorch/torchtune/main/torchtune/training/_activation_offloading.py>
- <https://raw.githubusercontent.com/Nerogar/OneTrainer/master/docs/RamOffloading.md>
- <https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/docs/hunyuan_video.md>
- <https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/NEWS.md>
- <https://raw.githubusercontent.com/kohya-ss/musubi-tuner/main/src/musubi_tuner/krea2/krea2_utils.py>
- <https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/FINETUNE.md>
- <https://raw.githubusercontent.com/pytorch/pytorch/main/torch/_inductor/config.py>
- <https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/ops.py>
- <https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/model_patcher.py>
- <https://raw.githubusercontent.com/city96/ComfyUI-GGUF/main/ops.py>
- <https://raw.githubusercontent.com/kohya-ss/sd-scripts/main/networks/resize_lora.py>
- <https://raw.githubusercontent.com/KohakuBlueleaf/KohakuMossTrainer/main/docs/performance/training-b300.md>
- <https://raw.githubusercontent.com/KohakuBlueleaf/KohakuMossTrainer/main/docs/performance/training-5090.md>
- <https://raw.githubusercontent.com/KohakuBlueleaf/KohakuMossTrainer/main/README.md>
- <https://raw.githubusercontent.com/Nerogar/OneTrainer/master/training_presets/Krea%202/%23krea2%20LoRA%2016GB.json>
- <https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/transformers/attention.cpp>
- <https://raw.githubusercontent.com/ostris/ai-toolkit/main/extensions_built_in/diffusion_models/flux3/src/transformer.py>
- <https://raw.githubusercontent.com/NVIDIA-NeMo/Automodel/main/examples/diffusion/finetune/flux3_t2i_flow_lora.yaml>
- <https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_flux3_klein.py>
- <https://raw.githubusercontent.com/ostris/ai-toolkit/main/extensions_built_in/diffusion_models/flux3/src/text_encoder.py>
- <https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/FLUX3.md>
- <https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/lora_convert.py>
- <https://raw.githubusercontent.com/black-forest-labs/flux-3-image-klein/main/src/flux3/text_encoder.py>
- <https://raw.githubusercontent.com/ostris/ai-toolkit/main/extensions_built_in/diffusion_models/flux3/flux3.py>
- <https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/text_encoders/flux.py>
- <https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/modular_pipelines/flux3/before_denoise.py>
- <https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/supported_models.py>
- <https://raw.githubusercontent.com/ostris/ai-toolkit/main/extensions_built_in/diffusion_models/flux3/src/pipeline.py>
- <https://raw.githubusercontent.com/black-forest-labs/flux-3-image-klein/main/src/flux3/pipeline.py>
- <https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/model_base.py>
- <https://raw.githubusercontent.com/black-forest-labs/flux-3-image-klein/main/src/flux3/dit.py>
- <https://raw.githubusercontent.com/KohakuBlueleaf/KohakuMossTrainer/main/docs/concepts/numerics.md>
- <https://raw.githubusercontent.com/KohakuBlueleaf/KohakuMossTrainer/main/docs/concepts/massive-channels.md>
- <https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/ldm/moss_image/model.py>
- <https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/schedulers/scheduling_minimax_h3.py>
- <https://raw.githubusercontent.com/tdrussell/diffusion-pipe/main/examples/minimax_h3_example.toml>
- <https://raw.githubusercontent.com/huggingface/diffusers/main/src/diffusers/models/transformers/transformer_minimax_h3.py>
- <https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/ldm/minimax/model.py>
- <https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/comfy/lora.py>

**huggingface.co**

- <https://huggingface.co/CalamitousFelicitousness/Krea-2-Base-Diffusers/raw/main/transformer/config.json>
- <https://huggingface.co/Comfy-Org/Krea-2/discussions/6>
- <https://huggingface.co/lightx2v/Qwen-Image-2.1-Lightning>
- <https://huggingface.co/linoyts/Qwen-Image-2.1-DMD2-2steps>
- <https://huggingface.co/nynxz/Qwen-Image-2.1>
- <https://huggingface.co/PrunaAI/P-Q-Image-2.1-Comfy>
- <https://huggingface.co/blog/flux-3>
- <https://huggingface.co/Comfy-Org/Qwen-Image/discussions/2>
- <https://huggingface.co/lodestones/Kromoss>
- <https://huggingface.co/docs/peft/package_reference/hra>
- <https://huggingface.co/SimpleTuner/flux-3-klein-image-LoRA-experiments/raw/main/chapters/03-assistant.md>
- <https://huggingface.co/MiniMaxAI/MiniMax-H3>
- <https://huggingface.co/docs/peft/package_reference/delora>
- <https://huggingface.co/ostris/krea2_turbo_training_adapter>
- <https://huggingface.co/krea/Krea-2-Turbo/discussions/10>
- <https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v2>
- <https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v3>
- <https://huggingface.co/circlestone-labs/MiniMax-H3-Image-Training-Adapter>
- <https://huggingface.co/api/models/ostris/minimax_h3_training_adapter/tree/main>
- <https://huggingface.co/MATLOWAI/MOSS-Image-SFT-DeBand>
- <https://huggingface.co/ShootTheSound/Fizgig-MOSS-Image-SFT-Look-LoRA>
- <https://huggingface.co/SimpleTuner/flux-3-klein-image-LoRA-experiments/raw/main/chapters/06-base-to-distilled.md>
- <https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora>
- <https://huggingface.co/TheDivergentAI/krea2-turbo-distill-lora>
- <https://huggingface.co/black-forest-labs/flux-3-image-klein>
- <https://huggingface.co/ostris/Krea2OstrisEdit>
- <https://huggingface.co/Alissonerdx/CharacterSheet/blob/main/README.md>
- <https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA>
- <https://huggingface.co/yijunwang2/krea2-reid>
- <https://huggingface.co/lodestones/MOSS-Image-Base-8-steps-dmd>
- <https://huggingface.co/no3us/minimax-h3-refmod-pilot>
- <https://huggingface.co/blog/linoyts/new-advanced-flux-dreambooth-lora>
- <https://huggingface.co/BennyDaBall/Krea-2-Engineer-V1>
- <https://huggingface.co/ModelsLab/Krea-2-Turbo-W4A4-Nunchaku>
- <https://huggingface.co/ostris/accuracy_recovery_adapters>
- <https://huggingface.co/DiffSynth-Studio/MOSS-Image-EntroPack>
- <https://huggingface.co/blog/asyncgrpo-lora-hfjobs>
- <https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF>
- <https://huggingface.co/molbal/krea2-gguf>
- <https://huggingface.co/molbal/MiniMax-H3-GGUF>
- <https://huggingface.co/Gluttony10/MiniMax-H3-INT8-CONVROT>
- <https://huggingface.co/ausboss/Qwen-Image-2.1-Restore-LoRA>
- <https://huggingface.co/ausboss/FLUX-3-Image-Klein-Restore-LoRA>
- <https://huggingface.co/docs/diffusers/api/pipelines/krea2>
- <https://huggingface.co/MJSAA/hannahowo-krea2-lora>
- <https://huggingface.co/jayark/sammy-krea-v4-loras>
- <https://huggingface.co/pmczip/Krea2Raw_Loras/discussions/4>
- <https://huggingface.co/pmczip/Krea2Raw_Loras/discussions/3>
- <https://huggingface.co/Qwen/Qwen-Image-2512>
- <https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1>
- <https://huggingface.co/Qwen/Qwen-Image-2.1/discussions/12>
- <https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-1-and-krea-2>
- <https://huggingface.co/docs/diffusers/main/en/api/models/qwenimage21_transformer2d>
- <https://huggingface.co/Qwen/Qwen-Image-2.1/raw/main/LICENSE>
- <https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo>
- <https://huggingface.co/Qwen/Qwen-Image-2.1/discussions>
- <https://huggingface.co/Qwen/Qwen-Image-2.1/discussions/32>
- <https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy>
- <https://huggingface.co/KBlueLeaf/MOSS-Image-FP16-fixed>
- <https://huggingface.co/docs/diffusers/v0.25.1/tutorials/fast_diffusion>
- <https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/transformer/config.json>
- <https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/vae/config.json>
- <https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE>
- <https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/docs/QA-about-License.md>
- <https://huggingface.co/api/models/Comfy-Org/MiniMax-H3/tree/main/diffusion_models>
- <https://huggingface.co/coolthor/MiniMax-H3-pruned-NVFP4>
- <https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA>
- <https://huggingface.co/multimodalart/MiniMax-H3-Pruned>
- <https://huggingface.co/dipjeb/MiniMax-H3-Turbo-LoRA-Pruned>
- <https://huggingface.co/hyukudan/Pika-H3-Flex-Hybrid-b30-49>
- <https://huggingface.co/prithivMLmods/MiniMax-H3-Facial-Realism-CloseUp>

**docs.pytorch.org**

- <https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.scaled_dot_product_attention.html>
- <https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html>
- <https://docs.pytorch.org/docs/2.15/library.html>
- <https://docs.pytorch.org/tutorials/recipes/regional_compilation.html>
- <https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html>
- <https://docs.pytorch.org/docs/2.15/checkpoint.html>
- <https://docs.pytorch.org/docs/2.15/autograd.html>
- <https://docs.pytorch.org/docs/2.15/notes/cuda.html>
- <https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html>
- <https://docs.pytorch.org/docs/2.15/distributed.html>
- <https://docs.pytorch.org/docs/2.15/notes/windows.html>
- <https://docs.pytorch.org/ao/main/workflows/training.html>

**comfyui-wiki.com**

- <https://comfyui-wiki.com/en/custom-nodes/ComfyUI-KJNodes/nodes/pathch-sage-attention-kj>
- <https://comfyui-wiki.com/en/news/2026-07-04-ai-toolkit-krea2-edit-training>
- <https://comfyui-wiki.com/en/news/2026-09-07-minimax-h3-refmods>
- <https://comfyui-wiki.com/ko/custom-nodes/ComfyUI-KJNodes/nodes/lora-reduce-rank>
- <https://comfyui-wiki.com/ja/news/2026-09-24-texture-fix-vae-qwen-image-2-1>

**openaccess.thecvf.com**

- <https://openaccess.thecvf.com/content/CVPR2025/html/Chae_APT_Adaptive_Personalized_Training_for_Diffusion_Models_with_Limited_Data_CVPR_2025_paper.html>
- <https://openaccess.thecvf.com/content/CVPR2025W/SyntaGen/html/Ni_Noise_Consistency_Regularization_for_Improved_Subject-Driven_Image_Synthesis_CVPRW_2025_paper.html>
- <https://openaccess.thecvf.com/content/WACV2025/html/He_A_Data_Perspective_on_Enhanced_Identity_Preservation_for_Diffusion_Personalization_WACV_2025_paper.html>
- <https://openaccess.thecvf.com/content/CVPR2026W/P13N/supplemental/Koh_Let_Triggers_Control_CVPRW_2026_supplemental.pdf>

**ar5iv.labs.arxiv.org**

- <https://ar5iv.labs.arxiv.org/html/2107.04197>
- <https://ar5iv.labs.arxiv.org/html/2402.02347>
- <https://ar5iv.labs.arxiv.org/html/2304.06648>

**pytorch.org**

- <https://pytorch.org/blog/flexattention/>
- <https://pytorch.org/blog/activation-checkpointing-techniques>
- <https://pytorch.org/blog/pytorch2-5/>

**cvpr.thecvf.com**

- <https://cvpr.thecvf.com/virtual/2026/poster/39047>
- <https://cvpr.thecvf.com/virtual/2026/poster/36823>

**docs.comfy.org**

- <https://docs.comfy.org/tutorials/utility/seedvr2.md>
- <https://docs.comfy.org/tutorials/video/minimax/minimax-h3-fastvideo>

**glennklockwood.com**

- <https://glennklockwood.com/garden/processors/H100>
- <https://glennklockwood.com/garden/processors/b200>

**images.nvidia.com**

- <https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf>
- <https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf>

**krea.ai**

- <https://www.krea.ai/blog/krea-2-technical-report>
- <https://www.krea.ai/krea-2-open-source>

**aclanthology.org**

- <https://aclanthology.org/2024.amta-research.12.pdf>

**alan-lanfeng.github.io**

- <https://alan-lanfeng.github.io/PTD/>

**anyscale.com**

- <https://anyscale.com/blog/population-based-bandits>

**automl.github.io**

- <https://automl.github.io/neps/latest/reference/search_algorithms/multifidelity_prior>

**deepspeed.ai**

- <https://www.deepspeed.ai/tutorials/pipeline/>

**dev.to**

- <https://dev.to/furkangozukara/compared-civitai-default-flux-lora-vs-my-best-flux-and-more-developments-5133>

**diffsynth-studio-doc.readthedocs.io**

- <https://diffsynth-studio-doc.readthedocs.io/en/latest/Training/Direct_Distill.html>

**docs.bfl.ml**

- <https://docs.bfl.ml/flux_3/flux3_image_klein>

**docs.nvidia.com**

- <https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html>

**docs.sglang.io**

- <https://docs.sglang.io/cookbook/diffusion/FLUX/FLUX.3>

**hub.docker.com**

- <https://hub.docker.com/r/vastai/pytorch>

**isaac-sim.github.io**

- <https://isaac-sim.github.io/IsaacLab/main/source/features/population_based_training.html>

**lilting.ch**

- <https://lilting.ch/en/articles/muon-image-lora-feasibility>

**lr.us.psf.lt**

- <https://lr.us.psf.lt/r/StableDiffusion/comments/1dxocch/lora_regularization_images>

**merl.com**

- <https://merl.com/publications/docs/TR2025-073.pdf>

**meta-pytorch.org**

- <https://meta-pytorch.org/torchtune/main/tutorials/memory_optimizations.html>

**minimax.io**

- <https://www.minimax.io/blog/minimax-h3>

**neurips.cc**

- <https://neurips.cc/virtual/2026/poster/153796>

**optuna.readthedocs.io**

- <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.WilcoxonPruner.html>

**proceedings.iclr.cc**

- <https://proceedings.iclr.cc/paper_files/paper/2025/hash/48c368f105e8145b945227b73255635a-Abstract-Conference.html>

**proceedings.neurips.cc**

- <https://proceedings.neurips.cc/paper_files/paper/2024/hash/d4387c37b3b06e55f86eccdb8cd1f829-Abstract.html>

**py-free-threading.github.io**

- <https://py-free-threading.github.io/tracking/>

**pypi.org**

- <https://pypi.org/project/prodigy-plus-schedule-free/>

**runcomfy.com**

- <https://www.runcomfy.com/trainer/ai-toolkit/qwen-image-edit-2511-likeness-custom-lora>

**search.r-project.org**

- <https://search.r-project.org/CRAN/refmans/irace/html/irace-package.html>

**thinkingmachines.ai**

- <https://thinkingmachines.ai/blog/lora/>

**vast.ai**

- <https://vast.ai/pricing>

**weirdwonderfulai.art**

- <https://weirdwonderfulai.art/comfyui/qwen-image-edit-can-create-character-consistent-lora-dataset/>

**zhuhanqing.github.io**

- <https://zhuhanqing.github.io/APOLLO>
