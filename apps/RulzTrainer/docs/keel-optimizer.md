# Keel: an identity-first optimizer for character LoRAs (design)

Status: milestone 1 built (`rulztrainer/optimizers/keel.py`: E and F, `optimizer: keel`); the rest is design.
Working name "Keel" (the part of the boat that keeps it from drifting sideways). Every number below is a starting
value to race, not a measured one.

## 1. Goal

A drop-in `optimizer: keel` for RulzTrainer that trains character LoRAs with the highest held-out likeness
(the FaceNet gold face score of the probes) at a given step budget, without raising SSCD copy similarity, and
that keeps the likeness when the LoRA is used on the family's Turbo model, below strength 1, or stacked with a
style LoRA.

Non-goals: a general-purpose optimizer (style LoRAs, full fine-tunes keep Automagic 4 / adamw8bit), fused mode,
anything that changes the saved file (the output stays a plain LoRA / LoKr).

## 2. Why the optimizer, when the loss already knows about identity

Everything identity-shaped in RulzTrainer today acts on the loss, the data or the schedule: region weights,
the identity loss, `id_align`, DOP, the curriculum, self-training. The optimizer gets one summed gradient per
step ([trainer.py:1209](../rulztrainer/trainer.py:1209)) and cannot tell the face from the background, photo A
from photo B, or sigma 0.2 from sigma 0.9.

That matters because Adam-style optimizers (AdamW, Adafactor, Automagic) divide each weight's step by that
weight's own gradient scale. Scaling the gradient of a weight by a constant therefore barely changes its step:
a weight whose gradient comes mostly from background tokens sees `background_loss_weight: 0.3` shrink its
gradient, and the second moment scales it straight back. Region weights work only where face and background
gradients mix inside one weight. Two design rules follow:

1. **Emphasis goes on the update, after normalisation**, never on the gradient that feeds the moments
   (which is why the cautious mask of ICLR 2026 masks the update, not the gradient).
2. **Anything that must be compared across examples or noise levels has to be seen per micro-batch**, before
   the sum. The optimizer already has a per-backward hook for that
   ([automagic3.py:360](../rulztrainer/optimizers/automagic3.py:360)).

## 3. Shape of the design

`Keel(Automagic4)` in `rulztrainer/optimizers/keel.py`. It keeps Automagic 4's self-tuned learning rate,
factored second moment, bucketed update, 1 byte/weight sign history, state mirrors and the race / relay
behaviour, and adds six mechanisms, each behind its own switch so race rows can test them one at a time.
With every switch off it is Automagic 4 exactly (a test asserts bitwise equality on a fixed problem).

The trainer feeds it context through a small, optional API; without any call it behaves like Automagic 4:

| call | when | what for |
|---|---|---|
| `observe(sigma: Tensor)` | before each micro-batch's backward | the batch's noise band (B, C) |
| `gather()` | after the last backward, before `clip_grad_norm_` | hands the accumulated gradients back as `p.grad` (A, B); `step()` calls it itself if the trainer did not |
| `on_probe(score, band, ema)` | after each likeness probe decision | the lookahead accept / pull-back (D) |
| `face_share(p) = float` | set by the LoRA layer tap during backward | per-tensor face share of the gradient (C) |

Per micro-batch k (all K micro-batches of a step share one sigma band b, see B):

```
g_k           = grad of micro-batch k, per tensor t
rms[t, b]    <- ema(rms[t, b], rms(g_k[t]))                 B  band statistics
acc[t]       += g_k[t] / sqrt(rms[t, b])                    B  every band speaks equally
agree[t]     += sign(g_k[t])            (int8)              A  consensus
face[t]      += energy of g_k[t] from face tokens           C  (LoRA layer tap)
```

Per step:

```
g  = gauge(acc)                                             E  r x r preconditioner, LoRA pairs only
u  = automagic4_update(g)                                   normalise, trust-region clip, sensitivity balance
u *= floor + (1 - floor) * ramp(|agree| / K)                A  emphasis after normalisation
u *= band_weight[b]                                         B
u *= clamp((face[t] / ref) ** gamma, 1 / gmax, gmax)        C  per tensor
u *= mask(u * g > 0) / mean(mask)                           F  cautious
theta -= lr * u ;  lr <- automagic vote (unchanged)
```

At each probe: D.

## 4. The mechanisms

### A. Cross-photo consensus gate

**Idea.** A character LoRA should learn what is true in every photo (the face, the eyes, the hair line, the
build) and not what varies between them (the outfit, the background, the light). Averaging gradients learns
the union; the AND-mask (Parascandolo et al., ICLR 2021) and SAND-mask show that moving a weight only where
the per-example gradients agree in sign suppresses the per-example shortcuts and keeps the shared mechanism.

**Signal.** `gradient_accumulation` K >= 2 at batch size 1 (the wizard's usual character setup). The hook
adds `sign(g_k)` into an int8 `agree` buffer (K <= 127) and accumulates the gradient itself (fp32 sum, or
Automagic's stochastic low-precision sum), setting `p.grad = None` so the next backward produces the next
micro-batch's gradient on its own. `gather()` writes the sum back to `p.grad` so `clip_grad_norm_` and
`reduce_grads` see what they see today.

**Gate.** `a = |agree| / K` in [0, 1]; `gate = floor + (1 - floor) * a ** power`, with
`floor = 1 - keel_consensus` (consensus 0 = off, 1 = a hard AND-mask) and `power = keel_consensus_power` (2).
Applied to the update after the sensitivity balance, before the sign history is written, so the vote sees the
gated update's magnitude (a gated-off element carries no vote weight, like a dead one).

**Paired sigma.** Sign disagreement between micro-batches at different noise levels is mostly the noise
level, not the content. With `keel_pair_sigma` the step draws one sigma per step
([`sample_sigmas`](../rulztrainer/trainer.py:220) once instead of per batch) and every micro-batch uses it.
Noise cannot be shared (buckets differ in shape); sigma is what matters.

**Multi-GPU.** Replica LoRA parameters are not the optimizer's; [`reduce_grads`](../rulztrainer/multigpu.py:64)
sums them into the primary after all backwards. Phase 1 gates on the primary's own micro-batches only;
phase 1b lets `reduce_grads` call `optimizer.observe_grads(primary_param, replica_grad)` per replica before
the sum, so every GPU's photos vote.

**Cost.** 1 B/weight (`agree`), no extra compute, K >= 2 required (a warning and gate 1 otherwise).

**Risks.** The gate also amplifies anything the dataset shares that is not the person (one photographer's
light, a recurring outfit); the floor keeps those learnable, the copy check measures them. A hard AND-mask
(floor 0) on 4 micro-batches keeps only elements where all 4 agree, about 1/8 under pure noise: expect
learning to slow, so the Automagic vote will raise the lr to compensate; watch `automagic_max_lr`.

**Expected measurable effect.** Higher gold score per step, lower SSCD similarity to the training photos
(`copy_check: on` logs it on every probe).

### B. Noise-band normalisation

**Idea.** Gradient scale and variance vary a lot with sigma (Min-SNR; *Addressing Negative Transfer in
Diffusion Models*; adaptive non-uniform timestep sampling). One second moment per weight is set by the
loudest band, usually not the band where the face forms (sigma 0.2-0.5 at 1 MP). EDM2 learns a per-noise-level
weight for the same reason. Doing it in the optimizer makes it per tensor: a timestep embedder and an
attention output get different balances.

**Mechanism.** `keel_bands` (8) equal bands in logit(sigma) after the shift. Per tensor and band an EMA of the
micro-batch gradient RMS (scalar, `keel_bands` floats per tensor). The micro-batch gradient is divided by
`sqrt(rms[t, b])` (with the tensor's overall RMS while a band has fewer than 8 observations) before it enters
the sum. Then `band_weight[b]` (from `keel_band_weights`: `""` = 1 everywhere, `identity` = a hump over the
face bands, or 8 comma-separated numbers) multiplies the step's update.

**Interplay.** This removes the band imbalance that `loss_weighting` and `clean_share` partly steer with.
`loss_weighting` still changes which band's *content* the gradient carries within a step; its scale effect is
gone, which is the point. `timestep_sampling` keeps deciding how often each band is visited.

**Cost.** Negligible. Needs `observe(sigma)`; without it, one band (= off).

### C. Face-token gain

**Idea.** Weights whose gradient mostly flows through the face's tokens take a larger step; weights serving
the background a smaller one. Unlike the identity loss it cannot be gamed: the gain is a positive scalar on
the diffusion update, so the step still descends the diffusion loss and cannot steer toward features that
fool a face recognizer (the failure the FaceNet gold judge exists to catch).

**Signal.** In [`LoRALinear._added`](../rulztrainer/lora.py:385) (LoRA kind only): save `h = x A^T`
([tokens, r], detached) and register a tensor hook on the LoRA output so the backward hands back the
per-token output gradient `d` ([tokens, out]). The weight gradient of `lora_up` is `d^T h` summed over
tokens; with the batch's face mask over tokens the face part is `d[face]^T h[face]`, one extra rank-r
matmul per layer and no second backward. `share = ||G_face||^2 / ||G||^2` per tensor (a scalar; the
`lora_down` of the pair gets the same share). The face mask comes from the region map's `face_cells`
([regions.py:213](../rulztrainer/regions.py:213)) and the per-sample image token ranges the alignment
[`Taps`](../rulztrainer/alignment.py:50) already receive (`set_layout`): one token per 16-px cell, row-major.
Checkpointed blocks recompute the forward before their backward; the hook registered in the recompute is the
one that fires, and the saved `h` is the recompute's (last write wins), so they agree.

**Gain.** `gain[t] = clamp((share[t] / median_share) ** keel_face_gain, 1 / keel_face_gain_max,
keel_face_gain_max)`, median over the live tensors of the step, EMA'd over `keel_face_gain_beta` (0.9) so one
batch without a face does not swing it. Applied per tensor to the update, after the balance. Items without a
face (`face_cells None`) contribute nothing to `face[t]` and are left out of that step's median.

**Cost.** About 5-15 % of step time (one [tokens, r] x [tokens, out] product per LoRA layer, the `h` buffers
of a block). LoKr / OFT: gain 1.

**Risks.** The share is per tensor, not per element, so it cannot push background-only elements inside a
tensor; a per-element version (factored row/col face energy, like the second moment) is a phase-2 option.
Blocks where every tensor has the same share get gain 1 by construction (median).

### D. Probe-gated lookahead

**Idea.** The probes with the held-out FaceNet judge are the one signal nothing trains on. Lookahead (Zhang
et al. 2019) keeps slow weights and pulls the fast weights toward them; here the probe decides.

**Mechanism.** Keel keeps an anchor copy of every trained tensor in `state[p]["anchor"]` (fp32, on the
parameter's device; parked, moved and relayed with the rest of the state). `on_probe(score, band, ema)` after
the likeness controller's decision:

- accept: `score > anchor_score + band` -> anchor := the weights that scored (the EMA shadow when
  `lora_ema` is on, else the raw weights), `anchor_score := score`.
- reject (after `keel_lookahead_grace` probes, 2): raw weights <- anchor + (1 - pull) * (raw - anchor) with
  `pull = keel_lookahead_pullback` (0.5); the EMA shadow the same way; every Automagic lr mirror
  *= `keel_lookahead_lr` (0.7, clamped to the rails). The second moments and sign histories stay.
- the step-0 probe sets the first anchor.

The anchor is the weights that were actually scored, so a pull toward it is a pull toward a measured
likeness, not toward weights 50 images younger (the EMA's lag).

**Relation to what exists.** `likeness_track_best` saves the best file but training goes on from wherever
it is; `likeness_lr_control` scales the lr by the trend but never moves the weights; `race_follow` is this
across configurations. Lookahead makes a single run the best of its own path.

**Cost.** One fp32 copy of the LoRA (rank 16 on Krea 2: about 90 MB). Phase 2 could keep it on the CPU.

**Risks.** Probe noise: the accept test uses the controller's `likeness_noise_band` on the smoothed score;
with `probe_seeds: 1` the band must be wide (the `noise_files` check measures it). A run whose score climbs
slowly but steadily is never pulled (every probe accepts); a run that oscillates is damped to its best. A
pull after a plateau can undo a dip that would have recovered: `race_patience_rising`'s logic (stay while
still rising from the previous probe) applies: no pull while the smoothed score rose since the last probe.

### E. LoRA gauge preconditioner

**Idea.** `B A = (B S)(S^-1 A)` for any invertible r x r `S`; Adam is not invariant to that
reparameterisation, so one factor tends to dominate (LoRA-RITE, ICLR 2025). Riemannian preconditioned LoRA
(Zhang & Pilanci, ICML 2024) restores it with two r x r solves per layer:

```
g_up   <- g_up   (A A^T + delta I)^-1        A = lora_down [r, in]
g_down <- (B^T B + delta I)^-1 g_down        B = lora_up   [out, r]
```

and reports better quality and stability on a 6-image SD 1.5 fine-tune, with equal lrs for both factors
(so LoRA+ is not needed; RulzTrainer's Automagic already ignores it).

**Mechanism (as built).** `build_optimizer` passes Keel the `(down, up)` pairs from the injected modules
(`lora.lora_pairs`); the solves run in `step()` on the accumulated gradient, after the trainer's clipping and
before the second moment. `delta = keel_gauge_eps (0.01) x` the larger of the two Grams' mean diagonal: with
`lora_up` at zero (its start) `lora_down`'s gradient is zero too, and while a factor is still small its
preconditioner is about the identity, so there is no blow-up through a near-zero Gram. A race that switches a
wrapper to LoKr leaves empty LoRA tensors, which the step skips.

**Cost.** Two r x r inversions per LoRA layer per step: nothing. LoKr / OFT / DoRA magnitudes: untouched.

### F. Cautious mask

**Idea.** Zero the update where its sign disagrees with the current gradient (Liang et al., ICLR 2026, "one
line of code"; 1.3-1.5x sample efficiency on LLM pretraining with AdamW and Lion), so the step keeps a
non-negative inner product with the gradient.

**What it can act on here.** Every scaling in Automagic 4's step is positive (the second-moment normalisation,
the trust-region clip, the sensitivity balance), and so are A's gate, B's band weight and C's gain: the update's
sign is the gradient's sign and the mask would never fire on them. What can flip a sign is the gauge
preconditioner (E), which mixes the r directions of a factor; a later non-diagonal mechanism would be the same.

**Mechanism (as built).** The mask is applied to the preconditioned gradient against the raw one, in
`precondition()`: `g_pre *= (g_pre * g_raw > 0)`. Since everything after it is positive that is the mask on
the update, except that the masked elements also leave the second moment (where they would have counted at the
raw sign) and the sensitivity measurement. No rescale by the surviving share: the normalised step does not
depend on the gradient's scale. The kept share is logged every 250 steps; well under 1 means the preconditioner
is fighting the gradient (raise `keel_gauge_eps`).

**Cost.** None.

### Phase 2 (not in the first build)

- **Identity lock-in.** After a lookahead accept, decay tensors with a high face share toward the anchor
  (EWC-shaped, like [`finetune_anchor`](../rulztrainer/finetune.py) but toward the best likeness rather than the
  base model), so late training keeps gaining flexibility without eroding the face.
- **Flatness for Turbo transfer.** The LoRA trains on Raw / dev and is used on a distilled Turbo model, at
  strength below 1, and stacked with other LoRAs: all shifts in the weights a flat minimum survives.
  Flat-LoRA's random full-rank perturbations cost forward time and report modest gains; cheaper: sample the
  training-adapter / turbo strength per micro-batch, which the consensus gate then treats like any other
  nuisance variation.
- **Per-element face gain** (factored row/col face energy).
- **Every GPU's photos in the consensus** (phase 1b above).

### Set aside

- Muon-style orthogonalised updates: mixed results on LoRA (PoLoRA finds Muon per factor does not beat
  Adam reliably).
- PCGrad / NashMTL across noise bands: one momentum buffer per band is too much state.
- The ArcFace gradient inside the optimizer: the gaming risk the gold judge was built against.

## 5. Where it plugs in

| place | change |
|---|---|
| [config.py:26](../rulztrainer/config.py:26) `OPTIMIZERS`, `AUTOMAGIC` | add `keel` (it owns its lr like Automagic: no schedule, LoRA+, per-image lr) |
| [config.py](../rulztrainer/config.py) | the `keel_*` options (section 6) and their validation |
| [trainer.py:264](../rulztrainer/trainer.py:264) `build_optimizer` | construct `Keel(params, pairs=..., **automagic kw, **keel kw)`; pairs from the injected LoRA modules |
| [trainer.py:1209](../rulztrainer/trainer.py:1209) step loop | `observe(sigma)` before each backward; `gather()` before `clip_grad_norm_`; paired sigma when `keel_pair_sigma` |
| [lora.py:385](../rulztrainer/lora.py:385) `_added` | the face tap (`h` save + output hook), armed per batch with the face token mask |
| [likeness.py](../rulztrainer/likeness.py) controller / trainer probe path | `on_probe(...)` after the decision, before `likeness_track_best` saves |
| [multigpu.py:64](../rulztrainer/multigpu.py:64) `reduce_grads` | phase 1b: `observe_grads` per replica |
| [race.py:58](../rulztrainer/race.py:58) | `keel_*` in the settings a race row may change |
| [race.py:743](../rulztrainer/race.py:743) `follow_leader` | nothing: `type(wo) is type(vo)` already makes Keel its own kind; the anchor rides in `state[p]` |
| README | a *Keel* section under *Self-tuning optimizers*, the Wizard's optimizer choice |
| tests | `tests/test_keel.py` (section 8) |

## 6. Options

```yaml
optimizer: keel                 # Automagic 4 + the switches below; automagic_* options apply as well
keel_consensus: 0.5             # 0 = off .. 1 = hard AND-mask; gate = (1 - c) + c * agree^power
keel_consensus_power: 2.0
keel_pair_sigma: true           # one sigma per step for all of its micro-batches
keel_bands: 8                   # noise bands (logit sigma); 1 = band normalisation off
keel_band_weights: ""           # "" = flat | identity | 8 numbers
keel_face_gain: 0.0             # exponent on (face share / median); 0 = off
keel_face_gain_max: 2.0         # the gain stays within [1/max, max]
keel_face_gain_beta: 0.9
keel_lookahead: false           # needs probes with face_refs; accept / pull-back on the gold score
keel_lookahead_pullback: 0.5
keel_lookahead_lr: 0.7
keel_lookahead_grace: 2         # probes before the first pull
keel_gauge: true                # r x r preconditioner on LoRA pairs
keel_gauge_eps: 0.01
keel_cautious: true
```

Defaults for the first build: E and F on (cheap, well-evidenced), A at 0.5 with paired sigma, B on with flat
weights, C and D off until raced. The Wizard keeps recommending adamw8bit until the race says otherwise.

## 7. State, resume, race

- `state_dict()` adds per parameter: `agree` (int8), `band_rms` ([bands] fp32), `anchor` (fp32, when D is
  on), and `anchor_score` once. Automagic 4 resumes under Keel (the extras start empty: a measurement step for
  the bands, no anchor until the next probe), and a Keel state resumes under Automagic 4 (extras ignored),
  the way Automagic 3 and 4 already resume under each other.
- Keel hyper-parameters are not restored from a state (like Automagic's): a race may change them on resume.
- In a race every configuration owns its optimizer; `follow_leader` copies the full state including the
  anchor when both are Keel, and starts over otherwise (the log says so, as today).
- `apply_lr_ratio` ([likeness.py:443](../rulztrainer/likeness.py:443)) scales the lr mirrors by the presence
  of `get_avg_learning_rate`: inherited, works unchanged.

## 8. Validation

Baseline first: the README says Automagic 4 against adamw8bit on likeness "is not measured yet". Everything
below is a race with `race_drop_confidence: 0.9`, `probe_seeds: 4`, `face_gold: auto`, `copy_check: on`, on
the 12-photo set and the 8-image synthetic set the README's other measurements used, at 512 px, rank 16,
int8, one card.

| race | rows | reads |
|---|---|---|
| R0 baseline | adamw8bit, automagic4 | the bar |
| R1 cheap pair | automagic4, keel (E+F only), keel (E), keel (F) | E and F each earn their place or go |
| R2 bands | keel (R1 winner) + B flat, + B identity weights, with and without `keel_pair_sigma` | B |
| R3 consensus | keel_consensus 0, 0.3, 0.6, 1.0 at K = 4 | A; SSCD must not rise |
| R4 face gain | keel_face_gain 0, 0.5, 1.0 | C |
| R5 lookahead | keel_lookahead off / on, pullback 0.3 / 0.5 | D |
| R6 transfer | R5 winner probed on Turbo (`sample_model_path`), at strength 0.8, with a style LoRA via `sample_lora` | the goal's second half |

A row wins when its smoothed gold score beats the others by more than the paired-seed noise (the
`noise_files` check sizes it), its SSCD similarity is not above the baseline's, and R6 holds. Unit tests:
Keel with every switch off equals Automagic 4 bitwise; the gate under fair random signs averages
`floor + (1 - floor) * E[a^power]`; the gauge solve is invariant to `B A = (B S)(S^-1 A)`; the band RMS
equalises two bands of different scale; a reject pulls raw and EMA weights by the right fraction and leaves
the moments; `gather()` reproduces fp32 accumulation to 1 ulp.

## 9. Build order

1. **M0** R0 baseline race (no code).
2. **M1** (built) `keel.py` = Automagic 4 + E + F, `pairs` from `build_optimizer`, config, WebUI, README,
   `tests/test_keel.py`. No trainer-loop changes. R1 is still to run.
3. **M2** `observe(sigma)`, bands, paired sigma. R2.
4. **M3** the accumulation hook, `agree`, `gather()`. R3.
5. **M4** `on_probe`, the anchor, the EMA pull. R5.
6. **M5** the LoRA face tap and the gain. R4.
7. **M6** README, the Wizard recommendation (if a race earned it), phase 1b, phase 2 candidates.

M1-M3 are a few hundred lines on top of Automagic 4 and touch the trainer only at the step loop; M4 and M5
are the invasive ones (the probe path, the LoRA layer).

## 10. Open questions

- Should the consensus gate feed the Automagic vote with the gated magnitude (elements that rarely agree
  stop voting) or the raw one? The design says gated; the race decides.
- Face share per tensor or per element first? Per tensor is safe and cheap; per element is where the
  background-only elements inside a tensor would be reached.
- The band normalisation makes `loss_weighting` a content choice, not a scale one; the Wizard's text for
  `min_snr` needs a sentence once Keel is the recommendation.
- Lookahead and `likeness_curriculum` both react to the same probe; a pull resets the curriculum's bias or
  leaves it? Leave it (it belongs to the data, not the weights), and say so in the log.

## References

- LoRA-RITE: https://arxiv.org/abs/2410.20625 (ICLR 2025)
- Riemannian Preconditioned LoRA: https://arxiv.org/abs/2402.02347 (ICML 2024)
- Learning explanations that are hard to vary (AND-mask): https://arxiv.org/abs/2009.00329
- SAND-mask: https://arxiv.org/abs/2106.02266
- Cautious Optimizers: https://arxiv.org/abs/2411.16085 (ICLR 2026)
- Lookahead Optimizer: https://arxiv.org/abs/1907.08610
- Flat-LoRA: https://arxiv.org/abs/2409.14396
- Min-SNR: https://arxiv.org/abs/2303.09556; Addressing Negative Transfer in Diffusion Models:
  https://arxiv.org/abs/2306.00354; Adaptive non-uniform timestep sampling: https://arxiv.org/abs/2411.09998
- EDM2 (per-noise-level uncertainty weighting): https://arxiv.org/abs/2312.02696
- PoLoRA (Muon on LoRA, negative): https://www.alphaxiv.org/abs/2607.17620
- ai-toolkit Automagic 3 / 4 (the base): https://github.com/ostris/ai-toolkit
