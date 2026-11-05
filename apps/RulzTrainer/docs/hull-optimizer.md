# Hull: a body-identity optimizer for character LoRAs (design)

Status: H0, the base rule, is built (`rulztrainer/optimizers/hull.py`, `tests/test_hull.py`) with `hull_metric: balanced`,
`hull_frame: virtual` and `hull_step_scope: layer`, plus the periodic rebalance of section 4.1. On Krea 2 at 0.25 MP and
2000 steps it ties AdamW so far: best smoothed scores 42.4, 57.2 and 40.0 against 38.8 and 45.0, the 57.2 being the one
run that memorised a photo (SSCD 0.61), so differences of about 15 points between single runs can be chance. The first
part of M1b is built and off by default: the parts tensor (5.2), C1 and C3 (5.1, 5.3) with the cosine retreat
(`hull_regions: parts`, `tests/test_hull_regions.py`). Raced at gain 1.5 on H0 (one fictional character, 0.25 MP, 2000 steps, one RTX
5090), it did not raise the body: best smoothed score 35.6 against 35.8 (seed 1234) and 26.3 against 41.0 (seed 42), mean
body score from step 1000 on 14.5 against 14.5 and 14.0 against 19.6, SSCD at most 0.45, within the limit. A race whose
two rows ran the same plain Hull, by a bug since fixed, gave 37.4 and 40.9. Two seeds fall far short of the section 11
bar, so this does not close R3. M0(d) then showed no case for it: body gradients agree no better than face or background
ones, and body falls below background on purity (section 5.1, first reading). So C1 stays off, and the detail gate, the
held-out-loss retreat and G are not worth building on this evidence. A first A/A reading (`docs/keel-test-results.md`,
four runs each of H0 and AdamW 8-bit on one character) could not tell them apart: best smoothed score 33.5 against 33.6,
difference -0.1 ± 6.4, body +1.8 ± 4.1, with identical runs up to 10 points apart. That shows no gain from H0 and does
not show non-inferiority, so M1a's kill rule is still open. Not built yet: those, the `adam` base and everything
from M2 on. Everything else here is design. Every number below is a starting value to race unless it is cited, measured in
a synthetic benchmark or computed, and then it carries a tag. Working name "Hull": Keel is a ship's spine, the hull
its whole body and skin. Hull is for any character LoRA; Qwen-Image 2.1 is the main target.

**Evidence tags.** **[P]** the paper's own measurement; **[3P]** a third party's; **[C]** a claim without numbers;
**[A]** an anecdote; **[bench]** synthetic tensors at this LoRA's shapes on an RTX 4060 Ti, no model (the research's
optimizer-contract benchmark: 224 rank-32 pairs, 83,886,080 weights); **[N]** a numerical check written for this
design (numpy, random matrices, CPU; each becomes an M1 test); **[computed]** arithmetic from the code or shown here;
**[H]** hypothesis. `file:line` is under `apps/RulzTrainer/rulztrainer/` at commit c4f2835 unless it starts with
`README.md`, `webui.py`, `tests/` or `docs/`. No number here comes from a training run, except the README's step time
(section 10).

## 1. Goal

A drop-in `optimizer: hull` for LoRA-kind character adapters that gives the highest held-out body identity (build,
proportions, structure, skin tone, skin detail, marks) at a given step budget, while the held-out face is no worse
than a tuned AdamW control's, copy similarity, pose-following and trigger leakage are no worse, and the saved file
stays a plain LoRA. It is written from scratch: its own update rule and state, not a subclass of Automagic 4.

Non-goals: style LoRAs and full fine-tunes; LoKr, OFT and DoRA in the first build (the base rule needs a LoRA pair); a
sigma-dependent LoRA (ComfyUI and diffusers could not load it); any training signal from a judge.

## 2. Why a new optimizer

The optimizer gets one summed gradient per step (`trainer.py:1358-1370`). It adds no information about the body; it
decides how far each weight moves, in which geometry, and, when the trainer tells it, per noise level and token
region. Keel's rules stand (`docs/keel-optimizer.md` §2): (1) emphasis goes on the update, after normalisation, since
an Adam-type normaliser cancels a loss weight wherever one region dominates an element's gradient; (2) anything
compared across photos or noise levels is seen before the sum; (3) judges the training never sees keep a run honest.

### 2.1 What Keel and the research taught

| finding | evidence | for Hull |
|---|---|---|
| Keel's r x r preconditioner sits before an Adam-type normaliser, which cancels its scale | A x 10 moved its update by at most 2.7% [N]; +307 ms and 448 host syncs per step (`torch.linalg.solve` syncs) [bench] | no preconditioning; a balanced frame (4.1) |
| The cautious mask has nothing to act on when every scaling is positive | `docs/keel-optimizer.md` §F; +43 ms, 448 syncs [bench] | rejected (12) |
| Votes within one step are weak | a sign vote tells an 80%-consistent weight from noise by 1.36x at K = 2, 2.08x at K = 4, 12.2x over 32 photos (E[a²] = m² + (1 - m²)/K, m = 0.6) [computed]; a magnitude vote (its mean purity, 5.7 G, at SNR 0.5) 1.07x / 1.24x, against 2.43x across steps at beta 0.9 [N] | consensus from momentum (beta1 0.9 averages 19 photos); Automagic and Keel keep none (`optimizers/automagic3.py:383-411`) |
| Keel's sign vote favours the face in mixed batches; its band normalisation rescales every band to the tensor's overall gradient scale, so a band's share follows its visit rate | `docs/keel-optimizer.md:108-121, 155-166` | no vote; bands relative to natural shares (5.4) |
| Keel's LoRA tap is memory-bound | +16.5% / +25.5% (face / face and body) of the tapped layers' time per 1 MP micro-batch, +300 MB, from per-sample fp32 copies (`lora.py:439`) [bench]; compile off (`trainer.py:1065-1067`) | one widened GEMM in a custom backward, behind a cost gate (5.1) |
| Self-tuned step sizes climb on small datasets | Prodigy's d is non-decreasing by construction, d_(k+1) = max(d_k, d̂_(k+1)) ([2306.06101](https://arxiv.org/abs/2306.06101), Algorithms 1-3) [C]; upstream Automagic 4, whose default `max_lr` is 1e3, ran its lr up about 57x in ~400 steps on 85 images and gave noise or near-copies, a `max_lr` = `lr` cap proposed but untested ([#1091](https://github.com/ostris/ai-toolkit/issues/1091)), and gave noise by step 500-750 (Z-Image Turbo, Krea 2) until `max_lr` was capped at 1e-4 ([#1084](https://github.com/ostris/ai-toolkit/issues/1084)) [A] | nothing loss-driven raises the step |
| Tuned LoRA variants land within 1-2% of each other | [2602.04998](https://arxiv.org/abs/2602.04998) [P, LLM tasks] | the base rule alone should tie AdamW |
| Under Adam on factors the dW step drifts as B grows and, at alpha = rank, with the rank | effective lr rises with B ([Thinking Machines](https://thinkingmachines.ai/blog/lora/)) [P]; Adam at lr 1e-3 moved a layer by 5.2e-3 to 1.65e-2 of its weight at step 1 over ranks 4-32 with alpha = rank [N], which the Wizard sets (`webui.py:4414, 5229`) | the step in dW units (4.3) |
| Balance never emerges from the init | gradient flow conserves BᵀB - AAᵀ ([1806.00900](https://arxiv.org/abs/1806.00900)) [P]; A starts Kaiming-uniform, B at zero (`lora.py:305-319`); BaLoRA, rebalancing after every AdamW step, works over a wider range of lrs and init scales ([2605.31484](https://arxiv.org/abs/2605.31484); its authors call their Adam analysis heuristic) [P, 3B LLMs] | balance is imposed (4.1) |

### 2.2 Rules Hull adds

4. **Define the step where the model lives**: rho is a relative change of each layer's weight delta per step, in a
   direction independent of the factor gauge (B S, S⁻¹A).
5. **Nothing driven by the training loss may raise the step**; per-step multipliers are clamped to at most 1.
6. **The normaliser and the step size never see the emphasis**: the gained gradient feeds the momentum, the raw one
   the second moment and the step size, so a gain survives normalisation per element and one level up.
7. **Noise levels are a budget relative to their natural shares**, not a scale to equalise; the top band is capped.
8. **No optimizer creates detail the cell, the VAE or the photos lack.** A Qwen-Image 2.1 token is one 16 px cell of
   64 latent channels (`regions.py:22`; `qwen21/loading.py:214-221`): about 25 mm of skin at full-body 1 MP framing,
   13 mm upper body, 10 mm on a torso crop [computed: a 1.70 m figure filling 90% of an 832 x 1216 frame is 0.64
   px/mm, so a 16 px cell is 25 mm; 0.9 m crown to hip and 0.7 m shoulder to mid-thigh over the same height give 13 mm
   and 10 mm]. Turbo never evaluates sigma below 0.4146 (`qwen21/sampler.py:20`), so skin is judged on the base
   sampler.
9. **A judge decides only after it passes validation, and a score that decides is never reported**: decisions read
   the decision set, reports the disjoint reporting set (5.6).

## 3. Shape of the design

`Hull(torch.optim.Optimizer)` in `rulztrainer/optimizers/hull.py`. It is not in `AUTOMAGIC`, so `group["lr"]` is the
step rho_t from the trainer's scheduler. The base rule H0 reads `p.grad` in `step()`, with no parameter hooks and no
`gather()`: it runs unchanged in the parallel race and with replica reduces, keeps `torch.compile`, and no hook keeps
a dropped optimizer alive (contract F4). The body mechanisms enter through one custom autograd Function in the LoRA
layer (HullLoRA) and through a few probe decisions.

| call | when | what for |
|---|---|---|
| `Hull(params, pairs, ref_norms, scales, **kw)` | `build_optimizer` | pairs (`lora.py:757-759`); frozen-weight norms and alpha / rank, one per pair, looked up by module name in `_make_optimizer` (7) |
| `observe(sigma, hook, kind)` | before each micro-batch and, `kind="dop"`, the DOP pass (`trainer.py:1161-1163, 1248-1249`) | the step's band, the DOP flag; locked |
| `lora.arm_hull` / `disarm_hull` | per micro-batch in `work()`, primary copy (replicas from M2) | token gains, face mask, layout, C3's metric, the excess sink |
| `zero_grad()` / `step()` | step start and out-of-memory retry / once per step | clears the excess and band record / H0, no host sync |
| `on_probe_record(rec, ema, weights)` | after each probe while C1 or P is on | C1's retreat (cosine EMA, held-out decision loss) and P; returns `"pull"` when it moved the weights |
| `host_keys` | `race.move_optimizer` | state that stays on the CPU (anchors) |

```
per micro-batch (work(), trainer.py:1157-1251; primary copy)
  observe(sigma, kind);  arm HullLoRA: w[t] = gain(part(t), band), face[t]      C1, C2 (face, text tokens: 1)
  each LoRA backward, one widened GEMM per factor:
    lora_up [G | X | F] = dᵀ [h | (w - 1) h | face h];  lora_down: the same with dh = d B and the input x
    dx raw;  p.grad += G;  X <- X - proj_F(X) where <F, X>_U < 0;  excess[p] += X                  C3
per step, per pair (B = lora_up [o, r], A = lora_down [r, i], s = alpha / r; stored factors never re-gauged)
  T   from the Grams, T M T = N                               virtual balanced frame (4.1)
  v   <- per-channel EMA of the raw gradient's square in that frame;  D = v^-½          (4.2)
  g^  = D (G + lam X), per-channel clip;  m <- b1 m + (1 - b1) g^       normalise first, then average
  U   = (m^_B T⁻¹, T m^_A)                                    the balanced update, mapped back
  eta = rho_t mu ||W0||_F / sqrt(u^),  u = EMA ||dW image of the RAW normalised gradient||²   (4.3)
  B, A -= eta U    (caps, finite flag, first-step branch: 4.4-4.6)
at each probe: C1's retreat (lam);  P (levels mu, face floor F2, copy and pose vetoes, optional lookahead)
```

With C1 off, or every gain 1 (HullLoRA then computes dA and dB with autograd's own r-wide products and skips the X
and F blocks, known from the gain table without a sync), Hull is H0 exactly (a test asserts it).

| | mechanism | state, 83,886,080 weights | step time | switch (default) | built |
|---|---|---|---|---|---|
| H0 | balanced frame, per-channel v, momentum after normalising, dW-unit step | 2.125 B/w, 178 MB (fp32 m: 346 MB) | est. +25-60 ms over a 94 ms null step [H], 0 syncs | `optimizer: hull` | M1a |
| C1-C3 | token gains in the LoRA backward; parts, ring, detail gate, flips refused; face-safe excess; the cosine and held-out-loss retreat | excess 2 B/w while armed | gate: at most +8% of step time | `hull_regions: off` | M1b |
| F2 | face floor on every probe decision | ~0 | per probe | with P | M3 |
| N | band budget relative to natural shares, top band capped | ~0 | ~0 | `hull_band_budget: off` | M2 |
| K | skin crops, base-sampler skin probes | 0 | data | `crop_augment` skin mode | M2 |
| P | probe levels, vetoes, F2, optional lookahead | anchor 336 MB on the host | per probe | `hull_probe_control: false` | M3 |
| phase 2 | body-cell purity gate, subject vs DOP, environment weights, phases | up to 4 B/w | 5.7 | off | M4 |

## 4. The base rule H0

### 4.1 Gauge and the virtual balanced frame

B = `lora_up` [o, r], A = `lora_down` [r, i], dW = s B A with s = alpha / rank (`lora.py:239`); g_B = s G Aᵀ and g_A =
s Bᵀ G for G = dL/d(dW). Under (B S, S⁻¹A) the product is unchanged while g_B -> g_B S⁻ᵀ and g_A -> Sᵀ g_A, so
per-element Adam's step depends on the gauge. Balanced factorisations of one dW differ only by an orthogonal Q: all
are (U Σ^½ Q, Qᵀ Σ^½ Vᵀ) for dW/s = U Σ Vᵀ, with ||B||² = ||A||² = the nuclear norm of dW/s [N].

**The frame.** With M = BᵀB + δI and N = AAᵀ + δI (δ = 1e-6 x trace / r, per Gram), the unique SPD S that balances the
pair, (BS)ᵀ(BS) = (S⁻¹A)(S⁻¹A)ᵀ, is S = T^½ with T = M^-½ (M^½ N M^½)^½ M^-½ (T M T = N, the geometric mean of M⁻¹ and
N). Hull never forms S and never re-gauges the stored factors. In the balanced frame the gradients are g_B S⁻¹ and S
g_A, so its per-channel statistics are rowsum((g_B T⁻¹) ∘ g_B) and colsum(g_A ∘ (T g_A)), and its update maps back to
the stored factors as (m̂_B T⁻¹, T m̂_A), the momentum kept in the stored frame. Only T and T⁻¹ appear.

**The derivation, checked.** Explicit rebalancing (rebalance after every update, momentum transported as m_B -> m_B
S⁻¹, m_A -> S m_A) carries the factors in a frame X_t relative to Hull's stored ones; both apply the same balanced
update, so X_t = S̃_t Q_t with S̃_t the stored pair's own balancing and Q_t orthogonal. Its momentum is then m̃_t
X_t⁻ᵀ = m̃_t S̃_t⁻¹ Q_t (Q_tᵀ S̃_t m̃_t for A): the virtual frame's, rotated by Q_t, to which the per-channel
statistics, both dW terms, the normaliser and the caps are blind. [N] At δ = 1e-12 x trace / r (1e-9 for the explicit
gauged starts), on a 96 x 128 rank-8 layer over 200 noisy steps the two rules agree to 2.1e-13 in fp64 and 4.8e-6 in
fp32 with 30 Newton-Schulz iterations; the virtual rule from (0, A0) and (0, S0⁻¹A0), cond(S0) 10, agrees to 6.3e-12;
explicit rebalancing from gauged starts agreed to 6e-7 over 60 steps, against 63-200% for per-element Adam. At the
default δ = 1e-6 x trace / r the two rules agree to 2.1e-7 in fp64, and gauged starts to 7.3e-6 at cond(S0) 10 and
4.5e-4 at cond(S0) 100 [N]: the gap grows linearly in δ.

**Buys and costs.** The stored factors, the `lora_ema` shadow (a factor lerp, `ema.py:53-61`), anchors and HullLoRA's
activations stay in one gauge, nothing is written back or transported, and two coupled Newton-Schulz roots per pair
suffice (M^½ with M^-½; P^½ with P^-½ for P = M^½ N M^½; T⁻¹ = M^½ P^-½ M^½). But the stored factors stay unbalanced
(||B|| 0.14, ||A|| 15.2 at step 200 against 1.4 and 1.4 explicit [N]), so Newton-Schulz sees worse-conditioned Grams,
and the factor-average gap is no narrower (factor against product averages, decay 0.98: 2.17-2.18% virtual, 2.09-2.10%
explicit, 0.88-1.01% Adam at an unmatched step [N]). **Fallback** `hull_frame: explicit`: rebalance in place with
`copy_` / `_foreach_copy_` (never rebinding `p.data`), transport `m` and the `lora_ema` shadow (bound once the EMA
exists, 7), and test that the shadow's product is unchanged.

**What the first real run showed, and the rebalance.** The buys above assumed the drift stays modest. The first real
run (Krea 2, rank 32, 2000 steps, `hull_rho` 0.0002, H0 as built) ended with ||A|| 1e5 against ||B|| 0.04 on the worst
layer, median ||A|| 531 against ||B|| 0.12; on a synthetic layer of the same shape ||A|| is 121-370 at step 100 against
6-8 under AdamW [N]. The frame has no restoring force on the gauge, so nothing stops it. On that worst layer ||dW|| was
about 1000 times smaller than ||B|| ||A||, a cancellation bf16 cannot hold: one more rounding to bf16 moved a saved dW
by 0.3% at the median layer, 5-9% at the 90th percentile and about 100% at the worst, against 0.23% at every layer for
the same dW balanced [N, the file's factors], and the
forward (`_added` runs in the activations' dtype, bf16) showed the same scale. So every `REBALANCE_EVERY` = 10 steps
Hull applies the fallback's rebalance in place: B <- B X, A <- X⁻¹ A with X = T^½, in fp64, m_B <- m_B X⁻¹, m_A <- X m_A,
T <- I, and the shadow of a bound `lora_ema` the same way, so its product stays. v and u do not depend on the gauge. It
leaves dW's path alone with fp32 momentum: the same step, to two decimals, over 64 steps at rank 32, and within 1.64 rho over 800
steps at gradient ranks 1-16 [N]. **With bf16 momentum it does not.** X can be ill-conditioned, so m's rounding (0.4% of
an element) is multiplied by it at every rebalance: on a rank-2 gradient ||m_A|| reached 3.9e5 against 680 at step 30
and the step sat on the safety cap at 4 rho, along that noise [N]. An earlier check at gradient rank 16 missed this. The
trainer therefore always builds Hull with fp32 momentum (`hull_state_dtype: bf16` is ignored with a warning), at about
346 MB against 178 MB. **Pairs with a rank above min(o, i) are never rebalanced.** Krea 2's `text_fusion.projector` is
12 -> 1, so at rank 32 BtB and AAt can never meet: X stayed huge at every rebalance and multiplied itself into the
`lora_ema` shadow. The live weights stayed healthy, so the loss stayed flat, but the shadow's dW on that layer reached
2.5e7 x ||W0|| by step 100, and saves and probes use the shadow. That gave wrong subjects and abstract noise on v2.36.2
and v2.36.3. Reproduced on the real model at 210 steps: faces became buildings and bodies abstract noise; with the skip
the shadow's largest relative dW is 0.009 and the probes are sound [real run, one seed]. That pair's ||A|| still grows
(3.7e3 against ||B|| 0.03 at step 210), as all pairs did before the rebalance.
**The rebalance does not change dW's path, so it can only remove the precision damage.** The first run's loss and raw gradients also
turned at about step 1250, and whether that came from the precision or from the path itself is not settled. Its
eigendecompositions sync the host once in 10 steps (5 ms for 257 pairs at rank 32 [N]).

### 4.2 Per-channel second moments, normalise first, then momentum

```
q_o = rowsum((G_B T⁻¹) ∘ G_B) / r       q_i = colsum(G_A ∘ (T G_A)) / r            raw, balanced frame
v <- b2 v + (1 - b2) q;  k <- k + 1;    D = (v / (1 - b2^k) + eps)^-½               eps = 1e-12 mean(v) + 1e-30
g^_B = D_o (G_B + lam X_B)              g^_A = (G_A + lam X_A) D_i                  X = 0 without C1
m <- b1 m + (1 - b1) g^;   m^ = m / (1 - b1^k)                                       b1 0.9, b2 0.99
```

Rows of B and columns of A are the gauge-free axes (row norms of g_B Q equal those of g_B), so v is O(r)-invariant,
unlike per-element Adam or Adafactor's rank-axis factor. Normalising first gives each photo of a batch-1 stream a
comparable say in the momentum, as in ADOPT ([2411.02853](https://arxiv.org/abs/2411.02853)), but Hull's v includes
the current gradient (no zero-initialised channel divides by about zero), which voids ADOPT's convergence result. `G`
is the raw `p.grad` (the step's mean at K > 1) and `X` HullLoRA's side buffer (5.1), so armed and unarmed paths share
one statistic. On pure noise the normalised momentum is about sqrt(0.1 / 1.9) = 0.23 of a consistent one [computed].

### 4.3 The step in dW units

```
u   <- b2 u + (1 - b2) s² ||P A + B Q||²_F    P = (D_o G_B) T⁻¹,  Q = T (G_A D_i)           RAW normalised gradient
       ||P A + B Q||² = tr(PᵀP AAᵀ) + 2 tr((PᵀB)(Q Aᵀ)) + tr(Q Qᵀ BᵀB)                    r x r Grams
eta = rho_t mu ||W0||_F / sqrt(u / (1 - b2^k))
```

u never reads the gained gradient: gains change where a layer's step goes and add to it, while its size stays the one
rho asked for; a consistent momentum moves a layer by about 1.25-1.5 rho per step, a pure-noise one about 0.25 rho.
[N] With a consistent gradient the realised relative step was 1.46-1.53 rho over steps 6-20 at ranks 4, 8, 16 and 32,
with alpha = rank and alpha = 16, and 1.24-1.25 rho over steps 50-200 at rank 16; a pure-noise layer moved 0.25 rho
over steps 50-200. The first normaliser tried, the factor-norm proxy s sqrt(o ||A||² + i ||B||²), drifted to 2.39 /
3.77 / 4.93 / 7.29 rho at ranks 4 / 8 / 16 / 32 once dW had a dominant direction and was dropped; dividing by the
update's own norm would move every layer by rho whatever its coherence. ||W0||_F comes once per module name from the
column norms DoRA caches on the shared wrappers (`lora.py:298-303`, dequantised by `lora.py:163-167`), since the
parallel race's master views have no `.base` (`race.py:572-576`). The ancestor is LAMB's trust ratio
([1904.00962](https://arxiv.org/abs/1904.00962)). In LoRA-RITE's LoRA fine-tuning tables it beat Adam by 2.85 and 5.98
points on the LLM-benchmark averages (Gemma-2B, Gemma-7B; per-benchmark gaps +1.04 to +15.02, Table 2), and its gaps
to Adam on Super-Natural Instructions ran from -0.67 to +3.63 (Table 1)
([2410.20625](https://arxiv.org/abs/2410.20625)) [3P]; unmeasured on diffusion LoRAs [H], so `hull_step_scope: global`
is row R1b.

### 4.4 Caps

- **Factor caps**: eta <= kappa_f ||B̄|| / ||Ū_B|| and kappa_f ||Ā|| / ||Ū_A|| in the balanced frame (||B̄||² = tr(M
  T), ||Ā||² = tr(N T⁻¹), ||Ū_B||² = tr(m̂_Bᵀm̂_B T⁻¹), ||Ū_A||² = tr(m̂_A m̂_Aᵀ T)); kappa_f = 0.2 bounds the dB dA
  term, and in the check it bound only at steps 2-6 of 200 [N].
- **Safety cap** on the exact step: with D1 = s (U_B A + B U_A) and D2 = s U_B U_A (norms from Grams), eta is at most
  the positive root of eta ||D1|| + eta² ||D2|| = kappa rho_t mu ||W0||, a triangle-inequality bound on the realised
  step. kappa = 2 max(1, gain max), at least 1.5x the largest gain (a consistent layer reaches about 1.5 rho at gains
  1 [N], so a cap near 1.5 would bind by design). Both bind rates are logged; **M1a fails if the safety cap binds on
  more than 2% of layer-steps after step 20.**

### 4.5 Zero-initialised B

At step 1, B = 0, g_A = 0 and no balance exists. Hull moves B alone in the gauge-invariant direction: U_B = the rows
of G_B (AAᵀ + δI)⁻¹, each normalised by its dW-space norm; U_A = 0; eta = rho_t mu ||W0|| / ||s U_B A||. Then dW_1 =
-rho_t mu ||W0|| R / ||R||_F with R the row-normalised G P_A (P_A projects on A's row space; ||R||_F = sqrt(o)):
||dW_1|| = rho_t mu ||W0|| exactly (1.00 of it at every rank and alpha [N]). m, v, u and k then restart. The branch
is selected on the device (`torch.where` on ||B|| = 0, the solve by `torch.linalg.solve_ex`). PoLoRA also holds A
until B is non-zero ([2607.17620](https://arxiv.org/abs/2607.17620), appendix) [P].

### 4.6 Numerics, precision, syncs

- **The roots run in fp64, not fp32 (measured while building H0).** A rank-4 update held at rank 32 at the default
  damping has Gram eigenvalues under fp32's rounding error: an fp32 Gram has negative ones and the iteration diverges to
  NaN or a residual of 1e8, at every held rank below 32. Grams and roots in fp64 hold a residual of 1e-9 from fp32
  factors; T, T⁻¹ and everything after them stay fp32. The fp32 figures in 4.1 do not carry over to low-rank factors.
- **No host sync in `step()`.** In torch 2.9.1 the docstrings of `torch.linalg.eigh`, `solve`, `inv` and `cholesky`
  state that they synchronise a CUDA device with the CPU, the `_ex` forms only with `check_errors`. The roots are
  coupled Newton-Schulz iterations (Higham, *Functions of Matrices*, ch. 6,
  [doi:10.1137/1.9780898717778](https://doi.org/10.1137/1.9780898717778)), a fixed 30 per root in fp64 (above), on one stacked
  [n, r, r] batch per device across all shapes. [N] A rank-4 update held at rank 32 needs 30: the S error was 0.93
  after 10 iterations, 0.48 after 20, 2.4e-9 after 30.
- **Fallback.** A pair whose residual ||T M T - N|| / ||N|| exceeds 1e-3 keeps the previous step's T (stored,
  initially I) by a device-side select, counted in the log.
- **Finite flag.** One NaN makes every gradient NaN after `clip_grad_norm_` and `step()` still runs (contract F1
  [bench]); a device-side flag skips the update and keeps every moment bitwise.
- **Precision.** Pairs are fp32 masters (`lora.py:223-226`); m is bf16 with stochastic rounding
  ([2010.06192](https://arxiv.org/abs/2010.06192)); v, T and u are fp32.
- **Hot path.** Work is stacked per (o, i, device) bucket and written back with `_foreach_copy_`, as
  `optimizers/automagic4.py:243-256` does: about 12 r x r Gram products per pair and two roots on one [224, 32, 32]
  batch, est. +25-60 ms over the 94 ms null step [H] (fused AdamW +2, foreach +21, Automagic 4 +66, bnb AdamW8bit +82
  ms [bench]). M1a's bar: +80 ms, 0 syncs. **Measured on H0 (synthetic tensors at this LoRA's shapes, four buckets, the
  cards shared with other work):** 165 ms a step against fused AdamW's 16 on the RTX 4060 Ti, about 115 against 15 on
  the RTX 5060 Ti, 0 syncs by `set_sync_debug_mode`. So the bar is missed by roughly a factor of 2 on the 4060 Ti; of the
  165 ms about 50 are the fp64 Grams, 30 the roots, the rest the update. The real step's +6% or less at 2.5 s or more
  keeps it inside section 10's +10%.

### 4.7 Clipping

Hull sets `max_grad_norm` to 0, so `step_grad_norm` measures and logs the norm without clipping
(`trainer.py:320-331`); a global clip on factor gradients is gauge-dependent anyway. Instead each row of g^_B and
column of g^_A is scaled to an RMS of at most c = 5 measured in the balanced frame (rowsum((g^_B T⁻¹) ∘ g^_B) / r and
its column twin), an O(r)-invariant per-channel clip; per-element values in the stored frame are gauge-dependent.

### 4.8 Step size: what sets it, and why it cannot run away

rho_t = `group["lr"]` = `hull_rho` x schedule(t); the step uses rho_t mu. `hull_rho` seeds `group["lr"]`; `LambdaLR`
applies warmup and the schedule (`trainer.py:421-438`) and rewrites `group["lr"]` every step (`trainer.py:1380`); mu,
P's level, is at most 1 and multiplies inside `step()`, never through `group["lr"]`.

- **One-sided.** Per-step multipliers (true per-image lr, `clean_share`'s half rate: `trainer.py:1364-1369`) are
  clamped to at most 1; refused: `loss_watch_healthy_mult` above 1 (`config.py:550`, `losswatch.py:237`),
  `likeness_lr_control`, which can raise the lr through `apply_lr_ratio` (`likeness.py:468-482`), and
  `likeness_curriculum`, which moves sigma on the probe trend. No weight decay (it pulls dW toward the base by about 2
  eta lambda per step; `weight_decay` is ignored, logged); LoRA+ groups collapse as `trainer.py:374-377` does.
- **Bounded.** rho_t mu <= `hull_rho` x schedule; the realised step is at most kappa rho_t mu ||W0||; noise layers
  move about 0.25 rho [N]; P only lowers mu. **Calibrated:** `hull_rho` has no default; M0 logs the tuned AdamW
  control's realised relative dW step per layer (median, late steps), and R1 races 0.5x, 1x and 2x of it.
- **Logged.** Hull defines no `get_avg_learning_rate`, so `current_lr` shows rho_t (`trainer.py:414-418`); mu is
  logged per probe. The endpoint is picked afterwards on held-out judges under the copy guard, with `posthoc_ema.py`'s
  product averages (its docstring, lines 16-19).

### 4.9 Metrics and kinds

`hull_metric: balanced` is H0. `riemannian` (R1b) normalises g_B (AAᵀ)⁻¹ and (BᵀB)⁻¹ g_A by dW-space channel norms;
[N] undamped at rank 32 its step fell to 0.04 rho (1,700 rho with the caps opened); damped by 0.01 it held 1.9-2.4 rho
but invariance fell to 1e-2 to 1e-3. `adam` keeps per-element v, no frame, the same dW-unit step and caps: the
fallback base (section 11). LoKr, OFT and DoRA are refused in M1; zero-element tensors (`lora.py:338-349`) are
skipped.

## 5. The body mechanisms

| trait | where Hull puts the update | outside the optimizer | guard |
|---|---|---|---|
| silhouette, proportions | ring and body-skin gains; band columns after M0 | `timestep_sampling` from R0a | top band <= 25%; pose and copy vetoes |
| build, mass | body-skin gain in the person's interior | fitted clothing, several outfits | clothing gain <= 1 |
| skin tone, detail, marks | detail-gated skin gain; band columns after M0 | skin crops at native density; no flips | VAE ceiling; base-sampler probes |
| face | gain 1; gain-free step; C3 | none | F2 |

### 5.1 C1: per-token region gains in the LoRA backward (HullLoRA)

**Idea.** Each image token gets a gain from its region (after M0, also its sample's band), applied only inside the
weight-gradient products; the gained gradient feeds m, the raw one v and u. A weight fed only by body tokens moves g
times as far in normalised units, a face-only weight ignores the gains, and the layer's step size does not change: the
per-element emphasis Keel C left for phase 2, with no need for feature channels to specialise by region.

**Mechanism.** A custom `torch.autograd.Function`, HullLoRA, replaces the two linears of `LoRALinear._added`
(`lora.py:392-401`) while armed; the forward is unchanged. The backward computes dh = du B and dx = dh A raw, so the
network upstream sees autograd's gradient; one GEMM du_tokᵀ [h | (w - 1) h | face h] -> [out, 3r] gives lora_up's raw
gradient, excess X and face part F, and [dh | (w - 1) dh | face dh]ᵀ x -> [3r, in] does the same for lora_down with
the x autograd saves anyway. The raw parts go back to autograd (`p.grad` stays raw); X, after C3, joins a bf16 excess
buffer per parameter. w and face are per-sample token tensors built in `work()` from the parts tensor (C2), the
dilated face box and the alignment taps' layout (`alignment.py:48-60`; a bare `Taps()` as `trainer.py:2892-2895`
installs for Keel), which the model sets only while the taps are armed (`qwen21/model.py:334-335`), so `work()`
begins, stops and ends them for Hull too (7). Text tokens keep gain 1: in this single-stream model they pass the same
linears, modulated at t = 0 (`qwen21/model.py:312-317`). The gains are ctx attributes, not saved tensors, so under
non-reentrant checkpointing (`qwen21/model.py:277-279`) the one backward uses the forward's gains (tested as
`tests/test_keel.py:897-924`). Arming is per micro-batch on the primary copy, as Keel's tap
(`trainer.py:1170-1186, 1244-1245`).

**Default once M1b passes:** one uniform gain g_body = 1.5 on body cells, flat across noise bands. Body = body skin
plus the silhouette ring (C2), never clothing or hair; clothing and hair at most 1 (hard cap); face and text exactly
1; background 1; every gain within [0.5, 2]. The region x band table and any share targets are filled in only after M0
has measured when traits are decided on Qwen-Image 2.1.

**Non-conservative [H].** sum_t w_t d_t h_tᵀ is the gradient of no objective: attention mixes tokens and d_t is not
reweighted. It is the factor gradient of a LoRA with strength w_t per token, taken with the forward at strength 1, so
its fixed point is not the diffusion loss's and it can drift once the true gradient is small. Hull logs the per-tensor
cosine between gained and raw normalised gradients (balanced metric, tr(g^ᵀ g^_w T⁻¹) on the B side) and, at every
probe, the raw diffusion loss on the held-out decision photos (fixed sigmas and noise, no gains). **Retreat:** a
tensor whose cosine EMA stays negative over a probe interval halves its excess scale lam (0-dim, in state); every
tensor halves lam when the held-out loss rises beyond its A/A noise while the training loss falls, and on an F2
breach. lam never rises.

**Unarmed paths.** Replicas (in M1b) and the parallel race run without HullLoRA; `p.grad` is raw either way, so they
compute the same v and u (tested on the gains-1 path of 3: armed H0 at gains 1 equals unarmed H0 at K = 1 and K = 2);
the armed share is logged. The parallel race refuses C1, C3 and band-budget rows instead of silently running H0.

**Cost.** No state; the excess (2 B/weight) and widened operands while armed. One extra r-block is 2 T r (sum of outs)
= 0.38 TFLOP for lora_up and 0.31 TFLOP for lora_down per 4,096-token sample [computed: outs sum to 1,441,792, ins to
1,179,648 over the 224 modules]; C1 with C3 adds two. FLOPs did not predict the memory-bound tap [bench], so C1 is
unmeasured until M0's gate: at most +8% of whole-step time with the real model on both 16 GB cards, compile on and
off, 0 syncs. Compile stays off while armed (`trainer.py:1065-1067`) unless a `torch.library` custom op keeps it on.
First reading, one RTX 5090 at 0.25 MP, one run each, same seed, steps 25-80, no probes: armed +8.6% over H0 with compile
off, +2.2% over H0 with compile on (compile itself was +6.3% here), peak memory unchanged at 23.8 GB. So the gate is
close and unsettled: the 16 GB cards and 1 MP are unmeasured, and syncs were not counted.

**First reading of M0(d)** (`scripts/hull_agreement.py`). One fictional subject, not the synthetic subjects
M0 calls for, 0.25 MP, 800 steps, seeds 42 and 1234, plain Hull, parts of G in the balanced frame with Hull's normaliser,
unit norm per step. A random-sign copy of the body part is the control: it reads 0.228 on tensor coherence, the 0.23
floor, and 0.24 on element purity, since real gradients are heavy tailed. Medians over the 448 of 514 tensors that see
image tokens, steps 100-800, above that control, seed 42 / 1234:

| part | coherence | purity |
|---|---|---|
| body | +0.0037 / +0.0031 | +0.0044 / +0.0035 |
| face | +0.0038 / +0.0039 | +0.0034 / +0.0036 |
| background | +0.0025 / +0.0021 | +0.0055 / +0.0046 |
| whole gradient | +0.0036 / +0.0036 | +0.0011 / +0.0007 |

The region gate, body above background, fails on purity on both seeds (body led in 38% and 43% of tensors, about a third
late on), though coherence leans the other way. G's condition, body at or above face, is a tie. The excess above the
floor means about 0.2% of a step's gradient energy is consistent [computed], and that holds for the whole gradient too,
so a low reading is not proof that a part carries no signal. Body parts hold about 17% of the gradient's energy, face
26%, background 26%. Two runs of one config are not identical: one tensor's reading differed by about 0.005 between
them over the first 40 steps. The EMA window mixes photos with noise draws and timesteps, so this cannot say which of them causes the
disagreement, and the snapshots overlap, so differences this small are not resolved. It fits the races, where gain 1.5
did not raise the body, but does not explain seed 42's loss.

**Risks.** Body gradients that disagree across poses and outfits make the gain amplify nuisance (M0's agreement gate);
the token path says where the LoRA acts, not which loss asked (M0 compares loss-path attribution); unarmed replicas
dilute the gain (M2 arms them; races use one card until then).

**Evidence.** Loss-side region balancing under Adam gave small gains: FairHuman's face detection confidence 85.31
against 84.82 ([2507.02714](https://arxiv.org/abs/2507.02714)) [P]; DBS's region-aware loss moved DINOv2 from 0.3024
to 0.3031, its timestep gating to 0.3143 (Table 2), on FLUX.1-dev at per-step batch 1 with 2-step gradient
accumulation (effective batch 2) and 8-bit AdamW ([2607.25622](https://arxiv.org/abs/2607.25622)) [P]. That emphasis
after the normaliser does better is C1's hypothesis [H]; Keel's per-tensor gain was never raced.

**Fallback R.** If HullLoRA fails the gate or cannot compile: per-output-channel additive shares through the existing
tap (p_k = <G_k, G> / ||G||², gains on lora_up's rows after normalising), only if M0 shows output channels, which
every token passes through, specialise by region beyond sampling noise.

### 5.2 C2: regions from a parts tensor

**Idea.** The silhouette carries widths and lengths [H]; skin carries tone and marks; clothing and hair are per-photo
nuisance, so they never take a gain above 1. **Signal, the parser.** The MediaPipe multiclass selfie segmenter: Apache
2.0; classes background, hair, body-skin, face-skin, clothes, others; selfies and full-body images; mean IoU 81.10 for
the 512 px model, lowest on Monk tones 9-10 (71.86)
([card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf)) [P]. The card
puts identity recognition out of scope; Hull uses it for training masks only, and the judges' skin masks come from SAM
3 ([2511.16719](https://arxiv.org/abs/2511.16719)), so one mask error cannot bias both sides.

**Mechanism (cache time).** Per 16 px cell, the six class fractions as uint8 in a new `parts` tensor of the region
file, plus a seventh channel, the silhouette ring: the person's cells (background below 0.5) dilated minus eroded by
`hull_ring_cells` (1), outside the dilated face box (`regions.py:233-236`). `save_region_map` already writes a tensor
dict and metadata (`regions.py:247-251`); `RegionMap` and `load_region_map` gain `parts`
(`regions.py:212-219, 254-264`); a fingerprint (parser, model size, ring, face margin, threshold, bucket) goes into
the metadata and into a race's region keys (`race.py:94`), and a mismatch rebuilds the map (today maps are rebuilt
only on `recache` or when missing, `regions.py:292`, and the subject is defined by a weight equality,
`regions.py:237`). Token gain: max(g_ring, base) on ring cells, else base = sum_c f[c] g_c; face-box cells forced
to 1. Without parts (`hull_regions: basic`), today's maps (`lora.py:710-733`) can only attenuate background, since
their "body" is the person minus the face, clothing included.

**Cost, risks.** One parser pass per image, once. Accuracy on full-body photos and darker skin is unmeasured (M0
scores cells against the exact masks of the synthetic renders and against hand-labelled full-body photos of the
consenting adults, section 10, reported by Monk tone). The ring at high noise is largely pose, so the pose-following
judge guards every ring gain.

### 5.3 Face safety

1. **Face gain 1** under every table and controller.
2. **The step size ignores the gains** (4.3): a body boost adds to the step instead of taking the face's share.
3. **C3, the face-safe excess.** Per factor, micro-batch and tensor, with <F, X>_U = tr(F_Bᵀ D_o X_B T⁻¹) on lora_up
   and tr(F_Aᵀ T X_A D_i) on lora_down (the previous step's D and T): where <F, X>_U < 0, X <- X - (<F, X>_U / <F,
   F>_U) F, so the excess cannot undo the face's part of the step to first order. Device-side selects; a no-op at
   gains 1. PCGrad projects a whole task gradient off a conflicting one
   ([2001.06782](https://arxiv.org/abs/2001.06782)); projecting only the gained excess is new [H].
4. **F2, the face floor** on every probe-driven decision (a level change, keeping a gain, a lookahead accept, the
   final checkpoint): the decision set's raw FaceNet cosine, paired with the anchor on the same prompts and seeds,
   must pass mean(d) - 1.64 sigma_face / sqrt(n) > -delta_face, delta_face in raw FaceNet cosine from the A/A pilot's
   run-to-run spread. A breach lowers mu one level and halves every lam.

### 5.4 N: the noise-band budget

**Idea.** Layout, silhouette, pose and proportions are decided at high noise and texture at low noise (P2
[2204.00227](https://arxiv.org/abs/2204.00227), T2I-Adapter [2302.08453](https://arxiv.org/abs/2302.08453), Readout
Guidance [2312.02150](https://arxiv.org/abs/2312.02150)) [C]; the highest timesteps memorise poses and backgrounds
(T-LoRA, [2507.05964](https://arxiv.org/abs/2507.05964)) [C]; in one user's on/off test, a separately trained
high-noise LoRA for Wan 2.2 made body type and face shape slightly more faithful
([musubi-tuner #455](https://github.com/kohya-ss/musubi-tuner/discussions/455)) [A]. None of it is measured on
Qwen-Image 2.1 or on skin. A shared v sets a band's share by its gradient scale, Keel's normalisation by its visit
rate; neither decides it for the body. Visits at 4,096 tokens, provisional edges [computed from `TURBO_SIGMAS`, the
25-step `qwen21_sigmas` (`qwen21/sampler.py:20, 32-43`) and logit-normal(0, 1), unshifted or shifted by mu = 0.6935 as
`reference` does (`archs.py:586-593`)]:

| band | sigma | Turbo, 8 evaluations | base, 25 | visits `logit_normal` | visits `reference` |
|---|---|---|---|---|---|
| B4 | > 0.88 | 5 | 6 | 2.3% | 9.7% |
| B3 | 0.65-0.88 | 2 | 7 | 24.5% | 43.3% |
| B2 | 0.35-0.65 | 1 | 6 | 46.4% | 37.6% |
| B1 | 0.12-0.35 | 0 | 4 | 24.5% | 9.1% |
| B0 | < 0.12 | 0 | 2 | 2.3% | 0.4% |

**Mechanism (M2).** With the budget on, a step's micro-batches and DOP pass share one sigma (`pairs_sigma`,
`trainer.py:302-317`, made a capability; the DOP job takes it, `trainer.py:2733-2734`). M0 measures the natural shares
(per band and layer group, realised dW-step energy of the A/A pilot's AdamW control, one shared v; M1a's R1 runs
re-log them under H0 at gains 1); targets are set relative to them and the realised shares, gains included, are
logged. A slow controller moves the band columns of the non-face, non-text gains toward the targets (log g += 0.125
log(target / realised) per probe, within the clamps; clothing at most 1), and the highest band is capped at 25% of
gained update energy. The pose-following judge guards every ring and high-band gain through P's pose veto. `min_snr`
draws a warning: it weights sigma 0.9 at about 1/64 of sigma 0.3 (`trainer.py:346-354`) [computed]. **Cost, risks:**
per-band EMAs, about nothing; high-band emphasis memorises poses (hence the cap and the pose guard), low-band emphasis
copies softness (hence the detail gate). **Evidence:** uncertainty weighting over five timestep-interval clusters
(ANT-UW) took DiT-L/2 on ImageNet from FID 12.59 to 6.17, against 9.58 for Min-SNR
([2306.00354](https://arxiv.org/abs/2306.00354), Table 2) [P, pretraining, equalising]; whether Qwen-Image 2.1's
high-sigma gradients are larger, so that AdamW already favours them, is unmeasured [H].

### 5.5 K: the skin path

**Idea.** Skin texture and marks are fine detail, decided at low noise (5.4) [C] and resolved only at native pixel
density (rule 8), so skin needs photos where a cell covers millimetres and probes that reach low noise.

- **Detail gate (M1b, per item).** Round-trip the item through the VAE at its training size
  (`qwen21/loading.py:214-249`); if its body-skin cells keep at least half their high-band energy (Laplacian bands at
  1-4 px) the gate is 1, otherwise its skin gain is 1 at every band (the ring is unaffected): soft photos would
  otherwise teach softness [H].
- **Skin crops (M2).** `crop_augment: skin`, a crop mode beside `augment_items`' face and bust crops
  (`dataset.py:204-262`): torso and limb boxes from the MediaPipe pose landmarks (`bodysim.py:350-375`), squares never
  upscaled (skipped below the minimum side as `dataset.py:236-238` does), at least 30% body skin, with their own
  captions, latents, parts and gate, weighted by `crop_augment_weight` (`config.py:585`).
- **Flips refused with parts** (M1b): `flip_augment` mirrors latents and maps (`dataset.py:521-522`,
  `regions.py:267-275`), and marks are asymmetric.
- **Skin probes** use the base sampler (25 steps to sigma 0.02, a RenderJob flag: 7) on native-resolution patches;
  below sigma 0.41 a Turbo render's skin is the turbo LoRA's one-step guess.
- **The ceiling.** A mark a few millimetres across is a few pixels inside one cell at full-body framing. The closest
  published figure is the RGB Qwen-Image-VAE-2.0 f16c64: PSNR 39.14 on FFHQ faces at 1K
  ([2605.13565](https://arxiv.org/abs/2605.13565), Table 2) [P]. The report calls the VAE inside Qwen-Image-2.0 an
  intermediate variant and does not cover Qwen-Image 2.1's own RGBA VAE (`archs.py:8`; `qwen21/loading.py:8, 216`), so
  the ceiling of the VAE Hull trains through is unmeasured, and nothing measures skin through either; M0 measures it
  on the subjects' own photos.
- **Dataset advice.** Photos that show the build and visible skin: about a third each face, upper and full body;
  several outfits, fitted ones among them; several poses and lights; originals at or above the training resolution.

**Cost.** One VAE round trip per item at cache time for the gate; crops add items with their own latents, parts and
gate. **Risks.** Crops that dominate the high-noise band teach framing [H]; a gated-off soft photo still trains at
gain 1. **Evidence.** None on skin: the VAE figure is a face benchmark, so M0(f) measures skin retention.

### 5.6 P: probe control

**Sets.** A decision set D and a reporting set R, disjoint in photos, prompts and seeds (validation refuses overlaps).
D: per subject, prompts written from held-out photos (full body, upper body, face, one novel pose; never from training
photos, which rewards copying), 4 fixed seeds each [starting value]; structure and face on Turbo, as the LoRA is used,
skin on the base sampler every second probe. R: other held-out photos, 8 fresh seeds, every judge, fixed late
checkpoints and the final file, then a blind human A/B; the controller never sees R. Strangers: 8 base-model renders
per prompt, the trainer's current count (`bodyanchor.py:50`), a starting value the A/A pilot re-derives (section 10).

**Statistic.** Per decision judge j, paired with the anchor on the same prompts and seeds, d_jk = y_jk - y_jk(anchor);
sigma_j pools the A/A pilot's sigma_AA,j with the run's seed spread; z_j = mean(d_j) sqrt(n) / sigma_j and Q = sum_j
w_j z_j / ||w|| over the validated primaries. **Accept** when Q > 1 (one standard error of the paired differences, the
parameter-free Ladder's margin, about the 0.15 level, [1502.04585](https://arxiv.org/abs/1502.04585)), the F2 and
other non-inferiority guards hold and no veto fires.

**Levels.** mu in {1, 0.5, 0.25}, one way: down a level on a copy veto, a pose veto, an F2 breach, or after
`hull_patience` (3) probes with no accept and no rise since the previous probe (a steady climb is never cut, as in
Keel D); past the lowest level the run stops with its best accepted checkpoint. The training loss never moves mu.
**Vetoes.** Copy: the SSCD best cosine to the training photos (`copycheck.py:39-42` sets its limit) or the body
copy-paste index rises past its A/A spread from the anchor; the index is WithAnyone's M_CP = (theta_gt - theta_gr) /
max(theta_tr, eps), defined on faces ([2510.14975](https://arxiv.org/abs/2510.14975)), and its body version (anchored
canonical window, a held-out photo as ground truth) is new [H]. Pose: the novel-pose prompt's keypoint and limb-angle
error (MediaPipe pose) rises past its A/A spread; it also halves the excess (g - 1) of the ring and high-band gains,
so they never fall below 1.

**Lookahead (optional, off).** The anchor is the weights that scored (the EMA shadow, or the copy an async probe
scored, `trainer.py:3930-3959`), in pinned CPU memory in `state[p]` and in `host_keys`, which `move_optimizer`
(`race.py:705-711`) skips. After `hull_grace` (2) misses without a rise: Q = polar(BᵀB_a + A A_aᵀ) by Newton-Schulz
(orthogonal Procrustes), B <- B_a + (1 - p)(BQ - B_a), A <- A_a + (1 - p)(QᵀA - A_a), p = 0.5, the shadow likewise,
m_B <- (1 - p) m_B Q, m_A <- (1 - p) Qᵀ m_A [H]. p = 1 restores the anchor exactly; the product error
p (1 - p)(BQ - B_a)(QᵀA - A_a) is bounded in a test. Product-space interpolation truncated to rank r (QR, SVD of the
2r x 2r core, Procrustes back) is the exact but syncing option.

**Rules.** Every judge, the existing anchored body score included, must pass validation (section 6) before it may
decide; a failing judge stays a log line (without a validated body judge P runs F2 and the vetoes only). P's state
(mu, counts, the anchor's per-seed scores) is mirrored in `state[p]` as 0-dim or tiny tensors, re-read every step and
median-unified after a relay (as `optimizers/automagic4.py:189-203`); lam is per tensor (5.1), carried as it is and
never unified.

**Cost, evidence.** Probe cost is unmeasured on 16 GB cards; `probe_async` moves it to the second card, which then
cannot train. Ancestors: PBT ([1711.09846](https://arxiv.org/abs/1711.09846)) and Lookahead
([1907.08610](https://arxiv.org/abs/1907.08610)) [P]. Reusing a holdout adaptively overfits it
([1506.02629](https://arxiv.org/abs/1506.02629)) [P], hence D and R; identity rewards are gamed (IPRO: 58% of outputs
hacked without its KL term, 10% for the full method with the KL term and its reference pool,
[2510.14255](https://arxiv.org/abs/2510.14255)) [P], hence few, thresholded decisions.

### 5.7 Phase 2: raced options (M4, all off)

- **G, cross-step purity on body cells.** a = |m_b| / n_b, equal-decay (beta 0.9) EMAs of the normalised body-cell
  part of the gradient and of its absolute value (two bf16 buffers, 4 B/weight); the gate f + (1 - f) a^p (f 0.5, p 2)
  scales the body excess, keeping the emphasis where body gradients agree across photos. GradDrop's purity P gives
  |sum g| / sum |g| = |2P - 1| ([2010.06808](https://arxiv.org/abs/2010.06808)); AND-mask
  ([2009.00329](https://arxiv.org/abs/2009.00329)) and SAND-mask ([2106.02266](https://arxiv.org/abs/2106.02266)) move
  weights only where environments agree. [N] At beta 0.9 pure noise reads 0.23 and SNR 0.5 reads 0.55; an element seen
  in 30% of photos keeps its purity (0.88 -> 0.93 at SNR 1) while Adam's |m̂| / sqrt(v̂) halves (0.71 -> 0.37); one
  10-sigma photo makes 74% of noise elements read above 0.5 (7% without). **Only if M0 shows body-cell purity at or
  above face-cell purity**; otherwise it favours the face. Stored-frame purity is not O(r)-invariant [H]. First reading:
  a tie (section 5.1).
- **D, subject against DOP.** DOP's own backward (`trainer.py:2738-2757`) is flagged by `observe(..., kind="dop")`; a
  post-accumulate-grad hook armed only inside a step (Keel's pattern) routes it to a second accumulator (4 B/weight on
  DOP steps). Per tensor, when <G_s, G_d>_U < 0, `pcgrad` projects both, `protect_prior` the subject gradient only
  (A-GEM, [1812.00420](https://arxiv.org/abs/1812.00420)), `off` neither. Weighted sums of the losses
  ([2209.11379](https://arxiv.org/abs/2209.11379)) and even the plain sum with standard regularisation
  ([2201.04122](https://arxiv.org/abs/2201.04122)) often match projection methods [P]; for the body [H]. It also
  measures whether DOP pulls the body toward the base model's body.
- **E, environment-balanced loss weights** (data side, no state): clusters by pose (MediaPipe landmarks) or outfit
  (clothing-cell colours); w_e = (1/E) / freq(e), mean 1, clipped to [1/2, 2], beside `loss_scale`
  (`trainer.py:1223-1225`, which reaches replicas) and in `race_parallel.forward_backward` (`race_parallel.py:60-72`)
  for the parallel race; framing stays off. [H]
- **Phases: structure, then identity, then detail.** S: proportions and build with high-band and ring gains; I: those
  and the face, natural bands; D: skin, with low-band skin gains and crops, only with a validated skin judge. A phase
  ends after 4 probes without an accept; the next starts from the anchor at the current level (mu never rises), and
  earlier primaries guard later phases. Easy-to-hard timestep ordering helped in
  [2403.10348](https://arxiv.org/abs/2403.10348) [P]; here [H].

### 5.8 Other model families

All families share the 16 px cell (`archs.py:1-10`), so parts and gains carry over. MiniMax H3's tokens cover 2 x 2
cells (`h3/model.py:536-545`): a token's gain is its cells' mean. Text-only inputs (Krea 2's text-fusion targets) stay
unarmed. Each family's distilled sampler has its own lowest sigma, so the skin-probe rule is checked per family, and
`reference` follows each family's recipe (`reference_sigmas` in `archs.py`). State scales with the weights.

## 6. Measurement

| trait | held-out judge | status |
|---|---|---|
| proportions | SAM 3D Body ([2602.15989](https://arxiv.org/abs/2602.15989), SAM License, gated) fitting MHR ([2511.15586](https://arxiv.org/abs/2511.15586), Apache 2.0), which separates skeleton from surface: bone lengths over height | new |
| build | MHR's 20 body identity coefficients ([repo](https://github.com/facebookresearch/MHR)); neutral-pose girths over height | new |
| overall body | the anchored score (a colour-standardised window low-passed to 72 rows: blind to skin tone and to detail finer than its 72 rows, `bodyanchor.py:4-13, 48-49`); SigLIP 2 alone if a DINOv2 target trains (`body_align`, `alignment.py:21-24, 45`) | exists, unvalidated |
| skin tone | CIELAB L*, hue angle h* and chroma per part on eroded SAM 3 skin masks, relative to the face's skin (hue next to lightness: [2309.05148](https://arxiv.org/abs/2309.05148)) | new |
| skin detail | DISTS texture term ([2004.07728](https://arxiv.org/abs/2004.07728)) on native-density skin patches, base sampler | new |
| marks | a mark atlas on body-surface coordinates (DensePose [1802.00434](https://arxiv.org/abs/1802.00434) or the MHR mesh): recall and precision, after a feasibility test, as total-body trackers match lesions ([2307.09642](https://arxiv.org/abs/2307.09642)) [C] | new |
| face | FaceNet gold, raw cosine (`idloss.py:138-148`) | exists |
| copying | SSCD ([2202.10261](https://arxiv.org/abs/2202.10261), `copycheck.py`); the body copy-paste index (5.6) | SSCD exists |
| pose-following | keypoint and limb-angle error against a requested pose from a held-out photo | new |
| leakage | the body judge on renders without the trigger | new |
| final | blind human A/B | new |

Shape judges are anchored between strangers (0) and references (100) as `bodyanchor.py:89-96` does, averaged over
seeds, and limited to measurements that repeat across the subject's own photos (single-image shape is
centimetre-noisy: HBW waist errors of 5.6-8.8 cm across SHAPY, ShapeBoost
([2403.01345](https://arxiv.org/abs/2403.01345)) and PromptHMR ([2504.06397](https://arxiv.org/abs/2504.06397)), test
and validation splits [P]). **Validation before a judge decides or reports** (thresholds are proposals): (1) dose
response: one synthetic MHR body coefficient moved by ±0.5, ±1, ±2 between-subject SDs at fixed camera, pose and
outfit gives a monotone response (Spearman at least 0.9), 1 SD outweighing a mirror, colour grade, JPEG, crop, light
or outfit edit; (2) reliability: ICC over the subject's own photos with Bonett's interval
([doi:10.1002/sim.1108](https://doi.org/10.1002/sim.1108)); (3) separation: leave-one-out AUC of at least 0.8 against
same-sex others, error per Hanley and McNeil
([doi:10.1148/radiology.143.1.7063747](https://doi.org/10.1148/radiology.143.1.7063747)); (4) look insensitivity:
re-graded or softened strangers move it by less than a fraction of a stranger SD that is pre-registered before
validation starts, and |r| with pose class stays below 0.2; (5) for skin, the VAE ceiling and painted-mark recall set
the judgeable mark sizes; (6) agreement with a blind human panel. Shape judges also validate on HBW's validation split
(10 scanned, consenting subjects, [2206.07036](https://arxiv.org/abs/2206.07036)) and BodyM
([2210.05667](https://arxiv.org/abs/2210.05667)), marks on 3DBodyTex ([site](https://cvi2.uni.lu/3dbodytexv1/),
Saint et al., 3DV 2018: 400 scans of 200 subjects, under a licence agreement the recipient's institution signs); none
may be a training subject.

**May feed training:** the diffusion loss, the parser's masks, sigma, HullLoRA's decomposition of the diffusion
gradient, and probe decisions only through P's levels, the retreat and the checkpoint choice. **Never:** a judge's
score or gradient; a judge's encoder family as an alignment target; reward or self-training stages scored by a judge
(`trainer.py:3224-3259, 3394-3400`), off in races; any decision on the reporting set.

## 7. Where it plugs in

| place | change |
|---|---|
| `config.py:26-27` | `hull` in `OPTIMIZERS`, not `AUTOMAGIC`: schedules, per-image lr and `clean_share` then act on rho (contract F3) |
| `config.py:434-482`, `validate()` (`config.py:795-986`) | `hull_*` fields; require `hull_rho`; force `max_grad_norm: 0`; refuse non-LoRA kinds, DoRA, fine-tunes, `likeness_lr_control` (`config.py:664`), `likeness_curriculum`, `region_weights` with C1, `flip_augment` with parts, `loss_watch_healthy_mult` above 1, C1 / C3 / band-budget rows in the parallel race, D and R sets that overlap, `id_loss_weight > 0` with `id_loss_backend: facenet` (it trains on the face judge and turns the gold off, `trainer.py:4135-4139`), `reward_steps > 0` and `selftrain` (scored by the judges, `trainer.py:3224-3259, 3394-3400`); with `hull_probe_control` or in a race, require `face_gold: facenet` and `copy_check: on` (their `auto` defaults give no FaceNet gold unless the run trains on ArcFace, `config.py:770-778`, and no SSCD without self-training, `config.py:786-789`); warn on `min_snr` |
| `config.py:577` | `crop_augment` gains a `skin` mode (torso and limb crops, 5.5), combinable as `face+bust+skin`; M2 |
| `config.py:1039-1049` | `wants_regions()` true for Hull with C1, and `wants_person_masks()` true with `hull_regions: basic` |
| `trainer.py:302-317` | `pairs_sigma` as a capability (Hull with the budget); the parallel race draws through it (`race_parallel.py:222-223`) |
| `trainer.py:357-411` | `build_optimizer(cfg, params, pairs, ref_norms)` builds `Hull(groups, pairs, ref_norms, **kw)`, group lr = `hull_rho`; LoRA+ collapsed as at `trainer.py:374-377` |
| `trainer.py:2321-2332` (`_make_optimizer`) | computes `ref_norms` from the shared wrappers by module name (`self.race.wrappers[name].weight_col_norm()` in a race), as a list aligned with `lora_pairs(injected)`, and passes it on (`build_optimizer` sees neither `injected` nor names) |
| `trainer.py:1012, 2063` | with `hull_frame: explicit`, `optimizer.bind_ema(ema)` after `_make_ema`; Hull reads `ema.shadow` at each step (a relay replaces the list, `race.py:783`) |
| `trainer.py:421-438` | none; optional `wsd` (warmup, constant, cooldown: matches cosine, [2405.18392](https://arxiv.org/abs/2405.18392) [C]) |
| `trainer.py:1065-1067` | compile off keyed on "the optimizer arms the LoRA backward", until M0 |
| `trainer.py:1161-1163, 1248-1249` | `observe(sigma, hook, kind)`; `kind="dop"`; a lock (copies run `work()` in threads, `multigpu.py:95-106`) |
| `trainer.py:1174-1180, 1192-1193, 1242-1245`; `lora.py:392-401, 736-749` | `work()` begins, stops and ends the taps for Hull with C1 too (they key on Keel's `face_energy` today, `trainer.py:1174`); `lora.arm_hull(rinj, gains, face, layout, metric, sink)` on the primary copy, `sink` the optimizer's excess buffers; HullLoRA in `_added` while armed |
| `trainer.py:1174`; `multigpu.py:63-86` (M2) | arm the replicas; after `reduce_grads`, add each replica's excess into the primary's by module name |
| `trainer.py:1223-1225`; `race_parallel.py:60-72` | phase-2 environment weight beside `loss_scale`, and in the parallel race's `forward_backward` |
| `trainer.py:1364-1369` | per-step multiplier clamped to at most 1 for Hull |
| `trainer.py:2892-2895` | a bare `Taps()` for Hull with C1 |
| `trainer.py:55-64, 2528-2545, 3657-3674`; `probeworker.py:219` | a RenderJob flag for the base sampler, honoured in `_render_round` and the probe worker: that job switches only the Turbo LoRA's extra off and samples 25 steps on the base schedule (`qwen21/sampler.py:46-55` picks Turbo's while any extra is on); the extras stay enabled, so the training adapter stays off (`lora.py:526-530`) |
| `trainer.py:3676-3702` (`_probe_jobs`) | D and R sets, novel-pose prompt, base-sampler skin probes, trigger-free renders |
| `trainer.py:755` (`prepare_cache`), `3704-3725` (`_score_probe`), `3740`, `3772-3775` (`_run_probe`) | cache D's photos; score the pose, skin and leakage judges per prompt and seed; evaluate the held-out diffusion loss (fixed sigmas and noise, no gains) under the scored weights, into the record; the scored-copy gate also checks `on_probe_record` |
| `trainer.py:3755-3770` (`_keel_probe`) | `on_probe_record(record, ema, weights)` with raw per-judge, per-prompt, per-seed scores (`likeness.py:340-375`), SSCD, pose, held-out loss; the replica re-sync after a pull stays |
| `race.py:55-63, 705-727, 757-794` | `hull_*` in `RACE_KEYS`; `move_optimizer` / `offload_optimizer` skip `opt.host_keys`; `follow_leader` unchanged |
| `race.py:94` | `REGION_KEYS` gains the parts fingerprint settings (`hull_parser`, `hull_ring_cells`, the parser threshold, `hull_detail_gate` if stored in the map), so rows that differ there get their own maps |
| `race.py:817-819`; `trainer.py:750-751, 2841-2845` | the skin-crop category in `variant_items`, in the race's region-map jobs and in the crop weight |
| `race.py:572-576`; `race_parallel.py:253-270` | frozen-weight norms by module name; H0 unarmed in the parallel race |
| `regions.py:212-264, 292` | the `parts` tensor with ring, detail gate and fingerprint (`RegionMap`, save and load carry it); rebuild on mismatch |
| `dataset.py:204-262, 506-539` | the skin crop mode (M2); parts loaded with the map |
| `webui.py`, `README.md` | labels; the Wizard keeps `adamw8bit` (`webui.py:4414-4415`) until a race earns Hull |
| `tests/test_hull.py` | section 11 |

Unchanged: the out-of-memory retry (`trainer.py:1345-1357`, whose `zero_grad` clears Hull's buffers), the reduce and
clip (`trainer.py:1358-1363`: no `take_replica`, no `gather`; M2 only adds the excess join after the reduce),
`ema.update` (`trainer.py:1378-1379`), resume (`trainer.py:1765-1775` already refuses other optimizers' states outside
`AUTOMAGIC`); fine-tunes refuse Hull, so `finetune.py:54-55` needs no entry.

## 8. Options

```yaml
optimizer: hull                 # LoRA kind only; not in AUTOMAGIC: the trainer's schedule acts on rho
hull_rho: 0.0                   # rho_0 (relative dW per step and layer); required, from M0's calibration
hull_metric: balanced           # balanced (H0) | riemannian (R1b) | adam (per-element v, no frame: fallback base)
hull_frame: virtual             # virtual | explicit (rebalance in place; m and the lora_ema shadow transported)
hull_step_scope: layer          # layer | global
hull_betas: [0.9, 0.99]
hull_state_dtype: fp32          # m: fp32; bf16 is ignored with a warning (section 4.1)
hull_clip: 5.0                  # per-channel RMS clip, balanced frame
hull_factor_cap: 0.2            # kappa_f
hull_safety_cap: 0.0            # kappa (rho_t mu ||W0|| units); 0 = 2 max(1, hull_gain_max)
hull_ns_iters: 30
hull_damping: 1.0e-6            # x trace / r
hull_regions: off               # off | parts (C1 + C2) | basic (today's maps: background attenuation only)
hull_gain_body: 1.5             # body skin + ring, flat across bands
hull_gain_background: 1.0       # at most 1
hull_gain_max: 2.0              # gains in [0.5, max]; clothing, hair <= 1; face, text = 1
hull_ring_cells: 1
hull_parser: mediapipe          # mediapipe | sam3 (the judges' masks then use MediaPipe)
hull_face_safe: true            # C3
hull_detail_gate: true
hull_retreat: 0.5               # factor on lam; lam never rises
hull_band_budget: off           # off | natural | table (M2, after M0)
hull_band_edges: "0.12,0.35,0.65,0.88"
hull_top_band_cap: 0.25
hull_probe_control: false       # P (M3)
hull_rho_levels: "1,0.5,0.25"
hull_patience: 3
hull_accept_z: 1.0
hull_face_margin: 0.0           # delta_face (raw FaceNet cosine) from the A/A pilot; required with P
hull_lookahead: false
hull_pull: 0.5
hull_grace: 2
hull_pull_mode: procrustes      # procrustes | product
hull_purity_floor: 1.0          # G; 1 = off
hull_dop: off                   # off | pcgrad | protect_prior
hull_env: none                  # none | pose | outfit
hull_phases: ""                 # "" | S,I,D
```

M1a's default is H0 alone; M1b's rows turn on `hull_regions: parts` with C3; the rest stays off until its race. Skin
crops (R5) are `crop_augment: skin` (or `face+bust+skin`), weighted by `crop_augment_weight`. The Wizard keeps
recommending `adamw8bit` until a race says otherwise.

## 9. State, resume, race

| key, per parameter | on | dtype, shape | bytes per weight, r = 32 |
|---|---|---|---|
| `m` | both factors | fp32, like p | 4 |
| `v` | lora_up [o], lora_down [i] | fp32 | 0.125 |
| `T`, `Ti`, `u`, `k`, cap and fallback counters | lora_up | fp32, [r, r] or 0-dim | ~0 |
| `dither`, the stochastic rounding's step count | the first pair's lora_up | a host integer | ~0 |
| `lam`, the gained-raw cosine EMA | both factors, per tensor | fp32, 0-dim | ~0 |
| `mu`, miss and accept counts, the anchor's per-seed scores | every parameter (mirrors) | fp32, 0-dim or tiny 1-D | ~0 |
| `anchor` (lookahead) | every parameter, CPU, in `host_keys` | fp32 | 4 on the host |

- **Totals.** 2.125 B/weight, 178 MB (170 MiB) at 83,886,080 weights, against AdamW's 647 MiB, bnb AdamW8bit's 163 MiB
  and Automagic 4's 91 MiB [bench]; transient, the excess while C1 is armed (2 B/weight); G 4 B/weight; D 4 on DOP
  steps.
- **Flat and re-read.** Races park, move, clone and relay `opt.state[p]` top-level tensor by tensor
  (`race.py:705-727, 757-794`; contract F16): no nested containers, no tensors in instance attributes across steps;
  the frame is recomputed from the factors each step (the stored T only seeds the fallback); mirrors are
  median-unified after a relay.
- **Load.** Constructor options win for Hull's own options, never for `lr` and `initial_lr` (the scheduler's, which
  torch restores; `optimizers/automagic3.py:678-680` resets every default, `lr` included); fp32 values come back
  exactly (as `optimizers/automagic3.py:668-672`); torch casts state to the parameter's dtype and moves it to the
  parameter's device, so `m` is cast back to bf16 and every `host_keys` entry is moved back to pinned CPU memory. Hull
  states resume only under Hull (`trainer.py:1765-1775`).
- **Race.** `follow_leader` copies a Hull state into a Hull follower and starts other kinds over (`race.py:773-779`);
  CPU anchors stay on the host; a dropped Hull is freed (H0 has no hooks; D's hook holds the optimizer weakly).
- **Weight average.** `lora_ema` (0.98 by default, `config.py:418`) lerps factors (`ema.py:58`), which stay in one
  gauge; final picks use `posthoc_ema.py`'s product averages.

## 10. Validation

**Subjects.** Synthetic characters from MHR or Anny ([repo](https://github.com/naver/anny), Apache 2.0, CC0 assets;
not its non-commercial SMPL-X topology) across height, build, sex and age, with 6-10 marks painted at known UV
positions (2-20 mm moles, freckles, a scar, a tattoo, a tan line; MHR's README mentions no UVs, so marked subjects use
Anny, or MHR with UVs added and checked first), poses from the body model's own pose space, 30 training renders at
1 MP (a third each face, upper and full body; at least 3 outfits that show the build and visible skin, 6 poses, 4
lights) and 6 held-out renders in unseen outfits and scenes for D and R. A realism pass is allowed only if refitting
SAM 3D Body moves the MHR coefficients by less than 0.25 between-subject SD and the marks by less than 3 px
(proposals). Synthetic subjects are scored against **peer strangers** (other synthetic subjects rendered alike)
besides the base model's renders, so a learned CG look cannot read as likeness, and their **primary endpoints are
ground-truth errors**: girth and length ratios against the true mesh, mark recall and precision against the painted
marks. Consenting adults under a release written for this use (likeness training and generation for research
evaluation, revocable, paid, tape-measured), on the official Qwen-Image 2.1 weights, are the external check.

**Licences.** Qwen-Image 2.1 is under the Qwen Research License (`README.md:2196`): races are research or evaluation.
The Fizgig training adapter (`README.md:2165-2171`) has no licence, so the races must reach their conclusions with it
off; an adapter-on row is the user's call and decides nothing. The FaceNet VGGFace2 weights state no licence. SAM 3D
Body's weights are gated (SAM License); MHR, the MediaPipe segmenter and DISTS are Apache 2.0 or MIT. SMPL-X
([licence](https://smpl-x.is.tue.mpg.de/modellicense.html)) and BEDLAM
([licence](https://bedlam.is.tue.mpg.de/license.html)) are non-commercial and restrict content uses, among them fake,
misleading or defamatory content outside peer-reviewed analyses; Sapiens2's licence bans biometric processing
([LICENSE.md](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md), section 1.b.vi). So they stay out of
the stack. HBW is downloaded under the SHAPY licence agreement ([datasets](https://shapy.is.tue.mpg.de/datasets.html),
[licence](https://shapy.is.tue.mpg.de/license.html)), which bans fake, misleading or defamatory content of any kind
outside peer-reviewed analyses; 3DBodyTex comes only under a licence agreement signed by the recipient and their
institution's research administration office ([site](https://cvi2.uni.lu/3dbodytexv1/)).

**A/A pilot first.** Before it, the adapter-off control must not collapse (`README.md:2166-2167`): in R0a's runs, the
first to train, the held-out judges stay above the strangers at every late checkpoint and the renders show no texture
collapse; if they collapse, the races wait for the user's decision on an adapter (13, item 12) instead of running
adapter-off. R0b then tunes the control's lr. The control (AdamW, betas 0.9 / 0.99 as `trainer.py:361-362`, no
decay, R0's lr and sampler, `face_gold: facenet`, `copy_check: on`) trained twice on 6-8 subjects, or three times on
4, probed at the starting counts of 4 seeds x the decision set's prompts (full body, upper body, face, one novel
pose; 5.6) x 6 late checkpoints, plus the body prompts rendered without the trigger, with 8 strangers per prompt,
counts then re-derived from the pilot's variance components. Outputs per judge: s_run (paired run differences /
sqrt 2), the seed and checkpoint terms, sigma_AA,j, delta_face in raw FaceNet cosine, and the A/A spreads of SSCD, the
copy index, pose, leakage and the held-out decision loss.

**How many subjects** (method; no number before the pilot). With r runs per arm, m seeds, c checkpoints and q prompts,
Var(D_s) = tau_s² + 2 s_run²/r + 2 (s_seed²/m + s_ck²/(m c)) / q for a subject's paired difference. Plan with the
upper 80% confidence bound of the pilot's paired SD (chi-square on its degrees of freedom) and a pre-registered tau_s:
N is the smallest n with power 0.8 for Delta (the smallest body effect worth having, pre-registered on blind image
pairs, in stranger-SD units) at two-sided alpha 0.05, Holm-adjusted, by the noncentral t. The face test is sized on
delta_face (one-sided alpha 0.025); N is the larger and at least 6 (five unanimous subjects give a two-sided sign-test
p of 2 x 2⁻⁵ = 0.0625) [computed]. tau_s cannot come from an A/A pilot: one pre-registered interim look after half the
confirm subjects re-estimates Var(D_s), arms blinded, and may only raise N.

**Races.** One sampler and step budget, 1024 px, rank 32, EMA 0.98, adapter off, one card; rows screened on 4
subjects, survivors confirmed on N; results reported at matched steps and at matched displacement ||dW|| / ||W0||. A
screened row survives when its paired difference on the primary judge (the ground-truth error on synthetic subjects)
favours it on at least 3 of 4 subjects and no guard (face, SSCD, copy index, pose, leakage) leaves its A/A spread;
confirmation on N then applies the bar. R0a and R0b run before the pilot and pick on the primary judge alone.

| race | rows | reads |
|---|---|---|
| R0a | AdamW at 1e-4: `timestep_sampling: reference` (logit-normal plus the resolution shift, `archs.py:586-593`; SD3 found logit-normal(0, 1) best-ranked and trains and samples at one shift, [2403.03206](https://arxiv.org/abs/2403.03206) Eq. 23 [P]; RulzTrainer's MiniMax H3 recipe uses it, `README.md:2461-2462`) against `logit_normal` | the sampler of later rows |
| R0b | AdamW at 0.5x, 1x, 2x of 1e-4, widened if the best sits at an end | tuned control; rho_0 |
| R1 | H0 at 0.5x, 1x, 2x rho_0 | H0 against tuned AdamW |
| R1b | per-layer vs global step; balanced vs Riemannian metric; rank-component growth logged | scope; face bias of the GD-like ordering [H] |
| R2 | H0 and AdamW at ranks 16, 32, 64, fixed rho / lr | rank transfer |
| R3 | C1 (parts, body gain 1.5, flat) with C3, on H0 and on `hull_metric: adam` | the body lever on both bases |
| R4 | R3's winner without C3 | whether C3 holds the face |
| R5 | skin crops on / off; detail gate on / off | skin (base sampler) |
| R6 | band budget: natural vs the M0 table | structure, pose, copy |
| R7 | P off / on; lookahead off / on | all; copy |
| R8 | phase-2 options, one at a time | each option's judge |
| R9 | the winner on Turbo and base samplers, at strength 0.8, with a style LoRA | the LoRA where it is used |
| R10 | the winner against the control on 3-6 consenting adults | external validity (reported) |

**The bar.** A mechanism stays only if, against its parent row: its primary validated judge improves (two-sided 95%
paired interval over subjects excludes 0, Holm, stranger-SD units; on synthetic subjects the ground-truth error
falls); the face is non-inferior (the lower one-sided 97.5% bound of the raw FaceNet difference lies above
-delta_face); SSCD, the copy index, pose-following and leakage stay within their A/A spreads; and it costs at most
+10% of whole-step time on a 16 GB card (or wins at matched wall-clock) and fits in 16 GB. Otherwise it is switched
off, and this document says so.

**Costs.** M0 measures whole-step time and peak memory with the real model on the RTX 5060 Ti and RTX 4060 Ti at 1 MP,
rank 32, compile on and off. The only 16 GB figure known is the README's: RTX 5060 Ti, int8, 0.5 MP, rank 8, about 2.5
s per step with compile at a 13.1 GB peak (`README.md:2194-2195`); 1 MP at rank 32 is slower [H], so a 2,000-step run
(a starting value) takes at least 2,000 x 2.5 s = 1.4 GPU-hours [computed lower bound], plus probes.

| stage | runs | GPU-hours, at least |
|---|---|---|
| M0 screens: R0a 8, R0b 12 | 20 | 28 |
| A/A pilot | 12-16 (6-8 subjects x 2, or 4 x 3) | 17-22 |
| M1a screen: R1 12, R1b 8, R2 16 | 36 | 50 |
| M1b screen: R3 12, R4 4 | 16 | 22 |
| M2 screen: R5 12, R6 8 | 20 | 28 |
| M3 and M4 screens: R7 12, R8 16 | 28 | 39 |
| each confirmation, 2 arms x N subjects | 2N | 2.8N |
| R10 | 6-12 | 8-17 |

Shared arms (R1's 1x arm is R2's rank-32 H0) cut the counts; two cards halve the wall-clock; M0's real times replace
the bound.

## 11. Build order and tests

**M0, measurements** (harnesses, no Hull class): (a) judges built and validated, the anchored score included; (b) R0a,
then R0b (AdamW only, screened on 4 subjects), then the A/A pilot at R0's lr and sampler; (c) when traits are decided
per band: noise held-out photos to sigma in {0.95 ... 0.2} and denoise with the base model (SDEdit,
[2108.01073](https://arxiv.org/abs/2108.01073)), the sigma where a validated judge falls to the strangers dating its
trait, plus a band-switch test on an AdamW LoRA (on only above or below a cut in {0.95, 0.9, 0.8, 0.7, 0.5, 0.35} via
`lora.set_multiplier` per sampling step), at 1024 and 2048 px; (d) body-cell agreement across photos: HullLoRA at
gains 1, report-only, cross-step purity of body, face and background parts per tensor (noise floor 0.23 at beta 0.9
[N]); region rows need body above background, G needs body at or above face; (e) natural per-band update shares; (f)
VAE detail retention on skin at 1024 and 2048 px, painted-mark recall; (g) parser cell accuracy against the exact
masks of the synthetic renders and against hand-labelled full-body photos of the consenting adults (section 10),
reported by Monk tone; (h) the HullLoRA cost gate and a compile test; (i) rho_0 from the realised dW step of R0b's
tuned control; (j) channel-specialisation spread for fallback R; (k) token-path against loss-path attribution; (l)
step time and peak memory of H0 and the control on both cards.

**M1a, the smallest version that could beat AdamW: H0.** Tests: the virtual frame equals explicit rebalancing with
transported momentum over 200-step trajectories (fp32 tolerance; in fp64 1e-12 at `hull_damping: 1e-12`, 1e-6 at the
default 1e-6); starts (0, A0) and (0, S0⁻¹A0), cond(S0) 5-100, and a step from (BS, S⁻¹A) with B non-zero give one dW
to 1e-5 at `hull_damping: 1e-12` and to 2e-3 at the default; T M T = N to 1e-4 after 30 iterations through a run, a
rank-4 update at rank 32 included, and a forced failure keeps the previous T; step 1 moves only B, along the
row-normalised G P_A, with ||dW_1|| = rho_t mu ||W0|| to 1e-6; a coherent gradient on factors with a dominant
direction gives the same realised multiple of rho within ±10% at ranks 4-64 with alpha = rank or 16, random signs
about 0.25 rho (0.2 of the coherent multiple); v and the clip unchanged under B -> BQ; the realised step never exceeds
kappa rho_t mu ||W0||, second order included; a NaN or inf gradient leaves weights and state bitwise; no host sync in
`step()` (`torch.cuda.set_sync_debug_mode("error")`, which Keel lacked); an exact state round trip (m back in bf16,
Hull's own options winning, `lr` left to the scheduler) and bitwise next steps after move, offload and relay (as
`tests/test_automagic.py:253-286`); `group["lr"]` x 0.5 halves the step and multipliers above 1 are clamped; random
training losses never change a step size; with `hull_frame: explicit` balancing leaves the `lora_ema` shadow's product
unchanged (1e-6); zero-element tensors skipped, LoKr / OFT / DoRA refused; wiring; at most +80 ms over the null step.
Races R1-R2; in R1 the safety cap binds on at most 2% of layer-steps after step 20 (4.4), and a higher rate, which
means u no longer sets the step, blocks M1b on either base until the normaliser is fixed. **Kill rule with
fallback:** H0 stays only if it is non-inferior to tuned AdamW on every validated judge and shows R2's rank transfer
(one rho level serves ranks 16-64); otherwise the body mechanisms continue on `hull_metric: adam`, AdamW-style moments
with the same dW-unit step.

**M1b, the body lever: C1 with parts, C3 and the detail gate, on H0 and the `adam` base.** The held-out decision-loss
probe job (fixed sigmas, noise and D photos) is built here. Tests: HullLoRA at gains 1 returns autograd's dx, dA and
dB bitwise and a zero excess (Hull then equals H0 bitwise); at other gains the widened GEMM's raw block matches
autograd to bf16 rounding; at arbitrary gains dx and the weight gradients stay raw and the excess equals sum (w - 1)
d hᵀ and its lora_down twin; one backward per layer, identical with and without non-reentrant checkpointing; armed H0
at gains 1 equals unarmed H0 at K = 1 and 2; a body-only tensor moves g times as far in normalised units, a face-only
tensor ignores the gains, u ignores every gain; after C3 <F, X>_U >= 0, a no-op at gains 1; face and text gains
exactly 1, clothing and hair at most 1; parts fractions sum to 1 per cell, the ring excludes the face box, a
fingerprint mismatch rebuilds, flips are refused, a blurred photo gates 0; `zero_grad` clears the excess; lam only
falls, and a negative cosine EMA over a probe interval, or a held-out loss rise beyond its A/A noise while the
training loss falls, halves it; a cost harness on the 224 layer shapes. Races R3-R4.

**M2: skin crops, band budget, replica arming.** Tests: crops never upscaled, at least 30% body skin, their own parts
and gate; on a synthetic two-band stream with 5x gradient scales and 3:1 visit rates the realised shares land within
±5% of the targets, the top band never exceeds 25%, face and text gains stay 1, and the budget off equals M1b bitwise;
every micro-batch of a step, the DOP pass included, in one band; armed replicas whose excess joins the primary's equal
a one-device step. Races R5-R6.

**M3: P.** Tests: accept if and only if Q exceeds the threshold with every guard and no veto; mu never exceeds 1 or
rises over random decision sequences, and random losses never move it; an F2 breach forces a level down and a retreat;
p = 1 restores the anchor exactly, p = 0.5 stays within the product-error bound, the EMA shadow alike; anchors stay on
the CPU through park, move, relay and resume; an async probe anchors the copy it scored; mirrors survive relays and
resume. Race R7.

**M4: phase-2 options.** G: purity at most 1, 0.23 ± 0.01 on noise at beta 0.9, abstention, off equals M3 bitwise; D:
the hook armed only inside a step, two optimizers on shared tensors independent, <G_s', G_d>_U >= 0 after projection;
E: an 80/20 stream comes out equally weighted; phases: mu never rises. Races R8-R9.

**M5: README and the Wizard,** only if a race earned it; R10 reported.

## 12. Rejected

- **Keel's gauge preconditioner before an Adam-type normaliser**: its scale is cancelled, at +307 ms and 448 syncs.
- **The cautious mask**: a no-op when every scaling is positive; under Hull's frame it would act on gauge-dependent
  signs, and at batch 1 a sign clash between momentum and one photo is mostly noise [H]
  ([2411.16085](https://arxiv.org/abs/2411.16085)).
- **Within-step votes at K = 2-4**: 1.36-2.08x (sign vote, 80%-consistent weight) and 1.07-1.24x (magnitude vote, SNR
  0.5) contrast against noise. **Loss-driven self-tuned learning rates**: they climb on small datasets (2.1). **Judge
  gradients in training** (identity losses, reward backpropagation, alignment to a judge's encoder): gamed, and they
  would void the held-out guard.
- **Per-factor Muon as the base**: 83.0 against Adam's 87.1 on Llama-3-8B
  ([2507.12142](https://arxiv.org/abs/2507.12142), Table 1) [3P]; full-parameter Muon showed a strong
  rote-memorisation tendency in a synthetic GPT-2 experiment, which its authors tie to Muon's full-rank updates and
  say may hurt learning from small fine-tuning sets ([2605.06654](https://arxiv.org/abs/2605.06654)) [C]; whether
  per-factor LoRA Muon shares it is untested [H].
- **IRM** needs more training environments than environmental feature dimensions in the linear case
  ([2010.05761](https://arxiv.org/abs/2010.05761)) [P]; **V-REx** ([2003.00688](https://arxiv.org/abs/2003.00688))
  upweights unusual photos; **IGA** ([2008.01883](https://arxiv.org/abs/2008.01883)) and all-parameter **Fishr**
  ([2109.02934](https://arxiv.org/abs/2109.02934)) are second-order through the W8A8 base, whose backward computes
  only the input gradient (`quant.py:11-15`); **full SAM** doubles the compute (two backpropagations per update,
  [2010.01412](https://arxiv.org/abs/2010.01412)).
- **Region loss weights as the main body lever**: v cancels them wherever one region dominates an element, and with C1
  they would count twice.
- Also set aside: per-region second moments (they amplify noise where a region barely acts), equalised bands by
  default, LoRA-RITE as a base (no per-channel adaptivity), LoRA-Pro ([2407.18242](https://arxiv.org/abs/2407.18242))
  / SOAP ([2409.11321](https://arxiv.org/abs/2409.11321)) / Shampoo ([1802.09568](https://arxiv.org/abs/1802.09568))
  (state), weight decay.

## 13. Open questions

1. **Does the virtual frame hold numerically at rank 32 over 2-3k steps?** Its Grams are worse conditioned; if the
   fallback fires often, `hull_frame: explicit` becomes the default (in place, `m` and the shadow transported,
   tested).
2. **Does H0 beat, tie or lose to tuned AdamW, and does rho transfer across ranks?** A tie is expected; R1b also asks
   whether the GD-like ordering across rank components favours the face.
3. **Where does Qwen-Image 2.1 decide each trait?** Until M0 answers, the band table stays empty and the gains flat.
4. **What does HullLoRA cost, and can it compile?** FLOP counts did not predict the existing tap.
5. **Does C1's non-conservative field drift** once the true gradient is small? The guards only catch it afterwards.
6. **Do body-cell gradients agree across photos** more than background ones, and as much as face ones?
7. **Does 1024 px carry marks through the VAE, and does a realism pass keep shape and marks?** Unmeasured anywhere.
8. **Does DOP pull the body toward the base model's body?** Phase-2 D measures it.
9. **Probe cost on 16 GB cards**, and whether `probe_async` (no data parallel) is forced.
10. **The factor-average gap** (about 2.2% in the check, 1% for Adam): above 5% in M1, a rank-r product EMA.
11. **Deviations from the synthesis spec**, each following the evidence:
    - (a) The parts tensor and the per-item detail gate move from M2 into M1b: the spec's body (skin plus the ring,
      never clothing or hair; clothing capped at 1) cannot be built from today's maps, whose body is the person minus
      the face; applied per item at every band, the gate keeps the default gain flat while no skin gain reaches a soft
      photo.
    - (b) The clip is per channel in the balanced frame, not per element: under the virtual frame per-element values
      are gauge-dependent.
    - (c) `p.grad` stays raw and the gained excess goes to a side buffer (2 B/weight while armed), so armed and
      unarmed paths compute the same v and u at every K, not only at K = 1.
    - (d) kappa = 2 max(1, gain max): a consistent layer already moves about 1.5 rho at gains 1 [N], so a cap near 1.5
      would bind by design; "a few percent" becomes 2% of layer-steps after step 20.
    - (e) Geometry's factor caps (kappa_f 0.2) stay beside the safety cap to bound the second-order term early.
    - (f) The Newton-Schulz fallback keeps the previous T, not S = I, which under the virtual frame would move a
      pair's statistics into another frame; only T and T⁻¹ are computed.
    - (g) The Ladder supplies the threshold only (one standard error, about the 0.15 level); the control design's
      "scores rounded to sigma/2" is not its rounding (the parameter-free variant rounds to 1/n).
    - (h) "C1 on AdamW" runs as `hull_metric: adam` (per-element v, no frame, the same dW-unit step).
    - (i) R1 does not race the factor-norm proxy against u: the [N] check already shows the proxy realising 2.39-7.29
      rho at ranks 4-32 (4.3), where the safety cap would set the step. The M1a rank test stands in for that race; it
      checks one realised multiple of rho within ±10% across ranks 4-64, not rho itself, because u realises about
      1.25-1.5 rho on a consistent layer (4.3, 4.4).
    - (j) R0a and R0b run in M0, before the A/A pilot, which needs the control's lr and sampler; M1a races R1-R2.
    - (k) Found while building H0: the Grams and roots run in fp64, not fp32 (4.6), and step() costs more than the bar.
    - (l) `Hull` also takes each pair's alpha / rank (`scales`), which the pairs' tensors do not carry.
    - (m) `T⁻¹` is stored beside `T`: the fallback must keep both, and a fp32 inverse of a stored T would lose its
      conditioning.
    - (o) Built M1b differs from 5.1 and 5.3 in four ways. HullBackward runs one extra GEMM per kept block instead of
      one widened GEMM (simpler; the cost gate decides if that stays). The excess and the face part arrive in the compute
      dtype and are summed as they are. C3 projects once in `step()` on the step's sum, with that step's D and T, not per
      micro-batch with the previous step's; so it costs 2 B/weight more while armed. The parser is the published
      256 x 256 multiclass model: MediaPipe's model card describes a 512 px one, which is not among its downloads.
    - (n) The stochastic rounding's generator is private and reseeded from a step count in the state, so the dither
      leaves the run's noise stream alone (arms stay paired) and a relay or resume repeats it exactly.
12. **Licences the user must settle:** the Fizgig adapter (and, if the adapter-off control collapses, whether the
    races use it) and the FaceNet VGGFace2 weights (none stated), SAM 3D Body's gated SAM License, the HBW (SHAPY) and
    3DBodyTex licence agreements (3DBodyTex's signed by the recipient's institution), and the volunteers' release
    text.

## References

arXiv IDs open at https://arxiv.org/abs/<id>.

- Optimizers and LoRA: BaLoRA 2605.31484; Du, Hu, Lee 1806.00900; LAMB 1904.00962; ADOPT 2411.02853; PoLoRA
  2607.17620; LoRA-RITE 2410.20625; Riemannion 2507.12142; Learning Rate Matters 2602.04998; Prodigy 2306.06101;
  Cautious Optimizers 2411.16085; Optimizer-Model Consistency (Liu, Wang, Zhang) 2605.06654; BF16 training 2010.06192;
  constant lr with cooldown 2405.18392; LoRA-Pro 2407.18242; SOAP 2409.11321; Shampoo 1802.09568; SAM 2010.01412; LoRA
  Without Regret https://thinkingmachines.ai/blog/lora/; Higham https://doi.org/10.1137/1.9780898717778
- Consensus and multi-objective: AND-mask 2009.00329; SAND-mask 2106.02266; GradDrop 2010.06808; PCGrad 2001.06782;
  A-GEM 1812.00420; risks of IRM 2010.05761; V-REx 2003.00688; IGA 2008.01883; Fishr 2109.02934; Xin et al.
  2209.11379; Kurin et al. 2201.04122; FairHuman 2507.02714; Beyond Facial Consistency (DBS) 2607.25622
- Noise levels: SD3 2403.03206; P2 2204.00227; T2I-Adapter 2302.08453; Readout Guidance 2312.02150; T-LoRA 2507.05964;
  Go et al. 2306.00354; timestep curriculum 2403.10348; SDEdit 2108.01073; Qwen-Image-VAE-2.0 2605.13565; Turbo sigmas
  https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/raw/main/model_index.json
- Probes and selection: PBT 1711.09846; Lookahead 1907.08610; the Ladder 1502.04585; holdout reuse 1506.02629; IPRO
  2510.14255; WithAnyone 2510.14975; SSCD 2202.10261
- Bodies, masks, skin: SAM 3D Body 2602.15989; MHR 2511.15586 and https://github.com/facebookresearch/MHR; Anny
  https://github.com/naver/anny; MediaPipe multiclass card
  https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf; DISTS 2004.07728;
  Beyond Skin Tone 2309.05148; DensePose 1802.00434; lesion matching 2307.09642; HBW (SHAPY) 2206.07036; ShapeBoost
  2403.01345; PromptHMR 2504.06397; BodyM 2210.05667; 3DBodyTex https://cvi2.uni.lu/3dbodytexv1/; SAM 3 2511.16719;
  Bonett https://doi.org/10.1002/sim.1108; Hanley and McNeil https://doi.org/10.1148/radiology.143.1.7063747
- Practice: https://github.com/ostris/ai-toolkit/issues/1091, https://github.com/ostris/ai-toolkit/issues/1084,
  https://github.com/kohya-ss/musubi-tuner/discussions/455. The research reports behind this design are working notes
  outside the repository; their benchmarks and numerical checks become `tests/test_hull.py` and scripts in M0-M1.
