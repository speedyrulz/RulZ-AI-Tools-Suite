"""Keel: Automagic 4 for character LoRAs (docs/keel-optimizer.md). Milestones 1 and 2 of that design: the LoRA
gauge preconditioner, the cautious mask and the noise-band normalisation. With every switch off it is Automagic 4
exactly (test_keel asserts it bitwise).

Gauge preconditioner (keel_gauge)
---------------------------------
A LoRA's delta is B A (lora_up [out, r] times lora_down [r, in]), and (B S)(S^-1 A) is the same delta for any
invertible r x r S. Adam-style steps are not invariant to that reparameterisation, so one factor tends to take
most of the update while the other barely moves (LoRA-RITE, ICLR 2025). Riemannian preconditioned LoRA (Zhang &
Pilanci, ICML 2024; the scaled gradient of Tong et al. 2021) restores the invariance with two r x r solves per
layer, before the gradient enters the optimizer's moments:

    g_up   <- g_up   (A A^T + delta I)^-1
    g_down <- (B^T B + delta I)^-1 g_down

The paper reports better quality and stability on a 6-image SD 1.5 fine-tune with AdamW after it, and that equal
rates for both factors then suffice (no LoRA+, which Automagic ignores anyway). delta = keel_gauge_eps times the
larger of the two Grams' mean diagonal: lora_up starts at zero, so its Gram is 0 for a while and the solve for
lora_down would otherwise divide by nothing; with the floor the small factor's preconditioner is about the
identity until that factor has grown, and the proper scaling takes over from there. The solves run on the
accumulated gradient in step(), after the trainer's gradient clipping, so the second moment and the sensitivity
reference see the preconditioned gradient (as the paper's scaled AdamW does).

Cautious mask (keel_cautious)
-----------------------------
Cautious optimizers (Liang et al., ICLR 2026) zero the update where its sign disagrees with the gradient's, so
the step keeps a non-negative inner product with the gradient. Every scaling in Automagic 4's step is positive
(the second-moment normalisation, the trust-region clip, the sensitivity balance), so its update's sign is the
gradient's sign and the mask would never fire on its own; what can flip a sign is the gauge preconditioner,
which mixes the r directions of a factor. The mask is therefore applied to the preconditioned gradient against
the raw one: an element whose preconditioned gradient points the other way is zeroed, which is the mask on the
update too (those elements also leave the second moment, where they would have counted at the raw sign). No
rescale by the surviving share: the normalised step does not depend on the gradient's scale. The share that
survives is logged every 250 steps (cautious_share()); well under 1 means the preconditioner is fighting the
gradient (raise keel_gauge_eps).

Noise-band normalisation (keel_bands, keel_band_weights)
--------------------------------------------------------
Gradient power varies a lot with the noise level sigma (Min-SNR; Addressing Negative Transfer in Diffusion
Models; EDM2's learned per-level weights), and one second moment per weight is set by the loudest band, usually
not the one where the face forms (sigma 0.2 to 0.5 at 1 MP): the quiet bands take small steps. Keel keeps, per
weight tensor and per band of logit(sigma) (keel_bands equal bands over [-4, 4], the ends taking what lies
beyond), a running mean of the gradient's power (its mean square), plus the overall one; a step whose
micro-batches all sit in one band has its gradient scaled by sqrt(overall / band) before anything else, so it
enters the second moment at the tensor's usual scale whatever its band and every band's step is sized alike.
A band is trusted after BAND_MIN_OBS observations (until then the scale is 1), the lift is capped at GAIN_MAX,
and a step whose micro-batches span several bands (keel_pair_sigma off, or batch_size > 1 with its own sigmas)
is left alone (logged once). keel_band_weights then multiplies the step's update (not the decay) by the band's
weight (Automagic4.update_gain): "" = 1 everywhere, "identity" = a hump over the face bands, or one number per
band; on a mixed step the mean of its micro-batches' weights. The trainer feeds the noise levels through
observe(sigma) before each micro-batch's backward (zero_grad() starts a new step); without those calls the
bands do nothing. The per-band gradient RMS relative to the overall, averaged over the tensors, is logged
every 250 steps: the measured imbalance the mechanism answers.

Pairs: build_optimizer passes the (lora_down, lora_up) pairs of the injected LoRA layers; LoKr, OFT and a
fine-tune have none, and the step is then Automagic 4's. A race switching a wrapper to LoKr leaves its LoRA
tensors empty, which the step skips.
"""

import logging
import math
from typing import List, Optional, Sequence, Tuple

import torch

from ..config import KEEL_BAND_LOGIT_RANGE, keel_band_edges
from .automagic4 import Automagic4

log = logging.getLogger("rulztrainer")

GAUGE_FLOOR = 1e-12  # absolute floor of delta (both Grams zero: nothing to divide by, and nothing to scale)
LOG_EVERY = 250  # steps between the periodic log lines (cautious share, band statistics)
BAND_MIN_OBS = 8  # a band is normalised on its own statistics after this many observations
BAND_WINDOW = 10  # the band statistics: a running mean until this many observations, an EMA of that window after
GAIN_MAX = 10.0  # the band scale stays within [1/GAIN_MAX, GAIN_MAX] (a 100x quieter band is lifted at most 10x)


def band_of(sigma: float, bands: int) -> int:
    """The band of a noise level: `bands` equal bands over logit(sigma) in [-KEEL_BAND_LOGIT_RANGE, +range], the
    first and last taking what lies beyond."""
    s = min(max(float(sigma), 1e-6), 1.0 - 1e-6)
    logit = math.log(s / (1.0 - s))
    r = KEEL_BAND_LOGIT_RANGE
    return min(max(int(math.floor((logit + r) / (2.0 * r) * bands)), 0), bands - 1)


def gauge_precondition(g_down: torch.Tensor, g_up: torch.Tensor, down: torch.Tensor, up: torch.Tensor,
                       eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """The r x r preconditioner of a LoRA pair's gradients, in fp32: (g_down (B^T B + d I)^-1-scaled, g_up (A A^T +
    d I)^-1-scaled) with d = eps x the larger mean diagonal of the two Grams (plus GAUGE_FLOOR). With eps -> 0 the
    delta weight's first-order change is invariant to B A = (B S)(S^-1 A) (test_keel checks it)."""
    A = down.detach().to(torch.float32)
    B = up.detach().to(torch.float32)
    gd = g_down.to(torch.float32)
    gu = g_up.to(torch.float32)
    r = A.shape[0]
    m_up = A @ A.T  # [r, r]: scales g_up
    m_down = B.T @ B  # [r, r]: scales g_down
    delta = torch.maximum(m_up.diagonal().mean(), m_down.diagonal().mean()) * eps + GAUGE_FLOOR
    eye = torch.eye(r, device=A.device, dtype=torch.float32)
    new_up = torch.linalg.solve(m_up + delta * eye, gu.T).T  # g_up M^-1, M symmetric
    new_down = torch.linalg.solve(m_down + delta * eye, gd)
    return new_down, new_up


class Keel(Automagic4):
    """Automagic 4 plus the gauge preconditioner, the cautious mask and the noise bands (module docstring).

    Parameters (beyond Automagic 4's)
    ---------------------------------
    pairs : [(lora_down, lora_up)]
        The LoRA pairs among the optimizer's parameters (lora.lora_pairs); empty = no preconditioning.
    gauge : bool
        The r x r preconditioner on every pair's gradient.
    gauge_eps : float > 0
        delta as a fraction of the larger Gram's mean diagonal (0.01).
    cautious : bool
        Zero a preconditioned gradient element whose sign disagrees with the raw gradient's.
    bands : int >= 1
        Noise bands over logit(sigma) for the band normalisation; 1 = off. Fed by observe(sigma).
    band_weights : [float] or None
        A multiplier on the step's update per band (config.keel_band_weights resolves the setting); None = 1.
    """

    def __init__(self, params, *args, pairs: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
                 gauge: bool = True, gauge_eps: float = 0.01, cautious: bool = True, bands: int = 8,
                 band_weights: Optional[Sequence[float]] = None, **kwargs):
        if not gauge_eps > 0.0:
            raise ValueError(f"gauge_eps ({gauge_eps}) must be above 0")
        if int(bands) < 1:
            raise ValueError(f"bands ({bands}) must be at least 1")
        if band_weights is not None:
            band_weights = [float(w) for w in band_weights]
            if len(band_weights) != int(bands) or any(not w > 0.0 for w in band_weights):
                raise ValueError(f"band_weights must be {int(bands)} numbers above 0, got {band_weights}")
        super().__init__(params, *args, **kwargs)
        self.gauge = bool(gauge)
        self.gauge_eps = float(gauge_eps)
        self.cautious = bool(cautious)
        self.bands = int(bands)
        self.band_weights: Optional[List[float]] = band_weights if self.bands > 1 else None
        self.pairs: List[Tuple[torch.Tensor, torch.Tensor]] = self._check_pairs(pairs)
        self._kept: Optional[torch.Tensor] = None  # the last step's cautious counts (kept, seen), no sync
        self._seen: Optional[torch.Tensor] = None
        self._share_sum = 0.0  # summed over the steps since the last log line
        self._share_steps = 0
        self._steps = 0
        self._step_bands: List[int] = []  # the bands observed since zero_grad() (one per micro-batch sample)
        self._mixed_warned = False
        if self.gauge and not self.pairs:
            log.info("Keel: no LoRA pairs (a LoKr / OFT adapter or a fine-tune): the gauge preconditioner has nothing "
                     "to do and the step is Automagic 4's")
        elif self.gauge:
            log.info("Keel: gauge preconditioner on %d LoRA pairs (eps %g)%s", len(self.pairs), self.gauge_eps,
                     ", cautious mask" if self.cautious else "")
        if self.bands > 1:
            edges = keel_band_edges(self.bands)
            log.info("Keel: %d noise bands (sigma edges %s)%s", self.bands,
                     " ".join(f"{e:.3g}" for e in edges),
                     f", update weights {self.band_weights}" if self.band_weights else "")

    def _check_pairs(self, pairs) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        mine = {id(p) for g in self.param_groups for p in g["params"]}
        out = []
        for pair in pairs or ():
            down, up = pair
            if id(down) not in mine or id(up) not in mine:
                raise ValueError("Keel: a LoRA pair's tensors are not among the optimizer's parameters")
            if down.dim() != 2 or up.dim() != 2 or (down.numel() and up.numel() and down.shape[0] != up.shape[1]):
                raise ValueError(f"Keel: a LoRA pair must be down [r, in] and up [out, r], got {tuple(down.shape)} "
                                 f"and {tuple(up.shape)}")
            out.append((down, up))
        return out

    # ------------------------------------------------------------------ the trainer's feed

    def observe(self, sigma) -> None:
        """The noise level(s) of a micro-batch about to be backpropagated (a tensor of any shape, or a number); the
        trainer calls it for every micro-batch of the step (the DOP pass included), after zero_grad(). Nothing
        without bands."""
        if self.bands <= 1:
            return
        vals = sigma.detach().flatten().tolist() if torch.is_tensor(sigma) else [float(sigma)]
        self._step_bands.extend(band_of(v, self.bands) for v in vals)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._step_bands = []  # a new step: the trainer zeroes before its first micro-batch (and on a retry)
        super().zero_grad(set_to_none=set_to_none)

    # ------------------------------------------------------------------ the noise bands

    @torch.no_grad()
    def _normalise_bands(self) -> None:
        """Scale every gradient to the tensor's overall power from its band's (module docstring), update the band
        statistics, and set the step's update gain from keel_band_weights. Nothing without observed sigmas."""
        bands = self._step_bands
        self._step_bands = []
        self.update_gain = None
        if self.bands <= 1 or not bands:
            return
        B = self.bands
        if self.band_weights is not None:
            w = sum(self.band_weights[b] for b in bands) / len(bands)
            if w != 1.0:
                self.update_gain = w
        b = bands[0]
        if any(x != b for x in bands):
            if not self._mixed_warned:
                log.info("Keel: a step's micro-batches span %d noise bands (keel_pair_sigma is off, or batch_size > 1 "
                         "draws a sigma per image): band normalisation acts only on steps within one band; the band "
                         "weight is averaged", len(set(bands)))
                self._mixed_warned = True
            return
        self._refresh_group_lrs()
        by_dev: dict = {}  # device -> (params, grads, states): the vector work runs per device
        for gi, group in enumerate(self.param_groups):
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                acc = getattr(p, "_accum_grad", None)  # the parent's low-precision accumulation buffer
                if acc is not None:
                    p.grad = acc
                    del p._accum_grad
                g = p.grad
                if g is None or p.numel() == 0:
                    continue
                st = self.state[p]
                if len(st) == 0:  # as the parent's step would (it then finds the state ready)
                    self._init_state(p, group)
                    if self._group_lr[gi] is None:
                        self._group_lr[gi] = st["lr"]
                ms = st.get("band_ms")
                if not isinstance(ms, torch.Tensor) or ms.numel() != B + 1:
                    st["band_ms"] = torch.zeros(B + 1, dtype=torch.float32, device=p.device)  # [bands + overall]
                    st["band_n"] = torch.zeros(B + 1, dtype=torch.float32, device=p.device)
                entry = by_dev.setdefault(g.device, ([], [], []))
                entry[0].append(p)
                entry[1].append(g)
                entry[2].append(st)
        for dev, (ps, gs, sts) in by_dev.items():
            power = torch.stack(torch._foreach_norm(gs)).to(torch.float32).square_()  # [T] the gradients' mean square
            power.div_(torch.tensor([float(p.numel()) for p in ps], device=dev))
            ms = torch.stack([st["band_ms"] for st in sts])  # [T, B + 1]
            n = torch.stack([st["band_n"] for st in sts])
            ms_b, ms_all, n_b = ms[:, b], ms[:, B], n[:, b]
            # the scale from the statistics before this step's observation; 1 until the band is trusted
            gain = torch.where((n_b >= BAND_MIN_OBS) & (ms_b > 0.0), (ms_all / ms_b.clamp(min=1e-38)).sqrt(),
                               torch.ones_like(ms_b)).clamp_(1.0 / GAIN_MAX, GAIN_MAX)
            torch._foreach_mul_(gs, list(gain.unbind(0)))
            for idx in (b, B):  # the band's and the overall statistics: a running mean, then an EMA
                cnt = n[:, idx].add_(1.0)
                ms[:, idx].add_((power - ms[:, idx]) / cnt.clamp(max=float(BAND_WINDOW)))
            torch._foreach_copy_([st["band_ms"] for st in sts], list(ms.unbind(0)))
            torch._foreach_copy_([st["band_n"] for st in sts], list(n.unbind(0)))

    def band_stats(self) -> Optional[Tuple[List[float], List[int]]]:
        """(the per-band gradient RMS relative to the tensor's overall, averaged over the tensors that saw the band;
        the steps seen per band), or None before any observation. Reads the statistics back from the GPU."""
        rows = [st for st in self.state.values() if isinstance(st.get("band_ms"), torch.Tensor)
                and st["band_ms"].numel() == self.bands + 1]
        if not rows:
            return None
        ms = torch.stack([st["band_ms"].detach().cpu() for st in rows])
        n = torch.stack([st["band_n"].detach().cpu() for st in rows])
        B = self.bands
        seen = (n[:, :B] > 0) & (ms[:, :B] > 0) & (ms[:, B:] > 0)
        rel = torch.where(seen, (ms[:, :B] / ms[:, B:].clamp(min=1e-38)).sqrt(), torch.zeros_like(ms[:, :B]))
        cnt = seen.sum(0).clamp(min=1)
        return (rel.sum(0) / cnt).tolist(), n[:, :B].max(0).values.to(torch.int64).tolist()

    # ------------------------------------------------------------------ the preconditioner

    @torch.no_grad()
    def precondition(self) -> None:
        """The gauge preconditioner (and the cautious mask) on every pair's accumulated gradient, in place. step()
        calls it; calling it earlier (after the trainer's clipping) is the same, as step() then finds the gradients
        already done (a flag, cleared by step())."""
        if getattr(self, "_preconditioned", False) or not self.gauge or not self.pairs:
            return
        self._preconditioned = True
        kept = seen = None
        for down, up in self.pairs:
            for p in (down, up):  # the parent's low-precision accumulation buffer: hand it back first
                acc = getattr(p, "_accum_grad", None)
                if acc is not None:
                    p.grad = acc
                    del p._accum_grad
            gd, gu = down.grad, up.grad
            if gd is None or gu is None or down.numel() == 0 or up.numel() == 0:
                continue  # no gradient this step, or a race switched the wrapper to LoKr (empty LoRA tensors)
            new_down, new_up = gauge_precondition(gd, gu, down, up, self.gauge_eps)
            for new, raw in ((new_down, gd), (new_up, gu)):
                new.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                if self.cautious:
                    mask = new.mul(raw).gt(0.0)
                    new.mul_(mask)
                    k = mask.sum(dtype=torch.float64)
                    n = torch.tensor(float(mask.numel()), dtype=torch.float64, device=k.device)
                    if kept is None:
                        kept, seen = k, n
                    else:
                        kept.add_(k if k.device == kept.device else k.to(kept.device))
                        seen.add_(n if n.device == seen.device else n.to(seen.device))
            down.grad = new_down if gd.dtype == torch.float32 else new_down.to(gd.dtype)
            up.grad = new_up if gu.dtype == torch.float32 else new_up.to(gu.dtype)
        self._kept, self._seen = kept, seen

    def cautious_share(self) -> Optional[float]:
        """The share of the preconditioned elements the cautious mask kept on the last step (None: no pair had a
        gradient, or the mask is off). Reads the counts back from the GPU."""
        if self._kept is None or self._seen is None:
            return None
        seen = float(self._seen)
        return float(self._kept) / seen if seen > 0 else None

    # ------------------------------------------------------------------ the periodic log

    def _log_periodic(self) -> None:
        if self.cautious and self._kept is not None:
            share = self.cautious_share()
            if share is not None:
                self._share_sum += share
                self._share_steps += 1
        if self._steps % LOG_EVERY:
            return
        if self._share_steps:
            log.info("Keel: the cautious mask kept %.1f%% of the gauge-preconditioned gradient elements over the last "
                     "%d steps", 100.0 * self._share_sum / self._share_steps, self._share_steps)
            self._share_sum = 0.0
            self._share_steps = 0
        if self.bands > 1:
            stats = self.band_stats()
            if stats is not None:
                rel, counts = stats
                log.info("Keel bands: gradient RMS per band relative to the tensor's overall (mean over tensors) %s; "
                         "steps seen per band %s", " ".join(f"{r:.2f}" if c else "-" for r, c in zip(rel, counts)),
                         " ".join(str(c) for c in counts))

    # ------------------------------------------------------------------ optimizer API

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:  # the closure makes the gradients: before the preconditioner, as in the parent
            with torch.enable_grad():
                loss = closure()
        self._normalise_bands()
        self.precondition()
        self._preconditioned = False
        try:
            super().step()
        finally:
            self.update_gain = None
        self._steps += 1
        self._log_periodic()
        return loss

    def load_state_dict(self, state_dict):
        # The parent casts every fp state tensor to the param dtype: the band statistics come back from the saved
        # fp32 values, and start fresh when their geometry no longer matches (keel_bands changed on resume).
        exact_ms = self._saved_fp32(state_dict, "band_ms")
        exact_n = self._saved_fp32(state_dict, "band_n")
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if st is None or "band_ms" not in st:
                    continue
                ms, n = exact_ms.get(p), exact_n.get(p)
                if ms is not None and n is not None and ms.numel() == self.bands + 1 and n.numel() == self.bands + 1:
                    st["band_ms"], st["band_n"] = ms, n
                else:
                    st.pop("band_ms", None)
                    st.pop("band_n", None)
