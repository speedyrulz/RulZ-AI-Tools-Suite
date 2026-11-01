"""Keel: Automagic 4 for character LoRAs (docs/keel-optimizer.md). Milestones 1 to 5 of that design: the LoRA
gauge preconditioner, the cautious mask, the noise-band normalisation, the cross-photo consensus gate, the
probe-gated lookahead and the face-token gain. With every switch off it is Automagic 4 exactly (test_keel asserts
it bitwise).

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

The accumulation hook
---------------------
Keel takes every micro-batch's gradient as it lands (a post-accumulate-grad hook on each parameter, armed by
zero_grad() and disarmed by step(), so a backward outside a step, or another optimizer's step on the same
tensors in a race, is left alone): the gradient leaves p.grad for an fp32 sum in p._accum_grad (Automagic 3's
stochastic low-precision sum for a parameter that is not fp32), so the next backward produces the next
micro-batch's gradient on its own and the hook can look at each one: its noise band (below) and its signs (the
consensus gate). gather() writes the sum back to p.grad; the trainer calls it after the step's last backward,
before the replica reduce and gradient clipping, and step() does it itself otherwise. A data-parallel replica's
parameters are not the optimizer's, so its micro-batches never reach the hook: the reduce hands their sum to
take_replica(primary_param, grad), which scales it by the step's band (when the step sits in one band: paired
sigma) and counts it as one vote. A loop that hands the optimizer the step's sum directly (the parallel race's
collect_grads) never fires the hook: the bands then act per step, the gate is 1 (logged once).

Noise-band normalisation (keel_bands, keel_band_weights)
--------------------------------------------------------
Gradient power varies a lot with the noise level sigma (Min-SNR; Addressing Negative Transfer in Diffusion
Models; EDM2's learned per-level weights), and one second moment per weight is set by the loudest band, usually
not the one where the face forms (sigma 0.2 to 0.5 at 1 MP): the quiet bands take small steps. Keel keeps, per
weight tensor and per band of logit(sigma) (keel_bands equal bands over [-4, 4], the ends taking what lies
beyond), a running mean of the gradient's power (its mean square), plus the overall one; each micro-batch's
gradient is scaled by sqrt(overall / band) in the hook before it enters the sum (a loop without the hook scales
the step's sum by its one band), so the gradient enters the second moment at the tensor's usual scale whatever
its band and every band's step is sized alike. A band is trusted after BAND_MIN_OBS steps that saw it (until
then the scale is 1), the lift is capped at GAIN_MAX, and a micro-batch whose images span several bands
(batch_size > 1 with keel_pair_sigma off) is left alone. keel_band_weights then multiplies the step's update
(not the decay) by the band's weight (Automagic4.update_gain): "" = 1 everywhere, "identity" = a hump over the
face bands, or one number per band; the mean of the step's micro-batches' weights when they differ. The trainer
calls observe(sigma) right before each micro-batch's backward (the CPU sigma: no sync); without those calls the
bands do nothing. The per-band gradient RMS relative to the overall, averaged over the tensors, is logged every
250 steps: the measured imbalance the mechanism answers.

Consensus gate (keel_consensus, keel_consensus_power)
-----------------------------------------------------
A character LoRA should learn what is true in every photo (the face, the eyes, the hair line, the build) and
not what varies between them (the outfit, the background, the light). The AND-mask (Parascandolo et al., ICLR
2021) and SAND-mask show that moving a weight only where the per-example gradients agree in sign suppresses the
per-example shortcuts and keeps the shared mechanism. The hook adds sign(g_k) of every gradient of the step
(the DOP pass's included) into an int8 buffer per tensor (1 B/weight for the step; up to 127 gradients). With
K gradients seen, a = |agree| / K in [0, 1] and the gate is floor + (1 - floor) a^power, floor = 1 -
keel_consensus (0 = off, 1 = a hard AND-mask that keeps only the unanimous elements). It multiplies the update
after the sensitivity balance and before the sign history, so the vote sees the gated magnitude (an element
the photos disagree on carries little vote weight, like a dead one). It needs two or more gradients per step
(gradient_accumulation >= 2, or the DOP pass); with one, a = 1 everywhere and the gate is 1. Paired sigma
matters here: sign disagreement between micro-batches at different noise levels is mostly the noise level,
not the content. The mean gate is logged every 250 steps.

Probe-gated lookahead (keel_lookahead, keel_lookahead_pullback, keel_lookahead_lr, keel_lookahead_grace)
-----------------------------------------------------------------------------------------------------
The likeness probes with the held-out judge are the one signal nothing trains on. Lookahead (Zhang et al. 2019)
keeps slow weights and pulls the fast weights toward them; here the probe decides. Keel keeps an anchor copy of
every trained tensor (state["anchor"], fp32, with the anchor's score mirrored per tensor as state["anchor_score"]
so both ride the state dict, the park and the relay) and the trainer calls on_probe(score, band, ema) after each
probe decision with the controller's smoothed decision score, its noise band and the weight average the probe
scored (or, for a probe rendered on the probe GPU from an older copy, that copy): the first probe sets the anchor;
a score above the anchor's by more than the band accepts (the anchor becomes the weights that scored: the
average's shadow when lora_ema is on, the raw weights otherwise, the older copy for an async probe); any other
probe is a miss, and after more than keel_lookahead_grace misses in a row the weights are pulled: raw <- anchor +
(1 - pullback) (raw - anchor), the shadow the same way, and every lr mirror times keel_lookahead_lr (clamped to
the rails); the second moments and the sign histories stay. No pull while the score rose since the previous
probe (a slow climb is never pulled; an oscillating run is damped to its best), and a pull starts the miss count
over. One fp32 copy of the LoRA; off by default until raced.

Face-token gain (keel_face_gain, keel_face_gain_max, keel_face_gain_beta)
-------------------------------------------------------------------------
Weights whose gradient mostly flows through the face's tokens take a larger step, weights serving the background a
smaller one; a positive scalar on the diffusion update, so the step still descends the diffusion loss and cannot be
gamed toward features that fool a face recognizer. The LoRA layers' face tap (lora.arm_face_tap, armed by the
trainer per micro-batch with each sample's face token mask from its region map) hands face_energy(lora_up, e_face,
e_total) the squared norm of lora_up's gradient from the face tokens and from all tokens of the samples with a face;
summed over the step's micro-batches, share = e_face / e_total per tensor, and gain = clamp((share / median share
over the reported tensors) ** keel_face_gain, 1 / keel_face_gain_max, keel_face_gain_max), EMA'd with
keel_face_gain_beta into state["face_gain"] (a 0-dim fp32, the pair's lora_down the same), multiplied into the
update through _element_gain with the consensus gate. Tensors without a report keep their gain; a step without one
changes nothing. Off by default until raced.

Pairs: build_optimizer passes the (lora_down, lora_up) pairs of the injected LoRA layers; LoKr, OFT and a
fine-tune have none, and the step is then Automagic 4's. A race switching a wrapper to LoKr leaves its LoRA
tensors empty, which the step skips.
"""

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..config import KEEL_BAND_LOGIT_RANGE, keel_band_edges
from .automagic4 import Automagic4

log = logging.getLogger("rulztrainer")

GAUGE_FLOOR = 1e-12  # absolute floor of delta (both Grams zero: nothing to divide by, and nothing to scale)
LOG_EVERY = 250  # steps between the periodic log lines (cautious share, band statistics, consensus gate)
BAND_MIN_OBS = 8  # a band is normalised on its own statistics after this many steps that saw it
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


def consensus_gate(agree: torch.Tensor, votes: int, consensus: float, power: float) -> torch.Tensor:
    """floor + (1 - floor) (|agree| / votes)^power in fp32, floor = 1 - consensus."""
    a = agree.abs().to(torch.float32).div_(float(votes))
    if power != 1.0:
        a.pow_(power)
    floor = 1.0 - consensus
    return a.mul_(1.0 - floor).add_(floor)


class Keel(Automagic4):
    """Automagic 4 plus the gauge preconditioner, the cautious mask, the noise bands and the consensus gate
    (module docstring).

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
    consensus : float in [0, 1]
        The consensus gate's strength: 0 = off, 1 = a hard AND-mask. Needs the accumulation hook.
    consensus_power : float > 0
        The exponent on the agreement share.
    lookahead : bool
        The probe-gated lookahead (fed by on_probe).
    lookahead_pullback : float in (0, 1]
        How far a pull moves the weights toward the anchor.
    lookahead_lr : float in (0, 1]
        The lr multiplier at a pull.
    lookahead_grace : int >= 0
        Misses before a pull.
    face_gain : float >= 0
        The exponent on (face share / median share); 0 = off.
    face_gain_max : float >= 1
        The gain stays within [1 / max, max].
    face_gain_beta : float in [0, 1)
        The EMA of the gain over steps.
    """

    def __init__(self, params, *args, pairs: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
                 gauge: bool = True, gauge_eps: float = 0.01, cautious: bool = True, bands: int = 8,
                 band_weights: Optional[Sequence[float]] = None, consensus: float = 0.5,
                 consensus_power: float = 2.0, lookahead: bool = False, lookahead_pullback: float = 0.5,
                 lookahead_lr: float = 0.7, lookahead_grace: int = 2, face_gain: float = 0.0,
                 face_gain_max: float = 2.0, face_gain_beta: float = 0.9, **kwargs):
        if not gauge_eps > 0.0:
            raise ValueError(f"gauge_eps ({gauge_eps}) must be above 0")
        if int(bands) < 1:
            raise ValueError(f"bands ({bands}) must be at least 1")
        if band_weights is not None:
            band_weights = [float(w) for w in band_weights]
            if len(band_weights) != int(bands) or any(not w > 0.0 for w in band_weights):
                raise ValueError(f"band_weights must be {int(bands)} numbers above 0, got {band_weights}")
        if not 0.0 <= consensus <= 1.0:
            raise ValueError(f"consensus ({consensus}) must be within [0, 1]")
        if not consensus_power > 0.0:
            raise ValueError(f"consensus_power ({consensus_power}) must be above 0")
        if not 0.0 < lookahead_pullback <= 1.0:
            raise ValueError(f"lookahead_pullback ({lookahead_pullback}) must be within (0, 1]")
        if not 0.0 < lookahead_lr <= 1.0:
            raise ValueError(f"lookahead_lr ({lookahead_lr}) must be within (0, 1]")
        if int(lookahead_grace) < 0:
            raise ValueError(f"lookahead_grace ({lookahead_grace}) must be 0 or more")
        if not face_gain >= 0.0:
            raise ValueError(f"face_gain ({face_gain}) must be 0 (off) or above")
        if not face_gain_max >= 1.0:
            raise ValueError(f"face_gain_max ({face_gain_max}) must be 1 or more")
        if not 0.0 <= face_gain_beta < 1.0:
            raise ValueError(f"face_gain_beta ({face_gain_beta}) must be within [0, 1)")
        # Read by the hook (registered by the parent's constructor for low-precision parameters), so before it.
        self.gauge = bool(gauge)
        self.gauge_eps = float(gauge_eps)
        self.cautious = bool(cautious)
        self.bands = int(bands)
        self.band_weights: Optional[List[float]] = band_weights if self.bands > 1 else None
        self.consensus = float(consensus)
        self.consensus_power = float(consensus_power)
        self.lookahead = bool(lookahead)
        self.lookahead_pullback = float(lookahead_pullback)
        self.lookahead_lr = float(lookahead_lr)
        self.lookahead_grace = int(lookahead_grace)
        self._misses = 0  # probes since the last accept or pull that did not beat the anchor
        self._last_score: Optional[float] = None  # the previous probe's score (no pull while it rises)
        self.face_gain = float(face_gain)
        self.face_gain_max = float(face_gain_max)
        self.face_gain_beta = float(face_gain_beta)
        self._face_e: Dict[torch.Tensor, List[torch.Tensor]] = {}  # per lora_up: [face, total] energy of the step
        self._face_seen = False  # a face report has arrived at some step
        self._armed = False  # the hook takes gradients between zero_grad() and step() only
        self._cur_band: Optional[int] = None  # the band of the micro-batch about to be backpropagated
        self._step_bands: List[int] = []  # the bands observed since zero_grad() (one per micro-batch image)
        self._hooked = False  # whether the hook took a gradient this step
        self._agree: Dict[torch.Tensor, torch.Tensor] = {}  # param -> int8 sign sums of the step's gradients
        self._votes: Dict[torch.Tensor, int] = {}  # param -> gradients seen this step
        self._pw: Dict[torch.Tensor, torch.Tensor] = {}  # param -> [bands] fp32 power sums of the step's gradients
        self._cnt: Dict[torch.Tensor, List[int]] = {}  # param -> gradients seen per band this step
        self._band_gain: Dict[torch.Tensor, torch.Tensor] = {}  # param -> [bands] fp32 scale per band (statistics)
        super().__init__(params, *args, **kwargs)
        self._register_hooks([p for g in self.param_groups for p in g["params"]])
        self.pairs: List[Tuple[torch.Tensor, torch.Tensor]] = self._check_pairs(pairs)
        self._pair_down: Dict[torch.Tensor, torch.Tensor] = {up: down for down, up in self.pairs}
        self._kept: Optional[torch.Tensor] = None  # the last step's cautious counts (kept, seen), no sync
        self._seen: Optional[torch.Tensor] = None
        self._share_sum = 0.0  # summed over the steps since the last log line
        self._share_steps = 0
        self._gate_sum: Dict[torch.device, torch.Tensor] = {}  # the gate's element sum since the last log line
        self._gate_n = 0
        self._gate_votes = 0  # gradients per step, summed over the gated steps since the last log line
        self._gate_steps = 0
        self._steps = 0
        self._mixed_warned = False
        self._noted: set = set()  # one-time notes already logged
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
        if self.consensus > 0.0:
            log.info("Keel: consensus gate %g (power %g): floor %g on the elements the step's gradients disagree on",
                     self.consensus, self.consensus_power, 1.0 - self.consensus)
        if self.lookahead:
            log.info("Keel: probe-gated lookahead (pull-back %g, lr x%g at a pull, %d probes of grace)",
                     self.lookahead_pullback, self.lookahead_lr, self.lookahead_grace)
        if self.face_gain > 0.0:
            log.info("Keel: face-token gain (exponent %g, within [%.3g, %g], EMA %g) on %d LoRA pairs",
                     self.face_gain, 1.0 / self.face_gain_max, self.face_gain_max, self.face_gain_beta,
                     len(self.pairs))

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

    def _note(self, key: str, msg: str, *args) -> None:
        if key not in self._noted:
            self._noted.add(key)
            log.info(msg, *args)

    def _ensure_state(self, gi: int, group: dict, p: torch.Tensor) -> dict:
        """p's state, initialised as the parent's step would (which then finds it ready)."""
        st = self.state[p]
        if len(st) == 0:
            self._init_state(p, group)
            if self._group_lr[gi] is None:
                self._group_lr[gi] = st["lr"]
        return st

    # ------------------------------------------------------------------ the accumulation hook

    def _register_hooks(self, params) -> None:
        # The parent registered the hook for the low-precision parameters (through _make_accum_hook); the fp32 ones
        # (every LoRA parameter) get it here. Fused mode has its own hooks and never reaches this class in the trainer.
        if self.fused:
            return
        for p in params:
            if p.requires_grad and p.dtype == torch.float32:
                self._hook_handles.append(p.register_post_accumulate_grad_hook(self._make_accum_hook()))

    def add_param_group(self, param_group: dict) -> None:
        super().add_param_group(param_group)
        if hasattr(self, "_hook_handles"):  # after construction (torch's constructor adds the first groups through here)
            self._register_hooks(param_group["params"])

    def _make_accum_hook(self):
        def _hook(p: torch.Tensor) -> None:
            if p.grad is not None and self._armed:
                self._take(p)

        return _hook

    @torch.no_grad()
    def _take(self, p: torch.Tensor) -> None:
        """One gradient of the step for p, off p.grad: its band's power and scale, its signs, then into the sum."""
        g = p.grad
        p.grad = None
        self._hooked = True
        b = self._cur_band
        if self.bands > 1 and b is not None:
            g32 = g if g.dtype == torch.float32 else g.to(torch.float32)
            pw = self._pw.get(p)
            if pw is None:
                pw = self._pw[p] = torch.zeros(self.bands, dtype=torch.float32, device=g.device)
                self._cnt[p] = [0] * self.bands
            pw[b].add_(g32.square().mean())  # the raw power, before the scale
            self._cnt[p][b] += 1
            gain = self._band_gain.get(p)
            if gain is not None:
                g = g32 * gain[b]
        if self.consensus > 0.0:
            self._vote(p, g)
        acc = getattr(p, "_accum_grad", None)
        if p.dtype == torch.float32:
            if acc is None:
                p._accum_grad = g if g.dtype == torch.float32 else g.to(torch.float32)  # the hook owns it now
            else:
                acc.add_(g)
        elif acc is None:  # Automagic 3's stochastic low-precision sum
            p._accum_grad = g if g.dtype == p.dtype else self._stochastic_round(g, p.dtype)
        else:
            self._stochastic_copy_(acc, acc.to(torch.float32).add_(g.to(torch.float32)))

    def _vote(self, p: torch.Tensor, g: torch.Tensor) -> None:
        s = g.sign().to(torch.int8)
        agree = self._agree.get(p)
        if agree is None:
            self._agree[p] = s
        else:
            agree.add_(s)
        self._votes[p] = self._votes.get(p, 0) + 1

    @torch.no_grad()
    def take_replica(self, p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """A data-parallel replica's gradient of the step for p (its micro-batches' sum; the replica's parameters are
        not the optimizer's, so the hook never saw them), before the reduce adds it to the primary's: scaled by the
        step's band when the step's observations sit in one band (paired sigma; a replica's share of a step that
        spans bands joins unscaled), and one vote. The trainer passes it to DataParallelLoRA.reduce_grads."""
        if not self._armed:
            return g
        bands = self._step_bands
        if self.bands > 1 and bands and all(b == bands[0] for b in bands):
            gain = self._band_gain.get(p)
            if gain is not None:
                g = g * gain[bands[0]]
        if self.consensus > 0.0:
            self._vote(p, g)
        return g

    @torch.no_grad()
    def gather(self) -> None:
        """The step's accumulated gradient back on p.grad. The trainer calls it after the step's last backward, before
        gradient clipping and the replica reduce; step() calls it itself otherwise."""
        for group in self.param_groups:
            for p in group["params"]:
                acc = getattr(p, "_accum_grad", None)
                if acc is not None:
                    p.grad = acc
                    del p._accum_grad

    def zero_grad(self, set_to_none: bool = True) -> None:
        # A new step: the trainer zeroes before its first micro-batch (and before an out-of-memory retry, whose
        # partial sums must go).
        self._armed = True
        self._cur_band = None
        self._step_bands = []
        self._hooked = False
        self._agree.clear()
        self._votes.clear()
        self._pw.clear()
        self._cnt.clear()
        for group in self.param_groups:
            for p in group["params"]:
                if hasattr(p, "_accum_grad"):
                    del p._accum_grad
        super().zero_grad(set_to_none=set_to_none)

    # ------------------------------------------------------------------ the trainer's feed

    def observe(self, sigma, hook: bool = True) -> None:
        """The noise level(s) of the micro-batch about to be backpropagated (a tensor of any shape, or a number;
        the CPU tensor avoids a sync); the trainer calls it before every backward of the step (the DOP pass
        included), after zero_grad(). hook=False for a micro-batch of a data-parallel replica, whose gradient
        reaches take_replica instead of the hook: it counts toward the step's bands (the band weight, the one-band
        check) and nothing else. Nothing without bands."""
        if self.bands <= 1:
            return
        vals = sigma.detach().flatten().tolist() if torch.is_tensor(sigma) else [float(sigma)]
        bands = [band_of(v, self.bands) for v in vals]
        self._step_bands.extend(bands)
        if hook:
            self._cur_band = bands[0] if bands and all(b == bands[0] for b in bands) else None

    # ------------------------------------------------------------------ the noise bands

    def _gain_from(self, ms: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """[T, bands] scale per band from stacked statistics [T, bands + 1]: sqrt(overall / band) where the band is
        trusted, 1 elsewhere, within [1/GAIN_MAX, GAIN_MAX]."""
        B = self.bands
        ms_b, ms_all = ms[:, :B], ms[:, B:]
        return torch.where((n[:, :B] >= BAND_MIN_OBS) & (ms_b > 0.0), (ms_all / ms_b.clamp(min=1e-38)).sqrt(),
                           torch.ones_like(ms_b)).clamp_(1.0 / GAIN_MAX, GAIN_MAX)

    @torch.no_grad()
    def _bands_step(self) -> None:
        """At step(): the band weight as this step's update gain; the band statistics updated with the step's
        observations (the hook's per-gradient power sums, or, without the hook, the accumulated gradient in the
        step's one band, which is scaled here as the hook would have); the scale per band for the next step."""
        bands = self._step_bands
        self._step_bands = []
        self._cur_band = None
        self.update_gain = None
        if self.bands <= 1:
            return
        if bands and self.band_weights is not None:
            w = sum(self.band_weights[b] for b in bands) / len(bands)
            if w != 1.0:
                self.update_gain = w
        B = self.bands
        hooked = bool(self._pw)
        one = None
        if not hooked:
            if not bands:
                return
            one = bands[0]
            if any(x != one for x in bands):
                if not self._mixed_warned:
                    log.info("Keel: a step's gradient spans %d noise bands and reached the optimizer as one sum "
                             "(keel_pair_sigma off, or batch_size > 1 drawing a sigma per image, in a loop that "
                             "sums the micro-batches itself): band normalisation acts only on steps within one "
                             "band; the band weight is averaged", len(set(bands)))
                    self._mixed_warned = True
                return
        self._refresh_group_lrs()
        by_dev: dict = {}  # device -> (params, grads, states): the vector work runs per device
        for gi, group in enumerate(self.param_groups):
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                g = p.grad
                if g is None or p.numel() == 0 or (hooked and p not in self._pw):
                    continue
                st = self._ensure_state(gi, group, p)
                ms = st.get("band_ms")
                if not isinstance(ms, torch.Tensor) or ms.numel() != B + 1:
                    st["band_ms"] = torch.zeros(B + 1, dtype=torch.float32, device=p.device)  # [bands + overall]
                    st["band_n"] = torch.zeros(B + 1, dtype=torch.float32, device=p.device)
                entry = by_dev.setdefault(g.device, ([], [], []))
                entry[0].append(p)
                entry[1].append(g)
                entry[2].append(st)
        for dev, (ps, gs, sts) in by_dev.items():
            ms = torch.stack([st["band_ms"] for st in sts])  # [T, B + 1]
            n = torch.stack([st["band_n"] for st in sts])
            if hooked:
                pw = torch.stack([self._pw[p] for p in ps])  # [T, B] the step's power sums per band
                cnt = torch.tensor([self._cnt[p] for p in ps], dtype=torch.float32, device=dev)
            else:
                power = torch.stack(torch._foreach_norm(gs)).to(torch.float32).square_()  # [T] mean square
                power.div_(torch.tensor([float(p.numel()) for p in ps], device=dev))
                pw = torch.zeros(len(ps), B, dtype=torch.float32, device=dev)
                cnt = torch.zeros_like(pw)
                pw[:, one] = power
                cnt[:, one] = 1.0
                # the scale from the statistics before this step's observation, as the hook applies per gradient
                torch._foreach_mul_(gs, list(self._gain_from(ms, n)[:, one].unbind(0)))
            seen = cnt > 0.0
            mean_b = pw / cnt.clamp(min=1.0)
            n_b = n[:, :B]
            n_b.add_(seen.to(torch.float32))
            ms[:, :B].add_(torch.where(seen, mean_b - ms[:, :B], torch.zeros_like(mean_b))
                           / n_b.clamp(min=1.0, max=float(BAND_WINDOW)))
            tot = cnt.sum(1)
            any_ = tot > 0.0
            mean_all = pw.sum(1) / tot.clamp(min=1.0)
            n[:, B].add_(any_.to(torch.float32))
            ms[:, B].add_(torch.where(any_, mean_all - ms[:, B], torch.zeros_like(mean_all))
                          / n[:, B].clamp(min=1.0, max=float(BAND_WINDOW)))
            torch._foreach_copy_([st["band_ms"] for st in sts], list(ms.unbind(0)))
            torch._foreach_copy_([st["band_n"] for st in sts], list(n.unbind(0)))
            for p, row in zip(ps, self._gain_from(ms, n).unbind(0)):
                self._band_gain[p] = row

    def _refresh_band_gains(self) -> None:
        """The scale per band of every tensor with statistics (after a state load)."""
        self._band_gain.clear()
        if self.bands <= 1:
            return
        for p, st in self.state.items():
            ms, n = st.get("band_ms"), st.get("band_n")
            if isinstance(ms, torch.Tensor) and isinstance(n, torch.Tensor) and ms.numel() == self.bands + 1:
                self._band_gain[p] = self._gain_from(ms.unsqueeze(0), n.unsqueeze(0))[0]

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

    # ------------------------------------------------------------------ the consensus gate

    def _element_gain(self, p: torch.Tensor, i0: int = 0, i1: Optional[int] = None) -> Optional[torch.Tensor]:
        """The consensus gate on p's update (rows i0:i1), times the face-token gain (a 0-dim tensor when there is no
        gate), or None for neither."""
        gate = None
        agree = self._agree.get(p)
        if agree is not None and self._votes.get(p, 0) >= 2:
            gate = consensus_gate(agree if i1 is None else agree[i0:i1], self._votes[p], self.consensus,
                                  self.consensus_power)
            acc = self._gate_sum.get(gate.device)
            s = gate.sum()
            self._gate_sum[gate.device] = s if acc is None else acc.add_(s)
            self._gate_n += gate.numel()
        fg = None
        if self.face_gain > 0.0:
            st = self.state.get(p)
            fg = st.get("face_gain") if st else None
        if fg is None:
            return gate
        return fg if gate is None else gate.mul_(fg)

    def gate_mean(self) -> Optional[float]:
        """The consensus gate's mean over the elements gated since the last log line (None: none). Syncs."""
        if not self._gate_n:
            return None
        return sum(float(t) for t in self._gate_sum.values()) / self._gate_n

    def _consensus_step_notes(self) -> None:
        if self.consensus <= 0.0:
            return
        votes = max(self._votes.values(), default=0)
        if votes >= 2:
            self._gate_votes += votes
            self._gate_steps += 1
        elif self._hooked or self._agree:
            self._note("votes", "Keel: the consensus gate needs two or more gradients per step (gradient_accumulation "
                       ">= 2, or the DOP pass); with one gradient the gate is 1")
        elif any(p.grad is not None for g in self.param_groups for p in g["params"]):
            self._note("hook", "Keel: the consensus gate sees no per-micro-batch gradients in this loop (the parallel "
                       "race and a replica reduce hand the optimizer the step's sum): the gate is 1")

    # ------------------------------------------------------------------ the face-token gain

    def face_energy(self, p: torch.Tensor, e_face: torch.Tensor, e_total: torch.Tensor) -> None:
        """The face tap's report for one micro-batch (lora.arm_face_tap): for `p` (a lora_up weight) the squared norm
        of its gradient from the face tokens and from all tokens of the samples with a face, 0-dim tensors (no sync).
        Summed over the step's micro-batches; a backward outside a step is ignored."""
        if self.face_gain <= 0.0 or not self._armed:
            return
        acc = self._face_e.get(p)
        if acc is None:
            self._face_e[p] = [e_face.detach().to(torch.float32), e_total.detach().to(torch.float32)]
        else:
            acc[0] = acc[0] + e_face.detach().to(acc[0].device, torch.float32)
            acc[1] = acc[1] + e_total.detach().to(acc[1].device, torch.float32)

    @torch.no_grad()
    def _face_step(self) -> None:
        """Before the step: each reported tensor's face share, its gain against the median share of the step's
        reported tensors (torch's median: the lower middle of an even count), clamped, EMA'd into state["face_gain"]
        (the pair's lora_down the same). Tensors without a report, or whose report had no energy, keep their gain.
        No sync."""
        if self.face_gain <= 0.0 or not self._face_e:
            return
        self._face_seen = True
        ups = list(self._face_e)
        dev = self._face_e[ups[0]][1].device
        ef = torch.stack([self._face_e[p][0].to(dev) for p in ups])
        et = torch.stack([self._face_e[p][1].to(dev) for p in ups])
        live = et > 0.0
        share = ef / et.clamp(min=GAUGE_FLOOR)
        med = torch.where(live, share, torch.full_like(share, float("nan"))).nanmedian().clamp(min=GAUGE_FLOOR)
        gain = (share / med).pow(self.face_gain).clamp_(1.0 / self.face_gain_max, self.face_gain_max)
        todo: Dict[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = {}
        for p, g, ok in zip(ups, gain, live):
            todo[p] = (g, ok)
            down = self._pair_down.get(p)
            if down is not None:
                todo[down] = (g, ok)
        beta = self.face_gain_beta
        for gi, group in enumerate(self.param_groups):
            for t in group["params"]:
                got = todo.get(t)
                if got is None:
                    continue
                g, ok = got
                st = self._ensure_state(gi, group, t)
                fg = st.get("face_gain")
                if fg is None:
                    fg = st["face_gain"] = torch.ones((), dtype=torch.float32, device=t.device)
                new = fg * beta + g.to(fg.device) * (1.0 - beta)
                fg.copy_(torch.where(ok.to(fg.device), new, fg))

    def face_gain_stats(self) -> Optional[Tuple[float, float, float, int]]:
        """(min, median, max, tensors) of the face-token gain over the lora_up tensors that have one (None: none).
        Syncs."""
        downs = set(self._pair_down.values())
        vals = [st["face_gain"] for p, st in self.state.items() if p not in downs and "face_gain" in st]
        if not vals:
            return None
        t = torch.stack([v.to(vals[0].device) for v in vals])
        return float(t.min()), float(t.median()), float(t.max()), len(vals)

    # ------------------------------------------------------------------ the probe-gated lookahead

    @property
    def anchor_score(self) -> Optional[float]:
        """The anchor's probe score, read from the state (None: no anchor). Syncs."""
        for st in self.state.values():
            s = st.get("anchor_score")
            if isinstance(s, torch.Tensor):
                return float(s)
        return None

    @torch.no_grad()
    def on_probe(self, score, band: float, ema=None, weights: Optional[dict] = None) -> Optional[str]:
        """After a likeness probe's decision: `score` the controller's smoothed decision score (None: the probe found
        no face, and nothing happens), `band` its noise band, `ema` the weight average the probe scored (a LoraEMA:
        its shadow is the weights that scored and follows a pull; None for the raw weights), `weights` the weights
        that scored when they are neither ({param: tensor}: a probe rendered elsewhere from an older copy, which a
        pull leaves alone). Returns what happened:
        "anchor" (the first), "accept" (a new anchor), "rising" (no gain over the anchor, but up since the previous
        probe: no pull), "hold" (a miss within the grace), "pull", or None with the lookahead off."""
        if not self.lookahead or score is None:
            return None
        score, band = float(score), float(band)
        last, self._last_score = self._last_score, score
        anchored = self.anchor_score
        if anchored is None:
            self._set_anchor(score, ema, weights)
            log.info("Keel lookahead: anchor set at %.1f%% (optimizer step %d)", score, self._steps)
            return "anchor"
        if score > anchored + band:
            self._set_anchor(score, ema, weights)
            self._misses = 0
            log.info("Keel lookahead: accept, anchor %.1f%% -> %.1f%% (optimizer step %d)", anchored, score,
                     self._steps)
            return "accept"
        self._misses += 1
        if last is not None and score > last:
            return "rising"
        if self._misses <= self.lookahead_grace:
            return "hold"
        self._pull(ema)
        log.info("Keel lookahead: pull toward the anchor (%.1f%%) after %d probes without a gain (score %.1f%%): "
                 "weights %.0f%% of the way back, lr x%g (optimizer step %d)", anchored, self._misses, score,
                 100.0 * self.lookahead_pullback, self.lookahead_lr, self._steps)
        self._misses = 0
        return "pull"

    @torch.no_grad()
    def _set_anchor(self, score: float, ema, weights: Optional[dict] = None) -> None:
        shadow = dict(zip(ema.params, ema.shadow)) if ema is not None else {}
        if weights:
            shadow = {**shadow, **weights}
        self._refresh_group_lrs()
        for gi, group in enumerate(self.param_groups):
            for p in group["params"]:
                if not p.requires_grad or p.numel() == 0:
                    continue
                st = self._ensure_state(gi, group, p)
                src = shadow.get(p, p)
                a = st.get("anchor")
                if isinstance(a, torch.Tensor) and a.shape == p.shape and a.dtype == torch.float32:
                    a.copy_(src)
                else:
                    st["anchor"] = src.detach().to(p.device, torch.float32, copy=True)
                st["anchor_score"] = torch.tensor(score, dtype=torch.float32, device=p.device)

    @torch.no_grad()
    def _pull(self, ema) -> None:
        pull = self.lookahead_pullback
        shadow = dict(zip(ema.params, ema.shadow)) if ema is not None else {}
        for group in self.param_groups:
            lo, hi = float(group["min_lr"]), float(group["max_lr"])
            for p in group["params"]:
                st = self.state.get(p)
                a = st.get("anchor") if st else None
                if not isinstance(a, torch.Tensor):
                    continue
                p.copy_(torch.lerp(p.detach().to(torch.float32), a, pull))
                s = shadow.get(p)
                if s is not None:
                    s.lerp_(a, pull)
                lr = st.get("lr")
                if isinstance(lr, torch.Tensor):
                    lr.mul_(self.lookahead_lr).clamp_(min=lo, max=hi)

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
            for p in (down, up):  # a sum the hook still holds: hand it back first
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
        mean = self.gate_mean()
        if mean is not None:
            log.info("Keel: consensus gate mean %.3f over the last %d steps (%.1f gradients per step)", mean,
                     self._gate_steps, self._gate_votes / max(1, self._gate_steps))
        if self.face_gain > 0.0:
            stats = self.face_gain_stats()
            if stats is not None:
                lo, med, hi, n = stats
                log.info("Keel: face-token gain over %d LoRA pairs: min %.2f median %.2f max %.2f", n, lo, med, hi)
            elif not self._face_seen:
                self._note("face", "Keel: the face-token gain has had no face report (no region maps with a face, or "
                           "a loop without the face tap: a replica, the parallel race): the gain is 1")
        self._gate_sum.clear()
        self._gate_n = 0
        self._gate_votes = 0
        self._gate_steps = 0

    # ------------------------------------------------------------------ optimizer API

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:  # the closure makes the gradients: before the preconditioner, as in the parent
            with torch.enable_grad():
                loss = closure()
        self.gather()
        self._bands_step()
        self._face_step()
        self.precondition()
        self._preconditioned = False
        self._consensus_step_notes()
        try:
            super().step()
        finally:
            self.update_gain = None
            self._armed = False
            self._hooked = False
            self._agree.clear()
            self._votes.clear()
            self._pw.clear()
            self._cnt.clear()
            self._face_e.clear()
        self._steps += 1
        self._log_periodic()
        return loss

    def load_state_dict(self, state_dict):
        # The parent casts every fp state tensor to the param dtype: the band statistics and the anchor come back
        # from the saved fp32 values. The band statistics start fresh when their geometry no longer matches
        # (keel_bands changed on resume); the anchor is dropped when its shape does not match or the lookahead is off.
        exact = {k: self._saved_fp32(state_dict, k)
                 for k in ("band_ms", "band_n", "anchor", "anchor_score", "face_gain")}
        super().load_state_dict(state_dict)
        self._misses = 0
        self._last_score = None
        self._face_e.clear()
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if st is None:
                    continue
                if "band_ms" in st:
                    ms, n = exact["band_ms"].get(p), exact["band_n"].get(p)
                    if (ms is not None and n is not None and ms.numel() == self.bands + 1
                            and n.numel() == self.bands + 1):
                        st["band_ms"], st["band_n"] = ms, n
                    else:
                        st.pop("band_ms", None)
                        st.pop("band_n", None)
                if "anchor" in st or "anchor_score" in st:
                    a, s = exact["anchor"].get(p), exact["anchor_score"].get(p)
                    if self.lookahead and a is not None and s is not None and a.shape == p.shape:
                        st["anchor"], st["anchor_score"] = a, s
                    else:
                        st.pop("anchor", None)
                        st.pop("anchor_score", None)
                if "face_gain" in st:
                    fg = exact["face_gain"].get(p)
                    if self.face_gain > 0.0 and fg is not None and fg.dim() == 0:
                        st["face_gain"] = fg
                    else:
                        st.pop("face_gain", None)
        self._refresh_band_gains()
