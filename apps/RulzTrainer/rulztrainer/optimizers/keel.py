"""Keel: Automagic 4 for character LoRAs (docs/keel-optimizer.md). This is milestone 1 of that design: the LoRA
gauge preconditioner and the cautious mask. With both off it is Automagic 4 exactly (test_keel asserts it bitwise).

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

Pairs: build_optimizer passes the (lora_down, lora_up) pairs of the injected LoRA layers; LoKr, OFT and a
fine-tune have none, and the step is then Automagic 4's. A race switching a wrapper to LoKr leaves its LoRA
tensors empty, which the step skips.
"""

import logging
from typing import List, Optional, Sequence, Tuple

import torch

from .automagic4 import Automagic4

log = logging.getLogger("rulztrainer")

GAUGE_FLOOR = 1e-12  # absolute floor of delta (both Grams zero: nothing to divide by, and nothing to scale)
SHARE_LOG_EVERY = 250  # steps between the cautious share log lines


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
    """Automagic 4 plus the gauge preconditioner and the cautious mask (module docstring).

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
    """

    def __init__(self, params, *args, pairs: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
                 gauge: bool = True, gauge_eps: float = 0.01, cautious: bool = True, **kwargs):
        if not gauge_eps > 0.0:
            raise ValueError(f"gauge_eps ({gauge_eps}) must be above 0")
        super().__init__(params, *args, **kwargs)
        self.gauge = bool(gauge)
        self.gauge_eps = float(gauge_eps)
        self.cautious = bool(cautious)
        self.pairs: List[Tuple[torch.Tensor, torch.Tensor]] = self._check_pairs(pairs)
        self._kept: Optional[torch.Tensor] = None  # the last step's cautious counts (kept, seen), no sync
        self._seen: Optional[torch.Tensor] = None
        self._share_sum = 0.0  # summed over the steps since the last log line
        self._share_steps = 0
        if self.gauge and not self.pairs:
            log.info("Keel: no LoRA pairs (a LoKr / OFT adapter or a fine-tune): the gauge preconditioner has nothing "
                     "to do and the step is Automagic 4's")
        elif self.gauge:
            log.info("Keel: gauge preconditioner on %d LoRA pairs (eps %g)%s", len(self.pairs), self.gauge_eps,
                     ", cautious mask" if self.cautious else "")

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

    def _log_share(self) -> None:
        if self._kept is None:
            return
        share = self.cautious_share()
        if share is None:
            return
        self._share_sum += share
        self._share_steps += 1
        if self._share_steps >= SHARE_LOG_EVERY:
            log.info("Keel: the cautious mask kept %.1f%% of the gauge-preconditioned gradient elements over the last "
                     "%d steps", 100.0 * self._share_sum / self._share_steps, self._share_steps)
            self._share_sum = 0.0
            self._share_steps = 0

    # ------------------------------------------------------------------ optimizer API

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:  # the closure makes the gradients: before the preconditioner, as in the parent
            with torch.enable_grad():
                loss = closure()
        self.precondition()
        self._preconditioned = False
        super().step()
        if self.cautious:
            self._log_share()
        return loss
