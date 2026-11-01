"""Automagic v4 optimizer: Automagic 3 with sensitivity-balanced updates and a zero-centred learning-rate
controller. Derived from ai-toolkit (https://github.com/ostris/ai-toolkit), toolkit/optimizers/automagic4.py.
Copyright (c) 2024 Ostris, LLC. MIT License.

RulzTrainer uses it with fused=False so gradient clipping and gradient accumulation still apply. The file only
depends on automagic3.py next to it, so it also works as a drop-in replacement for ai-toolkit's
toolkit/optimizers/automagic4.py (fused mode included; `step_target` is accepted and ignored there).

Departures from upstream (ai-toolkit commit 534d462, 2026-10-23), each explained in the class docstring:

1. The lr controller's vote is zero-mean on gradient noise (upstream's climbs 1.6% per step on noise, the
   runaway of ai-toolkit issues #1084 and #1091), and the lr moves at most `max_vote` per step.
2. Tensors whose gradient is exactly zero (a LoRA down while its up is still zero) do not enter the sensitivity
   reference, and the reference is one quantile over the whole optimizer, not per param group.
3. A parameter that receives its first gradient late starts at the group's adapted lr, not the start lr.
4. Execution: parameters of a group that share a shape are updated as one stacked tensor, the sign flips come
   from XORed packed planes and the counts are unpacked one bucket or slab at a time (bounded by
   `bucket_elems`), sums are uint8, the reference's weights never leave the GPU (upstream syncs every step), and a
   single tensor larger than `bucket_elems` (a full fine-tune weight) is processed in row slabs. Same math, 12x
   faster on a rank-16 LoRA, transient VRAM bounded.
"""

import logging
import math
from typing import List, Optional

import torch

from .automagic3 import Automagic3

log = logging.getLogger("rulztrainer")

VOTES = ("graded", "extremes")


class Automagic4(Automagic3):
    """
    Automagic v4: Automagic 3 with sensitivity-balanced updates.

    v3 gives every element an Adam-normalised step of about lr, so how far one step moves the OUTPUT scales
    with that weight's sensitivity. A single shared lr cannot fit a model whose tensors differ in sensitivity by
    orders of magnitude (a head reading a huge raw residual, a modulation shared by every layer): at the lr the
    bulk needs, the sensitive ones overshoot.

    v4 multiplies each element's (clipped) v3 update by

        balance = min(1, reference / sensitivity)

    sensitivity = sqrt(bias-corrected second moment). For a noise-dominated MSE loss (diffusion) gradient power
    tracks the Gauss-Newton curvature, i.e. how far the output moves per unit change of the weight. It comes from
    the same row/col (or 1-D) state v3 keeps, so it is free and per element: per column for a head whose input
    dims differ in scale, per row for a modulation. Above the reference the step becomes Newton-like (grad / v),
    which is invariant to rescaling the weights, including coupled Q/K-style rescalings.

    One-sided on purpose: low gradient power cannot be told apart from a weight that is gated off (a LoRA down
    while its zero-initialised up is still ~0), where Newton would take an unbounded step. Those keep v3's
    normalised step, which is what gets zero-initialised LoRA pairs moving.

    reference: the ``sensitivity_quantile`` (default 0.9) quantile of the per-tensor geometric-mean sensitivity,
    weighted by element count, from the previous step, over every tensor of the optimizer that had a non-zero
    gradient. Tensors with an all-zero gradient are left out: upstream counted them at sensitivity ~1e-15, so a
    group whose dead LoRA downs were 90% of its elements got a 1e-15 reference and every live tensor's balance
    went to ~0 (training stalled; narrow lora_targets such as only ffn_down reach that). One reference for the
    whole optimizer rather than one per param group: with ``automagic_groups: by_type`` the timestep embedders
    and modulation projections sit in a small group of their own, where a per-group reference would be their
    own sensitivity and the balance would do nothing for exactly the tensors it is meant for.

    The first step only measures (second moments and the reference): without a reference the sensitive tensors
    would take a full v3 step.

    lr controller
    -------------
    Per element the last H update signs are kept (1 bit each, as in v3). Let k be the number of positive signs
    in the window, m = (2k - H) / H the mean sign and f = flips / (H - 1) the sign-flip rate. The default
    ``vote="graded"`` is

        vote = (m^2 - 1/H) - (f - 1/2)

    pooled over the group weighted by the balanced |update|, and lr *= exp(clamp(vote, -max_vote, max_vote)).
    On independent fair signs E[m^2] = 1/H and E[f] = 1/2 exactly, so gradient noise votes 0: the lr holds. A
    weight on a trend (consistent signs) has m^2 above 1/H and few flips: it votes up, by about 0.9 r^2 per step
    for a per-step signal-to-noise ratio r. A weight overshooting its minimum flips sign more than half the time
    (lag-1 gradient correlation -lr * curvature / 2): it votes down. The lr therefore has a fixed point set by
    the gradients' signal-to-noise ratio, climbs while the model holds a trajectory, anneals as the signal is
    consumed and walks a too-high start down. ``vote="extremes"`` is v3's rule (all signs agree: +1, perfect
    alternation: -1, anything else 0), also zero-mean, with about a third of the gain.

    Upstream v4 voted (neutral - f) with neutral = 0.5 + asin(step_target / 2) / pi = 0.516: the flip rate of a
    single weight bouncing at its own minimum with effective step 0.1. That reference is above 0.5, so every
    element whose gradient is noise (a flat direction: most of a diffusion LoRA, measured flip rate 0.50 on
    Krea 2) voted +0.016 per step at any lr, and the lr doubled every ~44 steps until the stiffest directions
    blew up (issue #1091: 1e-4 to 5.7e-3 by step 422, loss doubled; #1084: Krea 2 to noise after ~400 steps).

    ``max_vote`` (default 0.05, about 5% per step) bounds the lr's move per step: the lr set at step t is judged
    by signs that are only fully refreshed H - 1 steps later, so with v3's +/-1 clamp a unanimous pool could
    carry the lr e^8 = 3000x past the stability edge before alternation answers; at 0.05 the lag overshoot is
    under 1.5x. The rail never engaged in the simulations except in the first steps of a noise-free problem.

    Parameters (beyond Automagic3's)
    --------------------------------
    sensitivity_quantile : float in (0, 1]
        Numel-weighted quantile of per-tensor sensitivity that caps each element's step (balance 1 below it).
    vote : "graded" | "extremes"
        The lr vote rule (above).
    max_vote : float > 0
        Largest lr change per step in ln units (0.05 = x1.051).
    bucket_elems : int
        Elements processed per stacked bucket / per row slab (bounds the transient VRAM: about 20 bytes per
        element of the bucket); 0 = unlimited.
    bucketing : bool
        False updates every tensor on its own (for tests; same results).
    step_target : ignored
        Accepted so ai-toolkit job configs written for upstream v4 still load.
    """

    def __init__(
        self,
        params,
        *args,
        sensitivity_quantile: float = 0.9,
        vote: str = "graded",
        max_vote: float = 0.05,
        bucket_elems: int = 4_000_000,
        bucketing: bool = True,
        step_target: Optional[float] = None,
        **kwargs,
    ):
        if not 0.0 < sensitivity_quantile <= 1.0:
            raise ValueError(f"sensitivity_quantile ({sensitivity_quantile}) must be in (0, 1]")
        if vote not in VOTES:
            raise ValueError(f"vote must be one of {VOTES}, got {vote!r}")
        if not max_vote > 0.0:
            raise ValueError(f"max_vote ({max_vote}) must be above 0")
        self.sensitivity_quantile = float(sensitivity_quantile)
        self.vote = vote
        self.max_vote = float(max_vote)
        self.bucket_elems = max(0, int(bucket_elems))
        self.bucketing = bool(bucketing)
        if step_target is not None:
            log.info("Automagic4: step_target %s is ignored (the vote is zero-centred; there is no target flip rate)",
                     step_target)
        # A multiplier on this step's update (not on the weight decay), set per step by a subclass (Keel's noise-band
        # weight) and None otherwise; applied after the vote, which a positive gain leaves unchanged.
        self.update_gain: Optional[float] = None
        super().__init__(params, *args, **kwargs)

    # ------------------------------------------------------------------ bookkeeping

    def _rebuild_group_index(self) -> None:
        super()._rebuild_group_index()
        n = len(self.param_groups)
        # The group lr this step works with: the first param's mirror (every mirror of a group holds the same
        # value; they are re-read every step so anything that rewrote them -- the likeness controller, a race
        # moving the state, a relay copying a winner's state -- is seen).
        self._group_lr: List[Optional[torch.Tensor]] = [None] * n
        # One sensitivity reference for the whole optimizer (0-dim fp32), from the previous step's quantile.
        self._ref: Optional[torch.Tensor] = None
        # This step's per-tensor log sensitivities and their quantile weights (numel x alive), [n] tensors.
        self._sens_log: list = []
        self._sens_weight: list = []

    def _refresh_group_lrs(self) -> None:
        # Re-read the group lrs and the reference from their per-param mirrors: anything that rewrote the state
        # since the last step (the likeness controller, a race parking or moving it, a relay copying a winner's
        # state) is seen, and a state without a reference (Automagic 3's) gets a measurement-only step. With no
        # state at all (a fresh optimizer, a relay that started it over) a reference set by inherit_reference
        # is kept.
        ref = None
        seen = False
        for gi, group in enumerate(self.param_groups):
            cur = None
            for p in group["params"]:
                st = self.state.get(p)
                if not st:
                    continue
                seen = True
                if cur is None and "lr" in st:
                    cur = st["lr"]
                if ref is None and isinstance(st.get("sens_ref"), torch.Tensor):
                    ref = st["sens_ref"]
                if cur is not None and ref is not None:
                    break
            if cur is not None and cur is not self._group_lr[gi]:
                self._unify_lrs(group, cur)
            self._group_lr[gi] = cur
        if seen:
            self._ref = ref

    def _unify_lrs(self, group: dict, first: torch.Tensor) -> None:
        # The mirror tensors were replaced since the last step (a load, a race move, a relay): make the group's
        # mirrors one value again, their median, as load_state_dict does. They are already equal except after a
        # relay from a by_type winner into a single-group follower, which carries several rates into one group;
        # without this the first param's rate would win by parameter order.
        mirrors = [st["lr"] for p in group["params"] if (st := self.state.get(p)) and "lr" in st]
        if len(mirrors) < 2:
            return
        dev = first.device
        med = torch.stack([t if t.device == dev else t.to(dev) for t in mirrors]).median()
        by_dev: dict = {}
        for t in mirrors:
            by_dev.setdefault(t.device, []).append(t)
        for d, ts in by_dev.items():
            torch._foreach_copy_(ts, [med if med.device == d else med.to(d)] * len(ts))

    def inherit_reference(self, other) -> None:
        """Start from another Automagic 4's sensitivity reference (a fine-tune window taking over from the last)
        instead of a measurement-only first step. Safe because the balance only ever shrinks a step; the
        reference is measured again on this optimizer's first step."""
        ref = getattr(other, "_ref", None)
        if ref is not None and not self.state:
            self._ref = ref.detach().to(torch.float32, copy=True)

    def add_param_group(self, param_group: dict) -> None:
        super().add_param_group(param_group)
        if hasattr(self, "_group_lr"):  # after construction: size the per-group lists for the new group
            ref = self._ref
            self._rebuild_group_index()
            self._ref = ref  # an inherited reference survives (the refresh re-reads it when there is state)
            self._refresh_group_lrs()

    def _init_state(self, p: torch.Tensor, group: dict) -> None:
        super()._init_state(p, group)
        st = self.state[p]
        gi = self._param_group_index.get(p)
        cur = self._group_lr[gi] if gi is not None else None
        if cur is not None:
            # A parameter joining late (a relay follower training layers the winner did not, a resume before
            # every layer had a gradient) starts at the group's adapted lr, not the configured start lr; the
            # parent seeds group["lr"], which never moves, and the two would then stay a fixed ratio apart.
            st["lr"] = cur.detach().to(p.device, torch.float32, copy=True).clamp_(min=group["min_lr"],
                                                                                max=group["max_lr"])
        if self._ref is not None:
            st["sens_ref"] = self._ref.to(p.device, copy=True)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _stack_f32(tensors: List[torch.Tensor]) -> torch.Tensor:
        # [N, *shape] fp32 working copy of per-param state; a view when there is one fp32 tensor (no copy).
        if len(tensors) == 1 and tensors[0].dtype == torch.float32:
            return tensors[0].unsqueeze(0)
        out = torch.stack(tensors)
        return out if out.dtype == torch.float32 else out.to(torch.float32)

    @staticmethod
    def _write_back(tensors: List[torch.Tensor], stacked: torch.Tensor) -> None:
        if len(tensors) == 1 and tensors[0].dtype == torch.float32:
            return  # the stack is a view of the state tensor
        torch._foreach_copy_(tensors, list(stacked.unbind(0)))

    @classmethod
    def _pack_rows(cls, bits: torch.Tensor, width: int) -> torch.Tensor:
        # bool [N, numel] -> uint8 [N, width], 8 sign bits per byte (bits past numel are 0).
        weights, _ = cls._pack_consts(bits.device)
        n, numel = bits.shape
        if width * 8 != numel:
            bits = torch.nn.functional.pad(bits, (0, width * 8 - numel))
        return (bits.view(n, width, 8) * weights).sum(-1, dtype=torch.uint8)

    @classmethod
    def _window_counts(cls, hist: torch.Tensor, idx: int, numel: int):
        """Per element over the H-plane window: (positive signs, sign flips), uint8 [N, numel].

        hist: [N, H, W] packed planes, a ring buffer whose oldest plane is `idx`. The flips are the XOR of
        chronological neighbours, which are the physical neighbour pairs (j, j+1) except that the pair
        (idx-1, idx) = (newest, oldest) is replaced by the wrap pair (H-1, 0); no torch.roll of the whole window.
        """
        _, shifts = cls._pack_consts(hist.device)
        n, H, _ = hist.shape
        ones = (hist.unsqueeze(-1) >> shifts).bitwise_and_(1).view(n, H, -1)[:, :, :numel].sum(1, dtype=torch.uint8)
        x = hist[:, 1:] ^ hist[:, :-1]
        if idx != 0:
            x[:, idx - 1] = hist[:, H - 1] ^ hist[:, 0]
        flips = (x.unsqueeze(-1) >> shifts).bitwise_and_(1).view(n, H - 1, -1)[:, :, :numel].sum(1, dtype=torch.uint8)
        return ones, flips

    def _vote_sums(self, w: torch.Tensor, ones: torch.Tensor, flips: torch.Tensor, H: int):
        """(weighted vote mass, total weight) of a flat |update| w and its window counts (see the docstring)."""
        den = w.sum()
        if self.vote == "extremes":
            up = ones.eq(H).logical_or_(ones.eq(0))
            down = flips.eq(H - 1)
            return (w * up).sum().sub_((w * down).sum()), den
        kf = ones.to(w.dtype)
        wk = (w * kf).sum()
        wk2 = (w * kf.square_()).sum()
        wf = (w * flips).sum()
        # sum w * [(2k - H)^2 / H^2 - 1/H - flips / (H - 1) + 1/2]
        num = wk2.mul_(4.0 / (H * H)).sub_(wk.mul_(4.0 / H)).add_(den * (1.5 - 1.0 / H)).sub_(wf.div_(H - 1))
        return num, den

    def _accumulate_vote(self, gi: int, num: torch.Tensor, den: torch.Tensor) -> None:
        if self._group_num[gi] is None:
            self._group_num[gi] = num
            self._group_den[gi] = den
        else:
            acc = self._group_num[gi]
            if num.device != acc.device:
                num = num.to(acc.device)
                den = den.to(acc.device)
            acc.add_(num)
            self._group_den[gi].add_(den)

    # ------------------------------------------------------------------ the update

    @torch.no_grad()
    def _update_param(self, p: torch.Tensor, group: dict) -> None:
        # Fused mode (ai-toolkit) and the per-tensor fallback: one tensor is a bucket of one.
        if p.grad is None:
            return
        gi = self._param_group_index.get(p)
        if gi is None:
            self._rebuild_group_index()
            gi = self._param_group_index[p]
        if self._group_lr[gi] is None:
            self._refresh_group_lrs()
        if len(self.state[p]) == 0:
            self._init_state(p, group)
        if self._group_lr[gi] is None:
            self._group_lr[gi] = self.state[p]["lr"]
        self._update_bucket(gi, group, [p])

    @torch.no_grad()
    def _update_bucket(self, gi: int, group: dict, params: List[torch.Tensor]) -> None:
        """Update N parameters of one shape (and one step / history position) as a stacked [N, *shape] tensor."""
        p0 = params[0]
        n = len(params)
        shape = tuple(p0.shape)
        numel = p0.numel()
        if n == 1 and p0.dim() == 2 and self.bucket_elems and numel > self.bucket_elems and shape[1] % 8 == 0:
            self._update_large(gi, group, p0)
            return
        beta2, eps = group["beta2"], group["eps"]
        clip, wd = group["clip_threshold"], group["weight_decay"]
        states = [self.state[p] for p in params]
        if n == 1:
            g = p0.grad
            grad = (g if g.dtype == torch.float32 else g.to(torch.float32)).unsqueeze(0)
        else:
            grad = torch.stack([p.grad if p.grad.dtype == torch.float32 else p.grad.to(torch.float32)
                                for p in params])
            for p in params:
                p.grad = None
        grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        # Second moment (Adafactor-factored for >= 2-D), bias-corrected log sensitivity per tensor. The row/col
        # means of grad^2 come from vector norms, so grad^2 is never materialised.
        if p0.dim() >= 2:
            rows = [st["exp_avg_sq_row"] for st in states]
            cols = [st["exp_avg_sq_col"] for st in states]
            row, col = self._stack_f32(rows), self._stack_f32(cols)
            rs = torch.linalg.vector_norm(grad, dim=-1).square_().div_(shape[-1])
            cs = torch.linalg.vector_norm(grad, dim=-2).square_().div_(shape[-2])
            alive = rs.reshape(n, -1).sum(-1) > 0
            row.mul_(beta2).add_(rs.add_(eps), alpha=1.0 - beta2)
            col.mul_(beta2).add_(cs.add_(eps), alpha=1.0 - beta2)
            self._write_back(rows, row)
            self._write_back(cols, col)
            log_v = (row.log().mean(-1) - row.mean(-1).log() + col.log().mean(-1)).reshape(n, -1).mean(-1)
            v_eps = None
        else:
            vs = [st["exp_avg_sq"] for st in states]
            v = self._stack_f32(vs)
            alive = grad.reshape(n, -1).abs().sum(-1) > 0
            v.mul_(beta2).add_(grad.square(), alpha=1.0 - beta2)
            self._write_back(vs, v)
            v_eps = v.add(eps)
            log_v = v_eps.log().reshape(n, -1).mean(-1)
        step = states[0]["step"] + 1
        for st in states:
            st["step"] = step
        bias_corr = 1.0 - beta2 ** step
        self._sens_log.append(log_v.mul_(0.5).sub_(0.5 * math.log(bias_corr)))
        self._sens_weight.append(alive.to(torch.float64).mul_(float(numel)))
        if self._ref is None:
            if n == 1:  # measurement-only first step (no reference yet)
                p0.grad = None
            return

        # The v3 update (grad / sqrt(v), RMS clip, element clamp) times the balance, in place on the gradient.
        kbal = self._ref.to(grad.device) * math.sqrt(bias_corr)
        if v_eps is None:
            r = (row / row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
            c = col.unsqueeze(-2).rsqrt()
            isv = torch.mul(r, c)
        else:
            isv = v_eps.rsqrt_()
        update = grad.mul_(isv)
        norms = torch.linalg.vector_norm(update.reshape(n, -1), dim=1)
        scale = norms.div_(numel ** 0.5).div_(clip).clamp_(min=1.0)
        update.div_(scale.view((n,) + (1,) * len(shape)))
        update.clamp_(-clip, clip)
        balance = isv.mul_(kbal).clamp_(max=1.0)
        update.mul_(balance)
        del balance, isv

        # Sign history (packed planes, stacked [N, H, W]) and the pooled vote.
        hists = [st["sign_history"] for st in states]
        H, width = hists[0].shape
        hist = hists[0].unsqueeze(0) if n == 1 else torch.stack(hists)
        idx = states[0]["hist_idx"]
        hist[:, idx].copy_(self._pack_rows(update.gt(0.0).reshape(n, -1), width))
        idx = (idx + 1) % H
        fill = min(H, states[0]["hist_fill"] + 1)
        for st in states:
            st["hist_idx"] = idx
            st["hist_fill"] = fill
        if fill == H:
            ones, flips = self._window_counts(hist, idx, numel)
            num, den = self._vote_sums(update.abs().reshape(n, -1), ones, flips, H)
            del ones, flips
            self._accumulate_vote(gi, num, den)
        if n > 1:
            torch._foreach_copy_(hists, list(hist.unbind(0)))
        del hist
        if self.update_gain is not None:
            update.mul_(self.update_gain)

        # p -= lr * (update + wd * p), decoupled weight decay as upstream.
        lr_t = self._group_lr[gi]
        if lr_t.device != grad.device:
            lr_t = lr_t.to(grad.device)
        if p0.dtype == torch.float32:
            if n == 1:
                u = update[0]
                if wd != 0.0:
                    u.add_(p0, alpha=wd)
                p0.addcmul_(u, lr_t, value=-1.0)
            else:
                ups = list(update.unbind(0))
                if wd != 0.0:
                    torch._foreach_add_(ups, params, alpha=wd)
                update.mul_(lr_t.neg())
                torch._foreach_add_(params, ups)
        else:
            for i, p in enumerate(params):
                new_p = p.to(torch.float32)
                u = update[i]
                if wd != 0.0:
                    u.add_(new_p, alpha=wd)
                new_p.addcmul_(u, lr_t, value=-1.0)
                self._stochastic_copy_(p, new_p)
        if n == 1:
            p0.grad = None

    @torch.no_grad()
    def _update_large(self, gi: int, group: dict, p: torch.Tensor) -> None:
        """One 2-D tensor larger than bucket_elems (a full fine-tune weight), in row slabs: the same math with a
        transient of about 20 bytes per element of a slab instead of 33 per element of the tensor."""
        st = self.state[p]
        beta2, eps = group["beta2"], group["eps"]
        clip, wd = group["clip_threshold"], group["weight_decay"]
        R, C = p.shape
        numel = R * C
        g = p.grad
        grad = g if g.dtype == torch.float32 else g.to(torch.float32)
        grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        rows, cols = [st["exp_avg_sq_row"]], [st["exp_avg_sq_col"]]
        row, col = self._stack_f32(rows)[0], self._stack_f32(cols)[0]
        slab = max(1, self.bucket_elems // C)
        rs = torch.linalg.vector_norm(grad, dim=1).square_().div_(C)
        cs = torch.zeros(C, dtype=torch.float32, device=grad.device)  # per slab: the whole-tensor column norm
        for i0 in range(0, R, slab):  # needs a staging buffer of 0.5 B per element of the tensor
            cs += torch.linalg.vector_norm(grad[i0:i0 + slab], dim=0).square_()
        cs.div_(R)
        alive = rs.sum() > 0
        row.mul_(beta2).add_(rs.add_(eps), alpha=1.0 - beta2)
        col.mul_(beta2).add_(cs.add_(eps), alpha=1.0 - beta2)
        self._write_back(rows, row.unsqueeze(0))
        self._write_back(cols, col.unsqueeze(0))
        log_v = row.log().mean() - row.mean().log() + col.log().mean()
        st["step"] += 1
        bias_corr = 1.0 - beta2 ** st["step"]
        self._sens_log.append(log_v.mul_(0.5).sub_(0.5 * math.log(bias_corr)).reshape(1))
        self._sens_weight.append(alive.to(torch.float64).mul_(float(numel)).reshape(1))
        if self._ref is None:
            p.grad = None
            return
        kbal = self._ref.to(grad.device) * math.sqrt(bias_corr)
        r = (row / row.mean()).rsqrt_()
        c = col.rsqrt()
        # pass 1: the update's RMS over the whole tensor (the trust-region clip is per tensor)
        ss = torch.zeros((), dtype=torch.float32, device=grad.device)
        for i0 in range(0, R, slab):
            t = grad[i0:i0 + slab] * r[i0:i0 + slab, None]
            ss += t.mul_(c).square_().sum()
        scale = ss.sqrt_().div_(numel ** 0.5).div_(clip).clamp_(min=1.0)
        # pass 2: clipped, balanced update per slab; its sign plane, vote and parameter update
        hist = st["sign_history"]
        H = hist.shape[0]
        idx_old = st["hist_idx"]
        idx = (idx_old + 1) % H
        fill = min(H, st["hist_fill"] + 1)
        st["hist_idx"], st["hist_fill"] = idx, fill
        lr_t = self._group_lr[gi]
        if lr_t.device != grad.device:
            lr_t = lr_t.to(grad.device)
        num = den = None
        for i0 in range(0, R, slab):
            i1 = min(R, i0 + slab)
            u = grad[i0:i1]
            u.mul_(r[i0:i1, None]).mul_(c).div_(scale).clamp_(-clip, clip)
            balance = torch.mul(r[i0:i1, None] * kbal, c).clamp_(max=1.0)
            u.mul_(balance)
            del balance
            w0, w1 = i0 * C // 8, i1 * C // 8
            n_el = (i1 - i0) * C
            hist[idx_old, w0:w1] = self._pack_rows(u.gt(0.0).reshape(1, -1), w1 - w0)[0]
            if fill == H:
                ones, flips = self._window_counts(hist[:, w0:w1].unsqueeze(0), idx, n_el)
                nm, dn = self._vote_sums(u.abs().reshape(1, -1), ones, flips, H)
                del ones, flips
                if num is None:
                    num, den = nm, dn
                else:
                    num.add_(nm)
                    den.add_(dn)
            if self.update_gain is not None:
                u.mul_(self.update_gain)
            if p.dtype == torch.float32:
                if wd != 0.0:
                    u.add_(p[i0:i1], alpha=wd)
                p[i0:i1].addcmul_(u, lr_t, value=-1.0)
            else:
                new_p = p[i0:i1].to(torch.float32)
                if wd != 0.0:
                    u.add_(new_p, alpha=wd)
                new_p.addcmul_(u, lr_t, value=-1.0)
                self._stochastic_copy_(p[i0:i1], new_p)
        if num is not None:
            self._accumulate_vote(gi, num, den)
        p.grad = None

    # ------------------------------------------------------------------ optimizer API

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self.fused:
            self._refresh_group_lrs()
            for gi, group in enumerate(self.param_groups):
                buckets: dict = {}
                order: list = []
                for p in group["params"]:
                    if not p.requires_grad:
                        continue
                    accum = getattr(p, "_accum_grad", None)
                    if accum is not None:
                        p.grad = accum
                        del p._accum_grad
                    if p.grad is None:
                        continue
                    if p.numel() == 0:  # nothing to update (and no bucket size to divide by)
                        p.grad = None
                        continue
                    st = self.state[p]
                    if len(st) == 0:
                        self._init_state(p, group)
                        if self._group_lr[gi] is None:
                            self._group_lr[gi] = st["lr"]
                    key = ((tuple(p.shape), p.dtype, p.device, st["step"], st["hist_idx"], st["hist_fill"])
                           if self.bucketing else (id(p),))
                    if key not in buckets:
                        buckets[key] = []
                        order.append(key)
                    buckets[key].append(p)
                for key in order:
                    ps = buckets[key]
                    per = len(ps)
                    if self.bucket_elems and ps[0].numel() < self.bucket_elems:
                        per = max(1, self.bucket_elems // ps[0].numel())
                    elif self.bucket_elems:
                        per = 1
                    for k in range(0, len(ps), per):
                        self._update_bucket(gi, group, ps[k:k + per])
        self._apply_group_votes()
        self._refresh_group_lrs()
        return loss

    def _apply_group_votes(self) -> None:
        self._update_reference()
        for gi, group in enumerate(self.param_groups):
            num = self._group_num[gi]
            if num is None:
                continue
            den = self._group_den[gi]
            self._group_num[gi] = None
            self._group_den[gi] = None
            signal = num.div_(den.clamp_(min=1e-30)).clamp_(-self.max_vote, self.max_vote)
            lr_t = self._group_lr[gi]
            if lr_t is None:
                continue
            new = lr_t.to(signal.device) * signal.exp_()
            new.clamp_(min=group["min_lr"], max=group["max_lr"])
            by_dev: dict = {}
            for p in group["params"]:
                st = self.state.get(p)
                if st is not None and "lr" in st:
                    by_dev.setdefault(st["lr"].device, []).append(st["lr"])
            for dev, mirrors in by_dev.items():
                src = new if new.device == dev else new.to(dev)
                torch._foreach_copy_(mirrors, [src] * len(mirrors))

    def _update_reference(self) -> None:
        # Numel-weighted quantile of this step's per-tensor log sensitivities over the tensors that had a
        # gradient; stays on the device (no sync). A step without any live tensor keeps the previous reference.
        if not self._sens_log:
            return
        dev = self._sens_log[0].device
        vals = torch.cat([t if t.device == dev else t.to(dev) for t in self._sens_log])
        weights = torch.cat([w if w.device == dev else w.to(dev) for w in self._sens_weight])
        self._sens_log = []
        self._sens_weight = []
        order = vals.argsort()
        cum = weights[order].cumsum(0)
        total = cum[-1:]
        pos = torch.searchsorted(cum, total * self.sensitivity_quantile).clamp_(max=vals.numel() - 1)
        new = vals[order][pos].squeeze(0).exp_()
        if self._ref is None:
            if not bool(total.squeeze(0) > 0):  # nothing alive yet: measure again (a sync only before a reference)
                return
        else:
            new = torch.where(total.squeeze(0) > 0, new, self._ref.to(dev))
        self._ref = new
        by_dev: dict = {}
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if not st:
                    continue
                cur = st.get("sens_ref")
                if cur is None:
                    st["sens_ref"] = new.to(p.device, copy=True)
                else:
                    by_dev.setdefault(cur.device, []).append(cur)
        for d, mirrors in by_dev.items():
            src = new if new.device == d else new.to(d)
            torch._foreach_copy_(mirrors, [src] * len(mirrors))

    def get_learning_rates(self) -> List[float]:
        self._refresh_group_lrs()  # the mirrors may have been replaced since the last step (a race relay)
        return [float(t) if t is not None else float(group["lr"])
                for t, group in zip(self._group_lr, self.param_groups)]

    def load_state_dict(self, state_dict):
        exact = self._saved_fp32(state_dict, "sens_ref")
        super().load_state_dict(state_dict)
        # The parent cast the reference to the param dtype: restore the saved fp32 value (a state saved by
        # Automagic 3 has none: the next step only measures).
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                cur = st.get("sens_ref") if st is not None else None
                if isinstance(cur, torch.Tensor):
                    st["sens_ref"] = exact[p] if p in exact else cur.to(torch.float32)
        self._refresh_group_lrs()
