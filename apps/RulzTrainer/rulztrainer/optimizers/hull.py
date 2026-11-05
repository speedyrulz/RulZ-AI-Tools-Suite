"""Hull: a body-identity optimizer for character LoRAs (docs/hull-optimizer.md). This file is its base rule H0 (section 4)
and the optimizer's side of the region gains (C1 and C3, sections 5.1 and 5.3): written from scratch, not a subclass of Automagic 4, and not self-tuning: `group["lr"]` is the step rho_t from the
trainer's scheduler. H0 reads `p.grad` in `step()` with no parameter hooks and no gather, so it runs unchanged in the
parallel race and with replica reduces, and nothing keeps a dropped optimizer alive.

Per LoRA pair (B = lora_up [o, r], A = lora_down [r, i], dW = s B A, s = alpha / rank):

    T     T M T = N for M = BtB + delta I, N = AAt + delta I      the virtual balanced frame; stored factors never re-gauged
    v     <- per-channel EMA of the raw gradient's square in that frame;  D = v^-1/2
    g^    = D G, per-channel clip;  m <- b1 m + (1 - b1) g^       normalise first, then average
    U     = (m^_B T^-1, T m^_A)                                   the balanced update, mapped back
    u     <- EMA of ||dW image of the raw normalised gradient||^2
    eta   = rho_t ||W0||_F / sqrt(u^)                             the step in dW units, relative to the frozen weight
    B, A -= eta U                                                 under the factor and safety caps (eta <= both)

At step 1 B = 0 and no frame exists: B alone moves, in the gauge-free direction (the row-normalised G P_A), by exactly
rho_t ||W0||, and the moments start at step 2.

Every REBALANCE_EVERY steps the stored factors are rebalanced in place: B <- B X, A <- X^-1 A with X = T^1/2, dW
unchanged. The virtual frame keeps dW's path independent of the gauge but has no restoring force on the gauge itself:
in the first real run (Krea 2, rank 32, 2000 steps) ||A|| reached 1e5 against ||B|| 0.04, so a layer's dW was 1000 times
smaller than ||B|| ||A|| and rested on cancellation that bf16 (the saved LoRA, the forward's activations) cannot hold.
The momenta and the lora_ema shadow move with the factors (bind_ema); v and u do not depend on the gauge.

The rebalance needs fp32 momentum, which is why the trainer always builds Hull with it. X can be ill-conditioned, so bf16
momentum's rounding (0.4% of an element) is multiplied by it at every rebalance: ||m_A|| grew 1000-fold in 30 steps and
the step sat on the safety cap at 4 rho, along that noise.

Departure from the design (its section 4.6 runs the roots in fp32): the Grams and both roots are fp64. A rank-4 update
held at rank 32 has Gram eigenvalues under the damping, fp32 rounding makes them negative, and the iteration diverges to
NaN or 1e8; fp64 holds a residual of 1e-9 at the same damping. Everything after the roots stays fp32 (T included).

No host sync in `step()` outside a rebalance: the roots are inverse-free Newton-Schulz iterations and the selects are
device-side, so a NaN gradient, a failed root and the first-step branch all cost nothing on the host. A rebalance's
eigendecompositions do sync, once in REBALANCE_EVERY steps.
"""

import logging
import math

import torch

log = logging.getLogger("rulztrainer")

STATE_DTYPES = ("bf16", "fp32")
FRAME_TOLERANCE = 1e-3  # ||T M T - N|| / ||N|| above this keeps the previous step's T
CAP_REPORT_AFTER = 20  # the safety cap's bind rate counts from this step on (the M1a gate)
REBALANCE_EVERY = 10  # steps between in-place rebalances; dW's path does not depend on it, and a pair drifts little in 10
COSINE_DECAY = 0.9  # the smoothed gained-to-raw cosine the retreat reads, per armed step
REGION_STATS = ("cos", "cos_top", "armed", "interval")  # per pair, on lora_up: the region gains' records
STATS = ("u", "k", "cap_safety", "cap_factor", "cap_steps", "frame_fallback", *REGION_STATS)


def _sym(a: torch.Tensor) -> torch.Tensor:
    return 0.5 * (a + a.mT)


def _trace(a: torch.Tensor) -> torch.Tensor:
    return a.diagonal(dim1=-2, dim2=-1).sum(-1)


def _tr_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """tr(a bt) over a batch of matrices: tr(a b) when either is symmetric."""
    return (a * b).sum((-2, -1))


def newton_schulz(a: torch.Tensor, iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(a^1/2, a^-1/2) of a batch of SPD matrices with eigenvalues in (0, 1], by the coupled inverse-free iteration
    (Higham, Functions of Matrices, ch. 6): Y -> Y C / 2, Z -> C Z / 2, C = 3 - Z Y, from Y = a, Z = I."""
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    y, z = a, eye.expand_as(a)
    for _ in range(iters):
        c = 3.0 * eye - z @ y
        y, z = _sym(y @ c) * 0.5, _sym(c @ z) * 0.5
    return y, z


def balance_frame(m: torch.Tensor, n: torch.Tensor, iters: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(T, T^-1, good) for batches of SPD Grams: T M T = N, the geometric mean of M^-1 and N, T = M^-1/2 (M^1/2 N M^1/2)^1/2
    M^-1/2 with two coupled roots. M and N work at unit trace, since the factors' scales can sit decades apart. `good`:
    the residual is within FRAME_TOLERANCE (false for an all-zero Gram and for anything not finite)."""
    r = m.shape[-1]
    eye = torch.eye(r, dtype=m.dtype, device=m.device)
    a, b = _trace(m), _trace(n)
    usable = (a > 0) & (b > 0) & torch.isfinite(a) & torch.isfinite(b)
    flag = usable[:, None, None]
    ms = torch.where(flag, m / a[:, None, None].where(usable[:, None, None], 1.0), eye / r)
    ns = torch.where(flag, n / b[:, None, None].where(usable[:, None, None], 1.0), eye / r)
    mh, mi = newton_schulz(ms, iters)
    p = _sym(mh @ ns @ mh)
    pt = _trace(p)[:, None, None]
    ph, pi = newton_schulz(p / pt, iters)
    root = pt.sqrt()
    t, ti = _sym(mi @ (ph * root) @ mi), _sym(mh @ (pi / root) @ mh)
    residual = (t @ ms @ t - ns).flatten(1).norm(dim=1) / ns.flatten(1).norm(dim=1)
    scale = (b / a.where(usable, 1.0)).sqrt()[:, None, None]
    return t * scale, ti / scale, usable & (residual <= FRAME_TOLERANCE)


class Hull(torch.optim.Optimizer):
    """H0. `params`: exactly the tensors of `pairs`, in one group (zero-element tensors are skipped); `pairs`:
    (lora_down, lora_up) per layer (lora.lora_pairs); `ref_norms`: ||W0||_F of each layer's frozen weight; `scales`:
    alpha / rank of each layer. `lr` is rho_0, a relative change of a layer's weight per step.

    The options are attributes, not group defaults, so a loaded state never overrides them and only `lr` and
    `initial_lr` come back from it. State is flat, per parameter and re-read every step, for the race's park, move and
    relay; the frame is recomputed from the factors, the stored T only seeds the fallback.
    """

    def __init__(self, params, pairs, ref_norms, scales, lr: float, betas=(0.9, 0.99), state_dtype: str = "fp32",
                 clip: float = 5.0, factor_cap: float = 0.2, safety_cap: float = 0.0, gain_max: float = 2.0,
                 ns_iters: int = 30, damping: float = 1e-6, seed: int = 0, face_safe: bool = True,
                 retreat: float = 0.5):
        if not lr > 0:
            raise ValueError(f"Hull: lr (hull_rho) must be above 0, got {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 < betas[1] < 1.0:
            raise ValueError(f"Hull: betas must be in [0, 1) and (0, 1), got {betas}")
        if state_dtype not in STATE_DTYPES:
            raise ValueError(f"Hull: state_dtype must be one of {STATE_DTYPES}, got {state_dtype!r}")
        if not clip > 0 or not factor_cap > 0 or safety_cap < 0 or not gain_max >= 1:
            raise ValueError("Hull: clip and factor_cap must be above 0, safety_cap at least 0, gain_max at least 1")
        if ns_iters < 1 or not damping > 0:
            raise ValueError("Hull: ns_iters must be at least 1 and damping above 0")
        super().__init__(params, dict(lr=lr))
        if len(self.param_groups) != 1:
            raise ValueError("Hull takes one parameter group (the trainer collapses LoRA+ groups)")
        pairs = [(a, b) for a, b in pairs]
        if len(pairs) != len(ref_norms) or len(pairs) != len(scales):
            raise ValueError(f"Hull: {len(pairs)} pairs, {len(ref_norms)} frozen-weight norms, {len(scales)} scales")
        given = {id(p) for p in self.param_groups[0]["params"] if p.numel() > 0}
        paired = {id(p) for pair in pairs for p in pair}
        if given != paired:
            raise ValueError("Hull trains LoRA pairs only: the parameters must be exactly the pairs' tensors (LoKr, "
                             "OFT and DoRA are refused in the config)")
        self.pairs = pairs
        self.ref_norms = [float(x) for x in ref_norms]
        if not all(math.isfinite(x) and x > 0 for x in self.ref_norms):
            raise ValueError("Hull: every frozen weight must have a finite norm above 0")
        self.scales = [float(x) for x in scales]
        self.b1, self.b2 = float(betas[0]), float(betas[1])
        self.state_dtype = state_dtype
        self.clip = float(clip)
        self.factor_cap = float(factor_cap)
        self.safety_cap = float(safety_cap) if safety_cap > 0 else 2.0 * max(1.0, float(gain_max))
        self.ns_iters = int(ns_iters)
        self.damping = float(damping)
        self.seed = int(seed)
        self.face_safe = bool(face_safe)
        self.retreat = float(retreat)
        self._excess: dict = {}  # id(lora_up) -> this step's region excess: a [x_a, x_b, f_a, f_b] list (add_excess)
        self._gens: dict = {}  # device -> a private generator, reseeded each step: the dither leaves the run's RNG alone
        self._consts: dict = {}  # (device, dtype, pair indices) -> (scales, ref norms)
        self._ema = None  # the LoraEMA whose shadow a rebalance moves with the factors
        self._ema_slots: dict = {}  # id(parameter) -> its index in the shadow

    def bind_ema(self, ema) -> None:
        """Keep this weight average's shadow in the factors' gauge: a rebalance transports it, so its product stays."""
        self._ema = ema
        self._ema_slots = {id(p): i for i, p in enumerate(ema.params)} if ema is not None else {}

    # ------------------------------------------------------------------ state

    def _init_pair(self, a: torch.Tensor, b: torch.Tensor) -> None:
        for p, channels in ((a, a.shape[1]), (b, b.shape[0])):
            st = self.state[p]
            st["m"] = torch.zeros_like(p, dtype=self._moment_dtype(p))
            st["v"] = torch.zeros(channels, dtype=p.dtype, device=p.device)
        st = self.state[b]
        r = b.shape[1]
        st["T"] = torch.eye(r, dtype=b.dtype, device=b.device)
        st["Ti"] = st["T"].clone()
        for key in STATS:
            st[key] = torch.zeros((), dtype=b.dtype, device=b.device)
        st["lam"] = torch.ones((), dtype=b.dtype, device=b.device)

    def _moment_dtype(self, p: torch.Tensor) -> torch.dtype:
        return torch.bfloat16 if self.state_dtype == "bf16" and p.dtype == torch.float32 else p.dtype

    def load_state_dict(self, state_dict) -> None:
        super().load_state_dict(state_dict)  # torch casts state to the parameter's dtype and moves it to its device
        for p, st in self.state.items():
            if "m" in st:
                st["m"] = st["m"].to(self._moment_dtype(p))
            if "T" in st:  # a state from before the region gains
                for key in REGION_STATS:
                    st.setdefault(key, torch.zeros_like(st["u"]))
                st.setdefault("lam", torch.ones_like(st["u"]))

    def report(self) -> dict:
        """The caps' bind rates and the frame fallback count, for the log (a host sync: call it between steps). The
        safety rate counts layer-steps after step CAP_REPORT_AFTER (design section 4.4 fails H0 above 2%), the factor
        rate all of them."""
        totals = {key: 0.0 for key in ("k", "cap_safety", "cap_factor", "cap_steps", "frame_fallback")}
        for _, b in self.pairs:
            st = self.state.get(b)
            if st and "k" in st:
                for key in totals:
                    totals[key] += float(st[key])
        late = max(totals["cap_steps"], 1.0)
        out = {"safety_rate": totals["cap_safety"] / late, "factor_rate": totals["cap_factor"] / max(totals["k"], 1.0),
               "frame_fallbacks": int(totals["frame_fallback"]), "layer_steps": int(totals["k"])}
        armed = [self.state[b] for _, b in self.pairs if float(self.state.get(b, {}).get("armed", 0.0)) > 0]
        if armed:
            out["armed_share"] = sum(float(st["armed"]) for st in armed) / max(totals["k"], 1.0)
            out["gain_cosine"] = sum(float(st["cos"]) for st in armed) / len(armed)
            out["excess_scale"] = min(float(st["lam"]) for st in armed)
        return out

    # ------------------------------------------------------------------ region gains

    def add_excess(self, pa, pb, x_a, x_b, f_a, f_b) -> None:
        """A LoRA backward's region excess and face part (lora.HullBackward), summed over a step's micro-batches."""
        buf = self._excess.get(id(pb))
        new = [x_a, x_b, f_a, f_b]
        if buf is None:
            self._excess[id(pb)] = new
            return
        for i, value in enumerate(new):
            if value is not None:
                buf[i] = value if buf[i] is None else buf[i] + value

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._excess.clear()
        super().zero_grad(set_to_none=set_to_none)

    def on_probe(self, score, band, ema=None, weights=None) -> None:
        """The cosine retreat: a tensor whose smoothed cosine between gained and raw gradients stayed below 0 over the
        whole probe interval has its excess scale lam multiplied by `retreat` (it never rises)."""
        for _, b in self.pairs:
            st = self.state.get(b)
            if not st or "interval" not in st:
                continue
            if bool(st["interval"] > 0) and bool(st["cos_top"] < 0):
                st["lam"].mul_(self.retreat)
            st["interval"].zero_()

    # ------------------------------------------------------------------ helpers

    def _generator(self, device: torch.device, counter: int) -> torch.Generator:
        gen = self._gens.get(device)
        if gen is None:
            gen = self._gens[device] = torch.Generator(device=device)
        gen.manual_seed((self.seed * 1_000_003 + counter) % (2 ** 63))
        return gen

    def _next_dither(self) -> int:
        """The dither's step counter, a host integer in the first pair's state so a relay and a resume carry it. It
        moves on a skipped step too: it is no moment, and a skip must not repeat the rounding of the step before."""
        st = self.state[self.pairs[0][1]]
        counter = st.get("dither", 0) + 1
        st["dither"] = counter
        return counter

    @staticmethod
    def _round_bf16(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """Stochastic rounding of fp32 to bf16: unbiased, so the (1 - b1) increments smaller than half a bf16 ulp of m
        still add up over steps. bf16 is fp32's top 16 bits, so a uniform carry into them before truncating rounds the
        magnitude up with the probability of the dropped fraction, across exponent boundaries too."""
        carry = torch.randint(0, 65536, x.shape, dtype=torch.int32, device=x.device, generator=gen)
        return x.view(torch.int32).add(carry).bitwise_right_shift_(16).to(torch.int16).view(torch.bfloat16)

    def _constants(self, indices: tuple, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype, indices)
        out = self._consts.get(key)
        if out is None:
            out = tuple(torch.tensor([values[i] for i in indices], dtype=dtype, pin_memory=device.type == "cuda")
                        .to(device, non_blocking=True) for values in (self.scales, self.ref_norms))
            self._consts[key] = out
        return out

    # ------------------------------------------------------------------ the step

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        layout: dict = {}  # (device, r) -> (o, i) -> pair indices
        for idx, (a, b) in enumerate(self.pairs):
            if a.grad is None or b.grad is None:
                continue
            if "m" not in self.state[b]:
                self._init_pair(a, b)
            layout.setdefault((b.device, b.shape[1]), {}).setdefault((b.shape[0], a.shape[1]), []).append(idx)
        if not layout:
            return loss
        counter = self._next_dither()
        rho = self.param_groups[0]["lr"]
        for (device, r), shapes in layout.items():
            gen = self._generator(device, counter) if self.state_dtype == "bf16" else None
            buckets = [self._stack(tuple(indices)) for indices in shapes.values()]
            eye = torch.eye(r, dtype=torch.float64, device=device)
            m0 = torch.cat([bk["m0"] for bk in buckets])
            n0 = torch.cat([bk["n0"] for bk in buckets])
            m = m0 + self.damping * (_trace(m0) / r)[:, None, None] * eye
            n = n0 + self.damping * (_trace(n0) / r)[:, None, None] * eye
            t, ti, good = balance_frame(m, n, self.ns_iters)
            # for the first-step branch: it runs every step because telling whether any pair still needs it would sync
            n_inv = torch.linalg.inv_ex(n, check_errors=False)[0]
            at = 0
            for bk in buckets:
                count = bk["a"].shape[0]
                self._update(bk, t[at:at + count], ti[at:at + count], good[at:at + count], n_inv[at:at + count], rho,
                             gen)
                at += count
            if counter % REBALANCE_EVERY == 0:
                for bk in buckets:
                    if r <= min(bk["b"].shape[1], bk["a"].shape[2]):  # a smaller layer can never balance: _rebalance
                        self._rebalance(bk["indices"])
        return loss

    def _stack(self, indices: tuple, region: bool = True) -> dict:
        """One shape's pairs as [n, ...] tensors, with their fp64 Grams."""
        pairs = [self.pairs[i] for i in indices]
        a = torch.stack([p for p, _ in pairs])
        b = torch.stack([p for _, p in pairs])
        ad, bd = a.to(torch.float64), b.to(torch.float64)
        out = dict(indices=indices, a=a, b=b, ga=torch.stack([p.grad for p, _ in pairs]),
                   gb=torch.stack([p.grad for _, p in pairs]), m0=bd.mT @ bd, n0=ad @ ad.mT, region=None)
        found = [self._excess.get(id(p)) for _, p in pairs] if region else []
        if any(x is not None for x in found):
            def part(i, like):
                return torch.stack([x[i].to(like.dtype) if x is not None and x[i] is not None
                                    else torch.zeros_like(like) for x in found])

            armed = torch.tensor([x is not None for x in found], device=b.device)
            out["region"] = (part(0, a[0]), part(1, b[0]), part(2, a[0]), part(3, b[0]), armed)
        return out

    def _gained(self, bk: dict, ga, gb, gb_ti, t, ti, d_a, d_b, lam, old, seen) -> tuple:
        """(gradient for m on A, on B, new smoothed cosine, armed): the raw gradient plus lam X, X projected off the face
        where it opposes it (C3, in the metric of the normalised update), and the balanced-frame cosine between gained
        and raw normalised gradients on lora_up, which the retreat reads. Gains never reach v, u or the step size."""
        x_a, x_b, f_a, f_b, armed = bk["region"]
        db, da = d_b[:, :, None], d_a[:, None, :]
        if self.face_safe:
            over_b = (f_b * (db * (x_b @ ti))).sum((1, 2))
            self_b = (f_b * (db * (f_b @ ti))).sum((1, 2))
            x_b = x_b - (over_b / self_b.clamp_min(1e-30)).clamp_max(0.0)[:, None, None] * f_b
            over_a = (f_a * ((t @ x_a) * da)).sum((1, 2))
            self_a = (f_a * ((t @ f_a) * da)).sum((1, 2))
            x_a = x_a - (over_a / self_a.clamp_min(1e-30)).clamp_max(0.0)[:, None, None] * f_a
        mult = lam[:, None, None]
        full_a, full_b = ga + mult * x_a, gb + mult * x_b
        raw_ti = db * gb_ti
        full_ti = db * (full_b @ ti)
        inner = ((db * full_b) * raw_ti).sum((1, 2))
        raw2 = ((db * gb) * raw_ti).sum((1, 2)).clamp_min(1e-30)
        full2 = ((db * full_b) * full_ti).sum((1, 2)).clamp_min(1e-30)
        cosine = inner / (raw2 * full2).sqrt()
        smoothed = torch.where(seen, COSINE_DECAY * old + (1.0 - COSINE_DECAY) * cosine, cosine)
        return full_a, full_b, smoothed, armed & torch.isfinite(smoothed)

    def _update(self, bk: dict, t64, ti64, good, n_inv, rho: float, gen) -> None:
        a, b, ga, gb = bk["a"], bk["b"], bk["ga"], bk["gb"]
        wd = b.dtype
        count, _, r = b.shape
        pairs = [self.pairs[i] for i in bk["indices"]]
        sts_a = [self.state[p] for p, _ in pairs]
        sts_b = [self.state[p] for _, p in pairs]
        take = lambda sts, key: torch.stack([st[key] for st in sts])  # noqa: E731
        s, ref = self._constants(bk["indices"], b.device, wd)
        k, u = take(sts_b, "k"), take(sts_b, "u")
        v_a, v_b = take(sts_a, "v"), take(sts_b, "v")
        m_a, m_b = take(sts_a, "m"), take(sts_b, "m")
        t_old, ti_old = take(sts_b, "T"), take(sts_b, "Ti")
        flag = good[:, None, None]
        t = torch.where(flag, t64.to(wd), t_old)  # a failed root keeps the previous T, in the pair's own frame
        ti = torch.where(flag, ti64.to(wd), ti_old)
        m0, n0 = bk["m0"].to(wd), bk["n0"].to(wd)

        # the raw gradient's second moment per channel, in the balanced frame
        gb_ti = gb @ ti
        t_ga = t @ ga
        q_b = (gb_ti * gb).sum(-1).clamp_min(0.0) / r
        q_a = (t_ga * ga).sum(-2).clamp_min(0.0) / r
        v_b_new = self.b2 * v_b + (1.0 - self.b2) * q_b
        v_a_new = self.b2 * v_a + (1.0 - self.b2) * q_a
        k_new = k + 1.0
        unbias2 = -torch.expm1(k_new * math.log(self.b2))
        unbias1 = -torch.expm1(k_new * math.log(self.b1)) if self.b1 > 0 else torch.ones_like(k_new)
        hat_b = v_b_new / unbias2[:, None]
        hat_a = v_a_new / unbias2[:, None]
        d_b = (hat_b + 1e-12 * hat_b.mean(-1, keepdim=True) + 1e-30).pow(-0.5)
        d_a = (hat_a + 1e-12 * hat_a.mean(-1, keepdim=True) + 1e-30).pow(-0.5)

        # per-channel clip at RMS `clip`, measured in the balanced frame: a channel's RMS after D is sqrt(q) D
        clipped_b = (self.clip / (q_b.sqrt() * d_b).clamp_min(1e-30)).clamp_max(1.0) * d_b
        clipped_a = (self.clip / (q_a.sqrt() * d_a).clamp_min(1e-30)).clamp_max(1.0) * d_a
        full_a, full_b, region = ga, gb, None  # what m averages: the raw gradient, plus the region excess while armed
        if bk["region"] is not None:
            full_a, full_b, cos_new, used = self._gained(bk, ga, gb, gb_ti, t, ti, d_a, d_b, take(sts_b, "lam"),
                                                         take(sts_b, "cos"), take(sts_b, "armed") > 0)
            region = (cos_new, used)
        m_b_new = m_b.to(wd, copy=True).mul_(self.b1).add_(clipped_b[:, :, None] * full_b, alpha=1.0 - self.b1)
        m_a_new = m_a.to(wd, copy=True).mul_(self.b1).add_(full_a * clipped_a[:, None, :], alpha=1.0 - self.b1)
        u_b = m_b_new @ ti  # the balanced update, before the momentum's bias correction `lift`
        u_a = t @ m_a_new
        lift = 1.0 / unbias1

        # the step in dW units, from the raw normalised gradient: gains (C1) will never reach u
        p_b = d_b[:, :, None] * gb_ti
        q_t = t_ga * d_a[:, None, :]
        p_gram = p_b.mT @ p_b
        cross = (p_b.mT @ b) @ (q_t @ a.mT)
        norm2 = _tr_product(p_gram, n0) + 2.0 * _trace(cross) + _tr_product(q_t @ q_t.mT, m0)
        u_new = self.b2 * u + (1.0 - self.b2) * s.square() * norm2.clamp_min(0.0)
        step = rho * ref
        eta = step / (u_new / unbias2).sqrt()

        # caps: the factor cap bounds the dB dA term, the safety cap the realised step, second order included
        ub_gram = u_b.mT @ u_b
        ua_gram = u_a @ u_a.mT
        factor = self.factor_cap * torch.minimum(
            (_tr_product(m0, t).clamp_min(0.0) / _tr_product(ub_gram, t).clamp_min(1e-30)).sqrt(),
            (_tr_product(n0, ti).clamp_min(0.0) / _tr_product(ua_gram, ti).clamp_min(1e-30)).sqrt()) / lift
        cross = (u_b.mT @ b) @ (u_a @ a.mT)
        d1 = lift * s * (_tr_product(ub_gram, n0) + 2.0 * _trace(cross) + _tr_product(ua_gram, m0)).clamp_min(0.0).sqrt()
        d2 = lift.square() * s * _tr_product(ub_gram, ua_gram).clamp_min(0.0).sqrt()
        limit = self.safety_cap * step
        safety = 2.0 * limit / (d1 + (d1.square() + 4.0 * d2 * limit).sqrt()).clamp_min(1e-30)
        eta_final = torch.minimum(torch.minimum(eta, factor), safety)
        eta_final = torch.where(u_new > 0, eta_final, torch.zeros_like(eta_final))

        started = b.flatten(1).any(1)
        finite = (torch.isfinite(ga.flatten(1).sum(1)) & torch.isfinite(gb.flatten(1).sum(1))
                  & torch.isfinite(full_a.flatten(1).sum(1)) & torch.isfinite(full_b.flatten(1).sum(1))
                  & torch.isfinite(eta_final) & torch.isfinite(d1))
        ok = started & finite
        ok_row, ok_m = ok[:, None], ok[:, None, None]
        grow = (eta_final * lift)[:, None, None]
        new_a = torch.where(ok_m, a - grow * u_a, a)
        new_b = torch.where(ok_m, b - grow * u_b, b)

        first = self._first_step(gb, n_inv.to(wd), n0, s, step)
        take_first = (~started & torch.isfinite(gb.flatten(1).sum(1)))[:, None, None]
        new_b = torch.where(take_first, first, new_b)

        if gen is not None and m_b.dtype == torch.bfloat16:
            m_b_new, m_a_new = self._round_bf16(m_b_new, gen), self._round_bf16(m_a_new, gen)
        m_b_new = torch.where(ok_m, m_b_new.to(m_b.dtype), m_b)
        m_a_new = torch.where(ok_m, m_a_new.to(m_a.dtype), m_a)
        late = k_new > CAP_REPORT_AFTER
        bound_safety = ok & late & (safety < torch.minimum(eta, factor))
        bound_factor = ok & (factor < torch.minimum(eta, safety))

        a_params = [p for p, _ in pairs]
        b_params = [p for _, p in pairs]
        torch._foreach_copy_(a_params, list(new_a.unbind(0)))
        torch._foreach_copy_(b_params, list(new_b.unbind(0)))
        put = lambda sts, key, value: torch._foreach_copy_([st[key] for st in sts], list(value.unbind(0)))  # noqa: E731
        put(sts_a, "m", m_a_new)
        put(sts_b, "m", m_b_new)
        put(sts_a, "v", torch.where(ok_row, v_a_new, v_a))
        put(sts_b, "v", torch.where(ok_row, v_b_new, v_b))
        put(sts_b, "u", torch.where(ok, u_new, u))
        put(sts_b, "k", torch.where(ok, k_new, k))
        put(sts_b, "T", torch.where(ok_m, t, t_old))
        put(sts_b, "Ti", torch.where(ok_m, ti, ti_old))
        put(sts_b, "cap_safety", take(sts_b, "cap_safety") + bound_safety)
        put(sts_b, "cap_factor", take(sts_b, "cap_factor") + bound_factor)
        put(sts_b, "cap_steps", take(sts_b, "cap_steps") + (ok & late))
        put(sts_b, "frame_fallback", take(sts_b, "frame_fallback") + (ok & ~good))
        if region is not None:
            used = region[1] & ok
            put(sts_b, "cos", torch.where(used, region[0], take(sts_b, "cos")))
            put(sts_b, "armed", take(sts_b, "armed") + used)
            count = take(sts_b, "interval")
            top = torch.where(count > 0, take(sts_b, "cos_top"), -torch.ones_like(count))
            put(sts_b, "cos_top", torch.where(used, torch.maximum(top, region[0]), top))
            put(sts_b, "interval", count + used)

    @torch.no_grad()
    def _rebalance(self, indices: tuple) -> None:
        """B <- B X, A <- X^-1 A with X = T^1/2, so BtB and AAt meet and dW stays, in fp64 before the factors go back.
        The momenta are gradient-shaped and go the other way, m_B <- m_B X^-1, m_A <- X m_A; T becomes the identity. A
        pair with no frame (B still zero, or a failed root) is left as it is. The caller skips pairs with a rank above
        min(o, i), such as Krea 2's 12 -> 1 text_fusion.projector at rank 32: BtB and AAt can never meet there, so X stays
        huge at every rebalance and multiplies itself into the lora_ema shadow, whose dW reached 3e6 x ||W0|| by step 100
        while the live weights stayed healthy."""
        bk = self._stack(indices, region=False)
        a, b = bk["a"].to(torch.float64), bk["b"].to(torch.float64)
        wd = bk["b"].dtype
        r = b.shape[2]
        eye = torch.eye(r, dtype=torch.float64, device=b.device)
        m = bk["m0"] + self.damping * (_trace(bk["m0"]) / r)[:, None, None] * eye
        n = bk["n0"] + self.damping * (_trace(bk["n0"]) / r)[:, None, None] * eye
        t, _, good = balance_frame(m, n, self.ns_iters)
        lam, q = torch.linalg.eigh(_sym(torch.where(good[:, None, None], t, eye)))
        flag = (good & (lam.amin(-1) > 0))[:, None, None]
        root = lam.clamp_min(1e-300).sqrt()[:, None, :]
        x = torch.where(flag, (q * root) @ q.mT, eye)
        xi = torch.where(flag, (q / root) @ q.mT, eye)

        pairs = [self.pairs[i] for i in indices]
        sts_a = [self.state[p] for p, _ in pairs]
        sts_b = [self.state[p] for _, p in pairs]
        take = lambda sts, key: torch.stack([st[key] for st in sts])  # noqa: E731
        put = lambda sts, key, value: torch._foreach_copy_([st[key] for st in sts], list(value.unbind(0)))  # noqa: E731
        torch._foreach_copy_([p for p, _ in pairs], list((xi @ a).to(wd).unbind(0)))
        torch._foreach_copy_([p for _, p in pairs], list((b @ x).to(wd).unbind(0)))
        m_a, m_b = take(sts_a, "m"), take(sts_b, "m")
        put(sts_a, "m", (x @ m_a.to(torch.float64)).to(m_a.dtype))
        put(sts_b, "m", (m_b.to(torch.float64) @ xi).to(m_b.dtype))
        same = eye.to(wd).expand_as(take(sts_b, "T"))
        put(sts_b, "T", torch.where(flag, same, take(sts_b, "T")))
        put(sts_b, "Ti", torch.where(flag, same, take(sts_b, "Ti")))
        if self._ema is not None:
            for j, (pa, pb) in enumerate(pairs):
                sa = self._ema.shadow[self._ema_slots[id(pa)]]
                sb = self._ema.shadow[self._ema_slots[id(pb)]]
                product = xi[j].to(sa.device) @ sa.to(torch.float64)
                sa.copy_(product.to(sa.dtype))
                product = sb.to(torch.float64) @ x[j].to(sb.device)
                sb.copy_(product.to(sb.dtype))

    @staticmethod
    def _first_step(gb, n_inv, n0, s, step) -> torch.Tensor:
        """B for a pair whose B is still all zero: the rows of G_B (AAt + delta I)^-1, each of unit dW-space norm, scaled
        so ||dW|| = step."""
        solved = gb @ n_inv
        norms = ((solved @ n0) * solved).sum(-1).clamp_min(0.0).sqrt()
        live = norms > 0
        rows = solved / norms.where(live, 1.0)[:, :, None]
        rows = torch.where(live[:, :, None], rows, torch.zeros_like(rows))
        eta = step / (s * live.sum(1).clamp_min(1).to(gb.dtype).sqrt())
        return -eta[:, None, None] * rows
