"""M0(d) of docs/hull-optimizer.md: how well Hull's body, face and background gradients agree across steps.

Trains plain Hull at gains 1: nothing here reaches the optimizer. Each step's part of a region is put in Hull's balanced
frame with Hull's own normaliser, scaled to unit norm so every photo has one say, and fed to two EMAs at beta 0.9:

  coherence  ||m|| / unbias over a whole tensor, 0.23 on noise
  purity     sum |m| / sum n over a tensor's elements (design section 5.7), 0.23 on Gaussian noise

`control` is the body part with a random sign per step, the noise floor of that same stream of gradients. It reads
above 0.23 on purity because real gradients are heavy tailed, so read every region against it.

The config needs `optimizer: hull` and `hull_regions: parts`, which builds the parts maps; the gains themselves are not
used. It costs three more GEMMs per kept block and about 40 B per trained weight, so turn probes and saves off.

    python scripts/hull_agreement.py run --config cfg.yaml --out s42.npz --seed 42
    python scripts/hull_agreement.py read s42.npz s1234.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BETA = 0.9
REGIONS = ("body", "face", "background", "all", "control")
BODY, ALL, CONTROL = 0, 3, 4
SNAP_EVERY = 10
MIN_STEPS = 20  # a layer's region is read after this many steps with any of that region in it


def masks(rm, token_gains):
    """(body, face, background) shares per cell, flattened. The body is what hull_gain_body would scale, and hair,
    clothing and the rest are what they leave."""
    body = token_gains(rm.parts, rm.face_cells, 2.0, 1.0)[0] - 1.0
    gain, face = token_gains(rm.parts, rm.face_cells, 1.0, 0.0)
    return body.flatten(), face.to(torch.float32).flatten(), (1.0 - gain).flatten()


class Recorder:
    """The EMAs per (pairs of one shape, factor, region), and a snapshot of their readings every SNAP_EVERY steps."""

    def __init__(self, out: Path, seed: int):
        self.out, self.seed = out, seed
        self.parts, self.ema, self.energy, self.gens = {}, {}, {}, {}
        self.snaps, self.steps = [], []
        self.count = 0

    def add(self, pa, pb, parts_a, parts_b, _x, _y):
        """A LoRA backward's three regions for one layer, summed over a step's micro-batches."""
        old = self.parts.get(id(pb))
        new = (parts_a, parts_b)
        self.parts[id(pb)] = new if old is None else tuple([o + n for o, n in zip(was, now)]
                                                           for was, now in zip(old, new))

    def feed(self, key, side: int, region: int, part: torch.Tensor) -> torch.Tensor:
        """One shape's part [n, x, y] of a region, and its squared norms. A pair where it is empty keeps its EMA."""
        norm = part.flatten(1).norm(dim=1)
        has = norm > 0
        unit = (part / norm.clamp_min(1e-30)[:, None, None]).float()
        st = self.ema.get((key, side, region))
        if st is None:  # fp32: bf16's rounding reads a constant direction as 0.98
            st = self.ema[(key, side, region)] = dict(m=torch.zeros_like(unit), n=torch.zeros_like(unit),
                                                      k=torch.zeros(part.shape[0], device=part.device))
        flag = has[:, None, None]
        st["m"] = torch.where(flag, BETA * st["m"] + (1 - BETA) * unit, st["m"])
        st["n"] = torch.where(flag, BETA * st["n"] + (1 - BETA) * unit.abs(), st["n"])
        st["k"] += has
        return norm.square()

    def read(self, key, side: int, region: int) -> tuple:
        """(coherence, purity) per pair of one shape, nan before MIN_STEPS."""
        st = self.ema[(key, side, region)]
        m, n = st["m"], st["n"]
        unbias = -torch.expm1(st["k"] * float(np.log(BETA)))
        coherence = m.flatten(1).norm(dim=1) / unbias.clamp_min(1e-6)
        purity = m.abs().flatten(1).sum(1) / n.flatten(1).sum(1).clamp_min(1e-30)
        enough = st["k"] >= MIN_STEPS
        nan = torch.tensor(float("nan"), device=m.device)
        return coherence.where(enough, nan).cpu().numpy(), purity.where(enough, nan).cpu().numpy()

    def snapshot(self, step: int, npairs: int) -> None:
        readings = np.full((npairs, 2, len(REGIONS), 2), np.nan, dtype=np.float32)
        share = np.full((npairs, 2, 3), np.nan, dtype=np.float32)
        for key, side, region in self.ema:
            readings[list(key), side, region, 0], readings[list(key), side, region, 1] = self.read(key, side, region)
        for (key, side), e in self.energy.items():
            share[list(key), side] = (e["sum"] / max(e["steps"], 1)).float().cpu().numpy()
            e["sum"], e["steps"] = 0.0, 0
        self.snaps.append((readings, share))
        self.steps.append(step)
        if len(self.steps) % 5 == 0:
            self.save()

    def save(self) -> None:
        np.savez_compressed(self.out, steps=np.array(self.steps), readings=np.stack([s[0] for s in self.snaps]),
                            share=np.stack([s[1] for s in self.snaps]), regions=np.array(REGIONS))


def patched_backward(ctx, du):
    """lora.HullBackward's backward with three region parts in place of the excess and the face part."""
    x, h, a, b = ctx.saved_tensors
    sink, pa, pb, regions, _ = ctx.arm
    dh = du @ b
    grad_b = (du.mT @ h).sum(0)
    grad_a = (dh.mT @ x).sum(0)
    grad_x = dh @ a if ctx.needs_input_grad[0] else None
    parts_a, parts_b = [], []
    for i in range(regions.shape[-1]):
        e = regions[:, :, i, None]
        parts_b.append(((du * e).mT @ h).sum(0))
        parts_a.append(((dh * e).mT @ x).sum(0))
    sink(pa, pb, parts_a, parts_b, None, None)
    return grad_x, grad_a, grad_b, None


class RegionArm:
    """One lora.HullArm per region over one layout: tokens() gives the [count, n_tok, regions] masks."""

    def __init__(self, regions: list, layout, sink, hull_arm):
        self.sink = sink
        self.arms = [hull_arm(e, [None] * len(e), layout, sink) for e in regions]

    def tokens(self, x):
        got = [arm.tokens(x) for arm in self.arms]
        if got[0] is None:
            return None
        return torch.stack([g[0] if g is not None else torch.zeros_like(got[0][0]) for g in got], dim=-1), None


def install(rec: Recorder):
    """Wrap the trainer's and Hull's methods around `rec`. Returns the function that puts them back."""
    from rulztrainer import lora as L
    from rulztrainer import trainer as T
    from rulztrainer.optimizers import hull as H
    from rulztrainer.regions import token_gains

    saved = [(L.HullBackward, "backward", L.HullBackward.__dict__["backward"]),
             (T.Trainer, "_hull_arm", T.Trainer._hull_arm), (H.Hull, "_update", H.Hull._update),
             (H.Hull, "zero_grad", H.Hull.zero_grad), (H.Hull, "step", H.Hull.step)]
    update, zero, step = H.Hull._update, H.Hull.zero_grad, H.Hull.step

    def hull_arm(self, optimizer, taps, regions):
        if not hasattr(optimizer, "add_excess") or taps is None or regions is None:
            return None
        lists = ([], [], [])
        for rm in regions.meta:
            parts = (None, None, None) if rm.parts is None else masks(rm, token_gains)
            for out, part in zip(lists, parts):
                out.append(part)
        if all(b is None for b in lists[0]):
            return None
        return RegionArm(list(lists), lambda: taps.layout, rec.add, L.HullArm)

    def zero_grad(self, set_to_none=True):
        rec.parts.clear()
        return zero(self, set_to_none=set_to_none)

    def _update(self, bk, t64, ti64, good, n_inv, rho, gen):
        pairs = [self.pairs[i] for i in bk["indices"]]
        sts_a, sts_b = [self.state[p] for p, _ in pairs], [self.state[p] for _, p in pairs]
        wd = bk["b"].dtype
        flag = good[:, None, None]
        t = torch.where(flag, t64.to(wd), torch.stack([st["T"] for st in sts_b]))
        ti = torch.where(flag, ti64.to(wd), torch.stack([st["Ti"] for st in sts_b]))
        update(self, bk, t64, ti64, good, n_inv, rho, gen)
        k = torch.stack([st["k"] for st in sts_b])
        unbias = -torch.expm1(k * float(np.log(self.b2)))
        d_b = (torch.stack([st["v"] for st in sts_b]).float() / unbias[:, None] + 1e-30).pow(-0.5)
        d_a = (torch.stack([st["v"] for st in sts_a]).float() / unbias[:, None] + 1e-30).pow(-0.5)

        def balanced_b(g):
            return d_b[:, :, None] * (g.to(wd) @ ti)

        def balanced_a(g):
            return (t @ g.to(wd)) * d_a[:, None, :]

        key = tuple(bk["indices"])
        got = [rec.parts.get(id(p)) for _, p in pairs]
        signs = rec.gens.get(k.device)
        if signs is None:
            signs = rec.gens[k.device] = torch.Generator(device=k.device).manual_seed(rec.seed)
        sign = (torch.randint(0, 2, (len(pairs),), device=k.device, generator=signs).to(wd) * 2 - 1)[:, None, None]
        for side, (raw, balanced, like) in enumerate(((bk["gb"], balanced_b, bk["b"]), (bk["ga"], balanced_a, bk["a"]))):
            total = rec.feed(key, side, ALL, balanced(raw))
            shares = []
            for region in range(3):
                part = balanced(torch.stack([x[1 - side][region].to(like.device) if x is not None
                                             else torch.zeros_like(like[0]) for x in got]))
                shares.append(rec.feed(key, side, region, part) / total.clamp_min(1e-30))
                if region == BODY:
                    rec.feed(key, side, CONTROL, part * sign)
            e = rec.energy.setdefault((key, side), dict(sum=0.0, steps=0))
            e["sum"] = e["sum"] + torch.stack(shares, dim=1)
            e["steps"] += 1

    def counted_step(self, closure=None):
        out = step(self, closure)
        rec.count += 1
        if rec.count % SNAP_EVERY == 0:
            rec.snapshot(rec.count, len(self.pairs))
        return out

    L.HullBackward.backward = staticmethod(patched_backward)
    T.Trainer._hull_arm = hull_arm
    H.Hull._update, H.Hull.zero_grad, H.Hull.step = _update, zero_grad, counted_step

    def undo():
        for owner, name, value in saved:
            setattr(owner, name, value)

    return undo


def summary(path: str, after: int = 100) -> None:
    """Each region's median over tensors minus its control's, per snapshot after step `after`, and how often body wins."""
    with np.load(path) as d:
        steps, readings, share = d["steps"], d["readings"], d["share"]
    keep = steps > after
    tensors = np.isfinite(readings[keep, :, :, BODY, 0]).any(axis=0)  # text-only layers never see a body
    series = np.stack([readings[keep][:, tensors, region] for region in range(len(REGIONS))], axis=1)  # snap, region, n, m
    print(f"{path}: {int(keep.sum())} snapshots, steps {int(steps[keep][0])}-{int(steps[keep][-1])}, "
          f"{int(tensors.sum())} of {tensors.size} tensors")
    for metric, name in enumerate(("coherence", "purity")):
        medians = np.nanmedian(series[:, :, :, metric], axis=2)
        above = medians[:, :CONTROL] - medians[:, CONTROL:CONTROL + 1]
        text = ", ".join(f"{REGIONS[i]} {above[:, i].mean():+.4f}" for i in range(CONTROL))
        means = np.nanmean(series[:, :, :, metric], axis=0)
        body, face, background = means[BODY], means[1], means[2]
        print(f"  {name}: control {medians[:, CONTROL].mean():.4f}; above it: {text}; tensors with body above "
              f"background {100 * (body > background).mean():.0f}%, at or above face {100 * (body >= face).mean():.0f}%")
    energy = np.nanmedian(np.nanmean(share[keep][:, tensors], axis=0), axis=0)
    print(f"  energy share: body {energy[0]:.3f}, face {energy[1]:.3f}, background {energy[2]:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--seed", type=int, default=0, help="the control's random signs")
    read = commands.add_parser("read")
    read.add_argument("files", nargs="+")
    args = parser.parse_args()
    if args.command == "read":
        for path in args.files:
            summary(path)
        return
    from rulztrainer.train import main as train

    rec = Recorder(Path(args.out), args.seed)
    install(rec)
    sys.argv = ["train", "--config", args.config]
    try:
        train()
    finally:
        if rec.steps:
            rec.save()


if __name__ == "__main__":
    main()
