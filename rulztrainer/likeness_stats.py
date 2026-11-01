"""Probe noise: how far a likeness probe's score moves with its seeds, and paired tests between probes that share them.

A probe renders probe_seeds seeds (the same ones every time) and averages their scores. Two probes on the same seeds
are paired: each seed's difference cancels what that seed does to both (a hard pose, an odd crop), so the noise of a
comparison is the spread of the per-seed differences, sd_diff = sigma * sqrt(2 (1 - rho)), not of the scores
themselves (sigma: the seed-to-seed spread of one probe; rho: how alike two probes' seeds score).

    paired(a, b)          per-seed scores of two probes -> n, mean difference, sd_diff, rho, one-sided p (a > b)
    seeds_needed(sd, gap) seeds a probe needs for a paired test to catch a gap (one-sided, 90% confidence, 80% power)
    run_noise(records)    a run's own probes (likeness.jsonl with per-seed scores): seed spread and probe-to-probe noise
    noise_report(...)     the noise check stage (noise_files): sigma, sd_diff, seeds needed, a likeness_noise_band

Pure Python (no scipy): the Student t tail comes from the regularized incomplete beta function.

    python -m rulztrainer.likeness_stats <job folder>   a finished run's probe noise, from its likeness.jsonl
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

KINDS = ("face", "body", "combined")
GAPS = (3.0, 5.0, 8.0)  # normalized points: the gaps the seeds-needed table answers for


def _betacf(a: float, b: float, x: float) -> float:
    """The continued fraction of the incomplete beta function (Numerical Recipes, Lentz's method)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """P(T > t) for Student's t with df degrees of freedom."""
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t >= 0 else 1.0 - tail


def mean_sd(values) -> tuple[float | None, float | None]:
    """The mean and the sample standard deviation (None with fewer than 1 / 2 values)."""
    v = [float(x) for x in values if x is not None]
    if not v:
        return None, None
    m = sum(v) / len(v)
    if len(v) < 2:
        return m, None
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def _corr(a: list[float], b: list[float]) -> float | None:
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if sa == 0 or sb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def paired(a, b) -> dict:
    """Two probes' per-seed scores (same seeds, same order; None where a seed went unscored): the seeds both scored,
    their mean difference a - b, its spread sd_diff, the correlation rho of the two probes' seeds, and the one-sided
    p that a scores no higher than b (a paired t-test; None below 3 pairs, where it cannot tell anything apart)."""
    pairs = [(float(x), float(y)) for x, y in zip(a or [], b or []) if x is not None and y is not None]
    n = len(pairs)
    out = {"n": n, "diff": None, "sd_diff": None, "rho": None, "p": None}
    if not n:
        return out
    d = [x - y for x, y in pairs]
    out["diff"], out["sd_diff"] = mean_sd(d)
    if n >= 3:
        out["rho"] = _corr([x for x, _ in pairs], [y for _, y in pairs])
        sd = out["sd_diff"]
        if sd == 0:
            out["p"] = 0.0 if out["diff"] > 0 else 1.0
        else:
            out["p"] = t_sf(out["diff"] / (sd / math.sqrt(n)), n - 1)
    return out


def seeds_needed(sd_diff: float, gap: float, confidence: float = 0.9, power: float = 0.8) -> int:
    """Seeds a paired probe needs so a one-sided test at `confidence` catches a true gap of `gap` points with
    probability `power`: ((z_conf + z_power) sd_diff / gap)^2, at least 3 (a t-test's least)."""
    if sd_diff is None or gap <= 0:
        return 0
    z = NormalDist().inv_cdf(confidence) + NormalDist().inv_cdf(power)
    return max(3, math.ceil((z * sd_diff / gap) ** 2))


def suggested_band(sd_diff: float, probe_seeds: int, probe_ema: float) -> float:
    """A likeness_noise_band for a run probed with probe_seeds seeds and smoothed with probe_ema: one standard
    deviation of the change between two smoothed probes when nothing changed (the EMA keeps sqrt((1-a)/(1+a)) of
    independent noise), rounded to half a point."""
    a = min(max(float(probe_ema), 0.0), 0.99)
    sd = sd_diff / math.sqrt(max(1, int(probe_seeds))) * math.sqrt((1 - a) / (1 + a))
    return max(0.5, round(sd * 2) / 2)


def run_noise(records: list[dict]) -> dict:
    """A run's probe noise from its own probes (likeness.jsonl records with per-seed scores, normalized:
    face_norm_seeds, body_norm_seeds, combined_seeds): sigma, the pooled seed-to-seed spread inside a probe (the step-0
    probe left out: the base model), and sd_diff / rho, the spread of the per-seed changes between consecutive probes
    (true learning included, so an upper bound)."""
    out = {}
    recs = [r for r in records if r.get("step", 0) > 0]
    for kind in KINDS:
        key = f"{kind}_norm_seeds" if kind != "combined" else "combined_seeds"
        variances = []
        for r in recs:
            _, sd = mean_sd(r.get(key) or [])
            if sd is not None:
                variances.append(sd * sd)
        diffs, rhos = [], []
        for prev, cur in zip(recs, recs[1:]):
            p = paired(cur.get(key) or [], prev.get(key) or [])
            if p["sd_diff"] is not None:
                diffs.append(p["sd_diff"] ** 2)
            if p["rho"] is not None:
                rhos.append(p["rho"])
        out[kind] = {"sigma": math.sqrt(sum(variances) / len(variances)) if variances else None,
                     "probes": len(variances),
                     "sd_diff": math.sqrt(sum(diffs) / len(diffs)) if diffs else None,
                     "rho": sum(rhos) / len(rhos) if rhos else None}
    return out


def noise_report(files: dict[str, dict], probe_seeds: int, probe_ema: float) -> dict:
    """The noise check's summary. files: {name: {"face": [normalized per seed], "body": [...], "combined": [...]}},
    each file probed on the same seeds. Per kind: each file's mean and sigma, the pooled sigma, every pair's paired
    statistics and the pooled sd_diff (sigma * sqrt(2) when only one file was probed: no rho to lower it), then the
    seeds a probe needs per gap and a likeness_noise_band for probe_seeds / probe_ema."""
    names = list(files)
    out: dict = {"files": {}, "kinds": {}, "probe_seeds": probe_seeds, "probe_ema": probe_ema}
    for name in names:
        out["files"][name] = {}
        for kind in KINDS:
            vals = files[name].get(kind) or []
            m, sd = mean_sd(vals)
            out["files"][name][kind] = {"mean": m, "sigma": sd, "n": sum(1 for x in vals if x is not None)}
    for kind in KINDS:
        sig = [out["files"][n][kind]["sigma"] for n in names if out["files"][n][kind]["sigma"] is not None]
        sigma = math.sqrt(sum(s * s for s in sig) / len(sig)) if sig else None
        pairs = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pairs[f"{b} - {a}"] = paired(files[b].get(kind), files[a].get(kind))
        sds = [p["sd_diff"] for p in pairs.values() if p["sd_diff"] is not None and p["n"] >= 3]
        rhos = [p["rho"] for p in pairs.values() if p["rho"] is not None]
        if sds:
            sd_diff, source = math.sqrt(sum(s * s for s in sds) / len(sds)), "pairs"
        else:
            sd_diff, source = (sigma * math.sqrt(2) if sigma is not None else None), "sigma x 1.41 (one file)"
        entry = {"sigma": sigma, "rho": sum(rhos) / len(rhos) if rhos else None, "sd_diff": sd_diff,
                 "sd_diff_from": source, "pairs": pairs}
        if sd_diff is not None:
            entry["seeds_needed"] = {f"{g:g}": seeds_needed(sd_diff, g) for g in GAPS}
            entry["probe_sd"] = {str(k): sd_diff / math.sqrt(k) for k in (1, 2, 4, 8)}
        out["kinds"][kind] = entry
    comb = out["kinds"]["combined"]
    if comb.get("sd_diff") is not None:
        out["suggested_noise_band"] = suggested_band(comb["sd_diff"], probe_seeds, probe_ema)
    return out


def describe(report: dict) -> list[str]:
    """The noise check's report as log lines."""
    def fmt(v):
        return "n/a" if v is None else f"{v:.1f}"

    lines = []
    for kind, e in report["kinds"].items():
        if e.get("sd_diff") is None:
            continue
        need = ", ".join(f"{g} points {n}" for g, n in e["seeds_needed"].items())
        lines.append(f"{kind}: seeds spread {fmt(e['sigma'])} points around a probe's mean, alike across files rho "
                     f"{fmt(e['rho'])}, a per-seed difference spreads {fmt(e['sd_diff'])} ({e['sd_diff_from']}); "
                     f"seeds a probe needs to tell apart a gap of {need}")
    if "suggested_noise_band" in report:
        lines.append(f"likeness_noise_band {report['suggested_noise_band']:g} fits {report['probe_seeds']} probe "
                     f"seed(s) at probe_ema {report['probe_ema']:g} (one standard deviation of the change between two "
                     f"smoothed probes)")
    return lines


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("python -m rulztrainer.likeness_stats <job folder>")
        return 2
    from .likeness_export import read_probes

    records = read_probes(Path(argv[0]))
    if not any(r.get("combined_seeds") for r in records):
        print("No per-seed probe scores in this run's likeness.jsonl (a run from before v2.16.0, or no probes)")
        return 1
    print(json.dumps(run_noise(records), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
