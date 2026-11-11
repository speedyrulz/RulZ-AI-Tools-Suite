"""Training maps: where on a few fixed training pictures the model is still wrong, where the LoRA changes its
prediction, and where the trigger word's attention lands, captured before training and then about once an epoch.
A port of Fizgig's training maps (shootthesound/Fizgig, Apache-2.0: src/fizgig/training/train_maps.py and
attn_probe.py) onto RulZ Trainer's one model call, `model(latents, sigma * 1000, embeds, mask)`.

Per picture and per fixed noise level (sigma 0.25 / 0.5 / 0.75, with the same noise every time, so between captures
only the LoRA changes), one value per latent cell:
  * "loss"   - (prediction - target)^2, the mean over channels: what training still gets wrong ("Not learned yet");
  * "effect" - |prediction with the LoRA - prediction with it switched off| ("Where the LoRA acts"; not for a full
               fine-tune, which has no adapter to switch off);
  * "effect_off" - the same as "effect" with the trigger word taken out of the caption (training_maps_trigger_off):
               what the LoRA changes when it was not asked for. Its mean over the mean of "effect" is the capture's
               "bleed": near 0, the LoRA is tied to its trigger; near 1, it acts the same without it;
  * "attn"   - the trigger word's share of each image token's attention over the caption (softmax over the text
               keys, mean over heads and over every block with a text attention): rulztrainer.alignment's own
               measure, so it differs a little from Fizgig's, which normalises over every key. Per word too
               ("attn_by") when the caption has several trigger words.
"Learned since the start" is computed by the viewer: the first capture's loss minus the current one.

Nothing here changes training: no gradients, the model in eval mode for the capture and put back after, the global
RNG restored, the weight average (lora_ema) used as samples and probes use it. A failure is never fatal: one log
line, and the maps stop for the rest of the run.

Output, in <job>/training_maps/: probes.json (the pictures, their captions' trigger words, the noise levels),
probe_<n>.jpg (each picture exactly as training sees it: the bucket crop) and maps.jsonl, one line per capture:
{"step", "epoch", "maps": {probe id: {sigma: {"loss": grid, "effect": grid, "attn": grid, "attn_by": {...}}}}}
with each grid {"w", "h", "v": [row-major floats]}, pooled to at most MAX_SIDE cells on its long side.

The second half of the file draws a capture over its picture for the WebUI's Training maps tab (render_overlay).
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from pathlib import Path

log = logging.getLogger("rulztrainer")

DIRNAME = "training_maps"
SIGMAS = (0.25, 0.5, 0.75)  # low noise (detail) to high noise (layout); never tiny (the loss has no 1/sigma, but
# at sigma near 0 the target is almost pure noise minus the picture and says little)
MAX_SIDE = 48
SEED = 1234
_MISSING = object()


def maps_dir(job_dir) -> Path:
    return Path(job_dir) / DIRNAME


# ---------------------------------------------------------------- picking the pictures


def pick_keys(keys, n: int) -> list[str]:
    """n keys spread evenly through the sorted list (the same dataset picks the same pictures)."""
    keys = sorted({str(k) for k in keys})
    n = max(0, min(int(n), len(keys)))
    if n == 0:
        return []
    if n == 1:
        return [keys[len(keys) // 2]]
    return [keys[round(i * (len(keys) - 1) / (n - 1))] for i in range(n)]


def strip_trigger(caption: str, words) -> str:
    """The caption without its trigger word(s): each whole-word occurrence taken out (any case), the punctuation and
    spaces it leaves tidied. The caption itself when none occurs."""
    import re

    out = str(caption)
    for w in sorted({str(w) for w in words if w}, key=len, reverse=True):
        out = re.sub(rf"(?<!\w){re.escape(w)}(?!\w)", "", out, flags=re.IGNORECASE)
    if out == caption:
        return caption
    out = re.sub(r"\s+([,.;:])", r"\1", out)
    out = re.sub(r"([,;:])(\s*[,;:])+", r"\1", out)
    return re.sub(r"\s{2,}", " ", out).strip(" ,;:")


def bleed(maps: dict) -> float | None:
    """A capture's bleed: over every picture and noise level with both maps, the mean of "effect_off" over the mean of
    "effect" (None without a trigger-off pass)."""
    ratios = []
    for per in (maps or {}).values():
        for level in (per or {}).values():
            on, off = (level or {}).get("effect"), (level or {}).get("effect_off")
            if on and off and on.get("v") and off.get("v"):
                base = sum(on["v"]) / len(on["v"])
                if base > 0:
                    ratios.append((sum(off["v"]) / len(off["v"])) / base)
    return round(sum(ratios) / len(ratios), 4) if ratios else None


def eligible(item) -> bool:
    """A picture the maps can use: a whole photo (not a close-up crop), not one of LoRA Isolation's side-by-side
    pairs (those come in separately, see pick_items)."""
    return getattr(item, "category", "image") == "image" and not getattr(item, "pair_side", "")


def pick_items(items, n: int) -> list:
    """The map pictures: one from every dataset first (each subject of a multi-dataset run), one of LoRA Isolation's
    side-by-side pairs when the run has them (does the trigger land on the subject, not the stranger beside them?),
    then the rest spread evenly through the photos. At least one per dataset, so more than n with more datasets."""
    by_key, groups, pairs = {}, {}, []
    for it in items:
        k = str(it.key)
        if k in by_key:
            continue
        if eligible(it):
            by_key[k] = it
            groups.setdefault(str(getattr(it, "dataset", "") or ""), []).append(k)
        elif getattr(it, "pair_side", "") and getattr(it, "category", "image") == "image":
            by_key[k] = it
            pairs.append(k)
    out = [pick_keys(groups[g], 1)[0] for g in sorted(groups)] if len(groups) > 1 else []
    if pairs:
        out.append(pick_keys(pairs, 1)[0])
    want = max(int(n), len(out))
    for k in pick_keys([k for g in groups.values() for k in g], want + len(out)):
        if len(out) >= want:
            break
        if k not in out:
            out.append(k)
    return [by_key[k] for k in out]


# ---------------------------------------------------------------- maps from tensors


def to_map(err, latents):
    """A per-element error laid out as the model's prediction -> an (h, w) map. A latent grid [B, C, h, w] averages
    its channels (the first sample); tokens [B, N] or [N] are laid row by row on the patch grid the latents make."""
    if err.dim() == 4:
        return err.mean(dim=1)[0]
    tok = err.reshape(-1) if err.dim() == 1 else err[0].reshape(-1)
    h, w = latents.shape[-2:]
    n = tok.numel()
    p = max(1, round((h * w / max(1, n)) ** 0.5))
    gh, gw = -(-h // p), -(-w // p)
    if gh * gw != n:
        gh, gw = h // p, w // p
    if gh * gw != n:
        raise RuntimeError(f"can't lay {n} tokens on a {h}x{w} latent grid")
    return tok.view(gh, gw)


def grid(m) -> dict:
    """An (h, w) map -> {"w", "h", "v"}, pooled to at most MAX_SIDE on its long side, 4 significant figures."""
    import torch.nn.functional as F

    m = m.detach().float().cpu()
    h, w = m.shape
    k = max(1, -(-max(h, w) // MAX_SIDE))
    if k > 1:
        m = F.avg_pool2d(m[None, None], k, ceil_mode=True)[0, 0]
    h, w = m.shape
    return {"w": int(w), "h": int(h), "v": [float(f"{x:.4g}") for x in m.flatten().tolist()]}


# ---------------------------------------------------------------- the trigger's attention


class MapTaps:
    """Stands in for alignment.Taps on the model while a map forward runs: the model hands it the token layout, and
    the attention taps (MapAttnTap) add up the trigger's share of each image token's attention over the caption."""

    hidden_layers = frozenset()

    def __init__(self, groups: dict[str, list[int]]):
        self.groups = {w: sorted({int(p) for p in ps}) for w, ps in groups.items() if ps}
        if self.groups and "" not in self.groups:
            self.groups[""] = sorted({p for ps in self.groups.values() for p in ps})
        self.armed = self.recording = self.attn_on = True
        self.layout = None
        self.acc: dict[str, object] = {}
        self.n = 0

    def set_layout(self, sample, img, txt, txt_mask=None) -> None:
        self.layout = {"sample": int(sample), "img": img, "txt": txt, "mask": txt_mask}

    def record_hidden(self, layer, img_tokens) -> None:
        pass

    def see(self, q, k) -> None:
        from .alignment import text_attention

        lay = self.layout
        if lay is None or not self.groups:
            return
        (i0, n_img), (t0, n_txt) = lay["img"], lay["txt"]
        if n_txt < 0:
            n_txt = k.shape[2] - t0
        mask = lay["mask"]
        probs = text_attention(q[:1, :, i0:i0 + n_img], k[:1, :, t0:t0 + n_txt],
                               mask[:1] if mask is not None else None)[0]  # [n_img, n_txt]
        for word, pos in self.groups.items():
            cols = [p for p in pos if p < probs.shape[1]]
            if not cols:
                continue
            share = probs[:, cols].sum(-1).float()
            self.acc[word] = share if word not in self.acc else self.acc[word] + share.to(self.acc[word].device)
        self.n += 1

    def result(self) -> dict | None:
        """{word: its mean share of each image token's attention over the blocks seen}, "" for all of them."""
        return None if not self.acc or not self.n else {w: a / self.n for w, a in self.acc.items()}


class MapAttnTap:
    def __init__(self, taps: MapTaps):
        self.taps = taps

    def __call__(self, q, k) -> None:
        self.taps.see(q, k)


@contextlib.contextmanager
def attention_taps(model, taps: MapTaps):
    """MapTaps on the model and a tap on every block's text attention for the forwards inside, whatever taps
    training had there (attention localization, Keel's face tap, Hull) put back exactly afterwards."""
    from .alignment import attention_module

    saved = []
    old_taps = model.__dict__.get("_rz_taps", _MISSING)
    try:
        model._rz_taps = taps
        for block in model.layers:
            module = attention_module(block)
            if module is None:
                continue
            saved.append((module, module.__dict__.get("_rz_tap", _MISSING)))
            module._rz_tap = MapAttnTap(taps)
        yield taps
    finally:
        for module, old in saved:
            if old is _MISSING:
                module.__dict__.pop("_rz_tap", None)
            else:
                module._rz_tap = old
        if old_taps is _MISSING:
            model.__dict__.pop("_rz_taps", None)
        else:
            model._rz_taps = old_taps


def eager_stance():
    """Compiled blocks run eagerly for a map forward: the attention taps are Python a compiled block never calls, and
    a no-grad forward at a training shape must not trace graphs of its own. A null context without set_stance."""
    import torch

    if getattr(getattr(torch, "compiler", None), "set_stance", None) is None:
        return contextlib.nullcontext()
    try:
        return torch.compiler.set_stance("force_eager")
    except Exception:  # noqa: BLE001
        return contextlib.nullcontext()


# ---------------------------------------------------------------- capture


class TrainingMaps:
    """The run's map pictures and their captures. `probes`: [{"id", "item", "latents", "embeds", "mask",
    "trigger": {word: token positions}, "off": {"embeds", "mask"} of the caption without its trigger word, when the
    trigger-off pass is on}] built by the trainer (see Trainer._setup_maps)."""

    def __init__(self, job_dir, probes: list[dict], every: int, sigmas=SIGMAS, seed: int = SEED):
        self.dir = maps_dir(job_dir)
        self.probes = probes
        self.every = max(1, int(every))
        self.sigmas = tuple(float(s) for s in sigmas)
        self.seed = int(seed)
        self.failed = False
        self.last_step: int | None = None
        self.last_bleed: float | None = None
        self.next_step = 0

    def start(self, step: int, resume: bool, pictures: dict[str, object] | None = None,
              words: list[str] | None = None) -> None:
        """A fresh run clears the old maps; a resumed one adds to them. Writes probes.json and the pictures."""
        self.dir.mkdir(parents=True, exist_ok=True)
        if not resume:
            for p in [self.dir / "maps.jsonl", *self.dir.glob("probe_*.jpg")]:
                with contextlib.suppress(OSError):
                    p.unlink()
        meta = []
        for pr in self.probes:
            pic = (pictures or {}).get(pr["id"])
            name = f"probe_{pr['id']}.jpg"
            if pic is not None:
                with contextlib.suppress(OSError):
                    pic.convert("RGB").save(self.dir / name, quality=90)
            it = pr["item"]
            meta.append({"id": pr["id"], "key": str(it.key), "image": name, "source": str(it.image),
                         "dataset": str(getattr(it, "dataset", "") or ""),
                         "pair": bool(getattr(it, "pair_side", "")),
                         "words": [w for w in pr.get("trigger", {}) if w], "trigger_off": bool(pr.get("off")),
                         "size": [int(it.width), int(it.height)]})
        with open(self.dir / "probes.json", "w", encoding="utf-8") as f:
            json.dump({"probes": meta, "sigmas": list(self.sigmas), "every": self.every, "words": words or []}, f,
                      indent=1)
        self.next_step = step
        n_attn = sum(1 for pr in self.probes if pr.get("trigger"))
        log.info("Training maps on: %d picture(s) x %d noise level(s), every %d steps%s (the Training maps tab)",
                 len(self.probes), len(self.sigmas), self.every,
                 f"; the trigger word's attention on {n_attn} of them" if n_attn else "")

    def due(self, step: int, last: bool = False) -> bool:
        if self.failed or not self.probes or step == self.last_step:
            return False
        return step >= self.next_step or last

    def capture(self, step: int, epoch: float, model, injected: dict, device, *, averaged=None) -> None:
        """This step's maps appended to maps.jsonl; never fatal."""
        if self.failed or not self.probes:
            return
        t0 = time.time()
        try:
            out = self._run(model, injected, device, averaged, effect=step > 0 or self.last_step is not None)
        except Exception as e:  # noqa: BLE001
            self.failed = True
            log.warning("Training maps failed at step %d, so they are off for the rest of this run; training goes "
                        "on (%s: %s)", step, type(e).__name__, str(e)[:200])
            log.debug("training maps error", exc_info=True)
            return
        finally:
            self.last_step = step
            self.next_step = step + self.every
        row = {"step": int(step), "epoch": round(float(epoch), 3), "maps": out}
        self.last_bleed = bleed(out)
        if self.last_bleed is not None:
            row["bleed"] = self.last_bleed
        with open(self.dir / "maps.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        log.info("Training maps at step %d: %d picture(s), %.1fs%s", step, len(self.probes), time.time() - t0,
                 "" if self.last_bleed is None else
                 f"; without the trigger word the LoRA acts x{self.last_bleed:.2f} as much")

    def noise(self, j: int, i: int, shape):
        """The fixed noise of picture j at noise level i (its own generator: the global RNG is never drawn from)."""
        import torch

        g = torch.Generator("cpu").manual_seed(self.seed * 1000003 + j * 1009 + i)
        return torch.randn(shape, generator=g)

    def _run(self, model, injected: dict, device, averaged, effect: bool) -> dict:
        import torch

        from . import lora as lora_mod

        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        was_training = model.training
        out = {}
        try:
            model.eval()
            with (averaged() if averaged is not None else contextlib.nullcontext()), torch.no_grad(), eager_stance():
                for j, pr in enumerate(self.probes):
                    latents = pr["latents"].to(device, torch.float32)
                    embeds, mask = pr["embeds"].to(device), pr["mask"].to(device)
                    per = {}
                    for i, s in enumerate(self.sigmas):
                        noise = self.noise(j, i, latents.shape).to(device)
                        noisy = ((1 - s) * latents + s * noise).to(torch.bfloat16)
                        target = noise - latents
                        sigma = torch.full((latents.shape[0],), s * 1000.0, device=device)
                        maps = {}
                        trig = pr.get("trigger") or {}
                        if trig:
                            taps = MapTaps(trig)
                            with attention_taps(model, taps):
                                pred = model(noisy, sigma, embeds, mask).float()
                            attn = taps.result()
                            if attn is not None and "" in attn:
                                maps["attn"] = grid(to_map(attn[""], latents))
                                words = [w for w in attn if w]
                                if len(words) > 1:
                                    maps["attn_by"] = {w: grid(to_map(attn[w], latents)) for w in words}
                        else:
                            pred = model(noisy, sigma, embeds, mask).float()
                        maps["loss"] = grid(to_map((pred - target).pow(2), latents))
                        if effect and injected:
                            lora_mod.set_multiplier(injected, 0.0)
                            try:
                                base = model(noisy, sigma, embeds, mask).float()
                            finally:
                                lora_mod.set_multiplier(injected, 1.0)
                            maps["effect"] = grid(to_map((pred - base).abs(), latents))
                            del base
                            off = pr.get("off")
                            if off is not None:  # the trigger-off pass: the same picture, noise and LoRA, asked without
                                e_off, m_off = off["embeds"].to(device), off["mask"].to(device)  # its trigger word
                                asked = model(noisy, sigma, e_off, m_off).float()
                                lora_mod.set_multiplier(injected, 0.0)
                                try:
                                    plain = model(noisy, sigma, e_off, m_off).float()
                                finally:
                                    lora_mod.set_multiplier(injected, 1.0)
                                maps["effect_off"] = grid(to_map((asked - plain).abs(), latents))
                                del asked, plain, e_off, m_off
                        per[f"{s:g}"] = maps
                        del noise, noisy, target, pred
                    out[pr["id"]] = per
                    del latents, embeds, mask
        finally:
            model.train(was_training)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        return out


# ---------------------------------------------------------------- reading and drawing (the WebUI)


def read(job_dir) -> dict | None:
    """A run's maps: {"probes", "sigmas", "words", "captures": [{"step", "epoch", "maps"}]} (a step written twice,
    by a resumed run, keeps the later), or None when the folder has none."""
    d = maps_dir(job_dir)
    try:
        meta = json.loads((d / "probes.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    by = {}
    try:
        with open(d / "maps.jsonl", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # a line still being written
                by[int(row.get("step", 0))] = row
    except OSError:
        pass
    return {"probes": meta.get("probes") or [], "sigmas": meta.get("sigmas") or list(SIGMAS),
            "words": meta.get("words") or [], "dir": str(d),
            "captures": [{"step": s, "epoch": by[s].get("epoch"), "maps": by[s].get("maps") or {},
                          "bleed": by[s].get("bleed")} for s in sorted(by)]}


MAP_KINDS = {
    "learned": "Learned since the start",
    "loss": "Not learned yet",
    "effect": "Where the LoRA acts",
    "effect_off": "Where the LoRA acts without the trigger word",
    "attn": "Trigger word",
}


def _values(g: dict | None):
    import numpy as np

    if not g:
        return None
    try:
        return np.asarray(g["v"], dtype=np.float32).reshape(int(g["h"]), int(g["w"]))
    except (KeyError, ValueError, TypeError):
        return None


def level_map(capture: dict, probe_id: str, kind: str, sigma: str = "all", word: str = ""):
    """One capture's raw map (numpy [h, w]) for a picture: kind loss / effect / attn (word: one trigger word's own
    attention), at one noise level ("0.25") or the mean of every level ("all"). None where it has none."""
    import numpy as np

    per = (capture.get("maps") or {}).get(probe_id) or {}
    levels = list(per) if sigma == "all" else [sigma]
    found = []
    for lv in levels:
        maps = per.get(lv) or {}
        g = maps.get(kind)
        if kind == "attn" and word and "attn_by" in maps:  # (one word alone: its map is the "attn" one)
            g = maps["attn_by"].get(word)
        a = _values(g)
        if a is not None:
            found.append(a)
    if not found or any(a.shape != found[0].shape for a in found):
        return found[0] if found else None
    return np.mean(found, axis=0)


# How "Learned since the start" is measured (after Fizgig's viewer, 10 November 2026). The amount is the loss removed:
# the part that learned most sets the scale, so a part with little to learn (a plain background) stays pale even when
# it is nearly done. The share is the loss removed over the cell's own starting loss: how far along each part is,
# whatever it started at. "fixed" is the share on one scale for every run, so two runs' maps compare by eye.
LEARNED_AS = {"amount": "Amount", "share": "Share of its own start", "fixed": "Share, same scale for every run"}
FIXED_SHARE_TOP = 0.5  # the fixed scale's full colour: half of a cell's starting loss gone (or come back)
SHARE_FLOOR_PCT = 10  # a nearly clean cell's start is floored at this percentile of the picture's, or its share
#                       would be noise over almost nothing
ATTN_FLOOR_PCT = 2  # the trigger map counts from its quietest share up: this percentile, the lowest of any capture


def _learned_share(data: dict, probe_id: str, sigma: str) -> list:
    """Every capture's share of the starting loss removed, per cell: taken per noise level against that level's own
    start (a level's loss has its own size), then the plain mean of the levels."""
    import numpy as np

    caps = data.get("captures") or []
    levels = [sigma]
    if sigma == "all":
        levels = sorted({lv for c in caps for lv in ((c.get("maps") or {}).get(probe_id) or {})})
    per_level = []
    for lv in levels:
        losses = [level_map(c, probe_id, "loss", lv) for c in caps]
        first = next((a for a in losses if a is not None), None)
        if first is None:
            continue
        floor = max(float(np.percentile(first, SHARE_FLOOR_PCT)), 1e-8)
        start = np.maximum(first, floor)
        per_level.append([None if a is None or a.shape != first.shape else (first - a) / start for a in losses])
    out = []
    for i in range(len(caps)):
        found = [lv[i] for lv in per_level if lv[i] is not None]
        ok = bool(found) and all(a.shape == found[0].shape for a in found)
        out.append(np.mean(found, axis=0) if ok else None)
    return out


def series(data: dict, probe_id: str, kind: str, sigma: str = "all", word: str = "",
           learned_as: str = "amount") -> list:
    """Every capture's map for one view, [numpy or None] in capture order. "learned" is the first loss map minus
    each capture's (positive = improved), or with learned_as "share" / "fixed" that as a share of each cell's own
    start. "attn" counts from the series' quietest share up (ATTN_FLOOR_PCT): some models spread a new word's
    attention almost evenly, more so with long captions, and counted from zero the whole picture is one colour."""
    import numpy as np

    caps = data.get("captures") or []
    if kind == "attn":
        maps = [level_map(c, probe_id, kind, sigma, word) for c in caps]
        lows = [float(np.percentile(a, ATTN_FLOOR_PCT)) for a in maps if a is not None]
        low = min(lows) if lows else 0.0
        return [None if a is None else np.maximum(a - low, 0.0) for a in maps]
    if kind != "learned":
        return [level_map(c, probe_id, kind, sigma, word) for c in caps]
    if learned_as in ("share", "fixed"):
        return _learned_share(data, probe_id, sigma)
    losses = [level_map(c, probe_id, "loss", sigma) for c in caps]
    first = next((a for a in losses if a is not None), None)
    return [None if a is None or first is None or a.shape != first.shape else first - a for a in losses]


def scale(maps: list, kind: str, learned_as: str = "amount") -> float:
    """One colour scale for a whole series (so captures compare): the 99th percentile of every value (of the absolute
    value for "learned"), never 0. learned_as "fixed": FIXED_SHARE_TOP, whatever the series holds."""
    import numpy as np

    if kind == "learned" and learned_as == "fixed":
        return FIXED_SHARE_TOP
    vals = [np.abs(a).ravel() if kind == "learned" else a.ravel() for a in maps if a is not None]
    if not vals:
        return 1.0
    top = float(np.percentile(np.concatenate(vals), 99))
    return top if top > 0 and math.isfinite(top) else 1.0


_HEAT = ((0.0, (40, 80, 220)), (0.5, (250, 215, 40)), (1.0, (225, 30, 30)))  # blue, yellow, red
_GOOD, _BAD = (40, 210, 90), (225, 40, 200)  # green: improved, magenta: worse


def colorize(a, kind: str, top: float):
    """[h, w] values -> ([h, w, 3] uint8 colours, [h, w] float opacity 0..1)."""
    import numpy as np

    if kind == "learned":
        x = np.clip(a / top, -1.0, 1.0)
        rgb = np.where(x[..., None] >= 0, np.array(_GOOD, np.float32), np.array(_BAD, np.float32))
        return rgb.astype(np.uint8), np.abs(x) * 0.9
    x = np.clip(a / top, 0.0, 1.0)
    pos = np.array([p for p, _ in _HEAT], np.float32)
    rgb = np.stack([np.interp(x, pos, [c[ch] for _, c in _HEAT]) for ch in range(3)], axis=-1)
    return rgb.astype(np.uint8), 0.3 + 0.6 * x


def render_overlay(picture, a, kind: str, top: float, opacity: float = 1.0, smooth: bool = True,
                   colour: bool = False, width: int = 768):
    """The map over its picture as a PIL image `width` wide: the picture in grey (or colour), the map's colours
    blended in by value, times `opacity`. a None: the picture alone."""
    import numpy as np
    from PIL import Image, ImageOps

    pic = picture.convert("RGB")
    h = max(1, round(pic.height * width / max(1, pic.width)))
    pic = pic.resize((width, h), Image.Resampling.LANCZOS)
    base = np.asarray(pic if colour else ImageOps.grayscale(pic).convert("RGB"), dtype=np.float32)
    if a is None:
        return Image.fromarray(base.astype(np.uint8))
    rgb, alpha = colorize(np.asarray(a, np.float32), kind, top)
    resample = Image.Resampling.BILINEAR if smooth else Image.Resampling.NEAREST
    rgb_big = np.asarray(Image.fromarray(rgb).resize((width, h), resample), dtype=np.float32)
    alpha_big = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).resize((width, h), resample),
                           dtype=np.float32)[..., None] / 255.0
    alpha_big = alpha_big * max(0.0, min(1.0, float(opacity)))
    out = base * (1 - alpha_big) + rgb_big * alpha_big
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def capture_label(capture: dict) -> str:
    step = int(capture.get("step") or 0)
    ep = capture.get("epoch")
    if step == 0:
        return "before training"
    return f"step {step:,}" + (f" (epoch {ep:g})" if isinstance(ep, (int, float)) else "")


def sigma_label(s: str) -> str:
    try:
        v = float(s)
    except ValueError:
        return "All levels"
    return {0.25: "Low noise (detail)", 0.5: "Mid noise", 0.75: "High noise (layout)"}.get(v, f"sigma {v:g}")
