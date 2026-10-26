"""Training-only losses that read the model's insides: where the trigger word's attention lands, and what the face's
tokens represent. Neither changes what is saved: the LoRA file is the same kind of file, and the identity
alignment's small projector is thrown away.

  * attention localization (attn_localize): in a few attention layers, every image token's attention over the text
    is read (softmax over the text tokens only, averaged over heads) and the share of it that goes to the trigger
    word's tokens is taken as a map over the image. Normalized to sum 1, the loss is the part of that map outside
    the subject (the person mask and face box the region maps already hold): 1 - share inside. It ties the identity
    to the trigger and keeps the trigger's attention off the background and off other people, the kind of
    positional supervision MultiCrafter (2025) used on FLUX's attention to stop attributes leaking between subjects.
    MOSS-Image and Krea 2 read it in their joint attention; FIBO 2.0 condenses the caption into Perceiver "gist"
    tokens, so there it is read in the cross-attention of its injection blocks, which see the caption token by
    token.
  * identity feature alignment (id_align): after one block, the image tokens inside the face box are averaged and a
    small MLP maps them onto the training photo's own ArcFace embedding (1 - cosine). It is REPA's representation
    alignment (Yu et al. 2024, aligning a diffusion transformer's hidden states to a frozen encoder's features)
    pointed at identity: it gives identity feedback at every noise level, where the experimental identity loss
    (id_loss_weight) needs a nearly clean prediction and a VAE decode. Untested idea: race it.

Taps carry the side outputs. The trainer arms them for a batch with the trigger token positions of each sample,
the model records on its main forward, recording stops before any other forward, and they are disarmed after the
backward. A checkpointed block recomputes its forward in the backward and must save the same tensors, so armed taps
keep computing (without recording) until then; the gradient through a recorded attention map then reaches the
block like any other (checked against an unchecked block).
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn

log = logging.getLogger("rulztrainer")

FACE_EMB_DIM = 512  # ArcFace (InsightFace buffalo_l w600k_r50)


class Taps:
    """One model's side outputs for a batch: the image tokens' hidden states after `hidden_layers`, and in
    `attn_layers` the share of each image token's text attention that goes to the trigger tokens."""

    def __init__(self, hidden_layers=(), attn_layers=()):
        self.hidden_layers = frozenset(int(i) for i in hidden_layers)
        self.attn_layers = frozenset(int(i) for i in attn_layers)
        self.armed = False
        self.recording = False
        self.attn_on = False  # read the attention this batch (fixed from begin() to end(): the recompute matches)
        self.positions: list[list[int]] = []  # per sample: its trigger tokens' indices in its text
        self.layout: dict | None = None  # set by the model per forward (see set_layout)
        self.hidden: dict[int, dict[int, torch.Tensor]] = {}  # layer -> sample -> [n_img, dim]
        self.attn: dict[int, dict[int, torch.Tensor]] = {}  # layer -> sample -> [n_img]

    def begin(self, positions: list[list[int]], attention: bool = True) -> None:
        self.positions = [list(p) for p in positions]
        self.hidden, self.attn = {}, {}
        self.attn_on = bool(attention and self.attn_layers)
        self.armed = self.recording = True

    def stop(self) -> None:
        """Keep computing (a checkpointed block's recompute must match) but record nothing more."""
        self.recording = False

    def end(self) -> None:
        self.armed = self.recording = self.attn_on = False
        self.positions, self.hidden, self.attn, self.layout = [], {}, {}, None

    def set_layout(self, sample: int, img: tuple[int, int], txt: tuple[int, int], txt_mask=None) -> None:
        """Where this forward's tokens are: the first batch sample it holds, the image tokens (start, count) in the
        query sequence, the text tokens (start, count; count -1: to the end) in the key sequence and their [B, count]
        validity mask. A block's recompute in the backward may run under a later forward's layout, so what the tap
        computes depends only on the parts that are the same for every sample of the batch."""
        self.layout = {"sample": int(sample), "img": img, "txt": txt, "mask": txt_mask}

    def record_hidden(self, layer: int, img_tokens: torch.Tensor) -> None:
        """img_tokens [B, n_img, dim] after block `layer`."""
        if not (self.armed and self.recording and layer in self.hidden_layers and self.layout is not None):
            return
        first = self.layout["sample"]
        for j in range(img_tokens.shape[0]):
            self.hidden.setdefault(layer, {})[first + j] = img_tokens[j]


class AttnTap:
    """Installed on an attention module as `_rz_tap`: called with its rotated q [B, H, Lq, d] and k [B, H, Lk, d]
    (heads already matched) before the attention runs; records each image token's attention over the text."""

    def __init__(self, taps: Taps, layer: int):
        self.taps, self.layer = taps, int(layer)

    def __call__(self, q: torch.Tensor, k: torch.Tensor) -> None:
        t = self.taps
        if not (t.armed and t.attn_on) or t.layout is None:
            return
        lay = t.layout
        (i0, n_img), (t0, n_txt) = lay["img"], lay["txt"]
        if n_txt < 0:
            n_txt = k.shape[2] - t0
        probs = text_attention(q[:, :, i0:i0 + n_img], k[:, :, t0:t0 + n_txt], lay["mask"])
        if t.recording:
            for j in range(q.shape[0]):
                t.attn.setdefault(self.layer, {})[lay["sample"] + j] = probs[j]


def text_attention(q_img: torch.Tensor, k_txt: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """[B, n_img, n_txt]: each image token's attention over the text tokens (softmax over those alone, the invalid
    ones masked), averaged over heads."""
    logits = torch.matmul(q_img, k_txt.transpose(-1, -2)).float() * (q_img.shape[-1] ** -0.5)  # [B, H, n_img, n_txt]
    if mask is not None:
        logits = logits.masked_fill(~mask.to(logits.device, torch.bool)[:, None, None, :], float("-inf"))
    return logits.softmax(-1).mean(1)


def attention_module(block: nn.Module) -> nn.Module | None:
    """The module whose `_rz_tap` a block calls: FIBO 2.0's cross-attention (None outside its injection blocks),
    Krea 2's attention, MOSS-Image's block itself."""
    if hasattr(block, "cross_attention"):
        return block.cross_attention
    if hasattr(block, "attn"):
        return block.attn
    return block


def model_dim(model: nn.Module) -> int:
    """The transformer's hidden size (the width of the image tokens' states)."""
    block = model.layers[0]
    for name in ("proj.q", "attn.to_q", "attention.to_q"):
        try:
            return int(block.get_submodule(name).in_features)
        except AttributeError:
            continue
    raise ValueError("cannot tell the model's hidden size for identity alignment")


def subject_mask(rm) -> torch.Tensor | None:
    """The subject's cells [h * w] (1 = face or person) from a region map, or None when it has neither."""
    if rm is None or not (rm.face_cells or rm.has_person):
        return None
    return (~rm.prior).flatten().float()


def localization_loss(taps: Taps, regions) -> tuple[torch.Tensor | None, int]:
    """1 - the share of the trigger's attention map (normalized to sum 1 over the image) that lies on the subject,
    averaged over the tapped layers and the samples that have a subject and a trigger. Returns (loss, samples)."""
    terms, samples = [], set()
    for per in taps.attn.values():
        for b, probs in per.items():
            pos = [p for p in (taps.positions[b] if b < len(taps.positions) else []) if 0 <= p < probs.shape[-1]]
            mask = subject_mask(regions.meta[b]) if b < len(regions.meta) else None
            if not pos or mask is None or mask.numel() != probs.shape[0]:
                continue
            share = probs[:, pos].sum(-1)  # each image token's attention to the trigger
            p = share / share.sum().clamp_min(1e-8)
            terms.append(1.0 - (p * mask.to(p.device)).sum())
            samples.add(b)
    if not terms:
        return None, 0
    return torch.stack(terms).mean(), len(samples)


class IdAligner(nn.Module):
    """The identity alignment's projector: mean face-token state [dim] -> ArcFace space [512]. Training-only."""

    def __init__(self, dim: int, hidden: int = 1024, out: int = FACE_EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, out))

    def loss(self, taps: Taps, layer: int, regions, targets: list) -> tuple[torch.Tensor | None, int]:
        """1 - cosine between the projected mean of each sample's face tokens and its photo's face embedding, over
        the samples with a face box and an embedding. Returns (loss, samples)."""
        per = taps.hidden.get(layer, {})
        dev = next(self.parameters()).device
        terms = []
        for b, h in per.items():
            rm = regions.meta[b] if b < len(regions.meta) else None
            target = targets[b] if b < len(targets) else None
            if rm is None or rm.face_cells is None or target is None:
                continue
            hh, ww = regions.prior.shape[-2:]
            if h.shape[0] != hh * ww:
                continue
            cx0, cy0, cx1, cy1 = rm.face_cells
            face = h.reshape(hh, ww, -1)[cy0:cy1, cx0:cx1].reshape(-1, h.shape[-1])
            if face.shape[0] == 0:
                continue
            z = self.net(face.float().mean(0).to(dev))
            terms.append(1.0 - F.cosine_similarity(z, target.to(dev, torch.float32), dim=0))
        if not terms:
            return None, 0
        return torch.stack(terms).mean(), len(terms)


def auto_attn_layers(arch: str, n_layers: int, injection_layers=()) -> list[int]:
    """The default attention layers: FIBO 2.0's injection blocks (where it reads the caption token by token); for
    the joint-attention models two blocks at a third and at half the depth, where the layout of a picture forms."""
    if arch == "fibo2":
        return sorted(int(i) for i in injection_layers)
    return sorted({max(0, n_layers // 3), max(0, n_layers // 2)})


def auto_hidden_layer(n_layers: int) -> int:
    """The default block for identity alignment: a third of the depth (REPA aligned early-middle blocks)."""
    return max(0, n_layers // 3)


def parse_layers(spec: str, n_layers: int) -> list[int] | None:
    """"" -> None (auto); "10,21" -> [10, 21] (each within the model)."""
    spec = (spec or "").strip()
    if not spec:
        return None
    out = sorted({int(s) for s in spec.replace(";", ",").split(",") if s.strip()})
    bad = [i for i in out if not 0 <= i < n_layers]
    if bad:
        raise ValueError(f"attention layers {bad} are outside the model's {n_layers} blocks (0-{n_layers - 1})")
    return out


def token_positions(text: str, offsets: list, triggers: list[str], drop: int = 0) -> list[int]:
    """Indices (minus `drop` dropped leading tokens) of the tokens whose character span overlaps an occurrence of any
    trigger in `text` (offsets: the tokenizer's (start, end) per token)."""
    spans = []
    low = text.lower()
    for trig in {t.strip() for t in triggers if t and t.strip()}:
        start = low.find(trig.lower())
        while start >= 0:
            spans.append((start, start + len(trig)))
            start = low.find(trig.lower(), start + 1)
    out = []
    for i, (a, b) in enumerate(offsets):
        if b > a and any(a < e and b > s for s, e in spans) and i >= drop:
            out.append(i - drop)
    return out
