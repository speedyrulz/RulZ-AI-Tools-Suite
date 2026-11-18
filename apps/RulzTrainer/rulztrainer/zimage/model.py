"""Z-Image S3-DiT (ZImageTransformer2DModel) for training, in RulzTrainer's conventions.

Ported from diffusers' transformer_z_image.py (Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team, Apache-2.0)
with the diffusers mixins removed, so that:
  - the state-dict keys are exactly diffusers' (Tongyi-MAI/Z-Image[-Turbo] transformer/): layers.N.attention.to_q /
    to_k / to_v / to_out.0, norm_q / norm_k, feed_forward.w1 / w2 / w3, adaLN_modulation.0, noise_refiner,
    context_refiner, cap_embedder, t_embedder, all_x_embedder["2-1"], all_final_layer["2-1"], x_pad_token,
    cap_pad_token. A diffusers state dict loads with strict=True, and a LoRA exported with these names loads in
    diffusers and (through ComfyUI's z_image map) in ComfyUI. ComfyUI's fused single file is converted by
    zimage/single_file.py;
  - the forward takes what the trainer and sampler give the other families: `hidden_states` as the 2x2-patchified
    VAE latents [B, 64, H/16, W/16] (channel index c * 4 + ph * 2 + pw, the house order of h3/model.py), `timestep`
    in 0..1000 units (sigma * 1000), the caption as [B, S, cap_feat_dim] with a [B, S] mask, and it returns the
    velocity (noise - x0) in the input's layout. Z-Image itself works on t = 1 - sigma and predicts x0 - noise, so
    the forward flips both;
  - `layers` holds the 30 main blocks only. The refiners, embedders and pad tokens stay resident (see RESIDENT);
  - gradient checkpointing and the model-split hand-off work like the other ports; `skip_layers` skips main blocks;
  - QuantLinear / LoRALinear replace nn.Linear modules transparently (only their dtype handling is consulted).

What the diffusers forward does with a batch (a list of images and a list of captions of different lengths) and this
port reproduces, row by row:
  - the caption is the row's masked tokens, padded to a multiple of 32 with `cap_pad_token`. The pad tokens take
    part in attention and carry sequential position ids like the real tokens (1, 2, ...): only batch padding is
    masked;
  - the image tokens (row-major over the latent's 2x2 patches, each patch's features ordered (ph, pw, c), which is
    NOT the house order) are padded to a multiple of 32 with `x_pad_token`; the pad tokens have position ids
    (0, 0, 0). A real image token sits at (padded caption length + 1, row, col), so the image RoPE depends on the
    row's caption length;
  - one stream per row in the order [image, caption], then the final layer on the image tokens.
"""
from __future__ import annotations

import fnmatch
import math

import torch
import torch.nn.functional as F
from torch import nn

from ..attention import sdpa
from ..checkpointing import checkpointed
from ..compiling import thread_prepare

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32
PATCH = 2
F_PATCH = 1
PATCH_KEY = f"{PATCH}-{F_PATCH}"

# diffusers' ZImageTransformer2DModel config of Tongyi-MAI/Z-Image and Z-Image-Turbo (S3-DiT, 6.15B parameters)
ZIMAGE_CONFIG = dict(in_channels=16, dim=3840, n_layers=30, n_refiner_layers=2, n_heads=30, n_kv_heads=30,
                     norm_eps=1e-5, qk_norm=True, cap_feat_dim=2560, rope_theta=256.0, t_scale=1000.0,
                     axes_dims=(32, 48, 48), axes_lens=(1024, 512, 512), patch_size=PATCH, f_patch_size=F_PATCH)

# Modules that quantization must leave in full precision (ai-toolkit's list): matched by `matches` against the dotted
# module / parameter name, as a prefix on a module boundary or as an fnmatch pattern.
FULL_PRECISION = ("t_embedder", "cap_embedder", "all_x_embedder", "all_final_layer", "x_pad_token", "cap_pad_token",
                  "noise_refiner.*.adaLN_modulation", "*attention_norm1", "*attention_norm2", "*ffn_norm1",
                  "*ffn_norm2", "*.norm_q", "*.norm_k")
# Modules block swap must never move: the pad token Parameters are read before and after the blocks (ai-toolkit
# e6c5aea), the refiners, embedders and the final layer are small and run outside the block loop.
RESIDENT = ("noise_refiner", "context_refiner", "t_embedder", "cap_embedder", "all_x_embedder", "all_final_layer",
            "x_pad_token", "cap_pad_token")


def matches(name: str, patterns) -> bool:
    """Whether the dotted `name` is one of `patterns` or inside one, or fits an fnmatch pattern with a wildcard."""
    for p in patterns:
        if "*" in p or "?" in p:
            if fnmatch.fnmatchcase(name, p) or fnmatch.fnmatchcase(name, p + ".*"):
                return True
        elif name == p or name.startswith(p + "."):
            return True
    return False


def _wdtype(m: nn.Module) -> torch.dtype:
    d = getattr(m, "compute_dtype", None)
    return d if d is not None else m.weight.dtype


def _module_device(m: nn.Module) -> torch.device | None:
    p = next(m.parameters(), None)
    if p is not None:
        return p.device
    b = next(m.buffers(), None)
    return b.device if b is not None else None


def patchify(latents: torch.Tensor) -> torch.Tensor:
    """[B, 16, h, w] latents -> [B, 64, h/2, w/2], channel index c * 4 + ph * 2 + pw (house order)."""
    b, c, h, w = latents.shape
    if h % PATCH or w % PATCH:
        raise ValueError(f"latent size {h}x{w} is not a multiple of {PATCH}")
    x = latents.reshape(b, c, h // PATCH, PATCH, w // PATCH, PATCH).permute(0, 1, 3, 5, 2, 4)
    return x.reshape(b, c * PATCH * PATCH, h // PATCH, w // PATCH)


def unpatchify(x: torch.Tensor) -> torch.Tensor:
    """[B, 64, h, w] -> [B, 16, 2h, 2w]: the inverse of patchify."""
    b, cc, h, w = x.shape
    c = cc // (PATCH * PATCH)
    x = x.reshape(b, c, PATCH, PATCH, h, w).permute(0, 1, 4, 2, 5, 3)
    return x.reshape(b, c, h * PATCH, w * PATCH)


def _house_to_tokens(x: torch.Tensor) -> torch.Tensor:
    """[B, 64, h, w] (c, ph, pw) -> [B, h * w, 64] with diffusers' token features ordered (ph, pw, c)."""
    b, cc, h, w = x.shape
    c = cc // (PATCH * PATCH)
    x = x.reshape(b, c, PATCH, PATCH, h, w).permute(0, 4, 5, 2, 3, 1)
    return x.reshape(b, h * w, cc)


def _tokens_to_house(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """The inverse of _house_to_tokens: [B, h * w, 64] (ph, pw, c) -> [B, 64, h, w] (c, ph, pw)."""
    b, n, cc = t.shape
    c = cc // (PATCH * PATCH)
    t = t.reshape(b, h, w, PATCH, PATCH, c).permute(0, 5, 3, 4, 1, 2)
    return t.reshape(b, cc, h, w)


class RMSNorm(nn.Module):
    """diffusers.models.normalization.RMSNorm with elementwise affine (fp32 statistics; a bf16 / fp16 weight casts the
    normalized value to the weight's dtype first, an fp32 weight leaves the product in fp32)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        return x * self.weight


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size: int, mid_size: int | None = None, frequency_embedding_size: int = 256):
        super().__init__()
        mid_size = mid_size or out_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, mid_size, bias=True), nn.SiLU(),
                                 nn.Linear(mid_size, out_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t is the scaled model time (t * t_scale), kept in fp32 until the sinusoid."""
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(_wdtype(self.mlp[0])))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def _apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    cos, sin = freqs_cis.unsqueeze(2).unbind(-1)
    real, imag = pairs.unbind(-1)
    return torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1).flatten(3).type_as(x)


class Attention(nn.Module):
    """Self-attention with per-head q/k RMSNorm (eps 1e-5, as diffusers' Attention builds it) and RoPE."""

    def __init__(self, dim: int, heads: int, qk_norm: bool = True):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])
        self.norm_q = RMSNorm(dim // heads, eps=1e-5) if qk_norm else None
        self.norm_k = RMSNorm(dim // heads, eps=1e-5) if qk_norm else None

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None,
                freqs_cis: torch.Tensor | None = None) -> torch.Tensor:
        q = self.to_q(x).unflatten(-1, (self.heads, -1))
        k = self.to_k(x).unflatten(-1, (self.heads, -1))
        v = self.to_v(x).unflatten(-1, (self.heads, -1))
        if self.norm_q is not None:
            q, k = self.norm_q(q), self.norm_k(k)
        if freqs_cis is not None:
            q, k = _apply_rope(q, freqs_cis), _apply_rope(k, freqs_cis)
        tap = getattr(self, "_rz_tap", None)  # rulztrainer.alignment: the trigger's attention (training-only loss)
        if tap is not None:
            tap(q.transpose(1, 2), k.transpose(1, 2))
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        out = sdpa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attention_mask)
        return self.to_out[0](out.transpose(1, 2).flatten(2).to(q.dtype))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, norm_eps: float, qk_norm: bool, modulation: bool = True):
        super().__init__()
        self.attention = Attention(dim, n_heads, qk_norm)
        self.feed_forward = FeedForward(dim, int(dim / 3 * 8))
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)
        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None, freqs_cis: torch.Tensor,
                adaln_input: torch.Tensor | None = None) -> torch.Tensor:
        if self.modulation:
            mod = self.adaLN_modulation(adaln_input)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mod.unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
            attn_out = self.attention(self.attention_norm1(x) * scale_msa, attention_mask=attn_mask,
                                      freqs_cis=freqs_cis)
            x = x + gate_msa * self.attention_norm2(attn_out)
            return x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        attn_out = self.attention(self.attention_norm1(x), attention_mask=attn_mask, freqs_cis=freqs_cis)
        x = x + self.attention_norm2(attn_out)
        return x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size,
                                                                    bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = (1.0 + self.adaLN_modulation(c)).unsqueeze(1)
        x = self.norm_final(x) * scale
        return self.linear(x.to(_wdtype(self.linear)))


class RopeEmbedder:
    """Real cosine/sine tables per axis, looked up by integer position ids; built lazily on the ids' device so a
    zero-argument construction on the meta device holds no tensors."""

    def __init__(self, theta: float, axes_dims, axes_lens):
        self.theta, self.axes_dims, self.axes_lens = theta, list(axes_dims), list(axes_lens)
        self.freqs_cis: list[torch.Tensor] | None = None

    def _precompute(self) -> list[torch.Tensor]:
        out = []
        for d, e in zip(self.axes_dims, self.axes_lens):
            freqs = 1.0 / (self.theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
            pos = torch.arange(e, dtype=torch.float64, device="cpu")
            freqs = torch.outer(pos, freqs).float()
            out.append(torch.stack((freqs.cos(), freqs.sin()), dim=-1))
        return out

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [..., 3] int -> [..., sum(axes_dims) / 2, 2] cosine/sine pairs."""
        if self.freqs_cis is None or self.freqs_cis[0].device != ids.device:
            self.freqs_cis = [f.to(ids.device) for f in (self.freqs_cis or self._precompute())]
        return torch.cat([self.freqs_cis[i][ids[..., i]] for i in range(len(self.axes_dims))], dim=-2)


class ZImageTransformer(nn.Module):
    """ZImageTransformer2DModel. hidden_states: [B, 64, H/16, W/16], the 2x2-patchified Flux1-VAE latents (see
    patchify)."""

    CONFIG_KEYS = tuple(ZIMAGE_CONFIG)

    def __init__(self, **overrides):
        super().__init__()
        unknown = set(overrides) - set(ZIMAGE_CONFIG)
        if unknown:
            raise TypeError(f"unknown ZImageTransformer option(s): {', '.join(sorted(unknown))}")
        cfg = {**ZIMAGE_CONFIG, **overrides}
        if cfg["patch_size"] != PATCH or cfg["f_patch_size"] != F_PATCH:
            raise ValueError("only patch_size 2 / f_patch_size 1 (the released models) is supported")
        dim, n_heads = cfg["dim"], cfg["n_heads"]
        if dim % n_heads or sum(cfg["axes_dims"]) != dim // n_heads:
            raise ValueError(f"axes_dims {tuple(cfg['axes_dims'])} must add up to dim / n_heads ({dim // n_heads})")
        cfg["axes_dims"], cfg["axes_lens"] = tuple(cfg["axes_dims"]), tuple(cfg["axes_lens"])
        self.config = dict(cfg)
        self.in_channels = cfg["in_channels"]
        self.latent_channels = cfg["in_channels"] * PATCH * PATCH  # 64: the trainer's (patchified) layout
        self.text_dim = cfg["cap_feat_dim"]
        self.dim = dim
        self.t_scale = float(cfg["t_scale"])
        self.gradient_checkpointing = False
        self.is_distilled = False  # Turbo: the loaders set it from the checkpoint (its keys are Base's)
        patch_dim = F_PATCH * PATCH * PATCH * cfg["in_channels"]
        self.all_x_embedder = nn.ModuleDict({PATCH_KEY: nn.Linear(patch_dim, dim, bias=True)})
        self.all_final_layer = nn.ModuleDict({PATCH_KEY: FinalLayer(dim, patch_dim)})
        self.noise_refiner = nn.ModuleList(
            [TransformerBlock(dim, n_heads, cfg["norm_eps"], cfg["qk_norm"], modulation=True)
             for _ in range(cfg["n_refiner_layers"])])
        self.context_refiner = nn.ModuleList(
            [TransformerBlock(dim, n_heads, cfg["norm_eps"], cfg["qk_norm"], modulation=False)
             for _ in range(cfg["n_refiner_layers"])])
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(RMSNorm(cfg["cap_feat_dim"], eps=cfg["norm_eps"]),
                                          nn.Linear(cfg["cap_feat_dim"], dim, bias=True))
        self.x_pad_token = nn.Parameter(torch.zeros((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.zeros((1, dim)))
        self.layers = nn.ModuleList([TransformerBlock(dim, n_heads, cfg["norm_eps"], cfg["qk_norm"])
                                     for _ in range(cfg["n_layers"])])
        self.rope_embedder = RopeEmbedder(cfg["rope_theta"], cfg["axes_dims"], cfg["axes_lens"])

    @classmethod
    def from_config(cls, config: dict) -> "ZImageTransformer":
        return cls(**{k: config[k] for k in cls.CONFIG_KEYS if k in config})

    def split_tail_modules(self) -> list[nn.Module]:
        """Modules that run after the last block (they follow it in a model split)."""
        return [self.all_final_layer]

    def _run(self, block, *args):
        thread_prepare()  # dynamo's limits in this thread (autograd recompute, data-parallel workers)
        if (self.gradient_checkpointing and self.training and torch.is_grad_enabled()
                and not getattr(block, "keep_activations", False)):  # see rulztrainer.checkpointing
            return checkpointed(self, block, *args)
        return block(*args)

    @staticmethod
    def _stack(rows: list[torch.Tensor], length: int) -> torch.Tensor:
        """Rows [n_i, ...] zero-padded at the end to [B, length, ...]."""
        out = rows[0].new_zeros((len(rows), length) + tuple(rows[0].shape[1:]))
        for i, r in enumerate(rows):
            out[i, :r.shape[0]] = r
        return out

    def forward(self, hidden_states: torch.Tensor, timestep, encoder_hidden_states: torch.Tensor,
                encoder_attention_mask: torch.Tensor | None = None, skip_layers=None) -> torch.Tensor:
        """timestep: noise level in 0..1000 (sigma * 1000), scalar or [B]. Returns the velocity (noise - x0) in the
        input's layout."""
        skip_layers = frozenset(skip_layers or ())
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.latent_channels:
            raise ValueError(f"hidden_states must be [B, {self.latent_channels}, H/16, W/16]")
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[2] != self.text_dim:
            raise ValueError(f"encoder_hidden_states must be [B, S, {self.text_dim}]")
        batch, _, height, width = hidden_states.shape
        device = hidden_states.device
        dtype = _wdtype(self.all_x_embedder[PATCH_KEY])
        seq = encoder_hidden_states.shape[1]

        # 1. time: Z-Image's t = 1 - sigma (0..1), times t_scale, in fp32 until the sinusoid (a bf16 round of
        # 1 - sigma near sigma = 0 would be off by up to 4e-3 in t, which the embedding amplifies)
        sigma = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1).expand(batch) / 1000.0
        adaln_input = self.t_embedder((1.0 - sigma) * self.t_scale).to(dtype)

        # 2. captions: the row's masked tokens, padded to a multiple of 32 with cap_pad_token
        if encoder_attention_mask is None:
            valid = torch.ones(batch, seq, dtype=torch.bool, device=device)
        else:
            valid = encoder_attention_mask.to(device=device, dtype=torch.bool)
        counts = valid.sum(dim=1).tolist()
        if min(counts) == 0:
            raise ValueError("every caption needs at least one real token")
        cap_lens = [n + (-n) % SEQ_MULTI_OF for n in counts]
        max_cap = max(cap_lens)
        cap_all = self.cap_embedder(encoder_hidden_states.to(device, dtype))
        cap_rows = []
        for i in range(batch):
            row = cap_all[i][valid[i]]
            pad = self.cap_pad_token.to(row.dtype).expand(cap_lens[i] - counts[i], -1)
            cap_rows.append(torch.cat([row, pad], dim=0))
        cap = self._stack(cap_rows, max_cap)
        cap_ids = torch.zeros(batch, max_cap, 3, dtype=torch.int32, device=device)
        for i in range(batch):
            cap_ids[i, :cap_lens[i], 0] = torch.arange(1, cap_lens[i] + 1, dtype=torch.int32, device=device)
        cap_freqs = self.rope_embedder(cap_ids)
        cap_mask = None
        if min(cap_lens) < max_cap:
            cap_mask = torch.arange(max_cap, device=device)[None] < torch.tensor(cap_lens, device=device)[:, None]
        for block in self.context_refiner:
            cap = self._run(block, cap, cap_mask, cap_freqs)

        # 3. image tokens (ph, pw, c features), padded to a multiple of 32 with x_pad_token; ids (cap_len + 1, y, x)
        n_img = height * width
        pad_img = (-n_img) % SEQ_MULTI_OF
        n_pad = n_img + pad_img
        x = self.all_x_embedder[PATCH_KEY](_house_to_tokens(hidden_states).to(dtype))
        if pad_img:
            x = torch.cat([x, self.x_pad_token.to(x.dtype).expand(batch, pad_img, -1)], dim=1)
        ys, xs = torch.meshgrid(torch.arange(height, dtype=torch.int32, device=device),
                                torch.arange(width, dtype=torch.int32, device=device), indexing="ij")
        grid = torch.stack([torch.zeros_like(ys), ys, xs], dim=-1).reshape(n_img, 3)
        img_ids = torch.zeros(batch, n_pad, 3, dtype=torch.int32, device=device)
        img_ids[:, :n_img] = grid
        img_ids[:, :n_img, 0] = torch.tensor(cap_lens, dtype=torch.int32, device=device)[:, None] + 1
        x_freqs = self.rope_embedder(img_ids)
        for block in self.noise_refiner:
            x = self._run(block, x, None, x_freqs, adaln_input)

        # 4. one stream per row: [image, caption], batch padding masked
        unified = torch.cat([x, cap], dim=1)
        unified_freqs = torch.cat([x_freqs, cap_freqs], dim=1)
        length = n_pad + max_cap
        for i in range(batch):  # a row's batch padding sits at the end: zeros, as diffusers' pad_sequence
            if cap_lens[i] < max_cap:
                unified[i, n_pad + cap_lens[i]:] = 0
                unified_freqs[i, n_pad + cap_lens[i]:] = 0
        attn_mask = None
        if cap_mask is not None:
            attn_mask = torch.cat([torch.ones(batch, n_pad, dtype=torch.bool, device=device), cap_mask], dim=1)
        taps = getattr(self, "_rz_taps", None)  # rulztrainer.alignment (training-only losses)
        if taps is not None and taps.armed:
            txt_valid = torch.arange(max_cap, device=device)[None] < torch.tensor(counts, device=device)[:, None]
            taps.set_layout(0, (0, n_img), (n_pad, max_cap), txt_valid)

        # 5. main blocks
        split = getattr(self, "split_devices", None)  # blocks may live on several devices (model split)
        for lid, block in enumerate(self.layers):
            if lid in skip_layers:
                continue
            if split:
                ldev = _module_device(block)
                if ldev is not None and unified.device != ldev:
                    unified, unified_freqs = unified.to(ldev), unified_freqs.to(ldev)
                    adaln_input = adaln_input.to(ldev)
                    attn_mask = attn_mask.to(ldev) if attn_mask is not None else None
            unified = self._run(block, unified, attn_mask, unified_freqs, adaln_input)
            if taps is not None and taps.armed:
                taps.record_hidden(lid, unified[:, :n_img])

        # 6. final layer on the image tokens; Z-Image predicts x0 - noise: negate to the trainer's noise - x0
        img = unified[:, :n_img]
        final = self.all_final_layer[PATCH_KEY]
        if split:
            fdev = _module_device(final)
            if fdev is not None and img.device != fdev:
                img, adaln_input = img.to(fdev), adaln_input.to(fdev)
        out = final(img, adaln_input)
        return (-_tokens_to_house(out, height, width)).to(device)
