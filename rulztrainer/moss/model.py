"""MOSS-Image dense DiT, vendored from OpenMOSS pipeline_moss_image.py (Apache-2.0).

State-dict keys match the released checkpoints and the ComfyUI plugin, so a LoRA
trained against this module's Linear layers maps 1:1 onto ComfyUI's
"diffusion_model.<key>" names.

Changes from upstream: optional gradient checkpointing per block; config is a plain
dict instead of a diffusers ConfigMixin.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _wdtype(m: nn.Module) -> torch.dtype:
    """Compute dtype of a Linear-like module (plain, LoRA-wrapped or quantized)."""
    d = getattr(m, "compute_dtype", None)
    return d if d is not None else m.weight.dtype


def _module_device(m: nn.Module):
    p = next(m.parameters(), None)
    if p is not None:
        return p.device
    b = next(m.buffers(), None)
    return b.device if b is not None else None


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _attention(q, k, v, rope, heads):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), rope).transpose(1, 2)
    k = _rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v).transpose(1, 2).flatten(2)


class TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope):
        x = self.proj(x.to(_wdtype(self.proj)))
        for block in self.blocks:
            x = block(x, rope)
        return x


class Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.q_norm, self.k_norm = RMSNorm(dim // heads), RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = _wdtype(self.proj["q"])
        y = y.to(dtype)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](_attention(q, k, self.proj["v"](y), rope, self.heads)) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        return x + self.proj["ffn_down"](_situ(hidden, gate)) * gate_f[:, None]


class RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", self._default_bands(torch.device("cpu")))

    def _default_bands(self, device):
        exponent = torch.arange(self.num_bands, device="cpu", dtype=torch.float32) / self.num_bands
        return (1 / (self.base ** exponent)).to(device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.bands = self._default_bands(self.bands.device)
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.bands = self._default_bands(self.bands.device)

    def forward(self, height, width, text_length, device):
        if self.bands.dtype != torch.float32 or self.bands.device != device:
            self.bands = self._default_bands(device)
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class MOSSImageTransformer(nn.Module):
    """MOSS-Image DiT. hidden_states: [B, 128, H/16, W/16] normalized Flux2 latents."""

    CONFIG_KEYS = ("dim", "ffn_dim", "num_heads", "num_layers", "text_dim", "freq_dim",
                   "latent_channels", "rope_base")

    def __init__(self, dim=3072, ffn_dim=8192, num_heads=24, num_layers=42,
                 text_dim=4096, freq_dim=256, latent_channels=128, rope_base=10000.0):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError("Require dim divisible by heads, head_dim divisible by 8, and even freq_dim.")
        self.config = dict(dim=dim, ffn_dim=ffn_dim, num_heads=num_heads, num_layers=num_layers,
                           text_dim=text_dim, freq_dim=freq_dim, latent_channels=latent_channels,
                           rope_base=rope_base)
        self.latent_channels = latent_channels
        self.freq_dim = freq_dim
        self.gradient_checkpointing = False
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    @classmethod
    def from_config(cls, config: dict) -> "MOSSImageTransformer":
        return cls(**{k: config[k] for k in cls.CONFIG_KEYS if k in config})

    def _run_layer(self, layer, x, rope, mod):
        if (self.gradient_checkpointing and self.training and torch.is_grad_enabled()
                and not getattr(layer, "keep_activations", False)):  # see rulztrainer.checkpointing
            return checkpoint(layer, x, rope, mod, use_reentrant=False)
        return layer(x, rope, mod)

    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask=None,
                skip_layers=None):
        skip_layers = frozenset(skip_layers or ())
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.latent_channels:
            raise ValueError("hidden_states must be [B, latent_channels, H, W].")
        batch, _, height, width = hidden_states.shape
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1)
        times = times.expand(batch)
        half = self.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(_wdtype(self.time_embedding[0]))).float()
        mod = self.shared_block_modulation(time.to(_wdtype(self.shared_block_modulation))).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(_wdtype(self.video_head_modulation))).view(batch, 2, -1)
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index:index + 1, mask[index].bool()]
            if context.shape[1] == 0:
                raise ValueError("Each sample needs at least one valid text token.")
            video = hidden_states[index:index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(_wdtype(self.video_embedder)).contiguous())
            rope = self.rope(height, width, context.shape[1], hidden_states.device).to(video.dtype)
            text = self.text_embedder(context, rope[:, height * width:])
            x = torch.cat((video, text), 1)
            mod_i = mod[index:index + 1]
            rope_i = rope
            split = getattr(self, "split_devices", None)  # blocks may live on several devices (model split)
            for layer_index, layer in enumerate(self.layers):
                if layer_index in skip_layers:
                    continue
                if split:
                    ldev = _module_device(layer)
                    if ldev is not None and x.device != ldev:
                        x, rope_i, mod_i = x.to(ldev), rope_i.to(ldev), mod_i.to(ldev)
                x = self._run_layer(layer, x, rope_i, mod_i)
            head_i = head[index:index + 1]
            if split:
                fdev = self.final_norm_video.weight.device
                if x.device != fdev:
                    x = x.to(fdev)
                if head_i.device != fdev:
                    head_i = head_i.to(fdev)
            x = self.final_norm_video(x[:, :height * width].to(self.final_norm_video.weight.dtype))
            x = x * (1 + head_i[:, 1, None]) + head_i[:, 0, None]
            x = self.final_linear_video(x.to(_wdtype(self.final_linear_video)))
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        return torch.cat(outputs).to(hidden_states.device)
