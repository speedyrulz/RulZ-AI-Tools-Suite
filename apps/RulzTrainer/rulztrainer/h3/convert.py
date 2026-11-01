"""MiniMax H3 transformer state-dict layouts and their conversion to the port's (the original checkpoint's) names.

Three layouts exist (key names read from the MiniMaxAI/MiniMax-H3 index files and Comfy-Org's safetensors headers):

  original   FL2VA/transformer/ and Ref2VA/transformer/ of MiniMaxAI/MiniMax-H3 (config "MiniMaxH3DiTModel") and every
             Comfy-Org single file: blocks.N.attn.qkv_proj, mlp.fc1 / fc2, adaln_proj.linear, token_refiner.blocks.N,
             video_patch_proj, condition_proj, time_embedder.proj_in / proj_out, final_layer.*, rope.inv_freq.
             The fused qkv rows are [q; k; v] stacked in Comfy-Org's files ("the reference model's post-load stack")
             and per-head interleaved ([head0: q k v, head1: ...]) in MiniMaxAI's own shards (diffusers'
             convert_minimax_h3_transformer_checkpoint_to_diffusers documents both: the shapes cannot tell them apart).
             fc1 rows are [gate; value]. Pruned files add adaln_t_table and drop time_embedder.
  diffusers  transformer/ and transformer_ref/ of MiniMaxAI/MiniMax-H3 (MiniMaxH3Transformer3DModel):
             transformer_blocks.N.attn.to_q / to_k / to_v / to_out.0, norm_q / norm_k, ff.net.0.proj ([value; gate],
             diffusers' SwiGLU order) / ff.net.2, token_refiner.refiner_blocks.N, proj_in, audio_proj_in,
             context_embedder, time_embedder.linear_1 / linear_2, norm_out.norm / linear, proj_out, audio_proj_out.
  port       the original names with [q; k; v] stacked (rulztrainer.h3.model), so Comfy-Org's files load as they are.

The mappings are exact inverses of diffusers' own converter, which the tests run against them."""
from __future__ import annotations

import re

import torch

from .. import quant as Q

DIT_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
# a Comfy-Org / original H3 diffusion model (ComfyUI's model_detection keys on the first two)
MARKERS = ("video_patch_proj.weight", "audio_patch_proj.weight", "condition_proj.weight")

_TO_PORT = (
    (r"^transformer_blocks\.(\d+)\.", r"blocks.\1."),
    (r"^token_refiner\.refiner_blocks\.(\d+)\.", r"token_refiner.blocks.\1."),
    (r"\.attn\.norm_q\.", ".attn.q_norm."),
    (r"\.attn\.norm_k\.", ".attn.k_norm."),
    (r"\.attn\.to_out\.0\.", ".attn.out_proj."),
    (r"\.ff\.net\.0\.proj\.", ".mlp.fc1."),
    (r"\.ff\.net\.2\.", ".mlp.fc2."),
    (r"^proj_in\.", "video_patch_proj."),
    (r"^audio_proj_in\.", "audio_patch_proj."),
    (r"^context_embedder\.", "condition_proj."),
    (r"^time_embedder\.linear_1\.", "time_embedder.proj_in."),
    (r"^time_embedder\.linear_2\.", "time_embedder.proj_out."),
    (r"^norm_out\.norm\.", "final_layer.norm."),
    (r"^norm_out\.linear\.", "final_layer.adaln_proj.linear."),
    (r"^proj_out\.", "final_layer.video_out."),
    (r"^audio_proj_out\.", "final_layer.audio_out."),
)


def is_h3_state(keys) -> bool:
    keys = set(keys)
    return all(m in keys for m in MARKERS)


def is_diffusers_state(keys) -> bool:
    keys = set(keys)
    return "proj_in.weight" in keys and "audio_proj_in.weight" in keys and "context_embedder.weight" in keys


def swap_halves(w):
    """[a; b] -> [b; a] by rows: diffusers' [value; gate] <-> the checkpoint's [gate; value] (a permutation of output
    rows, so an int8 weight with its per-row scales swaps as it is)."""
    if isinstance(w, Q.QuantWeight):
        if w.fmt == "nvfp4":
            a, b = w.dequantize(torch.float32).chunk(2, 0)
            return Q.quantize_weight(torch.cat([b, a]), "nvfp4")
        qa, qb = w.qdata.chunk(2, 0)
        sa, sb = w.scale.chunk(2, 0)
        return Q.QuantWeight(w.fmt, torch.cat([qb, qa]), torch.cat([sb, sa]), None, w.orig_shape, w.group)
    a, b = w.chunk(2, 0)
    return torch.cat([b, a])


def deinterleave_qkv(w: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """MiniMaxAI's per-head interleaved fused qkv rows [head: q k v] -> [q_all; k_all; v_all]."""
    return w.reshape(heads, 3, head_dim, *w.shape[1:]).transpose(0, 1).reshape(w.shape)


def interleave_qkv(w: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """The inverse of deinterleave_qkv."""
    return w.reshape(3, heads, head_dim, *w.shape[1:]).transpose(0, 1).reshape(w.shape)


def diffusers_to_port(state: dict) -> dict:
    """diffusers' MiniMaxH3Transformer3DModel state -> the port's names (consumes `state`): to_q / to_k / to_v fused
    into qkv_proj ([q; k; v]), ff.net.0.proj's halves swapped into fc1's [gate; value]."""
    out, qkv = {}, {}
    for k in list(state):
        v = state.pop(k)
        m = re.match(r"^(.*\.attn)\.to_([qkv])\.weight$", k)
        if m:
            qkv.setdefault(m.group(1), {})[m.group(2)] = v
            continue
        nk = k
        for pat, rep in _TO_PORT:
            nk = re.sub(pat, rep, nk)
        if nk.endswith(".mlp.fc1.weight"):
            v = swap_halves(v)
        out[nk] = v
    for stem, parts in qkv.items():
        nk = stem
        for pat, rep in _TO_PORT:
            nk = re.sub(pat, rep, nk)
        if set(parts) != {"q", "k", "v"}:
            raise ValueError(f"{stem}: to_q / to_k / to_v incomplete")
        if any(isinstance(p, Q.QuantWeight) for p in parts.values()):
            raise ValueError("a quantized diffusers-layout MiniMax H3 transformer is not supported (load bf16)")
        out[f"{nk}.qkv_proj.weight"] = torch.cat([parts["q"], parts["k"], parts["v"]])
    out.pop("rope.inv_freq", None)
    return out


def port_to_diffusers(state: dict) -> dict:
    """The port's state -> diffusers' names (the inverse of diffusers_to_port; tests and exports)."""
    back = [(r"^blocks\.(\d+)\.", r"transformer_blocks.\1."),
            (r"^token_refiner\.blocks\.(\d+)\.", r"token_refiner.refiner_blocks.\1."),
            (r"\.attn\.q_norm\.", ".attn.norm_q."), (r"\.attn\.k_norm\.", ".attn.norm_k."),
            (r"\.attn\.out_proj\.", ".attn.to_out.0."), (r"\.mlp\.fc1\.", ".ff.net.0.proj."),
            (r"\.mlp\.fc2\.", ".ff.net.2."), (r"^video_patch_proj\.", "proj_in."),
            (r"^audio_patch_proj\.", "audio_proj_in."), (r"^condition_proj\.", "context_embedder."),
            (r"^time_embedder\.proj_in\.", "time_embedder.linear_1."),
            (r"^time_embedder\.proj_out\.", "time_embedder.linear_2."), (r"^final_layer\.norm\.", "norm_out.norm."),
            (r"^final_layer\.adaln_proj\.linear\.", "norm_out.linear."), (r"^final_layer\.video_out\.", "proj_out."),
            (r"^final_layer\.audio_out\.", "audio_proj_out.")]
    out = {}
    for k, v in state.items():
        nk = k
        for pat, rep in back:
            nk = re.sub(pat, rep, nk)
        if nk.endswith(".attn.qkv_proj.weight"):
            q, kk, vv = v.chunk(3, 0)
            stem = nk[: -len(".qkv_proj.weight")]
            out[f"{stem}.to_q.weight"], out[f"{stem}.to_k.weight"], out[f"{stem}.to_v.weight"] = q, kk, vv
            continue
        if nk.endswith(".ff.net.0.proj.weight"):
            v = swap_halves(v)
        out[nk] = v
    return out


def original_to_port(state: dict, interleaved: bool, heads: int, head_dim: int) -> dict:
    """An original-layout state (Comfy-Org file or MiniMaxAI shards) -> the port's: prefixes stripped, MiniMaxAI's
    interleaved qkv de-interleaved (bf16 only: a quantized interleaved file does not exist), rope.inv_freq dropped
    (recomputed from the config)."""
    out = {}
    for k in list(state):
        v = state.pop(k)
        for p in DIT_PREFIXES:
            if k.startswith(p):
                k = k[len(p):]
                break
        if k == "rope.inv_freq" or k.endswith("comfy_attention.config"):
            continue
        if interleaved and k.endswith(".attn.qkv_proj.weight"):
            if isinstance(v, Q.QuantWeight):
                raise ValueError("an interleaved-qkv quantized MiniMax H3 file is not supported")
            v = deinterleave_qkv(v, heads, head_dim)
        out[k] = v
    return out


def _count(keys, pattern: str) -> int:
    n = 0
    while pattern.format(n) in keys:
        n += 1
    return n


def config_from_state(state: dict) -> dict:
    """The port's config from an original-layout (maybe quantized, maybe pruned) state, as ComfyUI's model_detection
    reads it."""
    def shape(k):
        v = state[k]
        return tuple(v.orig_shape) if isinstance(v, Q.QuantWeight) else tuple(v.shape)

    keys = set(state)
    head_dim = shape("blocks.0.attn.q_norm.weight")[0]
    hidden = shape("video_patch_proj.weight")[0]
    cfg = dict(hidden_size=hidden, num_layers=_count(keys, "blocks.{}.norm1.weight"),
               token_refiner_num_layers=_count(keys, "token_refiner.blocks.{}.norm1.weight"),
               attention_head_dim=head_dim, num_attention_heads=shape("blocks.0.attn.qkv_proj.weight")[0] // (3 * head_dim),
               ffn_hidden_size=shape("blocks.0.mlp.fc1.weight")[0] // 2,
               latents_dim=shape("final_layer.video_out.weight")[0] // 4,
               audio_latents_dim=shape("final_layer.audio_out.weight")[0],
               text_dim=shape("condition_proj.weight")[1])
    if "adaln_t_table" in keys:
        grid, t_dim = shape("adaln_t_table")
        cfg.update(adaln_curve_grid=grid, time_embed_dim=t_dim)
    else:
        cfg.update(timestep_input_dim=shape("time_embedder.proj_in.weight")[1],
                   time_embed_hidden_size=shape("time_embedder.proj_in.weight")[0],
                   time_embed_dim=shape("time_embedder.proj_out.weight")[0])
    return cfg


def diffusers_config_to_port(config: dict) -> dict:
    """diffusers' MiniMaxH3Transformer3DModel config.json -> the port's config."""
    return dict(hidden_size=config.get("hidden_size", 5376), num_layers=config.get("num_layers", 50),
                token_refiner_num_layers=config.get("num_refiner_layers", 2),
                num_attention_heads=config.get("num_attention_heads", 56),
                attention_head_dim=config.get("attention_head_dim", 128), ffn_hidden_size=config.get("ffn_dim", 14336),
                latents_dim=config.get("in_channels", 24), audio_latents_dim=config.get("audio_in_channels", 32),
                patch_size=list(config.get("patch_size", (1, 2, 2))), text_dim=config.get("text_dim", 5120),
                timestep_input_dim=config.get("freq_dim", 256),
                time_embed_hidden_size=config.get("time_embed_hidden_dim", 5376),
                time_embed_dim=config.get("time_embed_dim", 2688), rope_inv_freq_len=config.get("rope_freq_dim", 16),
                rope_theta=config.get("rope_theta", 10000.0), norm_eps=config.get("norm_eps", 1e-5),
                qk_norm_eps=config.get("qk_norm_eps", 1e-5), final_norm_eps=config.get("final_norm_eps", 1e-5))
