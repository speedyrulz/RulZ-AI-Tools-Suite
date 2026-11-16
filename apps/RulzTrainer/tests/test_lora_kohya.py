import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from rulztrainer.lora import export_state_dict, file_format, inject_lora, load_lora_weights, read_metadata, save_lora


NAMES = [
    "unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q",
    "unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0",
    "unet.down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj",
    "unet.down_blocks.1.attentions.0.proj_in",
]


def make_model():
    model = nn.Module()
    for name in NAMES:
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            if part not in parent._modules:
                parent.add_module(part, nn.Module())
            parent = parent._modules[part]
        parent.add_module(parts[-1], nn.Linear(4, 4))
    return model


def inject(model, **kwargs):
    return inject_lora(model, rank=2, alpha=3, targets=[], names=NAMES, **kwargs)


def test_kohya_roundtrip_and_existing_format(tmp_path):
    torch.manual_seed(42)
    model = make_model()
    layers = inject(model)
    for layer in layers.values():
        with torch.no_grad():
            layer.lora_down.weight.copy_(torch.randn_like(layer.lora_down.weight))
            layer.lora_up.weight.copy_(torch.randn_like(layer.lora_up.weight))
    path = tmp_path / "kohya.safetensors"
    save_lora(layers, path, metadata={"modelspec.architecture": "sdxl"}, dtype=torch.float32,
              fmt="kohya", ss_base_model_version="sdxl_base_v1-0")
    state = load_file(str(path))
    expected = {"lora_unet_" + n.removeprefix("unet.").replace(".", "_") + suffix
                for n in NAMES for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha")}
    assert set(state) == expected
    x = torch.randn(3, 4)
    for name, layer in layers.items():
        stem = "lora_unet_" + name.removeprefix("unet.").replace(".", "_")
        alpha = state[stem + ".alpha"]
        assert alpha.ndim == 0 and alpha.item() == 3
        weight = layer.base.weight + alpha / 2 * (state[stem + ".lora_up.weight"] @ state[stem + ".lora_down.weight"])
        torch.testing.assert_close(layer(x), nn.functional.linear(x, weight, layer.base.bias))
    restored = inject(make_model())
    assert load_lora_weights(restored, path) == len(NAMES)
    for name in NAMES:
        for part in ("lora_down", "lora_up"):
            torch.testing.assert_close(getattr(restored[name], part).weight, getattr(layers[name], part).weight)
        assert restored[name].alpha == 3 and restored[name].scale == 1.5
    assert file_format(path) == "kohya"
    meta = read_metadata(path)
    assert meta["ss_network_module"] == "networks.lora"
    assert meta["ss_network_dim"] == "2" and float(meta["ss_network_alpha"]) == 3
    assert meta["ss_base_model_version"] == "sdxl_base_v1-0"
    assert meta["modelspec.architecture"] == "sdxl"
    comfy = export_state_dict(layers, torch.float32, "comfy")
    assert set(comfy) == {"diffusion_model." + n + suffix for n in NAMES
                          for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha")}
    for name, layer in layers.items():
        torch.testing.assert_close(comfy["diffusion_model." + name + ".lora_up.weight"], layer.lora_up.weight)
        torch.testing.assert_close(comfy["diffusion_model." + name + ".lora_down.weight"], layer.lora_down.weight)


@pytest.mark.parametrize("variant", ["dora", "multiplier", "scale"])
def test_kohya_rejects_unrepresentable_variants(tmp_path, variant):
    layers = inject(make_model(), dora=variant == "dora")
    first = next(iter(layers.values()))
    if variant == "multiplier":
        first.multiplier = 0.5
    if variant == "scale":
        first.scale = 2.0
    with pytest.raises(ValueError, match="Kohya"):
        save_lora(layers, tmp_path / "bad.safetensors", fmt="kohya")
