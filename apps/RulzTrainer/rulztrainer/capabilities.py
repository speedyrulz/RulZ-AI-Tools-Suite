"""What each model family cannot do: the one table behind both the run-start refusal (TrainConfig.validate_sdxl and
the video-frame check in TrainConfig.validate) and the WebUI (webui.py greys the setting out, filters its choices and
shows the reason as soon as the model is picked).

A rule names a TrainConfig field, the values the family accepts (Only) or refuses (Forbid), the value the WebUI puts
back when it has to reset the setting, and the reason. Nothing is imported from config.py (config.py imports this
module), so the table is plain data and holds no TrainConfig default except in `default`.

Settings that are rewritten and not refused (SDXL's max_token_length and timestep_sampling) stay in code
(TrainConfig.validate_sdxl)."""
from __future__ import annotations

from dataclasses import dataclass

# families that train on video clips (train_frames) and render samples and probes as clips (sample_frames,
# probe_frames). MiniMax H3 is the video + audio model, not MiniMax-H3-Image or P-H3-Image (stills only).
VIDEO_FAMILIES = ("pan2", "minimax_h3")
FRAME_KEYS = ("sample_frames", "probe_frames", "train_frames")

_NAMES = {"sdxl": "SDXL"}


def display_name(family: str) -> str:
    """The family as a person reads it (the model's label: "SDXL", "Krea 2"), the family key when it has none."""
    if family in _NAMES:
        return _NAMES[family]
    try:
        from .archs import arch_by_name

        return arch_by_name(family).label or family
    except Exception:  # noqa: BLE001 - an unknown family or an import problem must not break a message
        return family


def _same(a, b) -> bool:
    """a == b, where True is not 1, and strings compare without case (optimizer names)."""
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, str) and isinstance(b, str):
        return a.lower() == b.lower()
    try:
        return a == b
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True)
class Rule:
    """One setting of one family. `only`: the values it accepts; `forbid`: the values it refuses (use one of the two).
    `default`: what the WebUI resets the setting to (and what a missing or None value counts as). `when`: (field,
    values): the rule applies only while that other field holds one of them (the run-start check reads it; the WebUI
    ignores it and applies the rule always)."""

    key: str
    reason: str
    only: tuple = ()
    forbid: tuple = ()
    default: object = None
    when: tuple | None = None
    ignored: bool = False  # the run never reads the setting (harmless): the WebUI greys it out and resets it quietly,
    # and the run-start check lets any value through (a refusal is for a setting that would fail or silently differ)

    def allows(self, value) -> bool:
        if value is None:
            value = self.default
        if self.only:
            return any(_same(value, v) for v in self.only)
        return not any(_same(value, v) for v in self.forbid)

    def applies(self, get) -> bool:
        if self.when is None:
            return True
        other, values = self.when
        return any(_same(get(other), v) for v in values)

    def single(self, options=None) -> bool:
        """Only one value is left to choose (of `options`, when given): the WebUI disables the control."""
        if options is None:
            return True
        return sum(1 for o in options if self.allows(o)) <= 1


def _only(key, reason, *values, default=None, when=None, ignored=False) -> Rule:
    return Rule(key, reason, only=tuple(values), default=values[0] if default is None else default, when=when,
                ignored=ignored)


def _forbid(key, reason, *values, default) -> Rule:
    return Rule(key, reason, forbid=tuple(values), default=default)


# SDXL (a LoRA on the UNet): each rule is one refusal. "auto" settings that name nothing for SDXL (transformer_quant,
# compile_model, the downloads, the checkpointing's kept blocks) resolve to off; a fresh SDXL config trips none of them.
_SDXL = [
    _only("block_swap", "the UNet is not swapped block by block (its 2.6B weights fit a 12 GB card)", 0),
    _forbid("multi_gpu", "the UNet cannot be split across GPUs here (use off or data_parallel)", "model_split",
            default="off"),
    _only("activation_offload", "diffusers' checkpointing is all or nothing", False),
    _only("attn_localize", "it reads the transformer blocks' attention and states, which SDXL's UNet does not expose "
                           "that way", False),
    _only("id_align", "it reads the transformer blocks' attention and states, which SDXL's UNet does not expose "
                      "that way", False),
    _only("body_align", "it reads the transformer blocks' attention and states, which SDXL's UNet does not expose "
                        "that way", False),
    _forbid("optimizer", "keel_fb, keel3_fb and hull read the face-token taps and region gains of the transformer "
                         "blocks; use adamw, adamw8bit, adafactor, automagic3/4, keel, keel3, muon or normuon",
            "keel_fb", "keel3_fb", "hull", default="adamw8bit"),
    _only("keel_face_gain", "the face-token gain hooks the transformer blocks' outputs, which SDXL does not have in "
                            "that form", 0.0, when=("optimizer", ("keel", "keel_fb", "keel3", "keel3_fb"))),
    _only("training_maps", "its attention heat maps are read inside the transformer blocks, which SDXL's UNet does "
                           "not expose that way", False),
    _only("transformer_quant", "nvfp4 is not available for SDXL's UNet; use auto, none, int8 or int8_convrot (the "
                               "UNet's attention and feed-forward linears)", "auto", "none", "int8", "int8_convrot"),
    _only("sample_model_quant", "nvfp4 is not available for SDXL's UNet; use auto, none, int8 or int8_convrot",
          "auto", "none", "int8", "int8_convrot"),
    _forbid("compile_model", "true is not available (the UNet is not compiled; auto and false run it eagerly)", True,
            default="auto"),
    _only("loss_type", "its loss is the model's own epsilon / v-prediction MSE; use mse", "mse"),
    _forbid("loss_weighting", "sigma_sqrt is a flow-matching weight with no meaning for SDXL's epsilon / "
                              "v-prediction loss; use none or min_snr", "sigma_sqrt", default="none"),
    _only("finetune", "SDXL trains a LoRA only here", False),
    _only("network_type", "SDXL's LoRA file is kohya's plain lora_down / lora_up layout; use lora", "lora"),
    _only("dora", "SDXL's LoRA file is kohya's plain lora_down / lora_up layout; keep DoRA off", False),
    _only("train_text_embedder", "SDXL has no text projection inside its UNet, and its text encoders are not "
                                 "trained", False),
    # (turbo_lora is allowed: DMD2's LoRA, 8 steps at CFG 1 with an LCM sampler; "" = Euler at 30 steps, CFG 6)
    _only("sample_lora", "no such LoRA exists for SDXL here (auto and empty both mean none)", "auto", "",
          default="auto"),
    _only("training_adapter", "no such LoRA exists for SDXL here (auto and empty both mean none)", "auto", "",
          default="auto"),
    _only("checkpoint_keep_blocks", "SDXL's checkpointing is all or nothing (diffusers'); use -1 or 0", -1, 0),
]
# SDXL's transformer_quant and sample_model_quant: auto / none / int8 / int8_convrot pass (any case), nvfp4 and
# anything else is refused at the start, as the loader (sdxl/loading.py _validate_quant) would refuse it later

_FRAME_REASONS = {
    "sample_frames": "frames above 1 need a video model ({video}): this model renders stills",
    "probe_frames": "frames above 1 need a video model ({video}): this model renders stills",
    "train_frames": "frames above 1 (video clip training) need a video model ({video}): this model trains on stills",
}

# SDXL's CLIP encoders are loaded as they are (sdxl/loading.py never reads text_encoder_quant)
_SDXL.append(_only("text_encoder_quant", "SDXL's CLIP text encoders stay unquantized", "auto", ignored=True))

_TABLES: dict[str, list[Rule]] = {"sdxl": _SDXL}

# settings only some families read: (rules, the families that read them). Everywhere else they are ignored.
_ONLY_READ_BY: list[tuple[list[Rule], tuple]] = [
    ([_only("sample_shift", "this model's sampler sets its own noise schedule (only MOSS-Image reads the shift)", 5.0,
            ignored=True),
      _only("sample_skip_layer", "skip-layer guidance is MOSS-Image only", 22, ignored=True)], ("moss_image",)),
    ([_only("sdxl_prediction", "only SDXL has an epsilon / v-prediction choice", "auto", ignored=True),
      _only("sdxl_zero_terminal_snr", "only SDXL has a zero terminal SNR option", False, ignored=True)], ("sdxl",)),
    ([_only("train_video_mode", "only video models train on clips", "fps", ignored=True)], VIDEO_FAMILIES),
]


def rules_for(family: str) -> dict[str, Rule]:
    """The family's rules by TrainConfig field, in the order the run-start check reports them. Every family that
    is not a video model also holds the three frame settings at 1."""
    rules = {r.key: r for r in _TABLES.get(family, ())}
    for group, readers in _ONLY_READ_BY:
        if family not in readers:
            rules.update({r.key: r for r in group})
    if family not in VIDEO_FAMILIES:
        for key in FRAME_KEYS:
            rules[key] = _only(key, _FRAME_REASONS[key].format(video=", ".join(VIDEO_FAMILIES)), 1)
    return rules


def message(family: str, rule: Rule, value) -> str:
    """The refusal text: setting, the value, the model, why."""
    return f"{rule.key} {value!r}: not available for {display_name(family)}: {rule.reason}"


def violations(family: str, values, keys=None, ignore_when: bool = False) -> dict[str, str]:
    """{field: refusal text} for every setting of `values` (a TrainConfig or a dict) the family refuses, in table
    order. `keys` limits the check to those fields; a field missing from a dict is not checked."""
    if isinstance(values, dict):
        get = values.get
    else:
        def get(key, default=None):
            return getattr(values, key, default)

    out = {}
    for key, rule in rules_for(family).items():
        if keys is not None and key not in keys:
            continue
        if rule.ignored or (not ignore_when and not rule.applies(get)):
            continue
        if isinstance(values, dict) and key not in values:
            continue
        value = get(key)
        if not rule.allows(value):
            out[key] = message(family, rule, value)
    return out


def unsupported_keys(family: str) -> list[str]:
    """The settings the family restricts (the WebUI greys out or filters each)."""
    return list(rules_for(family))


def reason(family: str, key: str) -> str | None:
    """"Not available for <model>: <reason>" for the WebUI, None when the family has no rule for the setting."""
    rule = rules_for(family).get(key)
    return None if rule is None else f"Not available for {display_name(family)}: {rule.reason}"


def allowed_values(family: str, key: str, options):
    """`options` without the values the family refuses (all of them when it has no rule for the setting)."""
    rule = rules_for(family).get(key)
    return list(options) if rule is None else [o for o in options if rule.allows(o)]
