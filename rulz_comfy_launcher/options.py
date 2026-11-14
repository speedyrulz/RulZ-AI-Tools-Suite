"""The catalog of ComfyUI command line options.

The list comes from ComfyUI itself (``scripts/introspect_args.py`` reads its
argparse parser), so it always matches the installed version. A copy taken from
ComfyUI 0.45.0 ships with the app as a fallback. Categories are added here,
because argparse does not have them.

No Qt imports here, so this module can be unit-tested on its own.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Options the launcher sets itself (or that make no sense from a launcher).
# They are hidden from the generic option list.
MANAGED_OPTIONS = {
    "--port": "set in the Network section",
    "--listen": "set in the Network section",
    "--cuda-device": "set in the GPUs section",
    "--cpu": "set in the GPUs section",
    "--auto-launch": "use 'Open the browser when ready' instead",
    "--disable-auto-launch": "use 'Open the browser when ready' instead",
    "--windows-standalone-build": "not used with WSL",
    "--quick-test-for-ci": "not used from a launcher",
    "--list-feature-flags": "prints a list and exits",
}

CATEGORY_ORDER = [
    "Memory and VRAM",
    "Performance",
    "Attention",
    "Precision",
    "Devices",
    "Folders",
    "Previews",
    "Custom nodes and manager",
    "Network",
    "Logging and debugging",
    "Assets and API",
    "Other",
]
UNKNOWN_CATEGORY = "Not in this ComfyUI version"

_CATEGORY_BY_NAME = {
    # Memory and VRAM
    "--gpu-only": "Memory and VRAM", "--highvram": "Memory and VRAM",
    "--lowvram": "Memory and VRAM", "--novram": "Memory and VRAM",
    "--reserve-vram": "Memory and VRAM", "--vram-headroom": "Memory and VRAM",
    "--disable-nvml-pressure": "Memory and VRAM", "--async-offload": "Memory and VRAM",
    "--disable-async-offload": "Memory and VRAM", "--disable-dynamic-vram": "Memory and VRAM",
    "--enable-dynamic-vram": "Memory and VRAM", "--fast-disk": "Memory and VRAM",
    "--disable-fast-disk": "Memory and VRAM", "--disable-smart-memory": "Memory and VRAM",
    "--disable-pinned-memory": "Memory and VRAM", "--mmap-torch-files": "Memory and VRAM",
    "--disable-mmap": "Memory and VRAM", "--cache-ram": "Memory and VRAM",
    "--cache-classic": "Memory and VRAM", "--cache-lru": "Memory and VRAM",
    "--cache-none": "Memory and VRAM", "--high-ram": "Memory and VRAM",
    # Performance
    "--fast": "Performance", "--enable-triton-backend": "Performance",
    "--disable-triton-backend": "Performance", "--disable-cuda-graphs": "Performance",
    "--disable-comfy-compiler": "Performance", "--assert-graph-breaks": "Performance",
    "--force-non-blocking": "Performance", "--deterministic": "Performance",
    # Attention
    "--use-split-cross-attention": "Attention", "--use-quad-cross-attention": "Attention",
    "--use-pytorch-cross-attention": "Attention", "--use-sage-attention": "Attention",
    "--use-flash-attention": "Attention", "--use-ck-attention": "Attention",
    "--disable-xformers": "Attention", "--force-upcast-attention": "Attention",
    "--dont-upcast-attention": "Attention",
    # Precision
    "--force-fp32": "Precision", "--force-fp16": "Precision", "--cpu-vae": "Precision",
    "--fp16-intermediates": "Precision", "--force-channels-last": "Precision",
    "--supports-fp8-compute": "Precision",
    # Devices
    "--default-device": "Devices", "--oneapi-device-selector": "Devices",
    "--cuda-malloc": "Devices", "--disable-cuda-malloc": "Devices",
    "--cuda-device": "Devices", "--cpu": "Devices",
    # Folders
    "--base-directory": "Folders", "--extra-model-paths-config": "Folders",
    "--output-directory": "Folders", "--temp-directory": "Folders",
    "--input-directory": "Folders", "--previews-directory": "Folders",
    "--user-directory": "Folders", "--models-directory": "Folders",
    "--database-url": "Folders", "--front-end-root": "Folders",
    "--front-end-version": "Folders",
    # Previews
    "--preview-method": "Previews", "--preview-size": "Previews",
    # Custom nodes and manager
    "--enable-manager": "Custom nodes and manager", "--enable-agent": "Custom nodes and manager",
    "--disable-manager-ui": "Custom nodes and manager",
    "--enable-manager-legacy-ui": "Custom nodes and manager",
    "--disable-all-custom-nodes": "Custom nodes and manager",
    "--whitelist-custom-nodes": "Custom nodes and manager",
    "--disabled-nodes-config": "Custom nodes and manager",
    "--disable-partner-nodes": "Custom nodes and manager", "--offline": "Custom nodes and manager",
    "--disable-api-nodes": "Custom nodes and manager",
    # Network
    "--listen": "Network", "--port": "Network", "--tls-keyfile": "Network",
    "--tls-certfile": "Network", "--enable-cors-header": "Network",
    "--max-upload-size": "Network", "--enable-compress-response-body": "Network",
    "--multi-user": "Network",
    # Logging and debugging
    "--verbose": "Logging and debugging", "--log-stdout": "Logging and debugging",
    "--dont-print-server": "Logging and debugging",
    "--disable-product-tips": "Logging and debugging", "--debug-hang": "Logging and debugging",
    "--disable-metadata": "Logging and debugging",
    "--default-hashing-function": "Logging and debugging",
    # Assets and API
    "--enable-assets": "Assets and API", "--disable-assets": "Assets and API",
    "--enable-asset-hashing": "Assets and API", "--comfy-api-base": "Assets and API",
    "--feature-flag": "Assets and API",
}


def categorize(name: str) -> str:
    """Category for an option name. Options added by newer ComfyUI versions are
    placed by keywords in their name, and land in "Other" when nothing fits."""
    known = _CATEGORY_BY_NAME.get(name)
    if known:
        return known
    n = name.lower()
    if n.endswith(("-unet", "-vae", "-text-enc")) or "fp16" in n or "fp32" in n or "bf16" in n:
        return "Precision"
    if "attention" in n or "xformers" in n:
        return "Attention"
    if "vram" in n or "cache" in n or "memory" in n or "mmap" in n or "offload" in n:
        return "Memory and VRAM"
    if "directory" in n or "path" in n or n.endswith("-root"):
        return "Folders"
    if "manager" in n or "nodes" in n:
        return "Custom nodes and manager"
    if "preview" in n:
        return "Previews"
    if "device" in n or "cuda" in n:
        return "Devices"
    if "tls" in n or "cors" in n:
        return "Network"
    if "log" in n or "verbose" in n or "debug" in n:
        return "Logging and debugging"
    return "Other"


@dataclass(frozen=True)
class OptionSpec:
    name: str                       # "--vram-headroom"
    kind: str = "flag"              # flag | value | optional | multi
    type: str = "str"               # str | int | float | path
    choices: tuple[str, ...] | None = None
    default: object = None
    const: object = None
    metavar: str = ""
    help: str = ""
    repeat: bool = False            # argparse "append": the flag may be given several times
    min_values: int = 0             # for kind "multi"
    category: str = "Other"
    known: bool = True              # False for options a profile has but ComfyUI does not

    @staticmethod
    def from_dict(data: dict) -> "OptionSpec":
        name = str(data["name"])
        kind = data.get("kind", "flag")
        if kind not in ("flag", "value", "optional", "multi"):
            kind = "flag"
        type_ = data.get("type", "str")
        if type_ not in ("str", "int", "float", "path"):
            type_ = "str"
        choices = data.get("choices")
        try:
            min_values = int(data.get("min_values") or 0)
        except (TypeError, ValueError):
            min_values = 0
        return OptionSpec(
            name=name,
            kind=kind,
            type=type_,
            choices=tuple(str(c) for c in choices) if choices else None,
            default=data.get("default"),
            const=data.get("const"),
            metavar=str(data.get("metavar") or ""),
            help=str(data.get("help") or ""),
            repeat=bool(data.get("repeat", False)),
            min_values=min_values,
            category=categorize(name),
        )

    def default_text(self) -> str:
        """The default as short text for placeholders ("" when there is none)."""
        d = self.default
        if d is None or d is False or d == [] or d == "":
            return ""
        if isinstance(d, list):
            return " ".join(str(x) for x in d)
        if isinstance(d, float) and d.is_integer():
            return str(int(d))
        return str(d)


@dataclass
class Catalog:
    options: list[OptionSpec] = field(default_factory=list)
    groups: list[list[str]] = field(default_factory=list)   # mutually exclusive sets
    version: str = ""
    source: str = "built-in"

    def __post_init__(self):
        self._by_name = {o.name: o for o in self.options}
        self._group_of: dict[str, int] = {}
        for index, names in enumerate(self.groups):
            for name in names:
                self._group_of[name] = index

    @staticmethod
    def from_data(data: dict, source: str = "built-in") -> "Catalog":
        options = []
        seen = set()
        for raw in data.get("options") or []:
            if not isinstance(raw, dict) or not raw.get("name"):
                continue
            spec = OptionSpec.from_dict(raw)
            if spec.name in seen:
                continue
            seen.add(spec.name)
            options.append(spec)
        groups = []
        for names in data.get("groups") or []:
            if isinstance(names, list) and len(names) > 1:
                groups.append([str(n) for n in names])
        return Catalog(options=options, groups=groups,
                       version=str(data.get("version") or ""), source=source)

    def get(self, name: str) -> OptionSpec | None:
        return self._by_name.get(name)

    def visible(self) -> list[OptionSpec]:
        """Options shown in the generic option list."""
        return [o for o in self.options if o.name not in MANAGED_OPTIONS]

    def exclusive_with(self, name: str) -> list[str]:
        """The other options in the same mutually exclusive group."""
        index = self._group_of.get(name)
        if index is None:
            return []
        return [n for n in self.groups[index] if n != name]

    def group_index(self, name: str) -> int | None:
        return self._group_of.get(name)

    def label(self) -> str:
        if self.version:
            return f"ComfyUI {self.version} ({self.source})"
        return self.source


def unknown_spec(name: str, has_value: bool) -> OptionSpec:
    """A stand-in for an option that a profile uses but the catalog does not list."""
    return OptionSpec(name=name, kind="optional" if has_value else "flag",
                      help="This option is not offered by the ComfyUI version the list was read from. "
                           "ComfyUI will refuse to start if it does not know it.",
                      category=UNKNOWN_CATEGORY, known=False)


def builtin_catalog() -> Catalog:
    path = Path(__file__).with_name("builtin_catalog.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    return Catalog.from_data(data, source="built-in list")


def parse_introspection_output(text: str) -> dict | None:
    """Pick the JSON payload out of the introspection script's output."""
    for line in text.splitlines():
        if line.startswith("RCL_JSON:"):
            try:
                data = json.loads(line[len("RCL_JSON:"):])
            except ValueError:
                return None
            if isinstance(data, dict) and isinstance(data.get("options"), list) and data["options"]:
                return data
            return None
    return None
