"""Turn a profile into a ComfyUI command line and into the bash script that runs it.

No Qt imports here, so this module can be unit-tested on its own.
"""
from __future__ import annotations

import hashlib
import re
import shlex
from dataclasses import dataclass

from .model import GPU_AUTO, GPU_CPU, GPU_SELECT, OptionState, Profile
from .options import MANAGED_OPTIONS, Catalog, OptionSpec, unknown_spec

LOG_PREFIX = "[launcher] "
# "start" is the kernel's start time of the process. Together with the pid it
# identifies one process for good, even if the pid number is used again later.
PID_PATTERN = re.compile(r"^\[launcher\] pid=(\d+)(?: start=(\d+))?\s*$")
# Reads that start time: field 22 of /proc/PID/stat, counted after the "(name)" part.
_START_TIME = "sed 's/.*) //' /proc/{pid}/stat 2>/dev/null | cut -d' ' -f20"
READY_PATTERN = re.compile(r"To see the GUI go to:")

# Exit codes the launch script uses for its own checks.
EXIT_NO_FOLDER = 91
EXIT_NO_MAIN = 92
EXIT_PORT_BUSY = 93
EXIT_NO_SMI = 94
EXIT_NO_GPU = 95
EXIT_NO_PYTHON = 96

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# The only forms that bash expands inside an environment value: $NAME and ${NAME}.
_ENV_REFERENCE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\})")
_HOST_CHARS = re.compile(r"^[0-9A-Za-z.:-]+$")
_WILDCARD_HOSTS = {"", "0.0.0.0", "::", "*"}


@dataclass(frozen=True)
class Issue:
    level: str      # "error" stops a launch, "warning" asks first
    message: str


# --------------------------------------------------------------------------
# quoting
# --------------------------------------------------------------------------
def quote_path(path: str) -> str:
    """Quote a Linux path for bash, keeping a leading ~ working."""
    path = path.strip()
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        rest = path[2:]
        return '"$HOME"/' + (shlex.quote(rest) if rest else "")
    return shlex.quote(path)


def quote_env_value(value: str) -> str:
    """Quote an environment value for ``export NAME="..."``.

    ``$NAME`` and ``${NAME}`` stay live so values such as ``$CUDA_HOME/bin:$PATH``
    work. Everything else is literal: quotes, backticks, backslashes, command
    substitution ``$(...)`` and any other ``${...}`` form are escaped, so no
    value can break the script. ``\\$`` gives a literal dollar.
    """
    out = []
    i = 0
    n = len(value)
    while i < n:
        ch = value[i]
        nxt = value[i + 1] if i + 1 < n else ""
        if ch == "\\":
            if nxt == "$":
                out.append("\\$")
                i += 2
                continue
            out.append("\\\\")
        elif ch in '"`':
            out.append("\\" + ch)
        elif ch == "$":
            live = _ENV_REFERENCE.match(value, i)
            if live:
                out.append(live.group(0))
                i = live.end()
                continue
            out.append("\\$")
        else:
            out.append(ch)
        i += 1
    return '"' + "".join(out) + '"'


def split_words(text: str) -> list[str]:
    """Split like a shell does. Raises ValueError on an unclosed quote."""
    return shlex.split(text, posix=True)


def _safe_split(text: str) -> list[str]:
    try:
        return split_words(text)
    except ValueError:
        return text.split()


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------
def _pair(name: str, value: str) -> list[str]:
    # "--opt=-1" keeps argparse from reading a value that starts with "-" as a flag.
    if value.startswith("-"):
        return [f"{name}={value}"]
    return [name, value]


def option_args(spec: OptionSpec, state: OptionState) -> list[str]:
    """The command line words for one enabled option."""
    name = spec.name
    value = state.value.strip()
    if spec.kind == "flag":
        return [name]
    if spec.kind == "value":
        if spec.repeat:
            words = []
            for token in _safe_split(value):
                words += _pair(name, token)
            return words
        return _pair(name, value)
    if spec.kind == "optional":
        return [name] if not value else _pair(name, value)
    return [name, *_safe_split(value)]


def head_args(profile: Profile) -> list[str]:
    args = ["--port", str(profile.port)]
    listen = profile.listen.strip()
    if listen and listen != "127.0.0.1":
        args += ["--listen", listen]
    return args


def tail_args(profile: Profile, catalog: Catalog) -> list[str]:
    """Everything after the GPU selection: --cpu, the options, then extra arguments."""
    args: list[str] = []
    if profile.gpu_mode == GPU_CPU:
        args.append("--cpu")
    for spec in catalog.options:
        if spec.name in MANAGED_OPTIONS:
            continue
        state = profile.options.get(spec.name)
        if state and state.enabled:
            args += option_args(spec, state)
    for name in sorted(profile.options):
        state = profile.options[name]
        if state.enabled and catalog.get(name) is None and name not in MANAGED_OPTIONS:
            args += option_args(unknown_spec(name, bool(state.value.strip())), state)
    args += _safe_split(profile.extra_args)
    return args


def preview_command(profile: Profile, catalog: Catalog,
                    gpu_index_by_uuid: dict[str, int] | None = None) -> str:
    """A readable one-line version of what will run inside WSL."""
    words = [profile.python.strip() or "python", "main.py", *head_args(profile)]
    prefix = ""
    if profile.gpu_mode == GPU_SELECT and profile.gpus:
        lookup = gpu_index_by_uuid or {}
        ids = [str(lookup[g.uuid]) if g.uuid in lookup else "?" for g in profile.gpus]
        words += ["--cuda-device", ",".join(ids)]
        prefix = "CUDA_DEVICE_ORDER=PCI_BUS_ID "
    words += tail_args(profile, catalog)
    return prefix + shlex.join(words)


def _probe_host(profile: Profile) -> str:
    """The address to test for "is the port already taken": the first listen
    address, or this PC when ComfyUI listens on every address."""
    host = profile.listen.strip().split(",")[0].strip().strip("[]")
    if host in _WILDCARD_HOSTS or not _HOST_CHARS.match(host):
        return "127.0.0.1"
    return host


def instance_url(profile: Profile) -> str:
    """The address to open in a browser on this PC."""
    tls = all(profile.options.get(n) and profile.options[n].enabled and profile.options[n].value.strip()
              for n in ("--tls-keyfile", "--tls-certfile"))
    host = profile.listen.strip().split(",")[0].strip()
    if host in _WILDCARD_HOSTS:
        host = "127.0.0.1"
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{'https' if tls else 'http'}://{host}:{profile.port}"


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def _is_number(text: str, kind: str) -> bool:
    try:
        int(text) if kind == "int" else float(text)
        return True
    except ValueError:
        return False


def _check_option(spec: OptionSpec, state: OptionState) -> str | None:
    value = state.value.strip()
    if spec.kind == "flag":
        return None
    if spec.kind in ("value", "optional") and not spec.repeat:
        if not value:
            return f"{spec.name} needs a value." if spec.kind == "value" else None
        tokens = [value]
    else:
        try:
            tokens = split_words(value)
        except ValueError:
            return f"{spec.name}: the value has an unclosed quote."
        minimum = 1 if spec.kind == "value" else spec.min_values
        if len(tokens) < minimum:
            return f"{spec.name} needs at least {minimum} value{'s' if minimum != 1 else ''}."
    for token in tokens:
        if spec.type in ("int", "float") and not _is_number(token, spec.type):
            kind = "a whole number" if spec.type == "int" else "a number"
            return f"{spec.name}: '{token}' is not {kind}."
        if spec.choices and token not in spec.choices:
            return f"{spec.name}: '{token}' is not one of {', '.join(spec.choices)}."
    return None


def validate(profile: Profile, catalog: Catalog) -> list[Issue]:
    """Problems with one profile on its own."""
    issues: list[Issue] = []

    def error(message):
        issues.append(Issue("error", message))

    def warning(message):
        issues.append(Issue("warning", message))

    if not profile.name.strip():
        error("The configuration needs a name.")
    if not profile.comfy_dir.strip():
        error("The ComfyUI folder is empty.")
    if not profile.python.strip():
        error("The Python interpreter is empty.")
    if not 1 <= profile.port <= 65535:
        error(f"Port {profile.port} is not between 1 and 65535.")
    if any(ch.isspace() for ch in profile.listen.strip()):
        error("The listen address must not contain spaces (separate several addresses with commas).")
    if profile.gpu_mode == GPU_SELECT and not profile.gpus:
        error("'Selected GPUs' is chosen, but no GPU is ticked.")

    # Options the launcher sets itself are never taken from the option list.
    enabled = {name for name, state in profile.options.items()
               if state.enabled and name not in MANAGED_OPTIONS}
    for name in sorted(enabled):
        spec = catalog.get(name)
        if spec is None:
            warning(f"{name} is not in the option list of this ComfyUI version. "
                    "ComfyUI will not start if it does not know the option.")
            continue
        problem = _check_option(spec, profile.options[name])
        if problem:
            error(problem)

    if profile.gpu_mode == GPU_CPU:
        enabled.add("--cpu")
    for names in catalog.groups:
        on = [n for n in names if n in enabled]
        if len(on) > 1:
            error("Only one of these can be on at a time: " + ", ".join(on) + ".")

    try:
        extra = split_words(profile.extra_args)
    except ValueError:
        extra = []
        error("The extra arguments have an unclosed quote.")
    for word in extra:
        flag = word.split("=", 1)[0]
        if flag in ("--port", "--listen", "--cuda-device", "--cpu"):
            warning(f"{flag} in the extra arguments overrides or duplicates what the launcher sets.")

    for name, value in profile.env:
        name = name.strip()
        if not name and not value.strip():
            continue
        if not _ENV_NAME.match(name):
            error(f"'{name}' is not a valid environment variable name.")
        if "\n" in value or "\r" in value:
            error(f"The value of {name} must be a single line.")
    return issues


def _device_set(profile: Profile):
    """None = every GPU, otherwise the set of chosen GPU UUIDs (empty for CPU)."""
    if profile.gpu_mode == GPU_AUTO:
        return None
    if profile.gpu_mode == GPU_CPU:
        return set()
    return {g.uuid for g in profile.gpus}


def validate_together(starting: list[Profile], running: list[Profile]) -> list[Issue]:
    """Problems between profiles: the ones about to start, and the ones already running."""
    issues: list[Issue] = []
    everything = list(starting) + [r for r in running if all(r.id != s.id for s in starting)]
    for i, a in enumerate(everything):
        for b in everything[i + 1:]:
            if a not in starting and b not in starting:
                continue
            if a.port == b.port:
                issues.append(Issue("error", f"'{a.name}' and '{b.name}' both use port {a.port}."))
            da, db = _device_set(a), _device_set(b)
            if da is None and db is None:
                issues.append(Issue("warning", f"'{a.name}' and '{b.name}' both use every GPU "
                                               "(no GPU selection)."))
            elif da is None or db is None:
                every, other, chosen = (a, b, db) if da is None else (b, a, da)
                if chosen:
                    issues.append(Issue("warning", f"'{every.name}' uses every GPU, including the "
                                                   f"one(s) '{other.name}' uses."))
            else:
                shared = da & db
                if shared:
                    names = [g.name or g.uuid for g in a.gpus if g.uuid in shared]
                    issues.append(Issue("warning", f"'{a.name}' and '{b.name}' both use "
                                                   f"{', '.join(names)}."))
    return issues


# --------------------------------------------------------------------------
# the launch script
# --------------------------------------------------------------------------
def build_launch_script(profile: Profile, catalog: Catalog) -> str:
    """The bash script that starts ComfyUI. It is piped to ``bash -s`` inside WSL.

    Everything is inside one function that runs with stdin from /dev/null, so
    nothing the script starts can swallow the rest of the script, and the last
    step is ``exec``: ComfyUI replaces bash and keeps its process id, which the
    script printed just before.
    """
    q = shlex.quote
    folder = profile.comfy_dir.strip()
    python = profile.python.strip()
    port = int(profile.port)
    lines: list[str] = []
    add = lines.append

    add("# Generated by RulZ ComfyUI Launcher. Runs inside WSL.")
    add("rcl_say() { printf '[launcher] %s\\n' \"$*\"; }")
    add("rcl_main() {")
    add(f"  cd -- {quote_path(folder)} || {{ rcl_say {q('ERROR: the ComfyUI folder was not found: ' + folder)}; exit {EXIT_NO_FOLDER}; }}")
    add(f"  [ -f main.py ] || {{ rcl_say \"ERROR: there is no main.py in $PWD\"; exit {EXIT_NO_MAIN}; }}")
    add(f"  if timeout 2 bash -c 'exec 3<>/dev/tcp/{_probe_host(profile)}/{port}' 2>/dev/null; then")
    add(f"    rcl_say 'ERROR: port {port} is already in use. Another ComfyUI may be running on it.'")
    add(f"    exit {EXIT_PORT_BUSY}")
    add("  fi")

    for name, value in profile.env:
        name = name.strip()
        if _ENV_NAME.match(name) and "\n" not in value and "\r" not in value:
            add(f"  export {name}={quote_env_value(value)}")

    pre = profile.pre_commands.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if pre.strip():
        add("  # --- commands from the configuration ---")
        lines.extend(pre.split("\n"))
        add("  # --- end of commands from the configuration ---")

    add("  RCL_GPU_ARGS=()")
    if profile.gpu_mode == GPU_SELECT and profile.gpus:
        uuids = " ".join(q(g.uuid) for g in profile.gpus)
        add("  export CUDA_DEVICE_ORDER=PCI_BUS_ID")
        add("  RCL_SMI=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)")
        add("  RCL_MAP=$(\"$RCL_SMI\" --query-gpu=index,uuid,name --format=csv,noheader 2>&1) || "
            f"{{ rcl_say \"ERROR: nvidia-smi failed inside WSL: $RCL_MAP\"; exit {EXIT_NO_SMI}; }}")
        add("  RCL_DEV=\"\"")
        add(f"  for RCL_U in {uuids}; do")
        add("    RCL_I=$(awk -F', *' -v u=\"$RCL_U\" '$2 == u { print $1; exit }' <<<\"$RCL_MAP\")")
        add(f"    [ -n \"$RCL_I\" ] || {{ rcl_say \"ERROR: GPU $RCL_U was not found inside WSL.\"; exit {EXIT_NO_GPU}; }}")
        add("    RCL_N=$(awk -F', *' -v u=\"$RCL_U\" '$2 == u { print $3; exit }' <<<\"$RCL_MAP\")")
        add("    rcl_say \"GPU $RCL_I (PCI bus order): $RCL_N\"")
        add("    RCL_DEV=\"${RCL_DEV:+$RCL_DEV,}$RCL_I\"")
        add("  done")
        add("  RCL_GPU_ARGS=(--cuda-device \"$RCL_DEV\")")

    add(f"  RCL_PY={quote_path(python)}")
    add("  case \"$RCL_PY\" in")
    add("    */*)")
    add(f"      [ -x \"$RCL_PY\" ] || {{ rcl_say \"ERROR: Python was not found: $RCL_PY (looked in $PWD)\"; exit {EXIT_NO_PYTHON}; }}")
    add("      RCL_ACT=\"$(dirname -- \"$RCL_PY\")/activate\"")
    add("      [ -f \"$RCL_ACT\" ] && . \"$RCL_ACT\"")
    add("      ;;")
    add("    *)")
    add(f"      command -v \"$RCL_PY\" >/dev/null || {{ rcl_say \"ERROR: Python was not found: $RCL_PY\"; exit {EXIT_NO_PYTHON}; }}")
    add("      ;;")
    add("  esac")

    head = " ".join(q(a) for a in head_args(profile))
    tail = " ".join(q(a) for a in tail_args(profile, catalog))
    add(f"  RCL_ARGS=({head} \"${{RCL_GPU_ARGS[@]}}\" {tail})")
    add("  rcl_say \"command: $RCL_PY main.py ${RCL_ARGS[*]}\"")
    add("  rcl_say \"pid=$$ start=$(" + _START_TIME.format(pid="$$") + ")\"")
    add("  exec \"$RCL_PY\" -u main.py \"${RCL_ARGS[@]}\"")
    add("}")
    add("rcl_main </dev/null")
    return "\n".join(lines) + "\n"


EXIT_NOT_OURS = 3


def build_signal_script(pid: int, start_time: str, signal_name: str) -> str:
    """A bash script that signals one ComfyUI instance, and only that one.

    It first checks that the process with this pid still has the start time the
    launch script reported. If the instance has ended and the number now belongs
    to another program, nothing is signalled. The launch script makes ComfyUI
    the leader of its process group, so "-PID" reaches ComfyUI and what it started.
    """
    pid = int(pid)
    if not re.fullmatch(r"[A-Z]+", signal_name) or not re.fullmatch(r"\d+", start_time or ""):
        raise ValueError("bad signal request")
    check = _START_TIME.format(pid=pid)
    return (f"[ \"$({check})\" = \"{start_time}\" ] || exit {EXIT_NOT_OURS}\n"
            f"kill -s {signal_name} -- -{pid} 2>/dev/null || kill -s {signal_name} {pid}\n")


def launch_signature(profile: Profile, catalog: Catalog) -> str:
    """Changes whenever a setting that affects the running server changes."""
    text = "\n".join([profile.distro, profile.user, build_launch_script(profile, catalog)])
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_introspection_script(profile: Profile, python_source: str) -> str:
    """A bash script that runs the option-introspection Python file in the profile's ComfyUI."""
    marker = "RCL_PY_EOF_7f3a"
    lines = [
        "rcl_main() {",
        f"  cd -- {quote_path(profile.comfy_dir.strip())} || exit {EXIT_NO_FOLDER}",
        f"  RCL_PY={quote_path(profile.python.strip() or 'python3')}",
        "  [ -x \"$RCL_PY\" ] || command -v \"$RCL_PY\" >/dev/null 2>&1 || RCL_PY=python3",
        f"  exec \"$RCL_PY\" - \"$PWD\" <<'{marker}'",
        python_source.replace("\r\n", "\n").rstrip("\n"),
        marker,
        "}",
        "rcl_main",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# importing environment variables from the interactive shell
# --------------------------------------------------------------------------
# The interactive shell prints its own marker, so anything ~/.bashrc writes to
# the screen ends up in front of the marker and not inside the first variable.
# Entries are separated by the "record separator" character (octal 036) and not
# by NUL, because NUL bytes do not survive the trip through wsl.exe's output.
ENV_DUMP_SCRIPT = (
    "printf 'RCL_ENV_BASE\\036'; env -0 | tr '\\0' '\\036'\n"
    "bash -ic 'printf \"\\036RCL_ENV_INTERACTIVE\\036\"; env -0 | tr \"\\0\" \"\\036\"' 2>/dev/null </dev/null\n"
)
_ENV_SEPARATORS = re.compile("[\0\x1e]")

_SKIP_ENV = {
    "PS1", "PS2", "SHLVL", "_", "PWD", "OLDPWD", "LS_COLORS", "LESSOPEN", "LESSCLOSE", "TERM",
    "HISTSIZE", "HISTFILESIZE", "HISTCONTROL", "PROMPT_COMMAND", "SHELL", "COLUMNS", "LINES",
}


def parse_env_dump(text: str) -> tuple[dict[str, str], dict[str, str]]:
    """Split the output of ENV_DUMP_SCRIPT into (plain shell env, interactive shell env)."""
    base: dict[str, str] = {}
    interactive: dict[str, str] = {}
    current = None
    for entry in _ENV_SEPARATORS.split(text):
        if entry == "RCL_ENV_BASE":
            current = base
        elif entry == "RCL_ENV_INTERACTIVE":
            current = interactive
        elif current is not None and "=" in entry:
            name, value = entry.split("=", 1)
            # A name cannot hold a line break: text before one is screen output.
            current[name.rsplit("\n", 1)[-1]] = value
    return base, interactive


def _literal(value: str) -> str:
    # In the environment table "\$" means a literal dollar (see quote_env_value).
    return value.replace("$", "\\$")


def diff_shell_env(base: dict[str, str], interactive: dict[str, str]) -> list[list[str]]:
    """Variables that ~/.bashrc adds or changes: the ones a plain ``wsl`` start is missing."""
    result: list[list[str]] = []
    for name, value in interactive.items():
        if name in _SKIP_ENV or name.startswith("BASH_") or not _ENV_NAME.match(name):
            continue
        if "\n" in value or "\r" in value:
            continue
        old = base.get(name)
        if old == value:
            continue
        if old and value.endswith(":" + old):
            text = _literal(value[: -len(old)]) + "$" + name
        elif old and value.startswith(old + ":"):
            text = "$" + name + _literal(value[len(old):])
        else:
            text = _literal(value)
        result.append([name, text])
    return result
