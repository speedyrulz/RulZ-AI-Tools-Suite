"""Print ComfyUI's command line options as JSON.

This file is sent to a Python interpreter inside WSL (``python - <comfy dir>``).
It only needs the standard library and ComfyUI's ``comfy/cli_args.py``, which
imports nothing heavy (no torch), so it is safe to run next to a busy server.

Output: one line starting with ``RCL_JSON:`` followed by a JSON object::

    {"version": "0.45.0", "options": [...], "groups": [["--a", "--b"], ...]}
"""
import argparse
import enum
import json
import os
import sys


def _plain(value):
    """Turn an argparse default/const into something JSON can hold."""
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _describe(parser, action):
    flags = list(action.option_strings)
    nargs = action.nargs
    type_ = action.type
    choices = action.choices

    if nargs == 0:
        kind = "flag"
    elif nargs is None:
        kind = "value"
    elif nargs == "?":
        kind = "optional"
    else:
        kind = "multi"

    if isinstance(type_, type) and issubclass(type_, enum.Enum):
        if choices is None:
            choices = [m.value for m in type_]
        type_name = "str"
    elif type_ is int:
        type_name = "int"
    elif type_ is float:
        type_name = "float"
    elif callable(type_) and "director" in getattr(type_, "__name__", ""):
        type_name = "path"
    else:
        type_name = "str"

    if choices is not None:
        choices = [str(_plain(c)) for c in choices]

    try:
        help_text = parser._get_formatter()._expand_help(action)
    except Exception:
        help_text = action.help or ""
    if help_text == argparse.SUPPRESS:
        help_text = ""

    metavar = action.metavar
    if isinstance(metavar, (tuple, list)):
        metavar = " ".join(str(m) for m in metavar)

    return {
        "name": flags[0],
        "flags": flags,
        "dest": action.dest,
        "kind": kind,
        "type": type_name,
        "choices": choices,
        "default": _plain(action.default),
        "const": _plain(action.const),
        "metavar": metavar,
        "help": " ".join((help_text or "").split()),
        "repeat": isinstance(action, argparse._AppendAction),
        "min_values": 1 if nargs == "+" else (nargs if isinstance(nargs, int) else 0),
    }


def main():
    root = os.path.abspath(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "."))
    sys.path.insert(0, root)
    sys.argv = ["main.py"]

    version = ""
    try:
        import comfyui_version
        version = str(getattr(comfyui_version, "__version__", ""))
    except Exception:
        pass

    import comfy.cli_args as cli_args

    parser = cli_args.parser
    options = []
    for action in parser._actions:
        if not action.option_strings or isinstance(action, argparse._HelpAction):
            continue
        options.append(_describe(parser, action))

    groups = []
    for group in parser._mutually_exclusive_groups:
        names = [a.option_strings[0] for a in group._group_actions if a.option_strings]
        if len(names) > 1:
            groups.append(names)

    print("RCL_JSON:" + json.dumps({"version": version, "options": options, "groups": groups}))


if __name__ == "__main__":
    main()
