"""A sandboxed calculator as a stdio MCP server, so the model can do the arithmetic an RE task
ends in (hash rounds, XOR decoding) without a shell.

    python -I calc_mcp.py   (RulZ LLM Suite starts it as the built-in "calc" server)

One tool, `calc`: a few lines of Python-like code over ints, strings, bytes, lists, with
assignments, `for`/`if`, and a fixed set of helper functions. No attribute access, imports,
or names starting with `_`, so it can't reach files, processes or the network. The value of
the last line (or anything passed to `print`) is returned.
"""

import ast
import json
import sys

MAX_STEPS = 200_000
MAX_SIZE = 1_000_000  # longest sequence, and widest int in bits


def _range(*args):
    r = range(*args)
    if len(r) > MAX_STEPS:
        raise CalcError(f"range longer than {MAX_STEPS}")
    return r


def _mul(a, b):
    for seq, n in ((a, b), (b, a)):
        if isinstance(seq, (str, bytes, bytearray, list, tuple)) and isinstance(n, int) and len(seq) * n > MAX_SIZE:
            raise CalcError("result too large")
    if isinstance(a, int) and isinstance(b, int) and a.bit_length() + b.bit_length() > MAX_SIZE:
        raise CalcError("result too large")
    return a * b


def _lshift(a, n):
    if n > MAX_SIZE:
        raise CalcError("shift too large")
    return a << n


def _sized(value):
    if isinstance(value, (str, bytes, bytearray, list, tuple)) and len(value) > MAX_SIZE:
        raise CalcError("result too large")
    return value


def _add(a, b):
    return _sized(a + b)


def _meth(obj, name, *args):
    if name not in METHODS:
        raise CalcError(f".{name}() is not allowed")
    result = getattr(obj, name)(*args)
    _sized(obj)  # .extend/.append grow the list in place
    return _sized(result)


class _Guard(ast.NodeTransformer):
    """Route *, <<, + and method calls through size checks, so a loop that keeps doubling a
    value stops instead of exhausting memory."""

    def visit_Call(self, node):
        self.generic_visit(node)
        if not isinstance(node.func, ast.Attribute):
            return node
        args = [node.func.value, ast.Constant(node.func.attr), *node.args]
        return ast.copy_location(ast.Call(ast.Name("_meth", ast.Load()), args, []), node)

    def visit_BinOp(self, node):
        self.generic_visit(node)
        fn = {ast.Mult: "_mul", ast.LShift: "_lshift", ast.Add: "_add"}.get(type(node.op))
        if not fn:
            return node
        return ast.copy_location(ast.Call(ast.Name(fn, ast.Load()), [node.left, node.right], []), node)

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        fn = {ast.Mult: "_mul", ast.LShift: "_lshift", ast.Add: "_add"}.get(type(node.op))
        if not fn:
            return node
        target = ast.Name(node.target.id, ast.Load()) if isinstance(node.target, ast.Name) else node.target
        call = ast.Call(ast.Name(fn, ast.Load()), [target, node.value], [])
        return ast.copy_location(ast.Assign([node.target], call), node)


FUNCS = {
    "ord": ord, "chr": chr, "hex": hex, "bin": bin, "int": int, "str": str, "len": len,
    "range": _range, "list": list, "bytes": bytes, "bytearray": bytearray, "reversed": reversed,
    "sum": sum, "min": min, "max": max, "abs": abs, "enumerate": enumerate, "zip": zip,
    "rotl32": lambda x, n: ((x << (n & 31)) | (x >> ((32 - n) & 31))) & 0xFFFFFFFF,
    "rotr32": lambda x, n: ((x >> (n & 31)) | (x << ((32 - n) & 31))) & 0xFFFFFFFF,
    "u32": lambda x: x & 0xFFFFFFFF,
    "u8": lambda x: x & 0xFF,
    "hex32": lambda x: f"{x & 0xFFFFFFFF:08X}",
    "join": lambda items, sep="": sep.join(map(str, items)),
    "chars": lambda codes: "".join(chr(c & 0xFF) for c in codes),
}

# Methods of lists, strings and bytes the code may call. Nothing that reaches beyond the value
# itself, and nothing that pads to a caller-chosen size (f-strings cover formatting).
METHODS = {
    "append", "extend", "pop", "insert", "reverse", "index", "count", "copy",
    "upper", "lower", "strip", "split", "join", "replace", "startswith", "endswith", "find",
    "encode", "decode", "hex", "fromhex",
}

ALLOWED = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.For, ast.If, ast.Name, ast.Load,
    ast.Store, ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Call, ast.Subscript, ast.Slice, ast.List, ast.Tuple, ast.JoinedStr, ast.FormattedValue,
    ast.ListComp, ast.comprehension, ast.Pass, ast.operator, ast.unaryop, ast.boolop, ast.cmpop,
    ast.expr_context, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.While, ast.Break,
    ast.Continue,
)


class CalcError(Exception):
    pass


def run(code: str) -> str:
    tree = ast.parse(code, mode="exec")
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    methods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr not in METHODS:
                raise CalcError(f".{node.func.attr}() is not allowed; methods: " + ", ".join(sorted(METHODS)))
            methods.add(id(node.func))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if id(node) not in methods:
                raise CalcError("attributes are only allowed as method calls: " + ", ".join(sorted(METHODS)))
            continue
        if not isinstance(node, ALLOWED):
            raise CalcError(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.FunctionDef) and (node.decorator_list or node.name.startswith("_")):
            raise CalcError("decorators and names starting with _ are not allowed")
        if isinstance(node, (ast.Name, ast.arg)) and (getattr(node, "id", None) or node.arg).startswith("_"):
            raise CalcError("names starting with _ are not allowed")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Attribute) and not (
                isinstance(node.func, ast.Name) and
                (node.func.id in FUNCS or node.func.id in defined or node.func.id == "print")):
            raise CalcError("only these functions can be called: print, your own defs, " + ", ".join(FUNCS))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            raise CalcError("** is not allowed; use * or <<")

    out: list[str] = []
    steps = 0

    def tick(*_):
        nonlocal steps
        steps += 1
        if steps > MAX_STEPS:
            raise CalcError(f"stopped after {MAX_STEPS} steps")

    env = {"__builtins__": {}, **FUNCS, "print": lambda *a: out.append(" ".join(map(str, a)))}
    body, last = tree.body, None
    if body and isinstance(body[-1], ast.Expr):
        body, last = body[:-1], body[-1]
    # Count loop iterations so a runaway loop stops instead of hanging the server.
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, (ast.For, ast.While, ast.FunctionDef)):
            node.body.insert(0, ast.Expr(ast.Call(ast.Name("_tick", ast.Load()), [], [])))
    env.update(_tick=tick, _mul=_mul, _lshift=_lshift, _add=_add, _meth=_meth)
    tree = ast.fix_missing_locations(_Guard().visit(ast.Module(body=body, type_ignores=[])))
    exec(compile(tree, "<calc>", "exec"), env)
    if last is not None:
        expr = ast.fix_missing_locations(_Guard().visit(ast.Expression(last.value)))
        value = eval(compile(expr, "<calc>", "eval"), env)
        if isinstance(value, int) and not isinstance(value, bool):
            out.append(f"{value} (0x{value & (2**64 - 1):X})")
        elif value is not None:
            out.append(repr(value))
    return "\n".join(out) or "(no output)"


TOOL = {
    "name": "calc",
    "description": (
        "Evaluate a few lines of Python-like arithmetic: ints, strings, bytes, lists, assignments, "
        "for/while/if, def, list comprehensions, and list/str/bytes methods such as .append and .decode. Returns the last line's value (ints also in hex) and anything "
        "printed. Helpers: " + ", ".join(FUNCS) + ". No imports, other attributes, or files. Example:\n"
        "h = 0x12345678\nfor c in 'name':\n    h = rotl32(u32((h ^ ord(c)) * 0x01000193), 5)\nhex32(h)"
    ),
    "inputSchema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
}


def handle(msg: dict) -> dict | None:
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:
        return None
    if method == "initialize":
        result = {"protocolVersion": msg.get("params", {}).get("protocolVersion", "2024-11-05"),
                  "capabilities": {"tools": {}}, "serverInfo": {"name": "calc", "version": "1.0"}}
    elif method == "tools/list":
        result = {"tools": [TOOL]}
    elif method == "tools/call":
        try:
            text, err = run(msg["params"]["arguments"].get("code", "")), False
        except Exception as e:  # report every failure to the model as a tool error
            text, err = f"{type(e).__name__}: {e}", True
        result = {"content": [{"type": "text", "text": text}], "isError": err}
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown method {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        reply = handle(json.loads(line))
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
