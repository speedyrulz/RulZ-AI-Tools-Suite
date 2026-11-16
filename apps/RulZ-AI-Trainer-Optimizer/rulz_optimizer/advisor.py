"""Ask an AI coding CLI (Claude Code, Codex, Grok Build, Antigravity) about the knowledge, dataset, recommendation and
run analysis, read-only. The context goes into advisor_context.md in a fresh temp folder that is the CLI's working
folder; the CLI is told to read it and answer. Nothing here edits the user's files.

Call ask() from a worker thread (nicegui run.io_bound): it blocks until the CLI finishes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROVIDERS = ("claude", "codex", "grok", "agy")
PROVIDER_LABELS = {"claude": "Claude Code", "codex": "Codex", "grok": "Grok Build", "agy": "Antigravity (agy)"}
CONTEXT_FILE = "advisor_context.md"

# No quotes or shell characters in it: on Windows a .cmd shim goes through cmd.exe.
INSTRUCTION = ("Read advisor_context.md in the working directory. Answer the question at its end. "
               "Do not edit or create files.")

SETUP_HINT = {
    "claude": "Install Claude Code and sign in (run `claude` once).",
    "codex": "Install the Codex CLI and sign in (run `codex login`).",
    "grok": "Install Grok Build and sign in (run `grok` once).",
    "agy": "Install the Antigravity CLI and sign in (run `agy` once).",
}

PRESET_QUESTIONS = (
    "Review these starting settings for my dataset. What would you change, and why?",
    "Why did this run lose likeness and what should I change?",
    "Which checkpoint should I pick, and how should I judge it?",
    "What is the biggest risk in this recipe for my dataset?",
)


@dataclass
class AdvisorResult:
    provider: str
    ok: bool
    text: str
    command: list[str]
    seconds: float


def available() -> dict[str, Path | None]:
    """Where each CLI is (shutil.which resolves Windows .cmd and .exe shims), None when it is not installed."""
    found: dict[str, Path | None] = {}
    for name in PROVIDERS:
        path = shutil.which(name)
        found[name] = Path(path) if path else None
    return found


# ---------------------------------------------------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------------------------------------------------

def _g(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default


def _fmt_value(value: Any) -> str:
    text = repr(value) if not isinstance(value, str) else value
    return text if len(text) <= 120 else text[:117] + "..."


def _src(k: Any, model_id: str, ids: list[str]) -> str:
    if not ids:
        return ""
    try:
        from . import knowledge
        return "; ".join(knowledge.source_text(k, model_id, s) for s in ids)
    except Exception:  # noqa: BLE001
        return ", ".join(ids)


def _setting_line(k: Any, model_id: str, key: str, s: Any) -> str:
    op = f" ({s.op})" if _g(s, "op") else ""
    return (f"- `{key}` = {_fmt_value(_g(s, 'value'))}{op} [{_g(s, 'confidence', '?')}] {_g(s, 'why', '')} "
            f"(sources: {_src(k, model_id, list(_g(s, 'sources', []) or [])) or 'none'})")


def _flaw_lines(k: Any, model_id: str, flaws: list[Any]) -> list[str]:
    lines: list[str] = []
    for fl in flaws:
        lines.append(f"- **{fl.symptom}**" + (f" (causes: {', '.join(fl.causes)})" if fl.causes else ""))
        for fix in fl.fixes:
            parts = [f"{key}={_fmt_value(s.value)}" + (f" ({s.op})" if s.op else "")
                     for key, s in (fix.set or {}).items()]
            text = "; ".join(parts + ([fix.action] if fix.action else []))
            lines.append(f"  - {text} [{fix.confidence}] {fix.why or ''}".rstrip())
    return lines


def build_context(k: Any, model_id: str, *, scan: Any = None, rec: Any = None, run: Any = None,
                  findings: list[Any] | None = None, proposals: list[Any] | None = None, question: str = "") -> str:
    """Markdown for the AI: the model's knowledge, the dataset summary, the recommendation, the run analysis.
    No file paths outside the run/dataset, no personal details."""
    out: list[str] = ["# RulZ AI Trainer Optimizer: context", "",
                      "You are helping choose settings for a character (single person) LoRA in RulZ AI Trainer. "
                      "Confidence ratings: CONF = two or more independent sources or numbers agree, REP = one source, "
                      "DISP = sources disagree, MEAS = measured in our own runs, DOC = official documentation, "
                      "GUESS = no evidence. Say when you disagree with a rating and why.", ""]
    m = (_g(k, "models", {}) or {}).get(model_id)
    if m is not None:
        out += [f"## Model: {m.name} (`{m.id}`, {m.kind})", "", (m.summary or "").strip(), ""]
        if m.facts:
            out += ["### Facts", ""]
            out += [f"- [{f.get('confidence', '?')}] {f.get('text', '')} "
                    f"(sources: {_src(k, model_id, list(f.get('sources') or [])) or 'none'})" for f in m.facts]
            out.append("")
        if m.base:
            out += ["### Base recipe", ""] + [_setting_line(k, model_id, key, s) for key, s in m.base.items()] + [""]
        if m.evaluate:
            out += ["### How to evaluate", ""]
            for key, val in m.evaluate.items():
                if isinstance(val, list):
                    out += [f"- {key}:"] + [f"  - {v}" for v in val]
                else:
                    out.append(f"- {key}: {val}")
            out.append("")
        flaws = list(m.flaws) + list(_g(k, "general_flaws", []) or [])
        if flaws:
            out += ["### Known flaws and fixes", ""] + _flaw_lines(k, model_id, flaws) + [""]
        if m.open_questions:
            out += ["### Open questions (nobody has settled these)", ""] + [f"- {q}" for q in m.open_questions] + [""]
    else:
        out += [f"## Model: `{model_id}` (no knowledge file found)", ""]

    if scan is not None:
        out += ["## Dataset", "",
                f"- images: {_g(scan, 'image_count', len(_g(scan, 'images', []) or []))}, "
                f"videos: {len(_g(scan, 'videos', []) or [])}",
                f"- captioned: {_g(scan, 'captioned', 0)} (has captions: {_g(scan, 'has_captions', False)})",
                f"- median megapixels: {round(float(_g(scan, 'median_mp', 0) or 0), 2)}, "
                f"shortest side: {_g(scan, 'min_side', 0)} px",
                f"- aspect buckets: {_g(scan, 'aspect_buckets', {})}",
                f"- near-duplicate pairs: {len(_g(scan, 'duplicates', []) or [])}, "
                f"images under 512 px: {len(_g(scan, 'small', []) or [])}",
                f"- trigger word found in captions: {_g(scan, 'trigger_found', None)}"]
        out += [f"- warning: {scrub_paths(str(w))}" for w in (_g(scan, "warnings", []) or [])] + [""]

    if rec is not None:
        goals = _g(rec, "goals")
        out += ["## Recommendation", "",
                f"- goal: {_g(goals, 'goal')}, focus: {_g(goals, 'focus')}, target: {_g(goals, 'target')}, "
                f"vram_gb: {_g(goals, 'vram_gb')}, steps: {_g(rec, 'steps')}", "",
                "| key | value | confidence | why |", "| --- | --- | --- | --- |"]
        for key, c in (_g(rec, "choices", {}) or {}).items():
            why = str(c.why or "").replace("|", "/").replace("\n", " ")
            out.append(f"| {key} | {_fmt_value(c.value)} | {c.confidence} | {why} |")
        out += [""] + [f"- warning: {scrub_paths(str(w))}" for w in (_g(rec, "warnings", []) or [])]
        out += [f"- note: {n}" for n in (_g(rec, "notes", []) or [])] + [""]

    if run is not None:
        out += ["## Run analysis", "", f"- job: {_g(run, 'job')}, arch: {_g(run, 'arch')}, "
                f"finished: {_g(run, 'finished')}",
                f"- probe points: {len(_g(run, 'probes', []) or [])}, checkpoints: {len(_g(run, 'checkpoints', []) or [])}"]
        cfg = _g(run, "config", {}) or {}
        shown = {key: cfg[key] for key in ("learning_rate", "optimizer", "lora_rank", "lora_alpha", "steps",
                                          "batch_size", "network_type", "lr_scheduler", "timestep_sampling",
                                          "gradient_accumulation") if key in cfg}
        if shown:
            out.append(f"- settings: {shown}")
        probes = list(_g(run, "probes", []) or [])
        if probes:
            out += ["", "| step | face | body | combined |", "| --- | --- | --- | --- |"]
            step = max(1, len(probes) // 25)
            for p in probes[::step]:
                out.append(f"| {p.step} | {_num(p.face)} | {_num(p.body)} | {_num(p.combined)} |")
        if findings:
            out += ["", "### Findings (detected from the numbers)", ""]
            out += [f"- [{f.severity}] {f.symptom}: {f.detail}" for f in findings]
        if proposals:
            out += ["", "### Proposed fixes", ""]
            for pr in proposals:
                fix = pr.fix
                parts = [f"{key}={_fmt_value(s.value)}" + (f" ({s.op})" if s.op else "")
                         for key, s in (fix.set or {}).items()]
                out.append(f"- for {pr.symptom}: " + "; ".join(parts + ([fix.action] if fix.action else []))
                           + f" [{fix.confidence}]")
        out.append("")

    out += ["## Question", "", question.strip() or "(none given)", ""]
    return scrub_home("\n".join(out))


_WIN_PATH = re.compile(r"[A-Za-z]:[\\/](?:(?!: |['\"`|]).)*")
_POSIX_PATH = re.compile(r"/(?:home|Users|root|mnt|media|tmp|var)/(?:(?!: |['\"`|]).)*")


def scrub_paths(text: str) -> str:
    """Cut every file path in a text down to its file name, and hide the user's home folder."""
    def short(match: re.Match) -> str:
        return re.split(r"[\\/]", match.group(0).rstrip("\\/"))[-1]
    text = _WIN_PATH.sub(short, text)
    text = _POSIX_PATH.sub(short, text)
    home = str(Path.home())
    return text.replace(home, "~") if home else text


def scrub_home(text: str) -> str:
    home = str(Path.home())
    return text.replace(home, "~") if home else text


def _num(v: Any) -> str:
    return "" if v is None else f"{v:.3f}"


# ---------------------------------------------------------------------------------------------------------------------
# Running the CLIs
# ---------------------------------------------------------------------------------------------------------------------

def build_command(provider: str, exe: str, model: str | None = None) -> list[str]:
    """The exact command line for a provider, read-only (see docs/DESIGN.md)."""
    model = (model or "").strip() or None
    # The npm CLIs are .cmd shims that Windows runs through cmd.exe, so a typed model name must not carry cmd
    # metacharacters (& | < > ^ % " and spaces). Real model ids only use these characters.
    if model and not re.fullmatch(r"[A-Za-z0-9._:/\-\[\]]+", model):
        raise ValueError(f"Model name {model!r} has characters a model id never uses.")
    if provider == "claude":
        cmd = [exe, "-p", INSTRUCTION, "--permission-mode", "plan", "--output-format", "text"]
        return cmd + (["--model", model] if model else [])
    if provider == "codex":
        cmd = [exe, "exec", "--sandbox", "read-only", "--skip-git-repo-check"]
        return cmd + (["-m", model] if model else []) + [INSTRUCTION]
    if provider == "grok":
        cmd = [exe, "-p", INSTRUCTION, "--permission-mode", "plan", "--output-format", "plain"]
        return cmd + (["-m", model] if model else [])
    if provider == "agy":
        cmd = [exe, "-p", INSTRUCTION, "--mode", "plan", "--output-format", "text"]
        return cmd + (["--model", model] if model else [])
    raise ValueError(f"Unknown provider {provider!r}")


_AUTH = re.compile(r"not logged in|please log ?in|sign in|signin|login required|unauthori[sz]ed|authenticat|api key|"
                   r"invalid credentials|401|oauth", re.I)
_QUOTA = re.compile(r"quota|rate.?limit|usage limit|too many requests|429|exhausted|limit reached|credit", re.I)


def explain_failure(provider: str, code: int | None, text: str) -> str:
    """A plain explanation for a failed run, ahead of the CLI's own output."""
    label = PROVIDER_LABELS.get(provider, provider)
    if code == -1:
        return f"{label} did not finish in time."
    if _QUOTA.search(text):
        return f"{label} reports a quota or rate limit. Try again later or choose another provider."
    if _AUTH.search(text):
        return f"{label} is not signed in. {SETUP_HINT.get(provider, '')}".strip()
    return f"{label} failed (exit code {code})."


def ask(provider: str, context: str, question: str, model: str | None = None, timeout: int = 600) -> AdvisorResult:
    """Write the context to a fresh temp folder as advisor_context.md and run the CLI there, read-only."""
    started = time.time()
    if provider not in PROVIDERS:
        return AdvisorResult(provider, False, f"Unknown provider {provider!r}.", [], 0.0)
    exe = shutil.which(provider)
    if not exe:
        return AdvisorResult(provider, False,
                             f"{PROVIDER_LABELS[provider]} is not installed (`{provider}` not found on PATH). "
                             f"{SETUP_HINT[provider]}", [], 0.0)
    try:
        command = build_command(provider, exe, model)
    except ValueError as e:
        return AdvisorResult(provider, False, str(e), [], 0.0)
    q = question.strip()
    # build_context already ends with the question when it was given there; add it only when it is not.
    full = context if (not q or context.rstrip().endswith(q)) else context.rstrip() + "\n\n## Question\n\n" + q + "\n"
    env = dict(os.environ, PYTHONUTF8="1")
    with tempfile.TemporaryDirectory(prefix="rulz-advisor-") as tmp:
        (Path(tmp) / CONTEXT_FILE).write_text(full, encoding="utf-8")
        try:
            done = subprocess.run(command, cwd=tmp, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=timeout, env=env, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as e:
            partial = (e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else e.stdout) or ""
            text = explain_failure(provider, -1, "") + (f"\n\nPartial output:\n{partial}" if partial.strip() else "")
            return AdvisorResult(provider, False, text, command, time.time() - started)
        except OSError as e:
            return AdvisorResult(provider, False, f"Could not start {exe}: {e}", command, time.time() - started)
    out = (done.stdout or "").strip()
    err = (done.stderr or "").strip()
    seconds = time.time() - started
    if done.returncode == 0 and out:
        return AdvisorResult(provider, True, out, command, seconds)
    detail = "\n".join(x for x in (out, err) if x)
    reason = explain_failure(provider, done.returncode, detail)
    if done.returncode == 0:
        reason = f"{PROVIDER_LABELS[provider]} returned no answer."
    return AdvisorResult(provider, False, f"{reason}\n\n{detail}".strip(), command, seconds)
