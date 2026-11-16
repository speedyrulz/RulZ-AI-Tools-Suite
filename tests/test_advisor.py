import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from rulz_optimizer import advisor


def setting(value, why="w", conf="CONF", sources=None, op=None):
    return NS(value=value, why=why, confidence=conf, sources=sources or [], op=op)


def fake_knowledge():
    model = NS(id="krea2", name="Krea 2", kind="image", summary="Train on Raw.",
               facts=[{"text": "Use Raw", "confidence": "CONF", "sources": []}],
               base={"lora_rank": setting(16)}, evaluate={"strength": "1.0", "notes": ["a", "b"]},
               flaws=[NS(symptom="plastic_skin", causes=["overtrained"],
                         fixes=[NS(set={"learning_rate": setting(0.5, op="mul")}, action=None, why="halve",
                                   confidence="REP", sources=[])])],
               open_questions=["Is Turbo OK?"], sources={})
    return NS(models={"krea2": model}, general_flaws=[], general_sources={}, symptoms={})


def test_build_context_has_knowledge_and_question():
    text = advisor.build_context(fake_knowledge(), "krea2", question="Why plastic?")
    assert "Krea 2" in text and "`lora_rank` = 16 [CONF]" in text and "plastic_skin" in text
    assert "Is Turbo OK?" in text and text.rstrip().endswith("Why plastic?")


def test_build_context_unknown_model_and_run_parts():
    run = NS(job="j", arch="krea2", finished=True, config={"learning_rate": 1e-4, "steps": 100}, checkpoints=[],
             probes=[NS(step=100, face=0.5, body=None, combined=0.4)])
    findings = [NS(severity="warn", symptom="peaked_early", detail="best at 40%")]
    text = advisor.build_context(NS(models={}, general_flaws=[]), "nope", run=run, findings=findings)
    assert "no knowledge file" in text and "peaked_early" in text and "| 100 | 0.500 |  | 0.400 |" in text


@pytest.mark.parametrize("provider,expected", [
    ("claude", ["-p", advisor.INSTRUCTION, "--permission-mode", "plan", "--output-format", "text"]),
    ("grok", ["-p", advisor.INSTRUCTION, "--permission-mode", "plan", "--output-format", "plain"]),
    ("agy", ["-p", advisor.INSTRUCTION, "--mode", "plan", "--output-format", "text"]),
])
def test_command_lines(provider, expected):
    assert advisor.build_command(provider, "EXE") == ["EXE"] + expected


def test_codex_command_and_model_flags():
    assert advisor.build_command("codex", "EXE") == ["EXE", "exec", "--sandbox", "read-only",
                                                      "--skip-git-repo-check", advisor.INSTRUCTION]
    assert advisor.build_command("codex", "EXE", "m1")[-3:-1] == ["-m", "m1"]
    assert advisor.build_command("claude", "EXE", "m1")[-2:] == ["--model", "m1"]
    assert advisor.build_command("grok", "EXE", "m1")[-2:] == ["-m", "m1"]
    assert advisor.build_command("agy", "EXE", " m1 ")[-2:] == ["--model", "m1"]


def test_instruction_is_shell_safe():
    assert not set(advisor.INSTRUCTION) & set('"&|<>^%`$')


def test_available_uses_which(monkeypatch):
    monkeypatch.setattr(advisor.shutil, "which", lambda n: f"C:/bin/{n}.cmd" if n in ("claude", "agy") else None)
    found = advisor.available()
    assert found["claude"] == Path("C:/bin/claude.cmd") and found["codex"] is None


def test_ask_not_installed(monkeypatch):
    monkeypatch.setattr(advisor.shutil, "which", lambda n: None)
    res = advisor.ask("codex", "ctx", "q")
    assert not res.ok and "not installed" in res.text and res.command == []


def test_ask_runs_in_temp_folder_with_context_file(monkeypatch):
    seen = {}

    def fake_run(args, **kw):
        seen["args"], seen["kw"] = args, kw
        seen["file"] = (Path(kw["cwd"]) / "advisor_context.md").read_text(encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, "the answer\n", "")

    monkeypatch.setattr(advisor.shutil, "which", lambda n: "C:/bin/claude.cmd")
    monkeypatch.setattr(advisor.subprocess, "run", fake_run)
    res = advisor.ask("claude", "# ctx", "What now?", model="m", timeout=7)
    assert res.ok and res.text == "the answer" and res.command[0] == "C:/bin/claude.cmd"
    assert "--model" in res.command and seen["kw"]["timeout"] == 7 and not seen["kw"].get("shell")
    assert seen["file"].startswith("# ctx") and seen["file"].rstrip().endswith("What now?")
    assert "rulz-advisor-" in seen["kw"]["cwd"] and not Path(seen["kw"]["cwd"]).exists()  # cleaned up


@pytest.mark.parametrize("stderr,word", [("Error: not logged in, please login", "not signed in"),
                                         ("429 rate limit exceeded", "quota"),
                                         ("boom", "failed (exit code 2)")])
def test_ask_failures_are_explained(monkeypatch, stderr, word):
    monkeypatch.setattr(advisor.shutil, "which", lambda n: "C:/bin/x")
    monkeypatch.setattr(advisor.subprocess, "run",
                        lambda args, **kw: subprocess.CompletedProcess(args, 2, "", stderr))
    res = advisor.ask("grok", "c", "q")
    assert not res.ok and word in res.text and stderr in res.text


def test_ask_timeout_and_os_error(monkeypatch):
    monkeypatch.setattr(advisor.shutil, "which", lambda n: "C:/bin/x")

    def boom(args, **kw):
        raise subprocess.TimeoutExpired(cmd=args, timeout=1)

    monkeypatch.setattr(advisor.subprocess, "run", boom)
    res = advisor.ask("agy", "c", "q", timeout=1)
    assert not res.ok and "did not finish" in res.text

    def bad(args, **kw):
        raise OSError("cannot run")

    monkeypatch.setattr(advisor.subprocess, "run", bad)
    assert "Could not start" in advisor.ask("agy", "c", "q").text


def test_empty_answer_is_a_failure(monkeypatch):
    monkeypatch.setattr(advisor.shutil, "which", lambda n: "C:/bin/x")
    monkeypatch.setattr(advisor.subprocess, "run", lambda a, **k: subprocess.CompletedProcess(a, 0, "  ", ""))
    res = advisor.ask("claude", "c", "q")
    assert not res.ok and "no answer" in res.text


def test_unknown_provider():
    assert not advisor.ask("nope", "c", "q").ok
