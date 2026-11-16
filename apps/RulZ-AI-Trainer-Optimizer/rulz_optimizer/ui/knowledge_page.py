"""Knowledge tab: browse the knowledge base per model, read-only."""

from __future__ import annotations

from typing import Any

from nicegui import ui

from . import common
from .common import AppState


def setting_rows(k: Any, model_id: str, settings_map: dict) -> list[tuple[str, str, str, str, str]]:
    from .. import knowledge as kmod

    rows = []
    for key, s in settings_map.items():
        srcs = []
        for sid in s.sources:
            try:
                srcs.append(kmod.source_text(k, model_id, sid))
            except Exception:  # noqa: BLE001
                srcs.append(sid)
        value = common.fmt_value(s.value) + (f" ({s.op})" if s.op else "")
        rows.append((key, value, s.why or "", s.confidence, "; ".join(srcs)))
    return rows


def build(state: AppState) -> None:
    knowledge, kerr = common.get_knowledge()
    if knowledge is None:
        with common.section("Knowledge", "menu_book"):
            common.banner(kerr or "Knowledge base not found.", "warning", "info")
            ui.label("The knowledge base is the knowledge/ folder next to this app (general.yaml and "
                     "models/<arch>.yaml).").classes("text-sm rulz-muted")
        return

    problems = common.knowledge_problems()
    if problems:
        with common.section("Problems in the knowledge base", "report_problem",
                            f"{len(problems)} found when it was loaded. Recommendations still work; fix the files."):
            for problem in problems:
                with ui.row().classes("items-start gap-2 no-wrap"):
                    ui.icon("warning").classes("rz-warn")
                    ui.label(problem).classes("text-sm rz-warn")

    with common.section("Knowledge base", "menu_book",
                        "Read-only. What the recommendation is built from, with its confidence and sources."):
        options = {mid: f"{m.name} ({mid})" for mid, m in knowledge.models.items()}
        pick = ui.select(options, value=next(iter(options), None), label="Model") \
            .props("outlined dense").classes("w-full max-w-xl")
        legend = ui.row().classes("gap-3 items-center")
        with legend:
            for conf in common.CONF_HELP:
                common.conf_chip(conf)
            ui.label("Hover a chip for what it means.").classes("text-xs rulz-muted")
    box = ui.column().classes("w-full gap-3")

    def show() -> None:
        box.clear()
        m = knowledge.models.get(pick.value)
        if m is None:
            return
        with box:
            with common.section(f"{m.name}", "memory", f"{m.kind} | updated {m.updated}"):
                ui.label(m.summary or "").classes("text-sm")
                if m.model_preset or m.model_path:
                    ui.label(f"Model preset: {m.model_preset or '-'} | model path: {m.model_path or '-'}") \
                        .classes("text-xs rulz-muted rulz-mono")
            if m.facts:
                with common.section("Facts", "fact_check"):
                    for f in m.facts:
                        with ui.row().classes("items-start gap-2 no-wrap"):
                            common.conf_chip(f.get("confidence"))
                            ui.label(f.get("text", "")).classes("text-sm")
            if m.base:
                with common.section("Character starting recipe", "tune"):
                    common.settings_table(setting_rows(knowledge, m.id, m.base))
            if m.steps:
                with common.section("Steps", "format_list_numbered"):
                    ui.label(", ".join(f"{k}: {v}" for k, v in m.steps.items()
                                       if k not in ("why", "confidence", "sources"))).classes("text-sm rulz-mono")
                    if m.steps.get("why"):
                        with ui.row().classes("items-center gap-2"):
                            common.conf_chip(m.steps.get("confidence"))
                            ui.label(str(m.steps["why"])).classes("text-sm")
            if m.adjust:
                with common.section("Adjustments by dataset and goal", "rule"):
                    for rule in m.adjust:
                        with ui.column().classes("gap-1 w-full"):
                            ui.label(f"{rule.id}: when {rule.when}").classes("text-sm font-semibold")
                            if rule.note:
                                ui.label(rule.note).classes("text-xs rulz-muted")
                            common.settings_table(setting_rows(knowledge, m.id, rule.set))
            if m.evaluate:
                with common.section("How to evaluate", "visibility"):
                    for key, val in m.evaluate.items():
                        if isinstance(val, list):
                            ui.label(f"{key}:").classes("text-sm font-semibold")
                            for v in val:
                                ui.label(f"- {v}").classes("text-sm")
                        else:
                            ui.label(f"{key}: {val}").classes("text-sm")
            flaws = list(m.flaws) + list(knowledge.general_flaws)
            if flaws:
                with common.section("Known flaws and fixes", "build_circle"):
                    for fl in flaws:
                        sym = knowledge.symptoms.get(fl.symptom)
                        ui.label(sym.label if sym else fl.symptom).classes("text-sm font-semibold")
                        if fl.causes:
                            ui.label("Causes: " + ", ".join(fl.causes)).classes("text-xs rulz-muted")
                        for fix in fl.fixes:
                            parts = [f"{k} {s.op or '='} {common.fmt_value(s.value)}" for k, s in (fix.set or {}).items()]
                            with ui.row().classes("items-start gap-2 no-wrap"):
                                common.conf_chip(fix.confidence)
                                ui.label("; ".join(parts + ([fix.action] if fix.action else []))
                                         + (f"  ({fix.why})" if fix.why else "")).classes("text-sm")
            if m.open_questions:
                with common.section("Open questions", "help_outline", "Nobody has settled these."):
                    for q in m.open_questions:
                        ui.label(f"- {q}").classes("text-sm")
            with common.section("Sources", "link"):
                for sid, src in m.sources.items():
                    ui.label(f"{sid}: {src.get('kind', '')} | {src.get('where', '')} | {src.get('date', '')}"
                             + (f" | {src.get('people')} people" if src.get("people") else "")
                             + (f" | {src.get('url')}" if src.get("url") else "")).classes("text-xs rulz-muted")

    rules = knowledge.general.get("dataset_rules") or []
    if rules:
        with common.section("Dataset rules", "photo_library",
                            "Informational: advice on the dataset itself. The recommendation does not change settings for these."):
            for rule in rules:
                with ui.column().classes("gap-0 w-full"):
                    ui.label(str(rule.get("id", "")).replace("_", " ").capitalize()).classes("text-sm font-semibold")
                    ui.label(str(rule.get("text", ""))).classes("text-sm")

    pick.on_value_change(lambda _: show())
    show()
