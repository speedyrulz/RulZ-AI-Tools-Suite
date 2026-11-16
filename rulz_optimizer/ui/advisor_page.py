"""Ask AI tab: send the knowledge, dataset, recommendation and run analysis to an AI CLI (read-only) and show the
answer, with the command that ran, how long it took and a clear error when it failed."""

from __future__ import annotations

from typing import Any

from nicegui import ui

from .. import advisor, settings
from . import common
from .common import AppState


def build(state: AppState) -> None:
    knowledge, kerr = common.get_knowledge()
    cfg = settings.get()
    found = advisor.available()

    with common.section("Ask an AI", "psychology",
                        "Sends the knowledge for your model and your own results to a coding assistant on this PC. "
                        "It runs read-only in an empty temporary folder and never touches your files."):
        with ui.row().classes("items-center gap-3 w-full"):
            default = cfg.advisor if cfg.advisor in advisor.PROVIDERS else "claude"
            provider = ui.select({p: advisor.PROVIDER_LABELS[p] + ("" if found[p] else " (not installed)")
                                  for p in advisor.PROVIDERS}, value=default, label="Provider") \
                .props("outlined dense").classes("w-64")
            model_in = ui.input(label="Model (optional)", value=cfg.model_for(default)) \
                .props("outlined dense clearable").classes("w-60")
            ui.label("").classes("flex-grow")
        provider.on_value_change(lambda e: model_in.set_value(settings.get().model_for(e.value)))
        status = ui.label("").classes("text-xs rulz-muted")

        def show_status() -> None:
            path = found.get(provider.value)
            status.set_text(f"{advisor.PROVIDER_LABELS[provider.value]}: {path}" if path else
                            f"{advisor.PROVIDER_LABELS[provider.value]} was not found on PATH. "
                            f"{advisor.SETUP_HINT[provider.value]}")

        provider.on_value_change(lambda _: show_status())
        show_status()

        include_scan = ui.checkbox("Include the dataset summary", value=True)
        include_rec = ui.checkbox("Include the recommendation", value=True)
        include_run = ui.checkbox("Include the run analysis", value=True)

        ui.label("Questions to start from:").classes("text-xs rulz-muted")
        question = ui.textarea(label="Your question", placeholder="Type a question or pick one above.") \
            .props("outlined autogrow stack-label").classes("w-full")
        with ui.row().classes("gap-2 flex-wrap"):
            for q in advisor.PRESET_QUESTIONS:
                common.secondary_button(q, on_click=lambda q=q: question.set_value(q), small=True)

        with ui.expansion("Context that will be sent", icon="description").classes("w-full") as preview_exp:
            preview = ui.label("").classes("rz-pre rz-box p-3 w-full").style("max-height: 420px; overflow: auto")

        def make_context() -> str:
            mid = state.model_id or (getattr(state.run, "arch", None) if state.run else None) or ""
            return advisor.build_context(
                knowledge, mid,
                scan=state.scan if include_scan.value else None,
                rec=state.rec if include_rec.value else None,
                run=state.run if include_run.value else None,
                findings=state.findings if include_run.value else None,
                proposals=state.proposals if include_run.value else None,
                question=question.value or "")

        def refresh_preview() -> None:
            if knowledge is None:
                preview.set_text(f"{kerr}\n\n(Without the knowledge base only your own results are sent.)")
                return
            preview.set_text(make_context())

        for el in (include_scan, include_rec, include_run):
            el.on_value_change(lambda _: refresh_preview())
        question.on_value_change(lambda _: refresh_preview())
        state.on_context_change.append(refresh_preview)
        refresh_preview()

        with ui.row().classes("items-center gap-3"):
            ask_btn = common.primary_button("Ask", "send")
            spinner = ui.spinner(size="sm")
            spinner.set_visibility(False)
            timer_label = ui.label("").classes("text-xs rulz-muted")
        answer_box = ui.column().classes("w-full gap-2")

        async def do_ask() -> None:
            if not (question.value or "").strip():
                ui.notify("Type a question first.", type="warning")
                return
            prov = provider.value
            if prov not in advisor.PROVIDERS:
                ui.notify("Choose a provider.", type="warning")
                return
            context = make_context()
            ask_btn.disable()
            spinner.set_visibility(True)
            timer_label.set_text("Waiting for the answer. This can take a few minutes.")
            answer_box.clear()
            try:
                res = await common.io(advisor.ask, prov, context, question.value, model_in.value or None)
            except Exception as e:  # noqa: BLE001
                res = advisor.AdvisorResult(prov, False, f"Unexpected error: {e}", [], 0.0)
            finally:
                ask_btn.enable()
                spinner.set_visibility(False)
                timer_label.set_text("")
            show_answer(res)

        def show_answer(res: Any) -> None:
            with answer_box:
                with ui.row().classes("items-center gap-2"):
                    ui.icon("check_circle" if res.ok else "error").classes("rz-ok" if res.ok else "rz-err")
                    ui.label(f"{advisor.PROVIDER_LABELS.get(res.provider, res.provider)}: "
                             f"{'answered' if res.ok else 'failed'} in {res.seconds:.0f} s") \
                        .classes("text-sm font-semibold")
                if res.command:
                    shown = [c if len(c) < 90 else c[:87] + "..." for c in res.command]
                    ui.label("Command: " + " ".join(f'"{c}"' if " " in c else c for c in shown)) \
                        .classes("text-xs rulz-muted rulz-mono")
                if res.ok:
                    ui.markdown(res.text).classes("w-full")
                else:
                    ui.label(res.text).classes("rz-pre rz-err w-full")

        ask_btn.on_click(do_ask)
