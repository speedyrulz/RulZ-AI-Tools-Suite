"""Settings tab: the RulZ AI Trainer folder, the AI helper, export folder, extra run folders and the look."""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from .. import advisor, bridge, settings
from . import common
from .common import AppState


def build(state: AppState) -> None:
    cfg = settings.get()

    with common.section("RulZ AI Trainer", "dns", "The install this app validates and queues configs in."):
        trainer_in = ui.input(label="RulZ AI Trainer folder (empty: look next to this app)", value=cfg.trainer_dir) \
            .props("outlined dense clearable").classes("w-full max-w-3xl")
        trainer_label = ui.label("").classes("text-sm")

        def show_trainer() -> None:
            hint = common.clean_path(trainer_in.value or "")
            inst = bridge.find_trainer(Path(hint) if hint else None)
            if inst:
                trainer_label.set_text(f"Found: {inst.root} | version {inst.version or '?'} | python {inst.python}")
                trainer_label.classes(remove="rz-err", add="rz-ok")
            else:
                trainer_label.set_text("No RulZ AI Trainer found here (needs rulztrainer/config.py).")
                trainer_label.classes(remove="rz-ok", add="rz-err")

        show_trainer()
        common.secondary_button("Check", "search", show_trainer, small=True)

    with common.section("AI helper", "psychology", "Which assistant the Ask AI tab uses by default."):
        found = advisor.available()
        provider = ui.select({**{p: advisor.PROVIDER_LABELS[p] for p in advisor.PROVIDERS}, "none": "None"},
                             value=cfg.advisor, label="Default provider").props("outlined dense").classes("w-64")
        models: dict[str, ui.input] = {}
        for p in advisor.PROVIDERS:
            row = ui.row().classes("items-center gap-3")
            with row:
                models[p] = ui.input(label=f"{advisor.PROVIDER_LABELS[p]}: model (optional)", value=cfg.model_for(p)) \
                    .props("outlined dense clearable").classes("w-72")
                ui.label(str(found[p]) if found[p] else "not installed").classes(
                    "text-xs " + ("rulz-muted" if found[p] else "rz-warn"))

    with common.section("Folders", "folder", "Where configs are saved and where runs are searched."):
        export_in = ui.input(label="Export folder (empty: the trainer's configs folder, else Documents)",
                             value=cfg.export_dir).props("outlined dense clearable").classes("w-full max-w-3xl")
        ui.label(f"Currently: {common.export_dir()}").classes("text-xs rulz-muted")
        roots_in = ui.textarea(label="Extra folders to search for runs (one per line)",
                               value="\n".join(cfg.run_roots)).props("outlined autogrow").classes("w-full max-w-3xl")
        ui.label("The Runs tab always searches the trainer's output folder and the pulled vast.ai runs "
                 "(~/.rulz_vast/runs and ~/.rulz_vast/fleet/runs).").classes("text-xs rulz-muted")

    with common.section("Knowledge base", "menu_book"):
        k, err = common.get_knowledge()
        ui.label(err if err else f"Loaded {len(k.models)} model file(s).").classes("text-sm")

        def reload_knowledge() -> None:
            _k, e = common.get_knowledge(reload=True)
            ui.notify(e or "Knowledge base reloaded. Reload the page to see it.", type="warning" if e else "positive")

        common.secondary_button("Reload knowledge", "refresh", reload_knowledge, small=True)

    def save() -> None:
        cfg.trainer_dir = common.clean_path(trainer_in.value or "")
        cfg.advisor = provider.value or "claude"
        cfg.advisor_model = {p: (models[p].value or "").strip() for p in advisor.PROVIDERS}
        cfg.export_dir = common.clean_path(export_in.value or "")
        cfg.run_roots = [common.clean_path(x) for x in (roots_in.value or "").splitlines() if common.clean_path(x)]
        try:
            settings.save()
        except OSError as e:
            ui.notify(f"Could not save the settings: {e}", type="negative")
            return
        show_trainer()
        ui.notify("Settings saved.", type="positive")

    common.primary_button("Save settings", "save", save)
    ui.label(f"Settings file: {settings.settings_path()}").classes("text-xs rulz-muted")
