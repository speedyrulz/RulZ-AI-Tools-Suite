"""Recommend tab: model, dataset scan, goals, the settings table with reasons and confidence, and the Export panel."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from nicegui import ui

from .. import bridge, settings
from . import common
from .common import AppState, Output


def _trigger_from(folders: list[Path]) -> str:
    if not folders:
        return ""
    return re.sub(r"[^a-z0-9_]+", "", folders[0].name.lower().replace(" ", ""))


def build(state: AppState) -> None:
    from .. import dataset_scan, export, recommend

    knowledge, kerr = common.get_knowledge()
    last = settings.get().last if isinstance(settings.get().last, dict) else {}
    models = knowledge.models if knowledge else {}

    if kerr:
        common.banner(f"{kerr} The real knowledge base is added separately; until then pick a model by its id and "
                      "no recommendation can be made.", "warning", "info")

    # ------------------------------------------------------------------ model
    with common.section("Model", "memory", "The RulZ AI Trainer architecture to train a character LoRA on."):
        options = {mid: f"{m.name} ({mid})" for mid, m in models.items()} or {a: a for a in common.arch_ids()}
        first = last.get("model") if last.get("model") in options else (next(iter(options), None))
        state.model_id = first
        model_select = ui.select(options, value=first, label="Model").props("outlined dense").classes("w-full max-w-xl")
        variant_select = ui.select({}, label="Variant", value=None).props("outlined dense clearable") \
            .classes("w-full max-w-xl")
        summary_box = ui.column().classes("w-full gap-2")

        def show_model() -> None:
            mid = model_select.value
            if mid != state.model_id:
                state.rec = None  # a recommendation for another model must not be exported for this one
            state.model_id = mid
            m = models.get(mid)
            variants = {v["id"]: v.get("label", v["id"]) for v in (m.variants if m else [])}
            variant_select.set_options(variants, value=next(iter(variants), None))
            variant_select.set_visibility(bool(variants))
            summary_box.clear()
            with summary_box:
                if not m:
                    ui.label("No knowledge for this model.").classes("rulz-muted")
                    return
                ui.label(m.summary or "").classes("text-sm")
                if m.facts:
                    with ui.expansion(f"Facts ({len(m.facts)})", icon="fact_check").classes("w-full"):
                        for f in m.facts:
                            with ui.row().classes("items-start gap-2 no-wrap"):
                                common.conf_chip(f.get("confidence"))
                                ui.label(f.get("text", "")).classes("text-sm")
            settings.remember("model", mid)
            state.changed()

        model_select.on_value_change(lambda _: show_model())
        show_model()

    # ------------------------------------------------------------------ dataset
    with common.section("Dataset", "photo_library",
                        "Folders of images (or clips) and their caption files. One folder per line."):
        folders_box = ui.textarea(label="Dataset folders", value="\n".join(last.get("folders", []) or [])) \
            .props("outlined dense autogrow").classes("w-full max-w-3xl")
        with ui.row().classes("items-center gap-3"):
            trigger_in = ui.input(label="Trigger word", value=last.get("trigger", "")) \
                .props("outlined dense").classes("w-60")
            scan_btn = common.secondary_button("Scan folders", "search")
            scan_spinner = ui.spinner(size="sm")
            scan_spinner.set_visibility(False)
        scan_box = ui.column().classes("w-full gap-2")

        def show_scan() -> None:
            scan_box.clear()
            sc = state.scan
            with scan_box:
                if sc is None:
                    return
                with ui.row().classes("gap-6"):
                    for label, value in (("Images", sc.image_count), ("Videos", len(sc.videos)),
                                         ("Captioned", f"{sc.captioned}/{sc.image_count}"),
                                         ("Median MP", f"{sc.median_mp:.2f}"), ("Shortest side", f"{sc.min_side} px"),
                                         ("Near-duplicates", len(sc.duplicates)), ("Under 512 px", len(sc.small))):
                        with ui.column().classes("gap-0"):
                            ui.label(str(value)).classes("rz-stat-value")
                            ui.label(label).classes("rz-stat-sub")
                if sc.aspect_buckets:
                    ui.label("Aspect ratios: " + ", ".join(f"{k}: {v}" for k, v in sc.aspect_buckets.items())) \
                        .classes("text-xs rulz-muted")
                if sc.trigger_found:
                    ui.label(f"Trigger word found in the captions: {sc.trigger_found}").classes("text-xs rulz-muted")
                for w in sc.warnings:
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        ui.icon("warning").classes("rz-warn")
                        ui.label(w).classes("text-sm rz-warn")

        async def do_scan() -> None:
            folders = common.parse_folders(folders_box.value)
            if not folders:
                ui.notify("Enter at least one dataset folder.", type="warning")
                return
            missing = [str(f) for f in folders if not f.is_dir()]
            if missing:
                ui.notify("Not a folder: " + ", ".join(missing), type="negative")
                return
            scan_btn.disable()
            scan_spinner.set_visibility(True)
            try:
                trigger = (trigger_in.value or "").strip() or _trigger_from(folders)
                state.scan = await common.io(dataset_scan.scan, folders, trigger)
                state.rec = None  # the old recommendation was for the previous dataset
                state.folders = folders
                if not trigger_in.value:
                    trigger_in.set_value(trigger)
                settings.get().last["folders"] = [str(f) for f in folders]
                settings.get().last["trigger"] = trigger_in.value or ""
                settings.save()
            except Exception as e:  # noqa: BLE001
                ui.notify(f"Scan failed: {e}", type="negative")
            finally:
                scan_btn.enable()
                scan_spinner.set_visibility(False)
            show_scan()
            state.changed()

        scan_btn.on_click(do_scan)

    # ------------------------------------------------------------------ goals
    with common.section("Goals", "flag", "What the LoRA is for and where it will train."):
        with ui.row().classes("items-end gap-6 w-full"):
            goal = ui.toggle({"likeness": "Likeness", "balanced": "Balanced", "quick": "Quick test"},
                             value=last.get("goal", "balanced")).props("no-caps unelevated")
            focus = ui.toggle({"face": "Face only", "face_body": "Face and body"},
                              value=last.get("focus", "face_body")).props("no-caps unelevated")
            target = ui.toggle({"local": "This PC", "vast": "vast.ai"}, value=last.get("target", "local")) \
                .props("no-caps unelevated")
            vram = ui.number(label="GPU memory (GB)", value=last.get("vram_gb"), min=0, format="%.0f") \
                .props("outlined dense").classes("w-40")
            detect = common.ghost_button("Detect", "memory", small=True)
            ui.label("Likeness: most faithful identity. Balanced: identity that still follows prompts, clothes and "
                     "scenes. Quick test: a fast first try.").classes("text-xs rulz-muted w-full")
        base_cfg = ui.input(label="Base config (optional): a RulZ AI Trainer config.yaml whose model and hardware "
                                  "settings are kept", value=last.get("base_config", "")) \
            .props("outlined dense clearable").classes("w-full max-w-3xl")

        async def detect_vram() -> None:
            v = await common.io(common.detect_vram_gb)
            if v:
                vram.set_value(v)
            else:
                ui.notify("No NVIDIA GPU found with nvidia-smi. Type the memory of the card you will train on.",
                          type="warning")

        detect.on_click(detect_vram)
        with ui.row().classes("items-center gap-3"):
            rec_btn = common.primary_button("Recommend settings", "auto_fix_high")

    # ------------------------------------------------------------------ result
    result_box = ui.column().classes("w-full gap-3")

    # ------------------------------------------------------------------ export
    export_card = common.section("Export", "ios_share", "Save the config, or hand it to RulZ AI Trainer.")
    export_card.set_visibility(False)
    with export_card:
        job_in = ui.input(label="Job name").props("outlined dense").classes("w-80")
        with ui.row().classes("items-center gap-2 flex-wrap"):
            save_btn = common.secondary_button("Save YAML", "save")  # becomes the one red button once there is a recommendation
            val_btn = common.secondary_button("Validate", "rule")
            q_local = common.secondary_button("Queue locally", "playlist_add")
            q_vast = common.secondary_button("Queue on vast.ai", "cloud_upload")
            webui_btn = common.ghost_button("Open RulZ AI Trainer", "open_in_new")
        out = Output()
        trainer_line = ui.label("").classes("text-xs rulz-muted")
        ui.label("The local queue only starts runs while the RulZ AI Trainer WebUI is running. The vast.ai queue "
                 "copies the dataset to the instance and starts the run on the first idle one, while the WebUI's "
                 "Fleet tab (or the fleet daemon) is watching.").classes("text-xs rulz-muted")
        instances_box = ui.column().classes("gap-0")

    def show_instances() -> None:
        instances_box.clear()
        items = bridge.fleet_instances()
        with instances_box:
            if items:
                ui.label("vast.ai instances the fleet knows:").classes("text-xs rulz-muted")
                for it in items:
                    state_txt = "gone" if it["gone"] else (it["status"] or "?")
                    ui.label(f"#{it['id']} {it['label']} {it['gpu']} ({state_txt})").classes("text-xs rulz-muted")

    def collect_goals() -> Any:
        base = common.clean_path(base_cfg.value or "")
        g = recommend.Goals(goal=goal.value, focus=focus.value, target=target.value,
                            vram_gb=float(vram.value) if vram.value else None,
                            trigger=(trigger_in.value or "").strip() or None,
                            variant=variant_select.value or None,
                            base_config=Path(base).expanduser() if base else None)
        return g

    def remember_goals() -> None:
        s = settings.get()
        s.last.update({"goal": goal.value, "focus": focus.value, "target": target.value,
                       "vram_gb": vram.value, "base_config": base_cfg.value or ""})
        settings.save()

    def show_result() -> None:
        result_box.clear()
        rec = state.rec
        if rec is None:
            return
        with result_box:
            with common.section("Recommended settings", "tune",
                                f"{rec.steps} steps for {len(state.scan.images) if state.scan else '?'} images."):
                for w in rec.warnings:
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        ui.icon("warning").classes("rz-warn")
                        ui.label(w).classes("text-sm rz-warn")
                for n in rec.notes:
                    ui.label(n).classes("text-sm rulz-muted")
                k = knowledge
                rows = []
                for key, c in rec.choices.items():
                    srcs = []
                    for sid in c.sources:
                        try:
                            from .. import knowledge as kmod
                            srcs.append(kmod.source_text(k, rec.model_id, sid))
                        except Exception:  # noqa: BLE001
                            srcs.append(sid)
                    if not srcs and c.origin == "trainer_wizard":
                        srcs.append(common.WIZARD_SOURCE)
                    rows.append((key, common.fmt_value(c.value), c.why or "", c.confidence, "; ".join(srcs)))
                common.settings_table(rows)
            if rec.evaluate:
                with common.section("How to judge the checkpoints", "visibility"):
                    for key, val in rec.evaluate.items():
                        if isinstance(val, list):
                            ui.label(f"{key}:").classes("text-sm font-semibold")
                            for v in val:
                                ui.label(f"- {v}").classes("text-sm")
                        else:
                            ui.label(f"{key}: {val}").classes("text-sm")
            if rec.open_questions:
                with common.section("Open questions", "help_outline", "Nobody has settled these."):
                    for q in rec.open_questions:
                        ui.label(f"- {q}").classes("text-sm")
            with ui.row().classes("gap-2"):
                common.secondary_button("Ask an AI about this", "psychology", lambda: state.goto("advisor"))

    async def do_recommend() -> None:
        if knowledge is None or not state.model_id or state.model_id not in models:
            ui.notify("No knowledge for this model: the knowledge base is not loaded.", type="warning")
            return
        if state.scan is None:
            ui.notify("No dataset scanned: the recommendation uses defaults for the dataset size.", type="info")
        rec_btn.disable()
        try:
            goals = collect_goals()
            state.goals = goals
            state.rec = await common.io(recommend.recommend, knowledge, state.model_id, state.scan, goals)
            remember_goals()
        except Exception as e:  # noqa: BLE001
            ui.notify(f"Could not recommend: {e}", type="negative")
            return
        finally:
            rec_btn.enable()
        if not job_in.value:
            job_in.set_value(_default_job(state))
        export_card.set_visibility(True)
        common.promote(rec_btn, False)  # one red button per view: it moves to Save YAML
        common.promote(save_btn, True)
        trainer = common.get_trainer()
        trainer_line.set_text(f"RulZ AI Trainer: {trainer.root} (version {trainer.version or '?'})" if trainer else
                              "RulZ AI Trainer not found. Set its folder in Settings to validate and queue.")
        show_instances()
        show_result()
        state.changed()

    rec_btn.on_click(do_recommend)

    # ------------------------------------------------------------------ export actions
    def build_cfg(job: str) -> dict:
        base = None
        if state.goals and state.goals.base_config:
            base = export.load_trainer_config(state.goals.base_config)
        return export.build_config(state.rec, job, base)

    def header() -> str:
        trainer = common.get_trainer()
        m = models.get(state.rec.model_id)
        return common.export_header(m.name if m else state.rec.model_id, trainer.version if trainer else None)

    def save_yaml(job: str) -> Path:
        # job was checked by act(); it is passed in so a later edit of the field cannot change the path
        path = common.export_dir() / f"{job}.yaml"
        return export.write(build_cfg(job), path, header())

    async def act(kind: str) -> None:
        if state.rec is None:
            ui.notify("Press Recommend again: the model or dataset changed.", type="warning")
            return
        job = (job_in.value or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", job):
            ui.notify("Job name: letters, digits, _ . - only.", type="warning")
            return
        buttons = (save_btn, val_btn, q_local, q_vast)
        for b in buttons:
            b.disable()
        try:
            path = await common.io(save_yaml, job)
            trainer = common.get_trainer()
            if kind == "save":
                out.show(f"Saved {path}\nLoad it in the RulZ AI Trainer WebUI with 'Config file' -> Load.", True)
                return
            if trainer is None:
                out.show(f"Saved {path}\nRulZ AI Trainer not found: set its folder in Settings to "
                         "validate or queue.", False)
                return
            if kind == "validate":
                ok, msg = await common.io(bridge.validate, trainer, path)
            elif kind == "local":
                ok, msg = await common.io(bridge.queue_local, trainer, path, "RulZ AI Trainer Optimizer")
            else:
                folders = list(state.scan.folders) if state.scan else common.parse_folders(folders_box.value)
                if not folders:
                    out.show("Scan the dataset folders first: the vast.ai queue copies them to the instance.", False)
                    return
                ok, msg = await common.io(bridge.queue_vast, trainer, path, job, folders)
            out.show(f"{path}\n{msg}", ok)
        except Exception as e:  # noqa: BLE001
            out.show(f"Failed: {e}", False)
        finally:
            for b in buttons:
                b.enable()

    async def open_trainer() -> None:
        trainer = common.get_trainer()
        if trainer is None:
            out.show("RulZ AI Trainer not found: set its folder in Settings.", False)
            return
        out.show(await common.io(bridge.open_webui, trainer), None)

    save_btn.on_click(lambda: act("save"))
    val_btn.on_click(lambda: act("validate"))
    q_local.on_click(lambda: act("local"))
    q_vast.on_click(lambda: act("vast"))
    webui_btn.on_click(open_trainer)


def _default_job(state: AppState) -> str:
    base = (state.goals.trigger if state.goals and state.goals.trigger else None) or "lora"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{base}_{state.rec.model_id if state.rec else 'run'}")
