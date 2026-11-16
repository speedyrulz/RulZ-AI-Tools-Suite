"""Runs tab: find RulZ AI Trainer run folders, show likeness over steps, best checkpoints with probe images, detected
flaws, flaws ticked from the images, proposed fixes and the exported next config."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from nicegui import ui

from .. import bridge, settings
from . import common, theme
from .common import AppState, Output


def chart_options(run: Any, light: bool = False) -> dict:
    """ECharts options: face, body and combined likeness against step, in the colours of the current look.

    The colours are the --rz-* tokens read from the theme (a canvas cannot use CSS variables). The three lines differ in
    lightness, dash and marker as well as colour: face in the red link colour, body in the text colour, combined in grey.
    """
    def tok(name: str) -> str:
        return theme.token(name, light)

    def series(name: str, attr: str, color: str, dash: str, symbol: str) -> dict:
        data = [[p.step, round(getattr(p, attr), 4)] for p in run.probes if getattr(p, attr) is not None]
        return {"name": name, "type": "line", "showSymbol": True, "symbol": symbol, "symbolSize": 6, "data": data,
                "lineStyle": {"color": color, "width": 2, "type": dash}, "itemStyle": {"color": color}}

    muted, text, border = tok("muted"), tok("text"), tok("border")
    return {
        "animation": False,
        "backgroundColor": "transparent",
        "textStyle": {"color": muted, "fontFamily": "Segoe UI Variable Text, Segoe UI, system-ui, sans-serif",
                      "fontSize": 13},
        "tooltip": {"trigger": "axis", "backgroundColor": tok("menu-bg"), "borderColor": tok("border-strong"),
                    "textStyle": {"color": text, "fontSize": 13}},
        "legend": {"data": ["face", "body", "combined"], "top": 0, "right": 20, "textStyle": {"color": muted, "fontSize": 13}},
        "grid": {"left": 50, "right": 20, "top": 40, "bottom": 40},
        "xAxis": {"type": "value", "name": "step", "nameTextStyle": {"color": muted},
                  "axisLine": {"lineStyle": {"color": tok("border-strong")}}, "axisLabel": {"color": muted},
                  "splitLine": {"show": False}},
        "yAxis": {"type": "value", "scale": True, "axisLabel": {"color": muted},
                  "axisLine": {"lineStyle": {"color": tok("border-strong")}},
                  "splitLine": {"lineStyle": {"color": border}}},
        "series": [series("face", "face", tok("link"), "solid", "circle"),
                   series("body", "body", text, "dashed", "triangle"),
                   series("combined", "combined", muted, "dotted", "diamond")],
    }


def nearest(items: dict[int, list[Path]], step: int | None) -> tuple[int | None, list[Path]]:
    if step is None or not items:
        return None, []
    key = min(items, key=lambda s: abs(s - step))
    return key, items[key]


def run_datasets(cfg: dict) -> list[Path]:
    """The dataset folders a config names (for the vast.ai push)."""
    paths: list[str] = []
    ds = cfg.get("datasets")
    if isinstance(ds, list):
        paths += [d.get("path") for d in ds if isinstance(d, dict) and d.get("path")]
    if cfg.get("dataset_dir"):
        paths.append(cfg["dataset_dir"])
    return [Path(p) for p in paths if isinstance(p, str) and Path(p).is_dir()]


def build(state: AppState) -> None:
    from .. import diagnose, export, runs

    knowledge, kerr = common.get_knowledge()
    cur: dict[str, Any] = {"proposals": [], "checks": [], "visual": set()}

    # ------------------------------------------------------------------ run list
    with common.section("Find runs", "folder_open",
                        "Local output folders, pulled vast.ai runs and any extra folders you add."):
        extra = ui.input(label="Another folder to search (a run folder or a folder of runs)") \
            .props("outlined dense clearable").classes("w-full max-w-3xl")
        with ui.row().classes("items-center gap-3"):
            find_btn = common.secondary_button("Find runs", "search")
            spinner = ui.spinner(size="sm")
            spinner.set_visibility(False)
        run_select = ui.select({}, label="Run", value=None, with_input=True).props("outlined dense") \
            .classes("w-full max-w-3xl")
        run_select.set_visibility(False)
        found_label = ui.label("").classes("text-xs rulz-muted")

    def roots() -> list[Path]:
        trainer = common.get_trainer()
        out = list(runs.default_roots(trainer.root if trainer else None))
        out += [Path(p).expanduser() for p in settings.get().run_roots if p]
        if extra.value:
            out.append(Path(common.clean_path(extra.value)).expanduser())
        return out

    async def find() -> None:
        find_btn.disable()
        spinner.set_visibility(True)
        try:
            paths = await common.io(runs.find_runs, roots())
        except Exception as e:  # noqa: BLE001
            ui.notify(f"Search failed: {e}", type="negative")
            return
        finally:
            find_btn.enable()
            spinner.set_visibility(False)
        options = {str(p): f"{p.name}  ({p.parent})" for p in paths}
        run_select.set_options(options, value=None)
        run_select.set_visibility(True)
        found_label.set_text(f"{len(paths)} run folder(s) found." if paths else "No runs found in these folders.")

    find_btn.on_click(find)

    detail_box = ui.column().classes("w-full gap-3")

    # ------------------------------------------------------------------ one run
    async def open_run(path_text: str) -> None:
        detail_box.clear()
        if not path_text:
            return
        try:
            run = await common.io(runs.load_run, Path(path_text))
            findings = await common.io(diagnose.detect, run)
        except Exception as e:  # noqa: BLE001
            with detail_box:
                common.banner(f"Could not read this run: {e}", "error", "info")
            return
        state.run, state.findings, state.proposals = run, findings, []
        cur.update(proposals=[], checks=[], visual=set())
        common.run_token(run.path)
        with detail_box:
            render_run(run, findings)
        state.changed()

    run_select.on_value_change(lambda e: open_run(e.value))

    def model_of(run: Any) -> str | None:
        return run.arch or run.config.get("model_arch")

    def render_run(run: Any, findings: list[Any]) -> None:
        mid = model_of(run)
        with common.section(run.job or run.path.name, "show_chart",
                            f"{mid or 'unknown model'} | {'finished' if run.finished else 'not finished (or still running)'}"
                            f" | {len(run.probes)} probe points | {len(run.checkpoints)} checkpoints"):
            ui.label(str(run.path)).classes("text-xs rulz-muted rulz-mono")
            for err in run.errors[:5]:
                ui.label(err).classes("text-xs rz-warn")
            if run.probes:
                look = state.look
                chart = ui.echart(chart_options(run, bool(look and look.light))).classes("w-full") \
                    .style("height: 300px")

                def retheme(light: bool, chart: Any = chart) -> None:
                    if chart.is_deleted:
                        look.off(retheme)
                        return
                    chart.options.clear()
                    chart.options.update(chart_options(run, light))
                    chart.update()

                if look is not None:
                    look.on_change(retheme)
            else:
                ui.label("This run has no likeness probes yet.").classes("rulz-muted")

        best = runs.best_points(run)
        if any(best.values()):
            with common.section("Best checkpoints", "emoji_events",
                                "By face, body and combined likeness (normalised scores where the run has them)."):
                for name in ("face", "body", "combined"):
                    p = best.get(name)
                    if p is None:
                        continue
                    value = {"face": p.face, "body": p.body, "combined": p.combined}[name]
                    ck = _checkpoint_near(run, p.step)
                    step_key, images = nearest(run.probe_images, p.step)
                    with ui.column().classes("gap-1 w-full"):
                        ui.label(f"Best {name}: step {p.step}, score {common.fmt_num(value)}"
                                 + (f" | checkpoint {ck.path.name}" if ck else "")).classes("text-sm font-semibold")
                        if step_key is not None and step_key != p.step:
                            ui.label(f"(probe images from the nearest saved step, {step_key})") \
                                .classes("text-xs rulz-muted")
                        probe_images(run, images[:8])

        sample_groups = runs.sample_steps(run)
        if sample_groups:
            best_pt = runs.recorded_best_point(run)
            best_step = best_pt.step if best_pt else None
            steps = sorted(sample_groups, reverse=True)

            def sample_label(s: int) -> str:
                if s < 0:
                    return f"Other samples ({len(sample_groups[s])} images)"
                tags = (["newest"] if s == steps[0] else []) + (["best checkpoint"] if s == best_step else [])
                return f"Step {s}" + (f" ({', '.join(tags)})" if tags else "") + f", {len(sample_groups[s])} images"

            with common.section("Samples", "collections",
                                "Images the trainer rendered from your sample prompts while it trained. Pick a step."):
                picker = ui.select({s: sample_label(s) for s in steps}, value=steps[0], label="Step") \
                    .props("outlined dense").classes("w-96")
                if best_step is not None and best_step not in sample_groups and best_step >= 0:
                    near, _ = nearest(sample_groups, best_step)
                    ui.label(f"The best checkpoint is step {best_step}; no samples were rendered at that step "
                             f"(nearest: step {near}).").classes("rz-tagline")
                holder = ui.column().classes("w-full gap-1")

                def show_samples(step: int) -> None:
                    holder.clear()
                    with holder:
                        if step == best_step:
                            ui.label(f"Samples at step {step}, the best checkpoint").classes("text-sm font-semibold")
                        probe_images(run, sample_groups.get(step, [])[:12])

                picker.on_value_change(lambda e: show_samples(e.value))
                show_samples(steps[0])

        with common.section("What the numbers show", "troubleshoot"):
            if not findings:
                ui.label("No problems detected from the numbers.").classes("text-sm rz-ok")
            for f in findings:
                color = {"bad": "rz-err", "warn": "rz-warn"}.get(f.severity, "rulz-muted")
                with ui.row().classes("items-start gap-2 no-wrap"):
                    ui.icon({"bad": "error", "warn": "warning"}.get(f.severity, "info")).classes(color)
                    ui.label(f"{f.symptom}: {f.detail}").classes(f"text-sm {color}")

        # visual flaws the user ticks
        visual = []
        if knowledge:
            visual = [s for s in knowledge.symptoms.values() if s.kind in ("visual", "both")]
        with common.section("What do you see in the images?", "image_search",
                            "Tick the flaws you see in the probe or sample images; they add to the detected ones."):
            if not visual:
                ui.label("No visual flaw list: " + (kerr or "the knowledge base has none.")).classes("rulz-muted")
            with ui.row().classes("gap-x-6 gap-y-1"):
                for s in visual:
                    cb = ui.checkbox(s.label, on_change=lambda e, sid=s.id: _toggle(sid, e.value))
                    with cb:
                        ui.tooltip(s.description or s.label)
            propose_btn = common.primary_button("Propose fixes", "build")

        proposals_box = ui.column().classes("w-full gap-3")

        def _toggle(sid: str, on: bool) -> None:
            (cur["visual"].add if on else cur["visual"].discard)(sid)

        async def propose() -> None:
            if knowledge is None or not mid or mid not in knowledge.models:
                ui.notify("No knowledge for this model, so no fixes can be proposed.", type="warning")
                return
            symptoms = [f.symptom for f in findings if f.severity != "info"] + sorted(cur["visual"])
            proposals = await common.io(diagnose.propose, knowledge, mid, list(dict.fromkeys(symptoms)))
            cur["proposals"], cur["checks"] = proposals, [pr.alternative_of is None for pr in proposals]
            common.promote(propose_btn, not proposals)  # with proposals the red button is Save YAML below
            state.proposals = proposals
            render_proposals(proposals_box, run, mid)
            state.changed()

        propose_btn.on_click(propose)

    def render_proposals(box: Any, run: Any, mid: str) -> None:
        box.clear()
        proposals = cur["proposals"]
        with box:
            with common.section("Proposed fixes", "build_circle", "Tick the ones to apply to the next config."):
                if not proposals:
                    ui.label("Nothing to propose for these symptoms.").classes("rulz-muted")
                def fix_text(pr: Any) -> str:
                    parts = [f"{k} {s.op or '='} {common.fmt_value(s.value)}" for k, s in (pr.fix.set or {}).items()]
                    return "; ".join(parts + ([pr.fix.action] if pr.fix.action else []))

                def root(i: int) -> int:
                    seen = set()
                    while proposals[i].alternative_of is not None and i not in seen:
                        seen.add(i)
                        i = proposals[i].alternative_of
                    return i

                groups: dict[int, list[int]] = {}
                for i in range(len(proposals)):
                    groups.setdefault(root(i), []).append(i)
                boxes: dict[int, Any] = {}

                def toggled(i: int, on: bool) -> None:
                    cur["checks"][i] = on
                    if on:  # an alternative and the fix it competes with set the same key: only one can apply
                        for j in groups[root(i)]:
                            if j != i and cur["checks"][j]:
                                cur["checks"][j] = False
                                boxes[j].set_value(False)

                for first, members in sorted(groups.items()):
                    for i in [first] + [m for m in members if m != first]:
                        pr, fix = proposals[i], proposals[i].fix
                        alt = i != first
                        with ui.row().classes("items-start gap-2 no-wrap w-full" + (" pl-10" if alt else "")):
                            boxes[i] = ui.checkbox(value=cur["checks"][i],
                                                   on_change=lambda e, i=i: toggled(i, e.value))
                            with ui.column().classes("gap-0 min-w-0"):
                                with ui.row().classes("items-center gap-2"):
                                    ui.label(pr.symptom).classes("text-sm font-semibold")
                                    common.conf_chip(fix.confidence)
                                    if alt:
                                        ui.label("Alternative, the same setting as the fix above").classes("rz-tagline")
                                ui.label(fix_text(pr)).classes("text-sm rulz-mono")
                                if fix.why:
                                    ui.label(fix.why).classes("text-xs rulz-muted")
            if proposals:
                export_panel(run, mid)

    def export_panel(run: Any, mid: str) -> None:
        with common.section("Export the next config", "ios_share"):
            job_in = ui.input(label="Job name", value=diagnose.next_config(run, cur["proposals"], [])["job_name"]) \
                .props("outlined dense").classes("w-80")
            with ui.row().classes("items-center gap-2 flex-wrap"):
                save_btn = common.primary_button("Save YAML", "save")  # the proposals are made: the red button moves here
                val_btn = common.secondary_button("Validate", "rule")
                q_local = common.secondary_button("Queue locally", "playlist_add")
                q_vast = common.secondary_button("Queue on vast.ai", "cloud_upload")
                web_btn = common.ghost_button("Open RulZ AI Trainer", "open_in_new")
            out = Output()

            def make(job: str) -> Path:
                chosen = [i for i, on in enumerate(cur["checks"]) if on]
                cfg = diagnose.next_config(run, cur["proposals"], chosen)
                cfg["job_name"] = job  # checked by act(); passed in so a later edit cannot change the path
                trainer = common.get_trainer()
                m = knowledge.models.get(mid) if knowledge else None
                header = common.export_header(m.name if m else mid, trainer.version if trainer else None)
                return export.write(cfg, common.export_dir() / f"{cfg['job_name']}.yaml", header)

            async def act(kind: str) -> None:
                job = (job_in.value or "").strip()
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", job):
                    ui.notify("Job name: letters, digits, _ . - only.", type="warning")
                    return
                try:
                    path = await common.io(make, job)
                    if kind == "save":
                        out.show(f"Saved {path}\nLoad it in the RulZ AI Trainer WebUI with 'Config file' -> Load.", True)
                        return
                    trainer = common.get_trainer()
                    if trainer is None:
                        out.show(f"Saved {path}\nRulZ AI Trainer not found: set its folder in Settings.", False)
                        return
                    if kind == "validate":
                        ok, msg = await common.io(bridge.validate, trainer, path)
                    elif kind == "local":
                        ok, msg = await common.io(bridge.queue_local, trainer, path, "RulZ AI Trainer Optimizer")
                    else:
                        folders = run_datasets(export.load_trainer_config(path))
                        if not folders:
                            out.show("The config's dataset folders are not on this PC, so there is nothing to copy "
                                     "to vast.ai.", False)
                            return
                        ok, msg = await common.io(bridge.queue_vast, trainer, path, job, folders)
                    out.show(f"{path}\n{msg}", ok)
                except Exception as e:  # noqa: BLE001
                    out.show(f"Failed: {e}", False)

            async def open_trainer() -> None:
                trainer = common.get_trainer()
                out.show(await common.io(bridge.open_webui, trainer) if trainer else
                         "RulZ AI Trainer not found: set its folder in Settings.", None)

            save_btn.on_click(lambda: act("save"))
            val_btn.on_click(lambda: act("validate"))
            q_local.on_click(lambda: act("local"))
            q_vast.on_click(lambda: act("vast"))
            web_btn.on_click(open_trainer)
            common.ghost_button("Ask an AI why this happened", "psychology", lambda: state.goto("advisor"))


def _checkpoint_near(run: Any, step: int | None) -> Any:
    cks = [c for c in run.checkpoints if c.step is not None]
    if not cks or step is None:
        return None
    best = min(cks, key=lambda c: abs(c.step - step))
    return best if abs(best.step - step) <= max(50, step // 20) else None


def probe_images(run: Any, images: list[Path]) -> None:
    with ui.element("div").classes("rz-probe-grid"):
        for img in images:
            url = common.probe_url(run.path, img)
            if url is None:
                continue
            image = ui.element("img").props(f'src="{url}" alt="{img.name}" loading="lazy"')
            image.on("click", lambda _, u=url: _zoom(u))


def _zoom(url: str) -> None:
    with ui.dialog() as dialog, ui.card().classes("rulz-card p-2"):
        ui.element("img").props(f'src="{url}"').style("max-width: 90vw; max-height: 85vh")
        common.secondary_button("Close", on_click=dialog.close)
    dialog.open()
