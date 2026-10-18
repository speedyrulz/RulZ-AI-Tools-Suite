"""RulzTrainer WebUI (NiceGUI). Run: python webui.py [--port 7870] [--open]

Training runs in a subprocess so Stop frees GPU memory cleanly; the UI polls
<output>/<job>/status.json and train.log.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from nicegui import run, ui

from rulztrainer.config import MODEL_PRESETS, PRESETS, TrainConfig, apply_preset
from rulztrainer import bodysim, facesim
from rulztrainer.losswatch import read_verdicts

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
CUSTOM = "Custom path / repo id"
TARGETS = ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"]
QUANTS = ["auto", "none", "int8", "int8_convrot", "nvfp4"]
SAFETENSORS = [("safetensors", "*.safetensors"), ("all files", "*.*")]

_proc: subprocess.Popen | None = None
_job_dir: Path | None = None


# ---------------------------------------------------------------- backend helpers


def _gpus() -> list[str]:
    try:
        import torch

        return [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    except Exception:  # noqa: BLE001
        return ["cuda:0"]


def _running() -> bool:
    return _proc is not None and _proc.poll() is None


def _relative(path: str) -> str:
    """Store paths inside the project relative to it, so configs stay portable."""
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return path


def tk_dialog(kind: str, initial: str = "", filetypes=None) -> str:
    """Native pick dialog (runs in a worker thread). kind: 'dir' | 'file'."""
    import tkinter as tk
    from tkinter import filedialog

    init = initial if initial and Path(initial).exists() else str(ROOT)
    if Path(init).is_file():
        init = str(Path(init).parent)
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if kind == "dir":
            return filedialog.askdirectory(initialdir=init, parent=root) or ""
        return filedialog.askopenfilename(initialdir=init, filetypes=filetypes or SAFETENSORS, parent=root) or ""
    finally:
        root.destroy()


def start_training(cfg: TrainConfig) -> str:
    global _proc, _job_dir
    if _running():
        return "A training job is already running."
    if not Path(cfg.dataset_dir).is_dir():
        return f"Dataset dir not found: {cfg.dataset_dir}"
    if cfg.model_source == "files":
        for n in ("transformer_file", "text_encoder_file", "vae_file"):
            if getattr(cfg, n) and not Path(getattr(cfg, n)).is_file():
                return f"{n} not found: {getattr(cfg, n)!r}"
    _job_dir = cfg.job_dir()
    _job_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = _job_dir / "config.yaml"
    cfg.save(cfg_path)
    for stale in ("status.json", "STOP"):
        p = _job_dir / stale
        if p.exists():
            p.unlink()
    (_job_dir / "train.log").write_text("", encoding="utf-8")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    console = open(_job_dir / "console.log", "w", encoding="utf-8")  # noqa: SIM115
    _proc = subprocess.Popen([sys.executable, "-m", "rulztrainer.train", "--config", str(cfg_path)],
                             cwd=str(ROOT), env=env, stdout=console, stderr=subprocess.STDOUT)
    return f"Started job '{cfg.job_name}' (pid {_proc.pid}). Config saved to {cfg_path}"


def stop_training() -> str:
    if not _running():
        return "No job running."
    (_job_dir / "STOP").touch()
    return "Stop requested; the trainer will save the LoRA after the current step."


def kill_training() -> str:
    if not _running():
        return "No job running."
    _proc.kill()
    return "Process killed (no final save)."


def read_status() -> dict | None:
    if _job_dir is None:
        return None
    p = _job_dir / "status.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def read_log_tail(n: int = 80) -> str:
    if _job_dir is None:
        return ""
    text = ""
    lp = _job_dir / "train.log"
    if lp.exists():
        text = "\n".join(lp.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    st = read_status() or {}
    if not _running() and (st.get("error") or not st.get("done")):
        cp = _job_dir / "console.log"
        if cp.exists():
            tail = cp.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            text += "\n--- console.log ---\n" + "\n".join(tail)
    return text


def download_model(model_path: str) -> str:
    from rulztrainer.moss.loading import resolve_model_dir

    try:
        return f"Ready: {resolve_model_dir(model_path, 'models')}"
    except Exception as e:  # noqa: BLE001
        return f"Failed: {e}"


def scan_dataset(cfg: TrainConfig):
    from rulztrainer.dataset import scan_dataset as _scan

    items = _scan(cfg)
    sizes: dict[str, int] = {}
    for it in items:
        key = f"{it.width}x{it.height}"
        sizes[key] = sizes.get(key, 0) + 1
    missing = sum(1 for it in items if not it.caption)
    info = f"{len(items)} images. Buckets: {sizes}. Images without caption: {missing}."
    return info, [(str(it.image), it.caption[:120]) for it in items[:24]]


_task_progress = {"text": "", "frac": None, "active": False}


def set_progress(text: str, frac=None) -> None:
    """Called from worker threads (model downloads / loading); the UI timer shows it."""
    _task_progress.update(text=text, frac=frac)


def face_preview(paths: list[str], provider: str, det_size: int) -> tuple[str, list[tuple[str, str]]]:
    """Detect reference faces and save crops under output/_face_refs for the UI. Returns (info, [(crop, caption)])."""
    ok, msg = facesim.available()
    if not ok:
        return msg, []
    scorer = facesim.FaceScorer(provider, int(det_size), progress=set_progress)
    set_progress("Detecting faces...", None)
    paths = [str(p) for p in facesim.expand_refs(paths)]
    if not paths:
        return "No image files found in the listed paths (folders are expanded to the images inside).", []
    info = scorer.set_references(paths)
    out_dir = ROOT / "output" / "_face_refs"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops = []
    for i, item in enumerate(info):
        if item.get("faces"):
            crop = scorer.crop_face(item["image"], item["bbox"], out_dir / f"ref_{i}.jpg")
            crops.append((str(crop), f"{Path(item['image']).name}: {item['faces']} face(s), det {item['det_score']:.2f}"))
        else:
            crops.append(("", f"{Path(item['image']).name}: NO FACE FOUND"))
    self_sim = scorer.ref_self_similarity
    text = (f"{len(scorer.refs)} of {len(paths)} reference images have a face. "
            + (f"Reference self-similarity: {self_sim * 100:.1f}% (the realistic ceiling for samples)."
               if self_sim is not None else "Add 2+ references to see the self-similarity ceiling."))
    return text, crops


def body_preview(paths: list[str], backend: str = "mediapipe", sam3_model: str = "facebook/sam3",
                 device: str = "cuda:0") -> str:
    ok, msg = bodysim.available()
    if not ok:
        return msg
    sil = None
    note = ""
    if backend == "mediapipe+sam3":
        try:
            sil = bodysim.SilhouetteScorer(sam3_model, device, progress=set_progress)
        except Exception as e:  # noqa: BLE001
            note = f" (SAM 3 unavailable: {str(e)[:120]}; skeleton only)"
    scorer = bodysim.BodyScorer("models", silhouette=sil, progress=set_progress)
    set_progress("Detecting poses...", None)
    paths = [str(p) for p in facesim.expand_refs(paths)]
    if not paths:
        return "No image files found in the listed paths (folders are expanded to the images inside)."
    info = scorer.set_references(paths)
    found = [i for i in info if i.get("ratios")]
    text = f"{len(found)} of {len(paths)} body references have a usable pose."
    if scorer.profile:
        text += " Profile: " + ", ".join(f"{k} {v:.2f}" for k, v in scorer.profile.items())
    if scorer.ref_self_similarity is not None:
        text += f". Reference self-similarity {scorer.ref_self_similarity * 100:.1f}%."
    return text + note


def score_caption(sc: dict) -> str:
    parts = []
    parts.append(f"face {sc['percent']:.0f}%" if sc.get("percent") is not None else "no face")
    if sc.get("body_percent") is not None:
        parts.append(f"body {sc['body_percent']:.0f}%")
    if sc.get("combined_percent") is not None:
        parts.append(f"combined {sc['combined_percent']:.0f}%")
    return "  |  " + ", ".join(parts)


def face_scores_by_image(job_dir: Path | None) -> dict[str, dict]:
    if job_dir is None:
        return {}
    return {Path(sc["image"]).name: sc for sc in facesim.read_scores(job_dir)}


def analyse_and_preview(cfg_dict: dict, what: str) -> tuple[str, list[tuple[str, str]]]:
    """Run face analysis on the dataset in the UI process; render region overlays or crop thumbnails."""
    from rulztrainer import regions as R
    from rulztrainer.dataset import augment_items, load_image_fit, scan_dataset

    cfg = TrainConfig.from_dict(cfg_dict)
    ok, msg = facesim.available()
    if not ok:
        return msg, []
    items = scan_dataset(cfg)
    det = facesim.FaceScorer(cfg.face_provider, cfg.face_det_size, progress=set_progress)
    set_progress("Analysing faces...", None)
    analysis = R.analyze_faces(items, det, cfg.resolved_cache_dir() / "faces.json",
                               progress=lambda i, n: set_progress(f"Analysing faces {i}/{n}", i / max(1, n)))
    n_face = sum(1 for r in analysis.values() if r.get("bbox"))
    out_dir = ROOT / "output" / ("_regions" if what == "regions" else "_crops")
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs: list[tuple[str, str]] = []
    if what == "crops":
        all_items = augment_items(items, cfg, analysis)
        crops = [it for it in all_items if it.category != "image"]
        for i, it in enumerate(crops[:24]):
            img = load_image_fit(it.image, it.width, it.height, it.src_box)
            img.thumbnail((256, 256))
            pth = out_dir / f"crop_{i}.jpg"
            img.save(pth, "JPEG", quality=85)
            thumbs.append((str(pth), f"{it.category} of {it.image.name}: {it.caption[:60]}"))
        text = (f"{n_face} of {len(items)} images have a face; crop_augment={cfg.crop_augment} would add "
                f"{sum(1 for c in crops if c.category == 'face_crop')} face and "
                f"{sum(1 for c in crops if c.category == 'bust_crop')} bust crops.")
    else:
        for i, it in enumerate(items[:12]):
            rec = analysis.get(str(it.image.resolve()))
            bbox = R.remap_bbox(rec["bbox"], rec["W"], rec["H"], it.width, it.height) if rec and rec.get("bbox") else None
            rm = R.build_weight_map(it.width, it.height, bbox, None, cfg.face_loss_weight, cfg.body_loss_weight,
                                    cfg.background_loss_weight, cfg.face_margin, cfg.region_feather_cells)
            img = load_image_fit(it.image, it.width, it.height)
            ov = R.render_overlay(img, rm, it.width, it.height)
            ov.thumbnail((256, 256))
            pth = out_dir / f"region_{i}.jpg"
            ov.save(pth, "JPEG", quality=85)
            thumbs.append((str(pth), f"{it.image.name}: {'face' if rm.face_cells else 'no face'}"))
        text = (f"{n_face} of {len(items)} images have a face. Overlay: brighter red = higher loss weight "
                f"(face {cfg.face_loss_weight:g} / background {cfg.background_loss_weight:g}; person masks are "
                f"built during training).")
    return text, thumbs


def gpu_stats() -> list[dict]:
    """Per-GPU utilization and memory via nvidia-smi (no CUDA context in the UI process)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout
    except Exception:  # noqa: BLE001
        return []
    stats = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            stats.append({"index": int(parts[0]), "name": parts[1], "util": int(float(parts[2])),
                          "used": float(parts[3]), "total": float(parts[4])})
        except ValueError:
            continue
    return stats


# ---------------------------------------------------------------- UI


class Form:
    """Holds the NiceGUI widgets and converts to/from TrainConfig."""

    def __init__(self):
        self.w: dict[str, object] = {}

    def value(self, key):
        return self.w[key].value

    def to_config(self) -> TrainConfig:
        v = self.value
        preset = v("model_preset")
        model_path = (v("model_path") or "").strip() if preset == CUSTOM else MODEL_PRESETS[preset]
        prompts = [p.strip() for p in (v("sample_prompts") or "").splitlines() if p.strip()]
        s = lambda k: (v(k) or "").strip()  # noqa: E731
        return TrainConfig(
            preset="identity" if self.w.get("preset_toggle") is not None and self.w["preset_toggle"].value else "",
            job_name=s("job_name") or "my_lora", output_dir=s("output_dir") or "output", seed=int(v("seed")),
            model_source="files" if v("model_source") == "Single safetensors files (ComfyUI)" else "hf",
            model_path=model_path, transformer_file=s("transformer_file"), text_encoder_file=s("text_encoder_file"),
            vae_file=s("vae_file"), tokenizer_path=s("tokenizer_path"),
            device=v("device"), text_encoder_device=v("text_encoder_device"),
            multi_gpu=v("multi_gpu"), devices=(v("devices") or "").strip(), split_plan=(v("split_plan") or "").strip(),
            transformer_quant=v("transformer_quant"), text_encoder_quant=v("text_encoder_quant"),
            gradient_checkpointing=bool(v("gradient_checkpointing")),
            dataset_dir=s("dataset_dir"), resolution=int(v("resolution")),
            enable_buckets=bool(v("enable_buckets")), flip_augment=bool(v("flip_augment")),
            trigger_word=s("trigger_word"), caption_dropout=float(v("caption_dropout")),
            max_token_length=int(v("max_token_length")), recache=bool(v("recache")),
            lora_rank=int(v("lora_rank")), lora_alpha=float(v("lora_alpha")),
            lora_targets=[t for t in TARGETS if self.w[f"target_{t}"].value],
            train_text_embedder=bool(v("train_text_embedder")), resume_from=s("resume_from"),
            lora_blocks=s("lora_blocks"),
            lora_save_dtype=v("lora_save_dtype"), save_state=bool(v("save_state")), resume_state=s("resume_state"),
            save_state_every=int(v("save_state_every")), keep_states=int(v("keep_states")),
            compile_model=bool(v("compile_model")), sample_seed=int(v("sample_seed")),
            sample_walk_seed=bool(v("sample_walk_seed")), sample_before_training=bool(v("sample_before_training")),
            steps=int(v("steps")), batch_size=int(v("batch_size")),
            gradient_accumulation=int(v("gradient_accumulation")), learning_rate=float(v("learning_rate")),
            optimizer=v("optimizer"), lr_scheduler=v("lr_scheduler"), warmup_steps=int(v("warmup_steps")),
            max_grad_norm=float(v("max_grad_norm")), timestep_sampling=v("timestep_sampling"),
            automagic_min_lr=float(v("automagic_min_lr")), automagic_max_lr=float(v("automagic_max_lr")),
            automagic_polarity_history=int(v("automagic_polarity_history")),
            automagic_clip_threshold=float(v("automagic_clip_threshold")),
            timestep_shift=float(v("timestep_shift")), sigmoid_scale=float(v("sigmoid_scale")),
            logit_mean=float(v("logit_mean")), logit_std=float(v("logit_std")), loss_weighting=v("loss_weighting"),
            save_every=int(v("save_every")), sample_every=int(v("sample_every")), sample_prompts=prompts,
            sample_width=int(v("sample_width")), sample_height=int(v("sample_height")),
            sample_steps=int(v("sample_steps")), sample_cfg=float(v("sample_cfg")),
            sample_shift=float(v("sample_shift")), sample_negative=v("sample_negative") or "",
            turbo_lora=s("turbo_lora"), turbo_lora_strength=float(v("turbo_lora_strength")),
            sample_model_path=s("sample_model_path"), sample_model_quant=v("sample_model_quant"),
            sample_lora=s("sample_lora"), sample_lora_strength=float(v("sample_lora_strength")),
            face_refs=[p.strip() for p in (v("face_refs") or "").splitlines() if p.strip()],
            face_provider=v("face_provider"), face_det_size=int(v("face_det_size")),
            body_refs=[p.strip() for p in (v("body_refs") or "").splitlines() if p.strip()],
            face_weight=float(v("face_weight")), body_weight=float(v("body_weight")),
            body_backend=v("body_backend"), sam3_model=(v("sam3_model") or "facebook/sam3").strip(),
            body_device=v("body_device"),
            probe_every=int(v("probe_every")), probe_at_start=bool(v("probe_at_start")),
            probe_face_prompt=s("probe_face_prompt"), probe_body_prompt=s("probe_body_prompt"),
            probe_seeds=int(v("probe_seeds")), probe_seed=int(v("probe_seed")), probe_size=int(v("probe_size")),
            probe_steps=int(v("probe_steps")), probe_cfg=float(v("probe_cfg")), probe_ema=float(v("probe_ema")),
            likeness_noise_band=float(v("likeness_noise_band")), likeness_track_best=bool(v("likeness_track_best")),
            likeness_patience=int(v("likeness_patience")), likeness_lr_control=bool(v("likeness_lr_control")),
            likeness_lr_min=float(v("likeness_lr_min")), likeness_lr_max=float(v("likeness_lr_max")),
            likeness_lr_step=float(v("likeness_lr_step")), likeness_curriculum=bool(v("likeness_curriculum")),
            likeness_curriculum_strength=float(v("likeness_curriculum_strength")),
            loss_watch=bool(v("loss_watch")), per_image_lr=bool(v("per_image_lr")),
            loss_watch_buckets=int(v("loss_watch_buckets")), loss_watch_beta=float(v("loss_watch_beta")),
            loss_watch_high=float(v("loss_watch_high")), loss_watch_low=float(v("loss_watch_low")),
            loss_watch_improve_tol=float(v("loss_watch_improve_tol")),
            loss_watch_healthy_mult=float(v("loss_watch_healthy_mult")),
            loss_watch_exclude_after=int(v("loss_watch_exclude_after")),
            region_weights=bool(v("region_weights")), face_loss_weight=float(v("face_loss_weight")),
            body_loss_weight=float(v("body_loss_weight")), background_loss_weight=float(v("background_loss_weight")),
            face_margin=float(v("face_margin")), region_feather_cells=int(v("region_feather_cells")),
            region_prior_weight=float(v("region_prior_weight")),
            content_power=float(v("content_power")), sigma_min=float(v("sigma_min")), sigma_max=float(v("sigma_max")),
            lora_plus_ratio=float(v("lora_plus_ratio")),
            crop_augment=v("crop_augment"), crop_augment_min_px=int(v("crop_augment_min_px")),
            crop_face_scale=float(v("crop_face_scale")), crop_bust_scale=float(v("crop_bust_scale")),
            crop_skip_face_frac=float(v("crop_skip_face_frac")), crop_skip_cover=float(v("crop_skip_cover")),
            crop_face_caption=v("crop_face_caption") or "", crop_bust_caption=v("crop_bust_caption") or "",
            crop_augment_weight=float(v("crop_augment_weight")),
        )

    def from_config(self, cfg: TrainConfig) -> None:
        preset = next((k for k, v in MODEL_PRESETS.items() if v == cfg.model_path), CUSTOM)
        values = {**cfg.to_dict(), "model_preset": preset, "sample_prompts": "\n".join(cfg.sample_prompts),
                  "face_refs": "\n".join(cfg.face_refs), "body_refs": "\n".join(cfg.body_refs),
                  "model_source": "Single safetensors files (ComfyUI)" if cfg.model_source == "files"
                  else "Hugging Face repo / diffusers folder"}
        for t in TARGETS:
            values[f"target_{t}"] = t in cfg.lora_targets
        for key, widget in self.w.items():
            if key in values:
                widget.value = values[key]
        if "preset_toggle" in self.w and cfg.preset == "identity" and not self.w["preset_toggle"].value:
            self.w["preset_toggle"].value = True  # loading a preset config shows the switch on (no re-apply)


def path_input(f: Form, key: str, label: str, kind: str, value: str = "", width: str = "w-96", filetypes=None):
    """Text input with a Browse button that opens a native dialog."""
    with ui.row().classes("items-end gap-1"):
        inp = ui.input(label, value=value).classes(width)
        f.w[key] = inp

        async def browse():
            picked = await run.io_bound(tk_dialog, kind, inp.value or "", filetypes)
            if picked:
                inp.value = _relative(picked)

        ui.button(icon="folder_open" if kind == "dir" else "description", on_click=browse).props("flat dense") \
            .tooltip("Browse")
    return inp


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PAGE_SIZE = 24
THUMB_SIZE = 512


def list_dataset_images(folder: str) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and ".rulz_cache" not in p.parts)


def make_thumbs(images: list[Path], root: Path) -> list[Path]:
    """Create (or reuse) small JPEG thumbnails under <dataset>/.rulz_cache/thumbs. Runs in a worker thread."""
    import hashlib

    from PIL import Image, ImageOps

    tdir = root / ".rulz_cache" / "thumbs"
    tdir.mkdir(parents=True, exist_ok=True)
    out = []
    for img in images:
        try:
            st = img.stat()
            key = hashlib.sha1(f"{img.resolve()}|{st.st_mtime_ns}|{st.st_size}|sq".encode()).hexdigest()[:16]
            t = tdir / f"{key}.jpg"
            if not t.exists():
                with Image.open(img) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im = ImageOps.fit(im, (THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)  # square center crop
                    im.save(t, "JPEG", quality=85)
            out.append(t)
        except Exception:  # noqa: BLE001
            out.append(img)
    return out


def image_viewer():
    """Full-window image viewer dialog. Returns show(path) callable."""
    with ui.dialog().props("maximized") as dlg, ui.card().tight().classes("w-full h-full bg-black"):
        with ui.row().classes("w-full items-center justify-between px-3 py-1"):
            title = ui.label("").classes("text-sm text-white truncate")
            ui.button(icon="close", on_click=dlg.close).props("flat dense round color=white")
        big = ui.image("").props("fit=contain").classes("w-full").style("height: calc(100vh - 48px)")
        big.on("click", dlg.close)

    def show(path: Path):
        title.text = str(path)
        big.set_source(str(path))
        dlg.open()

    return show


def build_dataset_tab(f: Form):
    """Upload images/captions, view them in a grid and edit captions (saved to .txt on change)."""
    state = {"page": 0, "images": [], "busy": False}
    with ui.row().classes("w-full items-end gap-2"):
        folder = ui.input("Dataset folder", value=f.w["dataset_dir"].value).classes("w-96")
        folder.bind_value(f.w["dataset_dir"], "value")  # keep in sync with the Train tab

        async def browse():
            picked = await run.io_bound(tk_dialog, "dir", folder.value or "")
            if picked:
                folder.value = _relative(picked)
                await refresh()

        ui.button(icon="folder_open", on_click=browse).props("flat dense").tooltip("Browse")
        ui.button("Load / refresh", on_click=lambda: refresh())
        info = ui.label("").classes("text-sm")
        spinner = ui.spinner(size="sm")
        spinner.visible = False
    with ui.row().classes("w-full items-center gap-2"):
        def on_upload(e):
            dest = Path(folder.value or "datasets/new_dataset")
            dest.mkdir(parents=True, exist_ok=True)
            name = Path(e.name).name
            if Path(name).suffix.lower() not in IMAGE_EXTS | {".txt"}:
                ui.notify(f"Skipped {name}: not an image or .txt", type="warning")
                return
            (dest / name).write_bytes(e.content.read())

        ui.upload(label="Upload images and .txt captions (drop files here)", multiple=True, auto_upload=True,
                  on_upload=on_upload, on_multi_upload=lambda e: refresh()).props("accept=.png,.jpg,.jpeg,.webp,.bmp,.txt")             .classes("w-[36rem]")
        with ui.row().classes("items-center"):
            ui.button(icon="chevron_left", on_click=lambda: page(-1)).props("flat dense")
            page_label = ui.label("")
            ui.button(icon="chevron_right", on_click=lambda: page(1)).props("flat dense")
            columns = ui.select([2, 3, 4, 5, 6], value=4, label="Columns").classes("w-24")
    grid = ui.grid(columns=4).classes("w-full gap-3")
    show_full = image_viewer()

    def ref_set(key: str) -> set[str]:
        return {p.strip() for p in (f.w[key].value or "").splitlines() if p.strip()}

    def toggle_ref(key: str, path: Path, on: bool):
        refs = ref_set(key)
        rel = _relative(str(path))
        (refs.add if on else refs.discard)(rel)
        f.w[key].value = "\n".join(sorted(refs))

    def save_caption(path: Path, text: str, status_label):
        cap = path.with_suffix(".txt")
        try:
            cap.write_text(text, encoding="utf-8")  # creates the file if it did not exist
            status_label.text = "saved" if text.strip() else "saved (empty)"
        except OSError as e:
            status_label.text = f"error: {e}"
            ui.notify(f"Could not save {cap.name}: {e}", type="negative")

    async def render():
        imgs = state["images"]
        n_pages = max(1, (len(imgs) + PAGE_SIZE - 1) // PAGE_SIZE)
        state["page"] = min(state["page"], n_pages - 1)
        page_label.text = f"page {state['page'] + 1} / {n_pages}"
        start = state["page"] * PAGE_SIZE
        page_imgs = imgs[start:start + PAGE_SIZE]
        spinner.visible = True
        thumbs = await run.io_bound(make_thumbs, page_imgs, Path(folder.value))
        spinner.visible = False
        verdicts = read_verdicts(_job_dir)
        badge_color = {"stuck": "red", "excluded": "grey", "suspect": "orange", "exhausted": "blue", "easy": "green",
                       "healthy": "teal"}
        grid.style(f"grid-template-columns: repeat({columns.value}, minmax(0, 1fr))")
        grid.clear()
        with grid:
            for img, thumb in zip(page_imgs, thumbs):
                cap = img.with_suffix(".txt")
                text = cap.read_text(encoding="utf-8") if cap.is_file() else ""
                with ui.card().tight().classes("w-full"):
                    ui.image(str(thumb)).props("fit=cover ratio=1").classes("w-full cursor-pointer")                         .on("click", lambda _, p=img: show_full(p)).tooltip("Click to view full size")
                    with ui.row().classes("w-full items-center justify-between px-2 pt-1"):
                        ui.label(img.name).classes("text-xs truncate")
                        vd = verdicts.get(str(img)) or verdicts.get(_relative(str(img)))
                        if vd and vd != "new":
                            ui.badge(vd, color=badge_color.get(vd, "grey")).props("dense").tooltip("loss watch verdict")
                        ui.checkbox("Face ref", value=_relative(str(img)) in ref_set("face_refs"),
                                    on_change=lambda e, p=img: toggle_ref("face_refs", p, e.value)).props("dense") \
                            .classes("text-xs").tooltip("Use this image's face as a similarity reference")
                        ui.checkbox("Body ref", value=_relative(str(img)) in ref_set("body_refs"),
                                    on_change=lambda e, p=img: toggle_ref("body_refs", p, e.value)).props("dense") \
                            .classes("text-xs").tooltip("Use this image's body proportions as a reference")
                        status_label = ui.label("" if cap.is_file() else "no caption yet").classes("text-xs text-grey")
                    ui.textarea(value=text, placeholder="type a caption; it is saved to the .txt automatically",
                                on_change=lambda e, p=img, sl=status_label: save_caption(p, e.value, sl))                         .props("dense outlined autogrow").classes("w-full px-2 pb-2 text-xs")

    async def refresh():
        if state["busy"]:
            return
        state["busy"] = True
        try:
            state["images"] = await run.io_bound(list_dataset_images, folder.value or "")
            missing = sum(1 for p in state["images"] if not p.with_suffix(".txt").is_file())
            info.text = (f"{len(state['images'])} images, {missing} without caption" if state["images"]
                         else "No images found in that folder")
            await render()
        finally:
            state["busy"] = False

    async def page(delta: int):
        state["page"] = max(0, state["page"] + delta)
        await render()

    columns.on_value_change(lambda e: render())
    ui.timer(0.1, refresh, once=True)


def build_samples_tab(f: Form):
    """Live grid of generated sample images for the current (or any) job folder."""
    state = {"files": [], "columns": None}
    with ui.row().classes("w-full items-end gap-2"):
        folder = ui.input("Job folder (output/<job>)", value="").classes("w-96")

        async def browse():
            picked = await run.io_bound(tk_dialog, "dir", folder.value or str(ROOT / "output"))
            if picked:
                folder.value = _relative(picked)
                state["files"] = []

        ui.button(icon="folder_open", on_click=browse).props("flat dense").tooltip("Browse")
        follow = ui.checkbox("Follow current job", value=True)
        columns = ui.select([2, 3, 4, 6], value=4, label="Columns").classes("w-28")
        info = ui.label("").classes("text-sm")
    grid = ui.grid(columns=4).classes("w-full gap-3")
    show_full = image_viewer()

    def samples_dir() -> Path | None:
        if follow.value:
            jd = _job_dir or (Path(f.w["output_dir"].value or "output") / (f.w["job_name"].value or "my_lora"))
            rel = _relative(str(jd))
            if folder.value != rel:
                folder.value = rel
        if not folder.value:
            return None
        p = Path(folder.value)
        return p if p.name == "samples" else p / "samples"

    def refresh():
        sd = samples_dir()
        files = sorted(sd.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True) if sd and sd.is_dir() else []
        sig = [(str(p), p.stat().st_mtime) for p in files]
        fs_file = (sd.parent / "face_scores.jsonl") if sd else None
        if fs_file is not None and fs_file.exists():
            sig.append(("face_scores", fs_file.stat().st_mtime))
        if sig == state["files"] and state["columns"] == columns.value:
            return
        state["files"] = sig
        state["columns"] = columns.value
        grid.style(f"grid-template-columns: repeat({columns.value}, minmax(0, 1fr))")
        info.text = f"{len(files)} samples in {sd}" if sd else "No job selected"
        scores = face_scores_by_image(sd.parent if sd else None)
        grid.clear()
        with grid:
            for p in files[:120]:
                with ui.card().tight().classes("w-full"):
                    ui.image(str(p)).props("fit=contain").classes("w-full h-64 bg-black cursor-pointer")                         .on("click", lambda _, q=p: show_full(q)).tooltip("Click to view full size")
                    sc = scores.get(p.name)
                    label = p.name + (score_caption(sc) if sc is not None else "")
                    ui.label(label).classes("text-xs px-2 py-1")

    columns.on_value_change(lambda e: (state.__setitem__("files", []), refresh()))
    ui.timer(3.0, refresh)
    refresh()


def build_page():
    d = TrainConfig()
    gpus = _gpus()
    f = Form()
    w = f.w

    ui.page_title("RulzTrainer")
    with ui.header().classes("items-center"):
        ui.label("RulzTrainer").classes("text-xl font-bold")
        ui.label("LoRA trainer for MOSS-Image (Base / SFT). Output LoRAs load directly in ComfyUI.").classes("text-sm")

    with ui.tabs().classes("w-full") as tabs:
        tab_train = ui.tab("Train")
        tab_dataset = ui.tab("Dataset")
        tab_monitor = ui.tab("Monitor")
        tab_samples = ui.tab("Samples")
        tab_help = ui.tab("Help")

    with ui.tab_panels(tabs, value=tab_train).classes("w-full").props("keep-alive"):
        # ------------------------------------------------------------ Train
        with ui.tab_panel(tab_train):
            with ui.row().classes("w-full items-end"):
                w["job_name"] = ui.input("Job name", value=d.job_name).classes("w-56")
                path_input(f, "output_dir", "Output dir", "dir", d.output_dir, "w-72")
                w["seed"] = ui.number("Seed", value=d.seed, precision=0).classes("w-32")
                preset_state = {"snapshot": None, "applying": False}
                preset_note = ui.label("").classes("text-xs")

                def on_preset_toggle(e):
                    if preset_state["applying"]:
                        return
                    preset_state["applying"] = True
                    try:
                        if e.value:
                            preset_state["snapshot"] = f.to_config()
                            f.from_config(TrainConfig.from_dict(apply_preset(f.to_config().to_dict(), "identity")))
                            keys = sorted(PRESETS["identity"])
                            preset_note.text = (f"Identity preset applied ({len(keys)} settings: region loss, LoRA+ 4, "
                                                "adamw8bit, crops without upscaling, SAM 3 masks, probes every 25, "
                                                "best/early-stop, curriculum 0.5, loss watch + per-image lr). "
                                                "Set face/body references and check the probe prompts.")
                        elif preset_state["snapshot"] is not None:
                            f.from_config(preset_state["snapshot"])
                            preset_note.text = "Previous settings restored."
                    finally:
                        preset_state["applying"] = False

                w["preset_toggle"] = ui.switch("Identity training preset", value=False, on_change=on_preset_toggle) \
                    .tooltip("Applies the measured recipe for character LoRAs; switch off to restore your settings")

            with ui.expansion("Model", value=True).classes("w-full"):
                w["model_source"] = ui.radio(["Hugging Face repo / diffusers folder",
                                              "Single safetensors files (ComfyUI)"],
                                             value="Hugging Face repo / diffusers folder").props("inline")
                with ui.column().classes("w-full").bind_visibility_from(
                        w["model_source"], "value", lambda x: x.startswith("Hugging")):
                    with ui.row().classes("w-full items-end"):
                        w["model_preset"] = ui.select(list(MODEL_PRESETS) + [CUSTOM], value="MOSS-Image-Base",
                                                      label="Model").classes("w-56")
                        w["model_path"] = ui.input("Custom path or HF repo id", value=d.model_path).classes("w-96")
                        dl_status = ui.label("").classes("text-sm")

                        async def on_download():
                            dl_status.text = "Downloading / verifying... (about 30 GB on first download)"
                            cfg = f.to_config()
                            dl_status.text = await run.io_bound(download_model, cfg.model_path)

                        ui.button("Download / verify model", on_click=on_download)
                with ui.column().classes("w-full").bind_visibility_from(
                        w["model_source"], "value", lambda x: x.startswith("Single")):
                    path_input(f, "transformer_file", "Diffusion model (.safetensors: bf16, fp8, int8, nvfp4)", "file",
                               "", "w-[40rem]")
                    path_input(f, "text_encoder_file", "Text encoder (Qwen3.5-9B .safetensors)", "file", "", "w-[40rem]")
                    path_input(f, "vae_file", "VAE (Flux2 .safetensors)", "file", "", "w-[40rem]")
                    path_input(f, "tokenizer_path", "Tokenizer folder (optional; auto-downloads if empty)", "dir",
                               "", "w-[40rem]")
                    ui.label("Any file left empty is taken from the Hugging Face model above.").classes("text-xs")
                with ui.row().classes("w-full items-end"):
                    w["multi_gpu"] = ui.select({"off": "single GPU", "data_parallel": "2+ GPUs: data parallel (faster)",
                                                "model_split": "2+ GPUs: model split (memory only)"},
                                               value=d.multi_gpu, label="Multi-GPU").classes("w-72") \
                        .tooltip("data parallel: a model copy per GPU, each trains its own images, LoRA grads are "
                                 "averaged (about 1.7-2x throughput). model split: blocks divided across GPUs, half "
                                 "the memory per GPU, not faster.")
                    w["devices"] = ui.input("Devices (first = primary)", value=",".join(gpus[:2]) if len(gpus) > 1 else d.devices) \
                        .classes("w-56").bind_visibility_from(w["multi_gpu"], "value", lambda x: x != "off")
                    w["split_plan"] = ui.input("Split plan (empty = auto-balance), e.g. cuda:0=0-25;cuda:1=26-41",
                                               value=d.split_plan).classes("w-[28rem]") \
                        .bind_visibility_from(w["multi_gpu"], "value", lambda x: x == "model_split") \
                        .tooltip("Give the faster GPU more blocks. Auto-balance counts blocks before the first trained "
                                 "block as forward-only (cheap) and every later block as forward+backward.")
                with ui.row().classes("w-full items-end"):
                    w["device"] = ui.select(gpus, value=gpus[0], label="Training GPU").classes("w-40")
                    w["text_encoder_device"] = ui.select(["auto"] + gpus + ["cpu"], value="auto",
                                                         label="Text encoder device (caching)").classes("w-56")
                    w["transformer_quant"] = ui.select(QUANTS, value="auto", label="Diffusion model quant") \
                        .classes("w-44").tooltip("auto keeps the file's format; int8/int8_convrot/nvfp4 quantize on load")
                    w["text_encoder_quant"] = ui.select(QUANTS, value="auto", label="Text encoder quant").classes("w-44")
                    w["gradient_checkpointing"] = ui.checkbox("Gradient checkpointing", value=True)
                    w["compile_model"] = ui.checkbox("Compile model (torch.compile, needs triton)", value=False)

            with ui.expansion("Dataset", value=True).classes("w-full"):
                with ui.row().classes("w-full items-end"):
                    path_input(f, "dataset_dir", "Dataset dir (images + .txt captions)", "dir", d.dataset_dir)
                    with ui.column().classes("w-72"):
                        ui.label("Resolution (target area = res^2)").classes("text-xs")
                        w["resolution"] = ui.slider(min=512, max=2048, step=64, value=d.resolution)
                        ui.label().bind_text_from(w["resolution"], "value", lambda x: f"{int(x)} px")
                with ui.row().classes("w-full items-end"):
                    w["enable_buckets"] = ui.checkbox("Aspect-ratio buckets", value=True)
                    w["flip_augment"] = ui.checkbox("Horizontal flip augment", value=False)
                    w["trigger_word"] = ui.input("Trigger word (prepended to captions)").classes("w-64")
                    w["caption_dropout"] = ui.number("Caption dropout", value=d.caption_dropout, min=0, max=0.5,
                                                     step=0.01).classes("w-36")
                    w["max_token_length"] = ui.number("Max caption tokens", value=d.max_token_length, min=64,
                                                      max=4096, step=64, precision=0).classes("w-40")
                    w["recache"] = ui.checkbox("Force re-cache", value=False)
                scan_info = ui.label("").classes("text-sm")
                gallery = ui.row().classes("w-full gap-2")

                async def on_scan():
                    cfg = f.to_config()
                    scan_info.text = "Scanning..."
                    try:
                        info, items = await run.io_bound(scan_dataset, cfg)
                    except Exception as e:  # noqa: BLE001
                        scan_info.text = f"Error: {e}"
                        return
                    scan_info.text = info
                    gallery.clear()
                    with gallery:
                        for path, caption in items:
                            with ui.card().tight().classes("w-32"):
                                ui.image(path).classes("w-32 h-32 object-cover")
                                ui.tooltip(caption or "(no caption)")

                ui.button("Scan dataset", on_click=on_scan)
                with ui.row().classes("w-full items-end"):
                    w["crop_augment"] = ui.select(["none", "face", "face+bust"], value=d.crop_augment,
                                                  label="Auto crops (character LoRAs)").classes("w-48") \
                        .tooltip("Adds a face crop and/or a bust crop of every image with a detected face as extra "
                                 "training items, with templated captions. Needs insightface.")
                    w["crop_augment_min_px"] = ui.number("Min crop px (0 = resolution)", value=d.crop_augment_min_px, precision=0).classes("w-40")
                    w["crop_face_scale"] = ui.number("Face crop x face size", value=d.crop_face_scale, step=0.5).classes("w-40")
                    w["crop_bust_scale"] = ui.number("Bust crop x face size", value=d.crop_bust_scale, step=0.5).classes("w-40")
                    w["crop_augment_weight"] = ui.number("Crop sampling weight", value=d.crop_augment_weight, step=0.1).classes("w-40")
                with ui.row().classes("w-full items-end").bind_visibility_from(w["crop_augment"], "value", lambda x: x != "none"):
                    w["crop_face_caption"] = ui.input("Face crop caption ({caption}, {raw}, {trigger})",
                                                      value=d.crop_face_caption).classes("w-96")
                    w["crop_bust_caption"] = ui.input("Bust crop caption", value=d.crop_bust_caption).classes("w-96")
                    w["crop_skip_face_frac"] = ui.number("Skip face crop if face > frac", value=d.crop_skip_face_frac,
                                                         step=0.05).classes("w-44")
                    w["crop_skip_cover"] = ui.number("Skip crop covering > frac", value=d.crop_skip_cover, step=0.05).classes("w-44")
                crop_info = ui.label("").classes("text-sm")
                crop_row = ui.row().classes("w-full gap-2")

                async def on_crop_preview():
                    crop_info.text = "Working..."
                    _task_progress.update(text="Preparing face detector...", frac=None, active=True)
                    try:
                        text, thumbs = await run.io_bound(analyse_and_preview, f.to_config().to_dict(), "crops")
                    except Exception as e:  # noqa: BLE001
                        crop_info.text = f"Error: {e}"
                        return
                    finally:
                        _task_progress.update(active=False)
                    crop_info.text = text
                    crop_row.clear()
                    with crop_row:
                        for pth, cap in thumbs:
                            with ui.card().tight().classes("w-32"):
                                ui.image(pth).props("fit=contain").classes("w-32 h-32 bg-black")
                                ui.tooltip(cap)

                ui.button("Analyse faces & preview crops", on_click=on_crop_preview)

            with ui.expansion("LoRA", value=True).classes("w-full"):
                with ui.row().classes("w-full items-end"):
                    w["lora_rank"] = ui.number("Rank", value=d.lora_rank, min=1, max=256, precision=0).classes("w-28")
                    w["lora_alpha"] = ui.number("Alpha", value=d.lora_alpha).classes("w-28")
                    with ui.row().classes("items-center"):
                        ui.label("Targets:").classes("text-sm")
                        for t in TARGETS:
                            w[f"target_{t}"] = ui.checkbox(t, value=t in d.lora_targets)
                    w["train_text_embedder"] = ui.checkbox("Also train DiT text projection blocks", value=False)
                    w["lora_save_dtype"] = ui.select(["bf16", "fp16", "fp32"], value=d.lora_save_dtype,
                                                     label="Save LoRA as").classes("w-32")
                    w["lora_plus_ratio"] = ui.number("LoRA+ ratio (0 = off, 16 = paper)", value=d.lora_plus_ratio,
                                                     step=1).classes("w-56") \
                        .tooltip("lora_up learns at lr x ratio (faster convergence); ignored by automagic3")
                with ui.row().classes("w-full items-end"):
                    w["lora_blocks"] = ui.input("Blocks to train (e.g. 0-41, 21-41, 10-31,40, even; empty = all)",
                                                value=d.lora_blocks).classes("w-[32rem]")
                    ui.label("Training only later blocks also skips the backward pass through earlier ones.").classes("text-xs")
                block_boxes: list = []
                N_BLOCKS = 42

                def blocks_from_text():
                    from rulztrainer.lora import parse_blocks
                    try:
                        bl = parse_blocks(w["lora_blocks"].value or "", N_BLOCKS)
                    except ValueError:
                        return
                    sel = set(range(N_BLOCKS)) if bl is None else set(bl)
                    for i, cb in enumerate(block_boxes):
                        if cb.value != (i in sel):
                            cb.value = i in sel

                def text_from_blocks():
                    from rulztrainer.lora import format_blocks
                    sel = [i for i, cb in enumerate(block_boxes) if cb.value]
                    txt = "" if len(sel) == N_BLOCKS else format_blocks(sel)
                    if (w["lora_blocks"].value or "") != txt:
                        w["lora_blocks"].value = txt

                def set_blocks(sel: set[int]):
                    for i, cb in enumerate(block_boxes):
                        cb.value = i in sel
                    text_from_blocks()

                with ui.row().classes("w-full items-center gap-1"):
                    ui.label("Presets:").classes("text-xs")
                    for label, sel in (("All", set(range(N_BLOCKS))), ("None", set()),
                                       ("First half 0-20", set(range(0, 21))), ("Second half 21-41", set(range(21, 42))),
                                       ("Middle 10-31", set(range(10, 32))), ("Even", set(range(0, N_BLOCKS, 2))),
                                       ("Odd", set(range(1, N_BLOCKS, 2)))):
                        ui.button(label, on_click=lambda _, s_=sel: set_blocks(s_)).props("dense flat size=sm")
                    ui.button("Invert", on_click=lambda: set_blocks({i for i, cb in enumerate(block_boxes) if not cb.value})) \
                        .props("dense flat size=sm")
                with ui.grid(columns=14).classes("w-full gap-0"):
                    for i in range(N_BLOCKS):
                        block_boxes.append(ui.checkbox(str(i), value=True, on_change=lambda e: text_from_blocks())
                                           .props("dense size=xs").classes("text-xs"))
                w["lora_blocks"].on_value_change(lambda e: blocks_from_text())
                path_input(f, "resume_from", "Resume from LoRA weights (.safetensors, optional)", "file", "", "w-[40rem]")
                with ui.row().classes("w-full items-end"):
                    w["save_state"] = ui.checkbox("Save training state (optimizer, scheduler, RNG); final state always saved",
                                                  value=False)
                    w["save_state_every"] = ui.number("State every N steps (0 = with each LoRA save)",
                                                      value=d.save_state_every, precision=0).classes("w-72")
                    w["keep_states"] = ui.number("States to keep (rolling, 0 = all)", value=d.keep_states,
                                                 precision=0).classes("w-56")
                path_input(f, "resume_state", "Resume from training state folder (state_XXXXXX, optional)", "dir", "",
                           "w-[40rem]")

            with ui.expansion("Optimization", value=True).classes("w-full"):
                with ui.row().classes("w-full items-end"):
                    w["steps"] = ui.number("Steps", value=d.steps, precision=0).classes("w-28")
                    w["batch_size"] = ui.number("Batch size", value=d.batch_size, precision=0).classes("w-28")
                    w["gradient_accumulation"] = ui.number("Grad accumulation", value=d.gradient_accumulation,
                                                           precision=0).classes("w-36")
                    w["learning_rate"] = ui.number("Learning rate", value=d.learning_rate, format="%.2e",
                                                   step=1e-5).classes("w-36")
                    w["optimizer"] = ui.select(["adamw", "adamw8bit", "adafactor", "automagic3"], value="adamw8bit",
                                               label="Optimizer").classes("w-36")
                    w["lr_scheduler"] = ui.select(["constant", "cosine", "linear"], value=d.lr_scheduler,
                                                  label="LR schedule").classes("w-32")
                with ui.row().classes("w-full items-end").bind_visibility_from(
                        w["optimizer"], "value", lambda x: x == "automagic3"):
                    ui.label("Automagic v3 (self-tuning lr; LR schedule/warmup are ignored):").classes("text-xs")
                    w["automagic_min_lr"] = ui.number("Min lr", value=d.automagic_min_lr, format="%.1e").classes("w-32")
                    w["automagic_max_lr"] = ui.number("Max lr", value=d.automagic_max_lr, format="%.1e").classes("w-32")
                    w["automagic_polarity_history"] = ui.number("Polarity history", value=d.automagic_polarity_history,
                                                                min=2, max=64, precision=0).classes("w-36")
                    w["automagic_clip_threshold"] = ui.number("Clip threshold", value=d.automagic_clip_threshold,
                                                              step=0.1).classes("w-32")
                with ui.row().classes("w-full items-end"):
                    w["warmup_steps"] = ui.number("Warmup steps", value=d.warmup_steps, precision=0).classes("w-32")
                    w["max_grad_norm"] = ui.number("Max grad norm (0 = off)", value=d.max_grad_norm).classes("w-40")
                    w["timestep_sampling"] = ui.select(["logit_normal", "sigmoid", "uniform", "identity", "content", "style"],
                                                       value=d.timestep_sampling, label="Timestep sampling").classes("w-40") \
                        .tooltip("identity: logit-normal mean -0.5 (detail/low noise). content: sigma = u^p toward low "
                                 "noise (ai-toolkit style). style: mirror toward high noise.")
                    w["content_power"] = ui.number("Content power", value=d.content_power, step=0.5).classes("w-32") \
                        .bind_visibility_from(w["timestep_sampling"], "value", lambda x: x in ("content", "style"))
                    w["sigma_min"] = ui.number("Sigma min", value=d.sigma_min, min=0, max=1, step=0.05).classes("w-28")
                    w["sigma_max"] = ui.number("Sigma max", value=d.sigma_max, min=0, max=1, step=0.05).classes("w-28")
                    w["sigmoid_scale"] = ui.number("Sigmoid scale", value=d.sigmoid_scale).classes("w-32") \
                        .bind_visibility_from(w["timestep_sampling"], "value", lambda x: x == "sigmoid")
                    w["logit_mean"] = ui.number("Logit mean", value=d.logit_mean).classes("w-32") \
                        .bind_visibility_from(w["timestep_sampling"], "value", lambda x: x == "logit_normal")
                    w["logit_std"] = ui.number("Logit std", value=d.logit_std).classes("w-32") \
                        .bind_visibility_from(w["timestep_sampling"], "value", lambda x: x == "logit_normal")
                    w["timestep_shift"] = ui.number("Timestep shift", value=d.timestep_shift).classes("w-32")
                    w["loss_weighting"] = ui.select(["none", "min_snr", "sigma_sqrt"], value="none",
                                                    label="Loss weighting").classes("w-36")

            with ui.expansion("Saving & samples", value=True).classes("w-full"):
                with ui.row().classes("w-full items-end"):
                    w["save_every"] = ui.number("Save every N steps", value=d.save_every, precision=0).classes("w-40")
                    w["sample_every"] = ui.number("Sample every N steps (0 = off)", value=d.sample_every,
                                                  precision=0).classes("w-56")
                w["sample_prompts"] = ui.textarea("Sample prompts (one per line; empty = no samples)").classes("w-full")
                with ui.row().classes("w-full items-end"):
                    w["sample_seed"] = ui.number("Sample seed", value=d.sample_seed, precision=0).classes("w-32")
                    w["sample_walk_seed"] = ui.checkbox("Walk seed (+1 per prompt)", value=d.sample_walk_seed)
                    w["sample_before_training"] = ui.checkbox("Generate samples before training starts",
                                                              value=d.sample_before_training)
                with ui.row().classes("w-full items-end"):
                    w["sample_width"] = ui.number("Sample width", value=d.sample_width, precision=0).classes("w-32")
                    w["sample_height"] = ui.number("Sample height", value=d.sample_height, precision=0).classes("w-32")
                    w["sample_steps"] = ui.number("Sample steps", value=d.sample_steps, precision=0).classes("w-32")
                    w["sample_cfg"] = ui.number("Sample CFG", value=d.sample_cfg).classes("w-32")
                    w["sample_shift"] = ui.number("Sample shift", value=d.sample_shift).classes("w-32")
                w["sample_negative"] = ui.input("Sample negative prompt (optional)", value=d.sample_negative) \
                    .classes("w-full")
                with ui.row().classes("w-full items-end"):
                    path_input(f, "sample_model_path", "Sampling model (optional; .safetensors, diffusers folder or repo id)",
                               "file", "", "w-[36rem]")
                    w["sample_model_quant"] = ui.select(QUANTS, value="auto", label="Sampling model quant").classes("w-44")
                ui.label("Use a different checkpoint (e.g. SFT) for sample images only; the trained LoRA is copied onto it. "
                         "The training model is parked in system RAM while sampling.").classes("text-xs")
                with ui.row().classes("w-full items-end"):
                    path_input(f, "turbo_lora", "Turbo / step-distill LoRA for sampling (optional)", "file", "",
                               "w-[36rem]")
                    w["turbo_lora_strength"] = ui.number("Turbo LoRA strength", value=d.turbo_lora_strength,
                                                         step=0.05).classes("w-40")
                with ui.row().classes("w-full items-end"):
                    path_input(f, "sample_lora", "Secondary LoRA for sampling (optional, applied with the turbo LoRA)",
                               "file", "", "w-[36rem]")
                    w["sample_lora_strength"] = ui.number("Secondary LoRA strength", value=d.sample_lora_strength,
                                                          step=0.05).classes("w-40")

            with ui.expansion("Identity similarity: face + body (character LoRAs)", value=False).classes("w-full"):
                ui.label("Pick reference images on the Dataset tab (\"Face ref\" / \"Body ref\" toggles) or list paths "
                         "here. After each sampling pass, every sample's largest face is compared to the reference "
                         "faces with InsightFace (ArcFace; 50%+ generally means the same person), and its body "
                         "proportions (MediaPipe pose ratios) to the body references. Body references should show the "
                         "whole body in a fairly neutral, upright pose.").classes("text-xs")
                with ui.row().classes("w-full items-center gap-2") as progress_row:
                    progress_spinner = ui.spinner(size="sm")
                    progress_text = ui.label("").classes("text-sm")
                    progress_bar = ui.linear_progress(value=0, show_value=False).classes("w-64")
                progress_row.visible = False

                def _progress_tick():
                    if not _task_progress["active"]:
                        progress_row.visible = False
                        return
                    progress_row.visible = True
                    progress_text.text = _task_progress["text"]
                    frac = _task_progress["frac"]
                    progress_bar.visible = frac is not None
                    progress_bar.value = frac or 0

                ui.timer(0.5, _progress_tick)

                def _begin(text):
                    _task_progress.update(text=text, frac=None, active=True)

                def _end():
                    _task_progress.update(active=False)

                w["face_refs"] = ui.textarea("Face reference images or folders (one path per line)").classes("w-full")
                w["body_refs"] = ui.textarea("Body reference images or folders (one path per line; full-body shots)").classes("w-full")
                with ui.row().classes("w-full items-end"):
                    w["body_backend"] = ui.select(["mediapipe", "mediapipe+sam3"], value=d.body_backend,
                                                  label="Body backend").classes("w-44") \
                        .tooltip("mediapipe: skeleton proportion ratios. +sam3: also silhouette widths from a SAM 3 person mask")
                    w["sam3_model"] = ui.input("SAM 3 model (repo id or folder)", value=d.sam3_model).classes("w-64") \
                        .bind_visibility_from(w["body_backend"], "value", lambda x: x == "mediapipe+sam3")
                    w["body_device"] = ui.select(["cuda:0", "cuda:1", "cpu"], value=d.body_device,
                                                 label="SAM 3 device").classes("w-32") \
                        .bind_visibility_from(w["body_backend"], "value", lambda x: x == "mediapipe+sam3")
                with ui.row().classes("w-full items-end"):
                    w["face_weight"] = ui.number("Face weight", value=d.face_weight, step=0.1, min=0).classes("w-32")
                    w["body_weight"] = ui.number("Body weight", value=d.body_weight, step=0.1, min=0).classes("w-32")
                    body_info = ui.label("").classes("text-sm")

                    async def on_body_preview():
                        paths = [p.strip() for p in (w["body_refs"].value or "").splitlines() if p.strip()]
                        if not paths:
                            body_info.text = "No body reference images listed."
                            return
                        body_info.text = "Working..."
                        _begin("Preparing body models...")
                        try:
                            body_info.text = await run.io_bound(body_preview, paths, w["body_backend"].value,
                                                                w["sam3_model"].value, w["body_device"].value)
                        except Exception as e:  # noqa: BLE001
                            body_info.text = f"Error: {e}"
                        finally:
                            _end()

                    ui.button("Check body references", on_click=on_body_preview)
                with ui.row().classes("w-full items-end"):
                    w["face_provider"] = ui.select(["cpu", "cuda"], value=d.face_provider,
                                                   label="Face model device").classes("w-32")
                    w["face_det_size"] = ui.number("Detector size", value=d.face_det_size, precision=0).classes("w-32")
                    face_info = ui.label("").classes("text-sm")

                    async def on_face_preview():
                        paths = [p.strip() for p in (w["face_refs"].value or "").splitlines() if p.strip()]
                        if not paths:
                            face_info.text = "No reference images listed."
                            return
                        face_info.text = "Working..."
                        _begin("Preparing face model...")
                        try:
                            text, crops = await run.io_bound(face_preview, paths, w["face_provider"].value,
                                                             w["face_det_size"].value)
                        except Exception as e:  # noqa: BLE001
                            face_info.text = f"Error: {e}"
                            return
                        finally:
                            _end()
                        face_info.text = text
                        face_row.clear()
                        with face_row:
                            for crop, caption in crops:
                                with ui.card().tight().classes("w-32"):
                                    if crop:
                                        ui.image(crop).props("fit=contain").classes("w-32 h-32 bg-black")
                                    ui.label(caption).classes("text-xs px-1")

                    ui.button("Preview reference faces", on_click=on_face_preview)
                face_row = ui.row().classes("w-full gap-2")

            with ui.expansion("Region-weighted loss (character LoRAs)", value=False).classes("w-full"):
                ui.label("Weights each training image's loss by region: the detected face counts most, the person "
                         "silhouette next (SAM 3 or MediaPipe), the background least. Maps are built once at cache "
                         "time (16 px cells) and normalised to mean 1, so the loss scale is unchanged.").classes("text-xs")
                with ui.row().classes("w-full items-end"):
                    w["region_weights"] = ui.checkbox("Region-weighted loss", value=d.region_weights)
                    w["face_loss_weight"] = ui.number("Face weight", value=d.face_loss_weight, step=0.5).classes("w-32")
                    w["body_loss_weight"] = ui.number("Person weight", value=d.body_loss_weight, step=0.25).classes("w-32")
                    w["background_loss_weight"] = ui.number("Background weight", value=d.background_loss_weight,
                                                            step=0.1).classes("w-40")
                    w["face_margin"] = ui.number("Face box margin", value=d.face_margin, step=0.05).classes("w-36")
                    w["region_feather_cells"] = ui.number("Feather (cells)", value=d.region_feather_cells,
                                                          precision=0).classes("w-32")
                with ui.row().classes("w-full items-end"):
                    w["region_prior_weight"] = ui.number("Background prior weight (0 = off; +35% step time)",
                                                         value=d.region_prior_weight, step=0.1).classes("w-96") \
                        .tooltip("Regularises background cells toward the base model's prediction (extra no-grad forward)")
                    region_info = ui.label("").classes("text-sm")
                region_row = ui.row().classes("w-full gap-2")

                async def on_region_preview():
                    region_info.text = "Working..."
                    _task_progress.update(text="Preparing face detector...", frac=None, active=True)
                    try:
                        text, thumbs = await run.io_bound(analyse_and_preview, f.to_config().to_dict(), "regions")
                    except Exception as e:  # noqa: BLE001
                        region_info.text = f"Error: {e}"
                        return
                    finally:
                        _task_progress.update(active=False)
                    region_info.text = text
                    region_row.clear()
                    with region_row:
                        for pth, cap in thumbs:
                            with ui.card().tight().classes("w-32"):
                                ui.image(pth).props("fit=contain").classes("w-32 h-32 bg-black")
                                ui.tooltip(cap)

                ui.button("Preview region maps", on_click=on_region_preview)

            with ui.expansion("Per-image loss watch & per-image learning rate", value=False).classes("w-full"):
                ui.label("Tracks every image's loss normalised by its timestep bucket and, at each epoch boundary, "
                         "classes it as easy / healthy / suspect / stuck / exhausted. Per-image LR then throttles stuck "
                         "images (x0.5, x0.25, x0.125), eases exhausted ones (x0.6) and can exclude images that stay "
                         "stuck. A true per-image learning rate needs batch size 1 (otherwise per-sample loss weights). "
                         "Not available with automagic3.").classes("text-xs")
                with ui.row().classes("w-full items-end"):
                    w["loss_watch"] = ui.checkbox("Loss watch (observe)", value=d.loss_watch)
                    w["per_image_lr"] = ui.checkbox("Per-image learning rate (act on verdicts)", value=d.per_image_lr)
                    w["loss_watch_exclude_after"] = ui.number("Exclude after N epochs at lowest throttle (0 = never)",
                                                              value=d.loss_watch_exclude_after, precision=0).classes("w-96")
                with ui.row().classes("w-full items-end"):
                    w["loss_watch_buckets"] = ui.number("Sigma buckets", value=d.loss_watch_buckets, precision=0).classes("w-32")
                    w["loss_watch_beta"] = ui.number("Residual EMA", value=d.loss_watch_beta, min=0, max=0.99, step=0.05).classes("w-32")
                    w["loss_watch_high"] = ui.number("Hard above median x", value=d.loss_watch_high, step=0.05).classes("w-40")
                    w["loss_watch_low"] = ui.number("Easy below median x", value=d.loss_watch_low, step=0.05).classes("w-40")
                    w["loss_watch_improve_tol"] = ui.number("Improving = drop >", value=d.loss_watch_improve_tol, step=0.01).classes("w-36")
                    w["loss_watch_healthy_mult"] = ui.number("Healthy multiplier", value=d.loss_watch_healthy_mult, step=0.05).classes("w-36")

            with ui.expansion("Likeness probes & controller (character LoRAs)", value=False).classes("w-full"):
                ui.label("Renders a fixed-seed headshot and/or full-body probe every N steps (small, turbo LoRA), scores "
                         "it against the references, and lets a controller act on the smoothed scores: keep the best "
                         "checkpoint and stop early, nudge the learning rate, and weight the dataset / timesteps toward "
                         "whichever of face or body is lagging. Works with any optimizer. Needs face/body references "
                         "above.").classes("text-xs")
                with ui.row().classes("w-full items-end"):
                    w["probe_every"] = ui.number("Probe every N steps (0 = off)", value=d.probe_every, precision=0).classes("w-56")
                    w["probe_at_start"] = ui.checkbox("Probe at step 0 (baseline)", value=d.probe_at_start)
                    w["probe_seeds"] = ui.number("Seeds per probe", value=d.probe_seeds, min=1, max=8, precision=0).classes("w-32")
                    w["probe_seed"] = ui.number("Probe seed", value=d.probe_seed, precision=0).classes("w-32")
                w["probe_face_prompt"] = ui.input("Face probe prompt (headshot; empty = none)").classes("w-full")
                w["probe_body_prompt"] = ui.input("Body probe prompt (full body, upright; empty = none)").classes("w-full")
                with ui.row().classes("w-full items-end"):
                    w["probe_size"] = ui.number("Probe size (px)", value=d.probe_size, step=64, precision=0).classes("w-32")
                    w["probe_steps"] = ui.number("Probe steps", value=d.probe_steps, precision=0).classes("w-32")
                    w["probe_cfg"] = ui.number("Probe CFG", value=d.probe_cfg, step=0.5).classes("w-32")
                    w["probe_ema"] = ui.number("Score smoothing (EMA)", value=d.probe_ema, min=0, max=0.95, step=0.05).classes("w-44")
                    w["likeness_noise_band"] = ui.number("Noise band (% points)", value=d.likeness_noise_band, step=0.5).classes("w-44")
                ui.label("Controller actions").classes("text-sm font-bold")
                with ui.row().classes("w-full items-end"):
                    w["likeness_track_best"] = ui.checkbox("Keep best checkpoint (<job>_best.safetensors)", value=d.likeness_track_best)
                    w["likeness_patience"] = ui.number("Early stop after N probes w/o improvement (0 = never)",
                                                       value=d.likeness_patience, precision=0).classes("w-96")
                with ui.row().classes("w-full items-end"):
                    w["likeness_lr_control"] = ui.checkbox("LR multiplier from probe trend", value=d.likeness_lr_control)
                    w["likeness_lr_min"] = ui.number("Min factor", value=d.likeness_lr_min, step=0.1).classes("w-28")
                    w["likeness_lr_max"] = ui.number("Max factor", value=d.likeness_lr_max, step=0.1).classes("w-28")
                    w["likeness_lr_step"] = ui.number("Step (relative)", value=d.likeness_lr_step, step=0.05).classes("w-32")
                with ui.row().classes("w-full items-end"):
                    w["likeness_curriculum"] = ui.checkbox("Face/body curriculum (dataset weighting + timestep bias)",
                                                           value=d.likeness_curriculum)
                    w["likeness_curriculum_strength"] = ui.number("Strength", value=d.likeness_curriculum_strength,
                                                                  min=0, max=1, step=0.1).classes("w-28")

            status_msg = ui.label("").classes("text-sm")
            with ui.row().classes("items-center"):
                def on_start():
                    status_msg.text = start_training(f.to_config())
                    if status_msg.text.startswith("Started"):
                        tabs.value = tab_monitor

                ui.button("Start training", on_click=on_start, color="primary")
                ui.button("Stop (save)", on_click=lambda: setattr(status_msg, "text", stop_training()))
                ui.button("Kill", on_click=lambda: setattr(status_msg, "text", kill_training()), color="negative")

            with ui.row().classes("w-full items-end"):
                cfg_path = ui.input("Config file", value="configs/my_job.yaml").classes("w-96")

                def on_load():
                    try:
                        f.from_config(TrainConfig.load(cfg_path.value))
                        status_msg.text = f"Loaded {cfg_path.value}"
                    except Exception as e:  # noqa: BLE001
                        status_msg.text = f"Could not load config: {e}"

                def on_save():
                    cfg = f.to_config()
                    path = cfg_path.value.strip() or f"configs/{cfg.job_name}.yaml"
                    cfg.save(path)
                    status_msg.text = f"Saved {path}"

                async def browse_cfg():
                    picked = await run.io_bound(tk_dialog, "file", cfg_path.value, [("YAML", "*.yaml;*.yml")])
                    if picked:
                        cfg_path.value = _relative(picked)

                ui.button(icon="description", on_click=browse_cfg).props("flat dense").tooltip("Browse")
                ui.button("Load config", on_click=on_load)
                ui.button("Save config", on_click=on_save)

        # ------------------------------------------------------------ Dataset
        with ui.tab_panel(tab_dataset):
            build_dataset_tab(f)

        # ------------------------------------------------------------ Monitor
        with ui.tab_panel(tab_monitor):
            progress_label = ui.label("idle").classes("font-mono text-sm")
            progress_bar = ui.linear_progress(value=0, show_value=False).classes("w-full")
            gpu_box = ui.column().classes("w-full gap-1")
            gpu_widgets: list[dict] = []
            with ui.row().classes("w-full items-center gap-4"):
                ui.label("Loss smoothing").classes("text-sm")
                smoothing = ui.slider(min=0, max=0.98, step=0.02, value=0.8).classes("w-64")
                ui.label().bind_text_from(smoothing, "value", lambda x: f"{x:.2f}")
            loss_chart = ui.echart({
                "xAxis": {"type": "value", "name": "step", "min": "dataMin"},
                "yAxis": {"type": "value", "name": "loss", "scale": True},
                "tooltip": {"trigger": "axis"},
                "legend": {"data": ["raw", "smoothed"]},
                "grid": {"left": 60, "right": 20, "top": 30, "bottom": 40},
                "series": [
                    {"name": "raw", "type": "line", "showSymbol": False, "data": [],
                     "lineStyle": {"width": 1, "opacity": 0.35}},
                    {"name": "smoothed", "type": "line", "showSymbol": False, "data": [], "lineStyle": {"width": 2}},
                ],
            }).classes("w-full h-64")
            with ui.expansion("Identity similarity (face / body / combined)", value=True).classes("w-full") as face_exp:
                face_label = ui.label("").classes("text-sm")
                face_chart = ui.echart({
                    "xAxis": {"type": "value", "name": "step", "min": "dataMin"},
                    "yAxis": {"type": "value", "name": "similarity %", "min": 0, "max": 100},
                    "tooltip": {"trigger": "axis"},
                    "legend": {"data": []},
                    "grid": {"left": 60, "right": 20, "top": 30, "bottom": 40},
                    "series": [],
                }).classes("w-full h-56")
            face_exp.visible = False
            with ui.expansion("Per-image loss watch", value=True).classes("w-full") as lw_exp:
                lw_label = ui.label("").classes("text-sm")
                lw_table = ui.table(columns=[
                    {"name": "image", "label": "image", "field": "image", "align": "left", "sortable": True},
                    {"name": "verdict", "label": "verdict", "field": "verdict", "align": "left", "sortable": True},
                    {"name": "residual", "label": "residual (epoch)", "field": "residual", "sortable": True},
                    {"name": "ema", "label": "residual EMA", "field": "ema", "sortable": True},
                    {"name": "trend", "label": "trend %", "field": "trend", "sortable": True},
                    {"name": "multiplier", "label": "lr x", "field": "multiplier", "sortable": True},
                    {"name": "seen", "label": "seen", "field": "seen", "sortable": True},
                ], rows=[], row_key="path", pagination=15).classes("w-full text-xs")
            lw_exp.visible = False
            lw_state = {"sig": None}
            with ui.expansion("Likeness probes & controller", value=True).classes("w-full") as probe_exp:
                probe_label = ui.label("").classes("text-sm")
                probe_chart = ui.echart({
                    "xAxis": {"type": "value", "name": "step", "min": "dataMin"},
                    "yAxis": {"type": "value", "name": "likeness %", "min": 0, "max": 100},
                    "tooltip": {"trigger": "axis"},
                    "legend": {"data": ["face", "body", "combined", "face (raw)", "body (raw)"]},
                    "grid": {"left": 60, "right": 20, "top": 30, "bottom": 40},
                    "series": [
                        {"name": "face", "type": "line", "data": []},
                        {"name": "body", "type": "line", "data": []},
                        {"name": "combined", "type": "line", "data": [], "lineStyle": {"width": 3}},
                        {"name": "face (raw)", "type": "scatter", "symbolSize": 5, "data": []},
                        {"name": "body (raw)", "type": "scatter", "symbolSize": 5, "data": []},
                    ],
                }).classes("w-full h-56")
                probe_row = ui.row().classes("w-full gap-2")
            probe_exp.visible = False
            shown_probes: list[str] = []
            with ui.expansion("Latest samples", value=True).classes("w-full"):
                samples_row = ui.row().classes("w-full gap-2")
            saved_label = ui.label("").classes("font-mono text-xs whitespace-pre-wrap")
            ui.label("Log").classes("text-sm font-bold")
            log_view = ui.log(max_lines=600).classes("w-full h-72 font-mono text-xs")
            shown_samples: list[str] = []
            log_state = {"job": None, "n": 0, "console_shown": False}
            show_full = image_viewer()

            def update_gpus():
                stats = gpu_stats()
                if len(gpu_widgets) != len(stats):
                    gpu_box.clear()
                    gpu_widgets.clear()
                    with gpu_box:
                        for st in stats:
                            with ui.row().classes("w-full items-center gap-2"):
                                lbl = ui.label("").classes("font-mono text-xs w-[30rem]")
                                vram = ui.linear_progress(value=0, show_value=False).classes("w-48")
                                util = ui.linear_progress(value=0, show_value=False, color="orange").classes("w-48")
                            gpu_widgets.append({"lbl": lbl, "vram": vram, "util": util})
                for st, wdg in zip(stats, gpu_widgets):
                    wdg["lbl"].text = (f"GPU{st['index']} {st['name']}: VRAM {st['used'] / 1024:.1f}/{st['total'] / 1024:.1f} GB"
                                       f" | util {st['util']}%")
                    wdg["vram"].value = st["used"] / max(1, st["total"])
                    wdg["util"].value = st["util"] / 100

            def update_log():
                if _job_dir is None:
                    return
                if log_state["job"] != str(_job_dir):
                    log_state.update(job=str(_job_dir), n=0, console_shown=False)
                    log_view.clear()
                lp = _job_dir / "train.log"
                if lp.exists():
                    lines = lp.read_text(encoding="utf-8", errors="replace").splitlines()
                    if len(lines) < log_state["n"]:
                        log_view.clear()
                        log_state["n"] = 0
                    for line in lines[log_state["n"]:]:
                        log_view.push(line)
                    log_state["n"] = len(lines)
                st = read_status() or {}
                if not _running() and not log_state["console_shown"] and (st.get("error") or not st.get("done")):
                    cp = _job_dir / "console.log"
                    if cp.exists():
                        tail = cp.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
                        if tail:
                            log_view.push("--- console.log ---")
                            for line in tail:
                                log_view.push(line)
                    log_state["console_shown"] = True

            def poll():
                update_gpus()
                st = read_status()
                running = _running()
                if st is None:
                    progress_label.text = "running: starting..." if running else "idle"
                    return
                step, total = st.get("step", 0), st.get("total", 0) or 0
                eta = st.get("eta")
                eta_s = f"{eta / 60:.1f} min" if eta else "-"
                loss = st.get("loss")
                state = "RUNNING" if running else "STOPPED"
                if loss is not None:
                    lr_s = f" | lr {st['lr']:.2e}" if st.get("lr") is not None else ""
                    progress_label.text = (f"{state} | stage: {st.get('stage')} | step {step}/{total} | loss {loss:.4f}"
                                           f" | avg {st.get('avg_loss') or 0:.4f}{lr_s} | eta {eta_s}")
                else:
                    progress_label.text = f"{state} | stage: {st.get('stage')} | {step}/{total}"
                if st.get("error"):
                    progress_label.text += f"  ERROR: {st['error']}"
                progress_bar.value = (step / total) if total else 0
                losses = st.get("losses") or []
                if losses:
                    start = max(0, step - len(losses))
                    raw = [[start + i + 1, v] for i, v in enumerate(losses)]
                    a = float(smoothing.value)
                    sm, ema = [], None
                    for x, v in raw:
                        ema = v if ema is None else a * ema + (1 - a) * v
                        sm.append([x, ema])
                    loss_chart.options["series"][0]["data"] = raw
                    loss_chart.options["series"][1]["data"] = sm
                    loss_chart.update()
                fscores = st.get("face_scores") or []
                if fscores:
                    face_exp.visible = True
                    by_prompt: dict[int, list] = {}
                    per_step: dict[int, dict[str, list]] = {}
                    for sc in fscores:
                        if sc.get("combined_percent") is not None:
                            by_prompt.setdefault(sc["prompt_index"], []).append([sc["step"], round(sc["combined_percent"], 1)])
                        d_ = per_step.setdefault(sc["step"], {"face": [], "body": []})
                        if sc.get("percent") is not None:
                            d_["face"].append(sc["percent"])
                        if sc.get("body_percent") is not None:
                            d_["body"].append(sc["body_percent"])
                    series = [{"name": f"combined p{k + 1}", "type": "line", "data": v}
                              for k, v in sorted(by_prompt.items())]
                    for key, name in (("face", "mean face"), ("body", "mean body")):
                        pts = [[stp, round(sum(v[key]) / len(v[key]), 1)] for stp, v in sorted(per_step.items()) if v[key]]
                        if pts:
                            series.append({"name": name, "type": "line", "data": pts, "lineStyle": {"type": "dotted"}})
                    ref_sim = st.get("face_ref_similarity")
                    if ref_sim is not None and series:
                        xs = [pt[0] for v in by_prompt.values() for pt in v] or [0, 1]
                        series.append({"name": "face reference self-similarity", "type": "line", "showSymbol": False,
                                       "lineStyle": {"type": "dashed"},
                                       "data": [[min(xs), ref_sim * 100], [max(xs), ref_sim * 100]]})
                    face_chart.options["legend"]["data"] = [sr["name"] for sr in series]
                    face_chart.options["series"] = series
                    face_chart.update()
                    latest_step = max(sc["step"] for sc in fscores)
                    latest = [sc for sc in fscores if sc["step"] == latest_step]
                    fv = [sc["percent"] for sc in latest if sc.get("percent") is not None]
                    bv = [sc["body_percent"] for sc in latest if sc.get("body_percent") is not None]
                    cv = [sc["combined_percent"] for sc in latest if sc.get("combined_percent") is not None]
                    parts = [f"Step {latest_step}:"]
                    parts.append(f"face {sum(fv) / len(fv):.1f}% ({len(fv)}/{len(latest)} with a face)" if fv
                                 else f"face: none in {len(latest)} sample(s)")
                    if st.get("body_refs"):
                        parts.append(f"body {sum(bv) / len(bv):.1f}% ({len(bv)}/{len(latest)} with a pose)" if bv
                                     else "body: no usable pose")
                    if cv:
                        parts.append(f"combined {sum(cv) / len(cv):.1f}%")
                    if ref_sim is not None:
                        parts.append(f"face ref self-sim {ref_sim * 100:.1f}%")
                    if st.get("body_ref_similarity") is not None:
                        parts.append(f"body ref self-sim {st['body_ref_similarity'] * 100:.1f}%")
                    face_label.text = " | ".join(parts)
                lw = st.get("loss_watch")
                if lw:
                    lw_exp.visible = True
                    sig = (lw.get("epoch"), lw.get("step"))
                    if sig != lw_state["sig"]:
                        lw_state["sig"] = sig
                        counts = ", ".join(f"{k} {v}" for k, v in (lw.get("counts") or {}).items())
                        med = lw.get("median")
                        lw_label.text = (f"Epoch {lw.get('epoch')} (step {lw.get('step')}): {counts} | median residual "
                                         f"{med:.3f} | {lw.get('active')}/{lw.get('total')} images active"
                                         if med is not None else f"Epoch {lw.get('epoch')}: waiting for the first epoch")
                        lw_table.rows = lw.get("rows") or []
                        lw_table.update()
                probes = st.get("probes") or []
                if probes:
                    probe_exp.visible = True
                    def pts(key):
                        return [[pr["step"], round(pr[key], 1)] for pr in probes if pr.get(key) is not None]
                    probe_chart.options["series"][0]["data"] = pts("face_ema")
                    probe_chart.options["series"][1]["data"] = pts("body_ema")
                    probe_chart.options["series"][2]["data"] = pts("combined_ema")
                    probe_chart.options["series"][3]["data"] = pts("face")
                    probe_chart.options["series"][4]["data"] = pts("body")
                    probe_chart.update()
                    lk = st.get("likeness") or {}
                    last = probes[-1]
                    f2 = lambda x: "n/a" if x is None else f"{x:.1f}%"  # noqa: E731
                    probe_label.text = (f"Probe @ step {last['step']}: face {f2(last.get('face_ema'))} body {f2(last.get('body_ema'))}"
                                        f" combined {f2(last.get('combined_ema'))} | best {f2(lk.get('best'))} @ step {lk.get('best_step')}"
                                        f" | {lk.get('since_improve', 0)} probe(s) since improvement | lr x{lk.get('lr_factor', 1.0):.2f}"
                                        f" | face bias {lk.get('face_bias', 0.0):+.2f}" + (" | STOPPED" if lk.get("stopped") else ""))
                    latest_imgs = list(last.get("images") or [])[:6]
                    if latest_imgs != shown_probes:
                        shown_probes[:] = latest_imgs
                        probe_row.clear()
                        with probe_row:
                            for p in latest_imgs:
                                with ui.card().tight():
                                    ui.image(p).props("fit=contain").classes("w-40 h-40 bg-black cursor-pointer") \
                                        .on("click", lambda _, q=p: show_full(Path(q)))
                                    ui.label(Path(p).name).classes("text-xs px-1")
                score_by_name = {Path(sc["image"]).name: sc for sc in fscores}
                samples = (st.get("samples") or [])[:12]
                sample_sig = [(p, (score_by_name.get(Path(p).name) or {}).get("combined_percent")) for p in samples]
                if sample_sig != shown_samples:
                    shown_samples[:] = sample_sig
                    samples_row.clear()
                    with samples_row:
                        for p, pct in sample_sig:
                            with ui.card().tight():
                                ui.image(p).props("fit=contain").classes("w-48 h-48 bg-black cursor-pointer") \
                                    .on("click", lambda _, q=p: show_full(Path(q)))
                                caption = Path(p).name
                                sc_ = score_by_name.get(Path(p).name)
                                if sc_ is not None:
                                    caption += score_caption(sc_)
                                ui.label(caption).classes("text-xs px-1")
                saved = list(st.get("saved") or [])
                states = list(st.get("states") or [])
                saved_label.text = "Saved:\n" + "\n".join(saved) + ("\nStates:\n" + "\n".join(states) if states else "")
                update_log()

            ui.timer(2.0, poll)

        # ------------------------------------------------------------ Samples
        with ui.tab_panel(tab_samples):
            build_samples_tab(f)

        # ------------------------------------------------------------ Help
        with ui.tab_panel(tab_help):
            ui.markdown(HELP)


HELP = """
## Quick start
1. **Dataset**: a folder of images; each image may have a same-name `.txt` caption. Optional trigger word is prepended.
2. **Model**: either a Hugging Face repo (Base / SFT, about 30 GB, stored under `models/`) or three single
   ComfyUI-style files: diffusion model, Qwen3.5-9B text encoder and Flux2 VAE.
3. Click **Start training**. First run caches VAE latents and text embeddings; the text encoder is loaded once
   for caching and unloaded before the diffusion model is loaded.
4. LoRAs are written to `output/<job>/<job>_<step>.safetensors` and `output/<job>/<job>.safetensors` (bf16 by default).

## Quantization
*Diffusion model quant* and *Text encoder quant*: `auto` keeps whatever the file already is (bf16 stays bf16, a
ComfyUI int8/nvfp4 file stays quantized), `none` dequantizes to bf16, and `int8`, `int8_convrot` or `nvfp4`
quantize on load. Only the frozen base weights are quantized; the LoRA itself always trains in fp32.
int8 roughly halves VRAM for the base model (about 5 GB for the DiT); nvfp4 quarters it.

## Training states
Tick *Save training state* to also write `state_XXXXXX/` folders (LoRA in fp32 + optimizer + scheduler + RNG).
Point *Resume from training state folder* at one of them to continue exactly where it stopped.

## Sampling model
Optionally sample with a different checkpoint than you train on (for example train on Base, preview on SFT).
The trained LoRA is copied onto it before each sampling pass; the training model waits in system RAM meanwhile.

## Turbo LoRA
An optional ComfyUI-format LoRA (for example an 8-step distill) applied only while generating sample images,
never during the training step. Set *Sample steps* to match it.

## Identity training (character LoRAs)
The **Identity training preset** switch (top of the Train tab) applies the measured recipe below in one go and
restores your previous settings when switched off; the CLI equivalent is `--preset identity`.
Measured on a small test set (120 steps, same seeds): region-weighted loss gave the best face likeness (+2 points
over baseline), LoRA+ at ratio 4 a small gain (ratio 16 on a 2e-4 lr was unstable), `identity` timesteps were
neutral, and auto crops *hurt* when the crops had to be upscaled. Suggested recipe: *Region-weighted loss* on
(face 3 / person 1.5 / background 0.3), *LoRA+ ratio* 4 with `adamw8bit`, *Auto crops* only when your photos are
high-resolution (crops are never upscaled by default), timesteps left at `logit_normal`.
LoRA rules of thumb: LoRA lr is ~10x a full fine-tune lr and nearly rank-independent; keep MLP layers in the
targets; small batches with more steps beat large batches.

## Using in ComfyUI
ComfyUI supports MOSS-Image natively. Copy the `.safetensors` to `ComfyUI/models/loras/` and use the standard
**Load LoRA** node after *Load Diffusion Model*.
"""


def _free_port(host: str, start: int, tries: int = 10) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                print(f"Port {port} is busy, trying {port + 1}...")
    raise SystemExit(f"No free port in range {start}-{start + tries - 1}")


if __name__ in {"__main__", "__mp_main__"}:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7870)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="open the UI in the default browser once ready")
    a = ap.parse_args()
    port = _free_port(a.host, a.port)
    build_page()
    ui.run(host=a.host, port=port, title="RulzTrainer", show=a.open, reload=False, dark=None)
