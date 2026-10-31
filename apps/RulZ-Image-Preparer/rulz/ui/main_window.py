"""Main window: image grid with the global adjustment sliders, plus the per-image editor page."""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QRectF, QSize, QThreadPool, QTimer, Qt
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QPainterPath, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QListView, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QMessageBox, QProgressDialog, QPushButton, QStackedWidget, QStyle,
    QStyledItemDelegate, QVBoxLayout, QWidget,
)

from .. import ai_jpeg, ai_mask, config, imageio
from ..filters import ADJUSTMENTS, BALANCED, BEST, process
from . import theme
from .ai_setup import ensure_crop_ready, ensure_ready
from .auto_crop import AutoCropDialog
from .controls import AdjustRow, ModeSwitch
from .editor_page import EditorPage
from .workers import ImageItem, SaveAll, ThumbLoad, make_emitter, output_paths

ITEM_ROLE = Qt.UserRole + 1
CARD_W, CARD_H, PAD = 220, 250, 10
JPEG_MODE_KEY = "jpeg_mode"
OPEN_DIR_KEY = "last_open_dir"
SAVE_DIR_KEY = "last_save_dir"
FILE_FILTER = "Images (" + " ".join(f"*{e}" for e in sorted(imageio.IMAGE_EXTS)) + ")"
SHORT = {"color": "Noise", "jpeg": "JPEG", "detail": "Detail"}


def _thumb_key(it: ImageItem, globals_: dict) -> tuple:
    # Thumbnails are too small to show JPEG blocks or fine detail, so only color noise affects them.
    return ("color", it.value("color", globals_))


class CardDelegate(QStyledItemDelegate):
    def __init__(self, window: "MainWindow"):
        super().__init__(window)
        self.win = window

    def sizeHint(self, *_):
        return QSize(CARD_W, CARD_H)

    def paint(self, p: QPainter, opt, index):
        it: ImageItem = index.data(ITEM_ROLE)
        r = QRectF(opt.rect).adjusted(5, 5, -5, -5)
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        selected = bool(opt.state & QStyle.State_Selected)
        hover = bool(opt.state & QStyle.State_MouseOver)
        card = QPainterPath()
        card.addRoundedRect(r, 12, 12)
        p.fillPath(card, QColor(theme.C["hover"] if hover else theme.C["card"]))
        p.setPen(QColor(theme.C["accent"] if selected else theme.C["border"]))
        p.drawPath(card)

        img_r = QRectF(r.left() + PAD, r.top() + PAD, r.width() - 2 * PAD, r.height() - 2 * PAD - 44)
        pm = it.preview
        if pm is not None:
            s = min(img_r.width() / pm.width(), img_r.height() / pm.height())
            w, h = pm.width() * s, pm.height() * s
            target = QRectF(img_r.center().x() - w / 2, img_r.center().y() - h / 2, w, h)
            clip = QPainterPath()
            clip.addRoundedRect(target, 6, 6)
            p.setClipPath(clip)
            p.drawPixmap(target, pm, QRectF(pm.rect()))
            p.setClipping(False)
        else:
            p.setPen(QColor(theme.C["muted"]))
            p.drawText(img_r, Qt.AlignCenter, it.error and "Can't read this file" or "Loading…")

        f = QFont(opt.font)
        f.setPixelSize(12)
        p.setFont(f)
        p.setPen(QColor(theme.C["text"]))
        name_r = QRectF(r.left() + PAD, img_r.bottom() + 6, r.width() - 2 * PAD, 18)
        p.drawText(name_r, Qt.AlignLeft | Qt.AlignVCenter,
                   p.fontMetrics().elidedText(it.path.name, Qt.ElideMiddle, int(name_r.width())))

        settings = it.settings(self.win.globals)
        custom = bool(it.custom)
        badge = "  ".join(f"{SHORT[k]} {v}" for k, v in settings.items() if v) or "No changes"
        if custom:
            badge = "Custom · " + badge
        f.setBold(True)
        f.setPixelSize(11)
        p.setFont(f)
        badge = p.fontMetrics().elidedText(badge, Qt.ElideRight, int(r.width() - 2 * PAD - 16))
        bw = p.fontMetrics().horizontalAdvance(badge) + 16
        b_r = QRectF(r.left() + PAD, name_r.bottom() + 3, bw, 18)
        badge_path = QPainterPath()
        badge_path.addRoundedRect(b_r, 9, 9)
        p.fillPath(badge_path, QColor(theme.C["accent"] if custom else theme.C["placeholder"]))
        p.setPen(QColor(theme.C["on_accent"] if custom else theme.C["muted"]))
        p.drawText(b_r, Qt.AlignCenter, badge)
        p.restore()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.APP_NAME)
        self.resize(1320, 900)
        self.setAcceptDrops(True)
        self.items: list[ImageItem] = []
        s = config.settings()
        self.globals = {k: int(s.value(f"global/{k}", d)) for k, (_, d) in ADJUSTMENTS.items()}
        self.mode = BEST if s.value(JPEG_MODE_KEY, BALANCED) == BEST and ai_jpeg.status() == ai_jpeg.READY \
            else BALANCED
        if self.mode == BEST:
            QThreadPool.globalInstance().start(ai_jpeg.warm_up)
        self._render_queue: list[ImageItem] = []
        self._render_timer = QTimer(self, interval=0, timeout=self._render_some)
        self._pool = QThreadPool(self)
        self._emitter = make_emitter(self)
        self._emitter.done.connect(self._on_thumb)
        self._saver: SaveAll | None = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.stack.addWidget(self._build_library())
        self.editor = EditorPage()
        self.editor.back.connect(self._show_library)
        self.editor.navigate.connect(self._navigate)
        self.editor.itemChanged.connect(self._on_item_changed)
        self.editor.modeRequested.connect(self._request_mode)
        self.stack.addWidget(self.editor)
        self._update_counts()

    # ---- layout ----------------------------------------------------------
    def _build_library(self) -> QWidget:
        page = QWidget(objectName="Page")

        top = QFrame(objectName="TopBar")
        tl = QHBoxLayout(top)
        tl.setContentsMargins(16, 10, 16, 10)
        tl.setSpacing(10)
        logo = QLabel()
        pm = QPixmap(theme.logo_file())
        if not pm.isNull():
            logo.setPixmap(pm.scaledToHeight(30, Qt.SmoothTransformation))
        tl.addWidget(logo)
        tl.addWidget(QLabel("Image Preparer", objectName="PageTitle"))
        tl.addStretch(1)
        add = QPushButton(theme.icon("add"), "  Add images", clicked=self._browse)
        add.setToolTip("Pick photos to add (you can also drag photos or folders onto the window)")
        self.clear_btn = QPushButton(theme.icon("clear"), "  Clear", clicked=self._clear)
        self.save_btn = QPushButton(theme.icon("save", "#ffffff"), "  Save all…", clicked=self._save_all)
        self.save_btn.setProperty("primary", True)
        self.crop_btn = QPushButton(theme.icon("crop"), "  Auto crop…", clicked=self._auto_crop)
        self.crop_btn.setToolTip("Find something you describe (e.g. a face) in the selected pictures with AI, "
                                 "and crop them to it.\nCtrl+click or Shift+click to select; with none "
                                 "selected, every picture is used.")
        tl.addWidget(add)
        tl.addWidget(self.clear_btn)
        tl.addWidget(self.crop_btn)
        tl.addWidget(self.save_btn)

        tools = QFrame(objectName="ToolBar")
        grid = QGridLayout(tools)
        grid.setContentsMargins(16, 10, 16, 10)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)
        self.global_rows: dict[str, AdjustRow] = {}
        for r, key in enumerate(ADJUSTMENTS):
            row = AdjustRow(key)
            row.label.setText(f"{ADJUSTMENTS[key][0]}")
            row.set_value(self.globals[key])
            row.slider.setToolTip(row.slider.toolTip() + "\n\nApplies to every image that doesn't have its own value.")
            row.changed.connect(self._on_global)
            self.global_rows[key] = row
            grid.addWidget(row, r, 0)
        self.count_label = QLabel(objectName="Muted")
        self.mode_switch = ModeSwitch()
        self.mode_switch.set_mode(self.mode)
        self.mode_switch.requested.connect(self._request_mode)
        self.reset_all = QPushButton(theme.icon("reset"), "  Reset all to global", clicked=self._reset_all)
        self.reset_all.setToolTip("Clear every image's own settings so all follow the global sliders")
        grid.addWidget(self.count_label, 0, 1)
        grid.addWidget(self.mode_switch, list(ADJUSTMENTS).index("jpeg"), 1)
        grid.addWidget(self.reset_all, len(ADJUSTMENTS) - 1, 1)
        grid.setColumnStretch(0, 1)

        self.grid = QListWidget()
        self.grid.setViewMode(QListView.IconMode)
        self.grid.setResizeMode(QListView.Adjust)
        self.grid.setMovement(QListView.Static)
        self.grid.setUniformItemSizes(True)
        self.grid.setGridSize(QSize(CARD_W, CARD_H))
        self.grid.setSpacing(0)
        self.grid.setSelectionMode(QListWidget.ExtendedSelection)
        self.grid.setMouseTracking(True)
        self.grid.setItemDelegate(CardDelegate(self))
        self.grid.setAcceptDrops(False)
        self.grid.viewport().setAcceptDrops(False)
        self.grid.setVerticalScrollMode(QListView.ScrollPerPixel)
        self.grid.itemActivated.connect(lambda li: self._open(self.grid.row(li)))
        self.grid.itemClicked.connect(lambda li: self._open(self.grid.row(li)) if not self._modifiers() else None)
        self.grid.setContextMenuPolicy(Qt.CustomContextMenu)
        self.grid.customContextMenuRequested.connect(self._grid_menu)
        QShortcut(QKeySequence.Delete, self.grid, self._remove_selected, context=Qt.WidgetShortcut)

        self.drop_hint = QLabel(
            "Drag photos or folders here\n\nor click  Add images", objectName="DropHint", alignment=Qt.AlignCenter
        )
        self.body = QStackedWidget()
        hint_wrap = QWidget()
        hw = QVBoxLayout(hint_wrap)
        hw.setContentsMargins(40, 40, 40, 40)
        hw.addWidget(self.drop_hint)
        self.body.addWidget(hint_wrap)
        self.body.addWidget(self.grid)

        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(top)
        lay.addWidget(tools)
        lay.addWidget(self.body, 1)
        return page

    @staticmethod
    def _modifiers() -> bool:
        from PySide6.QtWidgets import QApplication

        return bool(QApplication.keyboardModifiers() & (Qt.ControlModifier | Qt.ShiftModifier))

    # ---- adding images ---------------------------------------------------
    def _browse(self):
        start = str(config.settings().value(OPEN_DIR_KEY, str(Path.home() / "Pictures")))
        files, _ = QFileDialog.getOpenFileNames(self, "Add images", start, FILE_FILTER)
        if files:
            config.settings().setValue(OPEN_DIR_KEY, str(Path(files[0]).parent))
            self.add_paths(files)

    def add_paths(self, paths):
        known = {str(it.path.resolve()).lower() for it in self.items}
        added = 0
        for path in self._expand(paths):
            key = str(path.resolve()).lower()
            if key in known:
                continue
            known.add(key)
            it = ImageItem(path)
            self.items.append(it)
            li = QListWidgetItem()
            li.setData(ITEM_ROLE, it)
            li.setToolTip(str(path))
            self.grid.addItem(li)
            self._pool.start(ThumbLoad(it, self._emitter))
            added += 1
        if added:
            self._update_counts()
        return added

    @staticmethod
    def _expand(paths):
        for raw in paths:
            p = Path(raw)
            if p.is_dir():
                yield from sorted(f for f in p.rglob("*") if f.is_file() and imageio.is_image(f))
            elif p.is_file() and imageio.is_image(p):
                yield p

    def _on_thumb(self, it: ImageItem, result):
        if isinstance(result, Exception):
            it.error = str(result)
        else:
            it.thumb = result
            self._queue_render([it])
        self._refresh_card(it)

    # ---- drag & drop -----------------------------------------------------
    def _drop_paths(self, e) -> list[str]:
        if not e.mimeData().hasUrls():
            return []
        out = []
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p and (Path(p).is_dir() or imageio.is_image(p)):
                out.append(p)
        return out

    def _set_drop_active(self, on: bool):
        self.drop_hint.setProperty("active", on)
        self.drop_hint.style().unpolish(self.drop_hint)
        self.drop_hint.style().polish(self.drop_hint)

    def dragEnterEvent(self, e):
        if self._drop_paths(e):
            e.acceptProposedAction()
            self._set_drop_active(True)

    def dragMoveEvent(self, e):
        e.acceptProposedAction()

    def dragLeaveEvent(self, e):
        self._set_drop_active(False)

    def dropEvent(self, e):
        self._set_drop_active(False)
        paths = self._drop_paths(e)
        if paths:
            e.acceptProposedAction()
            if not self.add_paths(paths):
                self.statusBar().showMessage("No new images in what you dropped", 4000)

    # ---- previews --------------------------------------------------------
    def _queue_render(self, items):
        queued = set(map(id, self._render_queue))
        self._render_queue.extend(it for it in items if it.thumb is not None and id(it) not in queued)
        if self._render_queue:
            self._render_timer.start()

    def _render_some(self):
        """Re-render thumbnails a few at a time so the UI stays smooth while sliding."""
        deadline = time.perf_counter() + 0.025
        while self._render_queue and time.perf_counter() < deadline:
            it = self._render_queue.pop(0)
            key = _thumb_key(it, self.globals)
            if it.preview_key != key:
                img = process(it.thumb, it.settings(self.globals), full_size=False)
                it.preview = QPixmap.fromImage(imageio.to_qimage(img))
                it.preview_key = key
                self._refresh_card(it)
        if not self._render_queue:
            self._render_timer.stop()

    def _refresh_card(self, it: ImageItem):
        for row in range(self.grid.count()):
            li = self.grid.item(row)
            if li.data(ITEM_ROLE) is it:
                self.grid.update(self.grid.indexFromItem(li))
                break

    # ---- settings --------------------------------------------------------
    def _on_global(self, key: str, v: int):
        self.globals[key] = v
        config.settings().setValue(f"global/{key}", v)
        self._queue_render([it for it in self.items if key not in it.custom])
        self.grid.viewport().update()
        self.editor.set_globals(self.globals, self.mode)

    def _request_mode(self, mode: str):
        if mode == BEST:
            if not ensure_ready(self):
                mode = BALANCED
            elif self.mode != BEST:
                QThreadPool.globalInstance().start(ai_jpeg.warm_up)  # load the model before it's needed
        self.mode = mode
        config.settings().setValue(JPEG_MODE_KEY, mode)
        self.mode_switch.set_mode(mode)
        self.editor.set_globals(self.globals, mode)

    def _on_item_changed(self, it: ImageItem):
        self._queue_render([it])
        self._refresh_card(it)
        self._update_counts()

    def _reset_all(self):
        for it in self.items:
            it.custom.clear()
        self._queue_render(self.items)
        self.grid.viewport().update()
        self._update_counts()

    def _update_counts(self):
        n = len(self.items)
        custom = sum(1 for it in self.items if it.custom)
        self.body.setCurrentIndex(1 if n else 0)
        self.count_label.setText(f"{n} image{'s' * (n != 1)}" + (f"  ·  {custom} custom" if custom else ""))
        self.save_btn.setEnabled(n > 0)
        self.crop_btn.setEnabled(n > 0)
        self.clear_btn.setEnabled(n > 0)
        self.reset_all.setEnabled(custom > 0)

    # ---- removing --------------------------------------------------------
    def _grid_menu(self, pos):
        li = self.grid.itemAt(pos)
        if li is None:
            return
        if not li.isSelected():
            self.grid.setCurrentItem(li)
        m = QMenu(self)
        m.addAction("Open", lambda: self._open(self.grid.row(li)))
        rows = self.grid.selectedItems()
        m.addAction(theme.icon("crop"), f"Auto crop ({len(rows)})…", self._auto_crop)
        if any(r.data(ITEM_ROLE).custom for r in rows):
            m.addAction("Use global settings", lambda: self._reset_rows(rows))
        m.addSeparator()
        m.addAction(theme.icon("delete"), f"Remove from list ({len(rows)})", self._remove_selected)
        m.exec(self.grid.viewport().mapToGlobal(pos))

    def _reset_rows(self, rows):
        for li in rows:
            li.data(ITEM_ROLE).custom.clear()
        self._queue_render([li.data(ITEM_ROLE) for li in rows])
        self.grid.viewport().update()
        self._update_counts()

    def _remove_selected(self):
        rows = sorted((self.grid.row(li) for li in self.grid.selectedItems()), reverse=True)
        for row in rows:
            it = self.grid.takeItem(row).data(ITEM_ROLE)
            self.items.remove(it)
        self._render_queue = [it for it in self._render_queue if it in self.items]
        self._update_counts()

    def _clear(self):
        if not self.items:
            return
        if QMessageBox.question(self, "Clear list", "Remove all images from the list?\n(Files on disk are not touched.)") \
                != QMessageBox.Yes:
            return
        self.grid.clear()
        self.items.clear()
        self._render_queue.clear()
        self._update_counts()

    # ---- editor ----------------------------------------------------------
    def _open(self, row: int):
        if not 0 <= row < len(self.items):
            return
        it = self.grid.item(row).data(ITEM_ROLE)
        if self.stack.currentWidget() is self.editor and self.editor.item is it:
            return  # a double-click lands here twice
        self.grid.setCurrentRow(row)
        self.editor.open(it, row, len(self.items), self.globals, self.mode)
        self.stack.setCurrentWidget(self.editor)

    def _navigate(self, delta: int):
        row = self.grid.currentRow() + delta
        if 0 <= row < self.grid.count():
            self._open(row)

    def _show_library(self):
        self.editor.close_image()
        self.stack.setCurrentIndex(0)
        self.grid.setFocus()

    # ---- auto crop -------------------------------------------------------
    def _auto_crop(self):
        rows = sorted(self.grid.row(li) for li in self.grid.selectedItems())
        items = [self.items[r] for r in rows] or list(self.items)
        if not items or not ensure_crop_ready(self):
            return
        QThreadPool.globalInstance().start(ai_mask.warm_up)  # load the model while the screen opens
        AutoCropDialog(items, self.globals, self.mode, self).exec()

    # ---- saving ----------------------------------------------------------
    def _save_all(self):
        if not self.items:
            return
        uses_ai = self.mode == BEST and any(it.value("jpeg", self.globals) for it in self.items)
        if uses_ai and not ensure_ready(self):
            return
        start = str(config.settings().value(SAVE_DIR_KEY, "") or self.items[0].path.parent)
        out = QFileDialog.getExistingDirectory(self, "Save all processed images to…", start)
        if not out:
            return
        config.settings().setValue(SAVE_DIR_KEY, out)
        out_dir = Path(out)
        dests = output_paths(self.items, out_dir)
        existing = [d for d in dests if d.exists()]
        if existing and QMessageBox.question(
            self, "Replace files?",
            f"{len(existing)} file(s) with the same name already exist in that folder. Replace them?",
        ) != QMessageBox.Yes:
            return
        jobs = [(it.path, d, it.settings(self.globals)) for it, d in zip(self.items, dests)]

        dlg = QProgressDialog("Saving…", "Cancel", 0, len(jobs), self)
        dlg.setWindowTitle("Save all")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setMinimumWidth(420)
        saver = SaveAll(jobs, self.mode)
        saver.progress.connect(lambda i, name: (dlg.setValue(i), dlg.setLabelText(f"Saving {name}…") if name else None))
        dlg.canceled.connect(saver.cancel)

        def finished(saved, errors):
            dlg.reset()
            msg = f"Saved {saved} of {len(jobs)} image(s) to\n{out_dir}"
            if errors:
                msg += "\n\nProblems:\n" + "\n".join(f"• {n}: {e}" for n, e in errors[:10])
            box = QMessageBox(QMessageBox.Warning if errors else QMessageBox.Information, "Save all", msg,
                              parent=self)
            open_btn = box.addButton("Open folder", QMessageBox.ActionRole)
            box.addButton(QMessageBox.Ok)
            box.exec()
            if box.clickedButton() is open_btn:
                from PySide6.QtCore import QUrl
                from PySide6.QtGui import QDesktopServices

                QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))
            self._saver = None

        saver.finished_all.connect(finished)
        self._saver = saver
        saver.start()

    def closeEvent(self, e):
        if self._saver is not None and self._saver.isRunning():
            self._saver.cancel()
            self._saver.wait()
        self._pool.clear()
        super().closeEvent(e)
