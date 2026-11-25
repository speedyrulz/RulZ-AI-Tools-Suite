"""Small reusable widgets."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QPen, QPolygonF, QTextCursor
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QFrame,
                               QGridLayout, QHBoxLayout, QHeaderView, QLabel, QPlainTextEdit,
                               QProgressBar, QPushButton, QRadioButton, QSizePolicy, QSpinBox,
                               QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from ..gpus import GpuInfo
from ..model import GPU_AUTO, GPU_CPU, GPU_SELECT, GpuRef, Profile
from ..textstream import TextStream
from . import theme


class NoWheelComboBox(QComboBox):
    """A combo box that does not change when the page is scrolled over it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event):
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)

    def wheelEvent(self, event):
        event.ignore()


class StatusDot(QWidget):
    """A small status mark. The shape says it as well as the colour: a ring when stopped, a dot
    when running or busy, a cross when failed, a triangle for a port used by another program.
    The colour is read from the active palette at paint time, so a theme switch needs no call."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tone = theme.TONE_STOPPED
        self.setFixedSize(QSize(14, 14))

    def set_tone(self, tone: str):
        if tone != self._tone:
            self._tone = tone
            self.update()

    def tone(self) -> str:
        return self._tone

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = theme.tone_colour(self._tone)
        tone = self._tone
        if tone == theme.TONE_STOPPED:
            painter.setPen(QPen(color, 1.8))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(2.5, 2.5, 9, 9))
        elif tone == theme.TONE_FAILED:
            painter.setPen(QPen(color, 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(3.5, 3.5), QPointF(10.5, 10.5))
            painter.drawLine(QPointF(10.5, 3.5), QPointF(3.5, 10.5))
        elif tone == theme.TONE_EXTERNAL:
            painter.setPen(QPen(color, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                                Qt.PenJoinStyle.RoundJoin))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPolygon(QPolygonF([QPointF(7, 2.5), QPointF(12, 11.5), QPointF(2, 11.5)]))
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QRectF(2.5, 2.5, 9, 9))


class ElidedLabel(QLabel):
    """A one-line label that ends in ... instead of widening its layout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setWidth(self.fontMetrics().horizontalAdvance(self.text()) + 2)
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(40)
        hint.setHeight(max(hint.height(), self.fontMetrics().height()))
        return hint

    def paintEvent(self, _event):
        painter = QPainter(self)
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, self.width())
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), text)


class StatusPill(QFrame):
    """A status badge: a Lucide icon and a short word in the status colour on a neutral fill."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pill")
        self._tone = theme.TONE_STOPPED
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 10, 0)
        layout.setSpacing(4)
        self._icon = QLabel()
        self._icon.setFixedSize(12, 12)
        self._text = QLabel()
        self._text.setObjectName("pillText")
        layout.addWidget(self._icon)
        layout.addWidget(self._text)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(24)
        self.setProperty("tone", self._tone)
        self._draw_icon()

    def set_status(self, text: str, tone: str):
        self._text.setText(text)
        self.updateGeometry()
        if tone != self._tone:
            self._tone = tone
            theme.set_role(self, tone=tone)
            theme.set_role(self._text)
            self._draw_icon()

    def text(self) -> str:
        return self._text.text()

    def tone(self) -> str:
        return self._tone

    def _draw_icon(self):
        token = theme.tone_token(self._tone)
        self._icon.setPixmap(theme.lucide_pixmap(theme.tone_icon(self._tone), theme.C[token], 12))

    def retheme(self):
        self._draw_icon()


def fit_button(button, *texts: str, padding: int = 30):
    """Make a button wide enough for its text (and for other texts it may show later)."""
    metrics = button.fontMetrics()
    widest = max(metrics.horizontalAdvance(t) for t in (texts or (button.text(),)))
    button.setMinimumWidth(widest + padding)


class Card(QFrame):
    """A titled box used for each section of the settings page."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 14, 16, 16)
        self._layout.setSpacing(12)
        self.header = QHBoxLayout()
        self.header.setSpacing(8)
        label = QLabel(title)
        label.setObjectName("cardTitle")
        self.header.addWidget(label)
        self.header.addStretch(1)
        self._layout.addLayout(self.header)

    def body(self) -> QVBoxLayout:
        return self._layout


class CollapsibleSection(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._badge = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.button = QToolButton()
        self.button.setObjectName("sectionHeader")
        self.button.setCheckable(True)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setArrowType(Qt.ArrowType.RightArrow)
        self.button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.button.toggled.connect(self._on_toggled)
        layout.addWidget(self.button)
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(22, 2, 0, 8)
        self.content_layout.setSpacing(2)
        self.content.setVisible(False)
        layout.addWidget(self.content)
        self._refresh_text()

    def _on_toggled(self, checked: bool):
        self.button.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self.content.setVisible(checked)

    def set_expanded(self, expanded: bool):
        self.button.setChecked(expanded)

    def is_expanded(self) -> bool:
        return self.button.isChecked()

    def set_badge(self, text: str):
        self._badge = text
        self._refresh_text()

    def _refresh_text(self):
        self.button.setText(f"  {self._title}" + (f"   ·   {self._badge}" if self._badge else ""))


class LogView(QPlainTextEdit):
    """Shows process output the way a terminal would (progress bars rewrite their line)."""

    def __init__(self, max_lines: int = 5000, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setObjectName("logView")   # coloured by the theme's style sheet, so it follows a switch
        self.setFont(theme.mono_font())
        self.setMaximumBlockCount(max_lines)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self._stream = TextStream()
        self._pending: list[str] = []
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._flush)

    def feed(self, text: str):
        """Add output. Writes are batched so heavy output does not freeze the window."""
        self._pending.append(text)
        if not self._timer.isActive():
            self._timer.start()

    def set_wrap(self, wrap: bool):
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth if wrap
                             else QPlainTextEdit.LineWrapMode.NoWrap)

    def clear_log(self):
        self._pending.clear()
        self.clear()

    def _flush(self):
        if not self._pending:
            return
        text = "".join(self._pending)
        self._pending.clear()
        steps = self._stream.feed(text)
        if not steps:
            return
        bar = self.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        cursor = QTextCursor(self.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.beginEditBlock()
        for step in steps:
            kind = step[0]
            if kind == "text":
                cursor.insertText(step[1])
            elif kind == "newline":
                cursor.insertBlock()
            else:  # restart: the next text replaces the current line
                cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock,
                                    QTextCursor.MoveMode.KeepAnchor)
                cursor.removeSelectedText()
        cursor.endEditBlock()
        if at_bottom:
            bar.setValue(bar.maximum())


class GpuSelector(QWidget):
    """Pick how a profile uses GPUs: all of them, chosen ones, or none (CPU)."""
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._profile: Profile | None = None
        self._gpus: list[GpuInfo] = []
        self._rows: dict[str, dict] = {}
        self._loading = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        modes = QHBoxLayout()
        modes.setSpacing(18)
        self._mode_group = QButtonGroup(self)
        self._mode_buttons: dict[str, QRadioButton] = {}
        for mode, text, tip in (
            (GPU_SELECT, "Selected GPUs", "ComfyUI only sees the cards ticked below (--cuda-device)."),
            (GPU_AUTO, "All GPUs", "No GPU selection is passed. ComfyUI in WSL then sees every card."),
            (GPU_CPU, "CPU only", "Run everything on the CPU (--cpu). Slow."),
        ):
            button = QRadioButton(text)
            button.setToolTip(tip)
            self._mode_group.addButton(button)
            self._mode_buttons[mode] = button
            button.toggled.connect(lambda checked, mode=mode: self._on_mode(mode, checked))
            modes.addWidget(button)
        modes.addStretch(1)
        layout.addLayout(modes)

        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(14)
        self._grid.setVerticalSpacing(6)
        self._grid.setColumnStretch(0, 1)
        layout.addWidget(self._grid_host)

        self._primary_group = QButtonGroup(self)
        self._hint = QLabel()
        self._hint.setObjectName("hint")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

    # -- data in ------------------------------------------------------------
    def set_profile(self, profile: Profile | None):
        self._profile = profile
        self._rebuild()

    def set_gpus(self, gpus: list[GpuInfo]):
        known = [g.uuid for g in self._gpus]
        self._gpus = list(gpus)
        if [g.uuid for g in gpus] != known:
            self._rebuild()
        else:
            self._update_stats()

    # -- building -----------------------------------------------------------
    def _entries(self) -> list[tuple[str, str, GpuInfo | None]]:
        """(uuid, label, live info) for detected cards plus saved cards that are missing."""
        entries = [(g.uuid, f"GPU {g.index}   {g.name}", g) for g in self._gpus]
        seen = {g.uuid for g in self._gpus}
        if self._profile:
            for ref in self._profile.gpus:
                if ref.uuid not in seen:
                    entries.append((ref.uuid, f"{ref.name or ref.uuid}   (not detected)", None))
        return entries

    def _rebuild(self):
        self._loading = True
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().hide()   # deleted later: must not be painted over the new rows meanwhile
                item.widget().deleteLater()
        for button in self._primary_group.buttons():
            self._primary_group.removeButton(button)
        self._rows.clear()

        profile = self._profile
        chosen = [g.uuid for g in profile.gpus] if profile else []
        for row, (uuid, label, _info) in enumerate(self._entries()):
            check = QCheckBox(label)
            check.setChecked(uuid in chosen)
            check.toggled.connect(lambda checked, uuid=uuid: self._on_check(uuid, checked))
            primary = QRadioButton("Primary")
            primary.setToolTip("The primary card is the one ComfyUI loads models on by default (cuda:0).")
            primary.toggled.connect(lambda checked, uuid=uuid: self._on_primary(uuid, checked))
            self._primary_group.addButton(primary)
            usage = QLabel()
            usage.setObjectName("muted")
            usage.setMinimumWidth(210)
            bar = QProgressBar()
            bar.setTextVisible(False)
            bar.setRange(0, 100)
            bar.setFixedWidth(120)
            self._grid.addWidget(check, row, 0)
            self._grid.addWidget(primary, row, 1)
            self._grid.addWidget(bar, row, 2)
            self._grid.addWidget(usage, row, 3)
            self._rows[uuid] = {"check": check, "primary": primary, "usage": usage, "bar": bar}

        if profile:
            self._mode_buttons[profile.gpu_mode].setChecked(True)
        self._loading = False
        self._update_stats()
        self._sync_enabled()

    def _update_stats(self):
        for gpu in self._gpus:
            row = self._rows.get(gpu.uuid)
            if not row:
                continue
            row["usage"].setText(f"{gpu.memory_text}   ·   {gpu.utilization}%   ·   {gpu.temperature}°C")
            row["bar"].setValue(int(100 * gpu.memory_used / gpu.memory_total) if gpu.memory_total else 0)

    def _sync_enabled(self):
        profile = self._profile
        selecting = bool(profile) and profile.gpu_mode == GPU_SELECT
        chosen = [g.uuid for g in profile.gpus] if profile else []
        self._loading = True
        self._primary_group.setExclusive(False)
        for uuid, row in self._rows.items():
            row["check"].setEnabled(selecting)
            is_chosen = uuid in chosen
            row["primary"].setEnabled(selecting and is_chosen and len(chosen) > 1)
            row["primary"].setVisible(selecting and len(chosen) > 1)
            row["primary"].setChecked(selecting and is_chosen and chosen[0] == uuid and len(chosen) > 1)
        self._primary_group.setExclusive(True)
        self._loading = False

        if not profile:
            self._hint.setText("")
        elif not self._rows:
            self._hint.setText("No NVIDIA GPU was detected (nvidia-smi was not found on Windows).")
        elif profile.gpu_mode == GPU_SELECT:
            if not chosen:
                self._hint.setText("Tick at least one GPU.")
            elif len(chosen) > 1:
                self._hint.setText("With several GPUs, the primary one is listed first, so it becomes cuda:0 "
                                   "inside ComfyUI. Multi-GPU nodes can use the others.")
            else:
                self._hint.setText("ComfyUI will only see this card.")
        elif profile.gpu_mode == GPU_AUTO:
            self._hint.setText("No GPU selection is passed, so ComfyUI sees every card and picks its own default.")
        else:
            self._hint.setText("ComfyUI runs on the CPU and uses no GPU.")

    # -- edits --------------------------------------------------------------
    def _name_for(self, uuid: str) -> str:
        for gpu in self._gpus:
            if gpu.uuid == uuid:
                return gpu.name
        if self._profile:
            for ref in self._profile.gpus:
                if ref.uuid == uuid:
                    return ref.name
        return ""

    def _on_mode(self, mode: str, checked: bool):
        if self._loading or not checked or not self._profile:
            return
        self._profile.gpu_mode = mode
        self._sync_enabled()
        self.changed.emit()

    def _on_check(self, uuid: str, checked: bool):
        if self._loading or not self._profile:
            return
        gpus = [g for g in self._profile.gpus if g.uuid != uuid]
        if checked:
            gpus.append(GpuRef(uuid=uuid, name=self._name_for(uuid)))
        self._profile.gpus = gpus
        self._sync_enabled()
        self.changed.emit()

    def _on_primary(self, uuid: str, checked: bool):
        if self._loading or not checked or not self._profile:
            return
        gpus = self._profile.gpus
        first = [g for g in gpus if g.uuid == uuid]
        if first and gpus[0].uuid != uuid:
            self._profile.gpus = first + [g for g in gpus if g.uuid != uuid]
            self._sync_enabled()
            self.changed.emit()

    def refresh_names(self):
        """Store the detected card names in the profile (kept for when a card is missing)."""
        if not self._profile:
            return
        names = {g.uuid: g.name for g in self._gpus}
        for ref in self._profile.gpus:
            if names.get(ref.uuid):
                ref.name = names[ref.uuid]


class EnvTable(QWidget):
    """Environment variables set inside WSL before ComfyUI starts."""
    changed = Signal()
    import_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loading = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Name", "Value"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 220)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setFont(theme.mono_font())
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(30)
        self.table.horizontalHeader().setFixedHeight(32)
        self.table.horizontalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        add = QPushButton("Add variable")
        add.clicked.connect(self._add_row)
        remove = QPushButton("Remove selected")
        remove.setProperty("danger", True)
        remove.clicked.connect(self._remove_selected)
        importer = QPushButton("Import from ~/.bashrc")
        importer.setToolTip("Find the variables your terminal sets in ~/.bashrc (for example CUDA paths) "
                            "and add them here. A server started from the launcher does not read ~/.bashrc.")
        importer.clicked.connect(self.import_requested)
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch(1)
        buttons.addWidget(importer)
        layout.addLayout(buttons)
        self._fit_height()

    def set_rows(self, rows: list[list[str]]):
        self._loading = True
        self.table.setRowCount(0)
        for name, value in rows:
            self._append(name, value)
        self._loading = False
        self._fit_height()

    def rows(self) -> list[list[str]]:
        result = []
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text().strip() if self.table.item(row, 0) else ""
            value = self.table.item(row, 1).text() if self.table.item(row, 1) else ""
            if name or value.strip():
                result.append([name, value])
        return result

    def _append(self, name: str, value: str):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(name))
        self.table.setItem(row, 1, QTableWidgetItem(value))

    def _add_row(self):
        self._loading = True
        self._append("", "")
        self._loading = False
        self._fit_height()
        row = self.table.rowCount() - 1
        self.table.setCurrentCell(row, 0)
        self.table.editItem(self.table.item(row, 0))

    def _remove_selected(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        if not rows and self.table.currentRow() >= 0:
            rows = [self.table.currentRow()]
        for row in rows:
            self.table.removeRow(row)
        self._fit_height()
        if rows:
            self.changed.emit()

    def _on_item_changed(self, _item):
        if not self._loading:
            self.changed.emit()

    def _fit_height(self):
        rows = max(2, min(8, self.table.rowCount()))
        row_height = self.table.verticalHeader().defaultSectionSize()
        self.table.setFixedHeight(self.table.horizontalHeader().height() + rows * row_height + 6)


class GpuStrip(QWidget):
    """Compact GPU usage shown in the status bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 2, 8, 2)
        self._layout.setSpacing(18)
        self._items: dict[str, tuple[QLabel, QProgressBar]] = {}
        self._empty = QLabel("No NVIDIA GPU detected")
        self._empty.setObjectName("muted")
        self._layout.addWidget(self._empty)
        self._layout.addStretch(1)

    def set_gpus(self, gpus: list[GpuInfo]):
        if [g.uuid for g in gpus] != list(self._items):
            for label, bar in self._items.values():
                label.deleteLater()
                bar.deleteLater()
            self._items.clear()
            self._empty.setVisible(not gpus)
            position = 0
            for gpu in gpus:
                label = QLabel()
                bar = QProgressBar()
                bar.setTextVisible(False)
                bar.setRange(0, 100)
                bar.setFixedWidth(70)
                self._layout.insertWidget(position, label)
                self._layout.insertWidget(position + 1, bar)
                position += 2
                self._items[gpu.uuid] = (label, bar)
        for gpu in gpus:
            label, bar = self._items[gpu.uuid]
            label.setText(f"GPU {gpu.index}  {gpu.short_name}   {gpu.memory_text}   {gpu.utilization}%   "
                          f"{gpu.temperature}°C")
            bar.setValue(int(100 * gpu.memory_used / gpu.memory_total) if gpu.memory_total else 0)
