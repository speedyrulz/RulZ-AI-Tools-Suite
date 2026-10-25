"""First-run "Choose your look" dialog: Dark (default) or Light, previewed live."""
from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                               QWidget)

from . import theme


class _Preview(QWidget):
    """A tiny mock window in the given theme's colours."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.setFixedSize(200, 120)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def paintEvent(self, e):
        c = theme.colors(self.mode)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(QColor(c["BORDER"]), 1))
        p.setBrush(QColor(c["BG"]))
        p.drawRoundedRect(r, 6, 6)
        # toolbar with the red underline and a primary button
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(c["RED"]))
        p.drawRect(QRectF(1, 22, r.width() - 1, 2))
        p.drawRoundedRect(QRectF(12, 7, 34, 10), 2, 2)
        p.setBrush(QColor(c["MUTED"]))
        p.drawRoundedRect(QRectF(52, 10, 26, 4), 2, 2)
        p.drawRoundedRect(QRectF(84, 10, 26, 4), 2, 2)
        # side panel
        p.setBrush(QColor(c["PANEL"]))
        p.drawRect(QRectF(1, 25, 52, r.height() - 25))
        p.setBrush(QColor(c["TEXT"]))
        for i in range(4):
            p.drawRoundedRect(QRectF(8, 34 + i * 14, 36 - (i % 2) * 10, 4), 2, 2)
        # thumbnail cards, one selected
        for i in range(3):
            card = QRectF(62 + i * 45, 34, 38, 72)
            p.setPen(QPen(QColor(c["RED"]), 2) if i == 1 else QPen(QColor(c["BORDER"]), 1))
            p.setBrush(QColor(c["CARD"]))
            p.drawRoundedRect(card, 3, 3)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(c["MUTED"]))
            p.drawRect(card.adjusted(4, 4, -4, -36))
            p.setBrush(QColor(c["TEXT"]))
            p.drawRoundedRect(QRectF(card.left() + 4, card.bottom() - 26, 30, 3), 1.5, 1.5)
            p.drawRoundedRect(QRectF(card.left() + 4, card.bottom() - 19, 22, 3), 1.5, 1.5)
        p.end()


class _Card(QFrame):
    clicked = Signal(str)

    def __init__(self, mode, title):
        super().__init__()
        self.mode = mode
        self.setObjectName("lookCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 14, 14, 12)
        v.setSpacing(10)
        v.addWidget(_Preview(mode), 0, Qt.AlignHCenter)
        self.label = QLabel(title)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setObjectName("lookTitle")
        v.addWidget(self.label)

    def set_selected(self, on):
        self.setProperty("selected", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.mode)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self.clicked.emit(self.mode)
        else:
            super().keyPressEvent(e)


class ChooseLookDialog(QDialog):
    """Asks Dark or Light. Switching cards previews the theme on the whole app; closing = Dark."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose your look")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.choice = theme.DARK
        v = QVBoxLayout(self)
        v.setContentsMargins(24, 20, 24, 20)
        v.setSpacing(14)
        head = QLabel("Choose your look")
        head.setObjectName("lookHead")
        v.addWidget(head)
        row = QHBoxLayout()
        row.setSpacing(16)
        self.cards = {}
        for mode, title in ((theme.DARK, "Dark"), (theme.LIGHT, "Light")):
            card = _Card(mode, title)
            card.clicked.connect(self._pick)
            row.addWidget(card)
            self.cards[mode] = card
        v.addLayout(row)
        note = QLabel("You can change this any time in View ▸ Light mode.")
        note.setObjectName("lookNote")
        v.addWidget(note)
        btns = QHBoxLayout()
        btns.addStretch(1)
        ok = QPushButton("Continue")
        ok.setDefault(True)
        ok.setMinimumWidth(110)
        ok.clicked.connect(self.accept)
        btns.addWidget(ok)
        v.addLayout(btns)
        self._pick(theme.DARK)
        self.cards[theme.DARK].setFocus()

    def sizeHint(self):
        return QSize(520, 320)

    def _pick(self, mode):
        self.choice = mode
        for m, card in self.cards.items():
            card.set_selected(m == mode)
        theme.apply(QApplication.instance(), mode)
        self._restyle()

    def _restyle(self):
        c = theme.colors()
        self.setStyleSheet(f"""
            QLabel#lookHead {{ font-size: 15pt; font-weight: bold; }}
            QLabel#lookNote {{ color: {c["MUTED"]}; }}
            QLabel#lookTitle {{ font-size: 11pt; font-weight: bold; }}
            QFrame#lookCard {{ background: {c["PANEL"]}; border: 1px solid {c["BORDER"]}; border-radius: 8px; }}
            QFrame#lookCard:hover {{ border-color: {c["RED_HOVER"]}; }}
            QFrame#lookCard[selected="true"] {{ border: 2px solid {c["RED"]}; }}
        """)

    def result_theme(self):
        """The choice to save: the selected card if Continue was pressed, Dark if the dialog was closed."""
        return self.choice if self.result() == QDialog.Accepted else theme.DARK
