"""AI tagging and search UI: the Library card, Advanced AI settings, and the Look for / Tag dialogs."""
from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QCompleter, QDialog, QDoubleSpinBox, QFormLayout,
                               QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit,
                               QRadioButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from ..ai import client, settings as ai_settings
from ..ai.jobs import Job
from . import dialogs, theme
from .common import Segmented, button, label, plural

INTRO = ("Optional. A vision AI model can tag your photos in more detail and look for things you name, like "
         "“red sweater”. It runs on an AI server you choose: Ollama, LM Studio or llama.cpp on this PC, or the "
         "server you set up in the Rulz Training Captioner. Nothing is sent anywhere while this is off.")
OWN_HINT = ("Ollama: http://127.0.0.1:11434/v1 with a vision model, for example qwen2.5vl:7b or gemma3 "
            "(get one with “ollama pull qwen2.5vl:7b”). LM Studio: http://127.0.0.1:1234/v1. Several addresses "
            "separated by commas share the work.")
CAPTIONER_NOTE = ("Power-user option: RulZ uses the Rulz Training Captioner's server settings. Change them in the "
                  "Captioner (Settings ▸ Backend): the next AI job here picks them up. The Captioner's captioning "
                  "prompts aren't used; RulZ has its own (Advanced…).")
REMOTE_WARNING = ("This AI server isn't on this PC: photos you tag or search are sent to it over the network.")


class _ModelSignals(QObject):
    done = Signal(object, str)  # model names (or None), error


class _ModelFetch(QRunnable):
    def __init__(self, url: str, key: str, sig: _ModelSignals):
        super().__init__()
        self.url, self.key, self.sig = url, key, sig

    def run(self):
        try:
            result = client.list_models(self.url, self.key), ""
        except Exception as e:  # noqa: BLE001
            result = None, str(e)
        try:
            self.sig.done.emit(*result)
        except RuntimeError:  # the window closed while the server was being asked
            pass


class AICard(QFrame):
    """Library ▸ AI tagging and search."""

    MODES = ai_settings.MODES  # the order of the segmented control

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 20)
        lay.setSpacing(10)
        lay.addWidget(label("AI tagging and search", "H2"))
        lay.addWidget(label(INTRO, "Muted", wrap=True))
        self.mode = Segmented(["Off", "My own server", "Training Captioner's server"])
        self.mode.changed.connect(self._mode_changed)
        lay.addWidget(self.mode)

        self.own = QWidget()
        form = QFormLayout(self.own)
        form.setContentsMargins(0, 4, 0, 0)
        form.setHorizontalSpacing(14)
        self.urls = QLineEdit()
        self.urls.setProperty("plain", True)
        self.urls.setPlaceholderText(ai_settings.DEFAULTS["urls"])
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.setInsertPolicy(QComboBox.NoInsert)
        self.model.lineEdit().setPlaceholderText("Model name, e.g. qwen2.5vl:7b")
        self.model.setMinimumWidth(260)
        self.load_btn = button("Load models", "refresh", tooltip="Ask the server which models it has")
        self.load_btn.clicked.connect(self._load_models)
        mrow = QHBoxLayout()
        mrow.addWidget(self.model, 1)
        mrow.addWidget(self.load_btn)
        self.key = QLineEdit()
        self.key.setProperty("plain", True)
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("Only if the server needs one")
        form.addRow("Server address", self.urls)
        form.addRow("Model", mrow)
        form.addRow("API key", self.key)
        hint = label(OWN_HINT, "Muted", wrap=True)
        hint.setStyleSheet("font-size: 12px;")
        form.addRow(hint)
        lay.addWidget(self.own)

        self.cap = QWidget()
        cl = QVBoxLayout(self.cap)
        cl.setContentsMargins(0, 4, 0, 0)
        cl.setSpacing(6)
        self.cap_text = label("", wrap=True)
        self.cap_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.cap_path = label("", "Muted", wrap=True)
        self.cap_path.setStyleSheet("font-size: 12px;")
        self.cap_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        note = label(CAPTIONER_NOTE, wrap=True)
        theme.set_style(note, "color: %(accent)s; font-size: 12px;")
        cl.addWidget(self.cap_text)
        cl.addWidget(self.cap_path)
        cl.addWidget(note)
        lay.addWidget(self.cap)

        self.remote_warn = label(REMOTE_WARNING, wrap=True)
        theme.set_style(self.remote_warn, "color: %(danger)s; font-weight: 600;")
        lay.addWidget(self.remote_warn)
        self.status = label("", "Muted", wrap=True)
        self.status.setTextFormat(Qt.PlainText)  # it quotes model replies and server errors
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        self.test_btn = button("Test connection", "check", tooltip="Sends a small test picture and checks the reply")
        self.test_btn.clicked.connect(self._test)
        self.adv_btn = button("Advanced…", "settings", tooltip="Prompts and model settings (for power users)")
        self.adv_btn.clicked.connect(self._advanced)
        self.stop_srv_btn = button("Stop AI server", "stop",
                                   tooltip="Ends the llama-server RulZ started, which frees its VRAM")
        self.stop_srv_btn.clicked.connect(self._stop_server)
        row.addWidget(self.test_btn)
        row.addWidget(self.adv_btn)
        row.addWidget(self.stop_srv_btn)
        row.addStretch(1)
        lay.addLayout(row)
        srow = QHBoxLayout()
        self.stats = label("", "Muted")
        self.clear_btn = button("Clear AI results…", flat=True, danger=True)
        self.clear_btn.clicked.connect(self._clear)
        srow.addWidget(self.stats, 1)
        srow.addWidget(self.clear_btn)
        lay.addLayout(srow)

        self._test_message: tuple[str, bool] | None = None
        self._save_timer = QTimer(self, singleShot=True, interval=600)
        self._save_timer.timeout.connect(self.flush)
        for w in (self.urls, self.key):
            w.textEdited.connect(self._edited)
        self.model.editTextChanged.connect(self._edited)
        self._models = _ModelSignals(self)
        self._models.done.connect(self._models_loaded)
        ctx.ai.started.connect(lambda *_: self._update())
        ctx.ai.finished.connect(self._ai_finished)
        self.load()

    # ---- settings <-> fields ------------------------------------------------------------------
    def load(self):
        """Show the saved settings (a change still waiting to be saved is saved first)."""
        self.flush()
        s = ai_settings.load()
        self.mode.set_current(self.MODES.index(s["mode"]))
        for edit, value in ((self.urls, s["urls"]), (self.key, s["api_key"])):
            if edit.text() != value:
                edit.setText(value)
        if self.model.currentText() != s["model"]:
            self.model.blockSignals(True)
            self.model.setEditText(s["model"])
            self.model.blockSignals(False)
        self._update(s)

    def _edited(self, *_):
        self._test_message = None
        self._save_timer.start()

    def flush(self):
        """Save the fields now if an edit is waiting."""
        if self._save_timer.isActive():
            self._save_timer.stop()
            self._save()

    def _save(self, mode: str | None = None):
        s = ai_settings.load()
        s.update(mode=mode or self.MODES[self.mode.current()], urls=self.urls.text().strip(),
                 model=self.model.currentText().strip(), api_key=self.key.text().strip())
        try:
            ai_settings.save(s)
        except OSError as e:
            self.status.setText(f"Couldn't save the AI settings: {e.strerror or e}")
            return
        self._update(s)
        self.ctx.ai_settings_changed()

    def _mode_changed(self, i: int):
        self._save_timer.stop()
        self._test_message = None
        self._save(self.MODES[i])

    def _update(self, s: dict | None = None):
        s = s or ai_settings.load()
        mode = s["mode"]
        self.own.setVisible(mode == "own")
        self.cap.setVisible(mode == "captioner")
        conn = ai_settings.resolve(s)
        if mode == "captioner":
            self.cap_text.setText(ai_settings.describe(conn) if conn.captioner else "")
            self.cap_path.setText(f"Settings read from {ai_settings.captioner_settings_file()}")
        self.remote_warn.setVisible(mode != "off" and conn.remote)
        busy = self.ctx.ai.running
        if mode == "off":
            text, bad = "Off.", False
        elif conn.problem:
            text, bad = conn.problem, True
        elif busy and self.ctx.ai.job.kind == "test":
            text, bad = "Testing… (a llama-server that has to start first can take a few minutes)", False
        elif self._test_message:
            text, bad = self._test_message
        else:
            text, bad = "Ready. Test connection checks that the model can see photos.", False
        self.status.setText(text)
        theme.set_style(self.status, "color: %(danger)s;" if bad else "color: %(muted)s;")
        self.test_btn.setEnabled(mode != "off" and not conn.problem and not busy)
        self.stop_srv_btn.setVisible(self.ctx.ai.server_running())
        st = self.ctx.db.ai_stats()
        self.stats.setText(f"{plural(st['tagged'], 'photo')} with AI tags · "
                           f"{plural(st['searches'], 'saved Look for search')}".replace("searchs", "searches"))
        self.clear_btn.setEnabled(bool(st["rows"] or st["searches"]))  # results of changed photos count too

    def refresh(self):
        self.load()

    # ---- actions ----------------------------------------------------------------------------------
    def _load_models(self):
        self.flush()
        eps = ai_settings.endpoints_from_urls(self.urls.text(), "", 1)
        if not eps:
            self.status.setText("Type the server address first.")
            return
        self.load_btn.setEnabled(False)
        self.status.setText(f"Asking {eps[0]['url']} for its models…")
        QThreadPool.globalInstance().start(_ModelFetch(eps[0]["url"], self.key.text().strip(), self._models))

    def _models_loaded(self, models, err: str):
        self.load_btn.setEnabled(True)
        if models is None:
            self._test_message = (err, True)
        elif not models:
            self._test_message = ("The server listed no models. Type the model name yourself.", True)
        else:
            current = self.model.currentText()
            self.model.blockSignals(True)
            self.model.clear()
            self.model.addItems(models)
            self.model.setEditText(current or models[0])
            self.model.blockSignals(False)
            if not current:
                self._edited()
                self.flush()  # saved now, so the status below doesn't still say "Choose a model"
            self._test_message = (f"The server has {plural(len(models), 'model')}. Pick a vision model "
                                  "(one that can see images).", False)
        self._update()

    def _test(self):
        self.flush()
        self._test_message = None
        self.ctx.run_ai(Job("test"))
        self._update()

    def _ai_finished(self, job, msg: str, ok: bool):
        if job.kind == "test":
            self._test_message = (msg, not ok or not msg.startswith("Connected"))
        self._update()

    def _advanced(self):
        self.flush()
        s = ai_settings.load()
        dlg = AdvancedDialog(self, s)
        if dlg.exec() == QDialog.Accepted:
            try:
                ai_settings.save(dlg.result_settings(s))
            except OSError as e:
                dialogs.info(self, "Not saved", f"Couldn't save the AI settings: {e.strerror or e}")
            self._update()

    def _stop_server(self):
        if self.ctx.ai.running and not dialogs.confirm(
                self, "Stop AI server", "An AI job is running. Stopping the server stops the job too.",
                ok_text="Stop", danger=True):
            return
        self.ctx.ai.stop()
        self.ctx.ai.stop_server()
        self._update()

    def _clear(self):
        st = self.ctx.db.ai_stats()
        if dialogs.confirm(self, "Clear AI results",
                           f"Forget the AI tags of {plural(st['tagged'], 'photo')} and "
                           f"{plural(st['searches'], 'saved Look for search')}, with all their answers?\n\n"
                           "Your photos, people and face tags are not touched. Asking the AI again takes as long "
                           "as the first time.".replace("searchs", "searches"),
                           ok_text="Clear", danger=True):
            self.ctx.db.ai_clear()
            self.ctx.data_changed()
            self._update()


# ---- Advanced AI settings -----------------------------------------------------------------------------
_LABELS = {
    "tag_system": "tagging instructions", "tag_user": "tagging request", "max_tags": "tags per photo",
    "tag_max_tokens": "tagging token limit", "look_system": "Look for instructions", "look_user": "Look for question",
    "look_max_tokens": "Look for token limit", "temperature": "temperature", "max_image_side": "image size",
    "concurrency": "requests at a time", "timeout": "reply timeout", "disable_thinking": "thinking",
    "managed_port": "llama-server port",
}
ADV_WARNING = ("<b>For power users.</b> These are the instructions RulZ gives the AI model and the settings it "
               "sends with them. Changing them can make tagging and Look for… fail or give wrong answers: RulZ reads "
               "a <b>comma-separated list</b> from tagging and a reply that <b>starts with YES or NO</b> from "
               "Look for…. Results saved before a change stay as they are.")


def _spin(lo, hi, value, step=1, suffix="") -> QSpinBox:
    sp = QSpinBox()
    sp.setRange(lo, hi)
    sp.setSingleStep(step)
    sp.setValue(value)
    if suffix:
        sp.setSuffix(suffix)
    _number_box(sp)
    return sp


def _number_box(sp):
    # The stylesheet hides the arrow buttons (Fusion drew empty squares); typing, the wheel and ↑/↓ still step.
    # Not setButtonSymbols(NoButtons): with a stylesheet, Qt then shrinks the text field to 1 px.
    sp.setMaximumWidth(170)


def _note(text: str) -> QLabel:
    lb = label(text, "Muted", wrap=True)
    lb.setStyleSheet("font-size: 12px;")
    return lb


class AdvancedDialog(dialogs._Base):
    def __init__(self, parent, s: dict):
        super().__init__(parent, "Advanced AI settings", 680)
        warn = QFrame()
        warn.setObjectName("Warning")
        wl = QVBoxLayout(warn)
        wl.setContentsMargins(14, 10, 14, 10)
        wl.setSpacing(6)
        text = QLabel(ADV_WARNING)
        text.setWordWrap(True)
        wl.addWidget(text)
        self.unlock = QCheckBox("I understand. Let me change these settings.")
        wl.addWidget(self.unlock)
        self.lay.addWidget(warn)

        tabs = QTabWidget()
        p = ai_settings.prompt
        # tagging
        w = QWidget()
        f = QFormLayout(w)
        self.tag_system = self._text(p(s, "tag_system"), 170)
        self.tag_user = self._text(p(s, "tag_user"), 56)
        self.max_tags = _spin(*ai_settings.LIMITS["max_tags"], s["max_tags"])
        self.tag_max_tokens = _spin(*ai_settings.LIMITS["tag_max_tokens"], s["tag_max_tokens"], 50)
        f.addRow("Instructions\n(system prompt)", self.tag_system)
        f.addRow("Request", self.tag_user)
        f.addRow("Tags kept per photo", self.max_tags)
        f.addRow("Token limit", self.tag_max_tokens)
        f.addRow(_note("The reply must be a comma-separated list of short tags. Longer phrases (over six words) are "
                       "dropped as sentences."))
        tabs.addTab(w, "Tagging")
        # look for
        w = QWidget()
        f = QFormLayout(w)
        self.look_system = self._text(p(s, "look_system"), 150)
        self.look_user = self._text(p(s, "look_user"), 70)
        self.look_max_tokens = _spin(*ai_settings.LIMITS["look_max_tokens"], s["look_max_tokens"], 8)
        f.addRow("Instructions\n(system prompt)", self.look_system)
        f.addRow("Question", self.look_user)
        f.addRow("Token limit", self.look_max_tokens)
        f.addRow(_note(f"{ai_settings.QUERY_PLACEHOLDER} in the question is replaced by what you look for. The reply "
                       "must start with YES or NO (a reply that is neither counts as failed and isn't saved)."))
        tabs.addTab(w, "Look for")
        # model and server
        w = QWidget()
        f = QFormLayout(w)
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(*ai_settings.LIMITS["temperature"])
        self.temperature.setSingleStep(0.05)
        self.temperature.setValue(s["temperature"])
        _number_box(self.temperature)
        self.max_image_side = _spin(*ai_settings.LIMITS["max_image_side"], s["max_image_side"], 128, " px")
        self.concurrency = _spin(*ai_settings.LIMITS["concurrency"], s["concurrency"])
        self.timeout = _spin(*ai_settings.LIMITS["timeout"], s["timeout"], 30, " s")
        self.disable_thinking = QCheckBox("Turn off thinking (recommended)")
        self.disable_thinking.setChecked(s["disable_thinking"])
        self.managed_port = _spin(*ai_settings.LIMITS["managed_port"], s["managed_port"])
        f.addRow("Temperature", self.temperature)
        f.addRow("", _note("Low values give steadier answers; tagging and YES/NO checks work best near 0."))
        f.addRow("Image size sent", self.max_image_side)
        f.addRow("Requests at a time", self.concurrency)
        f.addRow("", _note("Per server address, on your own server. The Training Captioner option uses its "
                           "“Concurrent requests per server”. More only helps if the server runs requests in "
                           "parallel (Ollama: OLLAMA_NUM_PARALLEL)."))
        f.addRow("Reply timeout", self.timeout)
        f.addRow("", self.disable_thinking)
        f.addRow("", _note("Thinking models (Qwen3, Gemma 4…) can use up the token limit before they answer, and "
                           "are much slower."))
        f.addRow("First port for RulZ's\nown llama-server", self.managed_port)
        f.addRow("", _note("Only for the Training Captioner's llama.cpp setup, when the Captioner's own server isn't "
                           "running and RulZ starts its own copy (it needs the VRAM for a second copy if the "
                           "Captioner starts its server too)."))
        tabs.addTab(w, "Model and server")
        self.lay.addWidget(tabs, 1)

        self._editors = [self.tag_system, self.tag_user, self.max_tags, self.tag_max_tokens, self.look_system,
                         self.look_user, self.look_max_tokens, self.temperature, self.max_image_side,
                         self.concurrency, self.timeout, self.disable_thinking, self.managed_port]
        row = QHBoxLayout()
        self.restore_btn = button("Restore defaults", flat=True)
        self.restore_btn.clicked.connect(self._restore)
        row.addWidget(self.restore_btn)
        row.addStretch(1)
        cancel = button("Cancel")
        cancel.clicked.connect(self.reject)
        self.save_btn = button("Save", primary=True)
        self.save_btn.clicked.connect(self.accept)
        row.addWidget(cancel)
        row.addWidget(self.save_btn)
        self.lay.addLayout(row)
        self.unlock.toggled.connect(self._set_editable)
        self._set_editable(False)
        self._initial = self.values()
        self.resize(720, 640)

    @staticmethod
    def _text(value: str, height: int) -> QPlainTextEdit:
        e = QPlainTextEdit(value)
        e.setMinimumHeight(height)
        e.setTabChangesFocus(True)
        return e

    def _set_editable(self, on: bool):
        for e in self._editors:
            e.setEnabled(on)
        self.restore_btn.setEnabled(on)
        self.save_btn.setEnabled(on)

    def values(self) -> dict:
        return {
            "tag_system": self.tag_system.toPlainText(), "tag_user": self.tag_user.toPlainText(),
            "max_tags": self.max_tags.value(), "tag_max_tokens": self.tag_max_tokens.value(),
            "look_system": self.look_system.toPlainText(), "look_user": self.look_user.toPlainText(),
            "look_max_tokens": self.look_max_tokens.value(), "temperature": round(self.temperature.value(), 2),
            "max_image_side": self.max_image_side.value(), "concurrency": self.concurrency.value(),
            "timeout": self.timeout.value(), "disable_thinking": self.disable_thinking.isChecked(),
            "managed_port": self.managed_port.value(),
        }

    def _restore(self):
        d = ai_settings.DEFAULTS
        for k, default in ai_settings.PROMPT_DEFAULTS.items():
            getattr(self, k).setPlainText(default)
        for k in ("max_tags", "tag_max_tokens", "look_max_tokens", "max_image_side", "concurrency", "timeout",
                  "managed_port"):
            getattr(self, k).setValue(d[k])
        self.temperature.setValue(d["temperature"])
        self.disable_thinking.setChecked(d["disable_thinking"])

    def changed(self) -> list[str]:
        now = self.values()
        return [_LABELS[k] for k, v in now.items() if v != self._initial[k]]

    def accept(self):
        changed = self.changed()
        if not changed:
            super().accept()
            return
        vals = self.values()
        notes = []
        if ai_settings.QUERY_PLACEHOLDER not in vals["look_user"] and vals["look_user"].strip():
            notes.append(f"The Look for question has no {ai_settings.QUERY_PLACEHOLDER}: what you look for is added "
                         "on a line after it.")
        blank = [_LABELS[k] for k in ai_settings.PROMPT_DEFAULTS if not vals[k].strip()]
        if blank:
            notes.append(f"Left empty, so the built-in text is used: {', '.join(blank)}.")
        if not vals["disable_thinking"]:
            notes.append("With thinking on, replies are slower and RulZ allows extra tokens for the thinking.")
        text = (f"You changed: {', '.join(changed)}.\n\nThe next AI job uses the new settings and may give different "
                "or failed results. Tags and Look for answers saved before stay as they are; use “Tag again” or "
                "“Check again” to redo them.")
        if notes:
            text += "\n\n" + "\n".join(notes)
        if dialogs.confirm(self, "Save advanced AI settings?", text, ok_text="Save changes", danger=True):
            super().accept()

    def result_settings(self, s: dict) -> dict:
        out = dict(s)
        out.update(self.values())
        return out


# ---- Look for / Tag dialog -----------------------------------------------------------------------------
class AIJobDialog(dialogs._Base):
    """Pick what to look for (or confirm tagging) and which photos: this one, the selected, the shown, or all."""

    def __init__(self, parent, ctx, kind: str, *, current: int | None = None, selected: list[int] | None = None,
                 shown: list[int] | None = None, query: str = "", force: bool = False):
        look = kind == "look"
        super().__init__(parent, "Look for something" if look else "Tag photos with AI", 500)
        self.ctx, self.db, self.kind = ctx, ctx.db, kind
        self.total = self.db.stats()["photos"]
        self.text: QLineEdit | None = None
        if look:
            self.lay.addWidget(label("What should the AI look for in each photo? For several things, separate them "
                                     "with commas.", "Muted", wrap=True))
            self.text = QLineEdit(query)
            self.text.setProperty("plain", True)
            self.text.setPlaceholderText("e.g. red sweater")
            comp = QCompleter([s["text"] for s in self.db.ai_searches()], self)
            comp.setCaseSensitivity(Qt.CaseInsensitive)
            comp.setFilterMode(Qt.MatchContains)
            self.text.setCompleter(comp)
            self.lay.addWidget(self.text)
        else:
            self.lay.addWidget(label("The AI lists what it sees in each photo: people, clothing, objects, places, "
                                     "colours and more. Search then finds photos by any of these words.",
                                     "Muted", wrap=True))
        self.scopes: list[tuple[QRadioButton, list[int] | None]] = []
        group = QButtonGroup(self)
        options = []
        if current is not None:
            options.append(("This photo", [current]))
        if selected:
            options.append((f"Selected photos ({len(selected):,})", list(selected)))
        if shown is not None and 0 < len(shown) < self.total:
            options.append((f"Photos shown ({len(shown):,})", list(shown)))
        options.append((f"All photos ({self.total:,})", None))
        for text, ids in options:
            rb = QRadioButton(text)
            group.addButton(rb)
            self.lay.addWidget(rb)
            self.scopes.append((rb, ids))
        self.scopes[0][0].setChecked(True)
        self.force = QCheckBox("Check again photos already checked for this" if look
                               else "Tag again photos that already have AI tags")
        self.force.setChecked(force)  # the viewer's "Tag again"
        self.lay.addWidget(self.force)
        self.info = label("", "Muted", wrap=True)
        self.lay.addWidget(self.info)

        self.saved_list: QListWidget | None = None
        if look:
            searches = self.db.ai_searches()
            if searches:
                self.lay.addWidget(label("Earlier searches", "Muted"))
                self.saved_list = QListWidget()
                self.saved_list.setMaximumHeight(150)
                self.saved_list.itemClicked.connect(lambda it: self.text.setText(it.data(Qt.UserRole)))
                self.lay.addWidget(self.saved_list)
                forget = button("Forget the selected search", "delete", flat=True)
                forget.clicked.connect(self._forget)
                frow = QHBoxLayout()
                frow.addStretch(1)
                frow.addWidget(forget)
                self.lay.addLayout(frow)
                self._fill_saved()
        self.ok = self.add_buttons("Look for it" if look else "Tag photos")
        # connected only now: _update_info needs the info label and the OK button
        group.buttonToggled.connect(self._update_info)
        self.force.toggled.connect(self._update_info)
        if self.text is not None:
            self._timer = QTimer(self, singleShot=True, interval=250)
            self._timer.timeout.connect(self._update_info)
            self.text.textChanged.connect(lambda: self._timer.start())
            self.text.returnPressed.connect(self.accept)
            self.text.setFocus()
        self._update_info()

    def _fill_saved(self):
        self.saved_list.clear()
        for s in self.db.ai_searches():
            it = QListWidgetItem(f"{s['text']}   ·   found in {s['found']:,} of {s['checked']:,} checked")
            it.setData(Qt.UserRole, s["text"])
            it.setData(Qt.UserRole + 1, s["id"])
            self.saved_list.addItem(it)

    def _forget(self):
        it = self.saved_list.currentItem() if self.saved_list else None
        if it is None:
            return
        if dialogs.confirm(self, "Forget search", f"Forget “{it.data(Qt.UserRole)}” and every answer the AI gave "
                           "for it? Looking for it again asks the AI again.", ok_text="Forget", danger=True):
            self.db.ai_forget_search(it.data(Qt.UserRole + 1))
            self.ctx.data_changed(quiet=True)
            self._fill_saved()
            self._update_info()

    def queries(self) -> list[str]:
        if self.text is None:
            return []
        return list(dict.fromkeys(q for q in (self.db.norm_search(x) for x in self.text.text().split(",")) if q))

    def scope(self) -> list[int] | None:
        return next(ids for rb, ids in self.scopes if rb.isChecked())

    def _update_info(self, *_):
        ids = self.scope()
        # photos the AI can be asked about: not ones the scan couldn't read (they are never sent)
        n = len(self.db.ai_todo_tags(ids, force=True))
        qs = self.queries()
        self.nothing_to_do = False  # True: every photo was checked before, so OK only shows the results
        self.ok.setText("Look for it" if self.kind == "look" else "Tag photos")
        if self.kind == "look" and not qs:
            self.info.setText("Type what to look for.")
            self.ok.setEnabled(False)
            return
        if not n:
            self.info.setText("There are no photos here the AI can read.")
            self.ok.setEnabled(False)
            return
        if self.force.isChecked():
            todo = n * max(1, len(qs))
            self.info.setText(f"{todo:,} photo check{'s' if todo != 1 else ''} to do; earlier answers are replaced."
                              if self.kind == "look" else f"{plural(n, 'photo')} to tag again.")
            self.ok.setEnabled(todo > 0)
            return
        if self.kind == "tag":
            todo = len(self.db.ai_todo_tags(ids))
            self.info.setText(f"{n - todo:,} of {n:,} already have AI tags. {plural(todo, 'photo')} to tag." if todo
                              else "All of these already have AI tags. Tick “Tag again” to redo them.")
            self.ok.setEnabled(todo > 0)
            return
        lines = []
        todo_all = 0
        for q in qs:
            sid = self.db.ai_search_id(q)
            todo = len(self.db.ai_todo_search(sid, ids)) if sid is not None else n
            todo_all += todo
            if sid is not None:
                found = self.db.ai_found(sid)
                f = len(found if ids is None else found & set(ids))
                lines.append(f"“{q}”: {n - todo:,} of {n:,} already checked ({f:,} found), {todo:,} to check.")
            else:
                lines.append(f"“{q}”: {plural(todo, 'photo')} to check.")
        if not todo_all:
            lines.append("Everything was checked before: the results show right away.")
            self.ok.setText("Show results")
            self.nothing_to_do = True
        self.info.setText("\n".join(lines))
        self.ok.setEnabled(True)

    def job(self) -> Job:
        return Job(self.kind, photo_ids=self.scope(), queries=self.queries(), force=self.force.isChecked())

    def accept(self):
        if self.ok.isEnabled():
            super().accept()
