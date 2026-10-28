import os
import shutil
import sys

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
                               QRadioButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from .config import detect_gpus


def find_llama_server():
    here = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
    for d in (os.path.join(here, "llama.cpp"), os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "llama.cpp")):
        p = os.path.join(d, "llama-server.exe")
        if os.path.isfile(p):
            return p
    return shutil.which("llama-server") or ""


class PathRow(QWidget):
    def __init__(self, text, filt, folder=False):
        super().__init__()
        self.edit = QLineEdit(text)
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: self._browse(filt))
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.edit, 1)
        lay.addWidget(b)

    def _browse(self, filt):
        p, _ = QFileDialog.getOpenFileName(self, "Select file", self.edit.text(), filt)
        if p:
            self.edit.setText(os.path.normpath(p))


class SettingsDialog(QDialog):
    def __init__(self, s, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(640, 560)
        self.s = dict(s)
        self.gpus = detect_gpus()
        tabs = QTabWidget()

        # ---- backend
        w = QWidget()
        f = QFormLayout(w)
        self.r_managed = QRadioButton("Managed llama.cpp (app starts llama-server for you)")
        self.r_external = QRadioButton("External OpenAI-compatible server(s) (LM Studio, Ollama, vLLM, running llama-server…)")
        self.r_managed.setChecked(s["backend"] == "managed")
        self.r_external.setChecked(s["backend"] != "managed")
        f.addRow(self.r_managed)
        self.server = PathRow(s["llama_server"] or find_llama_server(), "llama-server (llama-server.exe);;All files (*)")
        self.model = PathRow(s["model"], "GGUF models (*.gguf);;All files (*)")
        self.mmproj = PathRow(s["mmproj"], "GGUF projector (*.gguf);;All files (*)")
        f.addRow("llama-server.exe", self.server)
        f.addRow("Model (.gguf)", self.model)
        f.addRow("Vision projector (mmproj)", self.mmproj)
        self.ctx = QSpinBox(); self.ctx.setRange(512, 262144); self.ctx.setSingleStep(512); self.ctx.setValue(s["ctx_size"])
        self.ngl = QSpinBox(); self.ngl.setRange(0, 999); self.ngl.setValue(s["ngl"])
        self.port = QSpinBox(); self.port.setRange(1024, 65000); self.port.setValue(s["base_port"])
        self.extra = QLineEdit(s["extra_args"]); self.extra.setPlaceholderText("e.g. --flash-attn on --no-mmproj-offload")
        f.addRow("Context per request", self.ctx)
        f.addRow("GPU layers (-ngl)", self.ngl)
        f.addRow("First port", self.port)
        f.addRow("Extra llama-server args", self.extra)
        f.addRow(QLabel(""))
        f.addRow(self.r_external)
        self.urls = QPlainTextEdit(s["external_urls"]); self.urls.setFixedHeight(70)
        self.urls.setPlaceholderText("One base URL per line, e.g. http://127.0.0.1:1234/v1\nList several to caption in parallel.")
        self.ext_model = QLineEdit(s["external_model"]); self.ext_model.setPlaceholderText("model name (required by LM Studio/Ollama)")
        self.key = QLineEdit(s["api_key"]); self.key.setEchoMode(QLineEdit.Password)
        f.addRow("Endpoint URLs", self.urls)
        f.addRow("Model name", self.ext_model)
        f.addRow("API key (optional)", self.key)
        tabs.addTab(w, "Backend")

        # ---- GPUs
        w = QWidget()
        v = QVBoxLayout(w)
        if self.gpus:
            v.addWidget(QLabel("GPUs to use (managed backend):"))
        else:
            v.addWidget(QLabel("nvidia-smi not found — no GPU selection; llama-server will choose its own devices."))
        self.gpu_checks = []
        for idx, name, mem in self.gpus:
            c = QCheckBox(f"GPU {idx}: {name}  ({mem / 1024:.0f} GB)")
            c.setChecked(idx in s["gpus"] or (not s["gpus"]))
            self.gpu_checks.append((idx, c))
            v.addWidget(c)
        self.m_parallel = QRadioButton("Parallel — one model copy per GPU, images are shared out between them (fastest; model must fit on one card)")
        self.m_split = QRadioButton("Split — one model spread across the selected GPUs (for models too big for one card)")
        self.m_parallel.setChecked(s["gpu_mode"] != "split")
        self.m_split.setChecked(s["gpu_mode"] == "split")
        v.addSpacing(8)
        v.addWidget(self.m_parallel)
        v.addWidget(self.m_split)
        self.tsplit = QLineEdit(s["tensor_split"])
        self.tsplit.setPlaceholderText("blank = automatic; or ratios per GPU, e.g. 16,16 or 3,2")
        f2 = QFormLayout()
        f2.addRow("Tensor split (split mode)", self.tsplit)
        self.slots = QSpinBox(); self.slots.setRange(1, 16); self.slots.setValue(s["slots_per_server"])
        f2.addRow("Concurrent requests per server", self.slots)
        v.addLayout(f2)
        v.addWidget(QLabel("Extra concurrent requests use more VRAM (context is multiplied) but can raise throughput.\n"
                           "For the external backend this is the number of simultaneous requests per URL."))
        v.addStretch(1)
        tabs.addTab(w, "GPUs")

        # ---- generation
        w = QWidget()
        f = QFormLayout(w)
        self.temp = QDoubleSpinBox(); self.temp.setRange(0, 2); self.temp.setSingleStep(0.1); self.temp.setValue(s["temperature"])
        self.maxtok = QSpinBox(); self.maxtok.setRange(16, 8192); self.maxtok.setValue(s["max_tokens"])
        self.side = QSpinBox(); self.side.setRange(256, 4096); self.side.setSingleStep(128); self.side.setValue(s["max_image_side"])
        self.single = QCheckBox("Collapse captions to a single line"); self.single.setChecked(s["single_line"])
        f.addRow("Temperature", self.temp)
        f.addRow("Max new tokens", self.maxtok)
        f.addRow("Resize longest image side to (px)", self.side)
        f.addRow(self.single)
        self.nothink = QCheckBox("Disable thinking / reasoning (recommended; thinking models are much slower)")
        self.nothink.setChecked(s["disable_thinking"])
        f.addRow(self.nothink)
        self.budget = QSpinBox()
        self.budget.setRange(-1, 32768)
        self.budget.setSingleStep(256)
        self.budget.setSpecialValueText("Unlimited")
        self.budget.setSuffix(" tokens")
        self.budget.setValue(s["thinking_budget"])
        self.budget.setToolTip("Maximum tokens the model may spend thinking before it must answer.\n"
                               "Added on top of 'Max new tokens'. Enforced by the managed llama-server.")
        self.budget.setEnabled(not s["disable_thinking"])
        self.nothink.toggled.connect(lambda on: self.budget.setEnabled(not on))
        f.addRow("Thinking limit", self.budget)
        tabs.addTab(w, "Generation")

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(tabs)
        lay.addWidget(bb)

    def result_settings(self):
        s = dict(self.s)
        s.update(
            backend="managed" if self.r_managed.isChecked() else "external",
            llama_server=self.server.edit.text().strip(), model=self.model.edit.text().strip(),
            mmproj=self.mmproj.edit.text().strip(), ctx_size=self.ctx.value(), ngl=self.ngl.value(),
            base_port=self.port.value(), extra_args=self.extra.text(),
            external_urls=self.urls.toPlainText(), external_model=self.ext_model.text().strip(),
            api_key=self.key.text().strip(),
            gpus=[i for i, c in self.gpu_checks if c.isChecked()],
            gpu_mode="split" if self.m_split.isChecked() else "parallel",
            tensor_split=self.tsplit.text().strip(), slots_per_server=self.slots.value(),
            temperature=self.temp.value(), max_tokens=self.maxtok.value(),
            max_image_side=self.side.value(), single_line=self.single.isChecked(),
            disable_thinking=self.nothink.isChecked(), thinking_budget=self.budget.value(),
        )
        return s
