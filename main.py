import ctypes
import sys

from PySide6.QtWidgets import QApplication

from captioner import config, theme
from captioner.look_dialog import ChooseLookDialog
from captioner.main_window import MainWindow


def main():
    try:  # own taskbar group/icon instead of python.exe's
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Rulz.TrainingCaptioner")
    except Exception:
        pass
    app = QApplication(sys.argv)
    app.setApplicationName("Rulz Training Captioner")
    app.setWindowIcon(theme.icon())
    s = config.load_settings()
    if s.get("theme") in (theme.DARK, theme.LIGHT):
        theme.apply(app, s["theme"])
    else:  # first launch: ask once, remember the answer
        theme.apply(app, theme.DARK)
        dlg = ChooseLookDialog()
        dlg.exec()
        s["theme"] = dlg.result_theme()
        config.save_settings(s)
        theme.apply(app, s["theme"])
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
