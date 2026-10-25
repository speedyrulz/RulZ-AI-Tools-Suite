import ctypes
import sys

from PySide6.QtWidgets import QApplication

from captioner import theme
from captioner.main_window import MainWindow


def main():
    try:  # own taskbar group/icon instead of python.exe's
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Rulz.TrainingCaptioner")
    except Exception:
        pass
    app = QApplication(sys.argv)
    app.setApplicationName("Rulz Training Captioner")
    theme.apply(app)
    app.setWindowIcon(theme.icon())
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
