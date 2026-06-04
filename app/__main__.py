import sys
import platform
import os
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui import QFontDatabase, QFont
from PySide6.QtCore import QThreadPool, QTimer

from .resources import resource_path
from .utils.log import setup as setup_logging, setup_file_logging


def _load_fonts() -> None:
    """Load bundled Inter fonts when they are available."""
    fonts_dir = resource_path("assets/fonts")
    os.makedirs(fonts_dir, exist_ok=True)

    loaded = False
    for weight in ["Regular", "Medium", "SemiBold", "Bold", "ExtraBold", "Light"]:
        fp = os.path.join(fonts_dir, f"Inter-{weight}.ttf")
        if os.path.exists(fp):
            if QFontDatabase.addApplicationFont(fp) >= 0:
                loaded = True

    if loaded:
        fam = next((f for f in QFontDatabase.families() if "Inter" in f), None)
        if fam:
            app = QApplication.instance()
            default_font = QFont(fam)
            default_font.setStyleStrategy(QFont.PreferAntialias)
            app.setFont(default_font)

def main() -> int:
    """Start the EyeLog desktop application."""
    debug_mode = "--debug" in sys.argv
    setup_logging(debug=debug_mode)

    if platform.system() == "Windows":
        os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    elif platform.system() == "Darwin":
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    app = QApplication(sys.argv)

    from .splash_screen import SplashScreen
    splash = SplashScreen()
    splash.show()
    app.processEvents()

    if platform.system() == "Windows":
        import multiprocessing
        QThreadPool.globalInstance().setMaxThreadCount(
            max(2, multiprocessing.cpu_count() // 2)
        )
    elif platform.system() == "Darwin":
        QThreadPool.globalInstance().setMaxThreadCount(2)

    try:
        setup_file_logging(debug=debug_mode)
    except Exception as e:
        print(f"[WARN] file logging not active: {e}")

    _load_fonts()

    from .main_window import MainWindow
    win = MainWindow()

    def _show_main():
        win.show()
        splash.finish(win)

    QTimer.singleShot(2000, _show_main)
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
