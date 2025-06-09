from PySide6.QtWidgets import QSplashScreen, QApplication
from PySide6.QtGui import QPixmap, QColor
from PySide6.QtCore import Qt, QSize
from resources import resource_path


def SplashScreen() -> QSplashScreen:
    """
    Splash screen dengan ukuran tetap seperti QGIS (500x300), gambar auto-scale.
    """

    image_path = resource_path("assets/images/EyelogSplashScreen.png")
    pixmap = QPixmap(image_path)

    if pixmap.isNull():
        raise FileNotFoundError(f"Splash image tidak ditemukan: {image_path}")

    # Scale gambar agar pas ke ukuran splash
    splash_size = QSize(700, 500)
    scaled_pixmap = pixmap.scaled(
        splash_size,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation
    )

    splash = QSplashScreen(scaled_pixmap)
    splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    splash.setEnabled(False)
    splash.setFixedSize(splash_size)

    splash.showMessage(
        "Loading EyeLog system...",
        alignment=Qt.AlignBottom | Qt.AlignCenter,
        color=QColor("#ffffff")
    )

    # Pusatkan splash di layar utama
    screen = QApplication.primaryScreen()
    if screen:
        screen_rect = screen.availableGeometry()
        splash_rect = splash.frameGeometry()
        x = screen_rect.center().x() - splash_rect.width() // 2
        y = screen_rect.center().y() - splash_rect.height() // 2
        splash.move(x, y)

    return splash