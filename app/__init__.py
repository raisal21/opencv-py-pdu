import sys
import platform
import os
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui import QFontDatabase, QFont
from PySide6.QtCore import QThreadPool

# Platform specific settings BEFORE any imports
if platform.system() == "Windows":
    # Windows optimizations
    os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # Disable MSMF
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    
    # Set thread pool size based on CPU cores
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    # Defer thread pool configuration until after QApplication
    
elif platform.system() == "Darwin":  # macOS
    # macOS specific fixes
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Import after environment setup
from resources import resource_path
from utils.log import setup as setup_logging, setup_file_logging


def load_fonts():
    """Load the Inter font family into the application"""
    # Path ke direktori font relatif terhadap lokasi script
    fonts_dir = resource_path("assets/fonts")
    
    # Periksa apakah direktori ada
    if not os.path.exists(fonts_dir):
        if not os.path.exists(fonts_dir) and not getattr(sys, "_MEIPASS", None):
            os.makedirs(fonts_dir, exist_ok=True)
        print(f"Created fonts directory: {fonts_dir}")
        print("Please place Inter font files in this directory")
        return False
    
    # Cari dan muat semua file font Inter
    font_loaded = False
    for weight in ["Regular", "Medium", "SemiBold", "Bold", "ExtraBold", "Light"]:
        font_path = os.path.join(fonts_dir, f"Inter-{weight}.ttf")
        
        if os.path.exists(font_path):
            font_id = QFontDatabase.addApplicationFont(font_path)
            if font_id >= 0:
                font_loaded = True
            else:
                print(f"Failed to load: Inter-{weight}")
    
    if not font_loaded:
        print("No Inter fonts found. Please download Inter fonts from https://fonts.google.com/specimen/Inter")
        print(f"and place them in {fonts_dir}")
        return False
    
    return True


if __name__ == "__main__":
    # Setup basic console logging first
    debug_mode = "--debug" in sys.argv
    setup_logging(debug=debug_mode)
    
    # Create the application FIRST
    app = QApplication(sys.argv)
    
    # NOW configure thread pool and other Qt-dependent settings
    if platform.system() == "Windows":
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        QThreadPool.globalInstance().setMaxThreadCount(max(2, cpu_count // 2))
    elif platform.system() == "Darwin":
        # Limit global thread pool for macOS
        QThreadPool.globalInstance().setMaxThreadCount(2)
        # Import cv2 AFTER QApplication
        try:
            import cv2 as cv
            cv.setNumThreads(1)
        except ImportError:
            pass
    
    # Setup file logging AFTER QApplication is created
    try:
        setup_file_logging(debug=debug_mode)
    except Exception as e:
        print(f"Warning: Could not setup file logging: {e}")
    
    # Load and set Inter as the default font
    if load_fonts():
        # Get font family name
        font_families = QFontDatabase.families()
        inter_family = next((f for f in font_families if "Inter" in f), None)
        
        if inter_family:
            # Buat font dan terapkan ke seluruh aplikasi
            default_font = QFont(inter_family)
            default_font.setStyleStrategy(QFont.PreferAntialias)  # Untuk rendering yang lebih halus
            
            # Tetapkan sebagai font default
            app.setFont(default_font)
        else:
            print("Inter font family not found after loading")
    
    # Import dan jalankan aplikasi utama setelah semua setup selesai
    try:
        from main_window import MainWindow
        
        window = MainWindow()
        window.show()
        
        # Start the event loop
        sys.exit(app.exec())
        
    except Exception as e:
        import traceback
        logging.error(f"Failed to start application: {e}")
        traceback.print_exc()
        sys.exit(1)