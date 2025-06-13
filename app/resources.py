"""
resources.py
Utility satu‑satunya untuk menghasilkan path absolut ke asset
yang kompatibel dengan PyInstaller (sys._MEIPASS) & mode dev.
"""
import sys
from pathlib import Path
import os

def resource_path(relative: str) -> str:
    """Kembalikan path absolut ke asset.

    Fungsi ini mendukung mode pengembangan maupun saat aplikasi telah
    dikompilasi dengan PyInstaller atau Nuitka. Saat ``sys._MEIPASS`` tidak
    tersedia namun ``sys.frozen`` ada (mis. Nuitka ``--standalone``), maka
    ``sys.executable`` dipakai sebagai dasar path distribusi.
    """

    try:
        # Logika untuk aplikasi yang sudah dikompilasi (PyInstaller, Nuitka)
        # di mana sys._MEIPASS atau sys.executable menunjuk ke lokasi sementara/dist
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            # PyInstaller membuat folder temp dan menyimpan path di _MEIPASS
            base_path = Path(sys._MEIPASS)
        elif getattr(sys, 'frozen', False):
            # Nuitka & PyInstaller mode --onefile di Windows.
            # Basisnya adalah direktori tempat file .exe berada.
            base_path = Path(sys.executable).resolve().parent
        else:
            # Mode Pengembangan (menjalankan dari source code)
            # Asumsi: file ini berada satu level di dalam proyek (misal: my_project/app/resources.py)
            # maka kita naik satu level untuk mendapatkan akar proyek.
            base_path = Path(__file__).resolve().parent.parent
    except Exception:
        # Fallback ke direktori kerja jika ada masalah
        base_path = Path.cwd()

    return os.fspath(Path(base_path) / relative)
