"""
resources.py
Utility satu‑satunya untuk menghasilkan path absolut ke asset
yang kompatibel dengan PyInstaller (sys._MEIPASS) & mode dev.
"""
import sys
from pathlib import Path
import os

# Direktori project saat development
_DEV_BASE = Path(__file__).resolve().parent

def resource_path(relative: str) -> str:
    """Kembalikan path absolut ke asset.

    Fungsi ini mendukung mode pengembangan maupun saat aplikasi telah
    dikompilasi dengan PyInstaller atau Nuitka. Saat ``sys._MEIPASS`` tidak
    tersedia namun ``sys.frozen`` ada (mis. Nuitka ``--standalone``), maka
    ``sys.executable`` dipakai sebagai dasar path distribusi.
    """

    base_path = getattr(sys, "_MEIPASS", None)

    if base_path is None:
        if getattr(sys, "frozen", False):
            # Nuitka standalone build
            base_path = Path(sys.executable).resolve().parent
        else:
            base_path = _DEV_BASE

    return os.fspath(Path(base_path) / relative)
