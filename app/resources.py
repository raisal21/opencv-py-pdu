"""
resources.py
Utility satu‑satunya untuk menghasilkan path absolut ke asset
yang kompatibel dengan PyInstaller (sys._MEIPASS) & mode dev.
"""
import sys
from pathlib import Path
import os

# direktori project (saat dev) = folder tempat file ini berada
_DEV_BASE = Path(__file__).resolve().parent

def resource_path(relative: str) -> str:
    """
    Param  relative : str  -> mis. 'assets/icons/plus.png'
    Return          : str  -> path absolut siap dipakai Qt
    """
    base_path = getattr(sys, "_MEIPASS", _DEV_BASE)  # _MEIPASS ada hanya saat frozen
    return os.fspath(Path(base_path) / relative)
