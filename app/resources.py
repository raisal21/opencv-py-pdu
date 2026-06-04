"""Resolve asset paths in development and packaged builds."""
import sys
from pathlib import Path
import os

def resource_path(relative: str) -> str:
    """Return an absolute path for a bundled asset."""
    try:
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            base_path = Path(sys._MEIPASS)
        elif getattr(sys, 'frozen', False):
            base_path = Path(sys.executable).resolve().parent
        else:
            base_path = Path(__file__).resolve().parent.parent
    except Exception:
        base_path = Path.cwd()

    return os.fspath(Path(base_path) / relative)
