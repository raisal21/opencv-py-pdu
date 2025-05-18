# log.py
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

def setup(debug: bool = False) -> None:
    """
    Konfigurasikan logging global.
    - Menyimpan ke logs/app.log
    - Debug = True: juga tampilkan di stdout
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "app.log"
    level = logging.DEBUG if debug else logging.INFO

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handlers = []

    # File handler (rotating)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=1_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)

    # Stream handler (jika debug mode)
    if debug:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

    logging.basicConfig(level=level, handlers=handlers)

    # Redam noise dari lib eksternal
    logging.getLogger("cv2").setLevel(logging.WARNING)
    logging.getLogger("PySide6").setLevel(logging.WARNING)