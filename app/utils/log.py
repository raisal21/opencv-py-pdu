# log.py
# =========
import logging
import sys

def setup(debug: bool = False) -> None:
    """
    Konfigurasikan logging global.
    Gunakan debug=True (atau jalankan app dgn --debug) untuk level DEBUG.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # Redam kebisingan lib eksternal
    logging.getLogger("cv2").setLevel(logging.WARNING)
    logging.getLogger("PySide6").setLevel(logging.WARNING)
