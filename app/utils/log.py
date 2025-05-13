import logging
import sys
from datetime import datetime

def setup(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO

    # Tambahkan logging ke file juga
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),  # tetap tampil di console
            logging.FileHandler(f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")  # file log
        ]
    )

    logging.getLogger("cv2").setLevel(logging.WARNING)
    logging.getLogger("PySide6").setLevel(logging.WARNING)
