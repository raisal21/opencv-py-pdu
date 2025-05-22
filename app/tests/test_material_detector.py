"""
Unit‑tests gabungan untuk:
  • material_detector.ForegroundExtraction
  • material_detector.ContourProcessor.process_mask

Menjalankan:
    pytest -q tests/test_material_detector.py
"""

import sys
import os
import cv2 as cv
import numpy as np
import pytest

from utils.material_detector import ForegroundExtraction, ContourProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def black_frame():
    """100×100 utk baseline (semua hitam)."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def half_white_frame():
    """Frame 100×100 setengah putih di atas."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv.rectangle(img, (0, 0), (99, 49), (255, 255, 255), thickness=-1)  # white top‑half
    return img

# ---------------------------------------------------------------------------
# ForegroundExtraction tests
# ---------------------------------------------------------------------------

def test_foreground_blank_returns_zero(black_frame):
    fg = ForegroundExtraction(history=5, var_threshold=16, detect_shadows=False, learning_rate=0.01)
    for _ in range(30):
        res = fg.process_frame(black_frame)
    assert cv.countNonZero(res.binary) == 0


def test_foreground_detects_white_area(half_white_frame):
    fg = ForegroundExtraction(history=1, var_threshold=10, detect_shadows=False,
                              learning_rate=1.0)
    res = fg.process_frame(half_white_frame)
    nonzero = cv.countNonZero(res.binary)
    assert nonzero > 0  # ada foreground terdeteksi

# ---------------------------------------------------------------------------
# ContourProcessor tests
# ---------------------------------------------------------------------------

def test_contour_processor_coverage_half_mask():
    # Buat mask biner 100×100 setengah atas putih
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:50, :] = 255

    proc = ContourProcessor(min_contour_area=10,
                             use_convex_hull=True,
                             merge_overlapping=False)
    result = proc.process_mask(mask)

    metrics = result.metrics
    # kontur ditemukan
    assert metrics['contour_count'] >= 1
    # coverage ~50% dengan toleransi ±5%
    assert 45 <= metrics['contour_coverage_percent'] <= 55
