"""Tests for foreground extraction and contour coverage."""
import cv2 as cv
import numpy as np
import pytest

from ..utils.material_detector import (
    ForegroundExtraction,
    ContourProcessor,
)


@pytest.fixture
def black_frame():
    """Return a 100x100 black baseline frame."""
    return np.zeros((100, 100, 3), dtype=np.uint8)

@pytest.fixture
def half_white_frame():
    """Return a 100x100 frame with a white upper half."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv.rectangle(img, (0, 0), (99, 49), (255, 255, 255), thickness=-1)
    return img

def test_foreground_blank_returns_zero(black_frame):
    fg = ForegroundExtraction(history=5, var_threshold=16, detect_shadows=False,
                              learning_rate=0)

    for _ in range(2):
        res = fg.process_frame(black_frame)

    assert cv.countNonZero(res.binary) == 0

def test_foreground_detects_white_area(half_white_frame):
    fg = ForegroundExtraction(history=1, var_threshold=10, detect_shadows=False,
                              learning_rate=1.0)
    res = fg.process_frame(half_white_frame)
    nonzero = cv.countNonZero(res.binary)
    assert nonzero > 0

def test_contour_processor_coverage_half_mask():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:50, :] = 255

    proc = ContourProcessor(min_contour_area=10,
                             use_convex_hull=True,
                             merge_overlapping=False)
    result = proc.process_mask(mask)

    metrics = result.metrics

    assert metrics['contour_count'] >= 1

    assert 45 <= metrics['contour_coverage_percent'] <= 55
