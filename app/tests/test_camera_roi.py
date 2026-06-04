"""Tests for Camera.process_frame_with_roi."""
import numpy as np
import pytest

from ..models.camera import Camera


@pytest.fixture
def dummy_frame():
    return np.full((60, 120, 3), 255, dtype=np.uint8)

def test_process_frame_without_roi_returns_same(dummy_frame):
    cam = Camera()
    out = cam.process_frame_with_roi(dummy_frame)

    assert out is dummy_frame or np.array_equal(out, dummy_frame)

def test_process_frame_with_valid_roi(dummy_frame):
    cam = Camera()

    cam.roi_points = [(0, 0), (100, 0), (100, 50), (0, 50)]

    roi_out = cam.process_frame_with_roi(dummy_frame)

    assert roi_out.shape[:2] == (50, 100)

    assert roi_out.mean() == 255
