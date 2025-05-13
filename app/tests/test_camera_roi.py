"""
Unit‑test: Camera.process_frame_with_roi
Memastikan transform ROI bekerja & kasus tanpa ROI aman.
 Jalankan:
    pytest -q tests/test_camera_roi.py
"""

import numpy as np
import pytest

from camera import Camera

# ---------------------------------------------------------------------------
# FIXTURE – frame dummy 60×120 putih (BGR)
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_frame():
    return np.full((60, 120, 3), 255, dtype=np.uint8)  # white image

# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------

def test_process_frame_without_roi_returns_same(dummy_frame):
    cam = Camera()
    out = cam.process_frame_with_roi(dummy_frame)
    # Harus kembali frame yg sama (tidak diubah)
    assert out is dummy_frame or np.array_equal(out, dummy_frame)


def test_process_frame_with_valid_roi(dummy_frame):
    cam = Camera()
    # ROI = persegi panjang 100×50 px di sudut kiri‑atas
    cam.roi_points = [(0, 0), (100, 0), (100, 50), (0, 50)]

    roi_out = cam.process_frame_with_roi(dummy_frame)

    # Ukuran output harus (50, 100)
    assert roi_out.shape[:2] == (50, 100)
    # Karena frame asli putih, hasil ROI juga putih
    assert roi_out.mean() == 255
