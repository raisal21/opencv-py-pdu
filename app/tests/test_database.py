""" 
Unit‑test terfokus pada operasi utama DatabaseManager:
  • add_camera
  • get_camera  / get_all_cameras
  • update_camera
  • delete_camera

Test memakai SQLite file sementara (`tmp_path`) sehingga:
  – Tidak mengotori database produksi.
  – Performanya cepat & repeatable.

Jalankan dengan:
    pytest -q tests/test_database.py
"""

import os
import sqlite3
import pytest

# Import modul yang diuji
from models.database import DatabaseManager

# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """Buat DatabaseManager dan koneksi SQLite sementara."""
    db_file = tmp_path / "unit_db.sqlite"
    manager = DatabaseManager(db_path=os.fspath(db_file))
    conn = DatabaseManager.get_connection(db_path=os.fspath(db_file))
    DatabaseManager.ensure_tables(conn)
    yield manager, conn
    conn.close()

# ---------------------------------------------------------------------------
# TEST‑CASE PRIORITAS 1
# ---------------------------------------------------------------------------

def test_add_and_get_camera(db):
    manager, conn = db
    cam_id = manager.add_camera(
        conn,
        name="Cam1",
        ip_address="192.168.0.1",
        port=554,
        protocol="RTSP",
        username="admin",
        password="pass",
        stream_path="stream1",
        url="rtsp://admin:pass@192.168.0.1:554/stream1",
    )
    conn.commit()
    assert isinstance(cam_id, int) and cam_id > 0

    cam = manager.get_camera(conn, cam_id)
    assert cam is not None
    assert cam["name"] == "Cam1"
    assert cam["ip_address"] == "192.168.0.1"
    assert cam["port"] == 554


def test_update_camera(db):
    manager, conn = db
    cam_id = manager.add_camera(
        conn,
        "CamX",
        "192.168.0.2",
        8554,
        protocol="RTSP",
        stream_path="video",
        url="rtsp://192.168.0.2:8554/video",
    )
    conn.commit()

    ok = manager.update_camera(
        conn,
        camera_id=cam_id,
        name="CamX‑Renamed",
        ip_address="192.168.0.2",
        port=8554,
        protocol="RTSP",
        username="",
        password="",
        stream_path="video",
        url="rtsp://192.168.0.2:8554/video",
    )
    assert ok is True

    conn.commit()
    cam = manager.get_camera(conn, cam_id)
    assert cam["name"] == "CamX‑Renamed"


def test_get_all_cameras(db):
    manager, conn = db
    ids = [
        manager.add_camera(conn,f"Cam{i}", f"10.0.0.{i}", 554, url=f"rtsp://10.0.0.{i}:554/stream")
        for i in range(3)
    ]
    conn.commit()
    cams = manager.get_all_cameras(conn)
    retrieved_ids = {c["id"] for c in cams}
    assert set(ids) <= retrieved_ids  # semua id yang ditambahkan ada di hasil


def test_delete_camera(db):
    manager, conn = db
    cam_id = manager.add_camera(conn,"DelCam", "10.0.0.99", 554, url="rtsp://10.0.0.99/stream")
    conn.commit()
    ok = manager.delete_camera(conn, cam_id)
    assert ok is True
    assert manager.get_camera(conn, cam_id) is None

# ---------------------------------------------------------------------------
# TEST SCHÉMA /MIGRASI
# ---------------------------------------------------------------------------

def test_schema_tables_created(tmp_path):
    """Pastikan tabel 'cameras' & 'schema_version' tercipta pada DB baru."""
    db_file = tmp_path / "schema.sqlite"
    conn = DatabaseManager.get_connection(db_path=os.fspath(db_file))
    DatabaseManager.ensure_tables(conn)

    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cameras'")
    assert cur.fetchone() is not None
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
    assert cur.fetchone() is not None
    conn.close()
