import sqlite3
import json
import os
import logging
from pathlib import Path

# Inisialisasi logger
logger = logging.getLogger(__name__)

# Konfigurasi path database
BASE_DIR = Path(os.getenv('LOCALAPPDATA', Path.home())) / 'EyeLog'
BASE_DIR.mkdir(exist_ok=True)
DB_PATH = BASE_DIR / 'eyelog_database.db'

class DatabaseManager:
    """
    Kelas untuk mengelola operasi database.
    CATATAN: Kelas ini sekarang TIDAK mengelola koneksi. Koneksi harus
    disediakan oleh pemanggil (misalnya, DBWorker).
    """

    def __init__(self, db_path=DB_PATH):
        """Hanya menyimpan path, tidak membuat koneksi."""
        self.db_path = db_path
    
    @staticmethod
    def get_connection(db_path=DB_PATH):
        """Membuat dan mengembalikan koneksi database baru."""
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path, timeout=10) # Timeout untuk mencegah lock
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def ensure_tables(conn):
        """Memastikan tabel yang diperlukan sudah ada menggunakan koneksi yang diberikan."""
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, ip_address TEXT NOT NULL,
            port INTEGER NOT NULL, protocol TEXT DEFAULT 'HTTP', username TEXT DEFAULT '',
            password TEXT DEFAULT '', stream_path TEXT DEFAULT '', url TEXT DEFAULT '',
            resolution_width INTEGER DEFAULT 640, resolution_height INTEGER DEFAULT 480,
            roi_points TEXT DEFAULT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS schema_version (
            id INTEGER PRIMARY KEY AUTOINCREMENT, version INTEGER NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        conn.commit()

    # --- Metode Operasi CRUD ---
    # Setiap metode sekarang menerima 'conn' sebagai argumen pertama

    def add_camera(self, conn, name, ip_address, port=554, protocol="RTSP", username="", password="", stream_path="stream1", url="", resolution=(640, 480), roi_points=None):
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO cameras (name, ip_address, port, protocol, username, password, stream_path, url, resolution_width, resolution_height, roi_points)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (name, ip_address, port, protocol, username, password, stream_path, url, resolution[0], resolution[1], roi_points))
        return cursor.lastrowid

    def update_camera(self, conn, camera_id, name, ip_address, port=554, protocol="RTSP", username="", password="", stream_path="stream1", url="", resolution=(640, 480)):
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE cameras 
            SET name = ?, ip_address = ?, port = ?, protocol = ?, username = ?, 
                password = ?, stream_path = ?, url = ?, resolution_width = ?, resolution_height = ?
            WHERE id = ?
        ''', (name, ip_address, port, protocol, username, password, stream_path, url, resolution[0], resolution[1], camera_id))
        return cursor.rowcount > 0

    def delete_camera(self, conn, camera_id):
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
        return cursor.rowcount > 0

    def get_all_cameras(self, conn):
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM cameras ORDER BY name')
        rows = cursor.fetchall()
        cameras = []
        for row in rows:
            camera = dict(row)
            if camera.get('roi_points'):
                try:
                    camera['roi_points'] = [tuple(p) for p in json.loads(camera['roi_points'])]
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Could not parse ROI points for camera ID {camera.get('id')}")
                    camera['roi_points'] = None # Set ke None jika gagal parse
            cameras.append(camera)
        return cameras

    def get_camera(self, conn, camera_id):
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,))
        row = cursor.fetchone()
        if not row:
            return None
        camera = dict(row)
        if camera.get('roi_points'):
            try:
                camera['roi_points'] = [tuple(p) for p in json.loads(camera['roi_points'])]
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Could not parse ROI points for camera ID {camera.get('id')}")
                camera['roi_points'] = None
        return camera

    def update_roi_points(self, conn, camera_id, roi_points_json):
        cursor = conn.cursor()
        cursor.execute("UPDATE cameras SET roi_points = ? WHERE id = ?", (roi_points_json, camera_id))
        return cursor.rowcount > 0