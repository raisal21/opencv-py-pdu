import sqlite3
import json
import os, sys

from pathlib import Path, PureWindowsPath

BASE_DIR = Path(os.getenv('LOCALAPPDATA', Path.home())) / 'EyeLog'
BASE_DIR.mkdir(exist_ok=True)

DB_PATH = BASE_DIR / 'eyelog_database.db'
LOG_BASE = BASE_DIR / 'logs'

class DatabaseManager:
    """
    Kelas untuk mengelola koneksi dan operasi database kamera
    """
    def __init__(self, db_path=(DB_PATH)):
        # Pastikan direktori database ada
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.connection = None
        self.ensure_tables()
        self.migrate_if_needed()
        
    def ensure_tables(self):
        """Memastikan tabel yang diperlukan sudah ada di database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Buat tabel kamera jika belum ada
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            ip_address TEXT NOT NULL,
            port INTEGER NOT NULL,
            protocol TEXT DEFAULT 'HTTP',
            username TEXT DEFAULT '',
            password TEXT DEFAULT '',
            stream_path TEXT DEFAULT '',
            url TEXT DEFAULT '',
            resolution_width INTEGER DEFAULT 640,
            resolution_height INTEGER DEFAULT 480,
            roi_points TEXT DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Buat tabel untuk menyimpan versi skema database
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS schema_version (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version INTEGER NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def migrate_if_needed(self):
        """Melakukan migrasi jika skema database perlu diperbarui"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Cek versi skema saat ini
            cursor.execute("SELECT version FROM schema_version ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()
            current_version = result[0] if result else 0
            
            # Definisi migrasi
            if current_version < 1:
                # Migrasi ke versi 1: Tambahkan kolom untuk RTSP
                try:
                    # Cek apakah kolom protocol sudah ada
                    cursor.execute("SELECT protocol FROM cameras LIMIT 1")
                except sqlite3.OperationalError:
                    # Kolom belum ada, tambahkan kolom baru
                    cursor.execute("ALTER TABLE cameras ADD COLUMN protocol TEXT DEFAULT 'HTTP'")
                    cursor.execute("ALTER TABLE cameras ADD COLUMN username TEXT DEFAULT ''")
                    cursor.execute("ALTER TABLE cameras ADD COLUMN password TEXT DEFAULT ''")
                    cursor.execute("ALTER TABLE cameras ADD COLUMN stream_path TEXT DEFAULT ''")
                    cursor.execute("ALTER TABLE cameras ADD COLUMN url TEXT DEFAULT ''")
                    
                    # Update versi skema
                    cursor.execute("INSERT INTO schema_version (version) VALUES (1)")
                    conn.commit()
            
            # Tambahkan migrasi lain jika diperlukan di masa depan
            # if current_version < 2:
            #     # Migrasi ke versi 2
            #     ...
            
        except sqlite3.Error as e:
            logger.error(f"Migration error: {e}")
        finally:
            conn.close()
    
    def add_camera(self, name, ip_address, port=554, protocol="RTSP", username="", password="", stream_path="stream1", url="", resolution=(640, 480), roi_points=None):
        """
        Menambahkan kamera baru ke database
    
        Returns:
            int: ID kamera baru atau None jika gagal
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
    
        try:
            cursor.execute('''
            INSERT INTO cameras (
                name, ip_address, port, protocol, username, password, 
                stream_path, url, resolution_width, resolution_height,
                roi_points
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                name, ip_address, port, protocol, username, password, 
                stream_path, url, resolution[0], resolution[1], roi_points
            ))
        
            conn.commit()
            # Dapatkan ID dari kamera yang baru ditambahkan
            camera_id = cursor.lastrowid
            return camera_id
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return None
        finally:
            conn.close()

    def update_camera(self, camera_id, name, ip_address, port=554, protocol="RTSP", username="", password="", stream_path="stream1", url="", resolution=(640, 480), roi_points=None):
        """
        Memperbarui data kamera yang ada
    
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
    
        try:
            cursor.execute('''
            UPDATE cameras 
            SET name = ?, ip_address = ?, port = ?, protocol = ?, username = ?, 
                password = ?, stream_path = ?, url = ?, resolution_width = ?, resolution_height = ?
            WHERE id = ?
            ''', (
                name, ip_address, port, protocol, username, 
                password, stream_path, url, resolution[0], resolution[1], camera_id,
            ))
        
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return False
        finally:
            conn.close()
    
    def delete_camera(self, camera_id):
        """
        Menghapus kamera dari database
        
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Hapus kamera
            cursor.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
            
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return False
        finally:
            conn.close()
    
    def get_all_cameras(self):
        """
        Mengambil semua kamera dari database
        
        Returns:
            list: Daftar dictionary yang berisi data kamera
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Untuk mengakses kolom dengan nama
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            SELECT id, name, ip_address, port, protocol, username, password,
                   stream_path, url, resolution_width, resolution_height, roi_points
            FROM cameras
            ORDER BY name
            ''')
            
            cameras = []
            for row in cursor.fetchall():
                camera = {
                    'id': row['id'],
                    'name': row['name'],
                    'ip_address': row['ip_address'],
                    'port': row['port'],
                    'protocol': row['protocol'],
                    'username': row['username'],
                    'password': row['password'],
                    'stream_path': row['stream_path'],
                    'url': row['url'],
                    'resolution': (row['resolution_width'], row['resolution_height']),
                    'roi_points': row['roi_points']
                }

                if camera['roi_points']:
                    import json
                    try:
                        # Parse JSON string back to Python list
                        points_data = json.loads(camera['roi_points'])
                        # Convert lists back to tuples if needed
                        camera['roi_points'] = [tuple(point) for point in points_data]
                    except json.JSONDecodeError as e:
                        logger.error(f"Error deserializing ROI points: {e}")
                # Keep as string if deserialization fails
                cameras.append(camera)
            
            return cameras
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return []
        finally:
            conn.close()
    
    def get_camera(self, camera_id):
        """
        Mengambil data kamera berdasarkan ID
        
        Returns:
            dict: Data kamera atau None jika tidak ditemukan
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            SELECT id, name, ip_address, port, protocol, username, password,
                   stream_path, url, resolution_width, resolution_height, roi_points
            FROM cameras
            WHERE id = ?
            ''', (camera_id,))
            
            row = cursor.fetchone()
            if row:
                camera = {
                    'id': row['id'],
                    'name': row['name'],
                    'ip_address': row['ip_address'],
                    'port': row['port'],
                    'protocol': row['protocol'],
                    'username': row['username'],
                    'password': row['password'],
                    'stream_path': row['stream_path'],
                    'url': row['url'],
                    'resolution': (row['resolution_width'], row['resolution_height']),
                    'roi_points': row['roi_points']
                }
                
                if camera['roi_points']:
                    import json
                    try:
                        # Parse JSON string back to Python list
                        points_data = json.loads(camera['roi_points'])
                        # Convert lists back to tuples if needed
                        camera['roi_points'] = [tuple(point) for point in points_data]
                    except json.JSONDecodeError as e:
                        logger.error(f"Error deserializing ROI points: {e}")
                        # Keep as string if deserialization fails
                
                return camera
            return None
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return None
        finally:
            conn.close()
    
    def update_roi_points(self, camera_id, roi_points_json):
        """
        Update only the ROI points for a camera
    
        Args:
            camera_id: ID of the camera to update
            roi_points_json: JSON string of ROI points
        
        Returns:
            bool: True if successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
    
        try:
            cursor.execute('''
            UPDATE cameras 
            SET roi_points = ?
            WHERE id = ?
            ''', (roi_points_json, camera_id))
        
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return False
        finally:
            conn.close()