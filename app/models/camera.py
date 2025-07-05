import cv2 as cv
import numpy as np
import time
import json
import psutil
import logging
from PySide6.QtCore import Qt, QThread, Signal, QMutex, QMutexLocker, QSize
from PySide6.QtGui import QImage, QPixmap

logger = logging.getLogger(__name__) 

FPS_PREVIEW = 5 
FPS_DETAIL  = 15
FPS_IDLE    = 3
CPU_HIGH_THRESHOLD = 80


class Camera:
    """
    Class untuk mengelola koneksi dan properti kamera IP dengan RTSP.
    Menangani koneksi, streaming, dan status kamera.
    """
    def __init__(self, camera_id=None, name="Untitled Camera", ip_address="0.0.0.0", port=554,
                 username="", password="", stream_path="stream1", custom_url=None):
        """
        Inisialisasi objek kamera baru.
        
        Args:
            camera_id: ID unik untuk kamera (None jika kamera baru)
            name: Nama tampilan kamera
            ip_address: Alamat IP kamera
            port: Port RTSP (default 554)
            username: Username untuk autentikasi RTSP
            password: Password untuk autentikasi RTSP
            stream_path: Path stream RTSP (misalnya "stream1", "video")
            custom_url: URL custom lengkap untuk override URL otomatis
        """
        # Properti dasar
        self.id = camera_id
        self.name = name
        self.ip_address = ip_address
        self.port = port
        # Properti koneksi RTSP
        self.username = username
        self.password = password
        self.stream_path = stream_path
        self.custom_url = custom_url
        self.is_static = False
        self.video_path = video_path
        # Status dan resource kamera
        self.connection_status = False
        self.capture = None
        self.last_frame = None
        self.last_error = None
        self.last_reconnect_attempt = 0
        self.reconnect_interval = 10  # dalam detik
        # Properti tambahan
        self.resolution = (640, 480)
        self.roi_points = None
        self.roi_image = None
        self.is_preview_mode: bool = False

        self._roi_M:      np.ndarray | None = None 
        self._roi_size:   tuple[int, int] | None = None   
        self._roi_mutex = QMutex()
        # Thread dan mutex
        self.mutex = QMutex()
        self.thread = None

    def build_stream_url(self):
        """
        Membuat URL lengkap untuk stream kamera RTSP.
        Prioritaskan custom_url jika tersedia.
    
        Returns:
        String URL untuk koneksi ke kamera
        """
        if self.protocol == 'FILE' and self.custom_url:
            return self.custom_url
            
        # Jika custom URL tersedia, gunakan itu
        if self.custom_url and self.custom_url.strip():
            return self.custom_url.strip()

        # Format autentikasi untuk RTSP
        auth_part = ""
        if self.username and self.password:
            auth_part = f"{self.username}:{self.password}@"
    
        # Pastikan stream_path ada dan tidak memiliki slash awal
        stream_path = self.stream_path if self.stream_path else "stream1"
        if stream_path.startswith('/'):
            stream_path = stream_path[1:]
        
        # Bangun URL RTSP: rtsp://[username:password@]ip:port/stream
        return f"rtsp://{auth_part}{self.ip_address}:{self.port}/{stream_path}"
    
    def connect(self):
        """
        Mencoba membuat koneksi ke kamera.
        
        Returns:
            bool: True jika koneksi berhasil, False jika gagal
        """
        try:
            # Tutup koneksi yang ada jika ada
            if self.capture is not None and self.capture.isOpened():
                self.capture.release()
            
            # Coba koneksi baru
            url = self.build_stream_url()
            # Jika statis, tidak perlu timeout
            if self.is_static:
                self.capture = cv.VideoCapture(url)
            else:
                open_timeout_ms = 3000
                params = [cv.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout_ms]

                try:
                    self.capture = cv.VideoCapture(url, cv.CAP_FFMPEG, params)
                    if hasattr(cv, "CAP_PROP_THREAD_COUNT"):
                        self.capture.set(cv.CAP_PROP_THREAD_COUNT, 1)
                except (cv.error, AttributeError):
                    logging.warning("CAP_PROP_OPEN_TIMEOUT_MSEC tidak tersedia — memakai fallback tanpa timeout")
                    self.capture = cv.VideoCapture(url, cv.CAP_FFMPEG)
            
            # Verifikasi koneksi
            if self.capture.isOpened():
                # Atur resolusi jika perlu
                self.capture.set(cv.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.capture.set(cv.CAP_PROP_BUFFERSIZE, 1)
                
                self.connection_status = True
                self.last_error = None
                return True
            else:
                if self.capture:
                    self.capture.release()
                    self.capture = None
                self.connection_status = False
                self.last_error = "Gagal membuka koneksi kamera"
                logger.error(f"Koneksi kamera {self.name} gagal: {self.last_error}")
                return False
                
        except Exception as e:
            if self.capture:
                try:
                    self.capture.release()
                except:
                    pass
                self.capture = None
            self.connection_status = False
            self.last_error = str(e)
            logger.error(f"Error koneksi: {self.last_error}")
            return False
    
    def disconnect(self):
        """
        Memutuskan koneksi kamera dan membersihkan resource.
        """
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.deleteLater()
            self.thread = None
        
        with QMutexLocker(self.mutex):
            if self.capture is not None:
                self.capture.release()
                self.capture = None
            
            self.connection_status = False
            self.last_frame = None
    
    def start_stream(self):
        """
        Memulai thread streaming dari kamera.
        
        Returns:
            bool: True jika streaming berhasil dimulai
        """
        if not self.connection_status:
            success = self.connect()
            if not success:
                return False
        
        # Buat dan mulai thread jika belum ada
        if self.thread is None or not self.thread.isRunning():
            target_fps = 10 if self.is_preview_mode else 15
            self.thread = CameraThread(self, target_fps=target_fps)
            self.thread.start()
            
        return True
    
    def get_last_frame(self):
        """
        Mengambil frame terakhir yang diambil dari kamera.
        Thread-safe dengan mutex.
        
        Returns:
            numpy.ndarray: Frame gambar terakhir atau None
        """
        with QMutexLocker(self.mutex):
            if self.last_frame is not None:
                return self.last_frame.copy()
        return None
    
    def set_last_frame(self, frame):
        """
        Mengatur frame terakhir. Digunakan oleh thread.
        Thread-safe dengan mutex.
        
        Args:
            frame: numpy.ndarray frame gambar
        """
        with QMutexLocker(self.mutex):
            self.last_frame = frame
    
    def try_reconnect(self):
        """
        Mencoba menghubungkan kembali ke kamera dengan interval pembatasan.
        
        Returns:
            bool: True jika upaya reconnect berhasil dimulai
        """
        current_time = time.time()
        
        # Periksa apakah sudah waktunya untuk mencoba koneksi ulang
        if current_time - self.last_reconnect_attempt > self.reconnect_interval:
            self.last_reconnect_attempt = current_time
            return self.connect()
        
        return False
    
    
    def _prepare_roi_transform(self, frame: np.ndarray) -> None:
        """
        Hitung matriks perspektif & ukuran output SATU KALI setelah ROI di‑set.

        Args
        ----
        frame : np.ndarray
            Frame sumber (hanya dipakai untuk ukuran gambar).
        """
        if self.roi_points is None or len(self.roi_points) < 4:
            return  # Tidak cukup titik

        # Sumber & tujuan (sama persis dengan ROISelector.extract_roi)
        src_pts = np.array(self.roi_points, dtype=np.float32)

        # Estimasi lebar–tinggi ortonormal
        w1, w2 = np.linalg.norm(src_pts[0] - src_pts[1]), np.linalg.norm(src_pts[2] - src_pts[3])
        h1, h2 = np.linalg.norm(src_pts[1] - src_pts[2]), np.linalg.norm(src_pts[3] - src_pts[0])
        width, height = int(max(w1, w2)), int(max(h1, h2))

        dst_pts = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32
        )

        # Simpan ke cache
        self._roi_M = cv.getPerspectiveTransform(src_pts, dst_pts)
        self._roi_size = (width, height)

    def process_frame_with_roi(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or self.roi_points is None or len(self.roi_points) < 4:
            return frame
        
        with QMutexLocker(self._roi_mutex):  # <-- TAMBAH
            if self._roi_M is None:
                self._prepare_roi_transform(frame)
            
            if self._roi_M is None or self._roi_size is None:
                return frame
            
            # Copy matrix untuk menghindari race saat warpPerspective
            M_copy = self._roi_M.copy()
            size_copy = self._roi_size
        
        # warpPerspective di LUAR mutex untuk performance
        return cv.warpPerspective(frame, M_copy, size_copy)

    def set_roi(self, roi_points):
        """
        Mengatur region of interest untuk kamera.
        
        Args:
            points: List koordinat titik ROI [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            image: numpy.ndarray gambar ROI yang diproses
        """
        if roi_points and isinstance(roi_points, str):
            try:
                self.roi_points = json.loads(roi_points)
            except:
                self.roi_points = None
        else:   
            self.roi_points = roi_points

        with QMutexLocker(self._roi_mutex):
            self.roi_points = roi_points
            self._roi_M = None
            self._roi_size = None
    
    @staticmethod
    def from_dict(data):
        """
        Membuat objek kamera dari bentuk dictionary.
    
        Args:
            data: Dictionary dengan data kamera
    
        Returns:
            Camera: Objek kamera baru
        """
        camera = Camera(
            camera_id=data.get('id'),
            name=data.get('name'),
            ip_address=data.get('ip_address'),
            port=data.get('port', 554),  # Default port RTSP
            username=data.get('username', ''),
            password=data.get('password', ''),
            stream_path=data.get('stream_path', 'stream1'),
            custom_url=data.get('url', ''),
            is_static=data.get('is_static', False),
            video_path=data.get('video_path', '')
        )
    
        if 'resolution' in data:
            camera.resolution = data['resolution']
        elif 'resolution_width' in data and 'resolution_height' in data:
            camera.resolution = (data['resolution_width'], data['resolution_height'])
        
        # Muat ROI jika tersedia
        if 'roi_points' in data and data['roi_points']:
            try:
                # Asumsi roi_points disimpan sebagai JSON string
                camera.roi_points = json.loads(data['roi_points'])
            except:
                pass
        
        return camera
    
    def set_preview_mode(self, enable: bool):
        """
        Ganti mode preview/detail sambil jalan.
        """
        if self.thread is None:
            return
        self.is_preview_mode = enable
        new_fps = FPS_PREVIEW if enable else FPS_DETAIL
        self.thread.set_target_fps(new_fps)




class CameraThread(QThread):
    """
    Thread terpisah untuk menangani streaming video dari kamera.
    Mencegah UI freeze saat mengambil frame.
    """
    
    # Signals untuk komunikasi dengan UI
    frame_received = Signal(np.ndarray)
    error_occurred = Signal(str)
    connection_changed = Signal(bool)
    
    def __init__(self, camera, target_fps=15): 
        """
        Inisialisasi thread kamera.
        
        Args:
            camera: Objek Camera yang akan diproses
        """
        super().__init__()
        self.camera = camera
        self._target_fps = max(1, target_fps)
        self.frame_interval = int(1000 / self._target_fps)
        self._frame_counter = 0 

    def set_target_fps(self, fps: int):
        """
        Ubah FPS target saat thread sudah berjalan.
        Akan aman‑thread karena hanya men‑update variabel atomik.
        """
        fps = max(1, fps)
        self._target_fps = fps
        self.frame_interval = int(1000 / fps)
    
    def run(self):
        """
        Metode utama thread yang berjalan ketika thread dimulai.
        Loop yang terus mengambil frame dari kamera.
        """
        self.running = True
        consecutive_errors = 0
        frame_counter = 0
        start_time = time.time()
        
        while self.running:
            try:
                if self.camera.connection_status:
                    with QMutexLocker(self.camera.mutex):
                        if self.camera.capture is not None:
                            ret, frame = self.camera.capture.read()
                            
                            if ret:
                                consecutive_errors = 0
                                self.camera.last_frame = frame
                                self.frame_received.emit(frame)

                                frame_counter += 1
                                elapsed_time = time.time() - start_time

                                if elapsed_time >= 1.0:
                                    fps = frame_counter / elapsed_time
                                    logger.info(f"[Camera {self.camera.name}] FPS saat ini: {fps:.2f}")
                                    frame_counter = 0
                                    start_time = time.time()
                            else:
                                # Frame tidak berhasil dibaca
                                consecutive_errors += 1
                                if consecutive_errors >= 5:
                                    self.camera.connection_status = False
                                    self.connection_changed.emit(False)
                                    
                                    # TAMBAH: Cleanup VideoCapture on persistent errors
                                    with QMutexLocker(self.camera.mutex):
                                        if self.camera.capture:
                                            try:
                                                self.camera.capture.release()
                                                self.camera.capture = None
                                            except Exception as e:
                                                logger.error(f"Failed to release capture: {e}")
                                    
                                    self.camera.reconnect_interval = min(60, self.camera.reconnect_interval * 2)
                else:
                    # Coba reconnect otomatis
                    if self.camera.try_reconnect():
                        self.connection_changed.emit(True)
                        consecutive_errors = 0
            
            except Exception as e:
                consecutive_errors += 1
                self.error_occurred.emit(f"Error saat streaming: {str(e)}")
                
                if consecutive_errors >= 5:
                    self.camera.connection_status = False
                    self.connection_changed.emit(False)
            
            # Interval untuk mengurangi CPU usage
            self._frame_counter += 1

            # Adaptif: kurangi FPS jika CPU tinggi
            if self._frame_counter % 15 == 0:
                cpu_now = psutil.cpu_percent(interval=None)
                if cpu_now > CPU_HIGH_THRESHOLD and self._target_fps > FPS_IDLE:
                    self.set_target_fps(max(self._target_fps // 2, FPS_IDLE))

            # Delay agar sesuai target_fps
            self.msleep(self.frame_interval)
    
    def stop(self):
        """Menghentikan thread dengan aman."""
        self.running = False
        # TAMBAHAN: Ensure kita keluar dari capture.read()
        if hasattr(self, 'camera') and self.camera.capture:
            # Set timeout pendek untuk keluar dari blocking read
            try:
                self.camera.capture.set(cv.CAP_PROP_BUFFERSIZE, 0)
            except:
                pass
        self.quit()
        # Pindahkan wait() ke sini, jangan di caller
        if not self.wait(3000):
            import logging
            logging.error(f"Camera thread {self.camera.name} failed to stop")


# Fungsi utilitas untuk konversi frame
def convert_cv_to_pixmap(cv_frame, target_size=None):
    """
    Mengkonversi frame OpenCV (numpy array) ke QPixmap untuk tampilan di Qt.
    
    Args:
        cv_frame: numpy.ndarray frame dari OpenCV (format BGR)
        target_size: Optional QSize untuk resize hasilnya
    
    Returns:
        QPixmap: Hasil konversi untuk tampilan di Qt
    """
    if cv_frame is None:
        return QPixmap()
    
    # Konversi BGR ke RGB
    rgb_frame = cv.cvtColor(cv_frame, cv.COLOR_BGR2RGB)
    
    # Konversi ke QImage
    h, w, ch = rgb_frame.shape
    bytes_per_line = ch * w
    image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
    
    # Konversi ke QPixmap
    pixmap = QPixmap.fromImage(image)
    
    # Resize jika target_size diberikan
    if target_size is not None:
        pixmap = pixmap.scaled(target_size, 
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
        )
    
    return pixmap


# Contoh penggunaan dasar:
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
    import sys
    
    app = QApplication(sys.argv)
    
    # Buat window test sederhana
    window = QMainWindow()
    window.setWindowTitle("Camera Test")
    window.setGeometry(100, 100, 800, 600)
    
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    
    layout = QVBoxLayout(central_widget)
    
    # Label untuk menampilkan frame
    frame_label = QLabel()
    frame_label.setMinimumSize(640, 480)
    layout.addWidget(frame_label)
    
    # Buat kamera
    camera = Camera(name="Test Camera", ip_address="192.168.1.100", port=8080)
    
    def update_frame(frame):
        """Callback saat frame diterima"""
        pixmap = convert_cv_to_pixmap(frame)
        frame_label.setPixmap(pixmap)
    
    def handle_error(error_message):
        """Callback saat error terjadi"""
        logger.error(f"Error: {error_message}")
    
    # Connect signals
    camera.thread = CameraThread(camera)
    camera.thread.frame_received.connect(update_frame)
    camera.thread.error_occurred.connect(handle_error)
    
    # Mulai koneksi kamera
    if camera.connect():
        camera.start_stream()
    
    window.show()
    
    # Cleanup saat aplikasi ditutup
    app.aboutToQuit.connect(camera.disconnect)
    
    sys.exit(app.exec())