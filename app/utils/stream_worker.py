"""
Stream Worker - Unified single-thread approach for camera streaming and processing
Phase 1: Foundation with timeout mechanism and state management
"""
import time
import cv2 as cv
import numpy as np
import logging
import platform
import threading
import json
import gc
from enum import IntEnum
from typing import Optional, Tuple
from ..utils.material_detector import ForegroundExtraction, ContourProcessor
from PySide6.QtCore import (
    QThread, Signal, QMutex, QMutexLocker, QWaitCondition,
    Qt, Slot, QTimer
)

logger = logging.getLogger(__name__)

class StreamState(IntEnum):
    """Stream worker states"""
    RUNNING = 0
    PAUSED = 1
    EDIT_ROI = 2
    STOPPING = 3
    RECONNECTING = 4
    UPDATING_PRESETS = 5
    RESTARTING = 6

class StreamWorker(QThread):
    """
    Single-thread worker that handles capture, processing, and emission
    with proper timeout mechanism and state management.
    """
    
    # Signals
    frame_ready = Signal(np.ndarray, dict)  # (processed_frame, metrics)
    state_changed = Signal(int)  # StreamState
    connection_changed = Signal(bool)  # Connection status
    error_occurred = Signal(str)  # Error message
    
    def __init__(self, camera, bg_params: dict, contour_params: dict, parent=None):
        super().__init__(parent)
        
        # Camera and processing components
        self.camera = camera
        self.bg_params = bg_params
        self.contour_params = contour_params
        
        # State management
        self._state = StreamState.RUNNING
        self._state_mutex = QMutex()
        self._state_condition = QWaitCondition()
        
        # --- Perbaikan: Mekanisme stop dan update preset berbasis flag ---
        self._stop_flag = False
        self._stop_mutex = QMutex()
        self._pending_bg_params: Optional[dict] = None
        self._pending_contour_params: Optional[dict] = None
        self._params_mutex = QMutex()

        self._raw_frame_mutex = QMutex()
        self._latest_raw_frame: Optional[np.ndarray] = None

        self._pending_roi_restart = None
        self._restart_mutex = QMutex()
        
        # Performance tracking
        self.target_fps = 15
        self.frame_interval = 1.0 / self.target_fps
        self._last_process_time = 0
        self._process_time_history = []
        self._history_size = 10
        
        # Timeout configuration
        self.read_timeout_ms = 100
        self.reconnect_cooldown = 5.0
        self._last_reconnect_attempt = 0
        
        self._setup_platform_specific()
        
        # Will be initialized in run()
        self.bg_subtractor: Optional[ForegroundExtraction] = None
        self.contour_processor: Optional[ContourProcessor] = None
        self._processing_time_avg: float = 0.0
        self._capture: Optional[cv.VideoCapture] = None
        self._capture_mutex = QMutex()
        self._timer: Optional[QTimer] = None
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5
        self.latest_raw_frame: np.ndarray | None = None

    def _setup_platform_specific(self):
        """Platform-specific optimizations"""
        self.platform = platform.system()
        self._has_native_timeout = self._check_timeout_support()
        
        if self.platform == "Windows":
            self.read_timeout_ms = 150
        elif self.platform == "Darwin":
            self.read_timeout_ms = 100
    
    def _check_timeout_support(self) -> bool:
        """Check if OpenCV supports CAP_PROP_OPEN_TIMEOUT_MSEC"""
        try:
            test_cap = cv.VideoCapture()
            if hasattr(cv, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
                test_cap.set(cv.CAP_PROP_OPEN_TIMEOUT_MSEC, 1000)
                return True
        except:
            pass
        return False
    
    # ===== State Management =====
    
    @property
    def state(self) -> StreamState:
        with QMutexLocker(self._state_mutex):
            return self._state
    
    def set_state(self, new_state: StreamState):
        with QMutexLocker(self._state_mutex):
            if self._state != new_state:
                old_state = self._state
                self._state = new_state
                logger.info(f"State transition: {old_state.name} → {new_state.name}")
                self._state_condition.wakeAll()
        self.state_changed.emit(new_state)
    
    def pause(self):
        self.set_state(StreamState.PAUSED)
    
    def resume(self):
        self.set_state(StreamState.RUNNING)
    
    def enter_roi_edit(self):
        self.set_state(StreamState.EDIT_ROI)

    def stop(self):
        """Boleh dipanggil dari thread mana pun. Menggunakan flag yang aman."""
        logger.info("StreamWorker stop requested")
        with QMutexLocker(self._stop_mutex):
            self._stop_flag = True
        self._state_condition.wakeAll()
        
        if not self.wait(3000):
            logger.error("StreamWorker forced termination (3s timeout)")
            self.terminate()

    def _is_stopping(self) -> bool:
        """Pemeriksaan flag stop yang thread-safe."""
        with QMutexLocker(self._stop_mutex):
            return self._stop_flag

    # ===== Main Thread Loop =====
    def run(self):
        """Start thread dengan event‑loop Qt + QTimer adaptif."""
        logger.info("StreamWorker started (event‑loop mode)")

        self._initialize_processors()
        self._capture = None

        self._timer = QTimer()
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._iteration, Qt.DirectConnection)
        self._timer.start(0)

        self.exec()

        self._cleanup_capture()
        logger.info("StreamWorker stopped")
    
    def _cleanup_capture(self):
        with QMutexLocker(self._capture_mutex):
            if self._capture:
                try:
                    self._capture.release()
                except Exception:
                    pass
                self._capture = None
                # gc.collect()

    def _iteration(self):
        """Dipanggil periodik oleh QTimer – satu siklus penuh."""
        if self._is_stopping():
            if self._timer and self._timer.isActive():
                self._timer.stop()
            self.quit()
            return

        should_restart = False
        new_roi_points = None
        with QMutexLocker(self._restart_mutex):
            if self._pending_roi_restart is not None:
                new_roi_points = self._pending_roi_restart
                self._pending_roi_restart = None
        
        if new_roi_points is not None:
            logger.info("Restarting stream due to ROI change.")
            self.camera.roi_points = new_roi_points
            self._cleanup_capture()  # Hancurkan koneksi lama
            self.set_state(StreamState.RUNNING) # Pastikan state kembali running
            self.msleep(100) # Beri jeda singkat sebelum mencoba lagi
            return

        if should_restart:
            self.set_state(StreamState.RESTARTING)
            logger.info("Restarting stream due to ROI change. State: RESTARTING")
            
            # Lakukan semua operasi restart dalam satu blok terkontrol
            self._cleanup_capture()
            self.camera.roi_points = new_roi_points
            
            logger.info("Attempting direct reconnection within restart sequence.")
            self._capture = self._establish_connection()
            
            if self._capture and self._capture.isOpened():
                logger.info("Reconnection successful. Resuming normal operation.")
                self.set_state(StreamState.RUNNING)
            else:
                logger.warning("Reconnection failed. Entering RECONNECTING state.")
                self.set_state(StreamState.RECONNECTING)
                self._last_reconnect_attempt = time.time()
            
            # Restart timer setelah semua operasi selesai
            if self._timer and not self._timer.isActive():
                self._timer.start(0)
            return

        state = self.state
        if state != StreamState.RUNNING:
            if state in (StreamState.PAUSED, StreamState.EDIT_ROI, StreamState.UPDATING_PRESETS):
                 self.msleep(100) # Jeda singkat untuk state non-aktif
            elif state == StreamState.RECONNECTING:
                self._handle_reconnecting_state()
            # Jangan lakukan apa-apa untuk state RESTARTING karena sudah ditangani di atas
            return

        # --- Perbaikan: Terapkan update preset secara aman ---
        self._apply_pending_preset_updates()

        loop_start = time.time()

        try:
            if self._capture is None or not self._capture.isOpened():
                logger.warning("Capture is not available or closed. Entering RECONNECTING state.")
                self._capture = self._establish_connection()
                if self._capture is None:
                    self.set_state(StreamState.RECONNECTING)
                    self._adjust_timer(loop_start)
                    return
                else:
                    logger.info("Initial connection successful. Proceeding with frame read.")

            ret, frame = self._read_with_timeout(self._capture, self.read_timeout_ms)
            if not ret or frame is None:
                self._consecutive_errors += 1
                if self._consecutive_errors >= self._max_consecutive_errors:
                    self._handle_too_many_errors()
                self._adjust_timer(loop_start)
                return
            self._consecutive_errors = 0

            with QMutexLocker(self._raw_frame_mutex):
                self._latest_raw_frame = frame.copy()

            if self._should_process_frame():
                t0 = time.time()
                processed, metrics = self._process_frame(frame)
                self._update_processing_stats(time.time() - t0)
                self.frame_ready.emit(processed, metrics)
                self._last_process_time = loop_start

        except Exception as e:
            logger.error(f"StreamWorker error: {e}", exc_info=True)
            self.error_occurred.emit(str(e))
            self._consecutive_errors += 1
            if self._consecutive_errors >= self._max_consecutive_errors:
                self._handle_too_many_errors()
        finally:
            self._adjust_timer(loop_start)

    def get_latest_raw_frame(self) -> Optional[np.ndarray]:
        """
        Secara thread-safe mengambil frame mentah terakhir yang ditangkap.
        Mengembalikan salinan untuk mencegah race conditions.
        """
        with QMutexLocker(self._raw_frame_mutex):
            if self._latest_raw_frame is not None:
                return self._latest_raw_frame.copy()
        return None
    
    def _adjust_timer(self, loop_start: float):
        elapsed = time.time() - loop_start
        sleep_time = max(0, self.frame_interval - elapsed)
        new_interval = max(0, int(sleep_time * 1000))
        if self._timer and self._timer.interval() != new_interval:
            self._timer.setInterval(new_interval)
    
    def _handle_too_many_errors(self):
        logger.error("Too many consecutive errors, reconnecting")
        self._cleanup_capture()
        self.connection_changed.emit(False)
        self.set_state(StreamState.RECONNECTING)
    
    @Slot(str)
    def _restart_stream_internal(self, roi_points_json: str):
        """
        Slot internal yang aman untuk menangani restart stream.
        Langkah-langkahnya sekarang lebih eksplisit untuk mencegah race conditions.
        """

        if self._timer and self._timer.isActive():
            self._timer.stop()

        self.set_state(StreamState.RESTARTING)
        logger.info("Executing internal stream restart. Pausing processing loop.")

        # LANGKAH 2: Parse data ROI.
        try:
            roi_points = json.loads(roi_points_json)
            # LANGKAH 3: Perbarui konfigurasi dan cleanup koneksi lama.
            logger.info("Updating camera config and cleaning up old connection.")
            if hasattr(self, 'camera') and self.camera:
                if hasattr(self.camera, 'set_roi'):
                    self.camera.set_roi(roi_points)
                else:
                    self.camera.roi_points = roi_points
            self._cleanup_capture()

            # LANGKAH 4: Coba sambung ulang secara langsung di sini.
            logger.info("Attempting direct reconnection within restart sequence.")
            self._capture = self._establish_connection()

            # LANGKAH 5: Tentukan langkah selanjutnya berdasarkan hasil koneksi.
            if self._capture and self._capture.isOpened():
                logger.info("Direct reconnection successful. Resuming processing loop.")
                self.set_state(StreamState.RUNNING)
            else:
                logger.warning("Direct reconnection failed. Entering RECONNECTING state.")
                self.set_state(StreamState.RECONNECTING)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode ROI points from JSON: {roi_points_json}")
            self.set_state(StreamState.RUNNING)
            return

    def restart_with_new_roi(self, points: list):
        """
        Secara thread-safe meminta worker untuk me-restart stream dengan ROI baru.
        Menggunakan QMetaObject.invokeMethod untuk menjamin eksekusi yang aman.
        """
        """Request ROI restart in a thread-safe manner."""
        logger.info("Queueing ROI restart request.")
        with QMutexLocker(self._restart_mutex):
            self._pending_roi_restart = points
        self._state_condition.wakeAll()

    def _initialize_processors(self):
        logger.info("Initializing processors (Phase 2)")
        # if self.bg_subtractor is not None:
        #     self.bg_subtractor = None
        # if self.contour_processor is not None:
        #     self.contour_processor = None
        # gc.collect()
        
        self.bg_subtractor = ForegroundExtraction(**self.bg_params)
        self.contour_processor = ContourProcessor(**self.contour_params)
    
    def _establish_connection(self) -> Optional[cv.VideoCapture]:
        if not hasattr(self.camera, 'build_stream_url'):
            logger.error("Camera does not have build_stream_url method")
            return None
        url = self.camera.build_stream_url()
        logger.info(f"Attempting connection to: {url}")
        
        capture = self._create_capture_with_timeout(url)
        if capture:
            self.connection_changed.emit(True)
            logger.info("Connection established successfully")
        else:
            self.error_occurred.emit("Failed to connect to camera")
        return capture
    
    def _should_process_frame(self) -> bool:
        time_since_last = time.time() - self._last_process_time
        if self._process_time_history:
            self._processing_time_avg = sum(self._process_time_history) / len(self._process_time_history)
        if self._processing_time_avg > 0.030:
            return time_since_last > 0.066
        else:
            return time_since_last >= self.frame_interval

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        if not (self.bg_subtractor and self.contour_processor):
            return frame, {}
        roi_frame = self.camera.process_frame_with_roi(frame)
        if roi_frame is None:
            return frame, {}
        fg = self.bg_subtractor.process_frame(roi_frame)
        cnt = self.contour_processor.process_mask(fg.binary)
        display = self.contour_processor.visualize(roi_frame, cnt.contours, cnt.metrics, show_metrics=False)
        return display, cnt.metrics
    
    def _update_processing_stats(self, duration: float):
        self._process_time_history.append(duration)
        if len(self._process_time_history) > self._history_size:
            self._process_time_history.pop(0)

    # ---- Perbaikan: API preset yang thread-safe ----
    def _apply_pending_preset_updates(self):
        """Menerapkan pembaruan preset yang tertunda dari UI thread."""
        bg_updated, contour_updated = False, False

        if self.state != StreamState.RUNNING:
            return

        with QMutexLocker(self._params_mutex):
            if self._pending_bg_params:
                self.bg_params = self._pending_bg_params
                self._pending_bg_params = None
                bg_updated = True
            if self._pending_contour_params:
                self.contour_params = self._pending_contour_params
                self._pending_contour_params = None
                contour_updated = True
        
        if bg_updated or contour_updated:
            logger.info("Applying new preset parameters.")
            self.set_state(StreamState.UPDATING_PRESETS)
            self._initialize_processors()
            self._process_time_history.clear()
            self.set_state(StreamState.RUNNING)

    def set_bg_params(self, params: dict):
        """Dipanggil UI → simpan parameter baru untuk diterapkan oleh worker."""
        with QMutexLocker(self._params_mutex):
            self._pending_bg_params = params

    def set_contour_params(self, params: dict):
        """Dipanggil UI → simpan parameter baru untuk diterapkan oleh worker."""
        with QMutexLocker(self._params_mutex):
            self._pending_contour_params = params

    # ===== State Handlers =====
    def _handle_paused_state(self):
        with QMutexLocker(self._state_mutex):
            self._state_condition.wait(self._state_mutex, 100)
    
    def _handle_reconnecting_state(self):
        """Handle reconnection attempts. (Sedikit diubah untuk tidak memerlukan argumen)"""
        current_time = time.time()
        
        if current_time - self._last_reconnect_attempt < self.reconnect_cooldown:
            self.msleep(1000) # Tunggu 1 detik sebelum cek lagi
            return
        
        self._last_reconnect_attempt = current_time
        logger.info("Attempting reconnection from main loop...")
        
        # Biarkan iterasi berikutnya mencoba _establish_connection
        self.set_state(StreamState.RUNNING)

    def _read_with_timeout(self, capture, timeout_ms: int) -> Tuple[bool, Optional[np.ndarray]]:
        if self._has_native_timeout:
            return capture.read()
        else:
            return self._read_with_thread_timeout(capture, timeout_ms)
    
    def _read_with_thread_timeout(self, capture, timeout_ms: int) -> Tuple[bool, Optional[np.ndarray]]:
        result = [False, None]
        exception = [None]
        
        def read_thread():
            try:
                result[0], result[1] = capture.read()
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=read_thread)
        thread.daemon = True
        thread.start()
        thread.join(timeout_ms / 1000.0)
        
        if thread.is_alive():
            logger.warning("Frame read timeout")
            return False, None
        
        if exception[0]:
            raise exception[0]
            
        return result[0], result[1]
    
    def _create_capture_with_timeout(self, url: str) -> Optional[cv.VideoCapture]:
        try:
            if self._has_native_timeout:
                params = [cv.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000, cv.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms]
                cap = cv.VideoCapture(url, cv.CAP_FFMPEG, params)
            else:
                cap = cv.VideoCapture(url, cv.CAP_FFMPEG)
            
            if cap.isOpened():
                cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                if hasattr(self.camera, 'resolution'):
                    cap.set(cv.CAP_PROP_FRAME_WIDTH, self.camera.resolution[0])
                    cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.camera.resolution[1])
                return cap
            else:
                cap.release()
                return None
        except Exception as e:
            logger.error(f"Failed to create capture: {e}")
            return None
