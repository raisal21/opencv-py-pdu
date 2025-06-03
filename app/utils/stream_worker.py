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
from enum import IntEnum
from typing import Optional, Tuple
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker, QWaitCondition

logger = logging.getLogger(__name__)

class StreamState(IntEnum):
    """Stream worker states"""
    RUNNING = 0
    PAUSED = 1
    EDIT_ROI = 2
    STOPPING = 3
    RECONNECTING = 4

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
        
        # Performance tracking
        self.target_fps = 15
        self.frame_interval = 1.0 / self.target_fps
        self._last_process_time = 0
        self._process_time_history = []  # Rolling window of processing times
        self._history_size = 10
        
        # Timeout configuration
        self.read_timeout_ms = 100  # 100ms timeout for capture.read()
        self.reconnect_cooldown = 5.0  # Seconds between reconnect attempts
        self._last_reconnect_attempt = 0
        
        # Platform-specific setup
        self._setup_platform_specific()
        
        # Will be initialized in run()
        self.bg_subtractor = None
        self.contour_processor = None
        
    def _setup_platform_specific(self):
        """Platform-specific optimizations"""
        self.platform = platform.system()
        
        # Platform-specific timeout support check
        self._has_native_timeout = self._check_timeout_support()
        
        if self.platform == "Windows":
            # Windows-specific optimizations
            self.read_timeout_ms = 150  # Slightly higher for Windows
        elif self.platform == "Darwin":  # macOS
            # macOS-specific optimizations
            self.read_timeout_ms = 100
    
    def _check_timeout_support(self) -> bool:
        """Check if OpenCV supports CAP_PROP_OPEN_TIMEOUT_MSEC"""
        try:
            # Test with a dummy capture
            test_cap = cv.VideoCapture()
            if hasattr(cv, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
                # Try to set it
                test_cap.set(cv.CAP_PROP_OPEN_TIMEOUT_MSEC, 1000)
                return True
        except:
            pass
        return False
    
    # ===== State Management =====
    
    @property
    def state(self) -> StreamState:
        """Thread-safe state getter"""
        with QMutexLocker(self._state_mutex):
            return self._state
    
    def set_state(self, new_state: StreamState):
        """Thread-safe state setter with notification"""
        with QMutexLocker(self._state_mutex):
            if self._state != new_state:
                old_state = self._state
                self._state = new_state
                logger.info(f"State transition: {old_state.name} → {new_state.name}")
                self._state_condition.wakeAll()
                
        # Emit outside of mutex lock
        self.state_changed.emit(new_state)
    
    def pause(self):
        """Pause processing"""
        self.set_state(StreamState.PAUSED)
    
    def resume(self):
        """Resume processing"""
        self.set_state(StreamState.RUNNING)
    
    def enter_roi_edit(self):
        """Enter ROI editing mode"""
        self.set_state(StreamState.EDIT_ROI)
    
    def stop(self):
        """Stop the worker thread gracefully"""
        logger.info("StreamWorker stop requested")
        self.set_state(StreamState.STOPPING)
        
        # Wake up if waiting
        with QMutexLocker(self._state_mutex):
            self._state_condition.wakeAll()
        
        # Give thread time to exit gracefully
        if not self.wait(3000):
            logger.warning("StreamWorker did not stop gracefully in 3s")
    
    # ===== Timeout Mechanism =====
    
    def _read_with_timeout(self, capture, timeout_ms: int) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame with timeout mechanism.
        Uses native timeout if available, otherwise uses fallback method.
        """
        if self._has_native_timeout:
            # Use native OpenCV timeout
            return capture.read()
        else:
            # Fallback: Use threading for timeout
            return self._read_with_thread_timeout(capture, timeout_ms)
    
    def _read_with_thread_timeout(self, capture, timeout_ms: int) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Fallback timeout implementation using threading.
        """
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
        
        # Wait for thread with timeout
        thread.join(timeout_ms / 1000.0)
        
        if thread.is_alive():
            # Timeout occurred
            logger.warning("Frame read timeout")
            return False, None
        
        if exception[0]:
            raise exception[0]
            
        return result[0], result[1]
    
    def _create_capture_with_timeout(self, url: str) -> Optional[cv.VideoCapture]:
        """
        Create VideoCapture with timeout settings.
        """
        try:
            if self._has_native_timeout:
                # Use native timeout
                params = [
                    cv.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000,
                    cv.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms
                ]
                cap = cv.VideoCapture(url, cv.CAP_FFMPEG, params)
            else:
                # Create without timeout params
                cap = cv.VideoCapture(url, cv.CAP_FFMPEG)
            
            if cap.isOpened():
                # Set buffer size to reduce latency
                cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
                
                # Set resolution
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
    
    # ===== Main Thread Loop =====
    
    def run(self):
        """Main thread loop with integrated capture and processing"""
        logger.info("StreamWorker started")
        
        # Initialize processing components
        self._initialize_processors()
        
        # Initialize capture
        capture = None
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.state != StreamState.STOPPING:
            try:
                current_state = self.state
                
                # Handle different states
                if current_state == StreamState.PAUSED:
                    self._handle_paused_state()
                    continue
                    
                elif current_state == StreamState.EDIT_ROI:
                    self._handle_edit_roi_state()
                    continue
                    
                elif current_state == StreamState.RECONNECTING:
                    self._handle_reconnecting_state(capture)
                    capture = None  # Reset capture for reconnection
                    continue
                
                # Ensure we have a capture
                if capture is None or not capture.isOpened():
                    capture = self._establish_connection()
                    if capture is None:
                        self.set_state(StreamState.RECONNECTING)
                        continue
                
                # Normal processing loop
                loop_start = time.time()
                
                # Read frame with timeout
                ret, frame = self._read_with_timeout(capture, self.read_timeout_ms)
                
                if not ret or frame is None:
                    consecutive_errors += 1
                    logger.warning(f"Failed to read frame (error {consecutive_errors}/{max_consecutive_errors})")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error("Too many consecutive errors, attempting reconnection")
                        if capture:
                            capture.release()
                        capture = None
                        self.set_state(StreamState.RECONNECTING)
                        self.connection_changed.emit(False)
                    continue
                
                # Reset error counter on successful read
                consecutive_errors = 0
                
                # Check if we should process this frame
                if self._should_process_frame():
                    process_start = time.time()
                    
                    # Process frame (placeholder for now)
                    processed_frame, metrics = self._process_frame(frame)
                    
                    process_duration = time.time() - process_start
                    self._update_processing_stats(process_duration)
                    
                    # Emit result
                    self.frame_ready.emit(processed_frame, metrics)
                    self._last_process_time = loop_start
                
                # Frame pacing
                self._frame_pacing(loop_start)
                
            except Exception as e:
                logger.error(f"StreamWorker error: {e}", exc_info=True)
                self.error_occurred.emit(str(e))
                consecutive_errors += 1
                
                if consecutive_errors >= max_consecutive_errors:
                    if capture:
                        capture.release()
                    capture = None
                    self.set_state(StreamState.RECONNECTING)
        
        # Cleanup
        if capture:
            capture.release()
        logger.info("StreamWorker stopped")
    
    def _initialize_processors(self):
        """Initialize processing components"""
        # This will be implemented properly in Phase 2
        # For now, just create placeholders
        logger.info("Initializing processors (placeholder)")
        # TODO: Initialize ForegroundExtraction and ContourProcessor
    
    def _establish_connection(self) -> Optional[cv.VideoCapture]:
        """Establish connection to camera with timeout"""
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
        """Determine if current frame should be processed (placeholder)"""
        # Phase 2 will implement intelligent skipping
        # For now, simple time-based decision
        time_since_last = time.time() - self._last_process_time
        return time_since_last >= self.frame_interval
    
    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Process frame (placeholder for Phase 2)"""
        # For Phase 1, just return the frame with dummy metrics
        metrics = {
            'processed_coverage_percent': 0,
            'contour_count': 0
        }
        return frame, metrics
    
    def _update_processing_stats(self, duration: float):
        """Update processing time statistics"""
        self._process_time_history.append(duration)
        if len(self._process_time_history) > self._history_size:
            self._process_time_history.pop(0)
    
    def _frame_pacing(self, loop_start: float):
        """Maintain target frame rate"""
        elapsed = time.time() - loop_start
        sleep_time = max(0, self.frame_interval - elapsed)
        if sleep_time > 0:
            self.msleep(int(sleep_time * 1000))
    
    # ===== State Handlers =====
    
    def _handle_paused_state(self):
        """Handle paused state"""
        with QMutexLocker(self._state_mutex):
            # Wait until state changes
            self._state_condition.wait(self._state_mutex, 100)
    
    def _handle_edit_roi_state(self):
        """Handle ROI editing state"""
        # Similar to paused, but might have different behavior later
        with QMutexLocker(self._state_mutex):
            self._state_condition.wait(self._state_mutex, 100)
    
    def _handle_reconnecting_state(self, old_capture):
        """Handle reconnection attempts"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self._last_reconnect_attempt < self.reconnect_cooldown:
            remaining = self.reconnect_cooldown - (current_time - self._last_reconnect_attempt)
            self.msleep(min(1000, int(remaining * 1000)))
            return
        
        self._last_reconnect_attempt = current_time
        logger.info("Attempting reconnection...")
        
        # Clean up old capture if exists
        if old_capture:
            try:
                old_capture.release()
            except:
                pass
        
        # Try to reconnect
        # The main loop will attempt connection on next iteration
        self.set_state(StreamState.RUNNING)