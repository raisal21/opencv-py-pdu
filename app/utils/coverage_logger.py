import os
import csv
import datetime
import threading
import queue

from PySide6.QtCore import QStandardPaths, QRunnable, QThreadPool, QMetaObject, Qt, QTimer


DEFAULT_LOG_DIR = os.path.join(
    QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
    "EyeLog", "DataLogs"
)

class CoverageLogger:
    """Asynchronous CSV logger for per-camera coverage measurements."""
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or DEFAULT_LOG_DIR
        self._ensure_base_dir()

        self._queue = queue.Queue()

        self._stop_event = threading.Event()

        self._file_handles = {}

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _ensure_base_dir(self):
        os.makedirs(self.base_dir, exist_ok=True)

    def log_measurement(self, camera_id, coverage, timestamp=None):
        """Queue one coverage measurement for writing."""
        if timestamp is None:
            timestamp_dt = datetime.datetime.now()
        else:
            timestamp_dt = timestamp
        date_str = timestamp_dt.strftime("%Y-%m-%d")
        time_str = timestamp_dt.strftime("%H:%M:%S")
        timestamp_float = timestamp_dt.timestamp()

        self._queue.put({
            'camera_id': camera_id,
            'date_str': date_str,
            'time_str': time_str,
            'timestamp': timestamp_float,
            'coverage': coverage
        })
        return True

    def log_measurements_batch(self, measurements):
        """Queue a batch of coverage measurements."""
        count = 0
        for camera_id, date_str, time_str, coverage in measurements:
            try:
                ts = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                self.log_measurement(camera_id, coverage, ts)
                count += 1
            except Exception:
                continue
        return count

    def flush(self):
        """Wait until all queued measurements have been written."""
        self._queue.join()

    class _FlushRunner(QRunnable):
        """Run CoverageLogger.flush() in the global thread pool."""
        def __init__(self, logger, finished_cb=None):
            super().__init__()
            self._logger      = logger
            self._finished_cb = finished_cb

        def run(self):
            self._logger.flush()
            if callable(self._finished_cb):
                from PySide6.QtCore import QTimer
                from PySide6.QtWidgets import QApplication
                QTimer.singleShot(0, QApplication.instance(), self._finished_cb)

    def flush_async(self, finished_cb=None):
        """Run flush() without blocking the UI thread."""
        runner = self._FlushRunner(self, finished_cb)
        QThreadPool.globalInstance().start(runner)

    def _get_camera_dir(self, camera_id):
        path = os.path.join(self.base_dir, f"camera_{camera_id}")
        os.makedirs(path, exist_ok=True)
        return path

    def _get_log_file_path(self, camera_id, date_str):
        camera_dir = self._get_camera_dir(camera_id)
        return os.path.join(camera_dir, f"{date_str}.csv")

    def _worker(self):
        """Write queued measurements to daily CSV files."""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            cam_id = item['camera_id']
            date_str = item['date_str']
            key = (cam_id, date_str)

            if key not in self._file_handles:
                path = self._get_log_file_path(cam_id, date_str)
                is_new = not os.path.exists(path)
                f = open(path, 'a', newline='')
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(['timestamp', 'time', 'coverage'])
                    f.flush()
                self._file_handles[key] = (f, writer)

            f, writer = self._file_handles[key]
            writer.writerow([item['timestamp'], item['time_str'], item['coverage']])
            f.flush()

            self._queue.task_done()

        for f, _ in self._file_handles.values():
            try:
                f.close()
            except Exception:
                pass
        self._file_handles.clear()

    def stop(self):
        """Stop the worker after pending measurements are written."""
        self._stop_event.set()
        self._thread.join()

    def get_measurements(self, camera_id, date_str=None, limit=None):
        """Read measurements for a camera and date."""
        if date_str is None:
            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        path = self._get_log_file_path(camera_id, date_str)
        if not os.path.exists(path):
            return []

        measurements = []
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                measurements.append({
                    'camera_id': camera_id,
                    'date': date_str,
                    'time': row['time'],
                    'coverage': float(row['coverage']),
                    'timestamp': float(row['timestamp'])
                })

        measurements.sort(key=lambda x: x['timestamp'], reverse=True)
        if limit:
            return measurements[:limit]
        return measurements

    def get_dates_with_data(self, camera_id, max_days=30):
        """Return log dates that contain measurements."""
        camera_dir = self._get_camera_dir(camera_id)
        today = datetime.date.today()
        dates = []
        for i in range(max_days):
            d = today - datetime.timedelta(days=i)
            path = os.path.join(camera_dir, f"{d.strftime('%Y-%m-%d')}.csv")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                dates.append(d.strftime('%Y-%m-%d'))
        return dates

    def get_coverage_history(self, camera_id, limit=5):
        """Return recent coverage values for charts."""
        measurements = self.get_measurements(camera_id, limit=limit)
        history = []
        for m in measurements:
            ts = datetime.datetime.fromtimestamp(m['timestamp'])
            history.append({'timestamp': ts, 'value': m['coverage']})
        return history

    def get_measurements_count(self, camera_id, date_str=None):
        """Count measurement rows for a camera and date."""
        if date_str is None:
            date_str = datetime.datetime.now().strftime('%Y-%m-%d')
        path = self._get_log_file_path(camera_id, date_str)
        if not os.path.exists(path):
            return 0
        with open(path, 'r', newline='') as f:
            return sum(1 for _ in f) - 1

    def export_to_csv(self, camera_id, date_str, output_file):
        """Export measurements to an external CSV file."""
        measurements = self.get_measurements(camera_id, date_str)
        if not measurements:
            return False
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Date', 'Time', 'Coverage (%)'])
            measurements = sorted(measurements, key=lambda x: x['time'])
            for i, m in enumerate(measurements, start=1):
                writer.writerow([i, m['date'], m['time'], m['coverage']])
        return True
