import os
import csv
import datetime
import threading
import queue

from PySide6.QtCore import QStandardPaths, QRunnable, QThreadPool, QMetaObject, Qt, QTimer

# Lokasi default: Dokumen → EyeLog → DataLogs
DEFAULT_LOG_DIR = os.path.join(
    QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation),
    "EyeLog", "DataLogs"
)

class CoverageLogger:
    """
    Logger asinkron berbasis thread untuk menyimpan pengukuran coverage
    ke file CSV harian per kamera. Menjaga UI tetap responsif dengan
    memindahkan operasi I/O ke background thread.
    """
    def __init__(self, base_dir=None):
        # Direktori dasar untuk menyimpan log
        self.base_dir = base_dir or DEFAULT_LOG_DIR
        self._ensure_base_dir()

        # Antrian untuk pengukuran yang akan ditulis
        self._queue = queue.Queue()
        # Event untuk menghentikan worker
        self._stop_event = threading.Event()
        # Penyimpanan file handle agar tidak terus buka-tutup file
        self._file_handles = {}  # key: (camera_id, date_str) -> (file_obj, csv_writer)

        # Start worker thread sebagai daemon
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _ensure_base_dir(self):
        os.makedirs(self.base_dir, exist_ok=True)

    def log_measurement(self, camera_id, coverage, timestamp=None):
        """
        Tambahkan satu pengukuran ke antrian untuk penulisan.

        Args:
            camera_id (int): ID kamera
            coverage (float): Nilai coverage
            timestamp (datetime, optional): Waktu pengukuran. Default: sekarang

        Returns:
            bool: True jika berhasil menambahkan ke antrian
        """
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
        """
        Tambahkan batch pengukuran ke antrian.

        Args:
            measurements (list of tuple): (camera_id, date_str, time_str, coverage)

        Returns:
            int: Jumlah pengukuran yang ditambahkan
        """
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
        """
        Tunggu hingga seluruh antrian kosong (semua data sudah ditulis ke file).
        """
        self._queue.join()

    class _FlushRunner(QRunnable):
        """Jalankan CoverageLogger.flush() di thread‑pool global."""
        def __init__(self, logger, finished_cb=None):
            super().__init__()
            self._logger      = logger
            self._finished_cb = finished_cb

        def run(self):
            self._logger.flush()
            if callable(self._finished_cb):
                QTimer.singleShot(0, self._finished_cb)

    def flush_async(self, finished_cb=None):
        """
        Jalankan flush() tanpa mem‑blok UI.
        Args
        ----
        finished_cb : fungsi | lambda | None
            Dipanggil di GUI thread setelah semua antrian kosong.
        """
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
        """
        Worker thread: membaca item dari antrian dan menulis ke file CSV
        secara asynchronous.
        """
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            cam_id = item['camera_id']
            date_str = item['date_str']
            key = (cam_id, date_str)

            # Buka atau reuse file handle untuk kombinasi kamera dan tanggal
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

        # Tutup semua file handle saat thread berhenti
        for f, _ in self._file_handles.values():
            try:
                f.close()
            except Exception:
                pass
        self._file_handles.clear()

    def stop(self):
        """
        Hentikan worker thread dengan benar setelah menulis semua data.
        """
        self._stop_event.set()
        self._thread.join()

    # ----------------------------------------------------------
    # Bagian pembacaan log (synchronous, untuk DataLogsUI)
    # ----------------------------------------------------------

    def get_measurements(self, camera_id, date_str=None, limit=None):
        """
        Membaca pengukuran dari file CSV untuk tanggal tertentu.
        """
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
        # Urutkan descending
        measurements.sort(key=lambda x: x['timestamp'], reverse=True)
        if limit:
            return measurements[:limit]
        return measurements

    def get_dates_with_data(self, camera_id, max_days=30):
        """
        Mengembalikan tanggal dengan data yang ada di file log.
        """
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
        """
        Mengembalikan riwayat coverage terbaru untuk chart.
        """
        measurements = self.get_measurements(camera_id, limit=limit)
        history = []
        for m in measurements:
            ts = datetime.datetime.fromtimestamp(m['timestamp'])
            history.append({'timestamp': ts, 'value': m['coverage']})
        return history

    def get_measurements_count(self, camera_id, date_str=None):
        """
        Menghitung jumlah baris data pada file log untuk tanggal.
        """
        if date_str is None:
            date_str = datetime.datetime.now().strftime('%Y-%m-%d')
        path = self._get_log_file_path(camera_id, date_str)
        if not os.path.exists(path):
            return 0
        with open(path, 'r', newline='') as f:
            return sum(1 for _ in f) - 1  # kurangi header

    def export_to_csv(self, camera_id, date_str, output_file):
        """
        Ekspor data log ke file CSV eksternal.
        """
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
