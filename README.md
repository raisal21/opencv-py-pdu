# EyeLog - Aplikasi Pemantauan Kamera Realtime

**EyeLog** adalah aplikasi desktop yang mengelola pemantauan kamera secara realtime, dilengkapi dengan analisis berbasis OpenCV dan antarmuka intuitif menggunakan PySide6.

## 📌 Fitur Utama
- Pemantauan realtime melalui protokol RTSP.
- Deteksi area cakupan menggunakan OpenCV dengan berbagai mode latar belakang.
- Dukungan multi-kamera dengan pengaturan dan manajemen yang mudah.
- Visualisasi area cakupan secara real-time dengan grafik interaktif.
- Log otomatis dan penyimpanan data untuk analisis historis.

## 🛠️ Teknologi yang Digunakan
- **Python 3.12**
- **PySide6**
- **OpenCV**
- **SQLite** untuk penyimpanan data
- **Matplotlib** untuk visualisasi

## 📂 Struktur Proyek
opencv-py-pdu/
├── app/
│ ├── assets/ (font, ikon, gambar)
│ ├── models/ (database, kamera)
│ ├── tests/ (unit testing)
│ ├── utils/ (modul tambahan seperti log, scheduler, deteksi material)
│ ├── views/ (antarmuka UI seperti tambah kamera, detail kamera, dll.)
│ ├── init.py
│ ├── main_window.py (file utama aplikasi)
│ └── resources.py (pengelolaan path asset)
├── database/
│ └── eyelog_database.db (file database SQLite)
├── machine_learning/
│ ├── main.py
│ ├── material_detector.py (modul deteksi material OpenCV)
│ └── roi_selector.py (modul seleksi ROI OpenCV)
├── flowchart/
├── backups/ (folder backup code)
├── EyeLog.spec (konfigurasi build PyInstaller)
├── requirements.txt (dependensi aplikasi)
├── README.md
└── .gitignore

## 🚀 Cara Instalasi untuk Tujuan Pembelajaran
Clone repositori ini dan instal dependensi melalui pip:

```bash
git clone https://github.com/raisal21/opencv_py_pdu.git
cd eyelog
pip install -r requirements.txt

Langkah singkat memulai aplikasi:

```bash
python app/__init__.py



