# EyeLog - Computer Vision Realtime Monitoring

**EyeLog** adalah aplikasi desktop yang mengelola pemantauan kamera secara realtime, dilengkapi dengan analisis berbasis OpenCV dan antarmuka intuitif menggunakan PySide6.

## 📌 Main Feature
- Pemantauan realtime melalui protokol RTSP.
- Deteksi area cakupan menggunakan OpenCV dengan berbagai mode latar belakang.
- Dukungan multi-kamera dengan pengaturan dan manajemen yang mudah.
- Visualisasi area cakupan secara real-time dengan grafik interaktif.
- Log otomatis dan penyimpanan data untuk analisis historis.

## 🛠️ Tech Stacks
- **Python 3.12**
- **PySide6**
- **OpenCV**
- **SQLite**
- **Matplotlib**

## 📂 Struktur Proyek

```
opencv-py-pdu/
├── app/
│ ├── assets/ (font, ikon, gambar)
│ ├── models/ (database, kamera)
│ ├── tests/ (unit testing)
│ ├── utils/ (log, scheduler, deteksi material)
│ ├── views/ (UI seperti tambah kamera, detail kamera)
│ ├── __init__.py (inisialisasi aplikasi)
│ ├── main_window.py (file utama aplikasi)
│ ├── splash_screen.py (loading app)
│ └── resources.py (pengelolaan path asset)
├── database/
│ └── eyelog_database.db (file database SQLite)
├── machine_learning/
│ ├── main.py (skrip utama ML)
│ ├── material_detector.py(modul deteksi material OpenCV)
│ └── roi_selector.py (modul seleksi ROI OpenCV)
├── flowchart/
│ └── flowchart-desktop-app.mermaid (diagram alur aplikasi)
├── backups/ (folder backup code)
├── EyeLog.spec (konfigurasi build PyInstaller)
├── requirements.txt (dependensi aplikasi)
├── README.md
└── .gitignore
```

## 🚀 Instalation
Clone repositori ini dan instal dependensi melalui pip:

```bash
git clone https://github.com/raisal21/opencv_py_pdu.git
cd eyelog
pip install -r requirements.txt
```

Langkah singkat memulai aplikasi:

```bash
python -m app
```

## 📦 Packaging dengan PySide6 Deploy

Untuk membuat executable standalone gunakan `pyside6-deploy` bersama Nuitka. Pastikan
`pysidedeploy.spec` memiliki pengaturan berikut agar folder `app/assets` tetap
terkopi apa adanya:

```
extra_files = app/assets/**:app/assets
```

Perintah build yang umum dijalankan:

```bash
pyside6-deploy -c pysidedeploy.spec --mode standalone --keep-deployment-files -v
```

Folder hasil dapat ditemukan di `release/nuitka/Eyelog-Standalone.dist` dan
fungsi `resource_path()` secara otomatis akan mencari asset di lokasi tersebut
saat aplikasi dijalankan.


