# EyeLog: Realtime Computer Vision Monitoring

[\![Python 3.12.10][https://img.shields.io/badge/Python-3.12-blue.svg]][https://www.python.org/downloads/release/python-31210/]
[\![Latest Release][https://img.shields.io/github/v/release/raisal21/opencv_py_pdu]](https://github.com/raisal21/opencv-py-pdu/releases/tag/v1.0.0)

[cite\_start]**EyeLog** adalah sebuah aplikasi desktop robust yang dirancang untuk pemantauan kamera *realtime*, dilengkapi dengan analisis visual berbasis OpenCV dan antarmuka pengguna yang reaktif dan intuitif dibangun di atas PySide6. [cite: 1] [cite\_start]Aplikasi ini menawarkan solusi pemantauan visual yang andal, terutama untuk lingkungan industri dengan konektivitas terbatas di mana operasi lokal menjadi kunci. [cite: 44]

## 🚀 Get Started

### Downloads

Versi *executable* siap pakai untuk Windows tersedia di halaman **[GitHub Releases](https://github.com/raisal21/opencv-py-pdu/releases/tag/v1.0.0)**.

Kami menyediakan beberapa format untuk kemudahan Anda:

  * **Standalone (`.zip`)**: Paket folder portabel yang berisi semua dependensi. Cukup ekstrak dan jalankan.
  * **One-File (`.exe` di dalam `.zip`)**: Satu file *executable* untuk kemudahan maksimal.

### Installation from Source

Untuk menjalankan dari kode sumber, pastikan Anda memiliki Python 3.12+.

```bash
# 1. Clone the repository
git clone https://github.com/raisal21/opencv_py_pdu.git
cd opencv_py_pdu

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python -m run.py
```

-----

## ✨ Key Features

EyeLog tidak hanya menampilkan video, tetapi juga menganalisis dan mencatatnya secara cerdas.

| Fitur | Deskripsi Teknis |
| :--- | :--- |
| **Realtime Multi-Camera Monitoring** | [cite\_start]Menampilkan *stream* video dari berbagai IP Camera melalui protokol RTSP secara bersamaan. [cite: 5] [cite\_start]Setiap status kamera (`Online`/`Offline`) dipantau secara aktif menggunakan *ping worker* di *thread* terpisah untuk memastikan UI tetap responsif. [cite: 5] |
| **Condition-Aware Analysis** | [cite\_start]Melakukan deteksi cakupan material menggunakan *background subtraction* (MOG2) dari OpenCV. [cite: 5] [cite\_start]Tersedia berbagai *preset* deteksi (misalnya untuk kondisi malam, berdebu, atau bergetar) yang menyesuaikan parameter seperti `learningRate`, `shadows`, dan *morphology kernel* untuk akurasi maksimal dalam berbagai kondisi lapangan. [cite: 5, 13, 14] |
| **Interactive Visualization** | [cite\_start]Visualisasi data disajikan secara *realtime*, termasuk *overlay* kontur hasil deteksi langsung di atas video stream dan grafik riwayat cakupan (coverage history) yang interaktif menggunakan QtCharts. [cite: 5, 6] |
| **Autonomous CSV Logging** | [cite\_start]Sistem melakukan *auto-logging* metrik cakupan area (dalam persen) setiap interval waktu tertentu (misalnya 5 detik) ke dalam file `.csv`. [cite: 5, 10] [cite\_start]Setiap kamera memiliki file log harian sendiri, membuatnya mudah diakses dan dianalisis menggunakan tool standar seperti Excel tanpa memerlukan *database engine* yang berat. [cite: 5, 11] |

## 🧠 Core Concepts & Architecture

EyeLog dibangun di atas beberapa pilar arsitektur yang dirancang untuk performa dan keandalan.

### 1\. Non-Blocking UI by Design

Antarmuka pengguna (UI) harus tetap responsif, bahkan saat memproses beberapa *stream* video. [cite\_start]Semua operasi berat—seperti *streaming* kamera, pemrosesan *frame*, *ping status*, dan pencatatan ke disk—dijalankan di **thread terpisah**. [cite: 15] [cite\_start]EyeLog menggunakan *thread pool* untuk mengelola *workers* ini secara dinamis, memastikan tidak ada *blocking call* di *main thread*. [cite: 15, 41]

### 2\. Deep Dive: OpenCV Implementation

Analisis visual adalah inti dari EyeLog. [cite\_start]Kami tidak menggunakan model Deep Learning yang berat, melainkan pendekatan matematis klasik dari OpenCV yang lebih ringan dan cepat untuk skenario ini. [cite: 51, 52]

  * [cite\_start]**Background Subtraction (MOG2)**: Kami menggunakan algoritma MOG2 untuk memisahkan objek bergerak (*foreground*) dari latar belakang yang statis. [cite: 34] [cite\_start]Model latar belakang ini terus diperbarui, dan parameter seperti `history`, `varThreshold`, dan `learningRate` dapat disesuaikan melalui *preset* untuk beradaptasi dengan perubahan pencahayaan atau kondisi lainnya. [cite: 33]
  * [cite\_start]**Morphology Operations**: Untuk membersihkan hasil deteksi, operasi morfologi seperti `Erode`, `Dilate`, `Opening`, dan `Closing` diterapkan. [cite: 30] [cite\_start]Ini digunakan baik sebagai *pre-processing* untuk membersihkan *noise* pada frame awal, maupun *post-processing* untuk menyempurnakan *mask foreground* sebelum analisis kontur. [cite: 29, 31]
  * [cite\_start]**Contour Detection & Analysis**: Setelah mendapatkan *mask* biner dari *foreground*, kami mendeteksi kontur untuk mengidentifikasi setiap objek material. [cite: 38] [cite\_start]Kontur-kontur kecil disaring, dan kontur yang berdekatan dapat digabungkan untuk mendapatkan representasi area yang akurat, yang kemudian dihitung luasnya. [cite: 37]

### 3\. Architectural Choices (Tech Stack Justification)

| Teknologi | Alasan Pemilihan |
| :--- | :--- |
| **Desktop App** | [cite\_start]Dipilih karena kebutuhan operasi **lokal dan *realtime***. [cite: 43] [cite\_start]Arsitektur web akan memperkenalkan latensi (kamera → server → klien) dan sangat tidak andal di lokasi dengan konektivitas internet terbatas. [cite: 43, 44] [cite\_start]Akses *hardware* langsung juga jauh lebih superior di lingkungan desktop. [cite: 45] |
| **PySide6 (Qt)** | [cite\_start]Dipilih karena dukungan **multimedia dan *threading* yang kuat**, komponen UI yang modern dan lengkap, serta performa *rendering* yang lebih tinggi dibandingkan Tkinter. [cite: 48, 49, 50] [cite\_start]Kemampuannya untuk berjalan *cross-platform* dengan tampilan native adalah bonus besar. [cite: 47] |
| **SQLite** | [cite\_start]Dipilih karena sifatnya yang **ringan, *serverless*, dan *embedded***. [cite: 53] [cite\_start]Sempurna untuk aplikasi *single-user* yang membutuhkan persistensi data sederhana (seperti daftar kamera) tanpa overhead dari *database engine* eksternal seperti PostgreSQL. [cite: 53] |

-----

## 👨‍💻 Developer Guide

Tertarik untuk berkontribusi atau memahami lebih dalam? Berikut panduannya.

### Project Structure

Struktur proyek dirancang agar modular dan mudah dipahami.

```
opencv-py-pdu/
├── app/                  # Main application source code
│   ├── assets/           # Static assets (fonts, icons, images)
│   ├── models/           # Data models and database interactions
│   ├── tests/            # Unit tests
│   ├── utils/            # Utility modules (logging, detection)
│   ├── views/            # UI components and dialogs
│   ├── main_window.py    # Main application window
│   └── resources.py      # Asset path management
├── database/
│   └── eyelog_database.db # Default SQLite database file
├── machine_learning/     # OpenCV specific modules
├── ...
├── requirements.txt      # Application dependencies
└── run.py                # Main entry point script
```

### Application Flow & Multithreading

Aplikasi ini sangat bergantung pada *multithreading* untuk menjaga performa. Berikut adalah gambaran alur kerja dan pembagian *thread*-nya:

  * **Main Thread**: Hanya bertanggung jawab untuk me-*render* UI dan menangani interaksi pengguna.
  * **Stream Workers Thread**: Memproses *frame* yang diterima (analisis OpenCV) di *thread* terpisah.
  * **Ping Workers Thread**: Periodik memeriksa status koneksi setiap kamera.
  * [cite\_start]**Coverage Logger Thread**: Menulis data cakupan ke file CSV secara asinkron agar tidak memblokir UI. [cite: 6]
  * **Database Workers Thread**: Menangani semua operasi baca/tulis ke database SQLite.

*(Anda bisa menyisipkan gambar flowchart Anda di sini)*
`![Application Flowchart](flowchart/Eyelog-Flowchart.svg)`

### How to Contribute

Kontribusi dalam bentuk apapun sangat saya hargai\!

1.  **Fork** repositori ini.
2.  Buat **branch** baru untuk fitur Anda (`git checkout -b feature/AmazingFeature`).
3.  **Commit** perubahan Anda (`git commit -m 'Add some AmazingFeature'`).
4.  **Push** ke branch Anda (`git push origin feature/AmazingFeature`).
5.  Buka **Pull Request**.
