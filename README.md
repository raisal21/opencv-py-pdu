# EyeLog: Realtime Computer Vision Monitoring

[\![Python 3.12.10][https://img.shields.io/badge/Python-3.12-blue.svg]][https://www.python.org/downloads/release/python-31210/]
[\![Latest Release][https://img.shields.io/github/v/release/raisal21/opencv_py_pdu]](https://github.com/raisal21/opencv-py-pdu/releases/tag/v1.0.0)

**EyeLog** adalah sebuah aplikasi desktop robust yang dirancang untuk pemantauan kamera *realtime*, dilengkapi dengan analisis visual berbasis OpenCV dan antarmuka pengguna yang reaktif dan intuitif dibangun di atas PySide6. Aplikasi ini menawarkan solusi pemantauan visual yang andal, terutama untuk lingkungan industri dengan konektivitas terbatas di mana operasi lokal menjadi kunci.

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
| **Realtime Multi-Camera Monitoring** | Menampilkan *stream* video dari berbagai IP Camera melalui protokol RTSP secara bersamaan. Setiap status kamera (`Online`/`Offline`) dipantau secara aktif menggunakan *ping worker* di *thread* terpisah untuk memastikan UI tetap responsif. |
| **Condition-Aware Analysis** | Melakukan deteksi cakupan material menggunakan *background subtraction* (MOG2) dari OpenCV. Tersedia berbagai *preset* deteksi (misalnya untuk kondisi malam, berdebu, atau bergetar) yang menyesuaikan parameter seperti `learningRate`, `shadows`, dan *morphology kernel* untuk akurasi maksimal dalam berbagai kondisi lapangan. |
| **Interactive Visualization** | Visualisasi data disajikan secara *realtime*, termasuk *overlay* kontur hasil deteksi langsung di atas video stream dan grafik riwayat cakupan (coverage history) yang interaktif menggunakan QtCharts. |
| **Autonomous CSV Logging** | Sistem melakukan *auto-logging* metrik cakupan area (dalam persen) setiap interval waktu tertentu (misalnya 5 detik) ke dalam file `.csv`. Setiap kamera memiliki file log harian sendiri, membuatnya mudah diakses dan dianalisis menggunakan tool standar seperti Excel tanpa memerlukan *database engine* yang berat. |

## 🧠 Core Concepts & Architecture

EyeLog dibangun di atas beberapa pilar arsitektur yang dirancang untuk performa dan keandalan.

### 1\. Non-Blocking UI by Design

Antarmuka pengguna (UI) harus tetap responsif, bahkan saat memproses beberapa *stream* video. Semua operasi berat—seperti *streaming* kamera, pemrosesan *frame*, *ping status*, dan pencatatan ke disk—dijalankan di **thread terpisah**. EyeLog menggunakan *thread pool* untuk mengelola *workers* ini secara dinamis, memastikan tidak ada *blocking call* di *main thread*.

### 2\. Deep Dive: OpenCV Implementation

Analisis visual adalah inti dari EyeLog. Kami tidak menggunakan model Deep Learning yang berat, melainkan pendekatan matematis klasik dari OpenCV yang lebih ringan dan cepat untuk skenario ini.

  * **Background Subtraction (MOG2)**: Kami menggunakan algoritma MOG2 untuk memisahkan objek bergerak (*foreground*) dari latar belakang yang statis. Model latar belakang ini terus diperbarui, dan parameter seperti `history`, `varThreshold`, dan `learningRate` dapat disesuaikan melalui *preset* untuk beradaptasi dengan perubahan pencahayaan atau kondisi lainnya. 
  * **Morphology Operations**: Untuk membersihkan hasil deteksi, operasi morfologi seperti `Erode`, `Dilate`, `Opening`, dan `Closing` diterapkan. [cite: 30] Ini digunakan baik sebagai *pre-processing* untuk membersihkan *noise* pada frame awal, maupun *post-processing* untuk menyempurnakan *mask foreground* sebelum analisis kontur. 
  * **Contour Detection & Analysis**: Setelah mendapatkan *mask* biner dari *foreground*, kami mendeteksi kontur untuk mengidentifikasi setiap objek material. Kontur-kontur kecil disaring, dan kontur yang berdekatan dapat digabungkan untuk mendapatkan representasi area yang akurat, yang kemudian dihitung luasnya.

### 3\. Architectural Choices (Tech Stack Justification)

| Teknologi | Alasan Pemilihan |
| :--- | :--- |
| **Desktop App** | Dipilih karena kebutuhan operasi **lokal dan *realtime***. Arsitektur web akan memperkenalkan latensi (kamera → server → klien) dan sangat tidak andal di lokasi dengan konektivitas internet terbatas.  Akses *hardware* langsung juga jauh lebih superior di lingkungan desktop. |
| **PySide6 (Qt)** | Dipilih karena dukungan **multimedia dan *threading* yang kuat**, komponen UI yang modern dan lengkap, serta performa *rendering* yang lebih tinggi dibandingkan Tkinter.  Kemampuannya untuk berjalan *cross-platform* dengan tampilan native adalah bonus besar.  |
| **SQLite** | Dipilih karena sifatnya yang **ringan, *serverless*, dan *embedded***.  Sempurna untuk aplikasi *single-user* yang membutuhkan persistensi data sederhana (seperti daftar kamera) tanpa overhead dari *database engine* eksternal seperti PostgreSQL. |

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
  * **Coverage Logger Thread**: Menulis data cakupan ke file CSV secara asinkron agar tidak memblokir UI. 
  * **Database Workers Thread**: Menangani semua operasi baca/tulis ke database SQLite.

*(Anda bisa menyisipkan gambar flowchart Anda di sini)*
`![Application Flowchart](https://raw.githubusercontent.com/raisal21/opencv-py-pdu/main/flowchart/Eyelog-Flowchart.svg)`

### How to Contribute

Kontribusi dalam bentuk apapun sangat saya hargai\!

1.  **Fork** repositori ini.
2.  Buat **branch** baru untuk fitur Anda (`git checkout -b feature/AmazingFeature`).
3.  **Commit** perubahan Anda (`git commit -m 'Add some AmazingFeature'`).
4.  **Push** ke branch Anda (`git push origin feature/AmazingFeature`).
5.  Buka **Pull Request**.
