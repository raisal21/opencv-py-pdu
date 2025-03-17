import cv2 as cv
import numpy as np
import argparse
import time


class ForegroundExtractor:
    """
    Kelas yang mengimplementasikan ekstraksi foreground menggunakan algoritma MOG2.
    
    Kelas ini menyediakan:
    1. Inisialisasi dan konfigurasi background subtractor MOG2
    2. Pemrosesan frame untuk ekstraksi mask foreground
    3. Opsi penggunaan frame grayscale untuk optimasi
    4. Operasi morfologi untuk memperbaiki hasil deteksi
    """
    
    def __init__(self, history=500, var_threshold=16, detect_shadows=True, 
                 use_grayscale=False, learning_rate=0.01, 
                 use_morphology=False, kernel_size=7, morph_iterations=2):
        """
        Inisialisasi foreground extractor dengan parameter MOG2.
        
        Args:
            history (int): Jumlah frame yang digunakan untuk membangun model background
            var_threshold (float): Threshold pada jarak Mahalanobis kuadrat untuk menentukan
                                  apakah suatu piksel termasuk background atau tidak
            detect_shadows (bool): Jika True, algoritma mendeteksi bayangan
            use_grayscale (bool): Jika True, frame dikonversi ke grayscale sebelum diproses
            learning_rate (float): Kecepatan adaptasi model background (0-1, atau -1 untuk otomatis)
            use_morphology (bool): Jika True, operasi morfologi closing diterapkan pada mask
            kernel_size (int): Ukuran kernel untuk operasi morfologi
            morph_iterations (int): Jumlah iterasi untuk operasi morfologi
        """
        # Inisialisasi MOG2 background subtractor
        self.bg_subtractor = cv.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        
        # Parameter untuk referensi
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.use_grayscale = use_grayscale
        self.learning_rate = learning_rate
        
        # Parameter morfologi
        self.use_morphology = use_morphology
        self.kernel_size = kernel_size
        self.morph_iterations = morph_iterations
        self.kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE, 
            (self.kernel_size, self.kernel_size)
        )
    
    def extract(self, frame):
        """
        Ekstraksi foreground dari frame input.
        
        Args:
            frame (numpy.ndarray): Frame input
            
        Returns:
            numpy.ndarray: Mask foreground (0=background, 255=foreground)
        """
        # Buat salinan frame untuk memastikan frame asli tidak dimodifikasi
        process_frame = frame.copy()
        
        # Konversi ke grayscale jika diperlukan
        if self.use_grayscale:
            gray = cv.cvtColor(process_frame, cv.COLOR_BGR2GRAY)
            # Konversi kembali ke BGR agar kompatibel dengan MOG2
            process_frame = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        
        # Terapkan background subtraction dengan learning rate yang ditentukan
        fg_mask = self.bg_subtractor.apply(process_frame, learningRate=self.learning_rate)
        
        # Jika shadows dideteksi, konversi nilai bayangan (127) ke hitam (0)
        if self.detect_shadows:
            # Dalam MOG2 dengan detect_shadows=True, bayangan ditandai dengan 127
            binary_mask = cv.threshold(fg_mask, 127, 255, cv.THRESH_BINARY)[1]
        else:
            binary_mask = fg_mask
        
        # Terapkan operasi morfologi jika diaktifkan
        if self.use_morphology:
            # Closing: dilasi diikuti erosi (mengisi lubang)
            binary_mask = cv.morphologyEx(
                binary_mask, 
                cv.MORPH_CLOSE, 
                self.kernel, 
                iterations=self.morph_iterations
            )
        
        return binary_mask
    
    def visualize(self, frame, mask):
        """
        Buat visualisasi hasil ekstraksi foreground.
        
        Args:
            frame (numpy.ndarray): Frame original
            mask (numpy.ndarray): Mask foreground hasil ekstraksi
            
        Returns:
            numpy.ndarray: Visualisasi hasil (hanya bagian foreground)
        """
        # Terapkan mask ke frame asli untuk hanya menampilkan foreground
        return cv.bitwise_and(frame, frame, mask=mask)
    
    def update_parameters(self, history, var_threshold, detect_shadows, use_grayscale,
                         learning_rate, use_morphology, kernel_size, morph_iterations):
        """
        Update parameter untuk background subtractor.
        
        Args:
            history (int): Jumlah frame untuk model background
            var_threshold (float): Threshold sensitivitas deteksi
            detect_shadows (bool): Apakah deteksi bayangan diaktifkan
            use_grayscale (bool): Apakah menggunakan grayscale
            learning_rate (float): Kecepatan adaptasi model background
            use_morphology (bool): Apakah menggunakan operasi morfologi
            kernel_size (int): Ukuran kernel morfologi
            morph_iterations (int): Jumlah iterasi operasi morfologi
        """
        # Update parameter internal
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.use_grayscale = use_grayscale
        self.learning_rate = learning_rate
        self.use_morphology = use_morphology
        
        # Update parameter morfologi jika berubah
        if self.kernel_size != kernel_size:
            self.kernel_size = kernel_size
            self.kernel = cv.getStructuringElement(
                cv.MORPH_ELLIPSE, 
                (self.kernel_size, self.kernel_size)
            )
        
        self.morph_iterations = morph_iterations
        
        # Buat ulang background subtractor jika parameter MOG2 berubah
        if (self.bg_subtractor.getHistory() != history or 
            self.bg_subtractor.getVarThreshold() != var_threshold or
            self.bg_subtractor.getDetectShadows() != detect_shadows):
            
            self.bg_subtractor = cv.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=detect_shadows
            )


def run_parameter_tuning(video_path):
    """
    Alat interaktif untuk mengatur parameter ForegroundExtractor.
    
    Args:
        video_path (str): Path ke file video atau indeks kamera (0 untuk kamera default)
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    # Buat objek video capture
    try:
        if isinstance(video_path, str) and video_path.isdigit():
            cap = cv.VideoCapture(int(video_path))
        else:
            cap = cv.VideoCapture(video_path)
    except:
        print(f"Error: Tidak dapat membuka sumber video {video_path}")
        return False
    
    # Cek apakah video berhasil dibuka
    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka sumber video {video_path}")
        return False
    
    # Buat jendela untuk kontrol dan visualisasi
    main_window = "Foreground Extraction Tuning"
    cv.namedWindow(main_window, cv.WINDOW_NORMAL)
    
    # Buat panel kontrol di sisi kiri
    panel_width = 400
    panel_height = 600
    
    # Buat ForegroundExtractor dengan parameter default
    extractor = ForegroundExtractor(
        history=200,
        var_threshold=25,
        detect_shadows=True,
        use_grayscale=False,
        learning_rate=0.01,
        use_morphology=False,
        kernel_size=7,
        morph_iterations=2
    )
    
    # Buat trackbar untuk parameter
    def nothing(x):
        pass
    
    # Trackbar parameter MOG2 dasar
    cv.createTrackbar('History', main_window, 200, 1000, nothing)
    cv.createTrackbar('Var Threshold', main_window, 25, 100, nothing)
    cv.createTrackbar('Detect Shadows', main_window, 1, 1, nothing)
    cv.createTrackbar('Use Grayscale', main_window, 0, 1, nothing)
    
    # Trackbar parameter tambahan
    cv.createTrackbar('Learning Rate x100', main_window, 1, 100, nothing)  # x100 untuk presisi lebih baik
    cv.createTrackbar('Use Morphology', main_window, 0, 1, nothing)
    cv.createTrackbar('Kernel Size', main_window, 7, 21, nothing)
    cv.createTrackbar('Morph Iterations', main_window, 2, 10, nothing)
    
    # Tambahkan trackbar untuk kontrol video
    cv.createTrackbar('Pause/Play', main_window, 0, 1, nothing)
    
    print("\nForeground Extractor - Parameter Tuning")
    print("----------------------------------------")
    print("Kontrol:")
    print("  - Geser slider untuk mengubah parameter")
    print("  - Gunakan trackbar 'Pause/Play' untuk menghentikan/melanjutkan video")
    print("  - Tekan 's' untuk menyimpan parameter saat ini")
    print("  - Tekan 'q' untuk keluar")
    
    # Loop pemrosesan video
    frame_count = 0
    
    while True:
        # Periksa status pause
        is_paused = cv.getTrackbarPos('Pause/Play', main_window) == 1
        
        # Baca frame jika tidak di-pause
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                # Jika sudah mencapai akhir video, mulai dari awal
                cap.set(cv.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_count += 1
        
        # Dapatkan nilai parameter dari trackbar
        history = cv.getTrackbarPos('History', main_window)
        var_threshold = cv.getTrackbarPos('Var Threshold', main_window)
        detect_shadows = cv.getTrackbarPos('Detect Shadows', main_window) == 1
        use_grayscale = cv.getTrackbarPos('Use Grayscale', main_window) == 1
        
        # Dapatkan nilai parameter tambahan
        learning_rate_val = cv.getTrackbarPos('Learning Rate x100', main_window)
        # Konversi nilai trackbar ke learning rate (0-1)
        learning_rate = learning_rate_val / 100.0 if learning_rate_val > 0 else -1.0
        
        use_morphology = cv.getTrackbarPos('Use Morphology', main_window) == 1
        kernel_size = cv.getTrackbarPos('Kernel Size', main_window)
        # Pastikan kernel size selalu ganjil
        if kernel_size % 2 == 0:
            kernel_size += 1
            cv.setTrackbarPos('Kernel Size', main_window, kernel_size)
            
        morph_iterations = cv.getTrackbarPos('Morph Iterations', main_window)
        
        # Update parameter
        extractor.update_parameters(
            history, var_threshold, detect_shadows, use_grayscale,
            learning_rate, use_morphology, kernel_size, morph_iterations
        )
        
        # Proses frame
        mask = extractor.extract(frame)
        result = extractor.visualize(frame, mask)
        
        # Buat area untuk teks parameter
        h, w = frame.shape[:2]
        
        # Informasi parameter untuk ditampilkan
        param_text = [
            f"Frame: {frame_count}",
            f"History: {history}",
            f"Var Threshold: {var_threshold}",
            f"Detect Shadows: {'Yes' if detect_shadows else 'No'}",
            f"Use Grayscale: {'Yes' if use_grayscale else 'No'}",
            f"Learning Rate: {learning_rate:.2f}" if learning_rate > 0 else "Learning Rate: Auto",
            f"Use Morphology: {'Yes' if use_morphology else 'No'}",
            f"Kernel Size: {kernel_size}",
            f"Morph Iterations: {morph_iterations}",
            f"Coverage: {(np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1]) * 100):.2f}%",
            "",
            "Controls:",
            "s - Save parameters",
            "q - Quit",
            "Spacebar - Pause/Play"
        ]
        
        # Buat panel parameter di sisi kiri
        panel = np.zeros((h, panel_width, 3), dtype=np.uint8)
        
        # Tambahkan teks parameter ke panel
        for i, text in enumerate(param_text):
            cv.putText(
                panel,
                text,
                (10, 30 + i * 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
        
        # Tambahkan visualisasi kernel jika morfologi diaktifkan
        if use_morphology:
            kernel_vis_size = 100
            kernel_vis_y = 30 + len(param_text) * 30 + 20
            
            if kernel_vis_y + kernel_vis_size < h:
                # Buat visualisasi kernel
                kernel_vis = np.zeros((kernel_vis_size, kernel_vis_size), dtype=np.uint8)
                
                # Skalakan kernel untuk visualisasi
                scale_factor = kernel_vis_size / kernel_size
                center = kernel_vis_size // 2
                radius = int(kernel_size * scale_factor / 2)
                
                # Gambar lingkaran untuk kernel ellipse
                cv.circle(kernel_vis, (center, center), radius, 255, -1)
                
                # Tambahkan visualisasi kernel ke panel
                kernel_vis_color = cv.cvtColor(kernel_vis, cv.COLOR_GRAY2BGR)
                panel[kernel_vis_y:kernel_vis_y+kernel_vis_size, 
                      panel_width//2-kernel_vis_size//2:panel_width//2+kernel_vis_size//2] = kernel_vis_color
                
                cv.putText(
                    panel,
                    "Kernel Visualisasi:",
                    (10, kernel_vis_y - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1
                )
        
        # Gabungkan panel dan hasil visualisasi
        display_frame = np.hstack((panel, result))
        
        # Resize untuk display jika terlalu besar
        screen_width, screen_height = 1280, 720
        if display_frame.shape[1] > screen_width or display_frame.shape[0] > screen_height:
            scale = min(screen_width / display_frame.shape[1], screen_height / display_frame.shape[0])
            display_frame = cv.resize(display_frame, (0, 0), fx=scale, fy=scale)
        
        # Tampilkan hasil
        cv.imshow(main_window, display_frame)
        
        # Tangani input keyboard
        key = cv.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):  # Simpan parameter
            print("\nParameter saat ini:")
            print(f"History: {history}")
            print(f"Var Threshold: {var_threshold}")
            print(f"Detect Shadows: {detect_shadows}")
            print(f"Use Grayscale: {use_grayscale}")
            print(f"Learning Rate: {learning_rate}")
            print(f"Use Morphology: {use_morphology}")
            print(f"Kernel Size: {kernel_size}")
            print(f"Morph Iterations: {morph_iterations}")
            
            print("\nKode Python untuk parameter ini:")
            print(f"extractor = ForegroundExtractor(")
            print(f"    history={history},")
            print(f"    var_threshold={var_threshold},")
            print(f"    detect_shadows={detect_shadows},")
            print(f"    use_grayscale={use_grayscale},")
            print(f"    learning_rate={learning_rate},")
            print(f"    use_morphology={use_morphology},")
            print(f"    kernel_size={kernel_size},")
            print(f"    morph_iterations={morph_iterations}")
            print(f")")
        elif key == ord(' '):  # Toggle pause dengan spasi
            new_pause_value = 0 if is_paused else 1
            cv.setTrackbarPos('Pause/Play', main_window, new_pause_value)
    
    # Lepaskan resource
    cap.release()
    cv.destroyAllWindows()
    
    return True


def main():
    """
    Fungsi utama untuk aplikasi standalone.
    """
    parser = argparse.ArgumentParser(description='Foreground Extraction menggunakan MOG2')
    parser.add_argument('video', help='Path ke file video atau indeks kamera (contoh: 0 untuk kamera default)')
    
    args = parser.parse_args()
    
    # Jalankan parameter tuning
    success = run_parameter_tuning(args.video)
    
    if not success:
        print("Operasi gagal. Silakan periksa sumber video Anda.")
    else:
        print("Operasi selesai dengan sukses.")


if __name__ == "__main__":
    main()