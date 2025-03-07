import cv2 as cv
import numpy as np
import math

def calculate_perpendicular_point(p1, p2, p3):
    """Calculate the fourth point to create a rectangle given 3 points,
    where p3 forms a 90-degree angle with p2."""
    # Vector from p2 to p3
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    # Calculate the fourth point
    p4 = (p1[0] + v2[0], p1[1] + v2[1])
    
    return p4

class RectangleROISelector:
    def __init__(self, window_name, image):
        self.window_name = window_name
        self.image = image.copy()
        self.original = image.copy()
        
        # Warna Claude orange RGB (255, 149, 0)
        self.color = (0, 149, 255)  # BGR format untuk OpenCV
        
        self.points = []
        self.complete = False
        self.current_point = None
        self.dragging_idx = -1  # Indeks titik yang sedang di-drag (-1 = tidak ada)
        self.roi_image = None
        self.finalized = False  # Flag untuk menandai ROI sudah dikonfirmasi dengan Enter
        
        # Simpan arah vektor untuk setiap pasangan titik
        self.vectors = []
        
        cv.namedWindow(window_name)
        cv.setMouseCallback(window_name, self.mouse_callback)
    
    def find_nearest_point(self, x, y, threshold=10):
        """Temukan titik terdekat dalam jarak threshold"""
        if not self.points:
            return -1
            
        min_dist = float('inf')
        min_idx = -1
        
        for i, point in enumerate(self.points):
            dist = np.sqrt((point[0] - x)**2 + (point[1] - y)**2)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                min_idx = i
                
        return min_idx
    
    def calculate_vectors(self):
        """Hitung vektor-vektor untuk setiap sisi persegi panjang"""
        if len(self.points) != 4:
            return
        
        self.vectors = []
        for i in range(4):
            next_idx = (i + 1) % 4
            vector = np.array([
                self.points[next_idx][0] - self.points[i][0],
                self.points[next_idx][1] - self.points[i][1]
            ])
            self.vectors.append(vector)
    
    def project_point_to_vector(self, point, start_point, vector):
        """Proyeksikan titik pada vektor dengan awal start_point"""
        # Normalisasi vektor
        v_norm = vector / np.linalg.norm(vector)
        
        # Vektor dari start_point ke point
        point_vector = np.array([point[0] - start_point[0], point[1] - start_point[1]])
        
        # Proyeksi ke vektor arah
        proj_length = np.dot(point_vector, v_norm)
        
        # Titik hasil proyeksi
        projected_point = (
            int(start_point[0] + v_norm[0] * proj_length),
            int(start_point[1] + v_norm[1] * proj_length)
        )
        
        return projected_point
    
    def update_adjacent_points(self, point_idx, new_pos):
        """Update titik-titik bersebelahan ketika satu titik digeser"""
        if len(self.points) != 4 or not self.vectors:
            return
        
        # Indeks titik bersebelahan
        prev_idx = (point_idx - 1) % 4
        next_idx = (point_idx + 1) % 4
        
        # Vektor dari titik_prev ke titik_saat_ini (sebelum digeser)
        v_prev = self.vectors[prev_idx]
        
        # Vektor dari titik_saat_ini ke titik_next (sebelum digeser)
        v_next = self.vectors[point_idx]
        
        # Pergeseran titik
        delta_x = new_pos[0] - self.points[point_idx][0]
        delta_y = new_pos[1] - self.points[point_idx][1]
        
        # Update titik yang di-drag
        self.points[point_idx] = new_pos
        
        # Update titik sebelumnya (mengikuti arah vektor v_prev)
        new_prev = (
            self.points[prev_idx][0] + delta_x,
            self.points[prev_idx][1] + delta_y
        )
        self.points[prev_idx] = self.project_point_to_vector(
            new_prev, 
            self.points[(prev_idx - 1) % 4], 
            -v_prev  # Kebalikan arah vektor asli
        )
        
        # Update titik berikutnya (mengikuti arah vektor v_next)
        new_next = (
            self.points[next_idx][0] + delta_x,
            self.points[next_idx][1] + delta_y
        )
        self.points[next_idx] = self.project_point_to_vector(
            new_next, 
            self.points[(next_idx + 1) % 4], 
            -self.vectors[next_idx]  # Kebalikan arah vektor asli
        )
        
        # Hitung ulang vektor-vektor
        self.calculate_vectors()
    
    def constrain_to_vector(self, start_point, current_pos, vector):
        """Batasi pergerakan ke arah vektor dari titik awal"""
        return self.project_point_to_vector(current_pos, start_point, vector)
    
    def mouse_callback(self, event, x, y, flags, param):
        # Jika ROI sudah difinalisasi, abaikan mouse events
        if self.finalized:
            return
        
        if event == cv.EVENT_LBUTTONDOWN:
            if self.complete:
                # Mode pengeditan: cari titik terdekat untuk di-drag
                near_idx = self.find_nearest_point(x, y)
                if near_idx >= 0:
                    self.dragging_idx = near_idx
                    # Hitung vektor-vektor jika belum dilakukan
                    if not self.vectors:
                        self.calculate_vectors()
            else:
                # Mode normal: tambahkan titik baru
                if len(self.points) < 2:
                    self.points.append((x, y))
                # Third point must form a 90-degree angle
                elif len(self.points) == 2:
                    p1, p2 = self.points
                    
                    # Vector from p2 to p1
                    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
                    # Vector from click to p2
                    v_click = np.array([x - p2[0], y - p2[1]])
                    
                    # Project v_click onto v1 to find the perpendicular component
                    v1_norm = v1 / np.linalg.norm(v1)
                    proj = np.dot(v_click, v1_norm) * v1_norm
                    perp = v_click - proj
                    
                    # Calculate the point that is perpendicular to the line p1-p2
                    p3 = (int(p2[0] + perp[0]), int(p2[1] + perp[1]))
                    self.points.append(p3)
                    
                    # Calculate the fourth point automatically
                    p4 = calculate_perpendicular_point(p1, p2, p3)
                    self.points.append((int(p4[0]), int(p4[1])))
                    
                    self.complete = True
                    self.calculate_vectors()
                    
                    # Tampilkan petunjuk di console
                    print("ROI selesai. Anda dapat: ")
                    print("- Tarik titik sudut untuk mengedit")
                    print("- Tekan ENTER untuk mengakhiri pengeditan dan menampilkan ROI")
                    print("- Tekan ESC untuk batal")
        
        elif event == cv.EVENT_MOUSEMOVE:
            if self.complete and self.dragging_idx >= 0:
                # Mode pengeditan: batas pergerakan ke arah vektor dan update titik bersebelahan
                self.update_adjacent_points(self.dragging_idx, (x, y))
            else:
                # Mode normal: perbarui current_point untuk animasi
                if not self.complete:
                    self.current_point = (x, y)
            
            self.draw_roi()
        
        elif event == cv.EVENT_LBUTTONUP:
            # Selesai drag
            self.dragging_idx = -1
            
        # Update display
        self.draw_roi()
    
    def draw_roi(self):
        img = self.original.copy()
        
        # Draw points
        for point in self.points:
            cv.circle(img, point, 5, self.color, -1)
        
        # Draw lines between confirmed points
        if len(self.points) >= 2:
            for i in range(len(self.points)-1):
                cv.line(img, self.points[i], self.points[i+1], self.color, 2)
            
            # When we have all 4 points, connect the last point to first
            if len(self.points) == 4:
                cv.line(img, self.points[3], self.points[0], self.color, 2)
        
        # Draw animation line from last point to current mouse position
        # Hanya untuk titik pertama ke titik kedua
        if self.current_point and len(self.points) == 1 and not self.complete:
            last_point = self.points[-1]
            cv.line(img, last_point, self.current_point, self.color, 2, cv.LINE_AA)
            
        # If we already have 2 points, show the potential perpendicular line
        if self.current_point and len(self.points) == 2 and not self.complete:
            p1, p2 = self.points
            
            # Vector from p2 to p1
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            # Vector from mouse to p2
            v_mouse = np.array([self.current_point[0] - p2[0], self.current_point[1] - p2[1]])
            
            # Calculate perpendicular
            v1_norm = v1 / np.linalg.norm(v1)
            proj = np.dot(v_mouse, v1_norm) * v1_norm
            perp = v_mouse - proj
            
            # Potential p3 (perpendicular point)
            potential_p3 = (int(p2[0] + perp[0]), int(p2[1] + perp[1]))
            
            # Draw guide line showing perpendicular constraint
            cv.line(img, p2, potential_p3, self.color, 2, cv.LINE_AA)
            
            # Show potential fourth point
            potential_p4 = calculate_perpendicular_point(p1, p2, potential_p3)
            potential_p4 = (int(potential_p4[0]), int(potential_p4[1]))
            
            # Draw remaining sides of potential rectangle
            cv.line(img, potential_p3, potential_p4, self.color, 2, cv.LINE_AA)
            cv.line(img, potential_p4, p1, self.color, 2, cv.LINE_AA)
            
            # Menambahkan circle pada titik potensial 3 dan 4
            cv.circle(img, potential_p3, 5, self.color, -1)
            cv.circle(img, potential_p4, 5, self.color, -1)
        
        # Tampilkan instruksi di gambar jika ROI sudah lengkap tapi belum final
        if self.complete and not self.finalized:
            instruction = "Tarik titik sudut untuk mengubah ukuran. Tekan ENTER untuk selesai."
            cv.putText(img, instruction, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 
                       0.7, self.color, 2, cv.LINE_AA)
            
            # Jika sedang drag, tampilkan ikon "tangan" di dekat kursor
            if self.dragging_idx >= 0:
                cursor_icon = "✋"  # Simbol tangan
                cv.putText(img, cursor_icon, (self.points[self.dragging_idx][0] + 15, 
                                              self.points[self.dragging_idx][1] + 15), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2, cv.LINE_AA)
        
        self.image = img
        
        # Jika ROI sudah difinalisasi dan gambar ROI sudah diekstrak, tampilkan itu
        if self.finalized and self.roi_image is not None:
            cv.imshow(self.window_name, self.roi_image)
        else:
            cv.imshow(self.window_name, img)
    
    def finalize_roi(self):
        """Finalisasi ROI - dipanggil ketika user menekan ENTER"""
        if self.complete:
            self.finalized = True
            self.extract_roi()
            return True
        return False
    
    def extract_roi(self):
        """Ekstrak gambar dari dalam area ROI"""
        if len(self.points) != 4:
            return
        
        # Konversi points ke numpy array untuk transformasi perspektif
        src_pts = np.array(self.points, dtype=np.float32)
        
        # Temukan lebar dan tinggi persegi panjang
        width_1 = np.linalg.norm(np.array(self.points[0]) - np.array(self.points[1]))
        width_2 = np.linalg.norm(np.array(self.points[2]) - np.array(self.points[3]))
        height_1 = np.linalg.norm(np.array(self.points[1]) - np.array(self.points[2]))
        height_2 = np.linalg.norm(np.array(self.points[3]) - np.array(self.points[0]))
        
        width = max(int(width_1), int(width_2))
        height = max(int(height_1), int(height_2))
        
        # Tentukan titik-titik tujuan untuk persegi panjang
        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Hitung matriks transformasi perspektif
        M = cv.getPerspectiveTransform(src_pts, dst_pts)
        
        # Terapkan transformasi untuk mendapatkan gambar ROI yang lurus
        self.roi_image = cv.warpPerspective(self.original, M, (width, height))
    
    def get_roi(self):
        if len(self.points) == 4:
            return self.points, self.roi_image
        return None, None

# Contoh penggunaan
def main():
    cap = cv.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        selector = RectangleROISelector("Select ROI", frame)
        
        while True:
            key = cv.waitKey(1) & 0xFF
            
            if key == 27:  # ESC key
                break
            elif key == 13 and selector.complete:  # Enter key when ROI is complete
                if selector.finalize_roi():  # Finalisasi ROI
                    break
        
        roi_points, roi_image = selector.get_roi()
        if roi_points:
            print(f"ROI Points: {roi_points}")
            
            # Jika ROI berhasil diektrak, tampilkan dalam jendela baru
            if roi_image is not None:
                cv.imshow("Extracted ROI", roi_image)
                cv.waitKey(0)
        
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()