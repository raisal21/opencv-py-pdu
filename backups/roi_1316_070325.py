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
        
        # Warna orange RGB (255, 149, 0)
        self.color = (0, 149, 255)  # BGR format
        
        self.points = []
        self.complete = False
        self.confirmed = False
        self.current_point = None
        self.roi_image = None
        
        cv.namedWindow(window_name)
        cv.setMouseCallback(window_name, self.mouse_callback)

    def reset_selection(self):
        """Reset seleksi ROI"""
        self.points = []
        self.complete = False
        self.confirmed = False
        self.roi_image = None
        self.draw_roi()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_RBUTTONDOWN and not self.confirmed:
            self.reset_selection()
            return
        
        if self.complete and not self.confirmed:
            # Jika ROI sudah lengkap tapi belum dikonfirmasi dengan Enter,
            # hanya tampilkan outline tapi jangan proses input mouse lainnya
            return
        
        if event == cv.EVENT_LBUTTONDOWN:
            # First two points are free
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
        
        elif event == cv.EVENT_MOUSEMOVE:
            # Only update animation when moving and not complete
            if not self.complete:
                self.current_point = (x, y)
                self.draw_roi()
            
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
                # nambahin cv.circle buat titik-titiknya
            
            # When we have all 4 points, connect the last point to first
            if len(self.points) == 4:
                cv.line(img, self.points[3], self.points[0], self.color, 2)
        
        # Draw animation line from last point to current mouse position
        # Hanya untuk titik pertama ke titik kedua
        if self.current_point and len(self.points) == 1 and not self.complete:
            last_point = self.points[-1]
            cv.line(img, last_point, self.current_point, self.color, 2, cv.LINE_AA)
        # harusnya ini bisa diatas code yang diatasnya
            
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

            cv.circle(img, potential_p3, 5, self.color, -1)
            cv.circle(img, potential_p4, 5, self.color, -1)
        
        self.image = img

        if self.complete and not self.confirmed:
            text = "Tekan ENTER untuk konfirmasi atau klik kanan untuk mulai ulang"
            cv.putText(img, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif not self.complete:
            if len(self.points) == 0:
                text = "Klik untuk memilih titik pertama"
            elif len(self.points) == 1:
                text = "Klik untuk memilih titik kedua"
            elif len(self.points) == 2:
                text = "Klik untuk memilih titik ketiga (akan dibuat persegi panjang)"
            cv.putText(img, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Jika ROI sudah lengkap dan gambar ROI sudah diekstrak, tampilkan itu
        if self.complete and self.roi_image is not None:
            cv.imshow(self.window_name, self.roi_image)
        else:
            cv.imshow(self.window_name, img)
    
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
        extracted_roi = cv.warpPerspective(self.original, M, (width, height))

        # Hitung aspek rasio saat ini
        current_ratio = width / height if height > 0 else 0
        target_ratio = 16 / 9

        if current_ratio > target_ratio:
            # Terlalu lebar - tambahkan padding atas dan bawah
            new_width = width
            new_height = int(width / target_ratio)
            pad_top = (new_height - height) // 2
            pad_bottom = new_height - height - pad_top
            pad_left = 0
            pad_right = 0
        else:
            # Terlalu tinggi - tambahkan padding kiri dan kanan
            new_height = height
            new_width = int(height * target_ratio)
            pad_left = (new_width - width) // 2
            pad_right = new_width - width - pad_left
            pad_top = 0
            pad_bottom = 0

        padded_roi = cv.copyMakeBorder(
            extracted_roi,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv.BORDER_CONSTANT,
            value=self.color
        )

        
        self.roi_image = padded_roi
    
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
        cv.namedWindow("ROI Selector")
        cv.imshow("ROI Selector", frame)
        cv.waitKey(1)

        selector = RectangleROISelector("ROI Selector", frame)
        
        while True:
            key = cv.waitKey(20) & 0xFF
            
            if key == 27:  # ESC key
                break
            elif key == 13 and selector.complete:  # ENTER key
                selector.extract_roi()
                selector.confirmed = True
                selector.draw_roi()
        
        roi_points, roi_image = selector.get_roi()
        if roi_points:
            print(f"ROI Points: {roi_points}")
            cv.imshow("Extracted Image", roi_image)
            cv.waitKey(0)

        
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()