import cv2 as cv
import numpy as np


class ROISelector:
    """
    Interactive ROI selector that creates a rectangular selection with perpendicular constraints.
    
    The selection process:
    1. User selects first two points freely
    2. Third point selection is constrained to create a 90-degree angle
    3. Fourth point is calculated automatically to complete the rectangle
    4. User confirms selection with Enter or resets with right-click
    """
    
    # Constants
    COLOR_UI = (0, 149, 255)  # Orange (BGR format)
    COLOR_TEXT = (0, 0, 255)  
    POINT_RADIUS = 5
    LINE_THICKNESS = 2
    
    # Selection state constants
    STATE_POINT1 = 0
    STATE_POINT2 = 1
    STATE_POINT3 = 2
    STATE_COMPLETE = 3
    STATE_CONFIRMED = 4
    
    def __init__(self, window_name, image, use_opencv_window=True):
        """Initialize the ROI selector with an image and window name."""
        # Store inputs
        self.window_name = window_name
        self.original_image = image.copy()
        
        # Initialize state
        self.points = []
        self.current_point = None
        self.roi_image = None
        self.selection_state = self.STATE_POINT1
        self.use_opencv_window = use_opencv_window
        self.callback = None
        self._v1_unit = None
        
        # Set up UI
        if self.use_opencv_window:
            cv.namedWindow(window_name)
            cv.setMouseCallback(window_name, self._mouse_callback)

        # Initial display
        self._update_display()
    
    def set_callback(self, callback_func):
        """Set a callback function to be called on selection confirmation."""
        self.callback = callback_func

    def reset(self):
        """Reset the selection process."""
        self.points = []
        self.roi_image = None
        self.selection_state = self.STATE_POINT1
        self._update_display()
    
    def is_complete(self):
        """Check if ROI selection is complete."""
        return self.selection_state >= self.STATE_COMPLETE
    
    def is_confirmed(self):
        """Check if ROI selection is confirmed."""
        return self.selection_state == self.STATE_CONFIRMED
    
    def confirm_selection(self):
        """Confirm and process the selected ROI."""
        if self.is_complete() and not self.is_confirmed():
            self.selection_state = self.STATE_CONFIRMED
            self._extract_roi()
            self._update_display()
    
    def get_roi(self):
        """Return the selected points and extracted ROI image."""
        if len(self.points) == 4:
            return self.points, self.roi_image
        return None, None
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection."""
        # Handle right-click reset
        if event == cv.EVENT_RBUTTONDOWN and not self.is_confirmed():
            self.reset()
            return
        
        # Ignore mouse events if selection is already complete but not confirmed
        if self.is_complete() and not self.is_confirmed():
            return
            
        # Handle point selection
        if event == cv.EVENT_LBUTTONDOWN:
            self._handle_click(x, y)
        
        # Track mouse position for preview
        elif event == cv.EVENT_MOUSEMOVE:
            if not self.is_complete():
                self.current_point = (x, y)
                self._update_display()
    
    def _handle_click(self, x, y):
        """Process mouse click based on current state."""
        if self.selection_state == self.STATE_POINT1:
            # First point - add directly
            self.points.append((x, y))
            self.selection_state = self.STATE_POINT2
        
        elif self.selection_state == self.STATE_POINT2:
            # Second point - add directly
            self.points.append((x, y))
            p1 = self.points[0]           # titik‑1
            v1 = np.array([p1[0] - x, p1[1] - y], dtype=float)
            norm = np.hypot(v1[0], v1[1])   # lebih cepat drpd np.linalg.norm utk 2D
            self._v1_unit = v1 / norm if norm else v1
            self.selection_state = self.STATE_POINT3
        
        elif self.selection_state == self.STATE_POINT3:
            # Third point - apply perpendicular constraint
            self._add_constrained_point(x, y)
            self.selection_state = self.STATE_COMPLETE
        
        self._update_display()
    
    def _add_constrained_point(self, x, y):
        """Add third point with perpendicular constraint and calculate fourth point."""
        p1, p2 = self.points
        v1_norm = self._v1_unit 
        
        # Calculate vector from p2 to clicked point
        v_click = np.array([x - p2[0], y - p2[1]])
        
        # Project v_click onto v1 to find perpendicular component
        proj = np.dot(v_click, v1_norm) * v1_norm
        perp = v_click - proj
        
        # Add the perpendicular point
        p3 = (int(p2[0] + perp[0]), int(p2[1] + perp[1]))
        self.points.append(p3)
        
        # Calculate and add the fourth point
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        p4 = (int(p1[0] + v2[0]), int(p1[1] + v2[1]))
        self.points.append(p4)
    
    def _calculate_potential_rectangle(self):
        """Calculate potential rectangle when two points are selected."""
        if len(self.points) != 2 or self.current_point is None:
            return None
        
        p1, p2 = self.points
        
        # Vector calculations
        v_mouse = np.array([self.current_point[0] - p2[0], self.current_point[1] - p2[1]])
        
        # Find perpendicular component
        v1_norm = self._v1_unit
        proj = np.dot(v_mouse, v1_norm) * v1_norm
        perp = v_mouse - proj
        
        # Calculate potential points
        potential_p3 = (int(p2[0] + perp[0]), int(p2[1] + perp[1]))
        v2 = np.array([potential_p3[0] - p2[0], potential_p3[1] - p2[1]])
        potential_p4 = (int(p1[0] + v2[0]), int(p1[1] + v2[1]))
        
        return [potential_p3, potential_p4]
    
    def _update_display(self):
        """Update the display with current selection state."""
        # Use the extracted ROI if confirmed
        if self.is_confirmed() and self.roi_image is not None:
            display_image = self.roi_image.copy()
        else:
            display_image = self.original_image.copy()
            self._draw_selection(display_image)
            self._add_instructions(display_image)

        if self.use_opencv_window:
            cv.imshow(self.window_name, display_image)
        
        if self.callback:
            self.callback(display_image)
        
        return display_image
    
    def _draw_selection(self, image):
        """Draw the current selection state on the image."""
        # Draw confirmed points
        for point in self.points:
            cv.circle(image, point, self.POINT_RADIUS, self.COLOR_UI, -1)
        
        # Draw lines between confirmed points
        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                cv.line(image, self.points[i], self.points[i+1], 
                        self.COLOR_UI, self.LINE_THICKNESS)
            
            if len(self.points) == 4:
                cv.line(image, self.points[3], self.points[0], 
                        self.COLOR_UI, self.LINE_THICKNESS)
        
        # Draw preview line from last point to cursor
        if len(self.points) == 1 and self.current_point:
            cv.line(image, self.points[0], self.current_point, 
                    self.COLOR_UI, self.LINE_THICKNESS, cv.LINE_AA)
        
        # Draw potential rectangle preview
        if len(self.points) == 2 and self.current_point:
            potential_points = self._calculate_potential_rectangle()
            if potential_points:
                p1, p2 = self.points
                p3, p4 = potential_points
                
                # Draw perpendicular guide and rectangle
                cv.line(image, p2, p3, self.COLOR_UI, self.LINE_THICKNESS, cv.LINE_AA)
                cv.line(image, p3, p4, self.COLOR_UI, self.LINE_THICKNESS, cv.LINE_AA)
                cv.line(image, p4, p1, self.COLOR_UI, self.LINE_THICKNESS, cv.LINE_AA)
                
                # Draw potential points
                cv.circle(image, p3, self.POINT_RADIUS, self.COLOR_UI, -1)
                cv.circle(image, p4, self.POINT_RADIUS, self.COLOR_UI, -1)
    
    def _add_instructions(self, image):
        """Add instruction text based on current state."""
        if self.is_complete() and not self.is_confirmed():
            text = "Tekan ENTER untuk konfirmasi atau klik kanan untuk mulai ulang"
        elif self.selection_state == self.STATE_POINT1:
            text = "Klik untuk memilih titik pertama"
        elif self.selection_state == self.STATE_POINT2:
            text = "Klik untuk memilih titik kedua"
        elif self.selection_state == self.STATE_POINT3:
            text = "Klik untuk memilih titik ketiga (akan dibuat persegi panjang)"
        else:
            return
            
        cv.putText(image, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 
                  0.7, self.COLOR_TEXT, 2)
    
    def _extract_roi(self):
        """Extract and process the ROI from the selected points."""
        if len(self.points) != 4:
            return
        
        # Convert points to numpy array for perspective transform
        src_pts = np.array(self.points, dtype=np.float32)
        
        # Calculate rectangle dimensions
        width_1 = np.hypot(*(np.array(self.points[0]) - np.array(self.points[1])))
        width_2 = np.hypot(*(np.array(self.points[2]) - np.array(self.points[3])))
        height_1 = np.hypot(*(np.array(self.points[1]) - np.array(self.points[2])))
        height_2 = np.hypot(*(np.array(self.points[3]) - np.array(self.points[0])))
        
        width = max(int(width_1), int(width_2))
        height = max(int(height_1), int(height_2))
        
        # Define destination points for rectangle
        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Apply perspective transform
        M = cv.getPerspectiveTransform(src_pts, dst_pts)
        extracted_roi = cv.warpPerspective(self.original_image, M, (width, height))
        
        # Apply padding to maintain 16:9 aspect ratio
        self.roi_image = self._apply_aspect_ratio_padding(extracted_roi)
    
    def _apply_aspect_ratio_padding(self, image, target_ratio=16/9):
        """Apply padding to maintain target aspect ratio."""
        height, width = image.shape[:2]
        current_ratio = width / height if height > 0 else 0
        
        if current_ratio > target_ratio:
            # Too wide - add padding to top and bottom
            new_width = width
            new_height = int(width / target_ratio)
            pad_top = (new_height - height) // 2
            pad_bottom = new_height - height - pad_top
            pad_left = 0
            pad_right = 0
        else:
            # Too tall - add padding to left and right
            new_height = height
            new_width = int(height * target_ratio)
            pad_left = (new_width - width) // 2
            pad_right = new_width - width - pad_left
            pad_top = 0
            pad_bottom = 0
        
        # Apply padding with consistent color
        return cv.copyMakeBorder(
            image,
            pad_top, pad_bottom, pad_left, pad_right,
            cv.BORDER_CONSTANT,
            value=self.COLOR_UI
        )
    
# Create Standalone Testing Module
def run_roi_selector_standalone():
    """Run the ROI selector as a standalone application."""
    # Capture a single frame from camera
    cap = cv.VideoCapture(0)
    
    # Optimize camera settings
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to capture image from camera")
        return None, None
    
    # Display initial frame immediately for better responsiveness
    cv.namedWindow("ROI Selector")
    cv.imshow("ROI Selector", frame)
    cv.waitKey(1)
    
    # Create selector
    selector = ROISelector("ROI Selector", frame)
    
    # Main loop
    while True:
        key = cv.waitKey(20) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == 13 and selector.is_complete():  # ENTER
            selector.confirm_selection()
            
            # Display extracted ROI in separate window once confirmed
            roi_points, roi_image = selector.get_roi()
            if roi_points and roi_image is not None:
                cv.imshow("Extracted ROI", roi_image)
        
    
    cv.destroyAllWindows()
    return None, None

# Create Export Based Module
def select_roi_from_frame(frame, window_name="ROI Selector"):
    """
    Select ROI from a provided frame
    
    Args:
        frame: The input image/frame
        window_name: Name for the selection window
        
    Returns:
        tuple: (roi_points, roi_image) where roi_points are the corner coordinates
              and roi_image is the extracted and transformed ROI
    """
    # Display initial frame
    cv.namedWindow(window_name)
    cv.imshow(window_name, frame)
    cv.waitKey(1)
    
    # Create selector
    selector = ROISelector(window_name, frame)
    
    # Main loop
    while True:
        key = cv.waitKey(20) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == 13 and selector.is_complete():  # ENTER
            selector.confirm_selection()
            
            # Get ROI
            roi_points, roi_image = selector.get_roi()
            if roi_points and roi_image is not None:
                cv.destroyWindow(window_name)
                return roi_points, roi_image

    cv.destroyWindow(window_name)
    return None, None

# Modify the main execution part so the file works as both a module and a script
if __name__ == "__main__":
    run_roi_selector_standalone()