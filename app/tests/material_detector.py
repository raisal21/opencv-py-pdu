import cv2 as cv
import numpy as np
import argparse
import csv
from pathlib import Path
import os
from collections import namedtuple, deque


# ----------------------
# PARAMETER FILTER COVERAGE
# ----------------------
WIN_MED = 3     # ukuran jendela median
WIN_MA  = 5     # ukuran jendela moving‑average
THR_REL = 0.25  # ambang spike relatif (25 %)

# State buffer (diinisialisasi kosong, nanti di‐reset di main())
med_buf: deque = deque(maxlen=WIN_MED)
ma_buf:  deque = deque(maxlen=WIN_MA)
prev_cov: float | None = None

# ----------------------
# STRUKTUR DATACLASS EXISTING 
# ----------------------
FrameResult  = namedtuple('FrameResult', ['original', 'mask', 'binary'])
ContourResult = namedtuple('ContourResult', ['mask', 'contours', 'metrics'])

# Background subtraction presets - moved up to be globally accessible
BG_PRESETS = {
    "default": {
        'history': 500,
        'var_threshold': 16,
        'detect_shadows': False,
        'nmixtures': 5,
        'background_ratio': 0.9,
        'learning_rate': 0.01,
        'pre_process': 'erode',
        'pre_kernel_size': 3,
        'pre_iterations': 1,
        'post_process': 'close',
        'post_kernel_size': 5,
        'post_iterations': 2
    },
    "shale-day-clear": {
        'history': 500,
        'pre_process': 'erode',
        'pre_kernel_size': 3,
        'pre_iterations': 1,
        'post_process': 'close',
        'post_kernel_size': 5,
        'post_iterations': 2,
        'var_threshold': 30,
        'detect_shadows': False,
        'learning_rate': 0.003,
        'nmixtures': 4,
        'background_ratio': 0.85
    },
    "shale-day-rainy": {
        'history': 300,
        'pre_process': 'open',
        'pre_kernel_size': 5,
        'pre_iterations': 2,
        'post_process': 'close',
        'post_kernel_size': 7,
        'post_iterations': 2,
        'var_threshold': 18,
        'detect_shadows': False,
        'learning_rate': 0.01,
        'nmixtures': 6,
        'background_ratio': 0.8
    },
    "shale-night": {
        'history': 500,
        'pre_process': 'open',
        'pre_kernel_size': 3,
        'pre_iterations': 1,
        'post_process': 'dilate',
        'post_kernel_size': 5,
        'post_iterations': 1,
        'var_threshold': 15,
        'detect_shadows': False,
        'learning_rate': 0.002,
        'nmixtures': 3,
        'background_ratio': 0.9
    },
    "shale-vibration": {
        'history': 250,
        'learning_rate': 0.003,
        'var_threshold': 25,
        'background_ratio': 0.90,
        'pre_process': 'erode',  
        'pre_kernel_size': 3,
        'pre_iterations': 2,
        'post_process': 'close',
        'post_kernel_size': 5,
        'post_iterations': 1,
        'detect_shadows': False,
        'nmixtures': 7
    },
    "shale-dust": {
        'history': 400,
        'pre_process': 'erode',
        'pre_kernel_size': 3,
        'pre_iterations': 2,
        'post_process': 'open',
        'post_kernel_size': 5,
        'post_iterations': 2,
        'var_threshold': 25,
        'detect_shadows': False,
        'learning_rate': 0.008,
        'nmixtures': 5,
        'background_ratio': 0.85
    }
}

# Contour processing presets - independent from weather conditions
CONTOUR_PRESETS = {
    "standard": {
        'min_contour_area': 100,
        'use_convex_hull': True,
        'merge_overlapping': True,
        'merge_distance': 10,
        'show_contour_index': False,
        'show_contour_area': False
    },
    "detailed": {
        'min_contour_area': 50,
        'use_convex_hull': False,
        'merge_overlapping': False,
        'merge_distance': 5,
        'show_contour_index': True,
        'show_contour_area': True
    },
    "simplified": {
        'min_contour_area': 200,
        'use_convex_hull': True,
        'merge_overlapping': True,
        'merge_distance': 15,
        'show_contour_index': False,
        'show_contour_area': False
    }
}


class ForegroundExtraction:
    """
    Advanced background subtraction implementation using OpenCV's MOG2
    algorithm with configurable parameters and morphological operations.
    """

    def __init__(self, 
                 # MOG2 parameters
                 history=500,
                 var_threshold=20,  # 
                 detect_shadows=True,
                 nmixtures=5,  # Number of Gaussian components per background pixel
                 background_ratio=0.9,  # Background ratio threshold
                 # Learning rate parameters
                 learning_rate=0.01,
                 # Pre-processing morphological operation
                 pre_process=None,  # 'open', 'close', 'dilate', 'erode', or None
                 pre_kernel_size=5,
                 pre_iterations=1,
                 # Post-processing morphological operation 
                 post_process=None,  # 'open', 'close', 'dilate', 'erode', or None
                 post_kernel_size=5,
                 post_iterations=1):
        """
        Initialize the background subtractor with configurable parameters.
        
        Args:
            history: Number of frames to use for background model
            var_threshold: Threshold for foreground/background decision
            detect_shadows: Whether to detect shadows separately
            nmixtures: Number of Gaussian components per background pixel (3-7 typical)
            background_ratio: Threshold that defines whether a component is background or not (0-1)
            learning_rate: How fast the background model is updated (0-1)
            pre_process: Morphological operation before background subtraction
            pre_kernel_size: Kernel size for pre-processing operations
            pre_iterations: Number of iterations for pre-processing
            post_process: Morphological operation after background subtraction
            post_kernel_size: Kernel size for post-processing operations
            post_iterations: Number of iterations for post-processing
        """
        # Initialize parameters
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.nmixtures = nmixtures
        self.background_ratio = background_ratio
        self.learning_rate = learning_rate
        
        # Morphological parameters
        self.pre_process = pre_process
        self.pre_kernel_size = pre_kernel_size
        self.pre_iterations = pre_iterations
        self.post_process = post_process
        self.post_kernel_size = post_kernel_size
        self.post_iterations = post_iterations
        
        # Create BackgroundSubtractorMOG2 object
        self.bg_subtractor = cv.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows
        )
        
        # Set additional parameters using setters
        self.bg_subtractor.setNMixtures(self.nmixtures)
        self.bg_subtractor.setBackgroundRatio(self.background_ratio)
        
        # Create morphological kernels
        self.pre_kernel = np.ones((self.pre_kernel_size, self.pre_kernel_size), np.uint8)
        self.post_kernel = np.ones((self.post_kernel_size, self.post_kernel_size), np.uint8)
    
    @classmethod
    def from_preset(cls, preset_name="default"):
        """
        Create a ForegroundExtraction instance using a predefined preset.
        
        Args:
            preset_name: Name of the preset to use
            
        Returns:
            ForegroundExtraction instance configured with the preset
        """
        if preset_name in BG_PRESETS:
            return cls(**BG_PRESETS[preset_name])
        else:
            print(f"Warning: Preset '{preset_name}' not found, using default")
            return cls(**BG_PRESETS["default"])
    
    def apply_morphological_ops(self, image, morph_type, kernel, iterations):
        """
        Apply morphological operations to an image or mask.
        
        Args:
            image: The image or mask to process
            morph_type: Type of morphological operation ('open', 'close', 'dilate', 'erode', or None)
            kernel: The kernel to use for the operation
            iterations: Number of iterations to apply the operation
            
        Returns:
            Processed image or mask after morphological operations
        """
        if morph_type is None:
            return image
        
        if morph_type == 'open':
            return cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=iterations)
        elif morph_type == 'close':
            return cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=iterations)
        elif morph_type == 'dilate':
            return cv.dilate(image, kernel, iterations=iterations)
        elif morph_type == 'erode':
            return cv.erode(image, kernel, iterations=iterations)
        else:
            return image
    
    def process_frame(self, frame):
        """
        Process a single frame with background subtraction and morphological operations.
        
        Args:
            frame: Input video frame
            
        Returns:
            FrameResult containing original frame, display mask, and binary mask
        """
        # Create a copy of the original frame
        frame_to_process = frame
        
        # Apply pre-processing morphological operations if specified
        if self.pre_process is not None:
            frame_to_process = frame.copy()
            # For pre-processing, convert to grayscale first for better results
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if len(frame.shape) > 2 else frame
            pre_processed = self.apply_morphological_ops(
                gray_frame, 
                self.pre_process, 
                self.pre_kernel, 
                self.pre_iterations
            )
            # Convert back to BGR if needed
            if len(frame.shape) > 2:
                pre_processed = cv.cvtColor(pre_processed, cv.COLOR_GRAY2BGR)
            frame_to_process = pre_processed
        else:
            frame_to_process = frame
            
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame_to_process, learningRate=self.learning_rate)
        
        # Apply post-processing morphological operations if specified
        if self.post_process is not None:
            processed_mask = self.apply_morphological_ops(
                fg_mask, 
                self.post_process, 
                self.post_kernel, 
                self.post_iterations
            )
        else:
            processed_mask = fg_mask
        
        # Convert to strict binary mask (0 or 255)
        # This is important especially when shadow detection is on, as it produces gray values
        binary_mask = cv.threshold(processed_mask, 127, 255, cv.THRESH_BINARY)[1]
        
        # Create a clean display version of the foreground mask (for visualization)
        display_mask = cv.cvtColor(binary_mask, cv.COLOR_GRAY2BGR)    
        
        return FrameResult(original=frame_to_process, mask=display_mask, binary=binary_mask)
    
    def reset_background(self):
        """Reset the background model."""
        # Recreate the background subtractor with the same parameters
        self.bg_subtractor = cv.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows
        )
        
        # Reset additional parameters
        self.bg_subtractor.setNMixtures(self.nmixtures)
        self.bg_subtractor.setBackgroundRatio(self.background_ratio)
        
        print("Background model reset")


class ContourProcessor:
    """
    Class for processing contours from binary masks to analyze material coverage.
    """
    
    def __init__(self, 
                 min_contour_area=100,       # Minimum area to consider a contour valid
                 use_convex_hull=True,        # Whether to use convex hull for more solid representation
                 merge_overlapping=False,     # Whether to merge overlapping contours
                 merge_distance=10,           # Maximum distance to consider contours for merging
                 contour_color=(0, 255, 0),   # Color for drawing contours (BGR)
                 hull_color=(0, 200, 255),    # Color for drawing convex hulls (BGR)
                 text_color=(0, 0, 255),      # Color for text information (BGR)
                 show_contour_index=False,    # Display index number on each contour
                 show_contour_area=False):    # Display area value on each contour
        """
        Initialize the ContourProcessor with specific parameters.
        """
        # Store parameters
        self.min_contour_area = min_contour_area
        self.use_convex_hull = use_convex_hull
        self.merge_overlapping = merge_overlapping
        self.merge_distance = merge_distance
        
        # Visualization parameters
        self.contour_color = contour_color
        self.hull_color = hull_color
        self.text_color = text_color
        self.show_contour_index = show_contour_index
        self.show_contour_area = show_contour_area
    
    @classmethod
    def from_preset(cls, preset_name="standard"):
        """
        Create a ContourProcessor instance using a predefined preset.
        
        Args:
            preset_name: Name of the preset to use
            
        Returns:
            ContourProcessor instance configured with the preset
        """
        if preset_name in CONTOUR_PRESETS:
            return cls(**CONTOUR_PRESETS[preset_name])
        else:
            print(f"Warning: Preset '{preset_name}' not found, using standard")
            return cls(**CONTOUR_PRESETS["standard"])
    
    def process_mask(self, binary_mask):
        """
        Process a binary mask to find, filter, and analyze contours.
    
        Args:
            binary_mask: Binary mask from background subtraction (0 for background, 255 for foreground)
            
        Returns:
            Tuple of (processed_mask, contours, metrics)
            - processed_mask: The binary mask after contour processing
            - contours: List of filtered and processed contours
            - metrics: Dictionary with coverage statistics and measurements
        """
        if len(binary_mask.shape) > 2:
            mask = cv.cvtColor(binary_mask, cv.COLOR_BGR2GRAY)
        else:
            mask = binary_mask
        # Find contours in the binary mask
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
        # Filter contours by minimum area
        filtered_contours = [cnt for cnt in contours if cv.contourArea(cnt) >= self.min_contour_area]
    
        # Process contours (apply convex hull if requested)
        if self.use_convex_hull:
            processed_contours = [cv.convexHull(cnt) for cnt in filtered_contours]
        else:
            processed_contours = filtered_contours
    
        # Merge overlapping contours if requested
        if self.merge_overlapping and len(processed_contours) > 1:
            processed_contours = self._merge_close_contours(processed_contours, self.merge_distance)
    
        # Create a mask with only the processed contours
        result_mask = np.zeros_like(mask)
        cv.drawContours(result_mask, processed_contours, -1, 255, -1)
    
        # Calculate metrics
        metrics = self._calculate_metrics(mask, result_mask, processed_contours)
    
        return ContourResult(mask=result_mask, contours=processed_contours, metrics=metrics)
    
    def _merge_close_contours(self, contours, max_distance):
        """
        Merge contours that are close to each other.
        
        Args:
            contours: List of contours to potentially merge
            max_distance: Maximum distance between contours to consider merging
            
        Returns:
            List of merged contours
        """
        if not contours:
            return []
        
        # First find bounds of all contours to create an appropriately sized mask
        all_points = np.vstack([cnt.reshape(-1, 2) for cnt in contours])
        min_x, min_y = all_points.min(axis=0)
        max_x, max_y = all_points.max(axis=0)
        
        # Add padding for dilation
        padding = max_distance * 2
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = max_x + padding
        max_y = max_y + padding
        
        # Create a mask of all contours
        width = int(max_x - min_x + 1)
        height = int(max_y - min_y + 1)
        if width <= 0 or height <= 0:
            return contours  # Safety check
            
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Adjust contour coordinates for the new mask
        shifted_contours = [cnt - np.array([min_x, min_y]) for cnt in contours]
        cv.drawContours(mask, shifted_contours, -1, 255, -1)
        
        # Dilate the mask to connect close contours
        kernel = np.ones((max_distance, max_distance), np.uint8)
        dilated = cv.dilate(mask, kernel, iterations=1)
        
        # Find contours in the dilated mask
        merged_contours_shifted, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Shift contours back to original coordinates
        merged_contours = [cnt + np.array([min_x, min_y]) for cnt in merged_contours_shifted]
        
        return merged_contours
    
    def _calculate_metrics(self, original_mask, processed_mask, contours):
        """
        Calculate various metrics about the contours and coverage.
    
        Args:
            original_mask: The original binary mask
            processed_mask: The mask after contour processing
            contours: List of processed contours
        
        Returns:
            Dictionary with various metrics
        """
        # Calculate total area of the frame
        total_area = original_mask.shape[0] * original_mask.shape[1]
    
        # Calculate areas
        original_coverage_pixels = cv.countNonZero(original_mask)
        processed_coverage_pixels = cv.countNonZero(processed_mask)
    
        contour_properties = [(cv.contourArea(cnt), cv.arcLength(cnt, True), cv.boundingRect(cnt)) 
                            for cnt in contours]
        
        contour_areas, perimeters, bounding_rects = zip(*contour_properties) if contour_properties else ([], [], [])
        contour_areas = np.array(contour_areas)
        perimeters = np.array(perimeters)

        total_contour_area = sum(contour_areas)
    
        # Calculate percentages
        if total_area > 0:
            original_coverage_percent = (original_coverage_pixels / total_area) * 100
            processed_coverage_percent = (processed_coverage_pixels / total_area) * 100
            contour_coverage_percent = (total_contour_area / total_area) * 100
        else:
            original_coverage_percent = 0
            processed_coverage_percent = 0
            contour_coverage_percent = 0
    
        return {
            'total_pixels': total_area,
            'original_coverage_pixels': original_coverage_pixels,
            'processed_coverage_pixels': processed_coverage_pixels,
            'original_coverage_percent': original_coverage_percent,
            'processed_coverage_percent': processed_coverage_percent,
            'contour_coverage_percent': contour_coverage_percent,
            'contour_count': len(contours),
            'contour_areas': contour_areas,
            'total_contour_area': total_contour_area,
            'perimeters': perimeters,
            'bounding_rects': bounding_rects
        }
    
    def visualize(self, image, contours, metrics, show_metrics=True, scale_factor=1.0):
        """
        Create a visualization of the contours and metrics on the input image.
        
        Args:
            image: Input image to draw visualization on
            contours: List of contours to visualize
            metrics: Dictionary of metrics from process_mask
            show_metrics: Whether to show metrics on the image
            scale_factor: Scale factor for text size (useful for different resolutions)
            
        Returns:
            Visualization image with contours and information
        """
        # Create a copy of the input image
        if contours or show_metrics:
            vis_image = image.copy()
        else:
            return image
        
        # Draw all contours
        cv.drawContours(vis_image, contours, -1, self.contour_color, 2)
        
        # Draw convex hulls if they're different from original contours
        if self.use_convex_hull:
            hulls = [cv.convexHull(cnt) for cnt in contours]
            cv.drawContours(vis_image, hulls, -1, self.hull_color, 1)
        
        # Add contour indices and areas if requested
        for i, cnt in enumerate(contours):
            # Find center of contour for text placement
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw index number
                if self.show_contour_index:
                    cv.putText(vis_image, f"{i}", (cx, cy), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.5 * scale_factor, self.text_color, 2)
                
                # Draw area
                if self.show_contour_area:
                    area = int(cv.contourArea(cnt))
                    cv.putText(vis_image, f"{area}", (cx, cy + 20), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.5 * scale_factor, self.text_color, 2)
        
        return vis_image

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Material Detector dengan Background Subtraction dan Contour Processing')
    
    # Video source
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (camera index or video path). Default: 0 (webcam)')
    
    # Resolution
    parser.add_argument('--width', type=int, default=640,
                        help='Frame width. Default: 640')
    parser.add_argument('--height', type=int, default=480,
                        help='Frame height. Default: 480')
    
    # Presets
    parser.add_argument('--bg-preset', type=str, 
                        choices=list(BG_PRESETS.keys()),
                        default='default',
                        help='Background subtraction preset configuration')
    
    parser.add_argument('--contour-preset', type=str,
                        choices=list(CONTOUR_PRESETS.keys()),
                        default='standard',
                        help='Contour processing preset configuration')
    
    # MOG2 parameters (for custom configuration)
    parser.add_argument('--history', type=int, default=500,
                        help='Number of frames for background model. Default: 500')
    parser.add_argument('--var-threshold', type=float, default=16,
                        help='Threshold for foreground/background decision. Default: 16')
    parser.add_argument('--detect-shadows', action='store_true', default=True,
                        help='Whether to detect shadows')
    parser.add_argument('--no-detect-shadows', action='store_false', dest='detect_shadows',
                        help='Disable shadow detection')
    parser.add_argument('--nmixtures', type=int, default=5,
                        help='Number of Gaussian components per background pixel (3-7 typical). Default: 5')
    parser.add_argument('--background-ratio', type=float, default=0.9,
                        help='Threshold for background components (0-1). Default: 0.9')
    
    # Learning rate
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate for background model update (0-1). Default: 0.01')
    
    # Morphological operations
    parser.add_argument('--pre-process', type=str, default=None, 
                        choices=['open', 'close', 'dilate', 'erode', None],
                        help='Morphological operation BEFORE background subtraction. Default: None')
    parser.add_argument('--pre-kernel-size', type=int, default=5,
                        help='Kernel size for pre-processing operations. Default: 5')
    parser.add_argument('--pre-iterations', type=int, default=1,
                        help='Number of iterations for pre-processing operations. Default: 1')
    
    parser.add_argument('--post-process', type=str, default=None, 
                        choices=['open', 'close', 'dilate', 'erode', None],
                        help='Morphological operation AFTER background subtraction. Default: None')
    parser.add_argument('--post-kernel-size', type=int, default=5,
                        help='Kernel size for post-processing operations. Default: 5')
    parser.add_argument('--post-iterations', type=int, default=1,
                        help='Number of iterations for post-processing operations. Default: 1')
    
    # ContourProcessor arguments
    parser.add_argument('--min-contour-area', type=int, default=100,
                        help='Minimum contour area to consider valid in pixels. Default: 100')
    parser.add_argument('--use-convex-hull', action='store_true', default=True,
                        help='Use convex hull for contours')
    parser.add_argument('--no-convex-hull', action='store_false', dest='use_convex_hull',
                        help='Disable convex hull for contours')
    parser.add_argument('--merge-overlapping', action='store_true', default=False,
                        help='Merge overlapping contours')
    parser.add_argument('--merge-distance', type=int, default=10,
                        help='Distance threshold for merging contours. Default: 10')
    parser.add_argument('--show-contour-index', action='store_true', default=False,
                        help='Show contour indices on visualization')
    parser.add_argument('--show-contour-area', action='store_true', default=False,
                        help='Show contour areas on visualization')

    parser.add_argument('--save-csv', type=Path, 
                        help='Simpan hasil ke CSV (opsional)')
    parser.add_argument('--save-video', type=Path, 
                        help='Simpan video hasil (opsional)')
    parser.add_argument('--max-frames', type=int, 
                        help='Batasi jumlah frame yang diproses (opsional)')  

    # Display options
    parser.add_argument('--hide-original', action='store_true',
                        help='Hide original frame')
    parser.add_argument('--hide-mask', action='store_true',
                        help='Hide foreground mask')
    parser.add_argument('--hide-analysis', action='store_true',
                        help='Hide contour analysis visualization')
    parser.add_argument('--display-scale', type=float, default=1.0,
                    help='Scaling factor for display window. Default: 1.0 (no scaling)')
    parser.add_argument('--display-ratio', type=float, default=2.35,
                    help='Aspect ratio (W/H) for display output. Default: 2.35 (cinematic)')
    
    args = parser.parse_args()
    
    return args

def enforce_aspect_ratio(frame, target_ratio=2.35):
    """
    Crop frame to desired aspect ratio (e.g. 2.35:1).
    Keeps center crop only.
    """
    h, w = frame.shape[:2]
    current_ratio = w / h

    if abs(current_ratio - target_ratio) < 0.01:
        return frame  # Already close to desired

    if current_ratio > target_ratio:
        # Too wide -> crop width
        new_w = int(h * target_ratio)
        start_x = (w - new_w) // 2
        return frame[:, start_x:start_x + new_w]
    else:
        # Too tall -> crop height
        new_h = int(w / target_ratio)
        start_y = (h - new_h) // 2
        return frame[start_y:start_y + new_h, :]

# ----------------------
# FUNGSI BANTUAN FILTER
# ----------------------

def filter_coverage(raw_cov: float) -> tuple[float, bool]:
    """Terapkan median → moving‑average → threshold gate.

    Args:
        raw_cov: coverage% mentah dari ContourProcessor.

    Returns:
        (coverage_sah, spike_flag)
    """
    global prev_cov  # gunakan state global sederhana (cukup untuk 1 stream)

    # 1) Median filter
    med_buf.append(raw_cov)
    median_cov = np.median(med_buf)

    # 2) Moving‑average
    ma_buf.append(median_cov)
    ma_cov = float(sum(ma_buf) / len(ma_buf))

    # 3) Threshold‑based gate
    if prev_cov is None or abs(ma_cov - prev_cov) <= THR_REL * prev_cov:
        coverage = ma_cov
        prev_cov = ma_cov
        spike = False
    else:
        coverage = prev_cov  # tahan nilai lama
        spike = True
    return coverage, spike


# ----------------------
# MAIN LOOP (DIUBAH DI BAGIAN FILTER COVERAGE SAJA)
# ----------------------

def main():
    args = parse_arguments()
    cap = cv.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka video: {args.source}")

    if args.width: cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height: cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    fg = ForegroundExtraction(**BG_PRESETS[args.bg_preset])
    ct = ContourProcessor(**CONTOUR_PRESETS[args.contour_preset])

    writer = None
    if args.save_video:
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv.CAP_PROP_FPS) or 25
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        writer = cv.VideoWriter(str(args.save_video), fourcc, fps, (w, h))

    csv_writer = None
    if args.save_csv:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        f_csv = open(args.save_csv, "w", newline="")
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(["frame_number", "timestamp_ms", "coverage_percent"])

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (args.max_frames and frame_idx >= args.max_frames):
            break

        ts_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))

        fg_result = fg.process_frame(frame)
        contour_result = ct.process_mask(fg_result.binary)
        coverage = contour_result.metrics.get("contour_coverage_percent", 0.0)

        if csv_writer:
            csv_writer.writerow([frame_idx, ts_ms, f"{coverage:.2f}"])

        disp = ct.visualize(frame, contour_result.contours, {})

        cv.putText(disp, f"Coverage: {coverage:.1f}%", (10, 30),
                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if writer:
            writer.write(disp)
        else:
            if not args.hide_analysis:
                disp = enforce_aspect_ratio(disp, target_ratio=2.35)

                # Scale preview if requested
                if args.display_scale != 1.0:
                    disp = cv.resize(disp, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv.INTER_AREA)

                cv.imshow("Preview", disp)
                key = cv.waitKey(1) & 0xFF   # <-- WAJIB untuk merender window
                if key == ord('q'):
                    break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    if csv_writer:
        f_csv.close()
    cv.destroyAllWindows()
    cv.waitKey(1)

if __name__ == "__main__":
    main()