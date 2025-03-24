import cv2 as cv
import numpy as np
import argparse

class ForegroundExtraction:
    """
    Advanced background subtraction implementation using OpenCV's MOG2 algorithm
    with configurable parameters and morphological operations.
    """
    
    def __init__(self, 
                 # MOG2 parameters
                 history=500,
                 var_threshold=20, # 
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
                 post_iterations=1,
                 # Display parameters
                 show_original=True,
                 show_mask=True,
                 show_result=True):
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
            show_original: Whether to show the original frame
            show_mask: Whether to show the foreground mask
            show_result: Whether to show the result with foreground highlighted
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
        
        # Display parameters
        self.show_original = show_original
        self.show_mask = show_mask
        self.show_result = show_result
        
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
        
        # Window names
        self.window_names = []
        if self.show_original:
            self.window_names.append("Original")
        if self.show_mask:
            self.window_names.append("Foreground Mask")
        if self.show_result:
            self.window_names.append("Result")
    
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
            Tuple of (original frame, foreground mask, result frame)
        """
        # Create a copy of the original frame
        original = frame.copy()
        
        # Apply pre-processing morphological operations if specified
        if self.pre_process is not None:
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
        
        # Apply the binary mask to the original frame for the final result
        result = cv.bitwise_and(original, original, mask=binary_mask)
        
        return original, display_mask, result
    
    def run(self, source=0, width=640, height=480):
        """
        Run background subtraction on a video source.
        
        Args:
            source: Camera index or video file path
            width: Desired width for the frames
            height: Desired height for the frames
        """
        # Initialize video capture
        cap = cv.VideoCapture(source)
        
        # Set camera resolution if using webcam
        if isinstance(source, int):
            cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        
        # Check if video source is opened
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        
        # Create windows
        for name in self.window_names:
            cv.namedWindow(name, cv.WINDOW_NORMAL)
        
        # Main loop
        while True:
            # Read a frame
            ret, frame = cap.read()
            
            # Break if frame reading failed (end of video)
            if not ret:
                break
            
            # Process the frame
            original, fg_mask, result = self.process_frame(frame)
            
            # Display the frames
            if self.show_original:
                cv.imshow("Original", original)
            if self.show_mask:
                cv.imshow("Foreground Mask", fg_mask)
            if self.show_result:
                cv.imshow("Result", result)
            
            # Exit on 'q' key press
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the video capture and close windows
        cap.release()
        cv.destroyAllWindows()
    
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
    Designed to work with output from ForegroundExtraction class and ROI selector.
    
    Focuses on analyzing all detected material contours, calculating coverage metrics,
    and providing visualization tools for analysis.
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
        
        Args:
            min_contour_area: Minimum area (in pixels) to consider a contour valid
            use_convex_hull: Whether to apply convex hull to found contours
            merge_overlapping: Whether to merge contours that are close to each other
            merge_distance: Maximum distance for merging contours (if merge_overlapping is True)
            contour_color: BGR color for drawing contours
            hull_color: BGR color for drawing convex hulls
            text_color: BGR color for drawing text information
            show_contour_index: Whether to display contour index on visualization
            show_contour_area: Whether to display contour area on visualization
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
    
    def process_mask(self, binary_mask, roi_contour=None):
        """
        Process a binary mask to find, filter, and analyze contours.
        
        Args:
            binary_mask: Binary mask from background subtraction (0 for background, 255 for foreground)
            roi_contour: Optional contour defining the region of interest
            
        Returns:
            Tuple of (processed_mask, contours, metrics)
            - processed_mask: The binary mask after contour processing
            - contours: List of filtered and processed contours
            - metrics: Dictionary with coverage statistics and measurements
        """
        # Make a copy of the binary mask and ensure it's binary (0 or 255)
        mask = binary_mask.copy()
        if len(mask.shape) > 2:
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
        
        # Apply ROI mask if provided
        roi_mask = None
        if roi_contour is not None:
            roi_mask = np.zeros_like(mask)
            cv.drawContours(roi_mask, [roi_contour], 0, 255, -1)
            mask = cv.bitwise_and(mask, mask, mask=roi_mask)
        
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
        metrics = self._calculate_metrics(mask, result_mask, processed_contours, roi_mask)
        
        return result_mask, processed_contours, metrics
    
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
    
    def _calculate_metrics(self, original_mask, processed_mask, contours, roi_mask=None):
        """
        Calculate various metrics about the contours and coverage.
        
        Args:
            original_mask: The original binary mask
            processed_mask: The mask after contour processing
            contours: List of processed contours
            roi_mask: Optional ROI mask for calculating percentages
            
        Returns:
            Dictionary with various metrics
        """
        # Calculate total area (of ROI or entire image)
        if roi_mask is not None:
            total_area = cv.countNonZero(roi_mask)
        else:
            total_area = original_mask.shape[0] * original_mask.shape[1]
        
        # Calculate areas
        original_coverage_pixels = cv.countNonZero(original_mask)
        processed_coverage_pixels = cv.countNonZero(processed_mask)
        
        # Calculate individual contour metrics
        contour_areas = [cv.contourArea(cnt) for cnt in contours]
        total_contour_area = sum(contour_areas)
        
        # Calculate perimeters
        perimeters = [cv.arcLength(cnt, True) for cnt in contours]
        
        # Calculate bounding rectangles
        bounding_rects = [cv.boundingRect(cnt) for cnt in contours]
        
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
        vis_image = image.copy()
        
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
        
        # Add summary metrics
        if show_metrics:
            metrics_text = [
                f"Coverage: {metrics['processed_coverage_percent']:.2f}%",
                f"Contours: {metrics['contour_count']}",
                f"Total Area: {metrics['total_contour_area']} px"
            ]
            
            y_pos = 30
            for text in metrics_text:
                cv.putText(vis_image, text, (10, y_pos), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7 * scale_factor, self.text_color, 2)
                y_pos += 30
        
        return vis_image
    
    @staticmethod
    def create_preset(preset_name="default"):
        """
        Create a ContourProcessor with preset parameters for specific material types.
        
        Args:
            preset_name: Name of the preset ('default', 'liquid', 'solid')
            
        Returns:
            Configured ContourProcessor instance
        """
        if preset_name == "liquid":
            # For liquid materials - less strict with small contours, use convex hull
            return ContourProcessor(
                min_contour_area=50,
                use_convex_hull=True,
                merge_overlapping=True,
                merge_distance=15,
                contour_color=(0, 255, 255),  # Yellow for liquids
            )
        elif preset_name == "solid":
            # For solid materials (rocks) - more strict filtering, less merging
            return ContourProcessor(
                min_contour_area=200,
                use_convex_hull=True,
                merge_overlapping=False,
                contour_color=(0, 0, 255),  # Red for solids
            )
        else:  # default
            return ContourProcessor(
                min_contour_area=100,
                use_convex_hull=True,
                merge_overlapping=False,
            )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Advanced Background Subtraction')
    
    # Video source
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (camera index or video path). Default: 0 (webcam)')
    
    # MOG2 parameters
    parser.add_argument('--history', type=int, default=500,
                        help='Number of frames to use for background model. Default: 500')
    parser.add_argument('--var-threshold', type=float, default=16,
                        help='Threshold for foreground/background decision. Default: 16')
    parser.add_argument('--detect-shadows', type=bool, default=True,
                        help='Whether to detect shadows. Default: True')
    parser.add_argument('--nmixtures', type=int, default=5,
                        help='Number of Gaussian components per background pixel (3-7 typical). Default: 5')
    parser.add_argument('--background-ratio', type=float, default=0.9,
                        help='Threshold defining whether a component is background (0-1). Default: 0.9')
    
    # Learning rate
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate for background model update (0-1). Default: 0.01')
    
    # Pre-processing morphological operations
    parser.add_argument('--pre-process', type=str, default=None, choices=['open', 'close', 'dilate', 'erode', None],
                        help='Type of morphological operation applied BEFORE background subtraction. Default: None')
    parser.add_argument('--pre-kernel-size', type=int, default=5,
                        help='Size of kernel for pre-processing operations. Default: 5')
    parser.add_argument('--pre-iterations', type=int, default=1,
                        help='Number of iterations for pre-processing operations. Default: 1')
                        
    # Post-processing morphological operations
    parser.add_argument('--post-process', type=str, default=None, choices=['open', 'close', 'dilate', 'erode', None],
                        help='Type of morphological operation applied AFTER background subtraction. Default: None')
    parser.add_argument('--post-kernel-size', type=int, default=5,
                        help='Size of kernel for post-processing operations. Default: 5')
    parser.add_argument('--post-iterations', type=int, default=1,
                        help='Number of iterations for post-processing operations. Default: 1')
    
    # Display options
    parser.add_argument('--hide-original', action='store_true',
                        help='Hide original frame')
    parser.add_argument('--hide-mask', action='store_true',
                        help='Hide foreground mask')
    parser.add_argument('--hide-result', action='store_true',
                        help='Hide result frame')
    
    # Resolution
    parser.add_argument('--width', type=int, default=640,
                        help='Frame width. Default: 640')
    parser.add_argument('--height', type=int, default=480,
                        help='Frame height. Default: 480')
    
    # Recommended presets
    parser.add_argument('--preset', type=str, choices=['none', 'shale-day-clear', 'shale-day-rainy', 'shale-night', 'shale-vibration', 'shale-dust'],
                        help='Use a recommended preset configuration')
    
    args = parser.parse_args()
    
    # Apply presets if specified
    if hasattr(args, 'preset') and args.preset:
        if args.preset == 'shale-day-clear':
        # Kondisi siang hari cerah - kontras tinggi, bayangan jelas
            args.pre_process = 'erode'
            args.pre_kernel_size = 3
            args.pre_iterations = 1
            args.post_process = 'close'
            args.post_kernel_size = 5
            args.post_iterations = 2
            args.var_threshold = 30  # Nilai lebih tinggi karena kontras baik
            args.detect_shadows = False  # Matikan deteksi bayangan karena bisa mengacaukan deteksi material
            args.learning_rate = 0.003  # Learning rate rendah untuk stabilitas
            args.nmixtures = 4  # Lebih sedikit Gaussian karena kondisi stabil
            args.background_ratio = 0.85

        elif args.preset == 'shale-day-rainy':
            # Kondisi hujan - kontras rendah, banyak pergerakan air
            args.pre_process = 'open'
            args.pre_kernel_size = 5  # Kernel lebih besar untuk mengatasi noise hujan
            args.pre_iterations = 2
            args.post_process = 'close'
            args.post_kernel_size = 7
            args.post_iterations = 2
            args.var_threshold = 18  # Lebih rendah untuk mendeteksi objek dengan kontras rendah
            args.detect_shadows = False
            args.learning_rate = 0.01  # Learning rate lebih tinggi untuk adaptasi cepat terhadap perubahan
            args.nmixtures = 6  # Lebih banyak Gaussian untuk menangani variasi akibat hujan
            args.history = 300  # History lebih pendek untuk adaptasi cepat
            args.background_ratio = 0.8

        elif args.preset == 'shale-night':
            # Kondisi malam - pencahayaan buatan, kontras tinggi, bayangan tajam
            args.pre_process = 'open'
            args.pre_kernel_size = 3
            args.pre_iterations = 1
            args.post_process = 'dilate'  # Dilasi untuk memperluas area deteksi dalam kondisi cahaya kurang
            args.post_kernel_size = 5
            args.post_iterations = 1
            args.var_threshold = 15  # Lebih rendah karena kontras mungkin lebih rendah
            args.detect_shadows = False
            args.learning_rate = 0.002  # Sangat rendah untuk stabilitas dalam pencahayaan konsisten
            args.nmixtures = 3  # Lebih sedikit karena pencahayaan malam cenderung stabil
            args.background_ratio = 0.9

        elif args.preset == 'shale-vibration':
            # Kondisi dengan banyak getaran peralatan
            args.pre_process = 'open'
            args.pre_kernel_size = 5
            args.pre_iterations = 1
            args.post_process = 'close'
            args.post_kernel_size = 9  # Kernel lebih besar untuk mengatasi fragmentasi deteksi akibat getaran
            args.post_iterations = 3
            args.var_threshold = 20
            args.detect_shadows = False
            args.learning_rate = 0.015  # Lebih tinggi untuk adaptasi cepat
            args.nmixtures = 7  # Lebih banyak Gaussian untuk menangani variasi posisi akibat getaran
            args.history = 200
            args.background_ratio = 0.75

        elif args.preset == 'shale-dust':
            # Kondisi dengan banyak debu di udara
            args.pre_process = 'erode'  # Erosi untuk mengurangi noise debu halus
            args.pre_kernel_size = 3
            args.pre_iterations = 2
            args.post_process = 'open'  # Opening untuk menghilangkan deteksi debu kecil
            args.post_kernel_size = 5
            args.post_iterations = 2
            args.var_threshold = 25
            args.detect_shadows = False
            args.learning_rate = 0.008
            args.nmixtures = 5
            args.history = 400
            args.background_ratio = 0.85

    return args


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Convert source to int if it's a digit string (camera index)
    source = int(args.source) if args.source.isdigit() else args.source
    
    # Create background subtractor instance
    bg_subtractor = ForegroundExtraction(
        # MOG2 parameters
        history=args.history,
        var_threshold=args.var_threshold,
        detect_shadows=args.detect_shadows,
        nmixtures=args.nmixtures,
        background_ratio=args.background_ratio,
        # Learning rate
        learning_rate=args.learning_rate,
        # Pre-processing morphological operations
        pre_process=args.pre_process,
        pre_kernel_size=args.pre_kernel_size,
        pre_iterations=args.pre_iterations,
        # Post-processing morphological operations
        post_process=args.post_process,
        post_kernel_size=args.post_kernel_size,
        post_iterations=args.post_iterations,
        # Display options
        show_original=not args.hide_original,
        show_mask=not args.hide_mask,
        show_result=not args.hide_result
    )
    
    # Run background subtraction
    bg_subtractor.run(source, args.width, args.height)


if __name__ == "__main__":
    main()