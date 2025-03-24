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
            Tuple of (original frame, foreground mask, result frame, binary mask)
            - original frame: Copy of the input frame
            - foreground mask: Visualizable foreground mask (BGR format)
            - result frame: Original frame with foreground only
            - binary mask: Binary mask for further processing (grayscale)
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
        
        return original, display_mask, binary_mask
    
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
    Designed to work with output from ForegroundExtraction class.
    
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
        # Make a copy of the binary mask and ensure it's binary (0 or 255)
        mask = binary_mask.copy()
        if len(mask.shape) > 2:
            mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)
    
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
    parser.add_argument('--preset', type=str, 
                        choices=['none', 'shale-day-clear', 'shale-day-rainy', 'shale-night', 'shale-vibration', 'shale-dust'],
                        default='none',
                        help='Use a recommended preset configuration for environment conditions')
    
    # MOG2 parameters
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
    
    # Display options
    parser.add_argument('--hide-original', action='store_true',
                        help='Hide original frame')
    parser.add_argument('--hide-mask', action='store_true',
                        help='Hide foreground mask')
    parser.add_argument('--hide-analysis', action='store_true',
                        help='Hide contour analysis visualization')
    
    args = parser.parse_args()
    
    # Apply presets if specified
    if args.preset != 'none':
        if args.preset == 'shale-day-clear':
            # Kondisi siang hari cerah - kontras tinggi, bayangan jelas
            args.pre_process = 'erode'
            args.pre_kernel_size = 3
            args.pre_iterations = 1
            args.post_process = 'close'
            args.post_kernel_size = 5
            args.post_iterations = 2
            args.var_threshold = 30
            args.detect_shadows = False
            args.learning_rate = 0.003
            args.nmixtures = 4
            args.background_ratio = 0.85
            # Contour processing presets for daylight
            args.min_contour_area = 150
            args.use_convex_hull = True
            args.merge_overlapping = True
            args.merge_distance = 8

        elif args.preset == 'shale-day-rainy':
            # Kondisi hujan - kontras rendah, banyak pergerakan air
            args.pre_process = 'open'
            args.pre_kernel_size = 5
            args.pre_iterations = 2
            args.post_process = 'close'
            args.post_kernel_size = 7
            args.post_iterations = 2
            args.var_threshold = 18
            args.detect_shadows = False
            args.learning_rate = 0.01
            args.nmixtures = 6
            args.history = 300
            args.background_ratio = 0.8
            # Contour processing presets for rainy conditions
            args.min_contour_area = 200
            args.use_convex_hull = True
            args.merge_overlapping = True
            args.merge_distance = 15

        elif args.preset == 'shale-night':
            # Kondisi malam - pencahayaan buatan, kontras tinggi, bayangan tajam
            args.pre_process = 'open'
            args.pre_kernel_size = 3
            args.pre_iterations = 1
            args.post_process = 'dilate'
            args.post_kernel_size = 5
            args.post_iterations = 1
            args.var_threshold = 15
            args.detect_shadows = False
            args.learning_rate = 0.002
            args.nmixtures = 3
            args.background_ratio = 0.9
            # Contour processing presets for night conditions
            args.min_contour_area = 100
            args.use_convex_hull = True
            args.merge_overlapping = False

        elif args.preset == 'shale-vibration':
            # Kondisi dengan banyak getaran peralatan
            args.pre_process = 'open'
            args.pre_kernel_size = 5
            args.pre_iterations = 1
            args.post_process = 'close'
            args.post_kernel_size = 9
            args.post_iterations = 3
            args.var_threshold = 20
            args.detect_shadows = False
            args.learning_rate = 0.015
            args.nmixtures = 7
            args.history = 200
            args.background_ratio = 0.75
            # Contour processing presets for vibration conditions
            args.min_contour_area = 250
            args.use_convex_hull = True
            args.merge_overlapping = True
            args.merge_distance = 20

        elif args.preset == 'shale-dust':
            # Kondisi dengan banyak debu di udara
            args.pre_process = 'erode'
            args.pre_kernel_size = 3
            args.pre_iterations = 2
            args.post_process = 'open'
            args.post_kernel_size = 5
            args.post_iterations = 2
            args.var_threshold = 25
            args.detect_shadows = False
            args.learning_rate = 0.008
            args.nmixtures = 5
            args.history = 400
            args.background_ratio = 0.85
            # Contour processing presets for dusty conditions
            args.min_contour_area = 180
            args.use_convex_hull = True
            args.merge_overlapping = True
            args.merge_distance = 12

    return args


def main():
    """
    Main function for running the Material Detector as a standalone application.
    Integrates ForegroundExtraction and ContourProcessor for complete material analysis.
    """
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
        post_iterations=args.post_iterations
    )
    
    # Create contour processor instance
    contour_processor = ContourProcessor(
        min_contour_area=args.min_contour_area,
        use_convex_hull=args.use_convex_hull,
        merge_overlapping=args.merge_overlapping,
        merge_distance=args.merge_distance,
        show_contour_index=args.show_contour_index,
        show_contour_area=args.show_contour_area
    )
    
    # Initialize video capture
    cap = cv.VideoCapture(source)
    
    # Set camera resolution if using webcam
    if isinstance(source, int):
        cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Check if video source is opened
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Create display windows
    if not args.hide_original:
        cv.namedWindow("Original", cv.WINDOW_NORMAL)
    if not args.hide_mask:
        cv.namedWindow("Foreground Mask", cv.WINDOW_NORMAL)
    if not args.hide_analysis:
        cv.namedWindow("Contour Analysis", cv.WINDOW_NORMAL)
    
    # Print active preset if using one
    if args.preset != 'none':
        print(f"Using preset: {args.preset}")
    
    # Main processing loop
    while True:
        # Read a frame
        ret, frame = cap.read()
        
        # Break if frame reading failed (end of video)
        if not ret:
            print("End of video stream or error reading frame")
            break
        
        # Process the frame with background subtraction
        original, fg_mask_display, binary_mask = bg_subtractor.process_frame(frame)
        
        # Process the binary mask to find and analyze contours
        _, contours, metrics = contour_processor.process_mask(binary_mask)
        
        # Create contour visualization
        contour_vis = contour_processor.visualize(original, contours, metrics)
        
        # Display the frames
        if not args.hide_original:
            cv.imshow("Original", original)
        if not args.hide_mask:
            cv.imshow("Foreground Mask", fg_mask_display)
        if not args.hide_analysis:
            cv.imshow("Contour Analysis", contour_vis)
        
        # Handle key presses
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):  # Exit on 'q' key
            break
        elif key == ord('r'):  # Reset background model on 'r' key
            print("Resetting background model...")
            bg_subtractor.reset_background()
    
    # Release resources
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()