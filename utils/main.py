import cv2 as cv
import numpy as np
import argparse
import time
from utils.roi_selector import ROISelector
from utils.material_detector import ForegroundExtraction, ContourProcessor


class MaterialMonitoringApp:
    """
    Integrated application for material monitoring that combines:
    - ROI selection
    - Background subtraction for foreground extraction
    - Contour processing for material analysis

    The application flow:
    1. Select ROI from video feed
    2. Apply background subtraction within the ROI
    3. Analyze material coverage using contour processing
    4. Display results and metrics
    """

    # UI Constants
    WINDOW_TITLE_MAIN = "Material Monitoring"
    WINDOW_TITLE_ROI = "Select Monitoring Region"
    WINDOW_TITLE_FOREGROUND = "Foreground Extraction"  # New window title
    WINDOW_TITLE_ANALYSIS = "Material Analysis"

    # Color constants
    ROI_COLOR = (0, 149, 255)  # Orange color (BGR format) - same as in ROISelector
    TEXT_COLOR = (0, 0, 255)  # Red (BGR)
    TEXT_FONT = cv.FONT_HERSHEY_SIMPLEX
    TEXT_SCALE = 0.7
    TEXT_THICKNESS = 2

    # Analysis modes
    MODE_ROI_SELECTION = 0
    MODE_MONITORING = 1

    def __init__(
        self,
        source=0,
        width=1280,
        height=720,
        bg_subtractor_params=None,
        contour_processor_params=None,
    ):
        """
        Initialize the integrated application with video source and processing
        parameters.

        Args:
            source: Camera index or video file path
            width: Desired width for camera/video frames
            height: Desired height for camera/video frames
            bg_subtractor_params: Dictionary of parameters for ForegroundExtraction
            contour_processor_params: Dictionary of parameters for ContourProcessor
        """
        # Initialize video parameters
        self.source = int(source) if str(source).isdigit() else source
        self.width = width
        self.height = height

        # Set default parameters if none provided
        if bg_subtractor_params is None:
            bg_subtractor_params = {}
        if contour_processor_params is None:
            contour_processor_params = {}

        # Create processing components (but don't initialize them yet)
        self.bg_subtractor = ForegroundExtraction(**bg_subtractor_params)
        self.contour_processor = ContourProcessor(**contour_processor_params)

        # Initialize state variables
        self.mode = self.MODE_ROI_SELECTION
        self.roi_points = None
        self.roi_image = None
        self.roi_mask = None
        self.transform_matrix = None
        self.inverse_transform_matrix = None

        # Performance tracking
        self.last_frame_time = 0
        self.fps = 0

        # Initialize video capture
        self.cap = None

    def start(self):
        """Start the application and main processing loop."""
        # Initialize video capture
        self.cap = cv.VideoCapture(self.source)

        # Set camera resolution if using webcam
        if isinstance(self.source, int):
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)

        # Check if video source is opened successfully
        if not self.cap.isOpened():
            print(f"Error: Could not open video source {self.source}")
            return False

        # Create main window
        cv.namedWindow(self.WINDOW_TITLE_MAIN, cv.WINDOW_NORMAL)

        # Start in ROI selection mode
        self._select_roi()

        # Main processing loop
        try:
            while True:
                # Calculate FPS
                current_time = time.time()
                self.fps = (
                    1 / (current_time - self.last_frame_time)
                    if self.last_frame_time > 0
                    else 0
                )
                self.last_frame_time = current_time

                # Read a frame
                ret, frame = self.cap.read()

                # Break if frame reading failed (end of video)
                if not ret:
                    print("End of video or error reading frame")
                    break

                # Process frame based on current mode
                if self.mode == self.MODE_ROI_SELECTION:
                    cv.imshow(self.WINDOW_TITLE_MAIN, frame)
                    if cv.waitKey(1) & 0xFF == ord("r"):
                        self._select_roi()

                elif self.mode == self.MODE_MONITORING:
                    self._process_monitoring_frame(frame)

                # Check for key presses
                key = cv.waitKey(1) & 0xFF
                if key == ord("q"):  # Quit
                    break
                elif key == ord("r"):  # Reset/Re-select ROI
                    self._select_roi()
                elif key == ord("b"):  # Reset background model
                    print("Resetting background model...")
                    self.bg_subtractor.reset_background()

        finally:
            # Clean up resources
            self.cap.release()
            cv.destroyAllWindows()

    def _select_roi(self):
        """Enter ROI selection mode and handle the ROI selection workflow."""
        self.mode = self.MODE_ROI_SELECTION

        # Capture a single frame for ROI selection
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame for ROI selection")
            return

        # Create ROI selector
        selector = ROISelector(self.WINDOW_TITLE_ROI, frame)

        # ROI selection loop
        while True:
            key = cv.waitKey(20) & 0xFF

            if key == 27:  # ESC - cancel and exit
                cv.destroyWindow(self.WINDOW_TITLE_ROI)
                return

            elif key == 13 and selector.is_complete():  # ENTER - confirm
                selector.confirm_selection()

                # Get ROI and prepare transformation matrices
                self.roi_points, self.roi_image = selector.get_roi()

                if self.roi_points and self.roi_image is not None:
                    # Calculate transformation matrices for future processing
                    self._calculate_transform_matrices()

                    # Switch to monitoring mode
                    self.mode = self.MODE_MONITORING
                    cv.destroyWindow(self.WINDOW_TITLE_ROI)

                    # Create analysis windows
                    cv.namedWindow(self.WINDOW_TITLE_FOREGROUND, cv.WINDOW_NORMAL)
                    cv.namedWindow(self.WINDOW_TITLE_ANALYSIS, cv.WINDOW_NORMAL)
                    return

            # Allow capture of new frames while in selection mode (for video)
            if isinstance(self.source, str):  # If it's a video file
                ret, new_frame = self.cap.read()
                if ret:
                    selector = ROISelector(self.WINDOW_TITLE_ROI, new_frame)

    def _calculate_transform_matrices(self):
        """Calculate transformation matrices for ROI processing."""
        if len(self.roi_points) != 4:
            return

        # Get source points from ROI selection
        src_pts = np.array(self.roi_points, dtype=np.float32)

        # Calculate rectangle dimensions for destination points
        width_1 = np.linalg.norm(
            np.array(self.roi_points[0]) - np.array(self.roi_points[1])
        )
        width_2 = np.linalg.norm(
            np.array(self.roi_points[2]) - np.array(self.roi_points[3])
        )
        height_1 = np.linalg.norm(
            np.array(self.roi_points[1]) - np.array(self.roi_points[2])
        )
        height_2 = np.linalg.norm(
            np.array(self.roi_points[3]) - np.array(self.roi_points[0])
        )

        width = max(int(width_1), int(width_2))
        height = max(int(height_1), int(height_2))

        # Define destination points
        dst_pts = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )

        # Calculate the perspective transformation matrix
        self.transform_matrix = cv.getPerspectiveTransform(src_pts, dst_pts)

        # Calculate the inverse transformation matrix (for mapping back)
        self.inverse_transform_matrix = cv.getPerspectiveTransform(dst_pts, src_pts)

        # Create a mask for the ROI in the original frame
        self.roi_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv.fillPoly(self.roi_mask, [np.array(self.roi_points, dtype=np.int32)], 255)

    def _apply_aspect_ratio_padding(self, image, target_ratio=16 / 9):
        """
        Apply padding to maintain target aspect ratio.
        This is the same function used in ROISelector to ensure consistent
        display.

        Args:
            image: Input image to pad
            target_ratio: Desired width/height ratio (default: 16/9)

        Returns:
            Padded image with consistent aspect ratio
        """
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

        # Apply padding with consistent color (using the ROI color)
        return cv.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv.BORDER_CONSTANT,
            value=self.ROI_COLOR,
        )

    def _process_monitoring_frame(self, frame):
        """
        Process a frame in monitoring mode:
        1. Extract the ROI from the frame
        2. Apply background subtraction
        3. Process contours and calculate metrics
        4. Visualize results

        Args:
            frame: Input video frame
        """
        # Apply ROI perspective transform
        warped_roi = cv.warpPerspective(
            frame,
            self.transform_matrix,
            (self.roi_image.shape[1], self.roi_image.shape[0]),
        )

        # Process the ROI with background subtraction - use the full FrameResult
        bg_result = self.bg_subtractor.process_frame(warped_roi)

        # Process the binary mask to find and analyze contours
        contour_result = self.contour_processor.process_mask(bg_result.binary)

        # Create visualization with metrics
        analysis_vis = self.contour_processor.visualize(
            warped_roi, contour_result.contours, contour_result.metrics
        )

        # Apply consistent aspect ratio padding to all visualization windows
        analysis_vis_padded = self._apply_aspect_ratio_padding(analysis_vis)
        foreground_mask_padded = self._apply_aspect_ratio_padding(bg_result.mask)

        # Draw ROI outline on original frame using the orange ROI_COLOR
        display_frame = frame.copy()
        cv.polylines(
            display_frame,
            [np.array(self.roi_points, dtype=np.int32)],
            True,
            self.ROI_COLOR,
            2,
        )

        # Add metrics text to the original frame
        self._add_metrics_to_frame(display_frame, contour_result.metrics)

        # Display frames, now including the foreground mask
        cv.imshow(self.WINDOW_TITLE_MAIN, display_frame)
        cv.imshow(self.WINDOW_TITLE_FOREGROUND, foreground_mask_padded)
        cv.imshow(self.WINDOW_TITLE_ANALYSIS, analysis_vis_padded)

    def _add_metrics_to_frame(self, frame, metrics):
        """Add metrics information to the display frame."""
        # Format metrics text
        metrics_text = [
            f"FPS: {self.fps:.1f}",
            f"Coverage: {metrics['processed_coverage_percent']:.2f}%",
            f"Contours: {metrics['contour_count']}",
            f"Press 'R' to reset ROI",
            f"Press 'B' to reset background",
            f"Press 'Q' to quit",
        ]

        # Draw semi-transparent background for text
        text_bg = np.zeros_like(frame)
        cv.rectangle(
            text_bg, (10, 10), (300, 40 + 30 * len(metrics_text)), (0, 0, 0), -1
        )
        frame = cv.addWeighted(frame, 1, text_bg, 0.6, 0)

        # Draw text
        y_pos = 40
        for text in metrics_text:
            cv.putText(
                frame,
                text,
                (20, y_pos),
                self.TEXT_FONT,
                self.TEXT_SCALE,
                self.TEXT_COLOR,
                self.TEXT_THICKNESS,
            )
            y_pos += 30


def parse_arguments():
    """Parse command line arguments for the application."""
    parser = argparse.ArgumentParser(description="Material Monitoring Application")

    # Video source
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source (camera index or video path). Default: 0 (webcam)",
    )

    # Resolution
    parser.add_argument(
        "--width", type=int, default=1280, help="Frame width. Default: 1280"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Frame height. Default: 720"
    )

    # Presets
    parser.add_argument(
        "--preset",
        type=str,
        choices=[
            "none",
            "shale-day-clear",
            "shale-day-rainy",
            "shale-night",
            "shale-vibration",
            "shale-dust",
        ],
        default="none",
        help="Use a recommended preset configuration",
    )

    args = parser.parse_args()
    return args


def apply_preset(preset_name):
    """
    Apply a preset configuration.

    Args:
        preset_name: Name of the preset to apply

    Returns:
        Tuple of (bg_subtractor_params, contour_processor_params)
    """
    # Default background subtractor parameters
    bg_params = {
        "history": 500,
        "var_threshold": 16,
        "detect_shadows": True,
        "nmixtures": 5,
        "background_ratio": 0.9,
        "learning_rate": 0.01,
        "pre_process": None,
        "pre_kernel_size": 5,
        "pre_iterations": 1,
        "post_process": None,
        "post_kernel_size": 5,
        "post_iterations": 1,
    }

    # Default contour processor parameters
    contour_params = {
        "min_contour_area": 100,
        "use    _convex_hull": True,
        "merge_overlapping": False,
        "merge_distance": 10,
        "show_contour_index": False,
        "show_contour_area": False,
    }

    # Apply preset-specific parameters
    if preset_name == "shale-day-clear":
        # Kondisi siang hari cerah - kontras tinggi, bayangan jelas
        bg_params.update(
            {
                "pre_process": "erode",
                "pre_kernel_size": 3,
                "pre_iterations": 1,
                "post_process": "close",
                "post_kernel_size": 5,
                "post_iterations": 2,
                "var_threshold": 30,
                "detect_shadows": False,
                "learning_rate": 0.003,
                "nmixtures": 4,
                "background_ratio": 0.85,
            }
        )

        contour_params.update(
            {
                "min_contour_area": 150,
                "use_convex_hull": True,
                "merge_overlapping": True,
                "merge_distance": 8,
            }
        )

    elif preset_name == "shale-day-rainy":
        # Kondisi hujan - kontras rendah, banyak pergerakan air
        bg_params.update(
            {
                "pre_process": "open",
                "pre_kernel_size": 5,
                "pre_iterations": 2,
                "post_process": "close",
                "post_kernel_size": 7,
                "post_iterations": 2,
                "var_threshold": 18,
                "detect_shadows": False,
                "learning_rate": 0.01,
                "nmixtures": 6,
                "history": 300,
                "background_ratio": 0.8,
            }
        )

        contour_params.update(
            {
                "min_contour_area": 200,
                "use_convex_hull": True,
                "merge_overlapping": True,
                "merge_distance": 15,
            }
        )

    elif preset_name == "shale-night":
        # Kondisi malam - pencahayaan buatan, kontras tinggi, bayangan tajam
        bg_params.update(
            {
                "pre_process": "open",
                "pre_kernel_size": 3,
                "pre_iterations": 1,
                "post_process": "dilate",
                "post_kernel_size": 5,
                "post_iterations": 1,
                "var_threshold": 15,
                "detect_shadows": False,
                "learning_rate": 0.002,
                "nmixtures": 3,
                "background_ratio": 0.9,
            }
        )

        contour_params.update(
            {
                "min_contour_area": 100,
                "use_convex_hull": True,
                "merge_overlapping": False,
            }
        )

    elif preset_name == "shale-vibration":
        # Kondisi dengan banyak getaran peralatan
        bg_params.update(
            {
                "pre_process": "open",
                "pre_kernel_size": 5,
                "pre_iterations": 1,
                "post_process": "close",
                "post_kernel_size": 9,
                "post_iterations": 3,
                "var_threshold": 20,
                "detect_shadows": False,
                "learning_rate": 0.015,
                "nmixtures": 7,
                "history": 200,
                "background_ratio": 0.75,
            }
        )

        contour_params.update(
            {
                "min_contour_area": 250,
                "use_convex_hull": True,
                "merge_overlapping": True,
                "merge_distance": 20,
            }
        )

    elif preset_name == "shale-dust":
        # Kondisi dengan banyak debu di udara
        bg_params.update(
            {
                "pre_process": "erode",
                "pre_kernel_size": 3,
                "pre_iterations": 2,
                "post_process": "open",
                "post_kernel_size": 5,
                "post_iterations": 2,
                "var_threshold": 25,
                "detect_shadows": False,
                "learning_rate": 0.008,
                "nmixtures": 5,
                "history": 400,
                "background_ratio": 0.85,
            }
        )

        contour_params.update(
            {
                "min_contour_area": 180,
                "use_convex_hull": True,
                "merge_overlapping": True,
                "merge_distance": 12,
            }
        )

    return bg_params, contour_params


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()

    # Convert source to int if it's a digit string (camera index)
    source = int(args.source) if args.source.isdigit() else args.source

    # Apply preset if specified
    if args.preset != "none":
        print(f"Using preset: {args.preset}")
        bg_params, contour_params = apply_preset(args.preset)
    else:
        bg_params, contour_params = {}, {}

    # Create and start the application
    app = MaterialMonitoringApp(
        source=source,
        width=args.width,
        height=args.height,
        bg_subtractor_params=bg_params,
        contour_processor_params=contour_params,
    )

    app.start()


if __name__ == "__main__":
    main()
