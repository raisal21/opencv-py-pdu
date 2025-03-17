import cv2 as cv
import numpy as np
import argparse
import time


class ForegroundExtractor:
    """
    A class that implements foreground extraction using background subtraction
    and morphological operations.
    
    This class provides implementations for both MOG2 and KNN algorithms and
    applies morphological operations (opening and closing) to refine the results.
    """
    
    def __init__(self, method='MOG2', history=50, dist_threshold=20, 
                 detect_shadows=True, learning_rate=0.01, use_grayscale=False):
        """
        Initialize the foreground extractor with background subtraction parameters.
        
        Args:
            method (str): Background subtraction method ('MOG2' or 'KNN')
            history (int): Number of frames used to build the background model
            dist_threshold (float): Threshold value:
                                   - For MOG2: Mahalanobis distance squared threshold
                                   - For KNN: Squared distance threshold
            detect_shadows (bool): Whether to detect shadows
            learning_rate (float): Speed of background model adaptation (0-1, or -1 for auto)
            use_grayscale (bool): Whether to convert frames to grayscale before processing
        """
        # Initialize background subtractor based on method
        if method == 'MOG2':
            self.bg_subtractor = cv.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=dist_threshold,
                detectShadows=detect_shadows
            )
        elif method == 'KNN':
            self.bg_subtractor = cv.createBackgroundSubtractorKNN(
                history=history,
                dist2Threshold=dist_threshold,
                detectShadows=detect_shadows
            )
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'MOG2' or 'KNN'.")
        
        # Store parameters
        self.method = method
        self.history = history
        self.dist_threshold = dist_threshold
        self.detect_shadows = detect_shadows
        self.learning_rate = learning_rate
        self.use_grayscale = use_grayscale
        
        # Morphological operation parameters
        self.opening_kernel_size = 3
        self.opening_iterations = 2
        self.closing_kernel_size = 3
        self.closing_iterations = 3
        
        # Create morphological kernels
        self.opening_kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE, 
            (self.opening_kernel_size, self.opening_kernel_size)
        )
        self.closing_kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE, 
            (self.closing_kernel_size, self.closing_kernel_size)
        )
    
    def extract_foreground(self, frame):
        """
        Extract foreground mask from a frame using background subtraction
        followed by morphological operations.
        
        Args:
            frame (numpy.ndarray): Input frame
        
        Returns:
            tuple: (raw_mask, processed_mask, visualization)
                - raw_mask: Original foreground mask before morphological operations
                - processed_mask: Foreground mask after morphological operations
                - visualization: Visualization of the result
        """
        # Create a copy to avoid modifying the original frame
        process_frame = frame.copy()
        
        # Convert to grayscale if needed
        if self.use_grayscale:
            gray = cv.cvtColor(process_frame, cv.COLOR_BGR2GRAY)
            process_frame = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(process_frame, learningRate=self.learning_rate)
        
        # Convert shadow values (127) to black (0) if shadows are detected
        if self.detect_shadows:
            raw_mask = cv.threshold(fg_mask, 127, 255, cv.THRESH_BINARY)[1]
        else:
            raw_mask = fg_mask
        
        # Create a copy for morphological operations
        processed_mask = raw_mask.copy()
        
        # Apply opening to remove noise (erosion followed by dilation)
        processed_mask = cv.morphologyEx(
            processed_mask,
            cv.MORPH_OPEN,
            self.opening_kernel,
            iterations=self.opening_iterations
        )
        
        # Apply closing to fill holes (dilation followed by erosion)
        processed_mask = cv.morphologyEx(
            processed_mask,
            cv.MORPH_CLOSE,
            self.closing_kernel,
            iterations=self.closing_iterations
        )
        
        # Create visualization
        visualization = cv.bitwise_and(frame, frame, mask=processed_mask)
        
        return raw_mask, processed_mask, visualization


def process_video(video_path, method='MOG2', output_path=None):
    """
    Process a video using foreground extraction.
    
    Args:
        video_path (str): Path to video file or camera index
        method (str): Background subtraction method ('MOG2' or 'KNN')
        output_path (str, optional): Path to save the output video
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Open video capture
    if video_path.isdigit():
        cap = cv.VideoCapture(int(video_path))
    else:
        cap = cv.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_path}")
        return False
    
    # Get video properties
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default to 30fps if not available
    
    # Create video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        writer = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Create foreground extractor
    extractor = ForegroundExtractor(
        method=method,
        history=500,
        dist_threshold=16 if method == 'MOG2' else 400,  # Different default thresholds for MOG2 vs KNN
        detect_shadows=True,
        learning_rate=0.01,
        use_grayscale=False
    )
    
    # Create window
    window_name = f"Foreground Extraction ({method})"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    
    # Process frames
    start_time = time.time()
    frame_count = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Extract foreground
        raw_mask, processed_mask, result = extractor.extract_foreground(frame)
        
        # Calculate foreground coverage percentage
        height, width = processed_mask.shape
        total_pixels = height * width
        foreground_pixels = cv.countNonZero(processed_mask)
        coverage = (foreground_pixels / total_pixels) * 100
        
        # Add coverage information to frame
        cv.putText(
            frame,
            f"Coverage: {coverage:.2f}%",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        
        # Create visualization display
        # Original frame | Raw Mask | Processed Mask | Result
        raw_mask_colored = cv.cvtColor(raw_mask, cv.COLOR_GRAY2BGR)
        processed_mask_colored = cv.cvtColor(processed_mask, cv.COLOR_GRAY2BGR)
        
        # First row: Original and Raw Mask
        top_row = np.hstack((frame, raw_mask_colored))
        # Second row: Processed Mask and Result
        bottom_row = np.hstack((processed_mask_colored, result))
        
        # Stack rows
        display = np.vstack((top_row, bottom_row))
        
        # Resize for display if too large
        screen_width, screen_height = 1280, 720
        if display.shape[1] > screen_width or display.shape[0] > screen_height:
            scale = min(screen_width / display.shape[1], screen_height / display.shape[0])
            display = cv.resize(display, (0, 0), fx=scale, fy=scale)
        
        # Show result
        cv.imshow(window_name, display)
        
        # Write frame if output is enabled
        if writer:
            writer.write(result)
        
        # Check for exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Calculate and print statistics
    elapsed_time = time.time() - start_time
    if frame_count > 0:
        fps_achieved = frame_count / elapsed_time
        print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
        print(f"Average FPS: {fps_achieved:.2f}")
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv.destroyAllWindows()
    
    return True


def main():
    """
    Main function for the foreground extraction application.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Foreground Extraction with MOG2/KNN and Morphological Operations')
    parser.add_argument('video', help='Path to video file or camera index (0 for default camera)')
    parser.add_argument('--method', type=str, choices=['MOG2', 'KNN'], default='MOG2',
                        help='Background subtraction method')
    parser.add_argument('--output', type=str, help='Path to save output video')
    
    args = parser.parse_args()
    
    # Process video
    success = process_video(args.video, args.method, args.output)
    
    if success:
        print("Video processing completed successfully.")
    else:
        print("Video processing failed.")


if __name__ == "__main__":
    main()