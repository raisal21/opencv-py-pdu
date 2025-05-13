import cv2 as cv
import numpy as np
import argparse

class BackgroundSubtractor:
    """
    Background subtraction implementation using OpenCV's MOG2 algorithm
    with preset configurations for different environmental conditions.
    """
    
    def __init__(self, 
                 # MOG2 parameters
                 history=500,
                 var_threshold=16, 
                 detect_shadows=True,
                 nmixtures=5,
                 background_ratio=0.9,
                 # Learning rate parameter
                 learning_rate=0.01,
                 # Pre-processing morphological operation
                 pre_process=None,
                 pre_kernel_size=5,
                 pre_iterations=1,
                 # Post-processing morphological operation 
                 post_process=None,
                 post_kernel_size=5,
                 post_iterations=1):
        """
        Initialize the background subtractor with preset parameters.
        
        Args:
            history: Number of frames for background model
            var_threshold: Threshold for foreground/background decision
            detect_shadows: Whether to detect shadows separately
            nmixtures: Number of Gaussian components per background pixel
            background_ratio: Threshold for background component
            learning_rate: Background model update speed
            pre_process: Morphological operation before subtraction
            pre_kernel_size: Kernel size for pre-processing
            pre_iterations: Number of pre-processing iterations
            post_process: Morphological operation after subtraction
            post_kernel_size: Kernel size for post-processing
            post_iterations: Number of post-processing iterations
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
            Tuple of (original frame, foreground mask)
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
        
        return original, display_mask
    
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
        cv.namedWindow("Original", cv.WINDOW_NORMAL)
        cv.namedWindow("Foreground Mask", cv.WINDOW_NORMAL)
        
        # Add preset name to window title if applicable
        if hasattr(self, 'preset_name') and self.preset_name:
            cv.setWindowTitle("Foreground Mask", f"Foreground Mask - {self.preset_name}")
        
        # Main loop
        while True:
            # Read a frame
            ret, frame = cap.read()
            
            # Break if frame reading failed (end of video)
            if not ret:
                break
            
            # Process the frame
            original, fg_mask = self.process_frame(frame)
            
            # Display the frames
            cv.imshow("Original", original)
            cv.imshow("Foreground Mask", fg_mask)
            
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Background Subtraction with Presets')
    
    # Video source
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (camera index or video path). Default: 0 (webcam)')
    
    # Resolution
    parser.add_argument('--width', type=int, default=640,
                        help='Frame width. Default: 640')
    parser.add_argument('--height', type=int, default=480,
                        help='Frame height. Default: 480')
    
    # Preset selection
    parser.add_argument('--preset', type=str, 
                        choices=['default', 'shale-day-clear', 'shale-day-rainy', 'shale-night', 'shale-vibration', 'shale-dust'],
                        default='default',
                        help='Use a recommended preset configuration')
    
    args = parser.parse_args()
    return args


def apply_preset(preset_name):
    """
    Create a BackgroundSubtractor with preset parameters.
    
    Args:
        preset_name: Name of the preset to apply
        
    Returns:
        Configured BackgroundSubtractor instance
    """
    # Default parameters
    params = {
        'history': 500,
        'var_threshold': 16,
        'detect_shadows': True,
        'nmixtures': 5,
        'background_ratio': 0.9,
        'learning_rate': 0.01,
        'pre_process': None,
        'pre_kernel_size': 5,
        'pre_iterations': 1,
        'post_process': None,
        'post_kernel_size': 5,
        'post_iterations': 1
    }
    
    # Apply preset-specific parameters
    if preset_name == 'shale-day-clear':
        # Kondisi siang hari cerah - kontras tinggi, bayangan jelas
        params.update({
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
        })
    
    elif preset_name == 'shale-day-rainy':
        # Kondisi hujan - kontras rendah, banyak pergerakan air
        params.update({
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
            'history': 300,
            'background_ratio': 0.8
        })
    
    elif preset_name == 'shale-night':
        # Kondisi malam - pencahayaan buatan, kontras tinggi, bayangan tajam
        params.update({
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
        })
    
    elif preset_name == 'shale-vibration':
        # Kondisi dengan banyak getaran peralatan
        params.update({
            'pre_process': 'open',
            'pre_kernel_size': 5,
            'pre_iterations': 1,
            'post_process': 'close',
            'post_kernel_size': 9,
            'post_iterations': 3,
            'var_threshold': 20,
            'detect_shadows': False,
            'learning_rate': 0.015,
            'nmixtures': 7,
            'history': 200,
            'background_ratio': 0.75
        })
    
    elif preset_name == 'shale-dust':
        # Kondisi dengan banyak debu di udara
        params.update({
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
            'history': 400,
            'background_ratio': 0.85
        })
    
    # Create and return instance
    bg_subtractor = BackgroundSubtractor(**params)
    bg_subtractor.preset_name = preset_name
    return bg_subtractor


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Convert source to int if it's a digit string (camera index)
    source = int(args.source) if args.source.isdigit() else args.source
    
    # Create background subtractor with the selected preset
    bg_subtractor = apply_preset(args.preset)
    
    # Print selected preset information
    print(f"Using preset: {args.preset}")
    
    # Run background subtraction
    bg_subtractor.run(source, args.width, args.height)


if __name__ == "__main__":
    main()