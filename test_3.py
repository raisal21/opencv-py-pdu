import cv2 as cv
import numpy as np
import argparse

class BackgroundSubtractor:
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
    bg_subtractor = BackgroundSubtractor(
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