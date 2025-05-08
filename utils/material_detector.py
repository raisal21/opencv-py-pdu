import cv2 as cv
import numpy as np
import argparse
from collections import namedtuple

FrameResult = namedtuple('FrameResult', ['original', 'mask', 'binary'])
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
        'history': 200,
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
        'background_ratio': 0.75
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
    
    # [Rest of the ContourProcessor methods stay the same]
    # ... 

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
    
    # Display options
    parser.add_argument('--hide-original', action='store_true',
                        help='Hide original frame')
    parser.add_argument('--hide-mask', action='store_true',
                        help='Hide foreground mask')
    parser.add_argument('--hide-analysis', action='store_true',
                        help='Hide contour analysis visualization')
    
    args = parser.parse_args()
    
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
    
    # Create background subtractor instance using preset
    bg_subtractor = ForegroundExtraction.from_preset(args.bg_preset)
    
    # Create contour processor instance using preset
    contour_processor = ContourProcessor.from_preset(args.contour_preset)
    
    # Override with custom parameters if specified
    # (This allows users to start with a preset but tweak individual parameters)
    # ...
    
    # Initialize video capture
    cap = cv.VideoCapture(source)
    
    # Set camera resolution if using webcam
    if isinstance(source, int):
        cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # [Rest of the main function remains the same]
    # ...

if __name__ == "__main__":
    main()