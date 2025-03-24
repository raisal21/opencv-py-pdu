import cv2 as cv
import numpy as np
import argparse
import os
import time
from datetime import datetime

# Import custom modules
from roi_selector import RectangleROISelector
from material_detector import ForegroundExtraction
from material_detector import ContourProcessor

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Material Coverage Analysis System')
    
    # Video source
    parser.add_argument('--source', type=str, default='0',
                        help='Video source (camera index or video path). Default: 0 (webcam)')
    parser.add_argument('--width', type=int, default=1280,
                        help='Camera width. Default: 1280')
    parser.add_argument('--height', type=int, default=720,
                        help='Camera height. Default: 720')
    
    # Processing configuration
    parser.add_argument('--bg-preset', type=str, default='shale-day-clear',
                        choices=['none', 'shale-day-clear', 'shale-day-rainy', 
                                'shale-night', 'shale-vibration', 'shale-dust'],
                        help='Background subtraction preset')
    parser.add_argument('--material-preset', type=str, default='default',
                        choices=['default', 'liquid', 'solid'],
                        help='Material type preset for contour processing')
    
    # Output options
    parser.add_argument('--save-video', action='store_true',
                        help='Save output video')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save output files. Default: output')
    
    # Display options
    parser.add_argument('--skip-roi', action='store_true',
                        help='Skip ROI selection and use full frame')
    parser.add_argument('--hide-metrics', action='store_true',
                        help='Hide metrics display on visualization')
    
    args = parser.parse_args()
    return args

def select_roi(frame):
    """
    Use the RectangleROISelector to select an ROI in the frame.
    
    Args:
        frame: Input video frame
        
    Returns:
        np.ndarray: ROI contour points, or None if selection was canceled
    """
    window_name = "Select Material ROI"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    
    # Create selector
    selector = RectangleROISelector(window_name, frame)
    
    print("Select a rectangular ROI for material analysis:")
    print("  1. Click to place the first point")
    print("  2. Click to place the second point")
    print("  3. Click to place the third point (will form a rectangle)")
    print("  4. Press ENTER to confirm or right-click to reset")
    print("  5. Press ESC to cancel and use full frame")
    
    while True:
        key = cv.waitKey(20) & 0xFF
        
        if key == 27:  # ESC
            cv.destroyWindow(window_name)
            return None
        
        elif key == 13 and selector.is_complete():  # ENTER
            selector.confirm_selection()
            roi_points, _ = selector.get_roi()
            cv.destroyWindow(window_name)
            
            if roi_points:
                return np.array(roi_points, dtype=np.int32)
            else:
                return None
    
    return None

def configure_foreground_extraction(preset=None):
    """
    Configure a ForegroundExtraction instance based on the specified preset.
    
    Args:
        preset: Name of the preset to apply
        
    Returns:
        ForegroundExtraction: Configured instance
    """
    # Default parameters
    params = {
        'history': 500,
        'var_threshold': 16,
        'detect_shadows': False,
        'nmixtures': 5,
        'background_ratio': 0.9,
        'learning_rate': 0.01,
        'pre_process': None,
        'pre_kernel_size': 5,
        'pre_iterations': 1,
        'post_process': 'close',  # Default to closing for noise reduction
        'post_kernel_size': 5,
        'post_iterations': 2,
        'show_original': False,
        'show_mask': False,
        'show_result': False
    }
    
    # Apply preset-specific parameters
    if preset == 'shale-day-clear':
        params.update({
            'pre_process': 'erode',
            'pre_kernel_size': 3,
            'pre_iterations': 1,
            'post_process': 'close',
            'post_kernel_size': 5,
            'post_iterations': 2,
            'var_threshold': 30,
            'learning_rate': 0.003,
            'nmixtures': 4,
            'background_ratio': 0.85
        })
    elif preset == 'shale-day-rainy':
        params.update({
            'pre_process': 'open',
            'pre_kernel_size': 5,
            'pre_iterations': 2,
            'post_process': 'close',
            'post_kernel_size': 7,
            'post_iterations': 2,
            'var_threshold': 18,
            'learning_rate': 0.01,
            'nmixtures': 6,
            'history': 300,
            'background_ratio': 0.8
        })
    elif preset == 'shale-night':
        params.update({
            'pre_process': 'open',
            'pre_kernel_size': 3,
            'pre_iterations': 1,
            'post_process': 'dilate',
            'post_kernel_size': 5,
            'post_iterations': 1,
            'var_threshold': 15,
            'learning_rate': 0.002,
            'nmixtures': 3,
            'background_ratio': 0.9
        })
    elif preset == 'shale-vibration':
        params.update({
            'pre_process': 'open',
            'pre_kernel_size': 5,
            'pre_iterations': 1,
            'post_process': 'close',
            'post_kernel_size': 9,
            'post_iterations': 3,
            'var_threshold': 20,
            'learning_rate': 0.015,
            'nmixtures': 7,
            'history': 200,
            'background_ratio': 0.75
        })
    elif preset == 'shale-dust':
        params.update({
            'pre_process': 'erode',
            'pre_kernel_size': 3,
            'pre_iterations': 2,
            'post_process': 'open',
            'post_kernel_size': 5,
            'post_iterations': 2,
            'var_threshold': 25,
            'learning_rate': 0.008,
            'nmixtures': 5,
            'history': 400,
            'background_ratio': 0.85
        })
    
    # Create and return instance
    extractor = ForegroundExtraction(**params)
    return extractor

def setup_output_directory(output_dir):
    """
    Setup the output directory for saved files.
    
    Args:
        output_dir: Directory path
        
    Returns:
        str: Full path to the output directory
    """
    # Create timestamp for unique directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(output_dir, f"run_{timestamp}")
    
    # Create directory if it doesn't exist
    os.makedirs(full_path, exist_ok=True)
    
    return full_path

def save_metrics_to_file(metrics, frame_number, output_path):
    """
    Save metrics to a CSV file.
    
    Args:
        metrics: Dictionary of metrics
        frame_number: Current frame number
        output_path: Directory to save the file
    """
    csv_path = os.path.join(output_path, "metrics.csv")
    
    # Create header if file doesn't exist
    if not os.path.exists(csv_path):
        header = "frame,timestamp,coverage_percent,contour_count,total_area\n"
        with open(csv_path, 'w') as f:
            f.write(header)
    
    # Append metrics for this frame
    timestamp = time.time()
    coverage = metrics['processed_coverage_percent']
    contour_count = metrics['contour_count']
    total_area = metrics['total_contour_area']
    
    with open(csv_path, 'a') as f:
        f.write(f"{frame_number},{timestamp},{coverage:.2f},{contour_count},{total_area}\n")

def main():
    """Main function that integrates all components."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup output directory if saving outputs
    output_path = None
    if args.save_video:
        output_path = setup_output_directory(args.output_dir)
        print(f"Outputs will be saved to: {output_path}")
    
    # Convert source to int if it's a digit string (camera index)
    source = int(args.source) if args.source.isdigit() else args.source
    
    # Initialize video capture
    cap = cv.VideoCapture(source)
    
    # Set camera properties if using webcam
    if isinstance(source, int):
        cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Check if video source is opened
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Read the first frame for ROI selection
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read from video source")
        return
    
    # Select ROI or use full frame
    roi_contour = None
    if not args.skip_roi:
        roi_contour = select_roi(first_frame)
        
    # If ROI selection was skipped or canceled, use full frame
    if roi_contour is None:
        h, w = first_frame.shape[:2]
        roi_contour = np.array([
            [10, 10],
            [w-10, 10],
            [w-10, h-10],
            [10, h-10]
        ], dtype=np.int32)
        print("Using full frame as ROI")
    
    # Configure processing components
    fg_extractor = configure_foreground_extraction(args.bg_preset)
    contour_processor = ContourProcessor.create_preset(args.material_preset)
    
    # Create video writer if saving output
    video_writer = None
    if args.save_video:
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        output_video_path = os.path.join(output_path, "analysis.avi")
        frame_size = (first_frame.shape[1], first_frame.shape[0])
        video_writer = cv.VideoWriter(output_video_path, fourcc, 20.0, frame_size)
    
    # Create windows
    cv.namedWindow("Original", cv.WINDOW_NORMAL)
    cv.namedWindow("Material Detection", cv.WINDOW_NORMAL)
    cv.namedWindow("Analysis", cv.WINDOW_NORMAL)
    
    # Display information about the configuration
    print(f"\nSystem Configuration:")
    print(f"  Background Preset: {args.bg_preset}")
    print(f"  Material Preset: {args.material_preset}")
    print(f"  Save Output: {'Yes' if args.save_video else 'No'}")
    print("\nControls:")
    print("  R - Reset background model")
    print("  S - Save current frame")
    print("  Q - Quit")
    
    # Main processing loop
    frame_number = 0
    
    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame with background subtraction
        _, fg_mask, _ = fg_extractor.process_frame(frame)
        
        # Convert mask to grayscale if needed
        if len(fg_mask.shape) > 2:
            fg_mask_gray = cv.cvtColor(fg_mask, cv.COLOR_BGR2GRAY)
        else:
            fg_mask_gray = fg_mask
        
        # Ensure mask is binary
        _, binary_mask = cv.threshold(fg_mask_gray, 127, 255, cv.THRESH_BINARY)
        
        # Process the mask with contour processor
        processed_mask, contours, metrics = contour_processor.process_mask(binary_mask, roi_contour)
        
        # Create visualization
        show_metrics = not args.hide_metrics
        analysis_view = contour_processor.visualize(frame, contours, metrics, show_metrics)
        
        # Draw the ROI on the analysis view
        cv.polylines(analysis_view, [roi_contour], True, (255, 0, 0), 2)
        
        # Add configuration info
        cv.putText(analysis_view, f"BG: {args.bg_preset}", 
                  (10, frame.shape[0] - 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
        cv.putText(analysis_view, f"Material: {args.material_preset}", 
                  (10, frame.shape[0] - 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
        
        # Display results
        cv.imshow("Original", frame)
        cv.imshow("Material Detection", processed_mask)
        cv.imshow("Analysis", analysis_view)
        
        # Save the frame to output video
        if video_writer is not None:
            video_writer.write(analysis_view)
        
        # Save metrics to file
        if output_path is not None:
            save_metrics_to_file(metrics, frame_number, output_path)
        
        # Handle key presses
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            fg_extractor.reset_background()
            print("Background model reset")
        elif key == ord('s'):
            # Save current frame analysis
            if output_path is None:
                output_path = setup_output_directory(args.output_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_filename = os.path.join(output_path, f"snapshot_{timestamp}.jpg")
            cv.imwrite(snapshot_filename, analysis_view)
            
            # Save metrics snapshot
            metrics_filename = os.path.join(output_path, f"metrics_{timestamp}.txt")
            with open(metrics_filename, 'w') as f:
                f.write(f"Material Analysis Snapshot ({timestamp})\n")
                f.write(f"Background Preset: {args.bg_preset}\n")
                f.write(f"Material Preset: {args.material_preset}\n\n")
                f.write(f"Coverage: {metrics['processed_coverage_percent']:.2f}%\n")
                f.write(f"Contour Count: {metrics['contour_count']}\n")
                f.write(f"Total Material Area: {metrics['total_contour_area']} px\n")
                f.write(f"ROI Area: {metrics['total_pixels']} px\n")
            
            print(f"Saved snapshot to {snapshot_filename}")
        
        # Update frame counter
        frame_number += 1
        
    # Clean up
    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv.destroyAllWindows()
    
    print("\nProcessing complete")
    if output_path is not None:
        print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()