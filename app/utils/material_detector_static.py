import cv2 as cv
import numpy as np
from collections import namedtuple

FrameResult = namedtuple('FrameResult', ['original', 'mask', 'binary'])
ContourResult = namedtuple('ContourResult', ['mask', 'contours', 'metrics'])

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
    def __init__(self, 
                 history=500, var_threshold=20, detect_shadows=True,
                 nmixtures=5, background_ratio=0.9, learning_rate=0.01,
                 pre_process=None, pre_kernel_size=5, pre_iterations=1,
                 post_process=None, post_kernel_size=5, post_iterations=1):
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.nmixtures = nmixtures
        self.background_ratio = background_ratio
        self.learning_rate = learning_rate
        self.pre_process = pre_process
        self.pre_kernel_size = pre_kernel_size
        self.pre_iterations = pre_iterations
        self.post_process = post_process
        self.post_kernel_size = post_kernel_size
        self.post_iterations = post_iterations
        
        self.bg_subtractor = cv.createBackgroundSubtractorMOG2(
            history=self.history, varThreshold=self.var_threshold, detectShadows=self.detect_shadows
        )
        self.bg_subtractor.setNMixtures(self.nmixtures)
        self.bg_subtractor.setBackgroundRatio(self.background_ratio)
        
        self.pre_kernel = np.ones((self.pre_kernel_size, self.pre_kernel_size), np.uint8)
        self.post_kernel = np.ones((self.post_kernel_size, self.post_kernel_size), np.uint8)
    
    def get_params(self):
        """Mengembalikan dictionary dari parameter saat ini."""
        return {
            'history': self.history,
            'var_threshold': self.var_threshold,
            'detect_shadows': self.detect_shadows,
            'nmixtures': self.nmixtures,
            'background_ratio': self.background_ratio,
            'learning_rate': self.learning_rate,
            'pre_process': self.pre_process,
            'pre_kernel_size': self.pre_kernel_size,
            'pre_iterations': self.pre_iterations,
            'post_process': self.post_process,
            'post_kernel_size': self.post_kernel_size,
            'post_iterations': self.post_iterations
        }

    def apply_morphological_ops(self, image, morph_type, kernel, iterations):
        if morph_type is None: return image
        if morph_type == 'open': return cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=iterations)
        if morph_type == 'close': return cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=iterations)
        if morph_type == 'dilate': return cv.dilate(image, kernel, iterations=iterations)
        if morph_type == 'erode': return cv.erode(image, kernel, iterations=iterations)
        return image
    
    def process_frame(self, frame):
        frame_to_process = frame
        if self.pre_process is not None:
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) if len(frame.shape) > 2 else frame
            pre_processed = self.apply_morphological_ops(gray_frame, self.pre_process, self.pre_kernel, self.pre_iterations)
            if len(frame.shape) > 2:
                pre_processed = cv.cvtColor(pre_processed, cv.COLOR_GRAY2BGR)
            frame_to_process = pre_processed
            
        fg_mask = self.bg_subtractor.apply(frame_to_process, learningRate=self.learning_rate)
        
        processed_mask = self.apply_morphological_ops(fg_mask, self.post_process, self.post_kernel, self.post_iterations) if self.post_process else fg_mask
        
        binary_mask = cv.threshold(processed_mask, 127, 255, cv.THRESH_BINARY)[1]
        display_mask = cv.cvtColor(binary_mask, cv.COLOR_GRAY2BGR)    
        
        return FrameResult(original=frame_to_process, mask=display_mask, binary=binary_mask)

class ContourProcessor:
    def __init__(self, 
                 min_contour_area=100, use_convex_hull=True, merge_overlapping=False,     
                 merge_distance=10, contour_color=(0, 255, 0), hull_color=(0, 200, 255),    
                 text_color=(0, 0, 255), show_contour_index=False, show_contour_area=False):
        self.min_contour_area = min_contour_area
        self.use_convex_hull = use_convex_hull
        self.merge_overlapping = merge_overlapping
        self.merge_distance = merge_distance
        self.contour_color = contour_color
        self.hull_color = hull_color
        self.text_color = text_color
        self.show_contour_index = show_contour_index
        self.show_contour_area = show_contour_area
    
    def get_params(self):
        """Mengembalikan dictionary dari parameter saat ini."""
        return {
            'min_contour_area': self.min_contour_area,
            'use_convex_hull': self.use_convex_hull,
            'merge_overlapping': self.merge_overlapping,
            'merge_distance': self.merge_distance,
            'show_contour_index': self.show_contour_index,
            'show_contour_area': self.show_contour_area
        }
    
    def process_mask(self, binary_mask):
        mask = cv.cvtColor(binary_mask, cv.COLOR_BGR2GRAY) if len(binary_mask.shape) > 2 else binary_mask
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv.contourArea(cnt) >= self.min_contour_area]
        processed_contours = [cv.convexHull(cnt) for cnt in filtered_contours] if self.use_convex_hull else filtered_contours
        if self.merge_overlapping and len(processed_contours) > 1:
            processed_contours = self._merge_close_contours(processed_contours, self.merge_distance)
        result_mask = np.zeros_like(mask)
        cv.drawContours(result_mask, processed_contours, -1, 255, -1)
        metrics = self._calculate_metrics(mask, result_mask, processed_contours)
        return ContourResult(mask=result_mask, contours=processed_contours, metrics=metrics)
    
    def _merge_close_contours(self, contours, max_distance):
        if not contours: return []
        all_points = np.vstack([cnt.reshape(-1, 2) for cnt in contours])
        min_x, min_y = all_points.min(axis=0)
        max_x, max_y = all_points.max(axis=0)
        padding = max_distance * 2
        min_x, min_y = max(0, min_x - padding), max(0, min_y - padding)
        width, height = int(max_x - min_x + 1 + padding*2), int(max_y - min_y + 1 + padding*2)
        if width <= 0 or height <= 0: return contours
        mask = np.zeros((height, width), dtype=np.uint8)
        shifted_contours = [cnt - np.array([min_x, min_y]) for cnt in contours]
        cv.drawContours(mask, shifted_contours, -1, 255, -1)
        kernel = np.ones((max_distance, max_distance), np.uint8)
        dilated = cv.dilate(mask, kernel, iterations=1)
        merged_contours_shifted, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return [cnt + np.array([min_x, min_y]) for cnt in merged_contours_shifted]
    
    def _calculate_metrics(self, original_mask, processed_mask, contours):
        total_area = original_mask.shape[0] * original_mask.shape[1]
        if total_area == 0: return {}
        total_contour_area = sum(cv.contourArea(cnt) for cnt in contours)
        return {
            'contour_count': len(contours),
            'total_contour_area': total_contour_area,
            'contour_coverage_percent': (total_contour_area / total_area) * 100,
            'processed_coverage_percent': (cv.countNonZero(processed_mask) / total_area) * 100
        }
    
    def visualize(self, image, contours, metrics, show_metrics=True, scale_factor=1.0):
        vis_image = image.copy()
        cv.drawContours(vis_image, contours, -1, self.contour_color, 2)
        if self.use_convex_hull:
            hulls = [cv.convexHull(cnt) for cnt in contours]
            cv.drawContours(vis_image, hulls, -1, self.hull_color, 1)
        for i, cnt in enumerate(contours):
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                if self.show_contour_index:
                    cv.putText(vis_image, f"{i}", (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 0.5 * scale_factor, self.text_color, 2)
                if self.show_contour_area:
                    cv.putText(vis_image, f"{int(cv.contourArea(cnt))}", (cx, cy + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5 * scale_factor, self.text_color, 2)
        return vis_image
