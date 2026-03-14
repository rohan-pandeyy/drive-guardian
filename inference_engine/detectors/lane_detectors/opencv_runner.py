import cv2
import numpy as np
from .base_lane import BaseLaneDetector

class OpenCVRunner(BaseLaneDetector):
    def __init__(self):
        self.model_path = "opencv_GPU_heuristics"
        
    def load_model(self, model_path: str = ""):
        """No weights to load for mathematical heuristics!"""
        print("[INFO] OpenCV GPU Runner initialized (0GB VRAM)")
        
    def process_frame(self, frame):
        # 0. Upload frame to GPU via Transparent API (T-API)
        u_frame = cv2.UMat(frame)
        height, width = frame.shape[:2]

        # 1. HSL Color Filtering for White and Yellow
        hsl = cv2.cvtColor(u_frame, cv2.COLOR_BGR2HLS)
        
        # White mask (high lightness)
        lower_white = np.array([0, 200, 0], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsl, lower_white, upper_white)
        
        # Yellow mask (specific hue range)
        lower_yellow = np.array([10, 0, 100], dtype=np.uint8)
        upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hsl, lower_yellow, upper_yellow)
        
        # Combine masks
        mask_hsl = cv2.bitwise_or(white_mask, yellow_mask)
        filtered_frame = cv2.bitwise_and(u_frame, u_frame, mask=mask_hsl)
        
        # 2. Grayscale & Gaussian Blur
        gray = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Canny Edge Detection
        edges = cv2.Canny(blur, 50, 150)
        
        # 4. ROI Masking (lower half of the screen)
        # Allocate zeros mask explicitly for UMat
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Define a polygon for the region of interest (hood up to horizon)
        polygon = np.array([[
            (0, height),
            (width, height),
            (int(width * 0.55), int(height * 0.6)),
            (int(width * 0.45), int(height * 0.6))
        ]], np.int32)
        
        cv2.fillPoly(mask, polygon, 255)
        # Upload the created mask back to the GPU
        u_mask = cv2.UMat(mask)
        masked_edges = cv2.bitwise_and(edges, u_mask)
        
        # 5. Hough Line Transform (GPU Accelerated)
        u_lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=40,
            maxLineGap=150
        )
        
        # Download lines back to GPU for geometry math
        lines = u_lines.get() if u_lines is not None else None
        
        # 6. Extrapolate Lines
        formatted_lanes = []
        if lines is not None:
            left_slopes = []
            left_intercepts = []
            right_slopes = []
            right_intercepts = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1: continue
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter out perfectly horizontal and vertical lines
                if abs(slope) < 0.5 or abs(slope) > 5.0: continue
                
                intercept = y1 - slope * x1
                
                # In OpenCV, y goes down as it increases. 
                # Negative slope = Left Lane (from hood to vanishing point)
                if slope < 0: 
                    left_slopes.append(slope)
                    left_intercepts.append(intercept)
                else: 
                    right_slopes.append(slope)
                    right_intercepts.append(intercept)
            
            # Horizon is at height * 0.6 based on the ROI mask
            y1_extrap = height
            y2_extrap = int(height * 0.6)
            
            # To interface with ForwardCollision and LaneDeparture heuristics,
            # pass the strong extrapolated lines into point lists.
            if left_slopes:
                avg_slope = np.mean(left_slopes)
                avg_intercept = np.mean(left_intercepts)
                x1_extrap = int((y1_extrap - avg_intercept) / avg_slope)
                x2_extrap = int((y2_extrap - avg_intercept) / avg_slope)
                formatted_lanes.append([[x1_extrap, y1_extrap], [x2_extrap, y2_extrap]])
                
            if right_slopes:
                avg_slope = np.mean(right_slopes)
                avg_intercept = np.mean(right_intercepts)
                x1_extrap = int((y1_extrap - avg_intercept) / avg_slope)
                x2_extrap = int((y2_extrap - avg_intercept) / avg_slope)
                formatted_lanes.append([[x1_extrap, y1_extrap], [x2_extrap, y2_extrap]])
                
        return formatted_lanes
