import cv2
import numpy as np
from .base_lane import BaseLaneDetector

class OpenCVRunner(BaseLaneDetector):
    def __init__(self):
        self.model_path = "opencv_cpu_heuristics"
        
    def load_model(self, model_path: str = ""):
        """No weights to load for mathematical heuristics!"""
        print("[INFO] OpenCV CPU Runner initialized (0GB VRAM)")
        
    def process_frame(self, frame):
        # 1. Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Gaussian Blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Canny Edge Detection
        edges = cv2.Canny(blur, 50, 150)
        
        # 4. ROI Masking (lower half of the screen)
        height, width = frame.shape[:2]
        mask = np.zeros_like(edges)
        
        # Define a polygon for the region of interest (hood up to horizon)
        polygon = np.array([[
            (0, height),
            (width, height),
            (int(width * 0.55), int(height * 0.6)),
            (int(width * 0.45), int(height * 0.6))
        ]], np.int32)
        
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # 5. Hough Line Transform
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=100,
            maxLineGap=50
        )
        
        # 6. Format to match UFLD output format for draw_all_warnings
        # UFLD outputs lists of contiguous lane points [[x, y], [x, y]...]
        formatted_lanes = []
        if lines is not None:
            left_lines = []
            right_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1: continue
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter out perfectly horizontal lines
                if abs(slope) < 0.5: continue
                
                # In OpenCV, y goes down as it increases. 
                # Negative slope = Left Lane (from hood to vanishing point)
                if slope < 0: 
                    left_lines.append(line[0])
                else: 
                    right_lines.append(line[0])
            
            # To interface with ForwardCollision and LaneDeparture heuristics,
            # just pass the strongest Hough segments mapped into point lists.
            if left_lines:
                l = left_lines[0]
                formatted_lanes.append([[l[0], l[1]], [l[2], l[3]]])
            if right_lines:
                r = right_lines[0]
                formatted_lanes.append([[r[0], r[1]], [r[2], r[3]]])
                
        return formatted_lanes
