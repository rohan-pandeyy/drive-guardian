import cv2
import numpy as np

class HazeGenerator:
    def __init__(self, fog_color=(200, 200, 220)):
        self.fog_color = fog_color
        # Caching variables to prevent reallocation overhead
        self._cached_shape = None
        self._cached_gradient = None
        self._cached_atmospheric = None

    def apply_haze(self, frame, intensity=0.5):
        if intensity <= 0.01:
            return frame
            
        h, w = frame.shape[:2]
        
        # 1. OPTIMIZATION: Cache the massive matrices so they are only built ONCE
        if self._cached_shape != (h, w):
            # Build the gradient (1.0 at top, 0.0 at bottom)
            gradient = np.linspace(1.0, 0.0, h, dtype=np.float32).reshape(h, 1)
            gradient = np.repeat(gradient, w, axis=1)
            self._cached_gradient = np.expand_dims(gradient, axis=-1) # Shape: (H, W, 1)
            
            # Build the static atmospheric light mask
            self._cached_atmospheric = np.full((h, w, 3), self.fog_color, dtype=np.float32)
            
            self._cached_shape = (h, w)
            print(f"[INFO] HazeGenerator cached matrices for resolution {w}x{h}")

        # 2. FAST MATH: Calculate transmission map using the cached gradient
        # t(x) = 1.0 - (gradient * intensity)
        t_x = 1.0 - (self._cached_gradient * intensity)
        
        # 3. ATMOSPHERIC EQUATION: I(x) = J(x)t(x) + A(1 - t(x))
        frame_float = frame.astype(np.float32)
        hazed_frame = (frame_float * t_x) + (self._cached_atmospheric * (1.0 - t_x))
        
        # 4. OPTIMIZATION: cv2.convertScaleAbs is exponentially faster than np.clip().astype(np.uint8)
        return cv2.convertScaleAbs(hazed_frame)
