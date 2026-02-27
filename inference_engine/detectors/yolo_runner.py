import cv2
import torch
from ultralytics import YOLO
from .object_detector import ObjectDetector

class YoloRunner(ObjectDetector):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.device = 'cpu'
        
    def load_model(self, model_path: str = None):
        if model_path:
            self.model_path = model_path
            
        print(f"[INFO] Loading YOLO model from {self.model_path}...")
        try:
            self.model = YOLO(self.model_path)
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            print(f"[INFO] YOLO Runner initialized on device: {self.device}")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            raise e
            
    def process_frame(self, frame):
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
            
        # Run inference specifying the device and silencing the console
        results = self.model(frame, device=self.device, verbose=False)
        return results[0]
