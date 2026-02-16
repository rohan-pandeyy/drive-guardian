import cv2
from ultralytics import YOLO
from .object_detector import ObjectDetector

class YoloRunner(ObjectDetector):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        
    def load_model(self, model_path: str = None):
        if model_path:
            self.model_path = model_path
            
        print(f"Loading YOLO model from {self.model_path}...")
        try:
            # ultralytics abstracting PyTorch loading
            self.model = YOLO(self.model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            raise e
            
    def predict(self, frame):
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
            
        # Run inference
        results = self.model.predict(frame, verbose=False)
        return results
