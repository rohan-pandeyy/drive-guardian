import torch
from ultralytics import YOLO
from .object_detector import ObjectDetector
from core.config import settings

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

        if settings.ENABLE_TRACKING:
            # Tracking mode: each detected object gets a persistent integer ID
            # across frames. persist=True is required to maintain tracker state
            # between calls in a frame loop — without it IDs reset every frame.
            results = self.model.track(
                frame,
                persist=True,
                tracker=settings.TRACKER,
                device=self.device,
                verbose=False,
            )
        else:
            # Detection-only mode: faster, stateless, used on edge devices
            results = self.model(frame, device=self.device, verbose=False)

        return results[0]
