import torch
from ultralytics import YOLO
from .base_object import ObjectDetector
from core.config import settings
from inference_engine.preprocessing import dcp_dehaze

class YoloRunner(ObjectDetector):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.device = 'cpu'
        self.use_fuse = settings.YOLO_ENABLE_FUSE
        self.use_half = settings.YOLO_ENABLE_HALF
        self.use_dcp_dehaze = settings.YOLO_ENABLE_DCP_DEHAZE
        
    def load_model(self, model_path: str = ""):
        if model_path:
            self.model_path = model_path
            
        print(f"[INFO] Loading YOLO model from {self.model_path}...")
        try:
            self.model = YOLO(self.model_path)
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            if self.device == 'cuda:0':
                torch.backends.cudnn.benchmark = True

            if self.use_fuse:
                # Fuse Conv+BN blocks in the YOLO graph for improved throughput.
                self.model.fuse()

            if self.use_half and self.device == 'cuda:0':
                # Convert model weights to FP16 for faster CUDA inference.
                self.model.model.half()

            print(f"[INFO] YOLO Runner initialized on device: {self.device} (CuDNN Benchmark: {torch.backends.cudnn.benchmark})")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            raise e
            
    def process_frame(self, frame):
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        if self.use_dcp_dehaze:
            frame = dcp_dehaze(frame)

        use_half = self.use_half and self.device == 'cuda:0'

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
                half=use_half,
            )
        else:
            # Detection-only mode: faster, stateless, used on edge devices
            results = self.model(frame, device=self.device, verbose=False, half=use_half)

        return results[0]
