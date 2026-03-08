import cv2
import numpy as np
import onnxruntime as ort
from .base_object import ObjectDetector
from core.config import settings

class ONNXRunner(ObjectDetector):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        
    def load_model(self, model_path: str = None):
        if model_path:
            self.model_path = model_path
            
        print(f"Loading ONNX model from {self.model_path}...")
        
        # Configure threads for optimal performance (helps prevent thread thrashing)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        
        # Try to use GPU if available, fallback to CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        try:
            self.session = ort.InferenceSession(self.model_path, sess_options, providers=providers)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            raise e
            
    def predict(self, frame):
        if self.session is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
            
        # VERY basic ONNX preprocessing (needs to be adapted to specific YOLO version)
        input_name = self.session.get_inputs()[0].name
        
        # Resize to 640x640 (standard YOLO size)
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # HWC to CHW format
        img = img.transpose((2, 0, 1))
        # Add batch dimension and normalize
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        
        # Run inference
        outputs = self.session.run(None, {input_name: img})
        return outputs
