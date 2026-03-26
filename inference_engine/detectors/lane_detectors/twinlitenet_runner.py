import torch
import cv2
import numpy as np
from types import SimpleNamespace
from .base_lane import BaseLaneDetector
from .twinlitenet.model import TwinLiteNetPlus

class TwinLiteNetRunner(BaseLaneDetector):
    def __init__(self, model_version="medium"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # TwinLiteNetPlus expects an args object with the requested configuration size
        args = SimpleNamespace(config=model_version)
        self.model = TwinLiteNetPlus(args)
        self.model = self.model.to(self.device)
        self.model_path = ""
        
    def load_model(self, model_path: str):
        if not model_path:
            return
            
        print(f"[INFO] Loading TwinLiteNet+ ({model_path}) to {self.device}...")
        self.model_path = model_path
        
        # Load the raw PyTorch State Dictionary
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        
        # Run a warmup inference
        print("[INFO] Warming up TwinLiteNet+ tensors...")
        dummy_input = torch.zeros(1, 3, 320, 640).to(self.device)
        with torch.no_grad():
            self.model(dummy_input)

    def process_frame(self, frame):
        """
        Runs the neural network segmenter on the frame.
        We return both the Lane Lines mask and Drivable Area mask, properly formatted.
        """
        if not self.model:
            return []

        # Resize to network dimensions (TwinLiteNet requires height mod 16 == 0)
        target_size = (640, 320)
        img_resized = cv2.resize(frame, target_size)
        
        # TwinLiteNet expects RGB format [0.0 - 1.0] with NO ImageNet normalization
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = img_rgb.astype(np.float32) / 255.0
        
        img_tensor = img_tensor.transpose((2, 0, 1)) # HWC -> CHW
        img_tensor = np.expand_dims(img_tensor, axis=0) # CHW -> BCHW
        img_tensor = torch.from_numpy(img_tensor).to(self.device)

        # Inference
        with torch.no_grad():
            # out_da = Drivable Area (B, 2, H, W), out_ll = Lane Lines (B, 2, H, W)
            out_da, out_ll = self.model(img_tensor)

        # Apply Argmax to get the class indices (0=Background, 1=Class)
        # Using [0] to extract from the batch dimension
        seg_ll = torch.argmax(out_ll, dim=1)[0].cpu().numpy().astype(np.uint8)
        seg_da = torch.argmax(out_da, dim=1)[0].cpu().numpy().astype(np.uint8)
        
        # Resize output masks back to the original frame resolution
        orig_h, orig_w = frame.shape[:2]
        seg_ll = cv2.resize(seg_ll, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        seg_da = cv2.resize(seg_da, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Since ADAS features expect a list of line geometries, we can extract
        # the boundary points from the segmentation mask using contours.
        # This converts a blob of pixels back into the [[x,y], [x,y]] mathematical format.
        contours, _ = cv2.findContours(seg_ll, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        formatted_lanes = []
        for contour in contours:
            # Drop tiny noise blobs
            if cv2.contourArea(contour) < 50:
                continue
                
            # Squeeze and extract coordinate list
            pts = contour.squeeze(1).tolist()
            if len(pts) > 1:
                formatted_lanes.append(pts)

        return formatted_lanes
