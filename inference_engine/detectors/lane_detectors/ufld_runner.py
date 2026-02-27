import cv2
import numpy as np
import torch
import onnxruntime as ort
import os

from .base_lane import BaseLaneDetector

class UFLDRunner(BaseLaneDetector):
    def __init__(self, model_path: str = "models/ufldv2_tusimple_res18.onnx"):
        self.model_path = model_path
        self.session = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # UFLD Tusimple config
        self.input_width = 800
        self.input_height = 320
        
        self.original_shape = None

    def load_model(self, model_path: str = None):
        if model_path:
            self.model_path = model_path
            
        print(f"[INFO] Loading UFLD Lane Detection model from {self.model_path}...")
        
        if not os.path.exists(self.model_path):
            print(f"[WARNING] UFLD model not found at {self.model_path}. Placeholder mode active.")
            return

        try:
            if self.model_path.endswith('.onnx'):
                try:
                    ort.preload_dlls()
                except Exception as e:
                    print(f"[WARNING] ONNX DLL preload failed or wasn't needed: {e}")
                    
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.startswith('cuda') else ['CPUExecutionProvider']
                self.session = ort.InferenceSession(self.model_path, providers=providers)
            else:
                self.session = torch.load(self.model_path, map_location=self.device)
                self.session.eval()
            print(f"[INFO] UFLD Runner initialized using {self.device}")
        except Exception as e:
            print(f"[ERROR] Failed to load UFLD model: {e}")

    def process_frame(self, frame):
        """
        Runs UFLD inference and returns a list of lane points.
        """
        self.original_shape = frame.shape[:2] # (H, W)
        
        if self.session is None:
            return []
            
        # 1. Preprocessing
        img = cv2.resize(frame, (self.input_width, int(self.input_height / 0.8)))
        img = img[-self.input_height:, :, :] # Crop bottom 320 rows
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = img.astype(np.float32) / 255.0
        img -= np.array([0.485, 0.456, 0.406])
        img /= np.array([0.229, 0.224, 0.225])
        
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        
        # 2. Inference
        if self.model_path.endswith('.onnx'):
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: img})
            out_loc_row, out_loc_col, out_exist_row, out_exist_col = outputs[:4]
        else:
            with torch.no_grad():
                tensor_img = torch.from_numpy(img).to(self.device)
                pred = self.session(tensor_img)
                out_loc_row = pred['loc_row'].cpu().numpy()
                out_loc_col = pred['loc_col'].cpu().numpy()
                out_exist_row = pred['exist_row'].cpu().numpy()
                out_exist_col = pred['exist_col'].cpu().numpy()
                
        # 3. Post-processing
        num_row, num_col = 56, 41
        row_anchor = np.linspace(160, 710, num_row) / 720
        col_anchor = np.linspace(0, 1, num_col)
        
        b, num_grid_row, num_cls_row, num_lane_row = out_loc_row.shape
        b, num_grid_col, num_cls_col, num_lane_col = out_loc_col.shape
        
        max_indices_row = np.argmax(out_loc_row, axis=1)
        valid_row = np.argmax(out_exist_row, axis=1)
        max_indices_col = np.argmax(out_loc_col, axis=1)
        valid_col = np.argmax(out_exist_col, axis=1)
        
        def softmax(x, axis=0):
            e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return e_x / e_x.sum(axis=axis, keepdims=True)
            
        coords = []
        row_lane_idx = [1, 2]
        col_lane_idx = [0, 3]
        local_width = 1
        original_image_height, original_image_width = self.original_shape
        
        for i in row_lane_idx:
            tmp = []
            if valid_row[0, :, i].sum() > num_cls_row / 2:
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:
                        idx = max_indices_row[0, k, i]
                        all_ind = np.arange(max(0, idx - local_width), min(num_grid_row - 1, idx + local_width) + 1)
                        if len(all_ind) == 0: continue
                        out_tmp = out_loc_row[0, all_ind, k, i]
                        out_tmp = softmax(out_tmp, axis=0) * all_ind
                        out_tmp = out_tmp.sum() + 0.5
                        out_tmp = out_tmp / (num_grid_row - 1) * original_image_width
                        tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            if len(tmp) > 1: coords.append(tmp)
                
        for i in col_lane_idx:
            tmp = []
            if valid_col[0, :, i].sum() > num_cls_col / 4:
                for k in range(valid_col.shape[1]):
                    if valid_col[0, k, i]:
                        idx = max_indices_col[0, k, i]
                        all_ind = np.arange(max(0, idx - local_width), min(num_grid_col - 1, idx + local_width) + 1)
                        if len(all_ind) == 0: continue
                        out_tmp = out_loc_col[0, all_ind, k, i]
                        out_tmp = softmax(out_tmp, axis=0) * all_ind
                        out_tmp = out_tmp.sum() + 0.5
                        out_tmp = out_tmp / (num_grid_col - 1) * original_image_height
                        tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            if len(tmp) > 1: coords.append(tmp)
                
        return coords
