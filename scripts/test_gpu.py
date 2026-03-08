import torch
import onnxruntime as ort

print(f"--- GPU Diagnostic ---")
print(f"CUDA Available (PyTorch): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")

try:
    from ultralytics import YOLO
    # Loads the YOLO model to verify which hardware it connects to
    model = YOLO("models/yolov8n.pt")
    print(f"YOLO Ultralytics Device: {model.device}")
except Exception as e:
    print(f"YOLO Initialization Failed: {e}")

print(f"ONNX Runtime Available Providers: {ort.get_available_providers()}")
print(f"----------------------")
