from core.config import settings

def get_object_detector():
    model_path = settings.MODEL_PATH
    
    # Simple logic to choose between ONNX or PyTorch based on file extension
    if model_path.endswith('.onnx'):
        from .detectors.onnx_runner import ONNXRunner
        detector = ONNXRunner(model_path)
    else:
        from .detectors.yolo_runner import YoloRunner
        detector = YoloRunner(model_path)
        
    detector.load_model()
    return detector

def get_lane_detector():
    # We will default to a UFLD onnx model
    from .detectors.lane_detectors.ufld_runner import UFLDRunner
    detector = UFLDRunner("models/ufldv2_tusimple_res18.onnx")
    detector.load_model()
    return detector
