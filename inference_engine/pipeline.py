from core.config import settings

def get_object_detector():
    model_path = settings.MODEL_PATH
    
    if model_path.endswith('.onnx'):
        from .detectors.object_detectors.onnx_runner import ONNXRunner
        detector = ONNXRunner(model_path)
    else:
        from .detectors.object_detectors.yolo_runner import YoloRunner
        detector = YoloRunner(model_path)
        
    detector.load_model()
    return detector

def get_lane_detector():
    # Accelerate baseline OpenCV heuristics via GPU
    from .detectors.lane_detectors.opencv_runner import OpenCVRunner
    detector = OpenCVRunner()
    detector.load_model("")
    return detector
