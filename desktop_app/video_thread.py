import threading
import time
import cv2
import queue
import numpy as np
from PIL import Image
import customtkinter as ctk

# Import our inference engine
from inference_engine.pipeline import get_object_detector, get_lane_detector
from inference_engine.features.lane_departure import LaneDepartureWarning
from inference_engine.features.collision_warn import ForwardCollisionWarning
from core.config import settings

class VideoThread(threading.Thread):
    def __init__(self, video_queue: queue.Queue):
        super().__init__()
        self.video_queue = video_queue
        self.running = False
        self.source = 0
        self.source_changed = False
        self.source_lock = threading.Lock()
        
        try:
            self.yolo_detector = get_object_detector()
            self.lane_detector = get_lane_detector()
            self.ldw = LaneDepartureWarning(drift_threshold=50)
            self.fcw = ForwardCollisionWarning(critical_area_threshold=0.15)
            self.cap = cv2.VideoCapture(self.source)
        except Exception as e:
            print(f"Error initializing detector or camera: {e}")
            self.yolo_detector = None
            self.lane_detector = None
            self.ldw = None
            self.fcw = None
            self.cap = None

    def change_source(self, new_source):
        with self.source_lock:
            self.source = new_source
            self.source_changed = True

    def run(self):
        self.running = True
        
        # FPS Tracking variables
        prev_frame_time = 0
        new_frame_time = 0
        
        while self.running:
            with self.source_lock:
                if self.source_changed:
                    if self.cap:
                        self.cap.release()
                    self.cap = cv2.VideoCapture(self.source)
                    self.source_changed = False

            if not self.cap or not self.cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                # If it's a video file, loop it. If webcam, just wait.
                if isinstance(self.source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    time.sleep(0.1)
                continue
                
            # 1. Run Object Detection (GPU)
            yolo_results = self.yolo_detector.process_frame(frame) if self.yolo_detector else None
            
            # 2. Run Lane Detection (GPU/ONNX)
            lane_results = self.lane_detector.process_frame(frame) if self.lane_detector else None
            
            # 3. Combine Overlays and Compute ADAS Warnings
            annotated_frame = self.draw_all_warnings(frame, yolo_results, lane_results)
                
            # Convert OpenCV BGR to RGB
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(annotated_frame)
            
            # Convert PIL Image to CTkImage
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(800, 600))
            
            # FPS Calculation
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time
            fps_display_val = int(fps)
            
            # Drop stale frames — only keep the latest
            try:
                self.video_queue.put_nowait((ctk_img, fps_display_val))
            except queue.Full:
                try:
                    self.video_queue.get_nowait()
                except queue.Empty:
                    pass
                self.video_queue.put_nowait((ctk_img, fps_display_val))
            
            time.sleep(1 / settings.FPS) # Simple throttle based on profile

    def draw_all_warnings(self, frame, yolo_results, lane_results):
        """ Combines object detection boxes and lane detection lines onto the frame """
        annotated_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        # Draw YOLO Boxes
        if yolo_results is not None:
            # Using ultralytics built-in plotter
            annotated_frame = yolo_results.plot(img=annotated_frame)
            
        # Draw Lane Lines
        if lane_results:
            for lane in lane_results:
                if len(lane) > 1: # Make sure we have points to connect
                    pts = np.array(lane, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    # Draw a thick green line for the lane
                    cv2.polylines(annotated_frame, [pts], isClosed=False, color=(0, 255, 0), thickness=5)
                    # Draw red dots on the anchor points
                    for point in lane:
                        cv2.circle(annotated_frame, tuple(point), radius=4, color=(0, 0, 255), thickness=-1)

        # Evaluate ADAS Features
        if self.ldw and self.fcw:
            is_drifting, ldw_msg = self.ldw.evaluate(lane_results, frame_width, frame_height)
            is_crashing, fcw_msg = self.fcw.evaluate(yolo_results, lane_results, frame_width, frame_height)
            
            if is_drifting:
                cv2.putText(annotated_frame, ldw_msg, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

            if is_crashing:
                cv2.putText(annotated_frame, fcw_msg, (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
                
        return annotated_frame

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
