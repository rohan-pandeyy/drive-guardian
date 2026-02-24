import threading
import time
import cv2
import queue
from PIL import Image
import customtkinter as ctk

# Import our inference engine
from inference_engine.pipeline import get_detector
from core.config import settings

class VideoThread(threading.Thread):
    def __init__(self, video_queue: queue.Queue):
        super().__init__()
        self.video_queue = video_queue
        self.running = False
        
        try:
            self.detector = get_detector()
            self.cap = cv2.VideoCapture(0) # Change to a video path if you prefer
        except Exception as e:
            print(f"Error initializing detector or camera: {e}")
            self.detector = None
            self.cap = None

    def run(self):
        self.running = True
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Run inference if detector is ready
            if self.detector:
                results = self.detector.predict(frame)
                annotated_frame = results.plot()
            else:
                annotated_frame = frame
                
            # Convert OpenCV BGR to RGB
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(annotated_frame)
            
            # Convert PIL Image to CTkImage
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(800, 600))
            
            # Put the image into the queue safely
            self.video_queue.put(ctk_img)
            
            time.sleep(1 / settings.FPS) # Simple throttle based on profile

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
