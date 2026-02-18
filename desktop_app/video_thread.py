import threading
import time
import cv2
from PIL import Image
import customtkinter as ctk

# Import our inference engine
from inference_engine.pipeline import get_detector
from core.config import settings

class VideoThread(threading.Thread):
    def __init__(self, image_label: ctk.CTkLabel):
        super().__init__()
        self.image_label = image_label
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
                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame
                
            # Convert OpenCV BGR to RGB
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(annotated_frame)
            
            # Convert PIL Image to CTkImage
            # Getting dimensions to maintain aspect ratio could be added here
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(800, 600))
            
            # Update the UI canvas from the background thread safely
            # Using after ensures it is pushed to the main loop GUI thread
            self.image_label.after(0, self.update_image, ctk_img)
            
            time.sleep(1 / settings.FPS) # Simple throttle based on profile

    def update_image(self, ctk_img):
        # This runs on the main GUI thread
        self.image_label.configure(image=ctk_img, text="")

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
