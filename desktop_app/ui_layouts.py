import customtkinter as ctk
import queue

class DriveGuardianApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("Drive Guardian - Native ADAS")
        self.geometry("1100x700")

        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -- Sidebar --
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Drive Guardian", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Status: Online", text_color="green")
        self.status_label.grid(row=1, column=0, padx=20, pady=10)

        self.settings_button = ctk.CTkButton(self.sidebar_frame, text="Hardware Settings", command=self.open_settings)
        self.settings_button.grid(row=2, column=0, padx=20, pady=10)
        
        self.source_label = ctk.CTkLabel(self.sidebar_frame, text="Video Source:")
        self.source_label.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.source_optionmenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Webcam", "Video File"],
                                                   command=self.change_source_event)
        self.source_optionmenu.grid(row=4, column=0, padx=20, pady=(0, 10))

        self.lane_toggle = ctk.CTkSwitch(self.sidebar_frame, text="Lane Detection", 
                                         command=self.toggle_lane_event)
        self.lane_toggle.grid(row=5, column=0, padx=20, pady=10, sticky="w")
        self.lane_toggle.select()  # Default to ON
        
        self.lane_model_label = ctk.CTkLabel(self.sidebar_frame, text="Lane Model:")
        self.lane_model_label.grid(row=6, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.lane_model_optionmenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["OpenCV (GPU)", "UFLDv2 (ONNX)", "TwinLiteNet+ (Segmentation)"],
                                                       command=self.change_lane_model_event)
        self.lane_model_optionmenu.grid(row=7, column=0, padx=20, pady=(0, 10))
        
        self.yolo_toggle = ctk.CTkSwitch(self.sidebar_frame, text="Object Detection", 
                                         command=self.toggle_yolo_event)
        self.yolo_toggle.grid(row=8, column=0, padx=20, pady=10, sticky="w")
        self.yolo_toggle.select()  # Default to ON
        
        self.yolo_model_label = ctk.CTkLabel(self.sidebar_frame, text="YOLO Model:")
        self.yolo_model_label.grid(row=9, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.yolo_model_optionmenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["yolo12n.pt"],
                                                       command=self.change_yolo_model_event)
        self.yolo_model_optionmenu.grid(row=10, column=0, padx=20, pady=(0, 10))
        
        self.latency_label = ctk.CTkLabel(self.sidebar_frame, text="Latency: -- ms", font=ctk.CTkFont(weight="bold"))
        self.latency_label.grid(row=11, column=0, padx=20, pady=0, sticky="w")
        
        self.fps_label = ctk.CTkLabel(self.sidebar_frame, text="FPS: --", font=ctk.CTkFont(weight="bold"))
        self.fps_label.grid(row=12, column=0, padx=20, pady=10, sticky="w")
        
        self.on_source_change_callback = None
        self.on_lane_toggle_callback = None
        self.on_lane_model_change_callback = None
        self.on_yolo_toggle_callback = None
        self.on_yolo_model_change_callback = None
        
        # -- Main Video Area --
        self.main_frame = ctk.CTkFrame(self, fg_color="black")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # The label that will be updated constantly by the video thread
        self.video_frame = ctk.CTkLabel(self.main_frame, text="Initializing Camera & AI Pipeline...")
        self.video_frame.grid(row=0, column=0, sticky="nsew")
        
        self.video_queue = queue.Queue()
        self.update_video_feed()

    def update_video_feed(self):
        if not self.video_queue.empty():
            # Unpack the image and the latest fps/latency calculation
            new_image, fps_value, latency_value = self.video_queue.get()
            self.video_frame.configure(image=new_image, text="")
            self.fps_label.configure(text=f"FPS: {fps_value}")
            self.latency_label.configure(text=f"Latency: {latency_value} ms")
        
        # Poll the queue every 15ms
        self.after(15, self.update_video_feed)

    def change_source_event(self, new_source: str):
        if self.on_source_change_callback:
            self.on_source_change_callback(new_source)

    def toggle_lane_event(self):
        state = self.lane_toggle.get() == 1
        if self.on_lane_toggle_callback:
            self.on_lane_toggle_callback(state)

    def change_lane_model_event(self, new_model_name: str):
        if self.on_lane_model_change_callback:
            self.on_lane_model_change_callback(new_model_name)

    def toggle_yolo_event(self):
        state = self.yolo_toggle.get() == 1
        if self.on_yolo_toggle_callback:
            self.on_yolo_toggle_callback(state)

    def change_yolo_model_event(self, new_model_name: str):
        if self.on_yolo_model_change_callback:
            self.on_yolo_model_change_callback(new_model_name)

    def open_settings(self):
        print("Settings window placeholder")
