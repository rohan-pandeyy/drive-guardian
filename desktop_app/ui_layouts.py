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
        
        self.fps_label = ctk.CTkLabel(self.sidebar_frame, text="FPS: --", font=ctk.CTkFont(weight="bold"))
        self.fps_label.grid(row=5, column=0, padx=20, pady=10, sticky="w")
        
        self.on_source_change_callback = None
        
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
            # Unpack the image and the latest fps calculation
            new_image, fps_value = self.video_queue.get()
            self.video_frame.configure(image=new_image, text="")
            self.fps_label.configure(text=f"FPS: {fps_value}")
        
        # Poll the queue every 15ms
        self.after(15, self.update_video_feed)

    def change_source_event(self, new_source: str):
        if self.on_source_change_callback:
            self.on_source_change_callback(new_source)

    def open_settings(self):
        print("Settings window placeholder")
