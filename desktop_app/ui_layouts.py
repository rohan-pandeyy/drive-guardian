import customtkinter as ctk

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
        
        # -- Main Video Area --
        self.main_frame = ctk.CTkFrame(self, fg_color="black")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # The label that will be updated constantly by the video thread
        self.video_frame = ctk.CTkLabel(self.main_frame, text="Initializing Camera & AI Pipeline...")
        self.video_frame.grid(row=0, column=0, sticky="nsew")

    def open_settings(self):
        print("Settings window placeholder")
