import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import customtkinter as ctk
from desktop_app.ui_layouts import DriveGuardianApp
from desktop_app.video_thread import VideoThread

def main():
    # Set the overall appearance of the CustomTkinter app
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")

    # Initialize the main UI window
    app = DriveGuardianApp()
    
    # Initialize and start the background video thread (which runs the ML inference)
    # We pass the queue so the thread can push un-blocking updates to the main UI
    video_thread = VideoThread(app.video_queue)
    video_thread.start()

    # Setup source change callback
    def handle_source_change(new_source):
        if new_source == "Video File":
            file_path = ctk.filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")]
            )
            if file_path:
                video_thread.change_source(file_path)
            else:
                # Revert UI option if canceled
                app.source_optionmenu.set("Webcam")
                video_thread.change_source(0)
        else:
            video_thread.change_source(0)

    app.on_source_change_callback = handle_source_change

    def handle_lane_toggle(state):
        video_thread.toggle_lane_detection(state)
        
    app.on_lane_toggle_callback = handle_lane_toggle

    # Graceful shutdown handling
    def on_closing():
        video_thread.stop()
        app.destroy()
        
    app.protocol("WM_DELETE_WINDOW", on_closing)

    # Start the native UI event loop
    app.mainloop()

if __name__ == "__main__":
    main()
