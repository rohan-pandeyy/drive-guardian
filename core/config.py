import os
import yaml

class Settings:
    def __init__(self):
        self.PROFILE = os.getenv("DRIVE_GUARDIAN_PROFILE", "high_power")
        
        # These will be loaded from YAML
        self.MODEL_PATH = ""
        self.RESOLUTION = [1920, 1080]
        self.FPS = 30
        self.ENABLE_TRACKING = True
    
    def load_profile(self):
        profile_path = os.path.join(os.path.dirname(__file__), "profiles", f"{self.PROFILE}.yaml")
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                data = yaml.safe_load(f)
                if data:
                    self.MODEL_PATH = data.get("MODEL_PATH", self.MODEL_PATH)
                    self.RESOLUTION = data.get("RESOLUTION", self.RESOLUTION)
                    self.FPS = data.get("FPS", self.FPS)
                    self.ENABLE_TRACKING = data.get("ENABLE_TRACKING", self.ENABLE_TRACKING)
        else:
            print(f"Warning: Profile {profile_path} not found. Using defaults.")

settings = Settings()
settings.load_profile()
