from abc import ABC, abstractmethod

class BaseLaneDetector(ABC):
    @abstractmethod
    def load_model(self, model_path: str):
        """Loads the lane detection model into memory."""
        pass
        
    @abstractmethod
    def process_frame(self, frame):
        """
        Takes a raw BGR OpenCV frame, runs lane inference.
        Returns the data needed to draw lanes or trigger warnings.
        """
        pass
