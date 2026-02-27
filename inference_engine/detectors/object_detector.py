from abc import ABC, abstractmethod

class ObjectDetector(ABC):
    @abstractmethod
    def load_model(self, model_path: str):
        pass
        
    @abstractmethod
    def process_frame(self, frame):
        pass
