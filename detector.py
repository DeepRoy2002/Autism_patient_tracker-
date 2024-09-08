from ultralytics import YOLO

class Detector:
    def __init__(self, model_name='yolov8n.pt', device='cuda'):
        self.model = YOLO(model_name)
        self.device = device

    def detect(self, frame):
        results = self.model(frame)
        return results
