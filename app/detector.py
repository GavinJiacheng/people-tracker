import numpy as np
from ultralytics import YOLO


class PersonDetector:
    def __init__(self, weights: str, conf: float, imgsz: int, person_class: int):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.person_class = person_class

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Return detections in shape: [x1, y1, x2, y2, conf, cls]
        """
        results = self.model.predict(
            frame,
            classes=[self.person_class],
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        detections = np.column_stack((boxes, confs, classes)).astype(np.float32)
        return detections