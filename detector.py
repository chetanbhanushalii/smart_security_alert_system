# detector.py
from ultralytics import YOLO
from config import MODEL_PATH, CONFIDENCE_THRESHOLD, DEVICE, RELEVANT_CLASSES

class ObjectDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.device = DEVICE

    def detect(self, frame) -> list[dict]:
        """
        Runs YOLOv8 on a frame.
        Returns list of dicts: {label, confidence, bbox}
        Only returns classes we care about (defined in config).
        """
        results = self.model(frame, device=self.device, verbose=False)
        detections = []

        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)

            if cls_id not in RELEVANT_CLASSES:
                continue
            if conf < CONFIDENCE_THRESHOLD:
                continue

            detections.append({
                "label": RELEVANT_CLASSES[cls_id],
                "confidence": round(conf, 2),
                "bbox": box.xyxy[0].tolist()   # [x1, y1, x2, y2]
            })

        return detections