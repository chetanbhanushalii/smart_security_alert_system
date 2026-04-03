# config.py

VIDEO_SOURCE = "input.mp4"       # path to your test video file
OUTPUT_DIR = "flagged_frames"    # directory to save flagged frames

# YOLOv8 settings
MODEL_PATH = "yolov8n.pt"        # nano model — fastest, good enough for MVP
CONFIDENCE_THRESHOLD = 0.4       # ignore detections below this confidence
DEVICE = "cpu"                   # cpu / cuda / mps

# Motion detection (background subtraction)
MOTION_THRESHOLD = 500           # minimum contour area to count as motion
                                 # lower = more sensitive, higher = less noise

# Temporal smoothing
SMOOTHING_WINDOW = 5             # number of consecutive frames before firing alert
                                 # prevents single-frame false positives

# YOLO class IDs we care about (COCO dataset)
RELEVANT_CLASSES = {
    0: "person",
    24: "backpack",
    25: "umbrella",
    28: "suitcase",
    15: "cat",
    16: "dog",
}