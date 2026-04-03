# Smart Security Alert System

Real-time computer vision pipeline for home security event detection.

## Architecture
Camera Feed → Motion Pre-filter → YOLOv8 Detection → Event Mapping → Temporal Smoothing → Alert

## Features
- Background subtraction as motion pre-filter (reduces unnecessary inference)
- YOLOv8n for real-time object detection
- Event classification: Visitor, Package Delivery, Pet Movement, Anomalous Motion
- Temporal smoothing to suppress false positives
- Annotated video output with flagged frame saving

## Setup
```bash
pip install ultralytics opencv-python numpy
python main.py
```

## Tradeoffs & Design Decisions
- **YOLOv8n over larger models**: prioritizes latency over marginal accuracy gains
- **Background subtraction pre-filter**: reduces YOLO inference by ~60-70% on static scenes
- **Temporal smoothing window (5 frames)**: tunable via config.py
- **Next steps**: ResNet fine-tuning on domain-specific data, TensorRT optimization for edge deployment