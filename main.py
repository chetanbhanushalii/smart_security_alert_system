# main.py
import cv2
from config import VIDEO_SOURCE
from motion_detector import MotionDetector
from detector import ObjectDetector
from event_mapper import EventMapper
from smoother import TemporalSmoother
from alerter import Alerter

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_SOURCE}")

    motion_detector = MotionDetector()
    object_detector = ObjectDetector()
    event_mapper = EventMapper()
    smoother = TemporalSmoother()
    alerter = Alerter()

    frame_count = 0

    print("Pipeline started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        frame_count += 1
        display_frame = frame.copy()

        # step 1: motion pre-filter
        motion_detected = motion_detector.detect(frame)

        detections = []
        if motion_detected:
            # step 2: YOLO inference only on motion frames
            detections = object_detector.detect(frame)

        # step 3: map to security event 
        event = event_mapper.map(detections)

        # step 4: temporal smoothing for stable alerts
        smoothed_event = smoother.update(event)

        # step 5: alert if event confirmed
        if smoothed_event:
            alerter.handle(smoothed_event, frame, detections)

        # annotate and display live window
        label = smoothed_event or ("Motion..." if motion_detected else "Clear")
        color = (0, 0, 255) if smoothed_event else (0, 255, 0)
        cv2.putText(display_frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imshow("Smart Security Monitor", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()