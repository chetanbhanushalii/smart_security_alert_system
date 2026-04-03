# motion_detector.py
import cv2
from config import MOTION_THRESHOLD

class MotionDetector:
    def __init__(self):
        # MOG2 = Mixture of Gaussians background subtractor
        # handles gradual lighting changes well — good for security cameras
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,          # frames used to build background model
            varThreshold=50,      # sensitivity — lower = more sensitive
            detectShadows=False   # ignore shadows, reduces false positives
        )

    def detect(self, frame) -> bool:
        """Returns True if significant motion is detected in the frame."""
        mask = self.bg_subtractor.apply(frame)

        # morphological operations to clean up noise in the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # only count motion if contour area exceeds threshold
        for contour in contours:
            if cv2.contourArea(contour) > MOTION_THRESHOLD:
                return True
        return False