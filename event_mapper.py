# event_mapper.py

class EventMapper:
    def map(self, detections: list[dict]) -> str | None:
        """
        Maps a list of YOLO detections to a meaningful security event.
        Returns event string or None if nothing significant detected.
        """
        labels = [d["label"] for d in detections]

        if not labels:
            return None

        # anomalous motion: person detected with luggage
        if "person" in labels and (
            "suitcase" in labels or "backpack" in labels
        ):
            return "PACKAGE_DELIVERY"

        # visitor: person at door with no luggage
        if "person" in labels:
            return "VISITOR_PRESENCE"

        # pet movement
        if "cat" in labels or "dog" in labels:
            return "PET_MOVEMENT"

        return "ANOMALOUS_MOTION"