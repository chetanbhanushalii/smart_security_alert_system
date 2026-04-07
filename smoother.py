# smoother.py
from collections import deque
from config import SMOOTHING_WINDOW

class TemporalSmoother:
    def __init__(self):
        # sliding window of recent events
        self.window = deque(maxlen=SMOOTHING_WINDOW)

    def update(self, event: str | None) -> str | None:
        """
        Add latest event to window.
        Only return an alert if the same event dominates the window.
        This prevents single-frame false positives from firing alerts.
        """
        self.window.append(event)

        if len(self.window) < SMOOTHING_WINDOW:
            return None   # not enough frames yet

        # count most common event in window
        from collections import Counter
        counts = Counter(e for e in self.window if e is not None)

        if not counts:
            return None

        most_common_event, freq = counts.most_common(1)[0]

        # fire alert only if dominant event appears in majority of window
        if freq >= (SMOOTHING_WINDOW // 2 + 1):
            return most_common_event

        return None