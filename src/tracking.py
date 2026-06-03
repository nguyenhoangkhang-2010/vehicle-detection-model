# src/tracking.py
from deep_sort_realtime.deepsort_tracker import DeepSort
from src.config import MAX_AGE, N_INIT, MAX_COSINE_DIST

class VehicleTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=MAX_AGE,
            n_init=N_INIT,
            nms_max_overlap=0.8,
            max_cosine_distance=MAX_COSINE_DIST
        )

    def update_trajectories(self, detections, frame):
        return self.tracker.update_tracks(detections, frame=frame)