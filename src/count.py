# src/count.py
import time
from datetime import datetime, timedelta
from src.config import CLASS_NAMES, LINE_START, LINE_END, TIME_WINDOW_SECONDS
from src.utils import check_intersection, get_direction

class TrafficAnalyzer:
    def __init__(self):
        self.reference_time = datetime.now()
        self.track_history = {}
        self.counted_registry = set()
        self.total_counts = {cls: 0 for cls in CLASS_NAMES.values()}
        self.log_database = []

    def analyze_flow(self, track, frame_id, fps):
        if not track.is_confirmed():
            return None

        track_id = track.track_id
        cls_id = track.get_det_class()
        class_name = CLASS_NAMES[cls_id]
        
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        current_center = (int((x1 + x2) / 2), int(y2))

        if track_id not in self.track_history:
            self.track_history[track_id] = [current_center]
        else:
            self.track_history[track_id].append(current_center)
            
            if len(self.track_history[track_id]) > 10:
                self.track_history[track_id].pop(0)

        if track_id not in self.counted_registry and len(self.track_history[track_id]) >= 2:
            prev_center = self.track_history[track_id][-2]
            
            if check_intersection(LINE_START, LINE_END, prev_center, current_center):
                self.counted_registry.add(track_id)
                self.total_counts[class_name] += 1
                
                elapsed_seconds = frame_id / fps
                simulated_time = self.reference_time + timedelta(seconds=elapsed_seconds)
                
                direction_flag = get_direction(LINE_START, LINE_END, prev_center, current_center)
                direction_label = "Chieu_Di" if direction_flag == 1 else "Chieu_Ve"

                log_data = {
                    "Exact_Time": simulated_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Time_Slot": simulated_time.strftime("%Y-%m-%d %H:%M"),
                    "Vehicle_ID": track_id,
                    "Vehicle_Type": class_name,
                    "Direction": direction_label
                }
                self.log_database.append(log_data)
                return log_data
        return None