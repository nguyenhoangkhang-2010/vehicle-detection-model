import cv2
import os
import argparse
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from src.config import *
from src.detect import YOLODetector
from src.tracking import VehicleTracker
from src.count import TrafficAnalyzer

class MediaProcessor(ABC):
    def __init__(self, detector: YOLODetector, input_path: str):
        self.detector = detector
        self.input_path = input_path

    @abstractmethod
    def process(self) -> None:
        pass

class ImageTrafficProcessor(MediaProcessor):
    def __init__(self, detector: YOLODetector, input_path: str):
        super().__init__(detector, input_path)
        self.image_counts = {cls: 0 for cls in CLASS_NAMES.values()}

    def process(self) -> None:
        print(f"[HỆ THỐNG] Phát hiện định dạng ẢNH. Đang xử lý: {self.input_path}")
        frame = cv2.imread(self.input_path)
        if frame is None:
            print(f"Lỗi: Không thể nạp file ảnh tại {self.input_path}")
            return

        detections = self.detector.extract_boxes(frame)
        
        for box_data, conf, cls_id in detections:
            class_name = CLASS_NAMES[cls_id]
            self.image_counts[class_name] += 1
            
            x, y, w, h = box_data
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            color = CLASS_COLORS.get(cls_id, (255,255,255))
            
            cv2.rectangle(frame, (0, 0), (200, 120), (0, 0, 0), -1)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.rectangle(frame, (15, 15), (320, 180), (0, 0, 0), -1)
        cv2.putText(frame, "STATIC DENSITY ANALYSIS", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y_pos = 70
        for name, count in self.image_counts.items():
            cv2.putText(frame, f"• {name.upper()}: {count}", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_pos += 25

        base_name = os.path.basename(self.input_path)
        output_path = f"results/img/result_{base_name}"
        cv2.imwrite(output_path, frame)
        print(f"Đã lưu ảnh kết quả tại: {output_path}")
        
        self._export_report()

    def _export_report(self) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_data = [{"Analysis_Time": timestamp, "Source_Image": self.input_path, "Vehicle_Type": k, "Total_Detected": v} 
                       for k, v in self.image_counts.items()]
        df_img = pd.DataFrame(report_data)
        df_img.to_csv("results/img/image_density_report.csv", index=False, encoding='utf-8-sig')
        print(f"Đã cập nhật báo cáo mật độ ảnh tại: results/img/image_density_report.csv\n")
        print(df_img.to_string(index=False))


class VideoTrafficProcessor(MediaProcessor):
    def __init__(self, detector: YOLODetector, input_path: str):
        super().__init__(detector, input_path)
        self.tracker = VehicleTracker()
        self.analyzer = TrafficAnalyzer()

    def process(self) -> None:
        print(f"[HỆ THỐNG] Phát hiện định dạng VIDEO. Đang xử lý: {self.input_path}")
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            print(f"Lỗi: Không thể nạp video đầu vào tại {self.input_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        
        base_name = os.path.basename(self.input_path)
        output_path = f"results/video/result_{base_name}"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            detections = self.detector.extract_boxes(frame)
            tracks = self.tracker.update_trajectories(detections, frame)

            cv2.line(frame, LINE_START, LINE_END, (0, 0, 255), 3)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                self.analyzer.analyze_flow(track, frame_id, fps)
                
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                cls_id = int(track.get_det_class())
                class_name = CLASS_NAMES[cls_id]
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name} #{track.track_id}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.rectangle(frame, (15, 15), (320, 180), (0, 0, 0), -1)
            cv2.putText(frame, "TRAFFIC MONITOR SYSTEM", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos = 70
            for name, count in self.analyzer.total_counts.items():
                cv2.putText(frame, f"• {name.upper()}: {count}", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_pos += 25

            out.write(frame)

        cap.release()
        out.release()
        print(f"Đã lưu video kết quả tại: {output_path}")
        self._generate_analytics()

    def _generate_analytics(self) -> None:
        if self.analyzer.log_database:
            df_raw = pd.DataFrame(self.analyzer.log_database)
            df_raw.to_csv(REPORT_OUTPUT, index=False, encoding='utf-8-sig')
            frequency_df = df_raw.groupby(['Time_Slot', 'Vehicle_Type']).size().unstack(fill_value=0)
            frequency_df.to_csv("results/video/traffic_density_analysis.csv", encoding='utf-8-sig')
            print("Báo cáo tần suất lưu thông video đã được cập nhật tại thư mục results/video/.")
class TrafficMonitorSystem:
    def __init__(self):
        self.detector = YOLODetector(MODEL_PATH, imgsz=IMG_SIZE, conf_threshold=CONF_THRESHOLD)

    def run(self, input_source: str) -> None:
        if not os.path.exists(input_source):
            print(f"Lỗi: Đường dẫn '{input_source}' không tồn tại. Vui lòng kiểm tra lại!")
            return

        file_extension = os.path.splitext(input_source)[1].lower()
        
        if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            processor = ImageTrafficProcessor(self.detector, input_source)
        else:
            processor = VideoTrafficProcessor(self.detector, input_source)
            
        processor.process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hệ thống YOLO26 Giám sát & Đo lường giao thông")
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Đường dẫn tới file Ảnh hoặc file Video cần xử lý")
    
    args = parser.parse_args()
    app = TrafficMonitorSystem()
    app.run(input_source=args.input)