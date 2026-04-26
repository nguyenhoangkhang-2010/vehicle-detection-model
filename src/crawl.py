import cv2
import os
import glob

def extract_optimal_frames(video_folder, output_folder, seconds_between_frames=2):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    video_dir = os.path.join(current_dir, "..", video_folder)
    out_dir = os.path.join(current_dir, "..", output_folder)
    
    video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))
    if not video_paths:
        print(f"Không tìm thấy video .mp4 nào tại: {video_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    count = 0

    print(f"Bắt đầu vắt ảnh từ {len(video_paths)} video. Mỗi {seconds_between_frames} giây lấy 1 tấm!")

    for vid_path in video_paths:
        cap = cv2.VideoCapture(vid_path)
        fps = round(cap.get(cv2.CAP_PROP_FPS)) 
        
        step = fps * seconds_between_frames 
        
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_id % step == 0:
                img_name = os.path.join(out_dir, f"huit_traffic_{count:04d}.jpg")
                cv2.imwrite(img_name, frame)
                count += 1
                
            frame_id += 1
        cap.release()

    print(f"Đã lưu {count} ảnh (chống trùng lặp 100%) vào {out_dir}")

if __name__ == "__main__":
    extract_optimal_frames("data/raw/videos", "data/raw/images", seconds_between_frames=2)