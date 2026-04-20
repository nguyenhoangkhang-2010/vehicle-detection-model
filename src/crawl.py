import cv2
import os
import glob

def extract_frames_from_videos(video_folder, output_folder, total_images_needed=300):
    video_paths = glob.glob(os.path.join(video_folder, "*.mp4"))
    if not video_paths:
        print("Không tìm thấy video nào!")
        return

    os.makedirs(output_folder, exist_ok=True)
    images_per_video = total_images_needed // len(video_paths)
    count = 0

    print(f"Bắt đầu trích xuất {total_images_needed} ảnh từ {len(video_paths)} video...")

    for vid_path in video_paths:
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        step = max(1, total_frames // images_per_video)
        frame_id = 0
        extracted_from_this_vid = 0

        while cap.isOpened() and extracted_from_this_vid < images_per_video:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_id % step == 0:
                img_name = os.path.join(output_folder, f"traffic_frame_{count:04d}.jpg")
                cv2.imwrite(img_name, frame)
                count += 1
                extracted_from_this_vid += 1
                
            frame_id += 1
        cap.release()

    print(f"Hoàn tất! Đã lưu {count} ảnh vào {output_folder}")

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    VIDEO_DIR = os.path.join(CURRENT_DIR, "..", "data", "raw", "videos")
    OUTPUT_DIR = os.path.join(CURRENT_DIR, "..", "data", "raw", "images")
    
    print(f"Đang tìm video tại: {VIDEO_DIR}")
    
    extract_frames_from_videos(VIDEO_DIR, OUTPUT_DIR)