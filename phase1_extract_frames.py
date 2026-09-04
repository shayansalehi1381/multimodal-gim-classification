import os
import cv2
import re

# ۱. تعریف مسیرها
output_folder = os.path.join('data', 'extracted_frames')
os.makedirs(output_folder, exist_ok=True)

# استخراج کد ۴ رقمی بیمار
def get_case_id(filename):
    match = re.search(r'^(\d{4})_', filename)
    return match.group(1) if match else "Unknown"

print("=== شروع استخراج فریم از ویدیوها (۱ فریم در ثانیه) ===")

search_dirs = [os.path.join('data', 'raw'), '.']
video_exts = ('.mp4', '.mkv', '.avi', '.mov')
video_files = []
dataset_folder = '.'
for directory in search_dirs:
    if os.path.exists(directory):
        found = [f for f in os.listdir(directory) if f.lower().endswith(video_exts)]
        if found:
            dataset_folder = directory
            video_files = found
            break
print(f"🎬 تعداد کل ویدیوهای پیدا شده: {len(video_files)}")

total_saved_frames = 0

for video_file in video_files:
    case_id = get_case_id(video_file)
    video_path = os.path.join(dataset_folder, video_file)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25
        
    saved_frames = 0
    frame_idx = 0
    video_base_name = os.path.splitext(video_file)[0]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % fps == 0:
            frame_filename = f"CASE_{case_id}_{video_base_name}_frame{saved_frames:03d}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_filename), frame)
            saved_frames += 1
            total_saved_frames += 1
            
        frame_idx += 1
        
    cap.release()
    print(f"✅ ویدیو {video_file} (Case {case_id}): {saved_frames} فریم ذخیره شد.")

print(f"\n🎉 عملیات با موفقیت انجام شد! مجموعاً {total_saved_frames} فریم ذخیره گردید.")