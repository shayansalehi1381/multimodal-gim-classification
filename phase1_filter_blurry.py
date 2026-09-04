import os
# pyrefly: ignore [missing-import]
import cv2
import shutil

# ۱. مسیرهای ورودی و خروجی
input_folder = os.path.join('data', 'extracted_frames')
output_folder = os.path.join('data', 'filtered_frames')
os.makedirs(output_folder, exist_ok=True)

# ۲. آستانه تشخیص تاری (هرچه عدد بیشتر باشد، سخت‌گیری بیشتر است)
BLUR_THRESHOLD = 100.0

print("=== شروع بررسی تاری فریم‌ها (Laplacian Variance) ===")

frame_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
total_frames = len(frame_files)
kept_count = 0
dropped_count = 0

for idx, filename in enumerate(frame_files):
    img_path = os.path.join(input_folder, filename)
    image = cv2.imread(img_path)
    
    if image is None:
        continue
        
    # تبدیل به تصویر خاکستری
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # محاسبه واریانس لپلاسیان (معیار شفافیت)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if score >= BLUR_THRESHOLD:
        # تصویر شفاف است -> کپی به پوشه اصلی
        shutil.copy(img_path, os.path.join(output_folder, filename))
        kept_count += 1
    else:
        dropped_count += 1
        
    if (idx + 1) % 300 == 0 or (idx + 1) == total_frames:
        print(f"⏳ پردازش شد: {idx + 1}/{total_frames} فریم...")

print(f"\n🎉 پالایش فریم‌ها با موفقیت تمام شد!")
print(f"🔹 کل فریم‌ها: {total_frames}")
print(f"✅ فریم‌های شفاف و باکیفیت نگه داشته‌شده: {kept_count}")
print(f"❌ فریم‌های تار حذف‌شده: {dropped_count}")
print(f"📁 تصاویر نهایی در پوشه data/filtered_frames ذخیره شدند.")