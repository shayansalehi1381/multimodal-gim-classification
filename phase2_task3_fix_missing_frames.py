import os
import cv2
import shutil

# Missing cases
missing_cases = ['2601', '2602', '2603', '2604', '2605', '2606', '2607', '2608', '2611', '2613', '2614', '2615']

input_folder = '.'
output_folder = os.path.join('data', 'filtered_frames')
os.makedirs(output_folder, exist_ok=True)

BLUR_THRESHOLD = 100.0

print("=== Fixing Missing Frames ===")

# Find all image files in root matching the missing cases
image_files = []
for f in os.listdir(input_folder):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Check if starts with one of the missing case codes
        if any(f.startswith(c) for c in missing_cases):
            image_files.append(f)

print(f"Found {len(image_files)} raw images for missing cases.")

kept_count = 0
dropped_count = 0

for filename in image_files:
    # Identify which case code it belongs to
    case_code = next(c for c in missing_cases if filename.startswith(c))
    
    img_path = os.path.join(input_folder, filename)
    image = cv2.imread(img_path)
    
    if image is None:
        continue
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if score >= BLUR_THRESHOLD:
        # Save with CASE_XXXX_ prefix so the regex catches it!
        # Example: CASE_2601_2601_1_LBC...jpg
        new_filename = f"CASE_{case_code}_{filename}"
        shutil.copy(img_path, os.path.join(output_folder, new_filename))
        kept_count += 1
    else:
        dropped_count += 1

print(f"\nFiltering complete:")
print(f"  Total processed : {len(image_files)}")
print(f"  Kept (saved)    : {kept_count}")
print(f"  Dropped (blur)  : {dropped_count}")
