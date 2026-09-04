import os
# pyrefly: ignore [missing-import]
import cv2
import shutil

# Missing cases
missing_cases = ['2601', '2602', '2603', '2604', '2605', '2606', '2607', '2608', '2611', '2613', '2614', '2615']

search_dirs = [os.path.join('data', 'raw'), '.']
output_folder = os.path.join('data', 'filtered_frames')
os.makedirs(output_folder, exist_ok=True)

BLUR_THRESHOLD = 100.0

print("=== Fixing Missing Frames ===")

image_files_with_path = []
for directory in search_dirs:
    if os.path.exists(directory):
        found = [os.path.join(directory, f) for f in os.listdir(directory)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png')) and any(f.startswith(f"{c}_") for c in missing_cases)]
        if found:
            image_files_with_path = found
            break

print(f"Found {len(image_files_with_path)} raw images for missing cases.")

kept_count = 0
dropped_count = 0

for img_path in image_files_with_path:
    filename = os.path.basename(img_path)
    # Identify which case code it belongs to
    case_code = next(c for c in missing_cases if filename.startswith(f"{c}_"))

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
print(f"  Total processed : {len(image_files_with_path)}")
print(f"  Kept (saved)    : {kept_count}")
print(f"  Dropped (blur)  : {dropped_count}")
print(f"  Destination     : {output_folder}")
