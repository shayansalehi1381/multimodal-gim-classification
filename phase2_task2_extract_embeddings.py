"""
Task 2.2 – Extract BiomedCLIP Visual Embeddings from Filtered Frames
=====================================================================
Model  : hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
Input  : data/filtered_frames/  (1,976 quality-filtered JPG frames)
Outputs:
  - data/processed/frame_embeddings_1976x512.npy   (float32 matrix)
  - data/processed/frame_metadata.csv              (filename / case_code / idx)
"""

import os
import sys
import re
import csv
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ── Imports ──────────────────────────────────────────────────────────────────
try:
    # pyrefly: ignore [missing-import]
    import numpy as np
    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    import torch.nn.functional as F
    # pyrefly: ignore [missing-import]
    import open_clip
    from PIL import Image
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}")
    print("  Run: .venv/Scripts/python.exe -m pip install open_clip_torch torch pillow numpy")
    sys.exit(1)

print("=" * 70)
print("   Task 2.2 – BiomedCLIP Visual Embedding Extraction (1,976 frames)")
print("=" * 70)

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_ID       = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
INPUT_DIR      = os.path.join("data", "filtered_frames")
OUTPUT_DIR     = os.path.join("data", "processed")
EMB_PATH       = os.path.join(OUTPUT_DIR, "frame_embeddings_1976x512.npy")
META_PATH      = os.path.join(OUTPUT_DIR, "frame_metadata.csv")
BATCH_SIZE     = 32          # process frames in mini-batches for speed
IMG_EXTS       = {".jpg", ".jpeg", ".png"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1: Device ───────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[1/5] Compute device  : {device}")
if device.type == "cuda":
    print(f"      GPU             : {torch.cuda.get_device_name(0)}")

# ── Step 2: Load model ───────────────────────────────────────────────────────
print(f"\n[2/5] Loading BiomedCLIP model (weights cached from Task 2.1) …")
t0 = time.time()
model, _, preprocess_val = open_clip.create_model_and_transforms(MODEL_ID)
model = model.to(device).eval()
print(f"      Model ready in {time.time() - t0:.1f}s  |  device={device}")

# ── Step 3: Collect image paths ──────────────────────────────────────────────
print(f"\n[3/5] Scanning input directory: {INPUT_DIR}")
all_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if os.path.splitext(f)[1].lower() in IMG_EXTS
])
n_total = len(all_files)
print(f"      Found {n_total} image files")

if n_total == 0:
    print("[ERROR] No images found in filtered_frames. Aborting.")
    sys.exit(1)

# Case code parser: matches CASE_XXXX_ prefix
CASE_RE = re.compile(r'^CASE_(\d+)_')

def parse_case_code(filename: str) -> str:
    m = CASE_RE.match(filename)
    return m.group(1) if m else "unknown"

# ── Step 4: Extract embeddings in batches ────────────────────────────────────
print(f"\n[4/5] Extracting embeddings  (batch_size={BATCH_SIZE}, dim=512) …")

all_embeddings = np.zeros((n_total, 512), dtype=np.float32)
metadata_rows  = []           # (filename, case_code, embedding_idx)

n_batches   = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
errors      = []
t_start     = time.time()
last_print  = t_start

for batch_idx in range(n_batches):
    start = batch_idx * BATCH_SIZE
    end   = min(start + BATCH_SIZE, n_total)
    batch_files = all_files[start:end]

    # Load & preprocess images
    tensors = []
    valid_indices = []   # positions within the batch that loaded OK
    for local_i, fname in enumerate(batch_files):
        fpath = os.path.join(INPUT_DIR, fname)
        try:
            img = Image.open(fpath).convert("RGB")
            tensors.append(preprocess_val(img))
            valid_indices.append(local_i)
        except Exception as err:
            errors.append((fname, str(err)))

    if not tensors:
        continue

    batch_tensor = torch.stack(tensors).to(device)   # (B, 3, 224, 224)

    with torch.no_grad():
        features = model.encode_image(batch_tensor)          # (B, 512)
        features = F.normalize(features, dim=-1).cpu().float()

    features_np = features.numpy()

    for local_j, global_i in enumerate(range(start, end)):
        # Map valid_indices positions to actual global index
        if local_j < len(valid_indices):
            all_embeddings[global_i] = features_np[local_j]
            case_code = parse_case_code(all_files[global_i])
            metadata_rows.append((all_files[global_i], case_code, global_i))

    # Progress print every ~10 seconds or every 10 batches
    now = time.time()
    if now - last_print >= 10 or (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
        pct     = (end / n_total) * 100
        elapsed = now - t_start
        rate    = end / elapsed if elapsed > 0 else 0
        eta_s   = (n_total - end) / rate if rate > 0 else 0
        print(f"      [{batch_idx+1:3d}/{n_batches}]  {end:4d}/{n_total}  "
              f"({pct:5.1f}%)  |  {rate:5.1f} img/s  |  ETA {eta_s:5.0f}s")
        last_print = now

elapsed_total = time.time() - t_start
print(f"\n      Done! {n_total} frames processed in {elapsed_total:.1f}s "
      f"({n_total/elapsed_total:.1f} img/s)")

if errors:
    print(f"\n      WARNING: {len(errors)} images failed to load:")
    for fname, err in errors[:5]:
        print(f"        - {fname}: {err}")
    if len(errors) > 5:
        print(f"        ... and {len(errors) - 5} more")

# ── Step 5: Save outputs ─────────────────────────────────────────────────────
print(f"\n[5/5] Saving outputs …")

# 5a. Embedding matrix
np.save(EMB_PATH, all_embeddings)
size_mb = os.path.getsize(EMB_PATH) / (1024 * 1024)
print(f"      Embeddings  : {EMB_PATH}")
print(f"                    shape = {all_embeddings.shape}  |  {size_mb:.2f} MB")

# 5b. Metadata CSV
with open(META_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "case_code", "embedding_idx"])
    writer.writerows(metadata_rows)
size_kb = os.path.getsize(META_PATH) / 1024
print(f"      Metadata    : {META_PATH}")
print(f"                    rows = {len(metadata_rows)}  |  {size_kb:.1f} KB")

# ── Verification ─────────────────────────────────────────────────────────────
print(f"\n--- Verification ---")
emb_verify  = np.load(EMB_PATH)
print(f"  Loaded shape          : {emb_verify.shape}")
print(f"  dtype                 : {emb_verify.dtype}")
print(f"  First-row L2 norm     : {np.linalg.norm(emb_verify[0]):.6f}  (expected ~1.0)")
print(f"  Last-row  L2 norm     : {np.linalg.norm(emb_verify[-1]):.6f}")
print(f"  Value range           : [{emb_verify.min():.4f}, {emb_verify.max():.4f}]")

# Unique case codes
import collections
case_counts = collections.Counter(r[1] for r in metadata_rows)
print(f"\n  Unique case codes     : {len(case_counts)}")
print(f"  Frames per case (top 5):")
for code, cnt in sorted(case_counts.items(), key=lambda x: -x[1])[:5]:
    print(f"    Case {code:>4s} : {cnt} frames")

print("\n" + "=" * 70)
print("🎉 Task 2.2 COMPLETE – Embeddings saved and verified.")
print("=" * 70)
