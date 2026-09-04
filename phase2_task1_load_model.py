"""
Task 2.1 – Load BiomedCLIP pretrained model via open_clip
==========================================================
Model: hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
Steps:
  1. Check & report PyTorch + open_clip versions
  2. Load BiomedCLIP model + image preprocessing transform
  3. Detect CUDA device and move model to it
  4. Print model summary (param count, image resolution, device)
"""

import sys
import os

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

print("=" * 70)
print("   Task 2.1 – BiomedCLIP Model Loading & GPU Verification")
print("=" * 70)

# ── Step 1: Import and version check ─────────────────────────────────────────
print("\n[1/4] Importing libraries …")

try:
    # pyrefly: ignore [missing-import]
    import torch
    print(f"  ✔ PyTorch       : {torch.__version__}")
except ImportError as e:
    print(f"  ✘ PyTorch not found: {e}")
    sys.exit(1)

try:
    # pyrefly: ignore [missing-import]
    import open_clip
    print(f"  ✔ open_clip     : {open_clip.__version__}")
except ImportError as e:
    print(f"  ✘ open_clip not found: {e}")
    sys.exit(1)

try:
    # pyrefly: ignore [missing-import]
    import transformers
    print(f"  ✔ transformers  : {transformers.__version__}")
except ImportError as e:
    print(f"  ✘ transformers not found: {e}")
    sys.exit(1)

# ── Step 2: GPU detection ────────────────────────────────────────────────────
print("\n[2/4] Detecting compute device …")
cuda_available = torch.cuda.is_available()
if cuda_available:
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"  ✔ CUDA available — GPU : {gpu_name}")
    print(f"                      VRAM: {gpu_mem:.1f} GB")
else:
    device = torch.device("cpu")
    print("  ⚠  CUDA NOT available — running on CPU")
print(f"  → Target device : {device}")

# ── Step 3: Load BiomedCLIP model ────────────────────────────────────────────
MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

print(f"\n[3/4] Loading BiomedCLIP model …")
print(f"  Model ID : {MODEL_NAME}")
print("  (Downloading weights from HuggingFace Hub if not cached – please wait …)")

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL_NAME)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

# Move to target device
model = model.to(device)
model.eval()
print(f"  ✔ Model loaded and moved to {device}")

# ── Step 4: Model summary ────────────────────────────────────────────────────
print("\n[4/4] Model Summary:")

total_params   = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Extract image preprocessing info from the validation transform
image_size = None
norm_mean  = None
norm_std   = None
try:
    # pyrefly: ignore [missing-import]
    from torchvision import transforms as T
    for t in preprocess_val.transforms:
        if isinstance(t, T.Resize):
            image_size = t.size if isinstance(t.size, int) else t.size[0]
        elif isinstance(t, T.CenterCrop):
            image_size = t.size if isinstance(t.size, int) else t.size[0]
        elif isinstance(t, T.Normalize):
            norm_mean = [round(m, 4) for m in t.mean]
            norm_std  = [round(s, 4) for s in t.std]
except Exception:
    pass

print(f"  ┌─────────────────────────────────────────────────────────────┐")
print(f"  │ Model          : BiomedCLIP-PubMedBERT_256-ViT-B/16        │")
print(f"  │ Source         : microsoft (HuggingFace Hub)                │")
print(f"  │ Device         : {str(device):<43} │")
print(f"  │ Total Params   : {total_params:,}  ({total_params/1e6:.2f} M){' '*(24-len(f'{total_params:,}'))}│")
print(f"  │ Trainable      : {trainable_params:,}  ({trainable_params/1e6:.2f} M){' '*(24-len(f'{trainable_params:,}'))}│")
if image_size:
    print(f"  │ Input Size     : {image_size} × {image_size} px{' '*(38-len(f'{image_size} x {image_size} px'))}│")
if norm_mean:
    print(f"  │ Norm Mean      : {norm_mean}                 │")
if norm_std:
    print(f"  │ Norm Std       : {norm_std}                 │")
print(f"  └─────────────────────────────────────────────────────────────┘")

print("\n  Preprocessing transforms (validation):")
try:
    for i, t in enumerate(preprocess_val.transforms):
        print(f"    [{i}] {t}")
except Exception:
    print("    (unable to enumerate transforms)")

# Quick sanity-check: forward pass with a dummy image
print("\n[Check] Running quick forward-pass sanity check …")
dummy_img = torch.zeros(1, 3, 224, 224, device=device)
dummy_txt = tokenizer(["intestinal metaplasia complete", "intestinal metaplasia incomplete"]).to(device)
with torch.no_grad():
    img_feat = model.encode_image(dummy_img)
    txt_feat = model.encode_text(dummy_txt)
print(f"  ✔ Image feature  shape : {tuple(img_feat.shape)}")
print(f"  ✔ Text  feature  shape : {tuple(txt_feat.shape)}")

print("\n" + "=" * 70)
print("🎉 Task 2.1 COMPLETE – BiomedCLIP is loaded and ready for inference.")
print("=" * 70)
