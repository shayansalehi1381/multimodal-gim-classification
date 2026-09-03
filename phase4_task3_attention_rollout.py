"""
Task 3.4 – Visual Attention Rollout on BiomedCLIP ViT
=====================================================
"""

import os
import sys
import types
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import open_clip

print("=" * 70)
print("   Task 3.4 – Visual Attention Rollout")
print("=" * 70)

# ── Config ───────────────────────────────────────────────────────────────────
IMG_DIR = os.path.join("data", "filtered_frames")
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Select 2 Incomplete (2601, 2605) and 2 Complete (2602, 2603)
TARGET_CASES = {
    '2601': 'Incomplete',
    '2605': 'Incomplete',
    '2602': 'Complete',
    '2603': 'Complete'
}

selected_frames = {}
for fname in sorted(os.listdir(IMG_DIR)):
    if not fname.lower().endswith(('.jpg', '.png')):
        continue
    # Filenames are like CASE_2601_...
    if fname.startswith('CASE_'):
        case_id = fname.split('_')[1]
        if case_id in TARGET_CASES and case_id not in selected_frames:
            selected_frames[case_id] = os.path.join(IMG_DIR, fname)
    if len(selected_frames) == len(TARGET_CASES):
        break

# ── 1. Load Model ────────────────────────────────────────────────────────────
print("\n[1/4] Loading BiomedCLIP...")
model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
model = model.to(device)
model.eval()

# ── 2. Hook Attention Weights ────────────────────────────────────────────────
print("\n[2/4] Hooking into ViT attention layers...")

blocks = model.visual.trunk.blocks

attn_weights = []

def get_patched_forward(module):
    def patched_forward(self, x, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        gate = self.gate(x).sigmoid() if self.gate is not None else None
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Force fused_attn off
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        attn = attn.softmax(dim=-1)
        
        # Save weights (average across heads)
        with torch.no_grad():
            attn_weights.append(attn.mean(dim=1).detach())
            
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        x = self.norm(x)
        if gate is not None:
            x = x * gate
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    return types.MethodType(patched_forward, module)

for blk in blocks:
    blk.attn.forward = get_patched_forward(blk.attn)

def rollout(attentions):
    """
    attentions: list of tensors of shape [B, N, N]
    """
    B, N, _ = attentions[0].shape
    result = torch.eye(N).unsqueeze(0).repeat(B, 1, 1).to(attentions[0].device)
    
    with torch.no_grad():
        for attention in attentions:
            attention_heads_fused = attention + torch.eye(N).unsqueeze(0).to(attention.device)
            attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
            result = torch.bmm(attention_heads_fused, result)
            
    return result

# ── 3. Process Frames and Plot ───────────────────────────────────────────────
print("\n[3/4] Running Attention Rollout on selected frames...")

for case_id, img_path in selected_frames.items():
    print(f"      Processing Case {case_id} ({TARGET_CASES[case_id]})...")
    
    attn_weights.clear()
    
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size
    
    x = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _ = model.visual(x)
        
    rollout_mat = rollout(attn_weights)
    
    cls_attention = rollout_mat[0, 0, 1:]
    
    grid_size = int(np.sqrt(cls_attention.shape[0]))
    attention_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
    
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    heatmap = cv2.resize(attention_map, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    img_np = np.array(img)
    overlay = cv2.addWeighted(img_np, 0.5, heatmap_colored, 0.5, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title(f"Original Frame\n(Case {case_id} - {TARGET_CASES[case_id]})", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Attention Heatmap", fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title("Attention Overlay", fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    out_file = os.path.join(FIG_DIR, f"attention_rollout_{case_id}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()

print("\n[4/4] Output files:")
for case_id in selected_frames.keys():
    print(f"      ✔ figures/attention_rollout_{case_id}.png")

print("\n" + "=" * 70)
print("🎉 Task 3.4 COMPLETE – Attention Rollout Heatmaps generated.")
print("=" * 70)
