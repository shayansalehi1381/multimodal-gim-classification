"""Goal 1, Step B - frozen image embeddings for every image in results/goal1/capped_list.csv.

Preprocessing (both models): resize the FULL image to 224x224 (bicubic, no center-crop, no custom crop), then the
model's own normalisation. BiomedCLIP ViT-B/16: all images in the list. DINOv2 ViT-B/14 CLS token: A2-NBI images only.
Each embedding is L2-normalised. Batch 8, CPU fp32 (user decision: .venv torch has no CUDA), eval/no_grad, offline cache only.
Outputs: D:/gim_outputs/goal1/embeddings/{biomedclip,dinov2}.npy + {biomedclip,dinov2}_index.csv
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import normalize, pil_to_tensor

EMB = "D:/gim_outputs/goal1/embeddings/"
os.makedirs(EMB, exist_ok=True)
torch.manual_seed(42)
dev = "cpu"
BS, SIZE = 8, 224
cl = pd.read_csv("../results/goal1/capped_list.csv")


def load(path, mean, std):
    im = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
    return normalize(pil_to_tensor(im).float() / 255.0, mean, std)


@torch.inference_mode()
def run(paths, mean, std, encode, dim):
    out = np.zeros((len(paths), dim), np.float32)
    for i in range(0, len(paths), BS):
        x = torch.stack([load(p, mean, std) for p in paths[i:i + BS]]).to(dev)
        z = encode(x)
        out[i:i + BS] = torch.nn.functional.normalize(z.float(), dim=-1).cpu().numpy()
        if (i // BS) % 100 == 0:
            print(f"  {i}/{len(paths)}", flush=True)
    return out


def save(name, paths, E):
    assert np.isfinite(E).all(), f"non-finite embeddings in {name}"
    assert (np.abs(E).sum(1) > 0).all(), f"zero embeddings in {name}"
    np.save(f"{EMB}{name}.npy", E)
    pd.DataFrame({"path": paths, "row": np.arange(len(paths))}).to_csv(f"{EMB}{name}_index.csv", index=False)
    print(name, E.shape, "finite OK", flush=True)


import open_clip
from transformers import AutoImageProcessor, AutoModel

m, _, pre = open_clip.create_model_and_transforms("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
m = m.to(dev).eval()
tn = pre.transforms[-1]
paths = sorted(cl.path.unique())
save("biomedclip", paths, run(paths, list(tn.mean), list(tn.std), m.encode_image, 512))
del m

proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base", local_files_only=True)
d = AutoModel.from_pretrained("facebook/dinov2-base", local_files_only=True).to(dev).eval()
paths = sorted(cl[cl.dataset == "A2-NBI"].path.unique())
save("dinov2", paths, run(paths, list(proc.image_mean), list(proc.image_std),
                          lambda x: d(pixel_values=x).last_hidden_state[:, 0, :], 768))
print("done", flush=True)
