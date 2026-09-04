import os
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from PIL import Image
# pyrefly: ignore [missing-import]
from transformers import AutoImageProcessor, AutoModel
import re
from tqdm import tqdm

def main():
    print("Loading DINOv2 model...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    model.eval()
    
    frames_dir = "data/filtered_frames"
    output_npy = "data/processed/dinov2_frame_embeddings_2016x768.npy"
    output_csv = "data/processed/dinov2_frame_metadata.csv"
    
    if not os.path.exists(frames_dir):
        print(f"Error: Directory {frames_dir} not found.")
        return
    
    image_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if len(image_files) != 2016:
        print(f"Warning: Expected 2016 frames, found {len(image_files)} frames in {frames_dir}.")
    
    print(f"Found {len(image_files)} frames. Extracting features...")
    
    embeddings = []
    metadata = []
    
    for idx, filename in enumerate(tqdm(image_files)):
        img_path = os.path.join(frames_dir, filename)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
            
        # Parse Case Code. E.g., CASE_2601 or 2601_...
        match = re.search(r'(\d{4})', filename)
        if match:
            case_code = f"CASE_{match.group(1)}"
        else:
            case_code = "UNKNOWN"
            
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            
            # The prompt says: "extract the 768-dimensional [CLS] token representation"
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            
            # L2-normalize the resulting 768-dim embedding vector
            cls_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=-1)
            
            embeddings.append(cls_embedding.cpu().numpy()[0])
            metadata.append({
                'filename': filename,
                'case_code': case_code,
                'embedding_idx': idx
            })
            
    embeddings_matrix = np.array(embeddings)
    
    print(f"Extraction complete. Matrix shape: {embeddings_matrix.shape}")
    
    if embeddings_matrix.shape != (2016, 768) and len(image_files) == 2016:
        print(f"ERROR: Matrix shape is {embeddings_matrix.shape}, expected (2016, 768)")
    
    os.makedirs(os.path.dirname(output_npy), exist_ok=True)
    
    np.save(output_npy, embeddings_matrix)
    print(f"Saved embeddings to {output_npy}")
    
    df = pd.DataFrame(metadata)
    df.to_csv(output_csv, index=False)
    print(f"Saved metadata to {output_csv}")
    
    print("Task completed successfully!")

if __name__ == "__main__":
    main()
