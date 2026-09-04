import os
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import pickle

def main():
    print("--- Step 1: Patient-Level Pooling for DINOv2 ---")
    frame_embeddings_path = "data/processed/dinov2_frame_embeddings_2016x768.npy"
    frame_metadata_path = "data/processed/dinov2_frame_metadata.csv"
    metadata_xlsx_path = "data/processed/cleaned_metadata_n17.xlsx"
    
    frame_embeddings = np.load(frame_embeddings_path)
    frame_metadata = pd.read_csv(frame_metadata_path)
    metadata_df = pd.read_excel(metadata_xlsx_path)
    
    patient_cases = metadata_df['Case Code'].astype(str).tolist()
    
    # Strip "CASE_" if present in frame_metadata to match with patient_cases
    frame_metadata['case_code_clean'] = frame_metadata['case_code'].astype(str).str.replace('CASE_', '')
    
    pooled_features = []
    pooled_metadata = []
    
    for case in patient_cases:
        # Get frame indices for this patient
        case_indices = frame_metadata[frame_metadata['case_code_clean'] == case]['embedding_idx'].values
        
        if len(case_indices) == 0:
            print(f"Warning: No frames found for Case Code {case}")
            continue
            
        case_embeddings = frame_embeddings[case_indices] # Shape: (num_frames, 768)
        
        # Mean Pooling
        pooled_emb = np.mean(case_embeddings, axis=0)
        
        # L2 Normalization
        pooled_emb = pooled_emb / np.linalg.norm(pooled_emb)
        
        pooled_features.append(pooled_emb)
        pooled_metadata.append({'Case Code': int(case)})
        
    pooled_features_matrix = np.array(pooled_features)
    
    print(f"Pooled matrix shape: {pooled_features_matrix.shape}")
    assert pooled_features_matrix.shape == (17, 768), f"Expected (17, 768), got {pooled_features_matrix.shape}"
    
    np.save("data/processed/dinov2_patient_features_17x768.npy", pooled_features_matrix)
    
    dinov2_df = pd.DataFrame(pooled_features_matrix, columns=[f'dinov2_{i}' for i in range(768)])
    dinov2_df.insert(0, 'Case Code', [m['Case Code'] for m in pooled_metadata])
    dinov2_df.to_csv("data/processed/dinov2_patient_features_17x768.csv", index=False)
    
    print("--- Step 2: Multi-Vision Multimodal Fusion ---")
    biomedclip_features_path = "data/processed/patient_visual_features_n17.csv"
    biomedclip_df = pd.read_csv(biomedclip_features_path)
    
    # Prefix biomedclip columns to avoid confusion
    rename_dict = {col: f'biomedclip_{col.split("_")[1]}' for col in biomedclip_df.columns if col.startswith('feat_')}
    biomedclip_df.rename(columns=rename_dict, inplace=True)
    
    # Merge visual features on Case Code
    visual_fused_df = pd.merge(biomedclip_df, dinov2_df, on='Case Code', how='inner')
    
    # Now merge with clinical/IEE metadata
    multimodal_df = pd.merge(visual_fused_df, metadata_df, on='Case Code', how='inner')
    
    # Verify shape
    num_visual_features = 512 + 768 # 1280
    num_metadata_cols = len(metadata_df.columns)
    # The merged DF will have num_visual_features + num_metadata_cols columns
    expected_cols = 1280 + num_metadata_cols
    print(f"Fused Multimodal Matrix Shape: {multimodal_df.shape}")
    
    if multimodal_df.shape[1] != expected_cols:
        print(f"Warning: Expected {expected_cols} columns, got {multimodal_df.shape[1]}")
    
    # Target label check
    if 'Subtype' in multimodal_df.columns:
        print("Target label 'Subtype' successfully preserved.")
        print(multimodal_df['Subtype'].value_counts())
    else:
        print("Warning: Target label 'Subtype' not found in final dataframe!")
    
    multimodal_df.to_csv("data/processed/multimodal_biomedclip_dinov2_n17.csv", index=False)
    multimodal_df.to_pickle("data/processed/multimodal_biomedclip_dinov2_n17.pkl")
    
    print("Task completed successfully!")

if __name__ == "__main__":
    main()
