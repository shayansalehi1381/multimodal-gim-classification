import os
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def calculate_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    correct = sum(y_true == y_pred)
    # Sensitivity (Recall for class 1 - Incomplete)
    sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    # Specificity (Recall for class 0 - Complete)
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    auc = np.nan
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            pass
            
    return acc, correct, sens, spec, f1, auc

def find_optimal_threshold(y_true, y_prob):
    # Find Youden's J threshold
    thresholds = np.linspace(0.01, 0.99, 99)
    best_j = -1
    best_thresh = 0.5
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_thresh = thresh
    return best_thresh

def main():
    print("Loading fused multimodal dataset...")
    df = pd.read_csv("data/processed/multimodal_biomedclip_dinov2_n17.csv")
    
    # Extract target label
    y = df['Subtype'].map({'Complete': 0, 'Incomplete': 1}).values
    
    clinical_cols = ['LBC', 'MTB', 'WOS', 'TVF', 'MLE', 'TM', 'Age']
    actual_clinical_cols = [c for c in clinical_cols if c in df.columns]
    if len(actual_clinical_cols) < len(clinical_cols):
        print(f"Warning: Missing some clinical cols. Found: {actual_clinical_cols}")
        
    visual_cols = [c for c in df.columns if c.startswith('biomedclip_') or c.startswith('dinov2_')]
    print(f"Found {len(actual_clinical_cols)} clinical features and {len(visual_cols)} visual features.")
    
    X_clin = df[actual_clinical_cols].values
    X_vis = df[visual_cols].values
    groups = df['Case Code'].values
    
    logo = LeaveOneGroupOut()
    
    # Models to test
    models = {
        'RF': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
        'SVC': SVC(kernel='rbf', probability=True, random_state=42),
        'GB': GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=2, random_state=42)
    }
    
    pca_configs = [5, 8]
    
    results = []
    
    for n_pcs in pca_configs:
        for model_name, model in models.items():
            print(f"Evaluating {model_name} with {n_pcs} PCs...")
            
            y_pred_standard = np.zeros_like(y)
            y_pred_tuned = np.zeros_like(y)
            y_prob_all = np.zeros_like(y, dtype=float)
            
            for train_idx, test_idx in logo.split(X_clin, y, groups):
                X_clin_train, X_clin_test = X_clin[train_idx], X_clin[test_idx]
                X_vis_train, X_vis_test = X_vis[train_idx], X_vis[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Scale clinical
                scaler_clin = StandardScaler().fit(X_clin_train)
                X_clin_train_scaled = scaler_clin.transform(X_clin_train)
                X_clin_test_scaled = scaler_clin.transform(X_clin_test)
                
                # Scale visual
                scaler_vis = StandardScaler().fit(X_vis_train)
                X_vis_train_scaled = scaler_vis.transform(X_vis_train)
                X_vis_test_scaled = scaler_vis.transform(X_vis_test)
                
                # PCA on visual
                pca = PCA(n_components=n_pcs, random_state=42).fit(X_vis_train_scaled)
                X_vis_train_pca = pca.transform(X_vis_train_scaled)
                X_vis_test_pca = pca.transform(X_vis_test_scaled)
                
                # Concatenate
                X_train_final = np.hstack((X_clin_train_scaled, X_vis_train_pca))
                X_test_final = np.hstack((X_clin_test_scaled, X_vis_test_pca))
                
                # Internal CV for threshold optimization
                cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
                
                try:
                    y_prob_cv = cross_val_predict(model, X_train_final, y_train, cv=cv, method='predict_proba')[:, 1]
                except Exception:
                    # Fallback to model trained on whole train set (slight leakage for threshold, but acceptable fallback)
                    model.fit(X_train_final, y_train)
                    y_prob_cv = model.predict_proba(X_train_final)[:, 1]
                
                best_thresh = find_optimal_threshold(y_train, y_prob_cv)
                
                # Train final model on all 16 train patients
                model.fit(X_train_final, y_train)
                
                # Predict on 1 test patient
                prob_test = model.predict_proba(X_test_final)[:, 1]
                y_prob_all[test_idx] = prob_test
                y_pred_standard[test_idx] = (prob_test >= 0.5).astype(int)
                y_pred_tuned[test_idx] = (prob_test >= best_thresh).astype(int)
                
            # Compile results for Standard Threshold
            acc_std, corr_std, sens_std, spec_std, f1_std, auc_std = calculate_metrics(y, y_pred_standard, y_prob_all)
            results.append({
                'Model': model_name,
                'PCs': n_pcs,
                'Threshold_Type': 'Standard_0.5',
                'Accuracy': acc_std,
                'Correct': f"{corr_std}/17",
                'Sensitivity': sens_std,
                'Specificity': spec_std,
                'F1_Score': f1_std,
                'ROC_AUC': auc_std
            })
            
            # Compile results for Tuned Threshold
            acc_t, corr_t, sens_t, spec_t, f1_t, auc_t = calculate_metrics(y, y_pred_tuned, y_prob_all)
            results.append({
                'Model': model_name,
                'PCs': n_pcs,
                'Threshold_Type': 'Tuned_Youden',
                'Accuracy': acc_t,
                'Correct': f"{corr_t}/17",
                'Sensitivity': sens_t,
                'Specificity': spec_t,
                'F1_Score': f1_t,
                'ROC_AUC': auc_t
            })

    res_df = pd.DataFrame(results)
    
    # Save to CSV
    out_csv = "data/processed/dinov2_multimodal_evaluation_results.csv"
    res_df.to_csv(out_csv, index=False)
    
    print("\n--- Evaluation Summary ---")
    
    # Print direct comparison table
    print(f"{'Model Configuration':<35} | {'Accuracy':<10} | {'Correct Cases'}")
    print("-" * 65)
    print(f"{'Baseline Clinical LR':<35} | {0.412:<10.3f} | 7/17")
    print(f"{'BiomedCLIP-only RF':<35} | {0.647:<10.3f} | 11/17")
    
    for idx, row in res_df.iterrows():
        config = f"{row['Model']} (PCs={row['PCs']}, {row['Threshold_Type'][:6]})"
        print(f"{config:<35} | {row['Accuracy']:<10.3f} | {row['Correct']}")
        
    print("\nTask completed successfully!")

if __name__ == "__main__":
    main()
