"""
Master Reproduction Orchestrator
================================
Execute this script to cleanly rerun the core classification and evaluation pipeline.
Use `--eval-only` to skip heavy embedding extraction (assumes data/processed exists).
"""

import os
import sys
import argparse
import subprocess

def run_script(script_name):
    print(f"\n{'='*70}")
    print(f"🚀 Running: {script_name}")
    print(f"{'='*70}\n")
    
    python_exe = sys.executable
    result = subprocess.run([python_exe, script_name])
    
    if result.returncode != 0:
        print(f"\n❌ Error encountered in {script_name}. Pipeline halted.")
        sys.exit(result.returncode)
    else:
        print(f"\n✅ {script_name} completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="GIMENDO v2 Pipeline Orchestrator")
    parser.add_argument("--eval-only", action="store_true", 
                        help="Skip embedding extraction and multimodal fusion (phases 1-2). "
                             "Only run machine learning evaluation, SHAP, and visualizations (phases 3-4).")
    args = parser.parse_args()

    # The pipeline scripts in execution order
    phase1_2_scripts = [
        "phase1_process_metadata.py",
        "phase2_task2_extract_embeddings.py",
        "phase2_task3_patient_pooling.py",
        "phase2_task4_multimodal_fusion.py",
        "phase2_dinov2_extract_embeddings.py",
        "phase2_dinov2_patient_pooling_and_fusion.py"
    ]
    
    phase3_4_scripts = [
        "phase3_task1_lopocv_setup.py",
        "phase3_task2_baseline_lr.py",
        "phase3_task4_multimodal_classifiers.py",
        "phase3_task6_statistical_evaluation.py",
        "phase3_dinov2_multimodal_evaluation.py",
        "phase3_dinov2_final_stats_and_plots.py",
        "phase4_task1_shap_analysis.py",
        "phase4_task2_shap_plots.py",
        "phase4_task3_attention_rollout.py"
    ]

    print("\n" + "#"*70)
    print("   GIMENDO v2 – MASTER REPRODUCTION PIPELINE")
    print("#"*70)
    
    if not args.eval_only:
        print("\nStarting Phase 1 & 2: Feature Engineering & BiomedCLIP Extraction...")
        for script in phase1_2_scripts:
            if os.path.exists(script):
                run_script(script)
            else:
                print(f"⚠️ Warning: {script} not found. Skipping.")
    else:
        print("\nSkipping Phase 1 & 2 (--eval-only flag detected).")

    print("\nStarting Phase 3 & 4: Model Evaluation, Statistics, & Explainability...")
    for script in phase3_4_scripts:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"⚠️ Warning: {script} not found. Skipping.")
            
    print("\n" + "#"*70)
    print("🎉 PIPELINE EXECUTION COMPLETE")
    print("#"*70 + "\n")

if __name__ == "__main__":
    main()
