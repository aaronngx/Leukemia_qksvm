#!/usr/bin/env python3
"""
Comprehensive QKSVM Test Suite

Tests all combinations of:
- K values: 4, 8, 16
- Feature selection: ANOVA, SNR
- Angle encoding types: Simple RY, ZZ Feature Map, BPS Circuit
- Patient balance: Balanced (22) vs All (38)
- Gene balance: Balanced (k/2 + k/2) vs Unbalanced (top-k)

Outputs results to comprehensive_test_results.csv and generates analysis plots.
"""

import sys
import time
from pathlib import Path
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "feature-selection-methods"))

from anova_f import run_feature_selection as run_anova
from signal_to_noise import run_snr_selection
from angle_encoding import angle_encoding_circuit, AngleEncodingType
from qksvm_golub import (
    train_eval_qksvm,
    EncodingType,
    scale_to_angle,
    build_kernel,
)
from data_loader import load_preprocessed_data
from backend_config import BackendType, KernelMethod

# ============================================================
# Configuration
# ============================================================
K_VALUES = [4, 8, 16]
FEATURE_SELECTION_METHODS = ["anova", "snr"]
ANGLE_ENCODING_TYPES = [
    ("simple_ry", AngleEncodingType.SIMPLE_RY, 1),
    ("zz_feature_map", AngleEncodingType.ZZ_FEATURE_MAP, 2),
    ("bps_circuit", AngleEncodingType.BPS_CIRCUIT, 1),
]
PATIENT_BALANCE_OPTIONS = [True, False]  # True = balanced 22, False = all 38
GENE_BALANCE_OPTIONS = [True, False]     # True = k/2 ALL + k/2 AML, False = top-k

# Data paths
DATA_DIR = Path("data/raw")
TRAIN_CSV = DATA_DIR / "data_set_ALL_AML_train.csv"
IND_CSV = DATA_DIR / "data_set_ALL_AML_independent.csv"
LABELS_CSV = DATA_DIR / "actual.csv"
OUTPUT_DIR = Path("results_comprehensive")

# Test parameters
TEST_SIZE = 0.3
SEED = 42


def print_banner():
    print()
    print("=" * 80)
    print("  COMPREHENSIVE QKSVM TEST SUITE")
    print("  Testing all combinations of preprocessing and encoding options")
    print("=" * 80)
    print()


def run_feature_selection_for_test(
    method: str,
    k: int,
    balanced_genes: bool,
    balanced_patients: bool,
    output_subdir: str,
) -> tuple:
    """Run feature selection and return paths to generated files."""
    out_dir = OUTPUT_DIR / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Build validation strategy (simple 70/30 split)
    validation_strategy = {"method": "split", "ratio": 0.7}
    
    if method == "anova":
        run_anova(
            input_train=str(TRAIN_CSV),
            input_ind=str(IND_CSV) if IND_CSV.exists() else None,
            input_actual=None,
            k=k,
            out_dir=str(out_dir),
            balanced_genes=balanced_genes,
            use_balanced_patients=balanced_patients,
            validation_strategy=validation_strategy,
            labels_csv=str(LABELS_CSV),
            use_all_data=False,
        )
        train_file = out_dir / f"train_internal_top_{k}_anova_f.csv"
        test_file = out_dir / f"test_internal_top_{k}_anova_f.csv"
        ind_file = out_dir / f"independent_top_{k}_anova_f.csv"
    else:  # snr
        run_snr_selection(
            k=k,
            out_dir=str(out_dir),
            train_csv=str(TRAIN_CSV),
            ind_csv=str(IND_CSV) if IND_CSV.exists() else None,
            labels_csv=str(LABELS_CSV),
            use_all_data=False,
            balanced_genes=balanced_genes,
            use_balanced_patients=balanced_patients,
            validation_strategy=validation_strategy,
        )
        train_file = out_dir / f"train_internal_top_{k}_snr.csv"
        test_file = out_dir / f"test_internal_top_{k}_snr.csv"
        ind_file = out_dir / f"independent_top_{k}_snr.csv"
    
    return train_file, test_file, ind_file


def run_qksvm_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_ind: np.ndarray,
    y_ind: np.ndarray,
    angle_encoding_type: AngleEncodingType,
    angle_reps: int,
    n_features: int,
) -> dict:
    """Run QKSVM with specified angle encoding and return metrics."""
    from sklearn.svm import SVC
    
    # Scale features to [0, π]
    X_train_scaled, scaler = scale_to_angle(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_ind_scaled = scaler.transform(X_ind) if X_ind is not None else None
    
    # Build feature map
    feature_map, x_params = angle_encoding_circuit(
        n_features,
        encoding_type=angle_encoding_type,
        reps=angle_reps,
    )
    
    n_qubits = feature_map.num_qubits
    
    # Build kernels
    print(f"      Building training kernel ({len(X_train_scaled)}x{len(X_train_scaled)})...")
    K_train = build_kernel(
        X_train_scaled, X_train_scaled,
        feature_map, x_params,
        kernel_method=KernelMethod.STATEVECTOR,
        backend_type=BackendType.STATEVECTOR,
        verbose=False,
    )
    
    print(f"      Building test kernel ({len(X_test_scaled)}x{len(X_train_scaled)})...")
    K_test = build_kernel(
        X_test_scaled, X_train_scaled,
        feature_map, x_params,
        kernel_method=KernelMethod.STATEVECTOR,
        backend_type=BackendType.STATEVECTOR,
        verbose=False,
    )
    
    # Train SVM
    clf = SVC(kernel="precomputed", C=1.0)
    clf.fit(K_train, y_train)
    
    # Test predictions
    y_test_pred = clf.predict(K_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # Test AUROC
    try:
        y_test_scores = clf.decision_function(K_test)
        test_auroc = roc_auc_score(y_test, y_test_scores)
    except:
        test_auroc = None
    
    test_f1 = f1_score(y_test, y_test_pred, average='binary', zero_division=0)
    
    results = {
        "test_acc": test_acc,
        "test_auroc": test_auroc,
        "test_f1": test_f1,
        "n_qubits": n_qubits,
        "circuit_depth": feature_map.depth(),
    }
    
    # Independent set evaluation
    if X_ind_scaled is not None and len(X_ind_scaled) > 0:
        print(f"      Building independent kernel ({len(X_ind_scaled)}x{len(X_train_scaled)})...")
        K_ind = build_kernel(
            X_ind_scaled, X_train_scaled,
            feature_map, x_params,
            kernel_method=KernelMethod.STATEVECTOR,
            backend_type=BackendType.STATEVECTOR,
            verbose=False,
        )
        
        y_ind_pred = clf.predict(K_ind)
        ind_acc = accuracy_score(y_ind, y_ind_pred)
        
        try:
            y_ind_scores = clf.decision_function(K_ind)
            ind_auroc = roc_auc_score(y_ind, y_ind_scores)
        except:
            ind_auroc = None
        
        ind_f1 = f1_score(y_ind, y_ind_pred, average='binary', zero_division=0)
        
        results["ind_acc"] = ind_acc
        results["ind_auroc"] = ind_auroc
        results["ind_f1"] = ind_f1
    else:
        results["ind_acc"] = None
        results["ind_auroc"] = None
        results["ind_f1"] = None
    
    return results


def run_comprehensive_test():
    """Run all test combinations."""
    print_banner()
    
    # Check data files
    if not TRAIN_CSV.exists():
        print(f"[ERROR] Training data not found: {TRAIN_CSV}")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Calculate total experiments
    total_experiments = (
        len(K_VALUES) *
        len(FEATURE_SELECTION_METHODS) *
        len(ANGLE_ENCODING_TYPES) *
        len(PATIENT_BALANCE_OPTIONS) *
        len(GENE_BALANCE_OPTIONS)
    )
    
    print(f"Configuration:")
    print(f"  K values: {K_VALUES}")
    print(f"  Feature selection: {FEATURE_SELECTION_METHODS}")
    print(f"  Angle encodings: {[e[0] for e in ANGLE_ENCODING_TYPES]}")
    print(f"  Patient balance: {PATIENT_BALANCE_OPTIONS}")
    print(f"  Gene balance: {GENE_BALANCE_OPTIONS}")
    print(f"\n  Total experiments: {total_experiments}")
    print()
    
    # Confirm
    confirm = input("Proceed? (y/n) [default=y]: ").strip().lower()
    if confirm not in ["", "y", "yes"]:
        print("Cancelled.")
        return
    
    all_results = []
    experiment_num = 0
    start_time = time.time()
    
    # Iterate through all combinations
    for k in K_VALUES:
        for fs_method in FEATURE_SELECTION_METHODS:
            for balanced_patients in PATIENT_BALANCE_OPTIONS:
                for balanced_genes in GENE_BALANCE_OPTIONS:
                    # Create unique output directory for this combination
                    patient_str = "balanced_patients" if balanced_patients else "all_patients"
                    gene_str = "balanced_genes" if balanced_genes else "topk_genes"
                    subdir = f"{fs_method}_k{k}_{patient_str}_{gene_str}"
                    
                    print("\n" + "=" * 80)
                    print(f"[{experiment_num + 1}/{total_experiments}] Feature Selection")
                    print(f"  Method: {fs_method.upper()}, K={k}")
                    print(f"  Patients: {'Balanced 22' if balanced_patients else 'All 38'}")
                    print(f"  Genes: {'Balanced' if balanced_genes else 'Top-K'}")
                    print("=" * 80)
                    
                    try:
                        # Run feature selection
                        train_file, test_file, ind_file = run_feature_selection_for_test(
                            method=fs_method,
                            k=k,
                            balanced_genes=balanced_genes,
                            balanced_patients=balanced_patients,
                            output_subdir=subdir,
                        )
                        
                        # Load data
                        if not train_file.exists():
                            print(f"  [WARNING] Train file not found: {train_file}")
                            continue
                        
                        X_train, y_train = load_preprocessed_data(str(train_file))
                        
                        if test_file.exists():
                            X_test, y_test = load_preprocessed_data(str(test_file))
                        else:
                            # Fall back to train/test split
                            X_train, X_test, y_train, y_test = train_test_split(
                                X_train, y_train, test_size=TEST_SIZE, random_state=SEED, stratify=y_train
                            )
                        
                        if ind_file.exists():
                            X_ind, y_ind = load_preprocessed_data(str(ind_file))
                        else:
                            X_ind, y_ind = None, None
                        
                        print(f"  Loaded: {len(X_train)} train, {len(X_test)} test", end="")
                        if X_ind is not None:
                            print(f", {len(X_ind)} independent")
                        else:
                            print()
                        
                        # Test each angle encoding type
                        for enc_name, enc_type, enc_reps in ANGLE_ENCODING_TYPES:
                            experiment_num += 1
                            
                            print(f"\n    [{experiment_num}/{total_experiments}] QKSVM + {enc_name}")
                            print(f"    " + "-" * 50)
                            
                            try:
                                results = run_qksvm_test(
                                    X_train, y_train,
                                    X_test, y_test,
                                    X_ind, y_ind,
                                    angle_encoding_type=enc_type,
                                    angle_reps=enc_reps,
                                    n_features=k,
                                )
                                
                                # Store results
                                result_row = {
                                    "k": k,
                                    "feature_selection": fs_method,
                                    "patient_balance": "balanced" if balanced_patients else "all",
                                    "gene_balance": "balanced" if balanced_genes else "topk",
                                    "angle_encoding": enc_name,
                                    "angle_reps": enc_reps,
                                    "n_qubits": results["n_qubits"],
                                    "circuit_depth": results["circuit_depth"],
                                    "test_acc": results["test_acc"],
                                    "test_auroc": results["test_auroc"],
                                    "test_f1": results["test_f1"],
                                    "ind_acc": results["ind_acc"],
                                    "ind_auroc": results["ind_auroc"],
                                    "ind_f1": results["ind_f1"],
                                }
                                all_results.append(result_row)
                                
                                print(f"      Test Accuracy:  {results['test_acc']:.4f}")
                                if results['test_auroc']:
                                    print(f"      Test AUROC:     {results['test_auroc']:.4f}")
                                if results['ind_acc']:
                                    print(f"      Ind. Accuracy:  {results['ind_acc']:.4f}")
                                if results['ind_auroc']:
                                    print(f"      Ind. AUROC:     {results['ind_auroc']:.4f}")
                                
                            except Exception as e:
                                print(f"      [ERROR] {e}")
                                import traceback
                                traceback.print_exc()
                    
                    except Exception as e:
                        print(f"  [ERROR] Feature selection failed: {e}")
                        import traceback
                        traceback.print_exc()
    
    # Save results
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST COMPLETE")
    print("=" * 80)
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Experiments completed: {len(all_results)}")
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = OUTPUT_DIR / f"comprehensive_results_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Also save as latest
        latest_path = OUTPUT_DIR / "comprehensive_results_latest.csv"
        results_df.to_csv(latest_path, index=False)
        
        # Print summary table
        print("\n" + "=" * 140)
        print("RESULTS SUMMARY")
        print("=" * 140)
        print(f"{'K':<4} {'FS':<6} {'Patients':<10} {'Genes':<8} {'Encoding':<15} "
              f"{'Qubits':<7} {'Depth':<6} {'Test Acc':<10} {'Test AUC':<10} {'Ind Acc':<10} {'Ind AUC':<10}")
        print("-" * 140)
        
        for _, r in results_df.iterrows():
            def fmt(val):
                if val is None or pd.isna(val):
                    return "N/A"
                return f"{val:.4f}"
            
            print(f"{r['k']:<4} {r['feature_selection'].upper():<6} {r['patient_balance']:<10} "
                  f"{r['gene_balance']:<8} {r['angle_encoding']:<15} "
                  f"{r['n_qubits']:<7} {r['circuit_depth']:<6} "
                  f"{fmt(r['test_acc']):<10} {fmt(r['test_auroc']):<10} "
                  f"{fmt(r['ind_acc']):<10} {fmt(r['ind_auroc']):<10}")
        
        print("=" * 140)
        
        # Best results summary
        print("\n" + "=" * 80)
        print("BEST CONFIGURATIONS")
        print("=" * 80)
        
        # Best by test accuracy
        if 'test_acc' in results_df.columns:
            best_test = results_df.loc[results_df['test_acc'].idxmax()]
            print(f"\nBest Test Accuracy: {best_test['test_acc']:.4f}")
            print(f"  Config: K={best_test['k']}, {best_test['feature_selection'].upper()}, "
                  f"{best_test['patient_balance']} patients, {best_test['gene_balance']} genes, "
                  f"{best_test['angle_encoding']}")
        
        # Best by independent accuracy
        if 'ind_acc' in results_df.columns and results_df['ind_acc'].notna().any():
            best_ind = results_df.loc[results_df['ind_acc'].idxmax()]
            print(f"\nBest Independent Accuracy: {best_ind['ind_acc']:.4f}")
            print(f"  Config: K={best_ind['k']}, {best_ind['feature_selection'].upper()}, "
                  f"{best_ind['patient_balance']} patients, {best_ind['gene_balance']} genes, "
                  f"{best_ind['angle_encoding']}")
        
        # Best by AUROC
        if 'ind_auroc' in results_df.columns and results_df['ind_auroc'].notna().any():
            best_auroc = results_df.loc[results_df['ind_auroc'].idxmax()]
            print(f"\nBest Independent AUROC: {best_auroc['ind_auroc']:.4f}")
            print(f"  Config: K={best_auroc['k']}, {best_auroc['feature_selection'].upper()}, "
                  f"{best_auroc['patient_balance']} patients, {best_auroc['gene_balance']} genes, "
                  f"{best_auroc['angle_encoding']}")
    
    return all_results


if __name__ == "__main__":
    try:
        run_comprehensive_test()
    except KeyboardInterrupt:
        print("\n\n[INFO] Cancelled by user")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()



