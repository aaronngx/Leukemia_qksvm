#!/usr/bin/env python3
"""
Gene Expression Feature Selection - Golub Methodology

Follows the original Golub et al. (1999) approach:
- Gene selection uses ONLY training data (38 samples)
- Independent test set (34 samples) used ONLY for validation
- No data leakage between train and test
"""

import sys
from pathlib import Path

# Add feature-selection-methods to path
sys.path.insert(0, str(Path(__file__).parent / "feature-selection-methods"))

from anova_f import run_feature_selection as run_anova
from signal_to_noise import run_snr_selection
from scad_svm import run_scad_selection


def print_banner():
    """Print welcome banner."""
    print()
    print("=" * 70)
    print("  GENE EXPRESSION FEATURE SELECTION")
    print("  Golub Leukemia Dataset (ALL vs AML)")
    print("  Following Golub et al. (1999) Methodology")
    print("=" * 70)
    print()


def print_separator():
    """Print separator line."""
    print("=" * 70)


def ask_gene_count():
    """Ask how many genes to select and whether to balance them."""
    print("[1/4] How many genes/qubits to select?")
    print("      (Common: 16, 24, 32, 50)")
    print("      (Golub used 50 genes: 25 ALL + 25 AML)")

    while True:
        try:
            k = int(input("      -> "))
            if k < 1:
                print("      [ERROR] Please enter a positive number")
                continue
            if k > 7129:  # Total genes in dataset
                print("      [ERROR] Maximum 7129 genes (total in dataset)")
                continue

            # Sub-question: balance genes?
            print()
            print("      Balance gene selection?")
            print("      y = Yes (k/2 ALL-favoring + k/2 AML-favoring)")
            print("      n = No (pure top-k by score)")

            while True:
                balance_choice = input("      -> (y/n): ").lower().strip()
                if balance_choice in ["y", "yes"]:
                    return k, True
                elif balance_choice in ["n", "no"]:
                    return k, False
                else:
                    print("      [ERROR] Please enter 'y' or 'n'")

        except ValueError:
            print("      [ERROR] Please enter a valid integer")
        except KeyboardInterrupt:
            print("\n\n[INFO] Cancelled by user")
            sys.exit(0)


def ask_method():
    """Ask which feature selection method to use."""
    print("\n[2/4] Feature selection method:")
    print("      1. ANOVA F-test")
    print("      2. SNR (Signal-to-Noise Ratio / Golub's P-score)")
    print("      3. SCAD-SVM (Smoothly Clipped Absolute Deviation)")
    print("      4. All methods (ANOVA + SNR + SCAD)")

    while True:
        try:
            choice = input("      -> ")
            if choice in ["1", "2", "3", "4"]:
                return int(choice)
            else:
                print("      [ERROR] Please enter 1, 2, 3, or 4")
        except KeyboardInterrupt:
            print("\n\n[INFO] Cancelled by user")
            sys.exit(0)


def ask_patient_count():
    """Ask how many patients to use for training."""
    print("\n[3/4] Patient samples:")
    print("      1. All 38 patients (27 ALL + 11 AML) - imbalanced")
    print("      2. Balanced 22 patients (11 ALL + 11 AML)")

    while True:
        try:
            choice = input("      -> ")
            if choice == "1":
                return False  # use_all_patients
            elif choice == "2":
                return True   # use_balanced_patients
            else:
                print("      [ERROR] Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\n\n[INFO] Cancelled by user")
            sys.exit(0)


def ask_validation_strategy():
    """Ask which internal validation strategy to use."""
    print("\n[4/4] Internal validation strategy:")
    print("      1. Train/test split")
    print("         a. 70/30 split")
    print("         b. 80/20 split")
    print("      2. Cross-validation")
    print("         a. 5-fold CV")
    print("         b. 10-fold CV")
    print("         c. LOOCV (Leave-One-Out CV)")

    while True:
        try:
            choice = input("      -> (1a/1b/2a/2b/2c): ").lower().strip()
            if choice == "1a":
                return {"method": "split", "ratio": 0.7}
            elif choice == "1b":
                return {"method": "split", "ratio": 0.8}
            elif choice == "2a":
                return {"method": "cv", "folds": 5}
            elif choice == "2b":
                return {"method": "cv", "folds": 10}
            elif choice == "2c":
                return {"method": "loocv"}
            else:
                print("      [ERROR] Please enter 1a, 1b, 2a, 2b, or 2c")
        except KeyboardInterrupt:
            print("\n\n[INFO] Cancelled by user")
            sys.exit(0)


def print_config(k: int, balanced_genes: bool, method: int, use_balanced_patients: bool, validation_strategy: dict):
    """Print configuration summary."""
    print()
    print_separator()
    print("  Configuration Summary:")
    print(f"  • Genes/Qubits: {k}")

    if balanced_genes:
        print(f"  • Gene Balance: Yes ({k//2} ALL + {k//2} AML)")
    else:
        print(f"  • Gene Balance: No (pure top-k by score)")

    method_names = {1: "ANOVA F-test", 2: "SNR (Golub P-score)", 3: "SCAD-SVM", 4: "All (ANOVA + SNR + SCAD)"}
    print(f"  • Method: {method_names[method]}")

    if use_balanced_patients:
        print(f"  • Patients: Balanced 22 (11 ALL + 11 AML)")
    else:
        print(f"  • Patients: All 38 (27 ALL + 11 AML) - imbalanced")

    if validation_strategy["method"] == "split":
        ratio = validation_strategy["ratio"]
        print(f"  • Validation: Train/test split ({int(ratio*100)}/{int((1-ratio)*100)})")
    elif validation_strategy["method"] == "loocv":
        print(f"  • Validation: LOOCV (Leave-One-Out Cross-Validation)")
    else:
        folds = validation_strategy["folds"]
        print(f"  • Validation: {folds}-fold cross-validation")

    print()
    print("  Data Strategy (Golub Methodology):")
    print("  • Gene selection: Selected patients ONLY (38 or 22)")
    print("  • Internal validation: From selected patients")
    print("  • Independent test: Completely separate (34 samples)")
    print_separator()
    print()


def run_preprocessing():
    """Main preprocessing workflow."""
    print_banner()

    # Get user inputs
    k, balanced_genes = ask_gene_count()  # Returns tuple now
    method = ask_method()
    use_balanced_patients = ask_patient_count()
    validation_strategy = ask_validation_strategy()

    # Show configuration
    print_config(k, balanced_genes, method, use_balanced_patients, validation_strategy)

    # Set paths
    data_dir = Path("data/raw")
    train_csv = data_dir / "data_set_ALL_AML_train.csv"
    ind_csv = data_dir / "data_set_ALL_AML_independent.csv"
    labels_csv = data_dir / "actual.csv"
    out_dir = "results"

    # Check if files exist
    if not train_csv.exists():
        print(f"[ERROR] Training data not found: {train_csv}")
        print("[INFO] Please ensure data/raw/ contains the Golub dataset files")
        return

    # Run selected methods
    try:
        if method in [1, 4]:  # ANOVA or All
            print("Running ANOVA F-test...")
            print("  - Calculating F-scores from training set only")
            print("  - Selecting top genes based on training data")
            print()
            run_anova(
                input_train=str(train_csv),
                input_ind=str(ind_csv) if ind_csv.exists() else None,
                input_actual=None,
                k=k,
                out_dir=out_dir,
                balanced_genes=balanced_genes,  # NEW param
                use_balanced_patients=use_balanced_patients,  # NEW param
                validation_strategy=validation_strategy,  # NEW param
                labels_csv=str(labels_csv),
                use_all_data=False,  # ALWAYS False - training only!
            )
            print("✓ ANOVA complete\n")

        if method in [2, 4]:  # SNR or All
            print("Running SNR (Signal-to-Noise / Golub P-score)...")
            print("  - Calculating P(g,c) from training set only")
            print("  - Selecting top genes based on training data")
            print()
            run_snr_selection(
                k=k,
                out_dir=out_dir,
                train_csv=str(train_csv),
                ind_csv=str(ind_csv) if ind_csv.exists() else None,
                labels_csv=str(labels_csv),
                use_all_data=False,  # ALWAYS False - training only!
                balanced_genes=balanced_genes,  # NEW param
                use_balanced_patients=use_balanced_patients,  # NEW param
                validation_strategy=validation_strategy,  # NEW param
            )
            print("✓ SNR complete\n")

        if method in [3, 4]:  # SCAD-SVM or All
            print("Running SCAD-SVM (Smoothly Clipped Absolute Deviation)...")
            print("  - Using SCAD regularization for automatic feature selection")
            print("  - Lambda controls sparsity (auto-tuned via CV)")
            print()
            run_scad_selection(
                k=k,
                use_all_data=False,  # ALWAYS False - training only!
                auto_tune_lambda=True,
                output_dir=out_dir,
                verbose=True,
            )
            print("✓ SCAD-SVM complete\n")

        # Success summary
        print_separator()
        print("  ✓ FEATURE SELECTION COMPLETE")
        print_separator()
        print()
        print("Methodology Summary:")
        if use_balanced_patients:
            print("  • Patients selected: 22 balanced (11 ALL + 11 AML)")
        else:
            print("  • Patients selected: 38 all (27 ALL + 11 AML)")
        print("  • Genes selected from: Selected patients ONLY")
        if balanced_genes:
            print(f"  • Gene selection method: Balanced ({k//2} ALL + {k//2} AML)")
        else:
            print("  • Gene selection method: Pure top-k by score")
        print()

        if validation_strategy["method"] == "split":
            ratio = validation_strategy["ratio"]
            print(f"  • Internal split: {int(ratio*100)}/{int((1-ratio)*100)} train/test")
            print("    - train_internal_*.csv → for model training")
            print("    - test_internal_*.csv → for hyperparameter tuning")
        elif validation_strategy["method"] == "loocv":
            if use_balanced_patients:
                n_samples = 22  # 11 ALL + 11 AML
            else:
                n_samples = 38  # 27 ALL + 11 AML
            print(f"  • Cross-validation: LOOCV (Leave-One-Out)")
            print(f"    - {n_samples} iterations (one per patient)")
            print(f"    - fold_1_train/test through fold_{n_samples}_train/test")
        else:
            folds = validation_strategy["folds"]
            print(f"  • Cross-validation: {folds}-fold stratified CV")
            print(f"    - fold_1_train_*.csv through fold_{folds}_train_*.csv")
            print(f"    - fold_1_test_*.csv through fold_{folds}_test_*.csv")

        print("  • Independent test: Completely separate (34 samples)")
        print("    - independent_*.csv → for final validation ONLY")
        print()
        print("Output files in 'results/':")

        if method in [1, 4]:
            print(f"\n  ANOVA F-test:")
            print(f"    • topk_anova_f_{k}genes.csv")
            print(f"        - Full rankings, F-scores, class favorability")
            print(f"    • selected_genes_anova_f_{k}genes.csv")
            print(f"        - Gene accession numbers (for reference)")
            if validation_strategy["method"] == "split":
                print(f"    • train_internal_top_{k}_anova_f.csv")
                print(f"    • test_internal_top_{k}_anova_f.csv")
            else:
                folds = validation_strategy["folds"]
                print(f"    • fold_1_train/test through fold_{folds}_train/test (top_{k}_anova_f.csv)")
            if ind_csv.exists():
                print(f"    • independent_top_{k}_anova_f.csv")
                print(f"        - Independent: 34 samples × {k} genes")

        if method in [2, 4]:
            print(f"\n  SNR (Signal-to-Noise):")
            print(f"    • topk_snr_{k}genes.csv")
            print(f"        - Full rankings, P-scores, class favorability")
            print(f"    • selected_genes_snr_{k}genes.csv")
            print(f"        - Gene accession numbers (for reference)")
            if validation_strategy["method"] == "split":
                print(f"    • train_internal_top_{k}_snr.csv")
                print(f"    • test_internal_top_{k}_snr.csv")
            else:
                folds = validation_strategy["folds"]
                print(f"    • fold_1_train/test through fold_{folds}_train/test (top_{k}_snr.csv)")
            if ind_csv.exists():
                print(f"    • independent_top_{k}_snr.csv")
                print(f"        - Independent: 34 samples × {k} genes")

        if method in [3, 4]:
            print(f"\n  SCAD-SVM:")
            print(f"    • selected_genes_scad_{k}genes.csv")
            print(f"        - Gene rankings with SCAD weights")
            print(f"    • train_top_{k}_scad.csv")
            print(f"        - Training data with SCAD-selected genes")
            if ind_csv.exists():
                print(f"    • independent_top_{k}_scad.csv")
                print(f"        - Independent: 34 samples × {k} genes")

        print()
        print("Next Steps:")
        if validation_strategy["method"] == "split":
            print("  1. Use train_internal_*.csv to build your classifier")
            print("  2. Tune hyperparameters using test_internal_*.csv")
            print("  3. Test final model on independent_*.csv for validation")
            print("  4. Report accuracy on independent set as final result")
        elif validation_strategy["method"] == "loocv":
            print("  1. Use LOOCV folds (fold_*_train/test) for exhaustive validation")
            print("  2. Train on (n-1) samples, test on 1 sample per fold")
            print("  3. Average performance across all folds for robust estimation")
            print("  4. Test final model on independent_*.csv for validation")
            print("  5. Report accuracy on independent set as final result")
            print()
            print("  NOTE: LOOCV provides low-bias estimates but can be computationally expensive")
        else:
            print("  1. Use CV folds (fold_*_train/test) for model training and validation")
            print("  2. Tune hyperparameters using CV results")
            print("  3. Test final model on independent_*.csv for validation")
            print("  4. Report accuracy on independent set as final result")
        print()

    except Exception as e:
        print(f"\n[ERROR] Feature selection failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_preprocessing()