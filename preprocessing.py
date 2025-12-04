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
    """Ask how many genes to select."""
    print("[1/3] How many genes/qubits to select?")
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
            return k
        except ValueError:
            print("      [ERROR] Please enter a valid integer")
        except KeyboardInterrupt:
            print("\n\n[INFO] Cancelled by user")
            sys.exit(0)


def ask_method():
    """Ask which feature selection method to use."""
    print("\n[2/3] Feature selection method:")
    print("      1. ANOVA F-test")
    print("      2. SNR (Signal-to-Noise Ratio / Golub's P-score)")
    print("      3. Both methods")

    while True:
        try:
            choice = input("      -> ")
            if choice in ["1", "2", "3"]:
                return int(choice)
            else:
                print("      [ERROR] Please enter 1, 2, or 3")
        except KeyboardInterrupt:
            print("\n\n[INFO] Cancelled by user")
            sys.exit(0)


def ask_balanced():
    """Ask whether to use balanced gene selection."""
    print("\n[3/3] Use balanced gene selection?")
    print("      y = Yes (k/2 ALL-favoring + k/2 AML-favoring)")
    print("      n = No (pure top-k by score)")
    print()
    print("      Note: Golub used balanced selection (25+25=50)")

    while True:
        try:
            choice = input("      -> (y/n): ").lower().strip()
            if choice in ["y", "yes"]:
                return True
            elif choice in ["n", "no"]:
                return False
            else:
                print("      [ERROR] Please enter 'y' or 'n'")
        except KeyboardInterrupt:
            print("\n\n[INFO] Cancelled by user")
            sys.exit(0)


def print_config(k: int, method: int, balanced: bool):
    """Print configuration summary."""
    print()
    print_separator()
    print("  Configuration Summary:")
    print(f"  • Genes/Qubits: {k}")
    
    method_names = {1: "ANOVA F-test", 2: "SNR (Golub P-score)", 3: "ANOVA + SNR"}
    print(f"  • Method: {method_names[method]}")
    
    balance_str = f"Yes ({k//2} ALL + {k//2} AML)" if balanced else "No (top-k by score)"
    print(f"  • Gene Balance: {balance_str}")
    
    print()
    print("  Data Strategy (Golub Methodology):")
    print("  • Gene selection: Training set ONLY (38 samples)")
    print("    - 27 ALL patients")
    print("    - 11 AML patients")
    print("  • Independent test: Validation ONLY (34 samples)")
    print("    - Never used for gene selection")
    print("    - Used only to test predictor accuracy")
    print_separator()
    print()


def run_preprocessing():
    """Main preprocessing workflow."""
    print_banner()

    # Get user inputs
    k = ask_gene_count()
    method = ask_method()
    balanced = ask_balanced()

    # Show configuration
    print_config(k, method, balanced)

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
        if method in [1, 3]:  # ANOVA or Both
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
                balanced=balanced,
                labels_csv=str(labels_csv),
                use_all_data=False,  # ALWAYS False - training only!
            )
            print("✓ ANOVA complete\n")

        if method in [2, 3]:  # SNR or Both
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
                balanced=balanced,
            )
            print("✓ SNR complete\n")

        # Success summary
        print_separator()
        print("  ✓ FEATURE SELECTION COMPLETE")
        print_separator()
        print()
        print("Methodology Summary:")
        print("  • Genes selected from: 38 training samples ONLY")
        print("  • Selected genes applied to:")
        print("    - Training set (38 samples) → for model building")
        print("    - Independent test (34 samples) → for validation")
        print()
        print("Output files in 'results/':")

        if method in [1, 3]:
            print(f"\n  ANOVA F-test:")
            print(f"    • topk_anova_f_{k}genes.csv")
            print(f"        - Full rankings, F-scores, class favorability")
            print(f"    • selected_genes_anova_f_{k}genes.csv")
            print(f"        - Gene accession numbers (for reference)")
            print(f"    • train_top_{k}_anova_f.csv")
            print(f"        - Training data: 38 samples × {k} genes")
            if ind_csv.exists():
                print(f"    • independent_top_{k}_anova_f.csv")
                print(f"        - Test data: 34 samples × {k} genes (same genes!)")

        if method in [2, 3]:
            print(f"\n  SNR (Signal-to-Noise):")
            print(f"    • topk_snr_{k}genes.csv")
            print(f"        - Full rankings, P-scores, class favorability")
            print(f"    • selected_genes_snr_{k}genes.csv")
            print(f"        - Gene accession numbers (for reference)")
            print(f"    • train_top_{k}_snr.csv")
            print(f"        - Training data: 38 samples × {k} genes")
            if ind_csv.exists():
                print(f"    • independent_top_{k}_snr.csv")
                print(f"        - Test data: 34 samples × {k} genes (same genes!)")

        print()
        print("Next Steps:")
        print("  1. Use train_*.csv to build your classifier")
        print("  2. Test classifier on independent_*.csv for validation")
        print("  3. Report accuracy on independent set as final result")
        print()

    except Exception as e:
        print(f"\n[ERROR] Feature selection failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_preprocessing()