#!/usr/bin/env python3
"""
Interactive Preprocessing Interface for Gene Expression Feature Selection

Provides a user-friendly 3-question workflow for feature selection using:
- ANOVA F-test
- SNR (Signal-to-Noise Ratio / Golub method)

Generates ranked gene lists with class balance options.
"""

import sys
from pathlib import Path

# Add feature-selection-methods to path
sys.path.insert(0, str(Path(__file__).parent / "feature-selection-methods"))

from anova_f import run_feature_selection as run_anova
from signal_to_noise import run_snr_selection


def print_banner():
    """Print welcome banner."""
    print("=" * 70)
    print("  GENE EXPRESSION FEATURE SELECTION")
    print("  Golub Leukemia Dataset (ALL vs AML)")
    print("=" * 70)
    print()


def print_separator():
    """Print separator line."""
    print("=" * 70)
    print()


def ask_k_value():
    """Ask how many genes to select."""
    print("[1/3] How many genes/qubits to select?")
    print("      (Common values: 16, 24, 32, 50)")

    while True:
        try:
            k = int(input("      -> "))
            if 1 <= k <= 100:
                return k
            else:
                print("      [ERROR] Please enter a value between 1 and 100")
        except ValueError:
            print("      [ERROR] Please enter a valid integer")
        except KeyboardInterrupt:
            print("\n\n[INFO] Cancelled by user")
            sys.exit(0)


def ask_method():
    """Ask which feature selection method to use."""
    print("\n[2/3] Feature selection method:")
    print("      1. ANOVA F-test")
    print("      2. SNR (Signal-to-Noise Ratio / Golub)")
    print("      3. Both (run both methods)")

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
    print("      (Select k/2 ALL-favoring + k/2 AML-favoring genes)")

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


def print_config(k, method, balanced):
    """Print configuration summary."""
    print_separator()
    print("  Configuration:")
    print(f"  * Genes: {k}")

    method_names = {1: "ANOVA F-test", 2: "SNR", 3: "ANOVA + SNR"}
    print(f"  * Method: {method_names[method]}")

    balance_str = "Yes (k/2 per class)" if balanced else "No (top-k by score)"
    print(f"  * Gene Balance: {balance_str}")

    print("  * Dataset: Original Golub split (38 train / 34 test)")
    print_separator()
    print()


def run_preprocessing():
    """Main preprocessing workflow."""
    print_banner()

    # Ask questions
    k = ask_k_value()
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
            run_anova(
                input_train=str(train_csv),
                input_ind=str(ind_csv) if ind_csv.exists() else None,
                input_actual=None,
                k=k,
                out_dir=out_dir,
                balanced=balanced,
                labels_csv=str(labels_csv),
            )
            print("[OK] ANOVA complete\n")

        if method in [2, 3]:  # SNR or Both
            print("Running SNR (Signal-to-Noise)...")
            run_snr_selection(
                k=k,
                out_dir=out_dir,
                train_csv=str(train_csv),
                ind_csv=str(ind_csv) if ind_csv.exists() else None,
                labels_csv=str(labels_csv),
            )
            print("[OK] SNR complete\n")

        # Success summary
        print_separator()
        print("  [OK] ALL DONE!")
        print_separator()
        print()
        print("Output files created in results/:")

        if method in [1, 3]:
            print(f"  * topk_anova_f_{k}genes.csv           - ANOVA rankings + scores")
            print(f"  * selected_genes_anova_f_{k}genes.csv - ANOVA gene list")
            print(f"  * train_top_{k}_anova_f.csv           - Training data")
            if ind_csv.exists():
                print(f"  * independent_top_{k}_anova_f.csv     - Test data")

        if method in [2, 3]:
            print(f"  * topk_snr_{k}genes.csv               - SNR rankings + scores")
            print(f"  * selected_genes_snr_{k}genes.csv     - SNR gene list")
            print(f"  * train_top_{k}_snr.csv               - Training data")
            if ind_csv.exists():
                print(f"  * independent_top_{k}_snr.csv         - Test data")

        print()

    except Exception as e:
        print(f"\n[ERROR] Feature selection failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_preprocessing()
