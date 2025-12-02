#!/usr/bin/env python3
"""
Interactive Preprocessing Interface for Gene Expression Feature Selection

Provides a user-friendly workflow for feature selection using:
- ANOVA F-test
- SNR (Signal-to-Noise Ratio / Golub method)

Handles gene limits based on balancing and data split choices.
"""

import sys
from pathlib import Path

# Add feature-selection-methods to path
sys.path.insert(0, str(Path(__file__).parent / "feature-selection-methods"))

from anova_f import run_feature_selection as run_anova
from signal_to_noise import run_snr_selection


# Constants for balanced selection limits
# Based on minority class sizes: 11 AML in train, 25 AML total
# Balanced mode uses equal patients from each class
PATIENTS_TRAIN_BALANCED = 22   # 11 ALL + 11 AML
PATIENTS_ALL_BALANCED = 50     # 25 ALL + 25 AML
MAX_GENES_TRAIN_BALANCED = 22  # Limited by sample size
MAX_GENES_ALL_BALANCED = 50    # Limited by sample size


def print_banner():
    """Print welcome banner."""
    print()
    print("=" * 70)
    print("  GENE EXPRESSION FEATURE SELECTION")
    print("  Golub Leukemia Dataset (ALL vs AML)")
    print("=" * 70)
    print()


def print_separator():
    """Print separator line."""
    print("=" * 70)


def ask_gene_count(max_genes: int = None, balanced: bool = False, data_source: str = None):
    """Ask how many genes to select.
    
    Args:
        max_genes: Maximum allowed genes (None for no limit)
        balanced: Whether balanced selection is being used
        data_source: 'train' or 'all' for context in prompts
    """
    print("[1/4] How many genes/qubits to select?")
    if max_genes:
        print(f"      (Balanced mode with {data_source} data: max {max_genes} genes)")
        print(f"      Common values: 16, 22" + (", 32, 50" if max_genes >= 50 else ""))
    else:
        print("      (Common values: 16, 24, 32, 50)")

    while True:
        try:
            k = int(input("      -> "))
            if k < 1:
                print("      [ERROR] Please enter a positive number")
                continue
            if max_genes and k > max_genes:
                print(f"      [ERROR] Maximum {max_genes} genes for balanced selection with {data_source} data")
                continue
            if k > 7129:  # Total genes in Golub dataset
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
    print("\n[2/4] Feature selection method:")
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
    print("\n[3/4] Use balanced gene selection?")
    print("      (Select k/2 ALL-favoring + k/2 AML-favoring genes)")
    print("      y = Yes (recommended for biological interpretation)")
    print("      n = No (pure top-k by score)")

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


def ask_data_source(balanced: bool):
    """Ask whether to use training data only or all data."""
    print("\n[4/4] Data source for feature selection:")
    if balanced:
        print(f"      1. Training data only ({PATIENTS_TRAIN_BALANCED} balanced samples: 11 ALL + 11 AML)")
        print(f"      2. All data ({PATIENTS_ALL_BALANCED} balanced samples: 25 ALL + 25 AML)")
    else:
        print("      1. Training data only (38 samples) -> separate train/test output")
        print("      2. All data (72 samples) -> single combined output")

    while True:
        try:
            choice = input("      -> ")
            if choice == "1":
                return "train"
            elif choice == "2":
                return "all"
            else:
                print("      [ERROR] Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\n\n[INFO] Cancelled by user")
            sys.exit(0)


def validate_gene_count(k: int, balanced: bool, data_source: str) -> int:
    """Validate and potentially re-ask for gene count based on constraints.
    
    Returns the validated (possibly new) gene count.
    """
    if not balanced:
        return k  # No limit for unbalanced
    
    max_genes = MAX_GENES_ALL_BALANCED if data_source == "all" else MAX_GENES_TRAIN_BALANCED
    
    if k > max_genes:
        print(f"\n[WARNING] You selected {k} genes, but balanced selection with {data_source} data")
        print(f"          allows maximum {max_genes} genes (limited by minority class size).")
        print(f"\nOptions:")
        print(f"  1. Reduce to {max_genes} genes")
        print(f"  2. Enter a different number")
        
        while True:
            try:
                choice = input("      -> ")
                if choice == "1":
                    return max_genes
                elif choice == "2":
                    return ask_gene_count(max_genes, balanced, data_source)
                else:
                    print("      [ERROR] Please enter 1 or 2")
            except KeyboardInterrupt:
                print("\n\n[INFO] Cancelled by user")
                sys.exit(0)
    
    return k


def print_config(k: int, method: int, balanced: bool, data_source: str):
    """Print configuration summary."""
    print()
    print_separator()
    print("  Configuration Summary:")
    print(f"  * Genes/Qubits: {k}")
    
    method_names = {1: "ANOVA F-test", 2: "SNR (Signal-to-Noise)", 3: "ANOVA + SNR"}
    print(f"  * Method: {method_names[method]}")
    
    balance_str = f"Yes (k/2 per class = {k//2} each)" if balanced else "No (top-k by score)"
    print(f"  * Gene Balance: {balance_str}")
    
    if balanced:
        if data_source == "train":
            print(f"  * Data Source: Training only ({PATIENTS_TRAIN_BALANCED} balanced samples)")
            print("  * Output: Separate train/test files")
        else:
            print(f"  * Data Source: All data ({PATIENTS_ALL_BALANCED} balanced samples)")
            print("  * Output: Single combined file")
    else:
        if data_source == "train":
            print("  * Data Source: Training only (38 samples)")
            print("  * Output: Separate train/test files")
        else:
            print("  * Data Source: All data (72 samples)")
            print("  * Output: Single combined file")
    
    print_separator()
    print()


def run_preprocessing():
    """Main preprocessing workflow."""
    print_banner()

    # Step 1: Ask about balancing first (affects gene limits)
    balanced = ask_balanced()
    
    # Step 2: Ask about data source (affects gene limits for balanced)
    data_source = ask_data_source(balanced)
    
    # Step 3: Ask gene count with appropriate limits
    if balanced:
        max_genes = MAX_GENES_ALL_BALANCED if data_source == "all" else MAX_GENES_TRAIN_BALANCED
        k = ask_gene_count(max_genes, balanced, data_source)
    else:
        k = ask_gene_count()
    
    # Step 4: Ask method
    method = ask_method()

    # Show configuration
    print_config(k, method, balanced, data_source)

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

    use_all_data = (data_source == "all")

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
                use_all_data=use_all_data,
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
                use_all_data=use_all_data,
                balanced=balanced,
            )
            print("[OK] SNR complete\n")

        # Success summary
        print_separator()
        print("  ALL DONE!")
        print_separator()
        print()
        print("Output files created in results/:")

        # Determine sample counts for output description
        if balanced:
            all_samples = PATIENTS_ALL_BALANCED
            train_samples = PATIENTS_TRAIN_BALANCED
        else:
            all_samples = 72
            train_samples = 38
        
        if method in [1, 3]:
            print(f"\n  ANOVA F-test:")
            print(f"    * topk_anova_f_{k}genes.csv           - Rankings + scores")
            print(f"    * selected_genes_anova_f_{k}genes.csv - Gene list (accession numbers)")
            if use_all_data:
                print(f"    * all_top_{k}_anova_f.csv             - Combined data ({all_samples} samples)")
            else:
                print(f"    * train_top_{k}_anova_f.csv           - Training data ({train_samples} samples)")
                if ind_csv.exists() and not balanced:
                    print(f"    * independent_top_{k}_anova_f.csv     - Test data (34 samples)")

        if method in [2, 3]:
            print(f"\n  SNR (Signal-to-Noise):")
            print(f"    * topk_snr_{k}genes.csv               - Rankings + scores")
            print(f"    * selected_genes_snr_{k}genes.csv     - Gene list (accession numbers)")
            if use_all_data:
                print(f"    * all_top_{k}_snr.csv                 - Combined data ({all_samples} samples)")
            else:
                print(f"    * train_top_{k}_snr.csv               - Training data ({train_samples} samples)")
                if ind_csv.exists() and not balanced:
                    print(f"    * independent_top_{k}_snr.csv         - Test data (34 samples)")

        print()

    except Exception as e:
        print(f"\n[ERROR] Feature selection failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_preprocessing()
