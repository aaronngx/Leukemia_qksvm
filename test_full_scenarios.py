"""
COMPREHENSIVE experiment covering all scenarios from figures.

Configuration:
  - Angle (statevector): K=4, 8, 16
  - Amplitude (statevector): K=4, 8, 16, 32, 50
  - Angle (tensor_network): K=32, 50
  - Feature selection: ANOVA, SNR
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from Experiment import run_all_experiments


def test_full():
    """
    Complete experiment matrix covering all requested scenarios.
    """

    # Create output directory first
    output_dir = Path("results_full")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FULL EXPERIMENT TEST")
    print("=" * 70)
    print("\nConfiguration:")
    print("  - Angle (statevector): K=4, 8, 16")
    print("  - Amplitude (statevector): K=4, 8, 16, 32, 50")
    print("  - Angle (tensor_network): K=32, 50")
    print("  - Feature selection: ANOVA, SNR")
    print("=" * 70)

    all_results = []

    # ========================================================================
    # PART 1: Angle Encoding (Statevector) - K=4, 8, 16
    # ========================================================================
    print("\n" + "#" * 70)
    print("# PART 1: Angle Encoding (Statevector) - K=4, 8, 16")
    print("#" * 70)

    try:
        results1 = run_all_experiments(
            feature_methods=["anova", "snr"],
            k_values=[4, 8, 16],
            encodings=["angle"],
            models=["qksvm"],
            kernel_methods=["statevector"],
            output_dir="results_full",
            test_size=0.3,
            seed=42,
            max_bond_dimension=100,
        )
        all_results.extend(results1)
        print(f"\nPart 1 complete: {len(results1)} experiments")
    except Exception as e:
        print(f"Part 1 failed: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # PART 2: Amplitude Encoding - K=4, 8, 16, 32, 50
    # ========================================================================
    print("\n" + "#" * 70)
    print("# PART 2: Amplitude Encoding - K=4, 8, 16, 32, 50")
    print("#" * 70)

    try:
        results2 = run_all_experiments(
            feature_methods=["anova", "snr"],
            k_values=[4, 8, 16, 32, 50],
            encodings=["amplitude"],
            models=["qksvm"],
            kernel_methods=["statevector"],
            output_dir="results_full",
            test_size=0.3,
            seed=42,
            max_bond_dimension=100,
        )
        all_results.extend(results2)
        print(f"\nPart 2 complete: {len(results2)} experiments")
    except Exception as e:
        print(f"Part 2 failed: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # PART 3: Angle Encoding (Tensor Network) - K=32, 50
    # ========================================================================
    print("\n" + "#" * 70)
    print("# PART 3: Angle Encoding (Tensor Network) - K=32, 50")
    print("#" * 70)

    try:
        results3 = run_all_experiments(
            feature_methods=["anova", "snr"],
            k_values=[32, 50],
            encodings=["angle"],
            models=["qksvm"],
            kernel_methods=["tensor_network"],
            output_dir="results_full",
            test_size=0.3,
            seed=42,
            max_bond_dimension=100,
        )
        all_results.extend(results3)
        print(f"\nPart 3 complete: {len(results3)} experiments")
    except Exception as e:
        print(f"Part 3 failed: {e}")
        import traceback
        traceback.print_exc()

    # ========================================================================
    # RESULTS SUMMARY - Requested Format
    # ========================================================================
    print("\n" + "=" * 130)
    print("COMPREHENSIVE RESULTS - TEST SET PERFORMANCE")
    print("=" * 130)

    # Header in requested format
    print(f"{'Selection':<10} {'Optimizer':<12} {'Split':<8} {'Layer':<6} {'Qubit':<6} "
          f"{'Epochs':<8} {'Accuracy':<10} {'AUROC':<10} {'F-Score':<10} {'Recall':<10}")
    print("-" * 130)

    for r in all_results:
        def fmt(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "N/A"
            return f"{val:.4f}"

        # Map fields to requested columns
        selection = r.get('feature_selection', 'N/A').upper()
        optimizer = r.get('optimizer', r.get('kernel_method', 'N/A'))  # kernel_method for QKSVM
        split = "70/30"  # test_size=0.3
        layer = str(r.get('ansatz_reps', 1))  # Default 1 for QKSVM
        qubit = str(r.get('n_qubits', 'N/A'))
        epochs = str(r.get('max_iter', 'N/A'))

        print(
            f"{selection:<10} "
            f"{optimizer:<12} "
            f"{split:<8} "
            f"{layer:<6} "
            f"{qubit:<6} "
            f"{epochs:<8} "
            f"{fmt(r.get('ind_acc')):<10} "
            f"{fmt(r.get('ind_auroc')):<10} "
            f"{fmt(r.get('ind_f1')):<10} "
            f"{fmt(r.get('ind_recall')):<10}"
        )

    print("=" * 130)

    # Statistics
    successful = [r for r in all_results if r.get('ind_acc') is not None]

    print(f"\nTotal experiments: {len(all_results)}")
    print(f"Successful: {len(successful)}")

    # ========================================================================
    # SAVE RESULTS IN REQUESTED FORMAT
    # ========================================================================
    # Create DataFrame with requested column names
    formatted_results = []
    for r in all_results:
        formatted_results.append({
            'selection': r.get('feature_selection', 'N/A').upper(),
            'optimizer': r.get('optimizer', r.get('kernel_method', 'N/A')),
            'count_split': '70/30',
            'layer': r.get('ansatz_reps', 1),
            'qubit': r.get('n_qubits', 'N/A'),
            'epochs': r.get('max_iter', 'N/A'),
            'accuracy': r.get('ind_acc'),
            'AUROC': r.get('ind_auroc'),
            'F-score': r.get('ind_f1'),
            'recall': r.get('ind_recall'),
            # Also keep original fields for reference
            'encoding': r.get('encoding'),
            'k_features': r.get('k'),
        })

    results_df = pd.DataFrame(formatted_results)
    csv_path = output_dir / "comprehensive_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Also save raw results for full detail
    raw_df = pd.DataFrame(all_results)
    raw_path = output_dir / "raw_results.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"Raw results saved to: {raw_path}")

    # Scaling comparison table
    scaling_data = []
    for k in [4, 8, 16, 32, 50]:
        angle_res = [r for r in successful if r['k']==k and r['encoding']=='angle']
        amp_res = [r for r in successful if r['k']==k and r['encoding']=='amplitude']

        angle_qubits = angle_res[0]['n_qubits'] if angle_res else k
        amp_qubits = amp_res[0]['n_qubits'] if amp_res else int(np.ceil(np.log2(k)))
        reduction = f"{(1 - amp_qubits/k)*100:.0f}%"

        # Best accuracy for each encoding
        angle_best = max([r['ind_acc'] for r in angle_res]) if angle_res else None
        amp_best = max([r['ind_acc'] for r in amp_res]) if amp_res else None

        scaling_data.append({
            'K_features': k,
            'Angle_Qubits': angle_qubits,
            'Amplitude_Qubits': amp_qubits,
            'Qubit_Reduction': reduction,
            'Angle_Best_Acc': f"{angle_best:.4f}" if angle_best else "N/A",
            'Amplitude_Best_Acc': f"{amp_best:.4f}" if amp_best else "N/A",
        })

    scaling_df = pd.DataFrame(scaling_data)
    scaling_path = output_dir / "scaling_comparison.csv"
    scaling_df.to_csv(scaling_path, index=False)
    print(f"Scaling comparison saved to: {scaling_path}")

    print("\n" + scaling_df.to_string(index=False))

    # Top performers
    print("\n" + "=" * 80)
    print("TOP 5 CONFIGURATIONS BY TEST ACCURACY")
    print("=" * 80)

    if successful:
        top_5_acc = sorted(successful, key=lambda x: x['ind_acc'], reverse=True)[:5]
        for i, r in enumerate(top_5_acc, 1):
            auroc = f", AUROC={r.get('ind_auroc', 0):.4f}" if r.get('ind_auroc') else ""
            f1 = f", F1={r.get('ind_f1', 0):.4f}" if r.get('ind_f1') else ""
            recall = f", Recall={r.get('ind_recall', 0):.4f}" if r.get('ind_recall') else ""
            print(f"\n{i}. Acc={r['ind_acc']:.4f}{auroc}{f1}{recall}")
            print(f"   {r['feature_selection'].upper()} K={r['k']} {r['encoding']} "
                  f"{r.get('kernel_method', 'N/A')} ({r['n_qubits']} qubits)")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    try:
        results = test_full()
        sys.exit(0)
    except Exception as e:
        print(f"\n{'='*80}")
        print("CRITICAL ERROR")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
