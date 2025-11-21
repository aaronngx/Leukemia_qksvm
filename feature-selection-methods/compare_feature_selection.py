"""Compare SNR and ANOVA F-test feature selection methods."""
import argparse
import sys
import subprocess


def run_experiment(method: str, k: int):
    """Run feature selection and QKSVM (method: 'snr' or 'anova_f')."""
    print(f"\n{'='*70}")
    print(f"Running {method.upper()} with {k} qubits")
    print(f"{'='*70}\n")

    if method == 'snr':
        script = "feature-selection-methods/signal_to_noise_ratio.py"
        out_dir = f"data/processed_{method}_k{k}"
        train_csv = f"{out_dir}/train_topk_snr.csv"
        ind_csv = f"{out_dir}/independent_topk_snr.csv"
    elif method == 'anova_f':
        script = "feature-selection-methods/anova_f.py"
        out_dir = f"data/processed_{method}_k{k}"
        train_csv = f"{out_dir}/train_topk_anova_f.csv"
        ind_csv = f"{out_dir}/independent_topk_anova_f.csv"
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"[1/2] Running {method.upper()} feature selection for k={k}...")
    result = subprocess.run([
        sys.executable,
        script,
        "--k", str(k),
        "--out_dir", out_dir
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR in feature selection: {result.stderr}")
        return False

    print(result.stdout)

    print(f"[2/2] Running QKSVM with {k} qubits...")
    result = subprocess.run([
        sys.executable,
        "qksvm_golub.py",
        "--train_csv", train_csv,
        "--ind_csv", ind_csv,
        "--output_dir", f"results/qksvm_{method}_k{k}"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR in QKSVM: {result.stderr}")
        return False

    print(result.stdout)
    print(f"\n✓ Completed {method.upper()} experiment with {k} qubits\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Compare SNR and ANOVA F-test feature selection methods"
    )
    parser.add_argument(
        "--k_values",
        nargs='+',
        type=int,
        default=[2, 4, 8, 16],
        help="List of k values (number of features/qubits) to test"
    )
    parser.add_argument(
        "--methods",
        nargs='+',
        choices=['snr', 'anova_f', 'both'],
        default=['both'],
        help="Which feature selection methods to run"
    )
    args = parser.parse_args()

    # Determine which methods to run
    if 'both' in args.methods:
        methods = ['snr', 'anova_f']
    else:
        methods = args.methods

    print("="*70)
    print("QKSVM Feature Selection Comparison")
    print("="*70)
    print(f"Methods: {', '.join(m.upper() for m in methods)}")
    print(f"k values: {args.k_values}")
    print("="*70)

    results_summary = []

    for k in args.k_values:
        for method in methods:
            success = run_experiment(method, k)
            results_summary.append({
                'k': k,
                'method': method,
                'success': success
            })

    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    for result in results_summary:
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"{result['method'].upper():10s} k={result['k']:2d}  {status}")

    print("\n" + "="*70)
    print("Results saved in:")
    print("="*70)
    for k in args.k_values:
        for method in methods:
            print(f"  - results/qksvm_{method}_k{k}/")

    print("\nFeature scores saved in:")
    for k in args.k_values:
        for method in methods:
            if method == 'snr':
                print(f"  - data/processed_{method}_k{k}/  (no scores file for SNR)")
            else:
                print(f"  - data/processed_{method}_k{k}/anova_f_scores.csv")


if __name__ == "__main__":
    main()
