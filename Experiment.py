"""
Unified experiment runner for Quantum Machine Learning on Golub Leukemia dataset.

Supports:
- Feature selection: ANOVA F-test, SNR (Signal-to-Noise), or Both
- Encoding: Angle, Amplitude, or Both
- Models: VQC, QKSVM, or Both
- Kernel methods (QKSVM only): Statevector, Tensor Network, Swap Test, Hadamard Test
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from qiskit import QuantumCircuit

from angle_encoding import angle_encoding_circuit
from amplitude_encoding import get_num_qubits, build_amplitude_kernel
from backend_config import (
    BackendType,
    KernelMethod,
    compute_kernel_element_statevector,
    compute_kernel_element_tensor_network,
    compute_kernel_element_swap_test,
    compute_kernel_element_hadamard_test,
)

# Import VQC components (optional - requires qiskit_machine_learning)
VQC_AVAILABLE = False
try:
    from vqc_golub import train_eval_vqc, scale_to_angle
    VQC_AVAILABLE = True
except ImportError:
    print("[WARNING] VQC not available (qiskit_machine_learning not installed)")
    train_eval_vqc = None
    scale_to_angle = None

# Thresholds
MAX_STATEVECTOR_QUBITS = 20  # Safe limit for statevector simulation


# =============================================================================
# Feature Selection
# =============================================================================

def run_feature_selection(method: str, k: int, output_dir: str = "results"):
    """Run feature selection and return paths to generated files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if method == "snr":
        cmd = [
            sys.executable,
            "feature-selection-methods/signal_to_noise.py",
            "--k", str(k),
            "--out_dir", output_dir,
        ]
        print(f"[INFO] Running SNR feature selection with K={k}...")
        subprocess.run(cmd, check=True)
        train_csv = output_path / f"train_top_{k}_snr.csv"
        ind_csv = output_path / f"independent_top_{k}_snr.csv"

    else:  # anova
        cmd = [
            sys.executable,
            "feature-selection-methods/anova_f.py",
            "--k", str(k),
            "--out_dir", output_dir,
        ]
        print(f"[INFO] Running ANOVA-F feature selection with K={k}...")
        subprocess.run(cmd, check=True)
        train_csv = output_path / f"train_top_{k}_anova_f.csv"
        ind_csv = output_path / f"independent_top_{k}_anova_f.csv"

    return train_csv, ind_csv


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_data(train_csv: str, ind_csv: str = None):
    """Load processed CSV files."""
    df_train = pd.read_csv(train_csv)
    y_train = df_train["label"].values
    X_train = df_train.drop(columns=["label"]).values

    if ind_csv and Path(ind_csv).exists():
        df_ind = pd.read_csv(ind_csv)
        y_ind = df_ind["label"].values
        X_ind = df_ind.drop(columns=["label"]).values
    else:
        X_ind, y_ind = None, None

    return (X_train, y_train), (X_ind, y_ind)


def scale_features(X: np.ndarray):
    """Scale features to [0, pi] for angle encoding."""
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    return scaler.fit_transform(X), scaler


# =============================================================================
# Kernel Building Functions
# =============================================================================

def build_kernel(
    XA: np.ndarray,
    XB: np.ndarray,
    feature_map: QuantumCircuit,
    x_params,
    kernel_method: str = "statevector",
    max_bond_dimension: int = 100,
):
    """Build quantum kernel matrix."""
    n_qubits = feature_map.num_qubits
    K = np.zeros((len(XA), len(XB)))

    print(f"    Computing {len(XA)} x {len(XB)} kernel elements...")
    print(f"    Kernel method: {kernel_method}")

    for i, x in enumerate(XA):
        if i % 5 == 0 or i == len(XA) - 1:
            print(f"      Progress: {i+1}/{len(XA)} rows")

        for j, z in enumerate(XB):
            bind_x = {x_params[k]: float(x[k]) for k in range(n_qubits)}
            bind_z = {x_params[k]: float(z[k]) for k in range(n_qubits)}

            if kernel_method == "swap_test":
                qc_x = feature_map.assign_parameters(bind_x)
                qc_z = feature_map.assign_parameters(bind_z)
                K[i, j] = compute_kernel_element_swap_test(qc_x, qc_z)

            elif kernel_method == "hadamard_test":
                qc = QuantumCircuit(n_qubits)
                qc.compose(feature_map.assign_parameters(bind_x), inplace=True)
                qc.compose(feature_map.assign_parameters(bind_z).inverse(), inplace=True)
                K[i, j] = compute_kernel_element_hadamard_test(qc)

            elif kernel_method == "tensor_network":
                qc = QuantumCircuit(n_qubits)
                qc.compose(feature_map.assign_parameters(bind_x), inplace=True)
                qc.compose(feature_map.assign_parameters(bind_z).inverse(), inplace=True)
                K[i, j] = compute_kernel_element_tensor_network(qc, max_bond_dimension)

            else:  # statevector
                qc = QuantumCircuit(n_qubits)
                qc.compose(feature_map.assign_parameters(bind_x), inplace=True)
                qc.compose(feature_map.assign_parameters(bind_z).inverse(), inplace=True)
                K[i, j] = compute_kernel_element_statevector(qc)

    return K


# =============================================================================
# Model Runners
# =============================================================================

def run_qksvm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_ind: np.ndarray,
    y_ind: np.ndarray,
    encoding: str,
    n_features: int,
    kernel_method: str = "statevector",
    max_bond_dimension: int = 100,
):
    """Run QKSVM model and return results."""
    results = {"model": "QKSVM", "encoding": encoding, "kernel_method": kernel_method}

    if encoding == "amplitude":
        # Amplitude encoding: K features -> log2(K) qubits
        n_qubits = get_num_qubits(n_features)
        print(f"    Amplitude encoding: {n_features} features -> {n_qubits} qubits")
        print(f"    (Kernel method N/A for amplitude - uses inner product)")
        K_train = build_amplitude_kernel(X_train, X_train, verbose=False)
        K_val = build_amplitude_kernel(X_val, X_train, verbose=False)
    else:
        # Angle encoding: K features -> K qubits
        n_qubits = n_features
        print(f"    Angle encoding: {n_features} features -> {n_qubits} qubits")

        # Check if we need tensor network for statevector-based methods
        actual_method = kernel_method
        if n_qubits > MAX_STATEVECTOR_QUBITS and kernel_method == "statevector":
            actual_method = "tensor_network"
            print(f"    [AUTO] Switching to tensor network for {n_qubits} qubits")

        feature_map, x_params = angle_encoding_circuit(n_features)
        K_train = build_kernel(X_train, X_train, feature_map, x_params,
                               kernel_method=actual_method, max_bond_dimension=max_bond_dimension)
        K_val = build_kernel(X_val, X_train, feature_map, x_params,
                             kernel_method=actual_method, max_bond_dimension=max_bond_dimension)

    # Train SVM
    clf = SVC(kernel="precomputed", C=1.0)
    clf.fit(K_train, y_train)

    # Validation
    y_val_pred = clf.predict(K_val)
    results["val_acc"] = accuracy_score(y_val, y_val_pred)

    # Independent test
    if X_ind is not None:
        if encoding == "amplitude":
            K_ind = build_amplitude_kernel(X_ind, X_train, verbose=False)
        else:
            K_ind = build_kernel(X_ind, X_train, feature_map, x_params,
                                 kernel_method=actual_method, max_bond_dimension=max_bond_dimension)
        y_ind_pred = clf.predict(K_ind)
        results["ind_acc"] = accuracy_score(y_ind, y_ind_pred)

        # Compute additional metrics
        try:
            # For AUROC, need decision_function or predict_proba
            # SVC with precomputed kernel uses decision_function
            y_ind_scores = clf.decision_function(K_ind)
            results["ind_auroc"] = roc_auc_score(y_ind, y_ind_scores)
        except Exception:
            results["ind_auroc"] = None

        results["ind_f1"] = f1_score(y_ind, y_ind_pred, average='binary')
        results["ind_recall"] = recall_score(y_ind, y_ind_pred, average='binary')
    else:
        results["ind_acc"] = None
        results["ind_auroc"] = None
        results["ind_f1"] = None
        results["ind_recall"] = None

    results["n_qubits"] = n_qubits
    return results


def run_vqc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_ind: np.ndarray,
    y_ind: np.ndarray,
    encoding: str,
    n_features: int,
    max_bond_dimension: int = 100,
    max_iter: int = 200,
    optimizer: str = "COBYLA",
):
    """Run VQC model and return results."""
    if not VQC_AVAILABLE:
        print("    [SKIP] VQC not available (qiskit_machine_learning not installed)")
        return None

    results = {"model": "VQC", "encoding": encoding, "kernel_method": "N/A"}

    if encoding == "amplitude":
        print("    [SKIP] VQC with amplitude encoding not implemented")
        print("           (VQC requires angle encoding for feature map)")
        return None

    n_qubits = n_features
    print(f"    Angle encoding: {n_features} features -> {n_qubits} qubits")

    # Determine backend
    if n_qubits > MAX_STATEVECTOR_QUBITS:
        backend_type = BackendType.TENSOR_NETWORK
        print(f"    [AUTO] Using tensor network for {n_qubits} qubits")
    else:
        backend_type = BackendType.STATEVECTOR

    # Combine train and val for VQC (it does its own split internally)
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])

    ind_data = (X_ind, y_ind) if X_ind is not None else None

    try:
        circuit, clf = train_eval_vqc(
            X_combined,
            y_combined,
            ind_data=ind_data,
            reps=2,
            test_size=0.3,
            seed=42,
            backend_type=backend_type,
            max_bond_dimension=max_bond_dimension,
            max_iter=max_iter,
            optimizer_name=optimizer,
        )

        # Get accuracy from the trained model (re-evaluate on our val set)
        X_val_scaled, _ = scale_to_angle(X_val)
        y_val_pred = clf.predict(X_val_scaled)
        results["val_acc"] = accuracy_score(y_val, y_val_pred)

        if X_ind is not None:
            X_ind_scaled, _ = scale_to_angle(X_ind)
            y_ind_pred = clf.predict(X_ind_scaled)
            results["ind_acc"] = accuracy_score(y_ind, y_ind_pred)

            # Compute additional metrics
            try:
                # Try to get probability scores for AUROC
                y_ind_proba = clf.predict_proba(X_ind_scaled)
                if y_ind_proba.shape[1] == 2:
                    results["ind_auroc"] = roc_auc_score(y_ind, y_ind_proba[:, 1])
                else:
                    results["ind_auroc"] = None
            except Exception:
                results["ind_auroc"] = None

            results["ind_f1"] = f1_score(y_ind, y_ind_pred, average='binary')
            results["ind_recall"] = recall_score(y_ind, y_ind_pred, average='binary')
        else:
            results["ind_acc"] = None
            results["ind_auroc"] = None
            results["ind_f1"] = None
            results["ind_recall"] = None

        results["n_qubits"] = n_qubits
        results["optimizer"] = optimizer
        results["max_iter"] = max_iter

    except Exception as e:
        print(f"    [ERROR] VQC failed: {e}")
        return None

    return results


# =============================================================================
# Interactive Menu Functions
# =============================================================================

def prompt_menu(prompt: str, options: list, allow_multiple: bool = False) -> list:
    """Display a menu and get user choice(s)."""
    print(f"\n{prompt}")
    for i, (key, desc) in enumerate(options):
        print(f"  {i + 1}. {desc}")

    if allow_multiple:
        hint = f"Enter choices (1-{len(options)}, space-separated): "
    else:
        hint = f"Enter choice (1-{len(options)}): "

    while True:
        try:
            choice = input(hint).strip()
            if allow_multiple:
                indices = [int(c) - 1 for c in choice.split()]
                if all(0 <= idx < len(options) for idx in indices):
                    return [options[idx][0] for idx in indices]
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return [options[idx][0]]
            print(f"  Invalid choice. Please enter 1-{len(options)}")
        except ValueError:
            print(f"  Invalid input. Please enter number(s) 1-{len(options)}")


def prompt_k_values() -> list:
    """Get K values from user (space-separated)."""
    print("\nSelect K values (number of features)")
    print("  Available: 4, 8, 16, 32, 50")
    print("  Angle encoding:     K features -> K qubits")
    print("  Amplitude encoding: K features -> ceil(log2(K)) qubits")

    while True:
        choice = input("Enter K values (space-separated, e.g., '4 8 16'): ").strip()
        try:
            k_values = [int(k) for k in choice.split()]
            if all(k > 0 for k in k_values):
                return k_values
            print("  K values must be positive integers")
        except ValueError:
            print("  Invalid input. Please enter numbers separated by spaces")


def interactive_mode():
    """Run experiment with interactive prompts."""
    print("\n" + "=" * 60)
    print("   QUANTUM ML EXPERIMENT - INTERACTIVE SETUP")
    print("=" * 60)

    if not VQC_AVAILABLE:
        print("\n[NOTE] VQC not available (install qiskit_machine_learning)")
        print("       Only QKSVM experiments can be run.")

    # Step 1: Feature Selection Method
    feature_methods = prompt_menu(
        "Step 1: Feature Selection Method",
        [
            ("anova", "ANOVA F-test"),
            ("snr", "SNR (Signal-to-Noise)"),
            ("both", "Both"),
        ]
    )
    if "both" in feature_methods:
        feature_methods = ["anova", "snr"]

    # Step 2: K values (multi-select)
    k_values = prompt_k_values()

    # Step 3: Encoding Type
    encodings = prompt_menu(
        "Step 3: Encoding Type",
        [
            ("angle", "Angle"),
            ("amplitude", "Amplitude"),
            ("both", "Both"),
        ]
    )
    if "both" in encodings:
        encodings = ["angle", "amplitude"]

    # Step 4: Quantum Model
    if VQC_AVAILABLE:
        models = prompt_menu(
            "Step 4: Quantum Model",
            [
                ("vqc", "VQC (Variational Quantum Classifier)"),
                ("qksvm", "QKSVM (Quantum Kernel SVM)"),
                ("both", "Both"),
            ]
        )
        if "both" in models:
            models = ["vqc", "qksvm"]
    else:
        print("\nStep 4: Quantum Model")
        print("  [Auto-selected QKSVM - VQC not available]")
        models = ["qksvm"]

    # Step 5: Kernel Method (only for QKSVM)
    kernel_methods = ["statevector"]  # default
    if "qksvm" in models:
        kernel_methods = prompt_menu(
            "Step 5: Kernel Computation Method (QKSVM only)",
            [
                ("statevector", "Statevector (exact, up to ~20 qubits)"),
                ("tensor_network", "Tensor Network (MPS, for large circuits)"),
                ("swap_test", "Swap Test (measurement-based)"),
                ("hadamard_test", "Hadamard Test (measurement-based)"),
                ("all", "All methods"),
            ]
        )
        if "all" in kernel_methods:
            kernel_methods = ["statevector", "tensor_network", "swap_test", "hadamard_test"]

    # Confirmation
    print("\n" + "=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)
    print(f"  Feature Selection: {', '.join(m.upper() for m in feature_methods)}")
    print(f"  K values:          {', '.join(map(str, k_values))}")
    print(f"  Encodings:         {', '.join(e.capitalize() for e in encodings)}")
    print(f"  Models:            {', '.join(m.upper() for m in models)}")
    if "qksvm" in models:
        print(f"  Kernel Methods:    {', '.join(kernel_methods)}")
    print("=" * 60)

    # Count total experiments
    total = 0
    for fs in feature_methods:
        for k in k_values:
            for enc in encodings:
                for model in models:
                    if model == "vqc":
                        if enc == "angle":  # VQC only supports angle
                            total += 1
                    else:  # qksvm
                        if enc == "amplitude":
                            total += 1  # amplitude doesn't use kernel methods
                        else:
                            total += len(kernel_methods)

    print(f"\nTotal experiments to run: {total}")
    confirm = input("Proceed? (y/n) [default=y]: ").strip().lower()
    if confirm not in ["", "y", "yes"]:
        print("Experiment cancelled.")
        return

    # Run experiments
    run_all_experiments(
        feature_methods=feature_methods,
        k_values=k_values,
        encodings=encodings,
        models=models,
        kernel_methods=kernel_methods,
    )


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_all_experiments(
    feature_methods: list,
    k_values: list,
    encodings: list,
    models: list,
    kernel_methods: list,
    output_dir: str = "results",
    test_size: float = 0.3,
    seed: int = 42,
    max_bond_dimension: int = 100,
):
    """Run all experiment combinations."""
    all_results = []
    experiment_num = 0

    for fs_method in feature_methods:
        for k in k_values:
            print("\n" + "#" * 70)
            print(f"# Feature Selection: {fs_method.upper()}, K={k}")
            print("#" * 70)

            # Run feature selection once per (method, k) combination
            train_csv, ind_csv = run_feature_selection(fs_method, k, output_dir)

            # Load data
            (X, y), (X_ind, y_ind) = load_data(train_csv, ind_csv)
            print(f"  Loaded {len(X)} training samples, {X.shape[1]} features")

            # Scale and split (same split for all models with this data)
            X_scaled, scaler = scale_features(X)
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=test_size, random_state=seed, stratify=y
            )

            X_ind_scaled = scaler.transform(X_ind) if X_ind is not None else None

            # Run each model/encoding/kernel combination
            for encoding in encodings:
                for model in models:
                    if model == "vqc":
                        # VQC only supports angle encoding
                        if encoding != "angle":
                            continue

                        experiment_num += 1
                        print(f"\n[Exp {experiment_num}] VQC + {encoding.capitalize()}")
                        print("-" * 50)

                        result = run_vqc(
                            X_train, y_train, X_val, y_val, X_ind_scaled, y_ind,
                            encoding=encoding,
                            n_features=k,
                            max_bond_dimension=max_bond_dimension,
                        )

                        if result:
                            result["feature_selection"] = fs_method
                            result["k"] = k
                            all_results.append(result)
                            print(f"  -> Val Acc: {result['val_acc']:.4f}")
                            if result["ind_acc"]:
                                print(f"  -> Ind Acc: {result['ind_acc']:.4f}")

                    else:  # QKSVM
                        if encoding == "amplitude":
                            # Amplitude doesn't use kernel methods
                            experiment_num += 1
                            print(f"\n[Exp {experiment_num}] QKSVM + {encoding.capitalize()}")
                            print("-" * 50)

                            result = run_qksvm(
                                X_train, y_train, X_val, y_val, X_ind_scaled, y_ind,
                                encoding=encoding,
                                n_features=k,
                                kernel_method="statevector",
                                max_bond_dimension=max_bond_dimension,
                            )

                            result["feature_selection"] = fs_method
                            result["k"] = k
                            all_results.append(result)
                            print(f"  -> Val Acc: {result['val_acc']:.4f}")
                            if result["ind_acc"]:
                                print(f"  -> Ind Acc: {result['ind_acc']:.4f}")

                        else:  # angle encoding - run each kernel method
                            for km in kernel_methods:
                                experiment_num += 1
                                print(f"\n[Exp {experiment_num}] QKSVM + {encoding.capitalize()} + {km}")
                                print("-" * 50)

                                result = run_qksvm(
                                    X_train, y_train, X_val, y_val, X_ind_scaled, y_ind,
                                    encoding=encoding,
                                    n_features=k,
                                    kernel_method=km,
                                    max_bond_dimension=max_bond_dimension,
                                )

                                result["feature_selection"] = fs_method
                                result["k"] = k
                                all_results.append(result)
                                print(f"  -> Val Acc: {result['val_acc']:.4f}")
                                if result["ind_acc"]:
                                    print(f"  -> Ind Acc: {result['ind_acc']:.4f}")

    # Print summary
    print("\n" + "=" * 130)
    print("RESULTS SUMMARY (Independent Test Set)")
    print("=" * 130)
    print(f"{'Selection':<10} {'K':<4} {'Encoding':<10} {'Kernel':<15} {'Qubits':<7} "
          f"{'Accuracy':<10} {'AUROC':<10} {'F-Score':<10} {'Recall':<10}")
    print("-" * 130)

    for r in all_results:
        def fmt(val):
            if val is None:
                return "N/A"
            return f"{val:.4f}"

        print(f"{r['feature_selection'].upper():<10} {r['k']:<4} {r['encoding']:<10} "
              f"{r['kernel_method']:<15} {r['n_qubits']:<7} "
              f"{fmt(r.get('ind_acc')):<10} {fmt(r.get('ind_auroc')):<10} "
              f"{fmt(r.get('ind_f1')):<10} {fmt(r.get('ind_recall')):<10}")

    print("=" * 130)

    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_csv = Path(output_dir) / "experiment_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to: {results_csv}")

    return all_results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point - runs interactive mode."""
    interactive_mode()


if __name__ == "__main__":
    main()
