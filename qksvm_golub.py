from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import joblib

from qiskit import QuantumCircuit

from angle_encoding import angle_encoding_circuit, AngleEncodingType
from amplitude_encoding import amplitude_encoding_feature_map as amp_encoding_circuit
from backend_config import (
    BackendType,
    KernelMethod,
    compute_kernel_element_statevector,
    compute_kernel_element_tensor_network,
    compute_kernel_element_swap_test,
    compute_kernel_element_hadamard_test,
    get_backend_info,
)


class EncodingType:
    ANGLE = "angle"
    AMPLITUDE = "amplitude"


def scale_to_angle(X: np.ndarray):
    """Scale features to [0, π] for angle encoding."""
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def load_processed(train_csv: str, test_csv: str | None = None):
    """
    Load processed CSVs with auto-detection of label column.

    Supports both 'label' and 'cancer' column names.
    Converts labels: ALL → 0, AML → 1

    Returns
    -------
    (X_train, y_train), (X_test, y_test)
    """
    from data_loader import load_preprocessed_data

    X_train, y_train = load_preprocessed_data(train_csv)

    if test_csv is not None:
        X_test, y_test = load_preprocessed_data(test_csv)
    else:
        X_test, y_test = None, None

    return (X_train, y_train), (X_test, y_test)


def build_kernel(
    XA: np.ndarray,
    XB: np.ndarray,
    feature_map: QuantumCircuit,
    x_params,
    kernel_method: KernelMethod = KernelMethod.STATEVECTOR,
    backend_type: BackendType = BackendType.STATEVECTOR,
    max_bond_dimension: int = 100,
    verbose: bool = True,
) -> np.ndarray:
    """
    Build quantum kernel matrix K where K[i, j] = |<φ(x_i)|φ(x_j)>|^2.

    Parameters
    ----------
    XA, XB : np.ndarray
        Input matrices (already scaled to angles), shape (n_samples, n_features)
    feature_map : QuantumCircuit
        Parameterized feature map U(x)
    x_params : list
        Parameters of feature_map corresponding to classical features
    kernel_method : KernelMethod
        STATEVECTOR, SWAP_TEST, or HADAMARD_TEST
    backend_type : BackendType
        STATEVECTOR or TENSOR_NETWORK (used when kernel_method is STATEVECTOR)
    max_bond_dimension : int
        Max bond dimension for tensor-network backend
    verbose : bool
        If True, print progress

    Returns
    -------
    K : np.ndarray of shape (len(XA), len(XB))
    """
    n_qubits = feature_map.num_qubits
    nA, nB = len(XA), len(XB)
    K = np.zeros((nA, nB), dtype=float)

    total_elements = nA * nB
    if verbose:
        print(f"[INFO] Computing {nA} x {nB} = {total_elements} kernel elements...")
        print(f"[INFO] Kernel method: {kernel_method.value}")
        if kernel_method == KernelMethod.STATEVECTOR:
            print(f"[INFO] Backend: {backend_type.value}")
            if backend_type == BackendType.TENSOR_NETWORK:
                print(f"[INFO] Max bond dimension: {max_bond_dimension}")

    # Number of parameters may differ from qubits (e.g., amplitude encoding)
    n_params = len(x_params)
    
    for i, x in enumerate(XA):
        if verbose and (i % 5 == 0 or i == nA - 1):
            progress = ((i + 1) * nB) / total_elements * 100
            print(f"  Progress: {i+1}/{nA} rows ({progress:.1f}%)")

        for j, z in enumerate(XB):
            # Bind all parameters (use min to handle edge cases)
            n_to_bind = min(n_params, len(x))
            bind_x = {x_params[k]: float(x[k]) for k in range(n_to_bind)}
            bind_z = {x_params[k]: float(z[k]) for k in range(n_to_bind)}

            if kernel_method == KernelMethod.SWAP_TEST:
                # Swap test needs separate circuits for |φ(x)> and |φ(z)>
                qc_x = feature_map.assign_parameters(bind_x)
                qc_z = feature_map.assign_parameters(bind_z)
                K[i, j] = compute_kernel_element_swap_test(qc_x, qc_z)

            elif kernel_method == KernelMethod.HADAMARD_TEST:
                # Hadamard test uses U(x)U(z)†
                qc = QuantumCircuit(n_qubits)
                qc.compose(feature_map.assign_parameters(bind_x), inplace=True)
                qc.compose(feature_map.assign_parameters(bind_z).inverse(), inplace=True)
                K[i, j] = compute_kernel_element_hadamard_test(qc)

            else:  # STATEVECTOR (default)
                qc = QuantumCircuit(n_qubits)
                qc.compose(feature_map.assign_parameters(bind_x), inplace=True)
                qc.compose(feature_map.assign_parameters(bind_z).inverse(), inplace=True)

                if backend_type == BackendType.STATEVECTOR:
                    K[i, j] = compute_kernel_element_statevector(qc)
                elif backend_type == BackendType.TENSOR_NETWORK:
                    K[i, j] = compute_kernel_element_tensor_network(
                        qc, max_bond_dimension=max_bond_dimension
                    )
                else:
                    raise ValueError(f"Unknown backend type: {backend_type}")

    if verbose:
        print("[INFO] Kernel computation complete.")
    return K


def train_eval_qksvm(
    X: np.ndarray,
    y: np.ndarray,
    ind_data: tuple[np.ndarray, np.ndarray] | None = None,
    test_size: float = 0.3,
    seed: int = 42,
    output_dir: str = "results_qksvm",
    encoding_type: str = EncodingType.ANGLE,
    angle_encoding_type: AngleEncodingType = AngleEncodingType.SIMPLE_RY,
    angle_reps: int = 2,
    kernel_method: KernelMethod = KernelMethod.STATEVECTOR,
    backend_type: BackendType = BackendType.STATEVECTOR,
    max_bond_dimension: int = 100,
    # ENSGA optimization (optional)
    use_ensga: bool = False,
    ensga_pop_size: int = 30,
    ensga_generations: int = 50,
    C: float = 1.0,
):
    """
    Train and evaluate QKSVM with quantum kernel.

    Parameters
    ----------
    X, y : np.ndarray
        Features and labels for the main dataset.
    ind_data : (X_ind, y_ind) or None
        Independent test set (optional).
    test_size : float
        Validation split fraction.
    seed : int
        Random seed for train/validation split.
    output_dir : str
        Directory to save results.
    encoding_type : str
        "angle" or "amplitude" encoding.
    angle_encoding_type : AngleEncodingType
        For angle encoding: SIMPLE_RY, ZZ_FEATURE_MAP, or BPS_CIRCUIT.
    angle_reps : int
        Number of repetitions for ZZ/BPS circuits.
    kernel_method : KernelMethod
        STATEVECTOR, SWAP_TEST, or HADAMARD_TEST.
    backend_type : BackendType
        STATEVECTOR or TENSOR_NETWORK (for statevector kernel method).
    max_bond_dimension : int
        For TENSOR_NETWORK backend.
    use_ensga : bool
        If True, use ENSGA to optimize SVM parameters (C, gamma).
    ensga_pop_size : int
        ENSGA population size (if use_ensga=True).
    ensga_generations : int
        ENSGA generations (if use_ensga=True).
    C : float
        SVM regularization parameter (if use_ensga=False).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    n_features = X.shape[1]
    backend_info = get_backend_info(backend_type, n_features)

    print("\n" + "=" * 70)
    print("QUANTUM KERNEL SVM - CONFIGURATION")
    print("=" * 70)
    print(f"Encoding: {encoding_type}")
    print(f"Kernel method: {kernel_method.value}")
    if kernel_method == KernelMethod.STATEVECTOR:
        print(f"Backend: {backend_type.value}")
    print(f"Number of qubits/features: {n_features}")
    print(f"State dimension: {backend_info['state_dimension']:,}")
    if isinstance(backend_info["memory_required_gb"], float):
        print(f"Approx. statevector memory (GB): {backend_info['memory_required_gb']:.2f}")
    print(f"Exact simulation: {backend_info['exact_simulation']}")
    if backend_type == BackendType.TENSOR_NETWORK:
        print(f"Max bond dimension: {max_bond_dimension}")
    print("=" * 70 + "\n")

    # Scale to [0, π] and split
    X_scaled, _ = scale_to_angle(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # Build feature map based on encoding type
    if encoding_type == EncodingType.AMPLITUDE:
        feature_map, x_params, _ = amp_encoding_circuit(n_features)
    else:  # Angle encoding with selectable circuit type
        feature_map, x_params = angle_encoding_circuit(
            n_features,
            encoding_type=angle_encoding_type,
            reps=angle_reps,
        )
        print(f"Angle encoding type: {angle_encoding_type.value}")
        if angle_encoding_type != AngleEncodingType.SIMPLE_RY:
            print(f"Repetitions: {angle_reps}")

    # Save circuit visualization
    print("[INFO] Saving circuit diagram...")
    fig = feature_map.draw(output="mpl", style="iqp")
    plt.savefig(out_path / f"circuit_{timestamp}.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save circuit as text
    with open(out_path / f"circuit_{timestamp}.txt", "w", encoding="utf-8") as f:
        f.write(str(feature_map.draw(output="text")))

    # Training kernel
    print("[INFO] Building training kernel...")
    K_train = build_kernel(
        X_train,
        X_train,
        feature_map,
        x_params,
        kernel_method=kernel_method,
        backend_type=backend_type,
        max_bond_dimension=max_bond_dimension,
    )

    # Train SVM (with optional ENSGA optimization)
    if use_ensga:
        print("[INFO] Optimizing SVM parameters with ENSGA...")
        from ensga_optimizer import train_qsvm_ndsgoa
        
        ensga_results = train_qsvm_ndsgoa(
            K_train, y_train,
            pop_size=ensga_pop_size,
            n_generations=ensga_generations,
            verbose=True,
        )
        clf = ensga_results['model'].svm
        optimized_C = ensga_results['best_C']
        optimized_gamma = ensga_results['best_gamma']
        print(f"[INFO] ENSGA optimized: C={optimized_C:.4f}, gamma={optimized_gamma:.4f}")
    else:
        clf = SVC(kernel="precomputed", C=C)
        clf.fit(K_train, y_train)
        optimized_C = C
        optimized_gamma = None

    # Save trained model
    print("[INFO] Saving trained model...")
    model_data = {
        "classifier": clf,
        "X_train": X_train,
        "feature_map": feature_map,
        "x_params": x_params,
        "backend_type": backend_type.value,
        "max_bond_dimension": max_bond_dimension,
        "n_features": n_features,
        "optimized_C": optimized_C,
        "optimized_gamma": optimized_gamma,
        "use_ensga": use_ensga,
    }
    joblib.dump(model_data, out_path / f"qksvm_model_{timestamp}.pkl")

    # Validation kernel
    print("[INFO] Building validation kernel...")
    K_val = build_kernel(
        X_val,
        X_train,
        feature_map,
        x_params,
        kernel_method=kernel_method,
        backend_type=backend_type,
        max_bond_dimension=max_bond_dimension,
    )

    y_val_pred = clf.predict(K_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_report = classification_report(y_val, y_val_pred, digits=4)

    print("\n=== QKSVM (validation) ===")
    print("Accuracy:", round(val_acc, 4))
    print(val_report)

    # Save validation predictions
    val_df = pd.DataFrame(
        {"true_label": y_val, "predicted_label": y_val_pred}
    )
    val_df.to_csv(out_path / f"validation_predictions_{timestamp}.csv", index=False)

    # Prepare results text
    results_text: list[str] = []
    results_text.append("=" * 60)
    results_text.append("QUANTUM KERNEL SVM (QKSVM) RESULTS")
    results_text.append("=" * 60)
    results_text.append(f"\nTimestamp: {timestamp}")
    results_text.append(f"Kernel method: {kernel_method.value}")
    if kernel_method == KernelMethod.STATEVECTOR:
        results_text.append(f"Backend: {backend_type.value}")
    results_text.append(f"Number of qubits/features: {n_features}")
    if backend_type == BackendType.TENSOR_NETWORK:
        results_text.append(f"Max bond dimension: {max_bond_dimension}")
        results_text.append("Simulation type: Approximate (Tensor Network)")
    else:
        results_text.append("Simulation type: Exact (Statevector)")
    results_text.append(f"Training samples: {len(X_train)}")
    results_text.append(f"Validation samples: {len(X_val)}")
    results_text.append(f"\nOptimizer: {'ENSGA' if use_ensga else 'Default'}")
    results_text.append(f"SVM C parameter: {optimized_C:.4f}")
    if optimized_gamma is not None:
        results_text.append(f"Kernel gamma: {optimized_gamma:.4f}")
    results_text.append("\n" + "=" * 60)
    results_text.append("VALIDATION SET RESULTS")
    results_text.append("=" * 60)
    results_text.append(f"Accuracy: {round(val_acc, 4)}")
    results_text.append("\n" + val_report)

    # Independent test set (optional)
    if ind_data is not None and ind_data[0] is not None:
        X_ind, y_ind = ind_data
        X_ind_scaled, _ = scale_to_angle(X_ind)
        print("[INFO] Building independent-set kernel...")
        K_ind = build_kernel(
            X_ind_scaled,
            X_train,
            feature_map,
            x_params,
            kernel_method=kernel_method,
            backend_type=backend_type,
            max_bond_dimension=max_bond_dimension,
        )
        y_ind_pred = clf.predict(K_ind)
        ind_acc = accuracy_score(y_ind, y_ind_pred)
        ind_report = classification_report(y_ind, y_ind_pred, digits=4)

        print("\n=== QKSVM (independent set) ===")
        print("Accuracy:", round(ind_acc, 4))
        print(ind_report)

        # Save independent predictions
        ind_df = pd.DataFrame(
            {"true_label": y_ind, "predicted_label": y_ind_pred}
        )
        ind_df.to_csv(
            out_path / f"independent_predictions_{timestamp}.csv",
            index=False,
        )

        results_text.append("\n" + "=" * 60)
        results_text.append("INDEPENDENT TEST SET RESULTS")
        results_text.append("=" * 60)
        results_text.append(f"Test samples: {len(X_ind)}")
        results_text.append(f"Accuracy: {round(ind_acc, 4)}")
        results_text.append("\n" + ind_report)

    # Save results summary
    results_file = out_path / f"qksvm_results_{timestamp}.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("\n".join(results_text))

    print(f"\n[INFO] All results saved to: {out_path.absolute()}")
    print(f"  - Circuit diagram: circuit_{timestamp}.png")
    print(f"  - Circuit text:    circuit_{timestamp}.txt")
    print(f"  - Trained model:   qksvm_model_{timestamp}.pkl")
    print(f"  - Results summary: qksvm_results_{timestamp}.txt")
    print(f"  - Val predictions: validation_predictions_{timestamp}.csv")
    if ind_data is not None and ind_data[0] is not None:
        print(
            f"  - Independent predictions: "
            f"independent_predictions_{timestamp}.csv"
        )


def run_qksvm_from_csv(
    train_csv: str = "data/processed/train_topk_snr.csv",
    ind_csv: str | None = "data/processed/independent_topk_snr.csv",
    output_dir: str = "results_qksvm",
    encoding_type: str = EncodingType.ANGLE,
    angle_encoding_type: AngleEncodingType = AngleEncodingType.SIMPLE_RY,
    angle_reps: int = 2,
    kernel_method: KernelMethod = KernelMethod.STATEVECTOR,
    backend_type: BackendType = BackendType.STATEVECTOR,
    max_bond_dimension: int = 100,
    test_size: float = 0.3,
    seed: int = 42,
    # ENSGA parameters (optional)
    use_ensga: bool = False,
    ensga_pop_size: int = 30,
    ensga_generations: int = 50,
    C: float = 1.0,
):
    """
    Convenience wrapper: load CSVs and run QKSVM end-to-end.

    Usage (in a notebook):
        from qksvm_golub import run_qksvm_from_csv, BackendType, KernelMethod, EncodingType, AngleEncodingType
        
        # Simple RY (default)
        run_qksvm_from_csv(train_csv="results/train_top_16_anova_f.csv")
        
        # ZZ Feature Map with 2 reps
        run_qksvm_from_csv(
            train_csv="results/train_top_16_anova_f.csv",
            encoding_type=EncodingType.ANGLE,
            angle_encoding_type=AngleEncodingType.ZZ_FEATURE_MAP,
            angle_reps=2,
        )
        
        # With ENSGA optimization
        run_qksvm_from_csv(
            train_csv="results/train_top_16_anova_f.csv",
            use_ensga=True,
            ensga_pop_size=30,
            ensga_generations=50,
        )
    """
    (X_train, y_train), (X_ind, y_ind) = load_processed(train_csv, ind_csv)
    ind_data = (X_ind, y_ind) if X_ind is not None else None

    return train_eval_qksvm(
        X_train,
        y_train,
        ind_data=ind_data,
        test_size=test_size,
        seed=seed,
        output_dir=output_dir,
        encoding_type=encoding_type,
        angle_encoding_type=angle_encoding_type,
        angle_reps=angle_reps,
        kernel_method=kernel_method,
        backend_type=backend_type,
        max_bond_dimension=max_bond_dimension,
        use_ensga=use_ensga,
        ensga_pop_size=ensga_pop_size,
        ensga_generations=ensga_generations,
        C=C,
    )


def main():
    """CLI entry point for QKSVM."""
    import argparse

    parser = argparse.ArgumentParser(description="Quantum Kernel SVM for Golub dataset")
    parser.add_argument(
        "--train_csv",
        default="results/train_top_16_anova_f.csv",
        help="Path to training CSV",
    )
    parser.add_argument(
        "--ind_csv",
        default="results/independent_top_16_anova_f.csv",
        help="Path to independent test CSV",
    )
    parser.add_argument(
        "--output_dir",
        default="results_qksvm",
        help="Output directory",
    )
    parser.add_argument(
        "--encoding",
        choices=["angle", "amplitude"],
        default="angle",
        help="Feature encoding method",
    )
    parser.add_argument(
        "--kernel_method",
        choices=["statevector", "swap_test", "hadamard_test"],
        default="statevector",
        help="Kernel computation method",
    )
    parser.add_argument(
        "--backend",
        choices=["statevector", "tensor_network"],
        default="statevector",
        help="Backend for statevector kernel method",
    )
    parser.add_argument(
        "--max_bond_dimension",
        type=int,
        default=100,
        help="Max bond dimension for tensor network",
    )
    args = parser.parse_args()

    encoding_map = {
        "angle": EncodingType.ANGLE,
        "amplitude": EncodingType.AMPLITUDE,
    }
    kernel_method_map = {
        "statevector": KernelMethod.STATEVECTOR,
        "swap_test": KernelMethod.SWAP_TEST,
        "hadamard_test": KernelMethod.HADAMARD_TEST,
    }
    backend_map = {
        "statevector": BackendType.STATEVECTOR,
        "tensor_network": BackendType.TENSOR_NETWORK,
    }

    run_qksvm_from_csv(
        train_csv=args.train_csv,
        ind_csv=args.ind_csv,
        output_dir=args.output_dir,
        encoding_type=encoding_map[args.encoding],
        kernel_method=kernel_method_map[args.kernel_method],
        backend_type=backend_map[args.backend],
        max_bond_dimension=args.max_bond_dimension,
    )


if __name__ == "__main__":
    main()
