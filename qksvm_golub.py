import argparse
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

from angle_encoding import angle_encoding_circuit
from backend_config import (
    BackendType,
    compute_kernel_element_statevector,
    compute_kernel_element_tensor_network,
    get_backend_info,
    print_backend_recommendation,
)


def scale_to_angle(X: np.ndarray):
    """Scale features to [0, π] for angle encoding."""
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def load_processed(train_csv: str, test_csv: str | None = None):
    """Load SNR-processed CSVs."""
    df_train = pd.read_csv(train_csv)
    y_train = df_train["label"].values
    X_train = df_train.drop(columns=["label"]).values

    if test_csv is not None:
        df_test = pd.read_csv(test_csv)
        y_test = df_test["label"].values
        X_test = df_test.drop(columns=["label"]).values
    else:
        X_test, y_test = None, None

    return (X_train, y_train), (X_test, y_test)


def build_kernel(
    XA: np.ndarray,
    XB: np.ndarray,
    feature_map: QuantumCircuit,
    x_params,
    backend_type: BackendType = BackendType.STATEVECTOR,
    max_bond_dimension: int = 100,
    verbose: bool = True,
):
    """
    Build kernel matrix K where K[i, j] = |<φ(x_i)|φ(x_j)>|^2.

    Supports multiple backends: statevector, aer_statevector, or tensor_network (MPS).
    """
    n_qubits = feature_map.num_qubits
    K = np.zeros((len(XA), len(XB)))

    total_elements = len(XA) * len(XB)
    if verbose:
        print(f"[INFO] Computing {len(XA)} x {len(XB)} = {total_elements} kernel elements...")
        print(f"[INFO] Backend: {backend_type.value}")
        if backend_type == BackendType.TENSOR_NETWORK:
            print(f"[INFO] Max bond dimension: {max_bond_dimension}")

    for i, x in enumerate(XA):
        if verbose and (i % 5 == 0 or i == len(XA) - 1):
            progress = ((i + 1) * len(XB)) / total_elements * 100
            print(f"  Progress: {i+1}/{len(XA)} rows ({progress:.1f}%)")

        for j, z in enumerate(XB):
            qc = QuantumCircuit(n_qubits)

            # U(x)
            bind_x = {x_params[k]: float(x[k]) for k in range(n_qubits)}
            qc.compose(feature_map.assign_parameters(bind_x), inplace=True)

            # U(z)†
            bind_z = {x_params[k]: float(z[k]) for k in range(n_qubits)}
            qc.compose(feature_map.assign_parameters(bind_z).inverse(), inplace=True)

            # Compute kernel element based on backend
            if backend_type == BackendType.STATEVECTOR:
                K[i, j] = compute_kernel_element_statevector(qc)
            elif backend_type == BackendType.TENSOR_NETWORK:
                K[i, j] = compute_kernel_element_tensor_network(qc, max_bond_dimension)
            else:
                raise ValueError(f"Unknown backend type: {backend_type}")

    if verbose:
        print(f"[INFO] Kernel computation complete.")

    return K


def train_eval_qksvm(
    X: np.ndarray,
    y: np.ndarray,
    ind_data: tuple[np.ndarray, np.ndarray] | None = None,
    test_size: float = 0.3,
    seed: int = 42,
    output_dir: str = "results",
    backend_type: BackendType = BackendType.STATEVECTOR,
    max_bond_dimension: int = 100,
):
    """Train and evaluate QKSVM with angle-encoding quantum kernel."""
    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Print backend info
    n_features = X.shape[1]
    backend_info = get_backend_info(backend_type, n_features)
    print("\n" + "="*70)
    print("QUANTUM KERNEL SVM - CONFIGURATION")
    print("="*70)
    print(f"Backend: {backend_type.value}")
    print(f"Number of qubits: {n_features}")
    print(f"State dimension: {backend_info['state_dimension']:,}")
    if isinstance(backend_info['memory_required_gb'], float):
        print(f"Memory required: {backend_info['memory_required_gb']:.2f} GB")
    print(f"Exact simulation: {backend_info['exact_simulation']}")
    if backend_type == BackendType.TENSOR_NETWORK:
        print(f"Max bond dimension: {max_bond_dimension}")
    print("="*70 + "\n")

    X_scaled, _ = scale_to_angle(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    feature_map, x_params = angle_encoding_circuit(n_features)

    # Save circuit visualization
    print("[INFO] Saving circuit diagram...")
    fig = feature_map.draw(output='mpl', style='iqp')
    plt.savefig(out_path / f"circuit_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Also save circuit as text
    with open(out_path / f"circuit_{timestamp}.txt", 'w', encoding='utf-8') as f:
        f.write(str(feature_map.draw(output='text')))

    print("[INFO] Building training kernel...")
    K_train = build_kernel(
        X_train, X_train, feature_map, x_params,
        backend_type=backend_type,
        max_bond_dimension=max_bond_dimension,
    )

    clf = SVC(kernel="precomputed", C=1.0)
    clf.fit(K_train, y_train)

    # Save trained model
    print("[INFO] Saving trained model...")
    model_data = {
        'classifier': clf,
        'X_train': X_train,
        'feature_map': feature_map,
        'x_params': x_params,
        'backend_type': backend_type.value,
        'max_bond_dimension': max_bond_dimension,
        'n_features': n_features,
    }
    joblib.dump(model_data, out_path / f"qksvm_model_{timestamp}.pkl")

    print("[INFO] Building validation kernel...")
    K_val = build_kernel(
        X_val, X_train, feature_map, x_params,
        backend_type=backend_type,
        max_bond_dimension=max_bond_dimension,
    )
    y_pred = clf.predict(K_val)

    acc = accuracy_score(y_val, y_pred)
    val_report = classification_report(y_val, y_pred, digits=4)

    print("\n=== QKSVM (validation) ===")
    print("Accuracy:", round(acc, 4))
    print(val_report)

    # Save validation predictions
    val_df = pd.DataFrame({
        'true_label': y_val,
        'predicted_label': y_pred
    })
    val_df.to_csv(out_path / f"validation_predictions_{timestamp}.csv", index=False)

    # Prepare results text
    results_text = []
    results_text.append("="*60)
    results_text.append("QUANTUM KERNEL SVM (QKSVM) RESULTS")
    results_text.append("="*60)
    results_text.append(f"\nTimestamp: {timestamp}")
    results_text.append(f"Backend: {backend_type.value}")
    results_text.append(f"Number of qubits/features: {n_features}")
    if backend_type == BackendType.TENSOR_NETWORK:
        results_text.append(f"Max bond dimension: {max_bond_dimension}")
        results_text.append("Simulation type: Approximate (Tensor Network)")
    else:
        results_text.append("Simulation type: Exact (Statevector)")
    results_text.append(f"Training samples: {len(X_train)}")
    results_text.append(f"Validation samples: {len(X_val)}")
    results_text.append(f"\n{'='*60}")
    results_text.append("VALIDATION SET RESULTS")
    results_text.append('='*60)
    results_text.append(f"Accuracy: {round(acc, 4)}")
    results_text.append(f"\n{val_report}")

    # Evaluate on independent set if provided
    if ind_data is not None and ind_data[0] is not None:
        X_ind, y_ind = ind_data
        X_ind_scaled, _ = scale_to_angle(X_ind)
        print("[INFO] Building independent-set kernel...")
        K_ind = build_kernel(
            X_ind_scaled, X_train, feature_map, x_params,
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
        ind_df = pd.DataFrame({
            'true_label': y_ind,
            'predicted_label': y_ind_pred
        })
        ind_df.to_csv(out_path / f"independent_predictions_{timestamp}.csv", index=False)

        results_text.append(f"\n{'='*60}")
        results_text.append("INDEPENDENT TEST SET RESULTS")
        results_text.append('='*60)
        results_text.append(f"Test samples: {len(X_ind)}")
        results_text.append(f"Accuracy: {round(ind_acc, 4)}")
        results_text.append(f"\n{ind_report}")

    # Save all results to text file
    results_file = out_path / f"qksvm_results_{timestamp}.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results_text))

    print(f"\n[INFO] All results saved to: {out_path.absolute()}")
    print(f"  - Circuit diagram: circuit_{timestamp}.png")
    print(f"  - Circuit text: circuit_{timestamp}.txt")
    print(f"  - Trained model: qksvm_model_{timestamp}.pkl")
    print(f"  - Results summary: qksvm_results_{timestamp}.txt")
    print(f"  - Predictions: validation_predictions_{timestamp}.csv")
    if ind_data is not None and ind_data[0] is not None:
        print(f"  - Independent predictions: independent_predictions_{timestamp}.csv")


def main():
    parser = argparse.ArgumentParser(description="QKSVM with flexible backend support.")
    parser.add_argument(
        "--train_csv",
        default="data/processed/train_topk_snr.csv",
        help="Path to processed train CSV",
    )
    parser.add_argument(
        "--ind_csv",
        default="data/processed/independent_topk_snr.csv",
        help="Path to processed independent CSV",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--backend",
        choices=['statevector', 'tensor_network'],
        default='statevector',
        help="Backend type for simulation",
    )
    parser.add_argument(
        "--max_bond_dimension",
        type=int,
        default=100,
        help="Maximum bond dimension for tensor network backend (typical: 50-500)",
    )
    parser.add_argument(
        "--recommend_backend",
        action="store_true",
        help="Print backend recommendation and exit",
    )
    parser.add_argument(
        "--available_memory",
        type=float,
        default=16.0,
        help="Available RAM in GB (for recommendation)",
    )
    args = parser.parse_args()

    # Load data
    (X_train, y_train), (X_ind, y_ind) = load_processed(args.train_csv, args.ind_csv)

    if args.recommend_backend:
        print_backend_recommendation(X_train.shape[1], args.available_memory)
        return

    # Map string to BackendType
    backend_map = {
        'statevector': BackendType.STATEVECTOR,
        'tensor_network': BackendType.TENSOR_NETWORK,
    }
    backend_type = backend_map[args.backend]

    train_eval_qksvm(
        X_train, y_train,
        ind_data=(X_ind, y_ind),
        output_dir=args.output_dir,
        backend_type=backend_type,
        max_bond_dimension=args.max_bond_dimension,
    )


if __name__ == "__main__":
    main()
