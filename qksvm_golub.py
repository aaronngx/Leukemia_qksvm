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
)


def scale_to_angle(X: np.ndarray):
    """Scale features to [0, π] for angle encoding."""
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def load_processed(train_csv: str, test_csv: str | None = None):
    """
    Load processed CSVs in (features + 'label') format.

    Returns
    -------
    (X_train, y_train), (X_test, y_test)
    """
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
    backend_type : BackendType
        STATEVECTOR or TENSOR_NETWORK
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
        print(f"[INFO] Backend: {backend_type.value}")
        if backend_type == BackendType.TENSOR_NETWORK:
            print(f"[INFO] Max bond dimension: {max_bond_dimension}")

    for i, x in enumerate(XA):
        if verbose and (i % 5 == 0 or i == nA - 1):
            progress = ((i + 1) * nB) / total_elements * 100
            print(f"  Progress: {i+1}/{nA} rows ({progress:.1f}%)")

        for j, z in enumerate(XB):
            qc = QuantumCircuit(n_qubits)

            # U(x)
            bind_x = {x_params[k]: float(x[k]) for k in range(n_qubits)}
            qc.compose(feature_map.assign_parameters(bind_x), inplace=True)

            # U(z)†
            bind_z = {x_params[k]: float(z[k]) for k in range(n_qubits)}
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
    backend_type: BackendType = BackendType.STATEVECTOR,
    max_bond_dimension: int = 100,
):
    """
    Train and evaluate QKSVM with angle-encoding quantum kernel.

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
    backend_type : BackendType
        STATEVECTOR or TENSOR_NETWORK.
    max_bond_dimension : int
        For TENSOR_NETWORK backend.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    n_features = X.shape[1]
    backend_info = get_backend_info(backend_type, n_features)

    print("\n" + "=" * 70)
    print("QUANTUM KERNEL SVM - CONFIGURATION")
    print("=" * 70)
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

    # Build angle-encoding feature map
    feature_map, x_params = angle_encoding_circuit(n_features)

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
        backend_type=backend_type,
        max_bond_dimension=max_bond_dimension,
    )

    clf = SVC(kernel="precomputed", C=1.0)
    clf.fit(K_train, y_train)

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
    }
    joblib.dump(model_data, out_path / f"qksvm_model_{timestamp}.pkl")

    # Validation kernel
    print("[INFO] Building validation kernel...")
    K_val = build_kernel(
        X_val,
        X_train,
        feature_map,
        x_params,
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
    results_text.append(f"Backend: {backend_type.value}")
    results_text.append(f"Number of qubits/features: {n_features}")
    if backend_type == BackendType.TENSOR_NETWORK:
        results_text.append(f"Max bond dimension: {max_bond_dimension}")
        results_text.append("Simulation type: Approximate (Tensor Network)")
    else:
        results_text.append("Simulation type: Exact (Statevector)")
    results_text.append(f"Training samples: {len(X_train)}")
    results_text.append(f"Validation samples: {len(X_val)}")
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
    backend_type: BackendType = BackendType.STATEVECTOR,
    max_bond_dimension: int = 100,
    test_size: float = 0.3,
    seed: int = 42,
):
    """
    Convenience wrapper: load CSVs and run QKSVM end-to-end.

    Usage (in a notebook):
        from qksvm_golub import run_qksvm_from_csv, BackendType
        run_qksvm_from_csv(
            train_csv="data/processed/train_topk_snr.csv",
            ind_csv="data/processed/independent_topk_snr.csv",
            backend_type=BackendType.TENSOR_NETWORK,
            max_bond_dimension=100,
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
        backend_type=backend_type,
        max_bond_dimension=max_bond_dimension,
    )
