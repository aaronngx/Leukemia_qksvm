import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from qiskit import QuantumCircuit
from qiskit_aer import Aer  # requires qiskit-aer installed

from encodings.angle_encoding import angle_encoding_circuit


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
    backend,
):
    """
    Build kernel matrix K where K[i, j] = |<φ(x_i)|φ(x_j)>|^2
    using angle encoding and U(x) U(z)† on a statevector simulator.
    """
    n_qubits = feature_map.num_qubits
    K = np.zeros((len(XA), len(XB)))

    # Pre-bind U(z) once per column to avoid repeating work too much
    for i, x in enumerate(XA):
        for j, z in enumerate(XB):
            qc = QuantumCircuit(n_qubits)

            # U(x)
            bind_x = {x_params[k]: float(x[k]) for k in range(n_qubits)}
            qc.compose(feature_map.bind_parameters(bind_x), inplace=True)

            # U(z)†
            bind_z = {x_params[k]: float(z[k]) for k in range(n_qubits)}
            qc.compose(feature_map.bind_parameters(bind_z).inverse(), inplace=True)

            qc.save_statevector()
            result = backend.run(qc).result()
            sv = result.get_statevector()
            K[i, j] = float(np.abs(sv[0]) ** 2)  # prob(|0...0>)

    return K


def train_eval_qksvm(
    X: np.ndarray,
    y: np.ndarray,
    ind_data: tuple[np.ndarray, np.ndarray] | None = None,
    test_size: float = 0.3,
    seed: int = 42,
):
    """Train and evaluate QKSVM with angle-encoding quantum kernel."""
    X_scaled, _ = scale_to_angle(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    n_features = X_train.shape[1]

    feature_map, x_params = angle_encoding_circuit(n_features)
    backend = Aer.get_backend("statevector_simulator")

    print("[INFO] Building training kernel...")
    K_train = build_kernel(X_train, X_train, feature_map, x_params, backend)

    clf = SVC(kernel="precomputed", C=1.0)
    clf.fit(K_train, y_train)

    print("[INFO] Building validation kernel...")
    K_val = build_kernel(X_val, X_train, feature_map, x_params, backend)
    y_pred = clf.predict(K_val)

    acc = accuracy_score(y_val, y_pred)
    print("\n=== QKSVM (validation) ===")
    print("Accuracy:", round(acc, 4))
    print(classification_report(y_val, y_pred, digits=4))

    # Evaluate on independent set if provided
    if ind_data is not None and ind_data[0] is not None:
        X_ind, y_ind = ind_data
        X_ind_scaled, _ = scale_to_angle(X_ind)
        print("[INFO] Building independent-set kernel...")
        K_ind = build_kernel(X_ind_scaled, X_train, feature_map, x_params, backend)
        y_ind_pred = clf.predict(K_ind)
        ind_acc = accuracy_score(y_ind, y_ind_pred)
        print("\n=== QKSVM (independent set) ===")
        print("Accuracy:", round(ind_acc, 4))
        print(classification_report(y_ind, y_ind_pred, digits=4))


def main():
    parser = argparse.ArgumentParser(description="QKSVM (angle-encoding kernel) on Golub dataset.")
    parser.add_argument(
        "--train_csv",
        default="data/processed/train_topk_snr.csv",
        help="Path to processed train CSV (SNR-selected).",
    )
    parser.add_argument(
        "--ind_csv",
        default="data/processed/independent_topk_snr.csv",
        help="Path to processed independent CSV (SNR-selected).",
    )
    args = parser.parse_args()

    (X_train, y_train), (X_ind, y_ind) = load_processed(args.train_csv, args.ind_csv)
    train_eval_qksvm(X_train, y_train, ind_data=(X_ind, y_ind))


if __name__ == "__main__":
    main()
