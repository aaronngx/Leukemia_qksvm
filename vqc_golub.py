import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_algorithms.optimizers import COBYLA

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


def build_vqc(num_features: int, reps: int = 2):
    """Build VQC: angle encoding + TwoLocal ansatz."""
    feature_map, x_params = angle_encoding_circuit(num_features)
    ansatz = TwoLocal(
        num_qubits=num_features,
        rotation_blocks="ry",
        entanglement_blocks="cz",
        entanglement="full",
        reps=reps,
    )

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(num_features, name="VQC")
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    return qc, list(x_params), list(ansatz.parameters)


def train_eval_vqc(
    X: np.ndarray,
    y: np.ndarray,
    ind_data: tuple[np.ndarray, np.ndarray] | None = None,
    reps: int = 2,
    test_size: float = 0.3,
    seed: int = 42,
):
    """Train and evaluate VQC classifier on Golub data."""
    X_scaled, _ = scale_to_angle(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    n_features = X_train.shape[1]
    circuit, x_params, w_params = build_vqc(n_features, reps=reps)

    estimator = Estimator()
    qnn = EstimatorQNN(
        circuit=circuit,
        input_params=x_params,
        weight_params=w_params,
    )

    optimizer = COBYLA(maxiter=200)
    clf = NeuralNetworkClassifier(qnn, optimizer=optimizer)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    print("\n=== VQC (validation) ===")
    print("Accuracy:", round(acc, 4))
    print(classification_report(y_val, y_pred, digits=4))

    # Optional ROC-AUC if probabilities are available
    auc = None
    if hasattr(clf, "predict_proba"):
        try:
            y_proba = clf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_proba)
            print("ROC-AUC:", round(auc, 4))
        except Exception:
            pass

    # Evaluate on independent set if provided
    if ind_data is not None and ind_data[0] is not None:
        X_ind, y_ind = ind_data
        X_ind_scaled, _ = scale_to_angle(X_ind)
        y_ind_pred = clf.predict(X_ind_scaled)
        ind_acc = accuracy_score(y_ind, y_ind_pred)
        print("\n=== VQC (independent set) ===")
        print("Accuracy:", round(ind_acc, 4))
        print(classification_report(y_ind, y_ind_pred, digits=4))

    return circuit


def main():
    parser = argparse.ArgumentParser(description="VQC classifier on Golub dataset (angle encoding).")
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
    parser.add_argument(
        "--reps",
        type=int,
        default=2,
        help="TwoLocal reps (circuit depth).",
    )
    args = parser.parse_args()

    (X_train, y_train), (X_ind, y_ind) = load_processed(args.train_csv, args.ind_csv)
    train_eval_vqc(X_train, y_train, ind_data=(X_ind, y_ind), reps=args.reps)


if __name__ == "__main__":
    main()
