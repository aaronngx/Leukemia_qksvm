import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import TwoLocal
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_algorithms.optimizers import COBYLA

from angle_encoding import angle_encoding_circuit
from backend_config import (
    BackendType,
    get_estimator_for_backend,
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
    output_dir: str = "results_vqc",
    backend_type: BackendType = BackendType.STATEVECTOR,
    max_bond_dimension: int = 100,
    max_iter: int = 200,
):
    """Train and evaluate VQC classifier."""
    # Print backend info
    n_features = X.shape[1]
    backend_info = get_backend_info(backend_type, n_features)
    print("\n" + "="*70)
    print("VARIATIONAL QUANTUM CLASSIFIER - CONFIGURATION")
    print("="*70)
    print(f"Backend: {backend_type.value}")
    print(f"Number of qubits: {n_features}")
    print(f"Ansatz repetitions: {reps}")
    print(f"State dimension: {backend_info['state_dimension']:,}")
    if isinstance(backend_info['memory_required_gb'], float):
        print(f"Memory required: {backend_info['memory_required_gb']:.2f} GB")
    print(f"Exact simulation: {backend_info['exact_simulation']}")
    if backend_type == BackendType.TENSOR_NETWORK:
        print(f"Max bond dimension: {max_bond_dimension}")
    print(f"Max iterations: {max_iter}")
    print("="*70 + "\n")

    X_scaled, _ = scale_to_angle(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    circuit, x_params, w_params = build_vqc(n_features, reps=reps)

    print(f"[INFO] Initializing {backend_type.value} backend...")
    estimator = get_estimator_for_backend(backend_type, max_bond_dimension)

    print("[INFO] Building quantum neural network...")
    qnn = EstimatorQNN(
        circuit=circuit,
        input_params=x_params,
        weight_params=w_params,
        estimator=estimator,
    )

    optimizer = COBYLA(maxiter=max_iter)
    clf = NeuralNetworkClassifier(qnn, optimizer=optimizer)

    print(f"[INFO] Training VQC (max {max_iter} iterations)...")
    clf.fit(X_train, y_train)

    print("[INFO] Evaluating on validation set...")
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
        print("[INFO] Evaluating on independent test set...")
        y_ind_pred = clf.predict(X_ind_scaled)
        ind_acc = accuracy_score(y_ind, y_ind_pred)
        print("\n=== VQC (independent set) ===")
        print("Accuracy:", round(ind_acc, 4))
        print(classification_report(y_ind, y_ind_pred, digits=4))

    return circuit


def main():
    parser = argparse.ArgumentParser(description="VQC with flexible backend support.")
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
        default="results_vqc",
        help="Directory to save results",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=2,
        help="TwoLocal ansatz repetitions (circuit depth)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=200,
        help="Maximum optimizer iterations",
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
        help="Maximum bond dimension for tensor network backend",
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

    train_eval_vqc(
        X_train, y_train,
        ind_data=(X_ind, y_ind),
        reps=args.reps,
        output_dir=args.output_dir,
        backend_type=backend_type,
        max_bond_dimension=args.max_bond_dimension,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()
