import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import TwoLocal
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM

from angle_encoding import angle_encoding_circuit
from backend_config import (
    BackendType,
    get_backend_info,
    get_estimator_for_backend,
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


def build_vqc(num_features: int, reps: int = 2):
    """
    Build VQC circuit: angle encoding + TwoLocal ansatz.

    Returns
    -------
    qc        : QuantumCircuit for U(x, θ)
    x_params  : list of feature parameters
    w_params  : list of trainable parameters
    """
    feature_map, x_params = angle_encoding_circuit(num_features)
    ansatz = TwoLocal(
        num_qubits=num_features,
        rotation_blocks=["rx", "rz", "rx"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=reps,
    )

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(num_features, name="VQC")
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    return qc, list(x_params), list(ansatz.parameters)


def make_optimizer(name: str, max_iter: int = 200):
    """
    Create an optimizer by name.

    Supported:
      - 'COBYLA'
      - 'SPSA'
      - 'ADAM'
    """
    name = name.lower()
    if name == "cobyla":
        return COBYLA(maxiter=max_iter)
    if name == "spsa":
        # SPSA is stochastic; this is a reasonable default
        return SPSA(maxiter=max_iter)
    if name == "adam":
        # ADAM supports many hyperparams; keep defaults except maxiter
        return ADAM(maxiter=max_iter)
    raise ValueError(f"Unknown optimizer name: {name}. Use 'COBYLA', 'SPSA', or 'ADAM'.")


def train_eval_vqc(
    X: np.ndarray,
    y: np.ndarray,
    ind_data: tuple[np.ndarray, np.ndarray] | None = None,
    reps: int = 2,
    test_size: float = 0.3,
    seed: int = 42,
    output_dir: str = "results_vqc",  # placeholder for future saving
    backend_type: BackendType = BackendType.STATEVECTOR,
    max_bond_dimension: int = 100,
    max_iter: int = 200,
    optimizer_name: str = "COBYLA",
):
    """
    Train and evaluate a Variational Quantum Classifier (VQC).

    Parameters
    ----------
    X, y : np.ndarray
        Features and labels.
    ind_data : (X_ind, y_ind) or None
        Independent test set (optional).
    reps : int
        Number of TwoLocal repetitions.
    test_size : float
        Validation split fraction.
    seed : int
        Random seed for train/validation split.
    output_dir : str
        Currently unused (placeholder for future saving).
    backend_type : BackendType
        STATEVECTOR or TENSOR_NETWORK.
    max_bond_dimension : int
        For TENSOR_NETWORK backend.
    max_iter : int
        Max optimizer iterations.
    optimizer_name : str
        'COBYLA', 'SPSA', or 'ADAM'.
    """
    n_features = X.shape[1]
    backend_info = get_backend_info(backend_type, n_features)

    print("\n" + "=" * 70)
    print("VARIATIONAL QUANTUM CLASSIFIER - CONFIGURATION")
    print("=" * 70)
    print(f"Backend: {backend_type.value}")
    print(f"Number of qubits / features: {n_features}")
    print(f"Ansatz repetitions: {reps}")
    print(f"Optimizer: {optimizer_name.upper()}")
    print(f"Max iterations: {max_iter}")
    print(f"State dimension: {backend_info['state_dimension']:,}")
    if isinstance(backend_info["memory_required_gb"], float):
        print(f"Approx. statevector memory (GB): {backend_info['memory_required_gb']:.2f}")
    print(f"Exact simulation: {backend_info['exact_simulation']}")
    if backend_type == BackendType.TENSOR_NETWORK:
        print(f"Max bond dimension: {max_bond_dimension}")
    print("=" * 70 + "\n")

    # Scale and split
    X_scaled, _ = scale_to_angle(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # Build VQC circuit
    circuit, x_params, w_params = build_vqc(n_features, reps=reps)

    # Backend-specific Estimator
    print(f"[INFO] Initializing {backend_type.value} backend...")
    estimator = get_estimator_for_backend(backend_type, max_bond_dimension)

    print("[INFO] Building quantum neural network...")
    qnn = EstimatorQNN(
        circuit=circuit,
        input_params=x_params,
        weight_params=w_params,
        estimator=estimator,
    )

    optimizer = make_optimizer(optimizer_name, max_iter=max_iter)
    clf = NeuralNetworkClassifier(qnn, optimizer=optimizer)

    print(f"[INFO] Training VQC with {optimizer_name.upper()} (max {max_iter} iterations)...")
    clf.fit(X_train, y_train)

    print("[INFO] Evaluating on validation set...")
    y_val_pred = clf.predict(X_val)

    val_acc = accuracy_score(y_val, y_val_pred)
    print("\n=== VQC (validation) ===")
    print("Accuracy:", round(val_acc, 4))
    print(classification_report(y_val, y_val_pred, digits=4))

    # Optional ROC-AUC
    if hasattr(clf, "predict_proba"):
        try:
            y_proba = clf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_proba)
            print("ROC-AUC:", round(auc, 4))
        except Exception:
            pass

    # Independent test set (optional)
    if ind_data is not None and ind_data[0] is not None:
        X_ind, y_ind = ind_data
        X_ind_scaled, _ = scale_to_angle(X_ind)
        print("[INFO] Evaluating on independent test set...")
        y_ind_pred = clf.predict(X_ind_scaled)
        ind_acc = accuracy_score(y_ind, y_ind_pred)
        print("\n=== VQC (independent set) ===")
        print("Accuracy:", round(ind_acc, 4))
        print(classification_report(y_ind, y_ind_pred, digits=4))

    return circuit, clf


def run_vqc_from_csv(
    train_csv: str = "data/processed/train_topk_snr.csv",
    ind_csv: str | None = "data/processed/independent_topk_snr.csv",
    backend_type: BackendType = BackendType.STATEVECTOR,
    max_bond_dimension: int = 100,
    reps: int = 2,
    max_iter: int = 200,
    optimizer_name: str = "COBYLA",
    test_size: float = 0.3,
    seed: int = 42,
):
    """
    Convenience wrapper: load CSVs and run VQC end-to-end.

    Example (in notebook):

        from vqc_golub import run_vqc_from_csv, BackendType

        for opt in ["COBYLA", "SPSA", "ADAM"]:
            print("=== Optimizer:", opt, "===")
            circuit, clf = run_vqc_from_csv(
                train_csv="data/processed/train_topk_snr.csv",
                ind_csv="data/processed/independent_topk_snr.csv",
                backend_type=BackendType.TENSOR_NETWORK,
                max_bond_dimension=100,
                reps=2,
                max_iter=200,
                optimizer_name=opt,
            )
    """
    (X_train, y_train), (X_ind, y_ind) = load_processed(train_csv, ind_csv)
    ind_data = (X_ind, y_ind) if X_ind is not None else None

    return train_eval_vqc(
        X_train,
        y_train,
        ind_data=ind_data,
        reps=reps,
        test_size=test_size,
        seed=seed,
        output_dir="results_vqc",
        backend_type=backend_type,
        max_bond_dimension=max_bond_dimension,
        max_iter=max_iter,
        optimizer_name=optimizer_name,
    )
