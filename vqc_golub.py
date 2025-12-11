"""
VQC Training Pipeline for Golub Leukemia Dataset.

Supports two encoding types:
1. ANGLE ENCODING (via train_eval_vqc)
   - Features as rotation angles: RY(xᵢ)
   - Linear qubit scaling: n = features
   - Uses Qiskit ML NeuralNetworkClassifier

2. AMPLITUDE ENCODING (via train_eval_amplitude_vqc)
   - Features as state amplitudes: |ψ⟩ = Σxᵢ|i⟩
   - Logarithmic qubit scaling: n = log₂(features)
   - Uses custom training loop (initialize() not compatible with QNN)

Usage:
    # Angle encoding
    from vqc_golub import train_eval_vqc
    circuit, clf = train_eval_vqc(X, y, reps=2)
    
    # Amplitude encoding
    from vqc_golub import train_eval_amplitude_vqc
    clf = train_eval_amplitude_vqc(X, y, reps=1)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM

from angle_encoding import angle_encoding_circuit
from amplitude_encoding import (
    encode_amplitude,
    preprocess_for_amplitude_encoding,
    get_num_qubits,
)
from ansatz import create_amplitude_ansatz, create_angle_ansatz
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


# =============================================================================
# AMPLITUDE ENCODING VQC
# =============================================================================

def build_amplitude_vqc_circuit(x: np.ndarray, theta: np.ndarray, n_qubits: int, reps: int = 1):
    """
    Build complete amplitude VQC circuit: U(x) + V(θ).
    
    Architecture:
        |0⟩⊗ⁿ → [Amplitude Encoding U(x)] → [Ansatz V(θ)]
    
    Parameters
    ----------
    x : np.ndarray
        Normalized feature vector (length = 2^n_qubits)
    theta : np.ndarray
        Trainable parameters for ansatz
    n_qubits : int
        Number of qubits
    reps : int
        Number of ansatz layers
    
    Returns
    -------
    qc : QuantumCircuit
        Complete VQC circuit with bound parameters
    """
    # 1. Feature Map U(x): Amplitude Encoding (Mottonen Decomposition)
    qc = QuantumCircuit(n_qubits, name="AmplitudeVQC")
    qc.compose(encode_amplitude(x), inplace=True)
    qc.barrier(label="U(x)|V(θ)")
    
    # 2. Ansatz V(θ): RX-RZ-RX + linear CNOT
    ansatz, theta_params = create_amplitude_ansatz(n_qubits, reps)
    
    # Bind parameters
    param_dict = {theta_params[i]: theta[i] for i in range(len(theta_params))}
    bound_ansatz = ansatz.assign_parameters(param_dict)
    
    qc.compose(bound_ansatz, inplace=True)
    
    return qc


def compute_expectation_z0(circuit: QuantumCircuit) -> float:
    """
    Compute expectation value ⟨Z₀⟩ for measurement.
    
    The observable is Pauli-Z on qubit 0.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        Complete VQC circuit
    
    Returns
    -------
    float
        ⟨Z₀⟩ ∈ [-1, 1]
    """
    sv = Statevector.from_instruction(circuit)
    
    # Create Z₀ observable (Z on qubit 0, I on others)
    n_qubits = circuit.num_qubits
    z_string = 'I' * (n_qubits - 1) + 'Z'  # Z on qubit 0
    observable = SparsePauliOp.from_list([(z_string, 1.0)])
    
    return sv.expectation_value(observable).real


def amplitude_vqc_forward(x: np.ndarray, theta: np.ndarray, n_qubits: int, reps: int) -> float:
    """
    Forward pass through amplitude VQC.
    
    Returns probability p₁ = (1 + ⟨Z₀⟩) / 2 = P(AML).
    """
    qc = build_amplitude_vqc_circuit(x, theta, n_qubits, reps)
    expectation = compute_expectation_z0(qc)
    # Convert to probability: p₁ = (1 + ⟨Z₀⟩) / 2
    return (1 + expectation) / 2


def amplitude_vqc_gradient(x: np.ndarray, y: int, theta: np.ndarray, 
                           n_qubits: int, reps: int) -> np.ndarray:
    """
    Compute gradient using parameter-shift rule.
    
    ∂f/∂θᵢ = [f(θ + π/2·eᵢ) - f(θ - π/2·eᵢ)] / 2
    """
    n_params = len(theta)
    gradients = np.zeros(n_params)
    
    for i in range(n_params):
        # θ + π/2
        theta_plus = theta.copy()
        theta_plus[i] += np.pi / 2
        f_plus = amplitude_vqc_forward(x, theta_plus, n_qubits, reps)
        
        # θ - π/2
        theta_minus = theta.copy()
        theta_minus[i] -= np.pi / 2
        f_minus = amplitude_vqc_forward(x, theta_minus, n_qubits, reps)
        
        # Parameter-shift gradient
        gradients[i] = (f_plus - f_minus) / 2
    
    # Scale by loss gradient (binary cross-entropy)
    p = amplitude_vqc_forward(x, theta, n_qubits, reps)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    
    if y == 1:
        loss_grad = -1 / p
    else:
        loss_grad = 1 / (1 - p)
    
    return gradients * loss_grad


def train_eval_amplitude_vqc(
    X: np.ndarray,
    y: np.ndarray,
    ind_data: tuple[np.ndarray, np.ndarray] | None = None,
    reps: int = 1,
    test_size: float = 0.3,
    seed: int = 42,
    epochs: int = 50,
    learning_rate: float = 0.1,
    verbose: bool = True,
):
    """
    Train and evaluate Amplitude-Encoded VQC.
    
    Uses custom training loop because amplitude encoding (via initialize())
    is not compatible with Qiskit ML's NeuralNetworkClassifier.
    
    Architecture:
    1. U(x): Amplitude encoding (Mottonen Decomposition)
    2. V(θ): RX-RZ-RX ansatz with linear CNOT
    3. Measurement: ⟨Z₀⟩ → p₁ = P(AML)
    
    Parameters
    ----------
    X, y : np.ndarray
        Features and labels
    ind_data : tuple, optional
        Independent test set (X_ind, y_ind)
    reps : int
        Ansatz layers (default: 1 for 2 layers, 24 params)
    test_size : float
        Validation split
    seed : int
        Random seed
    epochs : int
        Training epochs
    learning_rate : float
        SGD learning rate
    verbose : bool
        Print progress
    
    Returns
    -------
    results : dict
        Training results including accuracy and trained parameters
    """
    n_features = X.shape[1]
    n_qubits = get_num_qubits(n_features)
    state_dim = 2 ** n_qubits
    
    # Get parameter count
    ansatz, theta_params = create_amplitude_ansatz(n_qubits, reps)
    n_params = len(theta_params)
    
    if verbose:
        print("\n" + "=" * 70)
        print("AMPLITUDE-ENCODED VQC - CONFIGURATION")
        print("=" * 70)
        print(f"\n1. FEATURE MAP U(x): Amplitude Encoding")
        print(f"   • Features: {n_features}")
        print(f"   • Qubits: {n_qubits} (log₂({n_features}) = {n_qubits})")
        print(f"   • State dimension: {state_dim}")
        print(f"   • Method: Mottonen Decomposition")
        print(f"\n2. ANSATZ V(θ): Variational Layers")
        print(f"   • Rotation blocks: RX-RZ-RX (SU(2))")
        print(f"   • Entanglement: Linear CNOT")
        print(f"   • Layers: {reps + 1}")
        print(f"   • Trainable parameters: {n_params}")
        print(f"\n3. MEASUREMENT: ⟨Z₀⟩ → p₁ = (1+⟨Z₀⟩)/2")
        print(f"\n4. TRAINING")
        print(f"   • Epochs: {epochs}")
        print(f"   • Learning rate: {learning_rate}")
        print(f"   • Gradient method: Parameter-shift rule")
        print("=" * 70 + "\n")
    
    # Preprocess for amplitude encoding
    X_prepared, scaler, _ = preprocess_for_amplitude_encoding(X)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_prepared, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    
    if verbose:
        print(f"[INFO] Training samples: {len(X_train)}")
        print(f"[INFO] Validation samples: {len(X_val)}")
    
    # Initialize parameters
    np.random.seed(seed)
    theta = np.random.uniform(-np.pi, np.pi, n_params)
    
    # Training loop
    if verbose:
        print(f"\n[INFO] Training with SGD (parameter-shift gradients)...")
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        
        for idx in indices:
            x = X_train[idx]
            label = y_train[idx]
            
            # Forward pass
            p = amplitude_vqc_forward(x, theta, n_qubits, reps)
            
            # Loss (binary cross-entropy)
            p_clipped = np.clip(p, 1e-10, 1 - 1e-10)
            loss = -label * np.log(p_clipped) - (1 - label) * np.log(1 - p_clipped)
            total_loss += loss
            
            # Prediction
            pred = 1 if p >= 0.5 else 0
            if pred == label:
                correct += 1
            
            # Gradient descent
            grad = amplitude_vqc_gradient(x, label, theta, n_qubits, reps)
            theta -= learning_rate * grad
        
        avg_loss = total_loss / len(X_train)
        train_acc = correct / len(X_train)
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}, Train Acc = {train_acc:.4f}")
    
    # Validation
    if verbose:
        print(f"\n[INFO] Evaluating on validation set...")
    
    y_val_pred = []
    y_val_proba = []
    for x in X_val:
        p = amplitude_vqc_forward(x, theta, n_qubits, reps)
        y_val_proba.append(p)
        y_val_pred.append(1 if p >= 0.5 else 0)
    
    y_val_pred = np.array(y_val_pred)
    y_val_proba = np.array(y_val_proba)
    
    val_acc = accuracy_score(y_val, y_val_pred)
    
    print("\n=== Amplitude VQC (validation) ===")
    print(f"Accuracy: {val_acc:.4f}")
    print(classification_report(y_val, y_val_pred, digits=4))
    
    try:
        val_auc = roc_auc_score(y_val, y_val_proba)
        print(f"ROC-AUC: {val_auc:.4f}")
    except:
        val_auc = None
    
    results = {
        "theta": theta,
        "n_qubits": n_qubits,
        "n_params": n_params,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "scaler": scaler,
        "reps": reps,
    }
    
    # Independent test set
    if ind_data is not None and ind_data[0] is not None:
        X_ind, y_ind = ind_data
        X_ind_prepared, _, _ = preprocess_for_amplitude_encoding(X_ind, scaler)
        
        if verbose:
            print(f"\n[INFO] Evaluating on independent test set...")
        
        y_ind_pred = []
        y_ind_proba = []
        for x in X_ind_prepared:
            p = amplitude_vqc_forward(x, theta, n_qubits, reps)
            y_ind_proba.append(p)
            y_ind_pred.append(1 if p >= 0.5 else 0)
        
        y_ind_pred = np.array(y_ind_pred)
        y_ind_proba = np.array(y_ind_proba)
        
        ind_acc = accuracy_score(y_ind, y_ind_pred)
        
        print("\n=== Amplitude VQC (independent set) ===")
        print(f"Accuracy: {ind_acc:.4f}")
        print(classification_report(y_ind, y_ind_pred, digits=4))
        
        try:
            ind_auc = roc_auc_score(y_ind, y_ind_proba)
            print(f"ROC-AUC: {ind_auc:.4f}")
        except:
            ind_auc = None
        
        results["ind_acc"] = ind_acc
        results["ind_auc"] = ind_auc
    
    return results


def run_amplitude_vqc_from_csv(
    train_csv: str = "results/train_internal_top_16_anova_f.csv",
    ind_csv: str | None = "results/independent_top_16_anova_f.csv",
    reps: int = 1,
    epochs: int = 50,
    learning_rate: float = 0.1,
    test_size: float = 0.3,
    seed: int = 42,
):
    """
    Convenience wrapper: load CSVs and run Amplitude VQC.
    
    Example:
        from vqc_golub import run_amplitude_vqc_from_csv
        
        results = run_amplitude_vqc_from_csv(
            train_csv="results/train_internal_top_16_snr.csv",
            ind_csv="results/independent_top_16_snr.csv",
            reps=1,
            epochs=50,
        )
    """
    (X_train, y_train), (X_ind, y_ind) = load_processed(train_csv, ind_csv)
    ind_data = (X_ind, y_ind) if X_ind is not None else None
    
    return train_eval_amplitude_vqc(
        X_train,
        y_train,
        ind_data=ind_data,
        reps=reps,
        test_size=test_size,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
    )
