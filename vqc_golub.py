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
    circuit, clf = train_eval_vqc(X, y, reps_ansatz=2)
    
    # Amplitude encoding
    from vqc_golub import train_eval_amplitude_vqc
    clf = train_eval_amplitude_vqc(X, y, reps_ansatz=1)
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


def build_vqc(num_features: int, reps_ansatz: int = 2):
    """
    Build VQC circuit: angle encoding + TwoLocal ansatz.
    
    Note: reps_feature = 1 (FIXED) - Feature map U(x) applied once
          reps_ansatz ∈ {1,...,5} - Ansatz V(θ) repetitions

    Returns
    -------
    qc        : QuantumCircuit for U(x, θ)
    x_params  : list of feature parameters
    w_params  : list of trainable parameters
    """
    # Feature map U(x) - applied once (reps_feature = 1)
    feature_map, x_params = angle_encoding_circuit(num_features)
    
    # Ansatz V(θ) - configurable repetitions
    if reps_ansatz < 1 or reps_ansatz > 5:
        raise ValueError(f"reps_ansatz must be in {{1,...,5}}, got {reps_ansatz}")
    
    ansatz = TwoLocal(
        num_qubits=num_features,
        rotation_blocks=["rx", "rz", "rx"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=reps_ansatz,
    )

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(num_features, name="VQC")
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    return qc, list(x_params), list(ansatz.parameters)


def make_optimizer(name: str, max_iter: int = 200, **kwargs):
    """
    Create an optimizer by name.

    Supported:
      - 'COBYLA': Constrained Optimization BY Linear Approximations
      - 'SPSA': Simultaneous Perturbation Stochastic Approximation
      - 'ADAM': Adaptive Moment Estimation
      - 'ENSGA': Elitist Non-Dominated Sorting Genetic Algorithm (returns None, handled separately)
    
    Parameters
    ----------
    name : str
        Optimizer name
    max_iter : int
        Maximum iterations
    **kwargs : dict
        Additional optimizer-specific parameters:
        - SPSA: learning_rate, perturbation
        - ADAM: lr, beta_1, beta_2
    """
    name = name.lower()
    if name == "cobyla":
        return COBYLA(maxiter=max_iter)
    if name == "spsa":
        learning_rate = kwargs.get('learning_rate', 0.05)
        perturbation = kwargs.get('perturbation', 0.1)
        return SPSA(maxiter=max_iter, learning_rate=learning_rate, perturbation=perturbation)
    if name == "adam":
        lr = kwargs.get('lr', 0.01)
        return ADAM(maxiter=max_iter, lr=lr)
    if name == "ensga":
        # ENSGA is handled separately in training loop
        return None
    raise ValueError(f"Unknown optimizer: {name}. Use 'COBYLA', 'SPSA', 'ADAM', or 'ENSGA'.")


def train_eval_vqc(
    X: np.ndarray,
    y: np.ndarray,
    ind_data: tuple[np.ndarray, np.ndarray] | None = None,
    reps_ansatz: int = 2,
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
    
    Note: reps_feature = 1 (FIXED) - Feature map U(x) applied once

    Parameters
    ----------
    X, y : np.ndarray
        Features and labels.
    ind_data : (X_ind, y_ind) or None
        Independent test set (optional).
    reps_ansatz : int
        Number of ansatz V(θ) repetitions. Range: {1, 2, 3, 4, 5}
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
        'COBYLA', 'SPSA', 'ADAM', or 'ENSGA'.
    """
    n_features = X.shape[1]
    backend_info = get_backend_info(backend_type, n_features)

    print("\n" + "=" * 70)
    print("VARIATIONAL QUANTUM CLASSIFIER - CONFIGURATION")
    print("=" * 70)
    print(f"Backend: {backend_type.value}")
    print(f"Number of qubits / features: {n_features}")
    print(f"reps_feature: 1 (FIXED)")
    print(f"reps_ansatz: {reps_ansatz} (configurable 1-5)")
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
    circuit, x_params, w_params = build_vqc(n_features, reps_ansatz=reps_ansatz)

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
    reps_ansatz: int = 2,
    max_iter: int = 200,
    optimizer_name: str = "COBYLA",
    test_size: float = 0.3,
    seed: int = 42,
):
    """
    Convenience wrapper: load CSVs and run VQC end-to-end.
    
    Note: reps_feature = 1 (FIXED) - Feature map U(x) applied once
          reps_ansatz ∈ {1,...,5} - Ansatz V(θ) repetitions

    Example (in notebook):

        from vqc_golub import run_vqc_from_csv, BackendType

        for opt in ["COBYLA", "SPSA", "ADAM"]:
            print("=== Optimizer:", opt, "===")
            circuit, clf = run_vqc_from_csv(
                train_csv="data/processed/train_topk_snr.csv",
                ind_csv="data/processed/independent_topk_snr.csv",
                backend_type=BackendType.TENSOR_NETWORK,
                max_bond_dimension=100,
                reps_ansatz=2,
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
        reps_ansatz=reps_ansatz,
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

def build_amplitude_vqc_circuit(x: np.ndarray, theta: np.ndarray, n_qubits: int, reps_ansatz: int = 1):
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
    reps_ansatz : int
        Number of ansatz V(θ) repetitions. Range: {1, 2, 3, 4, 5}
    
    Returns
    -------
    qc : QuantumCircuit
        Complete VQC circuit with bound parameters
    """
    # 1. Feature Map U(x): Amplitude Encoding (Mottonen Decomposition)
    # Note: reps_feature = 1 (FIXED) - applied once
    qc = QuantumCircuit(n_qubits, name="AmplitudeVQC")
    qc.compose(encode_amplitude(x), inplace=True)
    qc.barrier(label="U(x)|V(θ)")
    
    # 2. Ansatz V(θ): RX-RZ-RX + linear CNOT
    ansatz, theta_params = create_amplitude_ansatz(n_qubits, reps_ansatz)
    
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


def amplitude_vqc_forward(x: np.ndarray, theta: np.ndarray, n_qubits: int, reps_ansatz: int) -> float:
    """
    Forward pass through amplitude VQC.
    
    Returns probability p₁ = (1 + ⟨Z₀⟩) / 2 = P(AML).
    """
    qc = build_amplitude_vqc_circuit(x, theta, n_qubits, reps_ansatz)
    expectation = compute_expectation_z0(qc)
    # Convert to probability: p₁ = (1 + ⟨Z₀⟩) / 2
    return (1 + expectation) / 2


def amplitude_vqc_gradient(x: np.ndarray, y: int, theta: np.ndarray, 
                           n_qubits: int, reps_ansatz: int) -> np.ndarray:
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
        f_plus = amplitude_vqc_forward(x, theta_plus, n_qubits, reps_ansatz)
        
        # θ - π/2
        theta_minus = theta.copy()
        theta_minus[i] -= np.pi / 2
        f_minus = amplitude_vqc_forward(x, theta_minus, n_qubits, reps_ansatz)
        
        # Parameter-shift gradient
        gradients[i] = (f_plus - f_minus) / 2
    
    # Scale by loss gradient (binary cross-entropy)
    p = amplitude_vqc_forward(x, theta, n_qubits, reps_ansatz)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    
    if y == 1:
        loss_grad = -1 / p
    else:
        loss_grad = 1 / (1 - p)
    
    return gradients * loss_grad


# =============================================================================
# OPTIMIZER-SPECIFIC TRAINING FUNCTIONS
# =============================================================================

def _train_amplitude_vqc_spsa_adam(
    X_train: np.ndarray,
    y_train: np.ndarray,
    theta: np.ndarray,
    n_qubits: int,
    reps: int,
    epochs: int = 50,
    learning_rate: float = 0.1,
    optimizer_name: str = "SPSA",
    verbose: bool = True,
) -> np.ndarray:
    """Train amplitude VQC using SPSA or ADAM optimizer."""
    
    if optimizer_name == "ADAM":
        # ADAM parameters
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        m = np.zeros_like(theta)  # First moment
        v = np.zeros_like(theta)  # Second moment
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        # Compute batch gradient
        batch_grad = np.zeros_like(theta)
        
        for i in range(len(X_train)):
            x = X_train[i]
            label = y_train[i]
            
            # Forward pass
            p = amplitude_vqc_forward(x, theta, n_qubits, reps_ansatz)
            
            # Loss
            p_clipped = np.clip(p, 1e-10, 1 - 1e-10)
            loss = -label * np.log(p_clipped) - (1 - label) * np.log(1 - p_clipped)
            total_loss += loss
            
            # Prediction
            pred = 1 if p >= 0.5 else 0
            if pred == label:
                correct += 1
            
            # Accumulate gradient
            grad = amplitude_vqc_gradient(x, label, theta, n_qubits, reps_ansatz)
            batch_grad += grad
        
        batch_grad /= len(X_train)
        
        if optimizer_name == "ADAM":
            # ADAM update
            m = beta1 * m + (1 - beta1) * batch_grad
            v = beta2 * v + (1 - beta2) * (batch_grad ** 2)
            m_hat = m / (1 - beta1 ** (epoch + 1))
            v_hat = v / (1 - beta2 ** (epoch + 1))
            theta -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        else:
            # SPSA-style update (with some noise)
            perturbation = 0.1 / (epoch + 1) ** 0.602
            delta = np.random.choice([-1, 1], size=len(theta))
            theta -= learning_rate * batch_grad + perturbation * delta
        
        avg_loss = total_loss / len(X_train)
        train_acc = correct / len(X_train)
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}, Train Acc = {train_acc:.4f}")
    
    return theta


def _train_amplitude_vqc_cobyla(
    X_train: np.ndarray,
    y_train: np.ndarray,
    theta_init: np.ndarray,
    n_qubits: int,
    reps: int,
    max_iter: int = 100,
    verbose: bool = True,
) -> np.ndarray:
    """Train amplitude VQC using COBYLA (derivative-free) optimizer."""
    from scipy.optimize import minimize
    
    iteration_count = [0]
    
    def loss_fn(theta):
        total_loss = 0
        for i in range(len(X_train)):
            x = X_train[i]
            label = y_train[i]
            p = amplitude_vqc_forward(x, theta, n_qubits, reps_ansatz)
            p_clipped = np.clip(p, 1e-10, 1 - 1e-10)
            loss = -label * np.log(p_clipped) - (1 - label) * np.log(1 - p_clipped)
            total_loss += loss
        
        avg_loss = total_loss / len(X_train)
        
        iteration_count[0] += 1
        if verbose and iteration_count[0] % 20 == 0:
            # Calculate accuracy
            correct = sum(
                1 for i in range(len(X_train))
                if (amplitude_vqc_forward(X_train[i], theta, n_qubits, reps) >= 0.5) == y_train[i]
            )
            acc = correct / len(X_train)
            print(f"  Iter {iteration_count[0]:3d}: Loss = {avg_loss:.4f}, Train Acc = {acc:.4f}")
        
        return avg_loss
    
    result = minimize(
        loss_fn,
        theta_init,
        method='COBYLA',
        options={'maxiter': max_iter, 'rhobeg': 0.5}
    )
    
    return result.x


def _train_amplitude_vqc_ensga(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_qubits: int,
    reps: int,
    n_params: int,
    pop_size: int = 20,
    n_generations: int = 30,
    seed: int = 42,
    verbose: bool = True,
) -> np.ndarray:
    """
    Train amplitude VQC using ENSGA (Elitist Non-Dominated Sorting Genetic Algorithm).
    
    Multi-objective optimization:
    1. Classification error
    2. Parameter magnitude (regularization)
    """
    from ensga_optimizer import ENSGA, Individual, Population, non_dominated_sort
    
    np.random.seed(seed)
    
    def evaluate_individual(theta: np.ndarray) -> np.ndarray:
        """Compute objectives for an individual."""
        # Objective 1: Classification error
        correct = 0
        total_loss = 0
        for i in range(len(X_train)):
            x = X_train[i]
            label = y_train[i]
            p = amplitude_vqc_forward(x, theta, n_qubits, reps_ansatz)
            
            # Loss
            p_clipped = np.clip(p, 1e-10, 1 - 1e-10)
            loss = -label * np.log(p_clipped) - (1 - label) * np.log(1 - p_clipped)
            total_loss += loss
            
            # Accuracy
            pred = 1 if p >= 0.5 else 0
            if pred == label:
                correct += 1
        
        error = 1.0 - correct / len(X_train)
        avg_loss = total_loss / len(X_train)
        
        # Objective 2: Parameter magnitude (regularization)
        param_norm = np.linalg.norm(theta) / n_params
        
        return np.array([error, param_norm])
    
    # Initialize population
    population = Population()
    for _ in range(pop_size):
        genes = np.random.uniform(-np.pi, np.pi, n_params)
        ind = Individual(genes=genes)
        ind.objectives = evaluate_individual(genes)
        population.append(ind)
    
    # Evolution loop
    for gen in range(n_generations):
        # Create offspring
        offspring = Population()
        while len(offspring) < pop_size:
            # Tournament selection
            idx1 = np.random.randint(0, len(population))
            idx2 = np.random.randint(0, len(population))
            parent1 = population[idx1] if population[idx1].rank <= population[idx2].rank else population[idx2]
            
            idx1 = np.random.randint(0, len(population))
            idx2 = np.random.randint(0, len(population))
            parent2 = population[idx1] if population[idx1].rank <= population[idx2].rank else population[idx2]
            
            # Crossover (simple average)
            alpha = np.random.random()
            child_genes = alpha * parent1.genes + (1 - alpha) * parent2.genes
            
            # Mutation
            if np.random.random() < 0.2:
                mutation_idx = np.random.randint(0, n_params)
                child_genes[mutation_idx] += np.random.normal(0, 0.5)
                child_genes = np.clip(child_genes, -np.pi, np.pi)
            
            child = Individual(genes=child_genes)
            child.objectives = evaluate_individual(child_genes)
            offspring.append(child)
        
        # Merge and select
        combined = Population()
        combined.extend(population.individuals)
        combined.extend(offspring.individuals)
        
        # Non-dominated sorting
        fronts = non_dominated_sort(combined)
        
        # Select next generation
        new_pop = Population()
        front_idx = 0
        while len(new_pop) + len(fronts[front_idx]) <= pop_size and front_idx < len(fronts):
            for i in fronts[front_idx]:
                new_pop.append(combined[i])
            front_idx += 1
        
        # Fill remaining from last front
        if len(new_pop) < pop_size and front_idx < len(fronts):
            remaining = pop_size - len(new_pop)
            for i in fronts[front_idx][:remaining]:
                new_pop.append(combined[i])
        
        population = new_pop
        
        # Progress
        if verbose and (gen + 1) % 10 == 0:
            best_error = min(ind.objectives[0] for ind in population.individuals)
            print(f"  Gen {gen+1:3d}/{n_generations}: Best Error = {best_error:.4f}, Acc = {1-best_error:.4f}")
    
    # Return best individual
    best_idx = np.argmin([ind.objectives[0] for ind in population.individuals])
    return population[best_idx].genes


def train_eval_amplitude_vqc(
    X: np.ndarray,
    y: np.ndarray,
    ind_data: tuple[np.ndarray, np.ndarray] | None = None,
    reps: int = 1,
    test_size: float = 0.3,
    seed: int = 42,
    epochs: int = 50,
    learning_rate: float = 0.1,
    optimizer_name: str = "ADAM",
    ensga_pop_size: int = 20,
    ensga_generations: int = 30,
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
        Training epochs (for SGD/SPSA/ADAM)
    learning_rate : float
        Learning rate (for SGD/SPSA/ADAM)
    optimizer_name : str
        Optimizer: 'ADAM', 'SPSA', 'COBYLA', or 'ENSGA'
    ensga_pop_size : int
        ENSGA population size (if optimizer_name='ENSGA')
    ensga_generations : int
        ENSGA generations (if optimizer_name='ENSGA')
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
        print(f"   • Optimizer: {optimizer_name.upper()}")
        if optimizer_name.upper() == "ENSGA":
            print(f"   • Population: {ensga_pop_size}")
            print(f"   • Generations: {ensga_generations}")
        else:
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
    
    # Select training method based on optimizer
    optimizer_name_upper = optimizer_name.upper()
    
    if optimizer_name_upper == "ENSGA":
        # ENSGA: Multi-objective evolutionary optimization
        if verbose:
            print(f"\n[INFO] Training with ENSGA (evolutionary optimization)...")
        
        theta = _train_amplitude_vqc_ensga(
            X_train, y_train, n_qubits, reps, n_params,
            pop_size=ensga_pop_size,
            n_generations=ensga_generations,
            seed=seed,
            verbose=verbose,
        )
    
    elif optimizer_name_upper in ["SPSA", "ADAM"]:
        # SPSA/ADAM with parameter-shift gradients
        if verbose:
            print(f"\n[INFO] Training with {optimizer_name_upper} (parameter-shift gradients)...")
        
        theta = _train_amplitude_vqc_spsa_adam(
            X_train, y_train, theta, n_qubits, reps,
            epochs=epochs,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name_upper,
            verbose=verbose,
        )
    
    else:
        # COBYLA: Derivative-free optimization (default fallback)
        if verbose:
            print(f"\n[INFO] Training with COBYLA (derivative-free)...")
        
        theta = _train_amplitude_vqc_cobyla(
            X_train, y_train, theta, n_qubits, reps,
            max_iter=epochs,
            verbose=verbose,
        )
    
    # Validation
    if verbose:
        print(f"\n[INFO] Evaluating on validation set...")
    
    y_val_pred = []
    y_val_proba = []
    for x in X_val:
        p = amplitude_vqc_forward(x, theta, n_qubits, reps_ansatz)
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
            p = amplitude_vqc_forward(x, theta, n_qubits, reps_ansatz)
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
    optimizer_name: str = "ADAM",
    ensga_pop_size: int = 20,
    ensga_generations: int = 30,
    test_size: float = 0.3,
    seed: int = 42,
):
    """
    Convenience wrapper: load CSVs and run Amplitude VQC.
    
    Supported optimizers: ADAM, SPSA, COBYLA, ENSGA
    
    Example:
        from vqc_golub import run_amplitude_vqc_from_csv
        
        # ADAM (default) - adaptive learning rate
        results = run_amplitude_vqc_from_csv(
            train_csv="results/train_internal_top_16_snr.csv",
            ind_csv="results/independent_top_16_snr.csv",
            optimizer_name="ADAM",
            epochs=100,
            learning_rate=0.01,
        )
        
        # SPSA - stochastic perturbation
        results = run_amplitude_vqc_from_csv(
            train_csv="results/train_internal_top_16_snr.csv",
            optimizer_name="SPSA",
            epochs=100,
        )
        
        # COBYLA - derivative-free
        results = run_amplitude_vqc_from_csv(
            train_csv="results/train_internal_top_16_snr.csv",
            optimizer_name="COBYLA",
            epochs=100,
        )
        
        # ENSGA - multi-objective evolutionary
        results = run_amplitude_vqc_from_csv(
            train_csv="results/train_internal_top_16_snr.csv",
            optimizer_name="ENSGA",
            ensga_pop_size=30,
            ensga_generations=50,
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
        optimizer_name=optimizer_name,
        ensga_pop_size=ensga_pop_size,
        ensga_generations=ensga_generations,
    )
