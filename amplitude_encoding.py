#!/usr/bin/env python3
"""
Amplitude-Encoded Variational Quantum Circuit (VQC) for Leukemia Classification.

This implements a Quantum Neural Network with the following architecture:

1. FEATURE MAP U(x): Amplitude Encoding
   - Encodes 16 gene features into quantum state amplitudes
   - Uses Mottonen Decomposition (via Qiskit's initialize())
   - 4 qubits for 16 features: n = log₂(16) = 4
   - Deep circuit: O(2^n) gates

2. ANSATZ V(θ): Variational Layers
   - RX-RZ-RX rotation blocks (SU(2) parameterization)
   - Nearest-neighbor CNOT entanglement
   - 2-3 layers typical
   - 24 trainable parameters for 4 qubits, 2 layers

3. MEASUREMENT & PREDICTION
   - Observable: ⟨Z₀⟩ (Pauli-Z on qubit 0)
   - Prediction: p₁ = (1 + ⟨Z₀⟩) / 2 → probability of AML (class 1)

Full Pipeline:
    |0⟩⊗ⁿ → U(x) → V(θ) → Measure ⟨Z₀⟩ → p₁ = P(AML)
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp, Statevector
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_num_qubits(num_features: int) -> int:
    """
    Calculate qubits needed for amplitude encoding.
    
    n = ⌈log₂(d)⌉ qubits for d features
    
    Examples:
    - 16 features → 4 qubits (2⁴ = 16 amplitudes)
    - 50 features → 6 qubits (2⁶ = 64 amplitudes, pad with zeros)
    """
    return max(1, int(np.ceil(np.log2(num_features))))


def normalize_features(x: np.ndarray) -> np.ndarray:
    """
    L2-normalize feature vector for amplitude encoding.
    
    Required: Σᵢ|xᵢ|² = 1 for valid quantum state.
    """
    norm = np.linalg.norm(x)
    if norm < 1e-10:
        return np.ones(len(x)) / np.sqrt(len(x))
    return x / norm


def pad_to_power_of_2(x: np.ndarray) -> np.ndarray:
    """
    Pad feature vector to nearest power of 2.
    
    n qubits can only encode 2^n amplitudes.
    """
    d = len(x)
    n_qubits = get_num_qubits(d)
    target_size = 2 ** n_qubits
    
    if d == target_size:
        return x
    
    padded = np.zeros(target_size)
    padded[:d] = x
    return padded


def preprocess_for_amplitude_encoding(X: np.ndarray):
    """
    Preprocess feature matrix for amplitude encoding.
    
    Steps:
    1. Standardize features (zero mean, unit variance)
    2. Pad each sample to power of 2 length
    3. L2-normalize each sample
    
    Returns:
        X_prepared: Ready for amplitude encoding
        scaler: Fitted StandardScaler
        n_qubits: Number of qubits needed
    """
    n_samples, n_features = X.shape
    n_qubits = get_num_qubits(n_features)
    state_dim = 2 ** n_qubits
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_prepared = np.zeros((n_samples, state_dim))
    for i in range(n_samples):
        padded = pad_to_power_of_2(X_scaled[i])
        X_prepared[i] = normalize_features(padded)
    
    return X_prepared, scaler, n_qubits


# =============================================================================
# 1. FEATURE MAP U(x): AMPLITUDE ENCODING
# =============================================================================

def encode_amplitude(x: np.ndarray) -> QuantumCircuit:
    """
    Create amplitude encoding circuit for a single sample.
    
    Uses Qiskit's initialize() which internally implements
    Mottonen Decomposition for arbitrary state preparation.
    
    The resulting state is:
        |ψ(x)⟩ = Σᵢ xᵢ |i⟩
    
    where xᵢ are the normalized feature values.
    
    Parameters
    ----------
    x : np.ndarray
        Normalized feature vector (L2 norm = 1, length = 2^n)
    
    Returns
    -------
    qc : QuantumCircuit
        Circuit that prepares |ψ(x)⟩
    
    Notes
    -----
    Circuit depth is O(2^n) - this is the Mottonen decomposition.
    """
    n_qubits = int(np.log2(len(x)))
    
    qc = QuantumCircuit(n_qubits, name="U(x)")
    qc.initialize(x, range(n_qubits))
    
    return qc


# =============================================================================
# 2. ANSATZ V(θ): VARIATIONAL LAYERS
# =============================================================================

def create_ansatz(n_qubits: int, reps: int = 2) -> tuple:
    """
    Create variational ansatz with SU(2) rotation blocks.
    
    Architecture:
    - Rotation blocks: RX-RZ-RX (complete SU(2) parameterization)
    - Entanglement: Linear nearest-neighbor CNOT
    - Layers: reps repetitions
    
    For 4 qubits, 2 layers:
    - Parameters: 4 qubits × 3 rotations × (2 layers + 1) = 36 params
      (TwoLocal adds initial layer, so total = 3 × n × (reps + 1))
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits
    reps : int
        Number of ansatz layers/repetitions
    
    Returns
    -------
    ansatz : QuantumCircuit
        Parameterized ansatz circuit
    theta_params : list
        List of trainable parameters
    """
    ansatz = TwoLocal(
        num_qubits=n_qubits,
        rotation_blocks=["rx", "rz", "rx"],  # SU(2) decomposition
        entanglement_blocks="cx",             # CNOT gates
        entanglement="linear",                # Nearest-neighbor
        reps=reps,
    )
    
    return ansatz, list(ansatz.parameters)


# =============================================================================
# 3. MEASUREMENT & PREDICTION
# =============================================================================

def compute_expectation_z0(circuit: QuantumCircuit) -> float:
    """
    Compute expectation value ⟨Z₀⟩ for the circuit.
    
    The observable is Pauli-Z on qubit 0.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        Complete VQC circuit (U(x) + V(θ))
    
    Returns
    -------
    expectation : float
        ⟨Z₀⟩ ∈ [-1, 1]
    """
    sv = Statevector.from_instruction(circuit)
    
    # Create Z₀ observable (Z on qubit 0, I on others)
    n_qubits = circuit.num_qubits
    z_string = 'I' * (n_qubits - 1) + 'Z'  # Z on qubit 0
    observable = SparsePauliOp.from_list([(z_string, 1.0)])
    
    expectation = sv.expectation_value(observable).real
    return expectation


def expectation_to_probability(expectation: float) -> float:
    """
    Convert ⟨Z₀⟩ to probability p₁ (probability of class 1 / AML).
    
    p₁ = (1 + ⟨Z₀⟩) / 2
    
    Maps [-1, 1] → [0, 1]
    """
    return (1 + expectation) / 2


def predict_class(probability: float, threshold: float = 0.5) -> int:
    """
    Convert probability to class prediction.
    
    Returns:
        0 (ALL) if p₁ < threshold
        1 (AML) if p₁ >= threshold
    """
    return 1 if probability >= threshold else 0


# =============================================================================
# COMPLETE VQC CIRCUIT
# =============================================================================

def build_vqc_circuit(x: np.ndarray, theta: np.ndarray, n_qubits: int, reps: int = 2) -> QuantumCircuit:
    """
    Build complete VQC circuit: U(x) + V(θ).
    
    Architecture:
        |0⟩⊗ⁿ → [Amplitude Encoding U(x)] → [Ansatz V(θ)] → Measure
    
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
    # 1. Feature Map U(x): Amplitude Encoding
    qc = QuantumCircuit(n_qubits, name="VQC")
    qc.compose(encode_amplitude(x), inplace=True)
    qc.barrier(label="U(x)|V(θ)")
    
    # 2. Ansatz V(θ)
    ansatz, theta_params = create_ansatz(n_qubits, reps)
    
    # Bind parameters
    param_dict = {theta_params[i]: theta[i] for i in range(len(theta_params))}
    bound_ansatz = ansatz.assign_parameters(param_dict)
    
    qc.compose(bound_ansatz, inplace=True)
    
    return qc


def forward_pass(x: np.ndarray, theta: np.ndarray, n_qubits: int, reps: int = 2) -> float:
    """
    Forward pass through VQC.
    
    Returns probability p₁ (probability of AML).
    """
    qc = build_vqc_circuit(x, theta, n_qubits, reps)
    expectation = compute_expectation_z0(qc)
    probability = expectation_to_probability(expectation)
    return probability


# =============================================================================
# VQC CLASS
# =============================================================================

class AmplitudeEncodedVQC:
    """
    Amplitude-Encoded Variational Quantum Classifier.
    
    Architecture:
    1. U(x): Amplitude encoding (Mottonen decomposition)
    2. V(θ): RX-RZ-RX ansatz with linear CNOT
    3. Measurement: ⟨Z₀⟩ → p₁ = P(AML)
    
    Parameters
    ----------
    num_features : int
        Number of input features (e.g., 16 genes)
    reps : int
        Number of ansatz layers (default: 2)
    learning_rate : float
        Learning rate for gradient descent
    """
    
    def __init__(self, num_features: int, reps: int = 2, learning_rate: float = 0.1):
        self.num_features = num_features
        self.n_qubits = get_num_qubits(num_features)
        self.state_dim = 2 ** self.n_qubits
        self.reps = reps
        self.learning_rate = learning_rate
        
        # Create ansatz to get parameter count
        ansatz, theta_params = create_ansatz(self.n_qubits, reps)
        self.n_params = len(theta_params)
        
        # Initialize parameters randomly
        np.random.seed(42)
        self.theta = np.random.uniform(-np.pi, np.pi, self.n_params)
        
        # Preprocessing
        self.scaler = None
        
        self._print_config()
    
    def _print_config(self):
        """Print configuration summary."""
        print("\n" + "=" * 70)
        print("AMPLITUDE-ENCODED VQC CONFIGURATION")
        print("=" * 70)
        print(f"\n1. FEATURE MAP U(x): Amplitude Encoding")
        print(f"   • Features: {self.num_features}")
        print(f"   • Qubits: {self.n_qubits} (log₂({self.num_features}) = {self.n_qubits})")
        print(f"   • State dimension: {self.state_dim}")
        print(f"   • Method: Mottonen Decomposition (via initialize())")
        print(f"   • Depth: O(2^{self.n_qubits}) = O({self.state_dim}) gates")
        print(f"\n2. ANSATZ V(θ): Variational Layers")
        print(f"   • Rotation blocks: RX-RZ-RX (SU(2) parameterization)")
        print(f"   • Entanglement: Linear nearest-neighbor CNOT")
        print(f"   • Layers: {self.reps}")
        print(f"   • Trainable parameters: {self.n_params}")
        print(f"   • Formula: {self.n_qubits} qubits × 3 rotations × ({self.reps}+1) = {self.n_params}")
        print(f"\n3. MEASUREMENT & PREDICTION")
        print(f"   • Observable: ⟨Z₀⟩ (Pauli-Z on qubit 0)")
        print(f"   • Prediction: p₁ = (1 + ⟨Z₀⟩) / 2")
        print(f"   • Output: P(AML) ∈ [0, 1]")
        print("=" * 70 + "\n")
    
    def _preprocess(self, X: np.ndarray, fit: bool = False):
        """Preprocess features for amplitude encoding."""
        n_samples = X.shape[0]
        
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        X_prepared = np.zeros((n_samples, self.state_dim))
        for i in range(n_samples):
            padded = pad_to_power_of_2(X_scaled[i])
            X_prepared[i] = normalize_features(padded)
        
        return X_prepared
    
    def _forward(self, x: np.ndarray) -> float:
        """Forward pass for single sample."""
        return forward_pass(x, self.theta, self.n_qubits, self.reps)
    
    def _compute_gradient(self, x: np.ndarray, y: int, epsilon: float = 0.01) -> np.ndarray:
        """
        Compute gradient using parameter-shift rule.
        
        ∂f/∂θᵢ = [f(θ + π/2·eᵢ) - f(θ - π/2·eᵢ)] / 2
        """
        gradients = np.zeros(self.n_params)
        
        for i in range(self.n_params):
            # θ + π/2
            theta_plus = self.theta.copy()
            theta_plus[i] += np.pi / 2
            f_plus = forward_pass(x, theta_plus, self.n_qubits, self.reps)
            
            # θ - π/2
            theta_minus = self.theta.copy()
            theta_minus[i] -= np.pi / 2
            f_minus = forward_pass(x, theta_minus, self.n_qubits, self.reps)
            
            # Gradient
            gradients[i] = (f_plus - f_minus) / 2
        
        # Scale by loss gradient (for binary cross-entropy)
        p = self._forward(x)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        
        if y == 1:
            loss_grad = -1 / p
        else:
            loss_grad = 1 / (1 - p)
        
        return gradients * loss_grad
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, verbose: bool = True):
        """
        Train the VQC.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features
        y : np.ndarray of shape (n_samples,)
            Training labels (0 or 1)
        epochs : int
            Number of training epochs
        verbose : bool
            Print progress
        """
        if verbose:
            print("\n" + "=" * 70)
            print("TRAINING AMPLITUDE-ENCODED VQC")
            print("=" * 70)
            print(f"Samples: {len(X)}")
            print(f"Features: {X.shape[1]}")
            print(f"Epochs: {epochs}")
            print(f"Learning rate: {self.learning_rate}")
            print("=" * 70 + "\n")
        
        # Preprocess
        X_prepared = self._preprocess(X, fit=True)
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            
            for i in range(len(X_prepared)):
                x = X_prepared[i]
                label = y[i]
                
                # Forward pass
                p = self._forward(x)
                
                # Loss (binary cross-entropy)
                p_clipped = np.clip(p, 1e-10, 1 - 1e-10)
                loss = -label * np.log(p_clipped) - (1 - label) * np.log(1 - p_clipped)
                total_loss += loss
                
                # Prediction
                pred = predict_class(p)
                if pred == label:
                    correct += 1
                
                # Gradient descent (every sample for SGD)
                grad = self._compute_gradient(x, label)
                self.theta -= self.learning_rate * grad
            
            avg_loss = total_loss / len(X_prepared)
            accuracy = correct / len(X_prepared)
            
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
        
        if verbose:
            print("\n" + "=" * 70)
            print("TRAINING COMPLETE")
            print("=" * 70 + "\n")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities P(AML) for each sample."""
        X_prepared = self._preprocess(X, fit=False)
        probas = np.array([self._forward(x) for x in X_prepared])
        return probas
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probas = self.predict_proba(X)
        return (probas >= 0.5).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def get_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """Get the full VQC circuit for a sample."""
        x_prepared = self._preprocess(x.reshape(1, -1), fit=False)[0]
        return build_vqc_circuit(x_prepared, self.theta, self.n_qubits, self.reps)


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_vqc_architecture():
    """Demonstrate the VQC architecture."""
    
    print("\n" + "=" * 70)
    print("AMPLITUDE-ENCODED VQC DEMONSTRATION (16 features → 4 qubits)")
    print("=" * 70)
    
    num_features = 16
    n_qubits = get_num_qubits(num_features)
    reps = 2
    
    # Example normalized features
    np.random.seed(42)
    x = np.random.randn(num_features)
    x_padded = pad_to_power_of_2(x)
    x_norm = normalize_features(x_padded)
    
    print(f"\nInput: {num_features} gene expression features")
    print(f"After padding: {len(x_padded)} (nearest power of 2)")
    print(f"After normalization: L2 norm = {np.linalg.norm(x_norm):.6f}")
    
    # 1. Feature Map U(x)
    print("\n" + "-" * 70)
    print("1. FEATURE MAP U(x): AMPLITUDE ENCODING")
    print("-" * 70)
    
    qc_feature = encode_amplitude(x_norm)
    qc_feature_decomp = qc_feature.decompose()
    
    print(f"   Qubits: {n_qubits}")
    print(f"   State dimension: {2**n_qubits}")
    print(f"   Circuit depth (decomposed): {qc_feature_decomp.depth()}")
    print(f"   Gate count: {qc_feature_decomp.size()}")
    print(f"\n   High-level circuit:")
    print(qc_feature.draw(output='text'))
    
    # 2. Ansatz V(θ)
    print("\n" + "-" * 70)
    print("2. ANSATZ V(θ): VARIATIONAL LAYERS")
    print("-" * 70)
    
    ansatz, theta_params = create_ansatz(n_qubits, reps)
    
    print(f"   Rotation blocks: RX-RZ-RX (SU(2))")
    print(f"   Entanglement: Linear CNOT")
    print(f"   Layers: {reps}")
    print(f"   Trainable parameters: {len(theta_params)}")
    print(f"\n   Ansatz circuit:")
    print(ansatz.decompose().draw(output='text', fold=100))
    
    # 3. Measurement
    print("\n" + "-" * 70)
    print("3. MEASUREMENT & PREDICTION")
    print("-" * 70)
    
    # Build complete circuit
    theta = np.random.uniform(-np.pi, np.pi, len(theta_params))
    qc = build_vqc_circuit(x_norm, theta, n_qubits, reps)
    
    expectation = compute_expectation_z0(qc)
    probability = expectation_to_probability(expectation)
    prediction = predict_class(probability)
    
    print(f"   Observable: ⟨Z₀⟩")
    print(f"   Expectation value: {expectation:.4f}")
    print(f"   Probability P(AML): {probability:.4f}")
    print(f"   Prediction: {'AML (1)' if prediction == 1 else 'ALL (0)'}")
    
    print("\n" + "=" * 70)
    print("VQC ARCHITECTURE SUMMARY")
    print("=" * 70)
    print(f"""
    |0⟩⊗⁴ ──[U(x): Amplitude Encoding]──[V(θ): Ansatz]──[Measure ⟨Z₀⟩]── p₁
    
    U(x): Mottonen Decomposition
          16 features → 4 qubits
          O(2⁴) = O(16) gates
    
    V(θ): RX-RZ-RX + CNOT layers
          {len(theta_params)} trainable parameters
          {reps} repetitions
    
    Output: p₁ = (1 + ⟨Z₀⟩) / 2 = P(AML)
    """)
    
    return qc


if __name__ == "__main__":
    demonstrate_vqc_architecture()
