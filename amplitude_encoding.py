#!/usr/bin/env python3
"""
Amplitude Encoding Feature Map - Pure Encoding Functions ONLY.

This module provides amplitude encoding for quantum machine learning.
Features are encoded as probability amplitudes of the quantum state:

    |ψ(x)⟩ = Σᵢ xᵢ |i⟩

Key Properties:
- Logarithmic qubit scaling: n = ⌈log₂(d)⌉ qubits for d features
- Features encoded in AMPLITUDES, not rotation angles
- Uses Mottonen Decomposition (via Qiskit's initialize())
- Deep circuit: O(2^n) gates for exact state preparation

Usage:
    from amplitude_encoding import encode_amplitude, preprocess_for_amplitude_encoding
    
    # Preprocess features
    X_prepared, scaler, n_qubits = preprocess_for_amplitude_encoding(X)
    
    # Encode single sample
    qc = encode_amplitude(X_prepared[0])
"""

import numpy as np
from qiskit import QuantumCircuit
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CORE UTILITY FUNCTIONS
# =============================================================================

def get_num_qubits(num_features: int) -> int:
    """
    Calculate qubits needed for amplitude encoding.
    
    Logarithmic scaling: n = ⌈log₂(d)⌉
    
    Examples:
        4 features  → 2 qubits  (2² = 4 amplitudes)
        8 features  → 3 qubits  (2³ = 8 amplitudes)
        16 features → 4 qubits  (2⁴ = 16 amplitudes)
        50 features → 6 qubits  (2⁶ = 64 amplitudes, pad with zeros)
    
    Parameters
    ----------
    num_features : int
        Number of classical features to encode
    
    Returns
    -------
    int
        Number of qubits required
    """
    return max(1, int(np.ceil(np.log2(num_features))))


def normalize_features(x: np.ndarray) -> np.ndarray:
    """
    L2-normalize feature vector for amplitude encoding.
    
    The quantum state |ψ⟩ = Σᵢ αᵢ|i⟩ requires Σᵢ|αᵢ|² = 1.
    
    Parameters
    ----------
    x : np.ndarray
        Feature vector of shape (d,)
    
    Returns
    -------
    np.ndarray
        L2-normalized vector with ||x|| = 1
    """
    norm = np.linalg.norm(x)
    if norm < 1e-10:
        # Handle zero vector - return uniform distribution
        return np.ones(len(x)) / np.sqrt(len(x))
    return x / norm


def pad_to_power_of_2(x: np.ndarray) -> np.ndarray:
    """
    Pad feature vector to nearest power of 2.
    
    Required because n qubits can only encode 2^n amplitudes.
    
    Parameters
    ----------
    x : np.ndarray
        Feature vector of shape (d,)
    
    Returns
    -------
    np.ndarray
        Padded vector of shape (2^n,) where n = ⌈log₂(d)⌉
    """
    d = len(x)
    n_qubits = get_num_qubits(d)
    target_size = 2 ** n_qubits
    
    if d == target_size:
        return x
    
    # Pad with zeros
    padded = np.zeros(target_size)
    padded[:d] = x
    return padded


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_for_amplitude_encoding(X: np.ndarray, scaler: StandardScaler = None):
    """
    Preprocess feature matrix for amplitude encoding.
    
    Steps:
    1. Standardize features (zero mean, unit variance)
    2. Pad each sample to power of 2 length
    3. L2-normalize each sample (required for valid quantum state)
    
    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Raw feature matrix
    scaler : StandardScaler, optional
        Pre-fitted scaler for transform (used for test data)
    
    Returns
    -------
    X_prepared : np.ndarray of shape (n_samples, 2^n_qubits)
        Prepared features ready for amplitude encoding
    scaler : StandardScaler
        Fitted scaler for transforming new data
    n_qubits : int
        Number of qubits needed
    """
    n_samples, n_features = X.shape
    n_qubits = get_num_qubits(n_features)
    state_dim = 2 ** n_qubits
    
    # Standardize
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # Prepare each sample
    X_prepared = np.zeros((n_samples, state_dim))
    for i in range(n_samples):
        # Pad to power of 2
        padded = pad_to_power_of_2(X_scaled[i])
        # Normalize to unit vector (required for quantum state)
        X_prepared[i] = normalize_features(padded)
    
    return X_prepared, scaler, n_qubits


# =============================================================================
# AMPLITUDE ENCODING CIRCUIT
# =============================================================================

def encode_amplitude(x: np.ndarray) -> QuantumCircuit:
    """
    Create amplitude encoding circuit for a single sample.
    
    Uses Qiskit's initialize() which internally implements
    Mottonen Decomposition for arbitrary state preparation.
    
    The resulting quantum state is:
        |ψ(x)⟩ = Σᵢ xᵢ |i⟩
    
    where xᵢ are the normalized feature values stored as amplitudes.
    
    Parameters
    ----------
    x : np.ndarray
        Normalized feature vector (must have L2 norm = 1)
        Length must be power of 2
    
    Returns
    -------
    qc : QuantumCircuit
        Circuit that prepares |ψ(x)⟩ from |0⟩⊗ⁿ
    
    Notes
    -----
    Circuit depth is O(2^n) for n qubits - this is unavoidable
    for exact arbitrary state preparation (Mottonen decomposition).
    
    Examples
    --------
    >>> x = np.array([0.5, 0.5, 0.5, 0.5])  # 4 features, normalized
    >>> qc = encode_amplitude(x)
    >>> print(qc.num_qubits)  # 2 qubits for 4 features
    2
    """
    n_qubits = int(np.log2(len(x)))
    
    qc = QuantumCircuit(n_qubits, name="U(x)")
    
    # Use Qiskit's initialize() which uses Mottonen-like decomposition
    # This creates the state |ψ(x)⟩ = Σᵢ xᵢ |i⟩
    qc.initialize(x, range(n_qubits))
    
    return qc


def get_amplitude_encoding_info(num_features: int) -> dict:
    """
    Get information about amplitude encoding configuration.
    
    Parameters
    ----------
    num_features : int
        Number of features to encode
    
    Returns
    -------
    dict
        Configuration information
    """
    n_qubits = get_num_qubits(num_features)
    state_dim = 2 ** n_qubits
    
    return {
        "num_features": num_features,
        "n_qubits": n_qubits,
        "state_dimension": state_dim,
        "encoding_depth": f"O(2^{n_qubits}) = O({state_dim})",
        "method": "Mottonen Decomposition (via initialize())",
        "qubit_scaling": "Logarithmic",
    }


def print_amplitude_encoding_info(num_features: int):
    """Print amplitude encoding configuration summary."""
    info = get_amplitude_encoding_info(num_features)
    
    print(f"\n{'='*60}")
    print("AMPLITUDE ENCODING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Features:        {info['num_features']}")
    print(f"Qubits:          {info['n_qubits']} (log₂({info['num_features']}) = {info['n_qubits']})")
    print(f"State dimension: {info['state_dimension']}")
    print(f"Circuit depth:   {info['encoding_depth']}")
    print(f"Method:          {info['method']}")
    print(f"Qubit scaling:   {info['qubit_scaling']}")
    print(f"{'='*60}\n")


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("AMPLITUDE ENCODING DEMONSTRATION")
    print("="*70)
    
    # Example: 16 features
    num_features = 16
    print_amplitude_encoding_info(num_features)
    
    # Create sample data
    np.random.seed(42)
    x = np.random.randn(num_features)
    
    # Preprocess
    x_padded = pad_to_power_of_2(x)
    x_norm = normalize_features(x_padded)
    
    print(f"Original features: {num_features}")
    print(f"After padding: {len(x_padded)}")
    print(f"L2 norm: {np.linalg.norm(x_norm):.6f}")
    
    # Encode
    qc = encode_amplitude(x_norm)
    
    print(f"\nCircuit:")
    print(f"  Qubits: {qc.num_qubits}")
    print(f"  Depth: {qc.decompose().depth()}")
    print(f"  Gates: {qc.decompose().size()}")
    
    # Verify state
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_instruction(qc)
    
    print(f"\nState verification:")
    print(f"  Input amplitudes (first 4): {x_norm[:4]}")
    print(f"  Output amplitudes (first 4): {[sv.data[i].real for i in range(4)]}")
    print(f"  Match: {np.allclose(x_norm, sv.data.real)}")
    
    print("\n" + "="*70)
    print("Circuit diagram:")
    print("="*70)
    print(qc.draw(output='text'))
