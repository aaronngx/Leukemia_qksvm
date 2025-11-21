"""
Amplitude encoding for quantum classifiers.

Encodes K features into ceil(log₂K) qubits as quantum state amplitudes.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def get_num_qubits(num_features: int) -> int:
    """Calculate number of qubits needed: ceil(log₂(num_features))."""
    return max(1, int(np.ceil(np.log2(num_features))))


def amplitude_encoding_circuit(num_features: int):
    """
    Create amplitude encoding circuit.

    Encodes K features into ceil(log₂K) qubits.
    Example: 16 features → 4 qubits, 32 features → 5 qubits

    Parameters
    ----------
    num_features : int
        Number of input features K

    Returns
    -------
    qc : QuantumCircuit
    n_qubits : int
        Number of qubits (ceil(log₂K))
    """
    n_qubits = get_num_qubits(num_features)
    dim = 2 ** n_qubits

    qc = QuantumCircuit(n_qubits, name="AmpEnc")

    # Note: This is a placeholder circuit structure
    # Actual amplitude encoding requires state preparation
    # which depends on the specific input values

    return qc, n_qubits


def encode_amplitude(x: np.ndarray) -> QuantumCircuit:
    """
    Encode a feature vector as quantum state amplitudes.

    |ψ⟩ = Σᵢ xᵢ|i⟩ where xᵢ are normalized features

    Parameters
    ----------
    x : np.ndarray
        Feature vector of shape (K,)

    Returns
    -------
    qc : QuantumCircuit
        Circuit that prepares the amplitude-encoded state
    """
    x = np.asarray(x, dtype=float).ravel()
    num_features = len(x)
    n_qubits = get_num_qubits(num_features)
    dim = 2 ** n_qubits

    # Normalize to unit vector
    norm = np.linalg.norm(x)
    if norm < 1e-10:
        raise ValueError("Cannot encode zero vector")
    x_norm = x / norm

    # Pad to dimension 2^n_qubits
    amplitudes = np.zeros(dim)
    amplitudes[:num_features] = x_norm

    # Re-normalize after padding
    amplitudes = amplitudes / np.linalg.norm(amplitudes)

    qc = QuantumCircuit(n_qubits, name="AmpEnc")
    qc.initialize(amplitudes, range(n_qubits))

    return qc


def amplitude_kernel_element(x: np.ndarray, z: np.ndarray) -> float:
    """
    Compute kernel element |⟨φ(x)|φ(z)⟩|² for amplitude encoding.

    Parameters
    ----------
    x, z : np.ndarray
        Feature vectors

    Returns
    -------
    float
        Kernel value (squared inner product of normalized vectors)
    """
    from qiskit.quantum_info import Statevector

    # For amplitude encoding, kernel is simply |⟨x|z⟩|² (normalized)
    x_norm = x / np.linalg.norm(x)
    z_norm = z / np.linalg.norm(z)

    # Pad to same dimension
    n_qubits = get_num_qubits(max(len(x), len(z)))
    dim = 2 ** n_qubits

    x_padded = np.zeros(dim)
    x_padded[:len(x_norm)] = x_norm
    x_padded = x_padded / np.linalg.norm(x_padded)

    z_padded = np.zeros(dim)
    z_padded[:len(z_norm)] = z_norm
    z_padded = z_padded / np.linalg.norm(z_padded)

    # Inner product squared
    return float(np.abs(np.dot(x_padded, z_padded)) ** 2)


def build_amplitude_kernel(XA: np.ndarray, XB: np.ndarray, verbose: bool = True) -> np.ndarray:
    """
    Build kernel matrix using amplitude encoding.

    K[i,j] = |⟨φ(xᵢ)|φ(zⱼ)⟩|²

    For amplitude encoding, this is simply the squared inner product
    of the normalized feature vectors.

    Parameters
    ----------
    XA : np.ndarray of shape (n_samples_A, n_features)
    XB : np.ndarray of shape (n_samples_B, n_features)

    Returns
    -------
    K : np.ndarray of shape (n_samples_A, n_samples_B)
    """
    nA, nB = len(XA), len(XB)
    K = np.zeros((nA, nB))

    n_qubits = get_num_qubits(XA.shape[1])
    if verbose:
        print(f"[INFO] Amplitude encoding: {XA.shape[1]} features → {n_qubits} qubits")
        print(f"[INFO] Computing {nA} x {nB} kernel elements...")

    for i in range(nA):
        if verbose and (i % 5 == 0 or i == nA - 1):
            print(f"  Progress: {i+1}/{nA} rows")
        for j in range(nB):
            K[i, j] = amplitude_kernel_element(XA[i], XB[j])

    return K
