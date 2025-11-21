from typing import Tuple

import numpy as np
from qiskit import QuantumCircuit


def amplitude_encoding_circuit(x: np.ndarray) -> Tuple[QuantumCircuit, int]:
    """
    Build a circuit that prepares a data vector |x> as an amplitude-encoded state.

    This function assumes a *single* classical feature vector and returns a
    fully-specified QuantumCircuit (no free parameters). The vector is
    ℓ2-normalized and zero-padded to the next power-of-two dimension.

    Parameters
    ----------
    x : np.ndarray
        Real-valued feature vector of shape (d,) or (d, 1).

    Returns
    -------
    qc : QuantumCircuit
        Circuit that prepares the amplitude-encoded state |x>.
    n_qubits : int
        Number of qubits required (⌈log2 d⌉).
    """
    # Flatten and cast to 1D float array
    x = np.asarray(x, dtype=float).ravel()
    d = x.shape[0]

    if d == 0:
        raise ValueError(
            "Input vector x must have at least one element for amplitude encoding."
        )

    # ℓ2-normalize
    norm = np.linalg.norm(x)
    if norm == 0.0:
        raise ValueError(
            "Cannot amplitude-encode the all-zero vector (‖x‖₂ = 0)."
        )

    x_norm = x / norm

    # Determine number of qubits and pad to dimension 2**n_qubits
    n_qubits = max(1, int(np.ceil(np.log2(d))))

    dim = 2 ** n_qubits

    if d < dim:
        # Zero-pad the tail
        padded = np.zeros(dim, dtype=float)
        padded[:d] = x_norm
        amps = padded
    else:
        amps = x_norm

    # Sanity check: amplitudes must be normalized
    final_norm = np.linalg.norm(amps)
    if not np.isclose(final_norm, 1.0, atol=1e-8):
        # Re-normalize defensively
        amps = amps / final_norm

    qc = QuantumCircuit(n_qubits, name="AmplitudeEncoding")
    # Prepare |x> from |0...0>
    qc.initialize(amps, range(n_qubits))

    return qc, n_qubits
