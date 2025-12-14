#!/usr/bin/env python3
"""
Variational Ansatz V(θ) Definitions.

This module provides parameterized quantum circuits (ansatz) that form
the trainable layers in Variational Quantum Circuits (VQC).

Repetition Parameters:
- reps_feature = 1 (FIXED) - Feature map U(x) is applied once
- reps_ansatz ∈ {1,...,5} (CONFIGURABLE) - Ansatz V(θ) repetitions for VQC

Architecture:
- Rotation blocks: RX-RZ-RX (complete SU(2) parameterization)
- Entanglement: Linear nearest-neighbor CNOT gates
- Configurable depth via 'reps_ansatz' parameter

Parameter Count Formula:
    n_params = n_qubits × 3 rotations × (reps_ansatz + 1) layers

Examples:
    Amplitude encoding (4 qubits, reps_ansatz=1): 4 × 3 × 2 = 24 params
    Angle encoding (16 qubits, reps_ansatz=2): 16 × 3 × 3 = 144 params

Usage:
    from ansatz import create_amplitude_ansatz, create_angle_ansatz
    
    ansatz, theta_params = create_amplitude_ansatz(n_qubits=4, reps_ansatz=1)
    ansatz, theta_params = create_angle_ansatz(n_qubits=16, reps_ansatz=2)
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal


def create_su2_ansatz(n_qubits: int, reps_ansatz: int = 2) -> tuple:
    """
    Create SU(2) variational ansatz with RX-RZ-RX rotation blocks.
    
    This is the core ansatz builder used by both amplitude and angle
    encoding VQCs. The RX-RZ-RX combination provides a complete
    parameterization of SU(2) single-qubit rotations.
    
    Architecture:
        [RX-RZ-RX layer] → [Linear CNOT] → [RX-RZ-RX layer] → ...
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit
    reps_ansatz : int
        Number of repetitions of the ansatz V(θ). Range: {1, 2, 3, 4, 5}
        Total layers = reps_ansatz + 1 (includes initial rotation layer)
    
    Returns
    -------
    ansatz : QuantumCircuit
        Parameterized ansatz circuit
    theta_params : list
        List of Parameter objects for trainable weights
    
    Examples
    --------
    >>> ansatz, params = create_su2_ansatz(4, reps_ansatz=1)
    >>> print(len(params))  # 4 qubits × 3 rotations × 2 layers
    24
    """
    # Validate reps_ansatz for VQC (1-5 allowed)
    if reps_ansatz < 1 or reps_ansatz > 5:
        raise ValueError(f"reps_ansatz must be in {{1,...,5}}, got {reps_ansatz}")
    
    ansatz = TwoLocal(
        num_qubits=n_qubits,
        rotation_blocks=["rx", "rz", "rx"],  # SU(2) decomposition
        entanglement_blocks="cx",             # CNOT gates
        entanglement="linear",                # Nearest-neighbor
        reps=reps_ansatz,
    )
    
    return ansatz, list(ansatz.parameters)


def create_amplitude_ansatz(n_qubits: int, reps_ansatz: int = 1) -> tuple:
    """
    Create ansatz optimized for amplitude encoding VQC.
    
    Note: reps_feature = 1 (FIXED) - Feature map U(x) applied once
    
    Default Configuration:
    - reps_ansatz=1 → 2 layers (optimal for amplitude encoding)
    - For 4 qubits: 24 trainable parameters
    
    The shallower depth is appropriate because amplitude encoding
    already has a deep feature map (Mottonen decomposition).
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits (typically log₂(features))
    reps_ansatz : int
        Number of ansatz V(θ) repetitions. Range: {1, 2, 3, 4, 5}
        Default: 1 (2 layers)
    
    Returns
    -------
    ansatz : QuantumCircuit
        Parameterized ansatz circuit
    theta_params : list
        List of trainable parameters
    
    Notes
    -----
    For amplitude encoding with 16 features:
        - n_qubits = 4
        - reps_ansatz = 1 (default)
        - Parameters = 4 × 3 × 2 = 24
    """
    return create_su2_ansatz(n_qubits, reps_ansatz)


def create_angle_ansatz(n_qubits: int, reps_ansatz: int = 2) -> tuple:
    """
    Create ansatz optimized for angle encoding VQC.
    
    Note: reps_feature = 1 (FIXED) - Feature map U(x) applied once
    
    Default Configuration:
    - reps_ansatz=2 → 3 layers (optimal for angle encoding)
    - For 16 qubits: 144 trainable parameters
    
    The deeper ansatz compensates for the shallow feature map
    in angle encoding (just single RY gates).
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits (= number of features for angle encoding)
    reps_ansatz : int
        Number of ansatz V(θ) repetitions. Range: {1, 2, 3, 4, 5}
        Default: 2 (3 layers)
    
    Returns
    -------
    ansatz : QuantumCircuit
        Parameterized ansatz circuit
    theta_params : list
        List of trainable parameters
    
    Notes
    -----
    For angle encoding with 16 features:
        - n_qubits = 16
        - reps_ansatz = 2 (default)
        - Parameters = 16 × 3 × 3 = 144
    """
    return create_su2_ansatz(n_qubits, reps_ansatz)


def get_ansatz_info(n_qubits: int, reps_ansatz: int) -> dict:
    """
    Get information about ansatz configuration.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits
    reps_ansatz : int
        Number of ansatz repetitions (1-5 for VQC)
    
    Returns
    -------
    dict
        Configuration information including parameter count
    """
    n_layers = reps_ansatz + 1
    n_params = n_qubits * 3 * n_layers
    
    return {
        "n_qubits": n_qubits,
        "reps_feature": 1,  # FIXED: feature map applied once
        "reps_ansatz": reps_ansatz,
        "n_layers": n_layers,
        "n_params": n_params,
        "rotation_blocks": "RX-RZ-RX (SU(2))",
        "entanglement": "Linear nearest-neighbor CNOT",
        "param_formula": f"{n_qubits} × 3 × {n_layers} = {n_params}",
    }


def print_ansatz_info(n_qubits: int, reps_ansatz: int, encoding_type: str = ""):
    """Print ansatz configuration summary."""
    info = get_ansatz_info(n_qubits, reps_ansatz)
    
    title = f"{encoding_type.upper()} " if encoding_type else ""
    print(f"\n{'='*60}")
    print(f"{title}ANSATZ V(θ) CONFIGURATION")
    print(f"{'='*60}")
    print(f"Qubits:           {info['n_qubits']}")
    print(f"reps_feature:     {info['reps_feature']} (FIXED)")
    print(f"reps_ansatz:      {info['reps_ansatz']} (configurable 1-5)")
    print(f"Total layers:     {info['n_layers']}")
    print(f"Rotation blocks:  {info['rotation_blocks']}")
    print(f"Entanglement:     {info['entanglement']}")
    print(f"Parameters:       {info['param_formula']}")
    print(f"{'='*60}\n")


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("VARIATIONAL ANSATZ COMPARISON")
    print("="*70)
    print("\nNote: reps_feature = 1 (FIXED)")
    print("      reps_ansatz ∈ {1,...,5} (configurable for VQC)")
    
    # Amplitude encoding ansatz (4 qubits)
    print("\n1. AMPLITUDE ENCODING ANSATZ (4 qubits, reps_ansatz=1)")
    print("-" * 50)
    ansatz_amp, params_amp = create_amplitude_ansatz(4, reps_ansatz=1)
    print_ansatz_info(4, 1, "amplitude")
    print("Circuit:")
    print(ansatz_amp.decompose().draw(output='text', fold=100))
    
    # Angle encoding ansatz (4 qubits for comparison)
    print("\n2. ANGLE ENCODING ANSATZ (4 qubits, reps_ansatz=2)")
    print("-" * 50)
    ansatz_angle, params_angle = create_angle_ansatz(4, reps_ansatz=2)
    print_ansatz_info(4, 2, "angle")
    print("Circuit:")
    print(ansatz_angle.decompose().draw(output='text', fold=100))
    
    # Parameter count comparison for different reps_ansatz values
    print("\n" + "="*70)
    print("PARAMETER COUNT COMPARISON (reps_ansatz ∈ {1,...,5})")
    print("="*70)
    print(f"\n{'Encoding':<15} {'Qubits':<10} {'reps_ansatz':<12} {'Layers':<10} {'Parameters':<12}")
    print("-" * 70)
    
    configs = [
        ("Amplitude", 4, 1),
        ("Amplitude", 4, 2),
        ("Amplitude", 4, 3),
        ("Angle", 16, 1),
        ("Angle", 16, 2),
        ("Angle", 16, 3),
        ("Angle", 16, 4),
        ("Angle", 16, 5),
    ]
    
    for enc, n, r in configs:
        info = get_ansatz_info(n, r)
        print(f"{enc:<15} {n:<10} {r:<12} {info['n_layers']:<10} {info['n_params']:<12}")
    
    print("-" * 70)




