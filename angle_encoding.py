#!/usr/bin/env python3
"""
Angle Encoding Feature Maps for Quantum Machine Learning.

Provides three encoding strategies:
1. Simple RY - Basic single-axis encoding (no entanglement)
2. ZZ Feature Map - ZZ interactions with data re-uploading
3. BPS Circuit - Two-axis encoding with linear entanglement
"""

from enum import Enum
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
import numpy as np


class AngleEncodingType(Enum):
    """Available angle encoding strategies."""
    SIMPLE_RY = "simple_ry"
    ZZ_FEATURE_MAP = "zz_feature_map"
    BPS_CIRCUIT = "bps_circuit"


def simple_ry_circuit(n_qubits: int):
    """
    Simple RY angle encoding - one RY gate per qubit.
    
    Structure:
        q0: ─RY(x[0])─
        q1: ─RY(x[1])─
        ...
    
    Properties:
        - Single-axis encoding: RY only
        - Entanglement: None
        - Qubits: n_features
    """
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits, name="SimpleRY")

    for i in range(n_qubits):
        qc.ry(x[i], i)

    return qc, list(x)


def zz_feature_map(n_qubits: int, reps: int = 2):
    """
    ZZ Feature Map with data re-uploading.
    
    Structure (per rep):
        1. H layer (Hadamard on all qubits)
        2. RZ layer (single-qubit rotations)
        3. ZZ layer (pairwise interactions via CNOT-RZ-CNOT)
    
    Circuit diagram (2 qubits, 1 rep):
        q0: ─H─RZ(x[0])─●────────●─
        q1: ─H─RZ(x[1])─X─RZ(φ)─X─
        
        where φ = (π - x[0])(π - x[1]) for classical ZZFeatureMap
        or φ = x[0] * x[1] for simplified version
    
    Properties:
        - Single-axis encoding: RZ only
        - Entanglement: ZZ interactions (CNOT + RZ + CNOT)
        - Data re-uploading: Via reps parameter
        - Pairwise interactions: Encoded in ZZ gates
    """
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits, name=f"ZZFeatureMap_reps{reps}")
    
    for rep in range(reps):
        # Layer 1: Hadamard on all qubits
        for i in range(n_qubits):
            qc.h(i)
        
        # Layer 2: Single-qubit RZ rotations (data encoding)
        for i in range(n_qubits):
            qc.rz(2.0 * x[i], i)  # Factor of 2 for standard ZZFeatureMap
        
        # Layer 3: ZZ entanglement (pairwise interactions)
        # Linear connectivity: (0,1), (1,2), (2,3), ...
        for i in range(n_qubits - 1):
            # ZZ interaction: exp(-i * φ * Z⊗Z)
            # Implemented as: CNOT - RZ - CNOT
            qc.cx(i, i + 1)
            # φ = 2 * (π - x[i]) * (π - x[i+1]) for Qiskit's ZZFeatureMap
            # Simplified: φ = 2 * x[i] * x[i+1]
            qc.rz(2.0 * (np.pi - x[i]) * (np.pi - x[i + 1]), i + 1)
            qc.cx(i, i + 1)
        
        # Add barrier between reps for clarity
        if rep < reps - 1:
            qc.barrier()
    
    return qc, list(x)


def bps_circuit(n_qubits: int, reps: int = 1):
    """
    BPS (Basis-Product State) Circuit with two-axis encoding.
    
    Structure (per rep):
        1. RZ layer (first axis encoding)
        2. RY layer (second axis encoding)
        3. Linear CNOT chain (entanglement)
        4. Final RZ layer (data re-uploading)
    
    Circuit diagram (3 qubits):
        q0: ─RZ(x[0])─RY(x[0])─●───────RZ(x[0])─
        q1: ─RZ(x[1])─RY(x[1])─X─●─────RZ(x[1])─
        q2: ─RZ(x[2])─RY(x[2])───X─────RZ(x[2])─
    
    Properties:
        - Two-axis encoding: RZ + RY
        - Entanglement: Linear CNOT chain
        - Data re-uploading: Explicit RZ at end
        - Pairwise interactions: Through entanglement
    """
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits, name=f"BPS_reps{reps}")
    
    for rep in range(reps):
        # Layer 1: RZ rotations (first axis)
        for i in range(n_qubits):
            qc.rz(x[i], i)
        
        # Layer 2: RY rotations (second axis)
        for i in range(n_qubits):
            qc.ry(x[i], i)
        
        # Layer 3: Linear CNOT chain (entanglement)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        
        # Layer 4: Data re-uploading via RZ
        for i in range(n_qubits):
            qc.rz(x[i], i)
        
        # Add barrier between reps for clarity
        if rep < reps - 1:
            qc.barrier()
    
    return qc, list(x)


def angle_encoding_circuit(
    n_qubits: int,
    encoding_type: AngleEncodingType = AngleEncodingType.SIMPLE_RY,
    reps: int = 2,
):
    """
    Build angle-encoding feature map with selectable strategy.
    
    Parameters
    ----------
    n_qubits : int
        Number of qubits (= number of features)
    encoding_type : AngleEncodingType
        SIMPLE_RY, ZZ_FEATURE_MAP, or BPS_CIRCUIT
    reps : int
        Number of repetitions (for ZZ and BPS circuits)
    
    Returns
    -------
    qc : QuantumCircuit
        The parameterized feature map circuit
    x_params : list
        List of parameter objects for binding feature values
    """
    if encoding_type == AngleEncodingType.ZZ_FEATURE_MAP:
        return zz_feature_map(n_qubits, reps)
    elif encoding_type == AngleEncodingType.BPS_CIRCUIT:
        return bps_circuit(n_qubits, reps)
    else:  # Default: SIMPLE_RY
        return simple_ry_circuit(n_qubits)


# Backward compatibility - default to simple RY
def get_default_angle_encoding(n_qubits: int):
    """Backward compatible function returning simple RY encoding."""
    return simple_ry_circuit(n_qubits)


if __name__ == "__main__":
    # Demo: Show all three encoding types
    n = 4  # 4 qubits for demo
    
    print("=" * 70)
    print("ANGLE ENCODING COMPARISON")
    print("=" * 70)
    
    print("\n1. SIMPLE RY (No entanglement)")
    print("-" * 40)
    qc1, params1 = simple_ry_circuit(n)
    print(f"   Qubits: {qc1.num_qubits}")
    print(f"   Parameters: {len(params1)}")
    print(f"   Depth: {qc1.depth()}")
    print(qc1.draw(output='text'))
    
    print("\n2. ZZ FEATURE MAP (ZZ interactions)")
    print("-" * 40)
    qc2, params2 = zz_feature_map(n, reps=2)
    print(f"   Qubits: {qc2.num_qubits}")
    print(f"   Parameters: {len(params2)}")
    print(f"   Depth: {qc2.depth()}")
    print(f"   Reps: 2")
    print(qc2.draw(output='text', fold=80))
    
    print("\n3. BPS CIRCUIT (Two-axis + linear CNOT)")
    print("-" * 40)
    qc3, params3 = bps_circuit(n, reps=1)
    print(f"   Qubits: {qc3.num_qubits}")
    print(f"   Parameters: {len(params3)}")
    print(f"   Depth: {qc3.depth()}")
    print(f"   Reps: 1")
    print(qc3.draw(output='text'))
    
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Encoding':<20} {'Axes':<12} {'Entanglement':<20} {'Data Re-upload':<15}")
    print("-" * 70)
    print(f"{'Simple RY':<20} {'RY':<12} {'None':<20} {'No':<15}")
    print(f"{'ZZ Feature Map':<20} {'RZ':<12} {'ZZ (CNOT-RZ-CNOT)':<20} {'Via reps':<15}")
    print(f"{'BPS Circuit':<20} {'RZ + RY':<12} {'Linear CNOT':<20} {'Explicit RZ':<15}")
