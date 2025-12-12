#!/usr/bin/env python3
"""Demo script to visualize amplitude encoding circuit."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def demonstrate_amplitude_encoding():
    """Show amplitude encoding circuits for different feature sizes."""

    print("\n" + "="*80)
    print("AMPLITUDE ENCODING CIRCUIT DEMONSTRATION")
    print("="*80)

    # Example 1: 4 features (2 qubits)
    print("\n" + "-"*80)
    print("EXAMPLE 1: 4 Features -> 2 Qubits")
    print("-"*80)

    x_4 = np.array([0.5, 0.5, 0.5, 0.5])  # Already normalized
    n_qubits_4 = 2

    qc_4 = QuantumCircuit(n_qubits_4, name="U(x)")
    qc_4.initialize(x_4, range(n_qubits_4))

    print(f"\nInput features: {x_4}")
    print(f"Number of qubits: {n_qubits_4}")
    print(f"State dimension: 2^{n_qubits_4} = {2**n_qubits_4}")
    print(f"\nDecomposed circuit depth: {qc_4.decompose().depth()}")
    print(f"Decomposed circuit size: {qc_4.decompose().size()} gates")

    print("\nCircuit (high-level view):")
    print(qc_4.draw(output='text'))

    print("\nCircuit (decomposed - showing actual gates):")
    print(qc_4.decompose().draw(output='text', fold=90))

    # Verify the encoding
    sv = Statevector.from_instruction(qc_4)
    print(f"\nVerification:")
    print(f"  Input amplitudes:  {x_4}")
    print(f"  Output amplitudes: {np.abs(sv.data)}")
    print(f"  Match: {np.allclose(x_4, np.abs(sv.data))}")

    # Example 2: 8 features (3 qubits)
    print("\n" + "-"*80)
    print("EXAMPLE 2: 8 Features -> 3 Qubits")
    print("-"*80)

    x_8 = np.array([0.353, 0.353, 0.353, 0.353, 0.353, 0.353, 0.353, 0.353])
    n_qubits_8 = 3

    qc_8 = QuantumCircuit(n_qubits_8, name="U(x)")
    qc_8.initialize(x_8, range(n_qubits_8))

    print(f"\nInput features: {x_8}")
    print(f"Number of qubits: {n_qubits_8}")
    print(f"State dimension: 2^{n_qubits_8} = {2**n_qubits_8}")
    print(f"\nDecomposed circuit depth: {qc_8.decompose().depth()}")
    print(f"Decomposed circuit size: {qc_8.decompose().size()} gates")

    print("\nCircuit (high-level view):")
    print(qc_8.draw(output='text'))

    # Example 3: 16 features (4 qubits) - TYPICAL FOR THIS PROJECT
    print("\n" + "-"*80)
    print("EXAMPLE 3: 16 Features -> 4 Qubits (TYPICAL FOR LEUKEMIA DATASET)")
    print("-"*80)

    np.random.seed(42)
    x_16_raw = np.random.randn(16)
    x_16 = x_16_raw / np.linalg.norm(x_16_raw)  # L2 normalize
    n_qubits_16 = 4

    qc_16 = QuantumCircuit(n_qubits_16, name="U(x)")
    qc_16.initialize(x_16, range(n_qubits_16))

    print(f"\nNumber of qubits: {n_qubits_16}")
    print(f"State dimension: 2^{n_qubits_16} = {2**n_qubits_16}")
    print(f"\nInput features (first 4): {x_16[:4]}")
    print(f"L2 norm: {np.linalg.norm(x_16):.6f}")
    print(f"\nDecomposed circuit depth: {qc_16.decompose().depth()}")
    print(f"Decomposed circuit size: {qc_16.decompose().size()} gates")

    print("\nCircuit (high-level view):")
    print(qc_16.draw(output='text'))

    print("\n" + "="*80)
    print("KEY CONCEPTS")
    print("="*80)
    print("""
1. LOGARITHMIC SCALING:
   - 4 features  -> 2 qubits  (2^2 = 4)
   - 8 features  -> 3 qubits  (2^3 = 8)
   - 16 features -> 4 qubits  (2^4 = 16)
   - 50 features -> 6 qubits  (2^6 = 64, pad with 14 zeros)

2. STATE PREPARATION:
   |psi(x)> = sum(x_i |i>)
   where x_i are the normalized feature values

3. MOTTONEN DECOMPOSITION:
   - Qiskit's initialize() uses state preparation algorithms
   - Decomposes into RY, RZ, and CNOT gates
   - Circuit depth: O(2^n) for n qubits
   - Circuit size: ~4 * 2^n gates

4. NORMALIZATION REQUIRED:
   - Quantum state must satisfy: sum(|x_i|^2) = 1
   - Use StandardScaler then L2-normalize

5. ADVANTAGES:
   - Logarithmic qubit efficiency
   - Encodes exponentially more data per qubit
   - 16 features need only 4 qubits!

6. DISADVANTAGES:
   - Deep circuit (exponential gates)
   - State preparation is complex
   - Harder to implement on NISQ devices
    """)

if __name__ == "__main__":
    demonstrate_amplitude_encoding()
