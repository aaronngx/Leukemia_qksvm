"""
Circuit Visualization and Verification Script.

Verifies:
1. Circuit structure (gate types, order, layers)
2. Encoding layer (RY for angle, state prep for amplitude)
3. Entanglement pattern (linear vs full)
4. Parameter count
5. Circuit depth
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal

from angle_encoding import angle_encoding_circuit
from amplitude_encoding import get_num_qubits, amplitude_encoding_feature_map
from diagnostic_entanglement import analyze_ansatz_entanglement, print_diagnostic_report


def analyze_circuit(qc: QuantumCircuit, name: str):
    """Analyze and print circuit statistics."""
    print(f"\n{'='*60}")
    print(f"CIRCUIT: {name}")
    print('='*60)

    print(f"\n[Structure]")
    print(f"  Qubits: {qc.num_qubits}")
    print(f"  Depth: {qc.depth()}")
    print(f"  Total gates: {qc.size()}")

    # Count gate types
    gate_counts = {}
    for instruction in qc.data:
        gate_name = instruction.operation.name
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

    print(f"\n[Gate Counts]")
    for gate, count in sorted(gate_counts.items()):
        print(f"  {gate}: {count}")

    # Count parameters
    num_params = qc.num_parameters
    print(f"\n[Parameters]")
    print(f"  Total parameters: {num_params}")

    # List parameter names
    if num_params > 0 and num_params <= 20:
        names = [str(p.name).encode('ascii', errors='replace').decode('ascii') for p in qc.parameters]
        print(f"  Parameter names: {names}")
    elif num_params > 20:
        params = list(qc.parameters)
        first_10 = [str(p.name).encode('ascii', errors='replace').decode('ascii') for p in params[:10]]
        last_10 = [str(p.name).encode('ascii', errors='replace').decode('ascii') for p in params[-10:]]
        print(f"  First 10: {first_10}")
        print(f"  Last 10: {last_10}")

    return gate_counts


def visualize_angle_encoding(k: int = 16):
    """Visualize angle encoding circuit."""
    print("\n" + "="*70)
    print(f"ANGLE ENCODING (K={k} features -> {k} qubits)")
    print("="*70)

    qc, x_params = angle_encoding_circuit(k)

    analyze_circuit(qc, f"Angle Encoding (K={k})")

    print(f"\n[Circuit Diagram]")
    try:
        if k <= 8:
            diagram = qc.draw(output='text', fold=120)
            print(str(diagram).encode('ascii', errors='replace').decode('ascii'))
        else:
            print(f"  (Circuit too large to display, showing first 4 qubits)")
            qc_small, _ = angle_encoding_circuit(4)
            diagram = qc_small.draw(output='text', fold=120)
            print(str(diagram).encode('ascii', errors='replace').decode('ascii'))
    except Exception as e:
        print(f"  (Diagram display error: {e})")

    return qc


def visualize_amplitude_encoding(k: int = 16):
    """Visualize amplitude encoding circuit."""
    n_qubits = get_num_qubits(k)

    print("\n" + "="*70)
    print(f"AMPLITUDE ENCODING (K={k} features -> {n_qubits} qubits)")
    print("="*70)

    # Create amplitude encoding feature map
    qc, _ = amplitude_encoding_feature_map(k)

    # Decompose to see actual gates
    qc_decomposed = qc.decompose().decompose()

    analyze_circuit(qc, f"Amplitude Encoding (K={k})")

    print(f"\n[Note] Amplitude encoding uses qc.initialize() which creates")
    print(f"       a state preparation circuit. Decomposed depth: {qc_decomposed.depth()}")

    print(f"\n[Circuit Diagram (high-level)]")
    try:
        diagram = qc.draw(output='text', fold=120)
        print(str(diagram).encode('ascii', errors='replace').decode('ascii'))
    except Exception as e:
        print(f"  (Diagram display error: {e})")

    return qc


def visualize_vqc_ansatz(n_qubits: int = 4, reps: int = 2):
    """Visualize VQC ansatz (TwoLocal)."""
    print("\n" + "="*70)
    print(f"VQC ANSATZ (TwoLocal, {n_qubits} qubits, {reps} reps)")
    print("="*70)

    ansatz = TwoLocal(
        num_qubits=n_qubits,
        rotation_blocks=["rx", "rz", "rx"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=reps,
    )

    analyze_circuit(ansatz, f"TwoLocal Ansatz")

    print(f"\n[Configuration]")
    print(f"  Rotation blocks: ['rx', 'rz', 'rx']")
    print(f"  Entanglement: cx (CNOT)")
    print(f"  Entanglement pattern: linear (nearest-neighbor)")
    print(f"  Repetitions: {reps}")

    print(f"\n[Circuit Diagram]")
    try:
        diagram = ansatz.draw(output='text', fold=120)
        print(str(diagram).encode('ascii', errors='replace').decode('ascii'))
    except Exception as e:
        print(f"  (Diagram display error: {e})")

    # Show decomposed
    print(f"\n[Decomposed Circuit]")
    try:
        diagram = ansatz.decompose().draw(output='text', fold=120)
        print(str(diagram).encode('ascii', errors='replace').decode('ascii'))
    except Exception as e:
        print(f"  (Diagram display error: {e})")

    # Add entanglement pattern verification
    print(f"\n[Entanglement Pattern Verification]")
    analysis = analyze_ansatz_entanglement(n_qubits, reps, entanglement="linear")
    print_diagnostic_report(analysis, verbose=False)  # Brief summary

    return ansatz


def visualize_full_vqc(n_features: int = 4, reps: int = 2):
    """Visualize complete VQC (feature map + ansatz)."""
    print("\n" + "="*70)
    print(f"FULL VQC CIRCUIT ({n_features} features, {reps} reps)")
    print("="*70)

    # Feature map (angle encoding)
    feature_map, x_params = angle_encoding_circuit(n_features)

    # Ansatz
    ansatz = TwoLocal(
        num_qubits=n_features,
        rotation_blocks=["rx", "rz", "rx"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=reps,
    )

    # Compose
    qc = QuantumCircuit(n_features, name="VQC")
    qc.compose(feature_map, inplace=True)
    qc.barrier()
    qc.compose(ansatz, inplace=True)

    analyze_circuit(qc, f"Full VQC")

    print(f"\n[Structure]")
    print(f"  Feature map parameters (x): {len(x_params)}")
    print(f"  Ansatz parameters (theta): {ansatz.num_parameters}")
    print(f"  Total trainable: {ansatz.num_parameters}")

    print(f"\n[Circuit Diagram]")
    try:
        diagram = qc.draw(output='text', fold=120)
        print(str(diagram).encode('ascii', errors='replace').decode('ascii'))
    except Exception as e:
        print(f"  (Diagram display error: {e})")

    return qc


def verify_parameter_counts():
    """Verify parameter counts match expected values."""
    print("\n" + "="*70)
    print("PARAMETER COUNT VERIFICATION")
    print("="*70)

    print(f"\n{'K':<6} {'Encoding':<12} {'Qubits':<8} {'Params':<10} {'Formula'}")
    print("-" * 60)

    for k in [4, 8, 16, 32]:
        # Angle encoding
        qc_angle, x_params = angle_encoding_circuit(k)
        n_angle = len(x_params)
        print(f"{k:<6} {'Angle':<12} {k:<8} {n_angle:<10} K")

        # Amplitude encoding
        n_qubits_amp = get_num_qubits(k)
        print(f"{k:<6} {'Amplitude':<12} {n_qubits_amp:<8} {'N/A':<10} ceil(log2(K))")

        # VQC ansatz (for angle encoding)
        ansatz = TwoLocal(
            num_qubits=k,
            rotation_blocks=["rx", "rz", "rx"],
            entanglement_blocks="cx",
            entanglement="linear",
            reps=2,
        )
        n_ansatz = ansatz.num_parameters
        # Formula: 3 rotations * K qubits * (reps + 1) layers
        expected = 3 * k * 3  # 3 rotation types, K qubits, 3 layers (initial + 2 reps)
        print(f"{'':<6} {'+ Ansatz':<12} {k:<8} {n_ansatz:<10} 3*K*(reps+1)={expected}")
        print()


def main():
    print("\n" + "#"*70)
    print("#" + " "*20 + "CIRCUIT VISUALIZATION" + " "*21 + "#")
    print("#"*70)

    # 1. Angle encoding
    visualize_angle_encoding(k=4)

    # 2. Amplitude encoding
    visualize_amplitude_encoding(k=16)

    # 3. VQC Ansatz
    visualize_vqc_ansatz(n_qubits=4, reps=2)

    # 4. Full VQC
    visualize_full_vqc(n_features=4, reps=2)

    # 5. Parameter count verification
    verify_parameter_counts()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Angle Encoding:
  - Uses RY(x_i) gates on each qubit
  - K features -> K qubits
  - Depth: 1 (just RY gates)
  - Parameters: K (one per feature)

Amplitude Encoding:
  - Encodes features as state amplitudes |psi> = sum x_i|i>
  - K features -> ceil(log2(K)) qubits
  - Uses qc.initialize() for state preparation
  - Depth: O(2^n) when decomposed

VQC Ansatz (TwoLocal):
  - Rotation blocks: RX-RZ-RX sequence
  - Entanglement: CNOT (cx) gates
  - Pattern: Linear (nearest-neighbor)
  - Parameters: 3 * K * (reps + 1)
""")


if __name__ == "__main__":
    main()
