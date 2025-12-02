from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.circuit import ParameterVector
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def create_vqc_with_angle_encoding(n_qubits: int, reps: int = 2):
    """
    Create a VQC with angle encoding and TwoLocal ansatz.

    Args:
        n_qubits: Number of qubits
        reps: Number of repetitions in the TwoLocal ansatz

    Returns:
        Complete VQC circuit
    """
    # Feature parameters (input data)
    x = ParameterVector("x", n_qubits)

    # Create the full circuit
    qc = QuantumCircuit(n_qubits, name="VQC")

    # 1. Angle Encoding (Feature Map)
    # Apply RY rotations with input features
    for i in range(n_qubits):
        qc.ry(x[i], i)

    qc.barrier(label="Encoding")

    # 2. Variational Ansatz (TwoLocal)
    # TwoLocal creates a parameterized circuit with:
    # - rotation_blocks: RX-RZ-RX rotation sequence per layer
    # - entanglement_blocks: CX (CNOT) gates
    # - entanglement: linear (nearest-neighbor)
    # - reps: number of layers/repetitions
    ansatz = TwoLocal(
        n_qubits,
        rotation_blocks=["rx", "rz", "rx"],  # RX(γ) → RZ(β) → RX(α)
        entanglement_blocks="cx",            # CNOT gates
        entanglement="linear",               # nearest-neighbor connectivity
        reps=reps,
        skip_final_rotation_layer=False
    )

    # Compose the ansatz onto the feature map
    qc.compose(ansatz, inplace=True)

    return qc


if __name__ == "__main__":
    # Create VQC with 16 qubits and 2 repetitions
    n_qubits = 16
    reps = 2

    print(f"Creating VQC with {n_qubits} qubits and {reps} repetitions...")
    vqc = create_vqc_with_angle_encoding(n_qubits, reps)

    print(f"\nCircuit depth: {vqc.depth()}")
    print(f"Number of parameters: {vqc.num_parameters}")
    print(f"  - Feature parameters (x): {n_qubits}")
    print(f"  - Variational parameters (theta): {vqc.num_parameters - n_qubits}")

    # Draw the circuit
    print("\nGenerating circuit diagram...")
    fig = vqc.decompose().draw(
        output='mpl',
        style='iqp',
        fold=-1  # Don't fold the circuit
    )
    plt.tight_layout()
    plt.savefig('vqc_16qubits_reps2.png', dpi=300, bbox_inches='tight')
    print("Circuit saved as 'vqc_16qubits_reps2.png'")

    # Also create a folded version for better readability
    fig2 = vqc.decompose().draw(
        output='mpl',
        style='iqp',
        fold=20  # Fold at 20 gates
    )
    plt.tight_layout()
    plt.savefig('vqc_16qubits_reps2_folded.png', dpi=300, bbox_inches='tight')
    print("Folded circuit saved as 'vqc_16qubits_reps2_folded.png'")

    # Print circuit structure
    print("\n" + "="*60)
    print("CIRCUIT STRUCTURE:")
    print("="*60)
    print("\n1. ANGLE ENCODING LAYER:")
    print("   - RY(x[i]) applied to each qubit i")
    print(f"   - {n_qubits} feature parameters: x[0] to x[{n_qubits-1}]")
    print(f"\n2. TWOLOCAL ANSATZ (reps={reps}):")
    print("   - Rotation sequence per qubit: RX(gamma) -> RZ(beta) -> RX(alpha)")
    print("   - Entanglement: CX (CNOT) gates with linear (nearest-neighbor) connectivity")
    print("   - Structure per repetition:")
    print("     a) Rotation layer: RX-RZ-RX on each qubit")
    print("     b) Entanglement layer: CX gates between adjacent qubits")
    print(f"   - With {reps} reps, you get:")
    print(f"     * {reps} rotation layers (each with 3 rotations per qubit)")
    print(f"     * {reps} entanglement layers (linear CX)")
    print(f"     * 1 final rotation layer (RX-RZ-RX)")
    print(f"   - Total variational parameters: {vqc.num_parameters - n_qubits}")
    print(f"     (3 rotations × {n_qubits} qubits × ({reps} + 1) layers = {3 * n_qubits * (reps + 1)})")
    print("="*60)
    print("\nDone!")
