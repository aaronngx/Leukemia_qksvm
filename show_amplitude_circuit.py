#!/usr/bin/env python3
"""
Simple Amplitude Encoding Circuit Viewer

Shows the circuit structure without Unicode issues.
"""

import numpy as np
from amplitude_encoding import (
    amplitude_encoding_feature_map,
    build_amplitude_vqc,
    get_num_qubits
)


def show_circuit_info(num_features=16, reps=2, show_full_vqc=False):
    """Display circuit information and structure."""

    print("\n" + "=" * 70)
    if show_full_vqc:
        print("AMPLITUDE-ENCODED VQC CIRCUIT (Feature Map + Ansatz)")
    else:
        print("AMPLITUDE ENCODING FEATURE MAP")
    print("=" * 70)

    # Calculate qubits
    n_qubits = get_num_qubits(num_features)

    print(f"\nConfiguration:")
    print(f"  Features:         {num_features} genes")
    print(f"  Qubits needed:    {n_qubits}")
    print(f"  Scaling:          Logarithmic (ceil(log2({num_features})) = {n_qubits})")
    print(f"  Qubit savings:    {num_features - n_qubits} vs angle encoding ({((num_features-n_qubits)/num_features*100):.1f}%)")

    if show_full_vqc:
        # Build complete VQC
        circuit, x_params, theta_params, _ = build_amplitude_vqc(num_features, reps)

        print(f"\nCircuit Parameters:")
        print(f"  Input params (x):     {len(x_params)}")
        print(f"  Trainable params (θ): {len(theta_params)}")
        print(f"  Total parameters:     {len(x_params) + len(theta_params)}")
    else:
        # Build just feature map
        circuit, x_params, _ = amplitude_encoding_feature_map(num_features)

        print(f"\nCircuit Parameters:")
        print(f"  Input params (x):     {len(x_params)}")

    print(f"\nCircuit Statistics:")
    print(f"  Total gates:          {circuit.size()}")
    print(f"  Circuit depth:        {circuit.depth()}")
    print(f"  Qubits:               {circuit.num_qubits}")

    # Show gate breakdown
    print(f"\nGate Breakdown:")
    gate_counts = {}
    for gate, _, _ in circuit.data:
        gate_name = gate.name
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

    for gate_name, count in sorted(gate_counts.items()):
        print(f"  {gate_name:15s}: {count:3d}")

    # Show how features map to qubits (for feature map only)
    if not show_full_vqc:
        print(f"\nFeature Distribution Across Qubits:")
        print("-" * 70)
        for qubit_idx in range(n_qubits):
            features_on_qubit = [i for i in range(num_features) if i % n_qubits == qubit_idx]
            print(f"  Qubit {qubit_idx}: Features {features_on_qubit}")
            gates = [f"RY(x[{i}])" for i in features_on_qubit]
            print(f"            Gates: {' -> '.join(gates)}")

        print(f"\nGate Application Order:")
        print("-" * 70)
        for i in range(min(20, num_features)):  # Show first 20
            qubit = i % n_qubits
            print(f"  Step {i+1:2d}: RY(x[{i:2d}]) applied to qubit {qubit}")
        if num_features > 20:
            print(f"  ... ({num_features - 20} more gates)")

    # Try to save circuit diagram
    print(f"\n" + "=" * 70)
    print("SAVING CIRCUIT DIAGRAM...")
    print("=" * 70)

    try:
        # Save as image using matplotlib
        from qiskit.visualization import circuit_drawer
        import matplotlib.pyplot as plt

        if show_full_vqc:
            filename = f"amplitude_vqc_{num_features}features.png"
        else:
            filename = f"amplitude_featuremap_{num_features}features.png"

        fig = circuit.draw(output='mpl', style='iqp', fold=-1)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n  ✓ Circuit diagram saved to: {filename}")
        print(f"    Open this file to view the complete circuit diagram.")

    except Exception as e:
        print(f"\n  Note: Could not save as image: {e}")
        print(f"        Install matplotlib for circuit visualization:")
        print(f"        pip install matplotlib")

    # Also try text representation with ASCII only
    try:
        print(f"\n" + "=" * 70)
        print("TEXT REPRESENTATION (ASCII):")
        print("=" * 70)
        text_diagram = circuit.draw(output='text', encoding='ascii')
        print(text_diagram)
    except Exception as e:
        print(f"\n  Note: Text diagram not available: {e}")

    print("\n" + "=" * 70)


def compare_with_angle_encoding():
    """Compare amplitude vs angle encoding qubit requirements."""
    print("\n" + "=" * 70)
    print("AMPLITUDE vs ANGLE ENCODING COMPARISON")
    print("=" * 70)

    feature_counts = [8, 16, 24, 32, 50, 64, 100]

    print(f"\n{'Features':<12} {'Amplitude':<15} {'Angle':<15} {'Savings':<20}")
    print("-" * 70)

    for num_features in feature_counts:
        amp_qubits = get_num_qubits(num_features)
        angle_qubits = num_features
        savings = angle_qubits - amp_qubits
        savings_pct = (savings / angle_qubits) * 100

        print(f"{num_features:<12} {amp_qubits:<15} {angle_qubits:<15} {savings} qubits ({savings_pct:.1f}%)")

    print("\nKey Advantage:")
    print("  Amplitude encoding uses logarithmic qubit scaling,")
    print("  making it feasible to encode large feature sets on quantum hardware.")
    print()


if __name__ == "__main__":
    import sys

    # Parse arguments
    num_features = 16
    show_full = False

    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print("\nUsage:")
            print("  python show_amplitude_circuit.py [features] [--full]")
            print("\nOptions:")
            print("  features    Number of features (default: 16)")
            print("  --full      Show complete VQC (feature map + ansatz)")
            print("\nExamples:")
            print("  python show_amplitude_circuit.py")
            print("  python show_amplitude_circuit.py 16")
            print("  python show_amplitude_circuit.py 16 --full")
            print("  python show_amplitude_circuit.py --compare")
            sys.exit(0)

        if sys.argv[1] == '--compare':
            compare_with_angle_encoding()
            sys.exit(0)

        try:
            num_features = int(sys.argv[1])
        except ValueError:
            print(f"Error: Invalid number of features: {sys.argv[1]}")
            sys.exit(1)

        if len(sys.argv) > 2 and sys.argv[2] == '--full':
            show_full = True

    # Show circuit info
    show_circuit_info(num_features, reps=2, show_full_vqc=show_full)

    # Show comparison
    print()
    compare_with_angle_encoding()

    print("\n✓ Done! Check the generated PNG file for the circuit diagram.\n")
