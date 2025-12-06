#!/usr/bin/env python3
"""
Visualize Amplitude Encoding Feature Map

This script demonstrates how to inspect and visualize the amplitude encoding
quantum circuit used for gene expression data.
"""

import numpy as np
from amplitude_encoding import (
    amplitude_encoding_feature_map,
    build_amplitude_vqc,
    get_num_qubits
)


def visualize_feature_map(num_features: int = 16):
    """
    Visualize the amplitude encoding feature map for a given number of features.

    Parameters
    ----------
    num_features : int
        Number of input features (genes)
    """
    print("\n" + "=" * 70)
    print(f"AMPLITUDE ENCODING FEATURE MAP VISUALIZATION")
    print("=" * 70)

    # Calculate qubits needed
    n_qubits = get_num_qubits(num_features)

    print(f"\nConfiguration:")
    print(f"  Features (genes): {num_features}")
    print(f"  Qubits required:  {n_qubits}")
    print(f"  Scaling:          Logarithmic (ceil(log2({num_features})) = {n_qubits})")
    print(f"  Qubit savings:    {num_features - n_qubits} qubits saved vs angle encoding")

    # Build the feature map
    print(f"\n{'-' * 70}")
    print("Building feature map circuit...")
    feature_map, x_params, _ = amplitude_encoding_feature_map(num_features)

    print(f"\nFeature Map Details:")
    print(f"  Circuit name:     {feature_map.name}")
    print(f"  Input parameters: {len(x_params)} (x[0] to x[{len(x_params)-1}])")
    print(f"  Qubits:           {feature_map.num_qubits}")
    print(f"  Gates:            {feature_map.size()}")
    print(f"  Depth:            {feature_map.depth()}")

    # Print circuit structure
    print(f"\n{'-' * 70}")
    print("Circuit Structure:")
    print("-" * 70)
    print(feature_map.draw(output='text', fold=-1))
    print("-" * 70)

    # Show how features are distributed across qubits
    print(f"\nFeature Distribution Across Qubits:")
    print(f"{'-' * 70}")
    for qubit_idx in range(n_qubits):
        features_on_qubit = [i for i in range(num_features) if i % n_qubits == qubit_idx]
        print(f"  Qubit {qubit_idx}: Features {features_on_qubit}")
        print(f"           Gates: {', '.join([f'RY(x[{i}])' for i in features_on_qubit])}")

    # Show gate decomposition
    print(f"\n{'-' * 70}")
    print("Gate Sequence:")
    print("-" * 70)
    for i, (gate, qubits, _) in enumerate(feature_map.data):
        qubit_indices = [q._index for q in qubits]
        params = gate.params
        print(f"  {i+1:2d}. {gate.name:4s} on qubit {qubit_indices[0]} with parameter {params[0]}")

    return feature_map, x_params


def visualize_full_vqc(num_features: int = 16, reps: int = 2):
    """
    Visualize the complete VQC circuit (feature map + ansatz).

    Parameters
    ----------
    num_features : int
        Number of input features
    reps : int
        Number of repetitions for variational ansatz
    """
    print("\n" + "=" * 70)
    print(f"COMPLETE AMPLITUDE-ENCODED VQC CIRCUIT")
    print("=" * 70)

    n_qubits = get_num_qubits(num_features)

    print(f"\nConfiguration:")
    print(f"  Features:         {num_features}")
    print(f"  Qubits:           {n_qubits}")
    print(f"  Ansatz reps:      {reps}")

    # Build the complete VQC
    print(f"\nBuilding complete VQC...")
    circuit, x_params, theta_params, _ = build_amplitude_vqc(num_features, reps)

    print(f"\nCircuit Details:")
    print(f"  Input parameters (x):      {len(x_params)}")
    print(f"  Trainable parameters (θ):  {len(theta_params)}")
    print(f"  Total parameters:          {len(x_params) + len(theta_params)}")
    print(f"  Total gates:               {circuit.size()}")
    print(f"  Circuit depth:             {circuit.depth()}")

    # Print the complete circuit
    print(f"\n{'-' * 70}")
    print("Complete Circuit (Feature Map + Variational Ansatz):")
    print("-" * 70)
    print(circuit.draw(output='text', fold=-1))
    print("-" * 70)

    # Decompose to see the structure
    print(f"\nCircuit Composition:")
    print(f"  1. Feature Map U(x):    Amplitude encoding with RY rotations")
    print(f"                          {len(x_params)} input parameters")
    print(f"  2. Variational Layer:   TwoLocal ansatz with {reps} repetitions")
    print(f"                          {len(theta_params)} trainable parameters")

    return circuit, x_params, theta_params


def compare_encodings():
    """Compare amplitude encoding vs angle encoding for different feature counts."""
    print("\n" + "=" * 70)
    print("AMPLITUDE VS ANGLE ENCODING COMPARISON")
    print("=" * 70)

    feature_counts = [8, 16, 24, 32, 50, 64, 100]

    print(f"\n{'Features':<12} {'Amplitude':<15} {'Angle':<15} {'Savings':<15}")
    print("-" * 70)

    for num_features in feature_counts:
        amp_qubits = get_num_qubits(num_features)
        angle_qubits = num_features
        savings = angle_qubits - amp_qubits
        savings_pct = (savings / angle_qubits) * 100

        print(f"{num_features:<12} {amp_qubits:<15} {angle_qubits:<15} {savings} ({savings_pct:.1f}%)")

    print("\nKey Insight:")
    print("  Amplitude encoding provides logarithmic qubit scaling,")
    print("  drastically reducing qubit requirements for large feature sets.")


def visualize_with_sample_data(num_features: int = 16):
    """
    Visualize the feature map with actual sample data values.

    Parameters
    ----------
    num_features : int
        Number of features
    """
    print("\n" + "=" * 70)
    print("FEATURE MAP WITH SAMPLE DATA")
    print("=" * 70)

    # Create sample normalized data (simulating preprocessed gene expression)
    np.random.seed(42)
    sample_data = np.random.randn(num_features)
    sample_data = sample_data / np.linalg.norm(sample_data)  # Normalize

    print(f"\nSample Data (normalized gene expression values):")
    print(f"  Shape: ({num_features},)")
    print(f"  Norm:  {np.linalg.norm(sample_data):.6f} (should be ~1.0)")
    print(f"\nFirst 10 values:")
    for i in range(min(10, num_features)):
        print(f"  x[{i:2d}] = {sample_data[i]:+.6f}")

    # Build feature map
    feature_map, x_params, n_qubits = amplitude_encoding_feature_map(num_features)

    # Bind parameters
    bound_circuit = feature_map.assign_parameters({x_params[i]: sample_data[i] for i in range(num_features)})

    print(f"\n{'-' * 70}")
    print("Bound Circuit (with actual data values):")
    print("-" * 70)
    print(bound_circuit.draw(output='text', fold=-1))
    print("-" * 70)

    print(f"\nRotation Angles Applied:")
    for qubit_idx in range(n_qubits):
        features_on_qubit = [i for i in range(num_features) if i % n_qubits == qubit_idx]
        print(f"\n  Qubit {qubit_idx}:")
        for feat_idx in features_on_qubit:
            angle_rad = sample_data[feat_idx]
            angle_deg = np.degrees(angle_rad)
            print(f"    RY({angle_rad:+.4f} rad) = RY({angle_deg:+.2f}°)  [from x[{feat_idx}]]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize amplitude encoding feature maps"
    )
    parser.add_argument(
        "--features",
        type=int,
        default=16,
        help="Number of features (default: 16)"
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=2,
        help="Number of ansatz repetitions for full VQC (default: 2)"
    )
    parser.add_argument(
        "--mode",
        choices=["feature_map", "full_vqc", "compare", "sample_data", "all"],
        default="all",
        help="Visualization mode (default: all)"
    )

    args = parser.parse_args()

    try:
        if args.mode in ["feature_map", "all"]:
            visualize_feature_map(args.features)

        if args.mode in ["full_vqc", "all"]:
            visualize_full_vqc(args.features, args.reps)

        if args.mode in ["compare", "all"]:
            compare_encodings()

        if args.mode in ["sample_data", "all"]:
            visualize_with_sample_data(args.features)

        print("\n" + "=" * 70)
        print("✓ Visualization complete!")
        print("=" * 70)
        print()

    except Exception as e:
        print(f"\n[ERROR] Visualization failed: {e}")
        import traceback
        traceback.print_exc()
