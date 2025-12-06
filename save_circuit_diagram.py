#!/usr/bin/env python3
"""Save amplitude encoding circuit diagram as image."""

from amplitude_encoding import amplitude_encoding_feature_map, build_amplitude_vqc

def save_feature_map_diagram(num_features=16):
    """Save feature map circuit as PNG image."""
    print(f"\nGenerating amplitude encoding circuit for {num_features} features...")

    # Build circuit
    circuit, x_params, n_qubits = amplitude_encoding_feature_map(num_features)

    print(f"  Qubits: {n_qubits}")
    print(f"  Gates: {circuit.size()}")
    print(f"  Depth: {circuit.depth()}")

    # Save as image
    try:
        from qiskit.visualization import circuit_drawer
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Draw and save
        fig = circuit.draw(output='mpl', style='iqp')
        filename = f'amplitude_featuremap_{num_features}features.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"\nSUCCESS: Circuit diagram saved to: {filename}")
        print(f"Open the PNG file to view the complete circuit diagram.\n")
        return filename

    except ImportError:
        print("\nERROR: matplotlib not installed.")
        print("Install it with: pip install matplotlib\n")
        return None
    except Exception as e:
        print(f"\nERROR: Could not save diagram: {e}\n")
        return None


def save_full_vqc_diagram(num_features=16, reps=2):
    """Save complete VQC (feature map + ansatz) as PNG image."""
    print(f"\nGenerating complete VQC circuit ({num_features} features, {reps} reps)...")

    # Build circuit
    circuit, x_params, theta_params, n_qubits = build_amplitude_vqc(num_features, reps)

    print(f"  Qubits: {n_qubits}")
    print(f"  Input params: {len(x_params)}")
    print(f"  Trainable params: {len(theta_params)}")
    print(f"  Gates: {circuit.size()}")
    print(f"  Depth: {circuit.depth()}")

    # Save as image
    try:
        from qiskit.visualization import circuit_drawer
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Draw and save
        fig = circuit.draw(output='mpl', style='iqp')
        filename = f'amplitude_vqc_{num_features}features_{reps}reps.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"\nSUCCESS: Circuit diagram saved to: {filename}")
        print(f"Open the PNG file to view the complete VQC circuit.\n")
        return filename

    except ImportError:
        print("\nERROR: matplotlib not installed.")
        print("Install it with: pip install matplotlib\n")
        return None
    except Exception as e:
        print(f"\nERROR: Could not save diagram: {e}\n")
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        # Save complete VQC
        save_full_vqc_diagram(num_features=16, reps=2)
    else:
        # Save just feature map
        save_feature_map_diagram(num_features=16)
