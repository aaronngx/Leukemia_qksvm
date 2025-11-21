"""
Generate publication-quality figures for the Leukemia QKSVM report.

MUST HAVE:
- Fig 1: Angle vs Amplitude encoding comparison (K=4)
- Fig 2: VQC full circuit (encoding + ansatz, K=4)
- Fig 3: QKSVM kernel circuit U(x)U(z)† (K=4)
- Fig 4: Scaling table (K=4,8,16,32,50)

OPTIONAL:
- Fig 5: Ansatz detail (zoomed RX-RZ-RX pattern)
- Fig 6: Amplitude encoding for K=16 (show efficiency)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import TwoLocal

from angle_encoding import angle_encoding_circuit
from amplitude_encoding import get_num_qubits, encode_amplitude

# Output directory
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_circuit_as_figure(qc, filename, title=None, figsize=(12, 4), fold=80):
    """Save a quantum circuit as a matplotlib figure."""
    fig, ax = plt.subplots(figsize=figsize)

    # Draw circuit to matplotlib
    qc.draw(output='mpl', ax=ax, fold=fold, style={'fontsize': 10})

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / filename}")


def fig1_encoding_comparison():
    """
    Fig 1: Side-by-side comparison of Angle vs Amplitude encoding (K=4).
    """
    print("\n[Fig 1] Angle vs Amplitude Encoding Comparison...")

    k = 4

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Angle Encoding ---
    qc_angle, _ = angle_encoding_circuit(k)
    qc_angle.draw(output='mpl', ax=axes[0], style={'fontsize': 11})
    axes[0].set_title(f'Angle Encoding\nK={k} features -> {k} qubits',
                      fontsize=12, fontweight='bold')

    # --- Amplitude Encoding ---
    x_example = np.array([0.5, 0.3, -0.2, 0.4])  # Example 4 features
    qc_amp = encode_amplitude(x_example)
    qc_amp.draw(output='mpl', ax=axes[1], style={'fontsize': 11})
    n_qubits_amp = get_num_qubits(k)
    axes[1].set_title(f'Amplitude Encoding\nK={k} features -> {n_qubits_amp} qubits',
                      fontsize=12, fontweight='bold')

    # Add annotations
    fig.text(0.25, 0.02, 'RY(x_i) gate per feature\nDepth: 1',
             ha='center', fontsize=10, style='italic')
    fig.text(0.75, 0.02, 'State preparation |psi> = sum(x_i|i>)\nDepth: O(2^n) decomposed',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(OUTPUT_DIR / 'fig1_encoding_comparison.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'fig1_encoding_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig1_encoding_comparison.png/pdf")


def fig2_vqc_full_circuit():
    """
    Fig 2: Complete VQC circuit (feature map + ansatz) for K=4.
    """
    print("\n[Fig 2] VQC Full Circuit...")

    k = 4
    reps = 2

    # Feature map (angle encoding)
    feature_map, x_params = angle_encoding_circuit(k)

    # Ansatz
    ansatz = TwoLocal(
        num_qubits=k,
        rotation_blocks=["rx", "rz", "rx"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=reps,
    )

    # Compose full circuit
    qc = QuantumCircuit(k, name="VQC")
    qc.compose(feature_map, inplace=True)
    qc.barrier(label="Feature Map | Ansatz")
    qc.compose(ansatz, inplace=True)

    # Decompose for visualization
    qc_decomposed = qc.decompose()

    fig, ax = plt.subplots(figsize=(16, 5))
    qc_decomposed.draw(output='mpl', ax=ax, fold=100, style={'fontsize': 9})
    ax.set_title(f'VQC Circuit: Angle Encoding + TwoLocal Ansatz\n'
                 f'K={k} qubits, {reps} reps, {ansatz.num_parameters} trainable parameters',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_vqc_full_circuit.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'fig2_vqc_full_circuit.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig2_vqc_full_circuit.png/pdf")


def fig3_qksvm_kernel_circuit():
    """
    Fig 3: QKSVM kernel circuit U(x)U(z)† for computing <phi(x)|phi(z)>.
    """
    print("\n[Fig 3] QKSVM Kernel Circuit U(x)U(z)†...")

    k = 4

    # Create parameter vectors for x and z
    x_params = ParameterVector('x', k)
    z_params = ParameterVector('z', k)

    # Build U(x) - encoding for first data point
    qc = QuantumCircuit(k, name="Kernel")

    # U(x): Encode x
    for i in range(k):
        qc.ry(x_params[i], i)

    qc.barrier(label="U(x)")

    # U(z)†: Inverse encoding for second data point
    for i in range(k-1, -1, -1):
        qc.ry(-z_params[i], i)

    qc.barrier(label="U(z)†")

    # Measurement in computational basis
    qc.measure_all()

    fig, ax = plt.subplots(figsize=(14, 5))
    qc.draw(output='mpl', ax=ax, style={'fontsize': 11})
    ax.set_title('QKSVM Kernel Circuit: U(x)U(z)†\n'
                 'Kernel value K(x,z) = |<0|U(z)†U(x)|0>|²',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_qksvm_kernel_circuit.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'fig3_qksvm_kernel_circuit.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig3_qksvm_kernel_circuit.png/pdf")


def fig4_scaling_table():
    """
    Fig 4: Scaling table showing qubits/parameters for K=4,8,16,32,50.
    """
    print("\n[Fig 4] Scaling Table...")

    k_values = [4, 8, 16, 32, 50]
    reps = 2

    # Compute values
    data = []
    for k in k_values:
        # Angle encoding
        angle_qubits = k
        angle_params = k  # Feature map params (fixed during training)

        # Amplitude encoding
        amp_qubits = get_num_qubits(k)

        # Ansatz parameters (trainable)
        # Formula: 3 rotations * n_qubits * (reps + 1)
        ansatz_params_angle = 3 * k * (reps + 1)
        ansatz_params_amp = 3 * amp_qubits * (reps + 1)

        # Total trainable for each
        data.append({
            'K': k,
            'Angle Qubits': angle_qubits,
            'Amplitude Qubits': amp_qubits,
            'Angle Ansatz Params': ansatz_params_angle,
            'Amplitude Ansatz Params': ansatz_params_amp,
            'Qubit Reduction': f"{(1 - amp_qubits/angle_qubits)*100:.0f}%"
        })

    # Create table figure
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    # Table data
    columns = ['K\n(features)', 'Angle\nQubits', 'Amplitude\nQubits',
               'Angle Ansatz\nParams', 'Amplitude Ansatz\nParams', 'Qubit\nReduction']

    table_data = []
    for d in data:
        table_data.append([
            d['K'],
            d['Angle Qubits'],
            d['Amplitude Qubits'],
            d['Angle Ansatz Params'],
            d['Amplitude Ansatz Params'],
            d['Qubit Reduction']
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=['#4472C4']*6,
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#D6DCE5')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')

    ax.set_title('Quantum Resource Scaling: Angle vs Amplitude Encoding\n'
                 f'(TwoLocal Ansatz with {reps} repetitions)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add formulas
    fig.text(0.5, 0.05,
             'Angle: n_qubits = K    |    Amplitude: n_qubits = ceil(log₂K)    |    '
             'Ansatz params = 3 × n_qubits × (reps + 1)',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_scaling_table.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'fig4_scaling_table.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig4_scaling_table.png/pdf")


def fig5_ansatz_detail():
    """
    Fig 5 (Optional): Zoomed view of RX-RZ-RX ansatz pattern.
    """
    print("\n[Fig 5] Ansatz Detail (RX-RZ-RX pattern)...")

    # Create minimal ansatz to show pattern clearly
    ansatz = TwoLocal(
        num_qubits=3,
        rotation_blocks=["rx", "rz", "rx"],
        entanglement_blocks="cx",
        entanglement="linear",
        reps=1,
    )

    qc_decomposed = ansatz.decompose()

    fig, ax = plt.subplots(figsize=(14, 4))
    qc_decomposed.draw(output='mpl', ax=ax, style={'fontsize': 11})
    ax.set_title('TwoLocal Ansatz Structure (3 qubits, 1 rep)\n'
                 'Pattern: RX → RZ → RX → CNOT (linear) → repeat',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_ansatz_detail.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'fig5_ansatz_detail.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig5_ansatz_detail.png/pdf")


def fig6_amplitude_k16():
    """
    Fig 6 (Optional): Amplitude encoding for K=16 showing efficiency.
    """
    print("\n[Fig 6] Amplitude Encoding K=16...")

    k = 16
    n_qubits = get_num_qubits(k)

    # Create example
    x_example = np.random.randn(k)
    qc = encode_amplitude(x_example)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # High-level view
    qc.draw(output='mpl', ax=axes[0], style={'fontsize': 11})
    axes[0].set_title(f'Amplitude Encoding (High-level)\n'
                      f'K={k} features → {n_qubits} qubits',
                      fontsize=12, fontweight='bold')

    # Comparison diagram
    axes[1].axis('off')

    # Draw comparison boxes
    # Angle encoding representation
    rect1 = FancyBboxPatch((0.05, 0.6), 0.4, 0.3,
                            boxstyle="round,pad=0.02",
                            facecolor='#FFB3B3', edgecolor='black', linewidth=2)
    axes[1].add_patch(rect1)
    axes[1].text(0.25, 0.75, f'Angle Encoding\n{k} qubits',
                 ha='center', va='center', fontsize=12, fontweight='bold')

    # Amplitude encoding representation
    rect2 = FancyBboxPatch((0.55, 0.6), 0.4, 0.3,
                            boxstyle="round,pad=0.02",
                            facecolor='#B3FFB3', edgecolor='black', linewidth=2)
    axes[1].add_patch(rect2)
    axes[1].text(0.75, 0.75, f'Amplitude Encoding\n{n_qubits} qubits',
                 ha='center', va='center', fontsize=12, fontweight='bold')

    # Stats
    reduction = (1 - n_qubits / k) * 100
    axes[1].text(0.5, 0.4, f'Qubit Reduction: {reduction:.0f}%',
                 ha='center', fontsize=14, fontweight='bold', color='green')
    axes[1].text(0.5, 0.25, f'K={k} features encoded in {n_qubits} qubits\n'
                            f'using quantum state amplitudes',
                 ha='center', fontsize=11)

    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Qubit Efficiency Comparison', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_amplitude_k16.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'fig6_amplitude_k16.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig6_amplitude_k16.png/pdf")


def fig7_scaling_chart():
    """
    Fig 7: Visual chart showing scaling comparison.
    """
    print("\n[Fig 7] Scaling Chart...")

    k_values = [4, 8, 16, 32, 50, 64, 100]

    angle_qubits = k_values
    amplitude_qubits = [get_num_qubits(k) for k in k_values]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Qubit comparison
    ax1 = axes[0]
    x = np.arange(len(k_values))
    width = 0.35

    bars1 = ax1.bar(x - width/2, angle_qubits, width, label='Angle Encoding', color='#4472C4')
    bars2 = ax1.bar(x + width/2, amplitude_qubits, width, label='Amplitude Encoding', color='#ED7D31')

    ax1.set_xlabel('Number of Features (K)', fontsize=11)
    ax1.set_ylabel('Number of Qubits', fontsize=11)
    ax1.set_title('Qubit Requirements by Encoding Type', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(k_values)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    # Scaling curves
    ax2 = axes[1]
    k_continuous = np.linspace(4, 100, 100)
    angle_curve = k_continuous
    amplitude_curve = np.ceil(np.log2(k_continuous))

    ax2.plot(k_continuous, angle_curve, 'b-', linewidth=2, label='Angle: n = K')
    ax2.plot(k_continuous, amplitude_curve, 'r-', linewidth=2, label='Amplitude: n = ceil(log₂K)')
    ax2.fill_between(k_continuous, amplitude_curve, angle_curve, alpha=0.3, color='green',
                     label='Qubit savings')

    ax2.set_xlabel('Number of Features (K)', fontsize=11)
    ax2.set_ylabel('Number of Qubits', fontsize=11)
    ax2.set_title('Scaling: Linear vs Logarithmic', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xlim(4, 100)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig7_scaling_chart.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'fig7_scaling_chart.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: fig7_scaling_chart.png/pdf")


def main():
    print("="*60)
    print("GENERATING REPORT FIGURES")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR.absolute()}")

    # MUST HAVE figures
    print("\n" + "-"*40)
    print("MUST HAVE FIGURES")
    print("-"*40)

    fig1_encoding_comparison()
    fig2_vqc_full_circuit()
    fig3_qksvm_kernel_circuit()
    fig4_scaling_table()

    # OPTIONAL figures
    print("\n" + "-"*40)
    print("OPTIONAL FIGURES")
    print("-"*40)

    fig5_ansatz_detail()
    fig6_amplitude_k16()
    fig7_scaling_chart()

    print("\n" + "="*60)
    print("ALL FIGURES GENERATED!")
    print("="*60)
    print(f"\nFiles saved to: {OUTPUT_DIR.absolute()}")
    print("\nMUST HAVE:")
    print("  - fig1_encoding_comparison.png/pdf")
    print("  - fig2_vqc_full_circuit.png/pdf")
    print("  - fig3_qksvm_kernel_circuit.png/pdf")
    print("  - fig4_scaling_table.png/pdf")
    print("\nOPTIONAL:")
    print("  - fig5_ansatz_detail.png/pdf")
    print("  - fig6_amplitude_k16.png/pdf")
    print("  - fig7_scaling_chart.png/pdf")


if __name__ == "__main__":
    main()
