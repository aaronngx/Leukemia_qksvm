"""
Diagnostic tools for verifying entanglement patterns in quantum circuits.

This module provides functions to:
- Extract CNOT connections from quantum circuits
- Verify if TwoLocal ansatz follows specified entanglement patterns
- Compare different entanglement types (linear, full, circular)
- Visualize entanglement graphs (ASCII + optional matplotlib)
"""

from typing import List, Tuple, Dict, Optional
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal

# Optional visualization dependencies
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def extract_cnot_connections(circuit: QuantumCircuit) -> List[Tuple[int, int]]:
    """
    Extract all CNOT (CX) gate connections from a quantum circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to analyze (will be decomposed if needed)

    Returns
    -------
    List[Tuple[int, int]]
        List of (control_qubit, target_qubit) pairs for each CNOT gate,
        in the order they appear in the circuit

    Examples
    --------
    >>> from qiskit.circuit.library import TwoLocal
    >>> ansatz = TwoLocal(4, rotation_blocks=['rx'], entanglement_blocks='cx',
    ...                   entanglement='linear', reps=1)
    >>> connections = extract_cnot_connections(ansatz)
    >>> print(connections)
    [(0, 1), (1, 2), (2, 3)]
    """
    # Decompose circuit to basic gates
    decomposed = circuit.decompose()

    cnot_connections = []
    for instruction in decomposed.data:
        # Filter for CNOT gates (can be named 'cx' or 'cnot')
        if instruction.operation.name in ['cx', 'cnot']:
            # Extract qubit indices
            control_idx = instruction.qubits[0]._index
            target_idx = instruction.qubits[1]._index
            cnot_connections.append((control_idx, target_idx))

    return cnot_connections


def _generate_expected_pattern(
    n_qubits: int,
    reps: int,
    pattern_type: str
) -> List[Tuple[int, int]]:
    """
    Generate expected CNOT connections for a given entanglement pattern.

    Parameters
    ----------
    n_qubits : int
        Number of qubits
    reps : int
        Number of repetitions
    pattern_type : str
        Type of entanglement: 'linear', 'full', or 'circular'

    Returns
    -------
    List[Tuple[int, int]]
        Expected (control, target) pairs
    """
    connections = []

    if pattern_type == 'linear':
        # Linear: nearest-neighbor chain
        # q0-q1, q1-q2, ..., q(n-2)-q(n-1)
        base_pattern = [(i, i+1) for i in range(n_qubits - 1)]
        connections = base_pattern * reps

    elif pattern_type == 'full':
        # Full: all-to-all connections
        # For each qubit, connect to all higher-indexed qubits
        base_pattern = []
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                base_pattern.append((i, j))
        connections = base_pattern * reps

    elif pattern_type == 'circular':
        # Circular: linear + wraparound
        # q0-q1, q1-q2, ..., q(n-2)-q(n-1), q(n-1)-q0
        base_pattern = [(i, i+1) for i in range(n_qubits - 1)]
        base_pattern.append((n_qubits - 1, 0))  # wraparound
        connections = base_pattern * reps

    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    return connections


def analyze_ansatz_entanglement(
    n_qubits: int,
    reps: int = 2,
    entanglement: str = "linear",
    rotation_blocks: Optional[List[str]] = None,
    entanglement_blocks: str = "cx"
) -> Dict:
    """
    Perform comprehensive entanglement pattern analysis on TwoLocal ansatz.

    Parameters
    ----------
    n_qubits : int
        Number of qubits
    reps : int, default=2
        Number of repetitions
    entanglement : str, default='linear'
        Entanglement pattern type ('linear', 'full', 'circular', etc.)
    rotation_blocks : List[str], optional
        Rotation gates (default: ['rx', 'rz', 'rx'])
    entanglement_blocks : str, default='cx'
        Entanglement gate type

    Returns
    -------
    dict
        Analysis results containing:
        - 'circuit': QuantumCircuit - The analyzed circuit
        - 'actual_connections': List - Actual CNOT connections found
        - 'expected_connections': List - Expected connections for pattern type
        - 'matches': bool - Whether actual matches expected
        - 'extra_connections': List - CNOTs in actual but not expected
        - 'missing_connections': List - CNOTs in expected but not actual
        - 'is_linear': bool - Whether all connections are adjacent (distance=1)
        - 'max_distance': int - Maximum qubit distance in connections
        - 'total_cnots': int - Total number of CNOT gates
        - 'depth': int - Circuit depth
        - 'gate_counts': dict - Count of each gate type

    Examples
    --------
    >>> analysis = analyze_ansatz_entanglement(n_qubits=4, reps=2, entanglement='linear')
    >>> print(f"Matches expected: {analysis['matches']}")
    >>> print(f"Is linear: {analysis['is_linear']}")
    """
    if rotation_blocks is None:
        rotation_blocks = ['rx', 'rz', 'rx']

    # Build TwoLocal circuit
    circuit = TwoLocal(
        num_qubits=n_qubits,
        rotation_blocks=rotation_blocks,
        entanglement_blocks=entanglement_blocks,
        entanglement=entanglement,
        reps=reps,
    )

    # Extract actual connections
    actual_connections = extract_cnot_connections(circuit)

    # Generate expected pattern
    try:
        expected_connections = _generate_expected_pattern(n_qubits, reps, entanglement)
    except ValueError:
        # Unknown pattern type - can't generate expected
        expected_connections = []

    # Compare actual vs expected
    actual_set = set(actual_connections)
    expected_set = set(expected_connections)

    extra = list(actual_set - expected_set)
    missing = list(expected_set - actual_set)
    matches = (len(extra) == 0 and len(missing) == 0)

    # Check if pattern is truly linear (all distances = 1)
    is_linear = True
    max_distance = 0

    if actual_connections:
        for control, target in actual_connections:
            distance = abs(control - target)
            max_distance = max(max_distance, distance)
            if distance != 1:
                is_linear = False

    # Get circuit statistics
    gate_counts = {}
    for instruction in circuit.data:
        gate_name = instruction.operation.name
        gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

    return {
        'circuit': circuit,
        'config': {
            'n_qubits': n_qubits,
            'reps': reps,
            'entanglement': entanglement,
            'rotation_blocks': rotation_blocks,
            'entanglement_blocks': entanglement_blocks,
        },
        'actual_connections': actual_connections,
        'expected_connections': expected_connections,
        'matches': matches,
        'extra_connections': extra,
        'missing_connections': missing,
        'is_linear': is_linear,
        'max_distance': max_distance,
        'total_cnots': len(actual_connections),
        'depth': circuit.depth(),
        'gate_counts': gate_counts,
    }


def print_diagnostic_report(analysis: Dict, verbose: bool = True) -> None:
    """
    Print formatted diagnostic report for entanglement analysis.

    Parameters
    ----------
    analysis : dict
        Results from analyze_ansatz_entanglement()
    verbose : bool, default=True
        If True, show detailed information; if False, show summary only

    Examples
    --------
    >>> analysis = analyze_ansatz_entanglement(n_qubits=4, reps=2)
    >>> print_diagnostic_report(analysis)
    """
    cfg = analysis['config']

    print("\n" + "="*70)
    print("ENTANGLEMENT PATTERN DIAGNOSTIC")
    print("="*70)

    # Configuration
    print("\n[Configuration]")
    print(f"  Qubits: {cfg['n_qubits']}")
    print(f"  Repetitions: {cfg['reps']}")
    print(f"  Entanglement type: {cfg['entanglement']}")
    print(f"  Entanglement blocks: {cfg['entanglement_blocks']}")
    if verbose:
        print(f"  Rotation blocks: {cfg['rotation_blocks']}")

    # Actual connections
    print("\n[Actual CNOT Connections]")
    print(f"  Total: {analysis['total_cnots']} CNOT gates")

    if verbose and analysis['actual_connections']:
        # Group by repetition for readability
        n_qubits = cfg['n_qubits']
        cnots_per_rep = (n_qubits - 1) if cfg['entanglement'] == 'linear' else None

        if cnots_per_rep and len(analysis['actual_connections']) == cnots_per_rep * cfg['reps']:
            for rep in range(cfg['reps']):
                start_idx = rep * cnots_per_rep
                end_idx = start_idx + cnots_per_rep
                rep_connections = analysis['actual_connections'][start_idx:end_idx]
                conn_str = ", ".join(f"q{c}->q{t}" for c, t in rep_connections)
                print(f"  Rep {rep+1}: {conn_str}")
        else:
            # Can't cleanly group - show first 10 and last 5
            if len(analysis['actual_connections']) <= 15:
                conn_str = ", ".join(f"q{c}->q{t}" for c, t in analysis['actual_connections'])
                print(f"  Connections: {conn_str}")
            else:
                first_10 = analysis['actual_connections'][:10]
                last_5 = analysis['actual_connections'][-5:]
                print(f"  First 10: {', '.join(f'q{c}->q{t}' for c, t in first_10)}")
                print(f"  Last 5: {', '.join(f'q{c}->q{t}' for c, t in last_5)}")
                print(f"  ... ({analysis['total_cnots']} total)")

    # Expected pattern
    if analysis['expected_connections']:
        print("\n[Expected Pattern]")
        print(f"  Pattern: {cfg['entanglement'].capitalize()}")
        print(f"  Expected CNOTs: {len(analysis['expected_connections'])}")

        if cfg['entanglement'] == 'linear':
            n_qubits = cfg['n_qubits']
            print(f"  Should be: Nearest-neighbor chain (q0->q1, q1->q2, ... q{n_qubits-2}->q{n_qubits-1})")

    # Verification results
    print("\n[Verification Results]")

    def check_mark(condition):
        return "[OK]" if condition else "[X]"

    print(f"  {check_mark(analysis['matches'])} Pattern matches expected: {'YES' if analysis['matches'] else 'NO'}")
    print(f"  {check_mark(analysis['is_linear'])} Is truly linear: {'YES' if analysis['is_linear'] else 'NO'}")
    print(f"  {check_mark(analysis['is_linear'])} All connections adjacent: {'YES' if analysis['is_linear'] else 'NO'}")
    print(f"  {check_mark(len(analysis['extra_connections']) == 0)} Extra connections: {len(analysis['extra_connections'])}")
    print(f"  {check_mark(len(analysis['missing_connections']) == 0)} Missing connections: {len(analysis['missing_connections'])}")
    print(f"  Maximum qubit distance: {analysis['max_distance']}")

    # Show mismatches if present
    if not analysis['matches'] and verbose:
        if analysis['extra_connections']:
            print(f"\n  [WARNING] Extra connections found:")
            for c, t in analysis['extra_connections']:
                print(f"    - q{c}->q{t} (distance: {abs(c-t)})")

        if analysis['missing_connections']:
            print(f"\n  [WARNING] Missing expected connections:")
            for c, t in analysis['missing_connections']:
                print(f"    - q{c}->q{t}")

    # Circuit statistics
    if verbose:
        print("\n[Circuit Statistics]")
        print(f"  Depth: {analysis['depth']}")
        print(f"  Total gates: {sum(analysis['gate_counts'].values())}")
        print(f"  Gate breakdown:")
        for gate, count in sorted(analysis['gate_counts'].items()):
            print(f"    {gate}: {count}")

    print("="*70 + "\n")


def visualize_entanglement_graph(
    connections: List[Tuple[int, int]],
    n_qubits: int,
    pattern_name: str = "",
    save_path: Optional[str] = None
) -> None:
    """
    Visualize entanglement pattern as a graph.

    Uses ASCII art for small circuits (≤8 qubits) or when matplotlib unavailable.
    Uses matplotlib/networkx for larger circuits if available.

    Parameters
    ----------
    connections : List[Tuple[int, int]]
        CNOT connections to visualize
    n_qubits : int
        Number of qubits
    pattern_name : str, optional
        Name of the pattern (for title)
    save_path : str, optional
        If provided and matplotlib available, save graph to this path

    Examples
    --------
    >>> connections = [(0, 1), (1, 2), (2, 3)]
    >>> visualize_entanglement_graph(connections, 4, "Linear")
    """
    print(f"\n[Entanglement Graph{': ' + pattern_name if pattern_name else ''}]")

    # ASCII visualization (always available)
    if n_qubits <= 8:
        # Show full graph for small circuits
        print("  ", end="")
        for i in range(n_qubits):
            print(f"q{i}", end="")
            if i < n_qubits - 1:
                # Check if there's a connection i -> i+1
                if (i, i+1) in connections:
                    print(" -> ", end="")
                else:
                    print("   ", end="")
        print()

        # Check for non-linear connections
        has_nonlinear = any(abs(c - t) != 1 for c, t in connections)
        if has_nonlinear:
            print("  [Note: Contains non-adjacent connections]")
        else:
            print("  (Linear nearest-neighbor)")
    else:
        # Summary for large circuits
        print(f"  Total qubits: {n_qubits}")
        print(f"  Total connections: {len(connections)}")
        if connections:
            print(f"  First 5: {', '.join(f'q{c}->q{t}' for c, t in connections[:5])}")
            print(f"  Last 5: {', '.join(f'q{c}->q{t}' for c, t in connections[-5:])}")

    # Matplotlib visualization (optional)
    if HAS_MATPLOTLIB and n_qubits > 8 and save_path:
        try:
            G = nx.DiGraph()
            G.add_nodes_from(range(n_qubits))
            G.add_edges_from(connections)

            plt.figure(figsize=(12, 8))
            pos = nx.circular_layout(G)

            nx.draw(G, pos,
                   with_labels=True,
                   node_color='lightblue',
                   node_size=500,
                   arrowsize=20,
                   arrowstyle='->',
                   font_size=10,
                   font_weight='bold')

            title = f"Entanglement Graph: {pattern_name}" if pattern_name else "Entanglement Graph"
            plt.title(title)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  [Graph saved to: {save_path}]")
        except Exception as e:
            print(f"  [Matplotlib visualization failed: {e}]")


def compare_entanglement_patterns(n_qubits: int, reps: int = 2) -> None:
    """
    Compare different entanglement patterns side-by-side.

    Tests linear, full, and circular patterns and displays comparison table.

    Parameters
    ----------
    n_qubits : int
        Number of qubits
    reps : int, default=2
        Number of repetitions

    Examples
    --------
    >>> compare_entanglement_patterns(n_qubits=4, reps=2)
    """
    print("\n" + "="*70)
    print(f"ENTANGLEMENT PATTERN COMPARISON ({n_qubits} qubits, {reps} reps)")
    print("="*70)

    patterns = ['linear', 'full', 'circular']
    results = {}

    for pattern in patterns:
        try:
            analysis = analyze_ansatz_entanglement(n_qubits, reps, entanglement=pattern)
            results[pattern] = analysis
        except Exception as e:
            print(f"[WARNING] Could not analyze '{pattern}' pattern: {e}")
            results[pattern] = None

    # Print comparison table
    print(f"\n{'Pattern':<12} {'Total CNOTs':<15} {'Is Linear?':<12} {'Max Distance':<15} {'Depth':<10}")
    print("-" * 70)

    for pattern in patterns:
        if results[pattern]:
            r = results[pattern]
            is_linear_str = "Yes" if r['is_linear'] else "No"
            print(f"{pattern.capitalize():<12} {r['total_cnots']:<15} {is_linear_str:<12} "
                  f"{r['max_distance']:<15} {r['depth']:<10}")
        else:
            print(f"{pattern.capitalize():<12} {'N/A':<15} {'N/A':<12} {'N/A':<15} {'N/A':<10}")

    print("\n[Key Insights]")
    if results['linear']:
        print(f"  - Linear: Nearest-neighbor only ({n_qubits-1} CNOTs per rep)")
    if results['full']:
        n_full = n_qubits * (n_qubits - 1) // 2
        print(f"  - Full: All-to-all connections ({n_full} CNOTs per rep)")
    if results['circular']:
        print(f"  - Circular: Linear + wraparound ({n_qubits} CNOTs per rep)")

    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Diagnose TwoLocal ansatz entanglement patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze 16-qubit linear pattern
  python diagnostic_entanglement.py --qubits 16 --reps 2 --entanglement linear

  # Compare all patterns for 8 qubits
  python diagnostic_entanglement.py --qubits 8 --compare

  # Quick test of 4-qubit circuit
  python diagnostic_entanglement.py --qubits 4 --reps 1
        """
    )

    parser.add_argument('--qubits', type=int, default=4,
                       help='Number of qubits (default: 4)')
    parser.add_argument('--reps', type=int, default=2,
                       help='Number of repetitions (default: 2)')
    parser.add_argument('--entanglement', type=str, default='linear',
                       help='Entanglement pattern: linear, full, circular (default: linear)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all entanglement patterns')
    parser.add_argument('--brief', action='store_true',
                       help='Show brief output instead of verbose')

    args = parser.parse_args()

    if args.compare:
        compare_entanglement_patterns(args.qubits, args.reps)
    else:
        analysis = analyze_ansatz_entanglement(args.qubits, args.reps, args.entanglement)
        print_diagnostic_report(analysis, verbose=not args.brief)
        visualize_entanglement_graph(
            analysis['actual_connections'],
            args.qubits,
            pattern_name=args.entanglement.capitalize()
        )
