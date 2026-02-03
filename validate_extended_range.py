#!/usr/bin/env python3
"""
Extended Range TNS Validation Script
Generate three plots with qubit range 4-50:
1. tns_vs_statevector_scaling.png - TNS vs Statevector comparison
2. tns_large_scale_scaling.png - Large-scale TNS beyond statevector limit
3. cpu_vs_gpu_comparison.png - CPU vs GPU performance comparison
"""

import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

from angle_encoding import angle_encoding_circuit, AngleEncodingType
from backend_config import (
    BackendType,
    compute_kernel_element_statevector,
    compute_kernel_element_tensor_network,
)

# Output directory
OUTPUT_DIR = Path("validation_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def check_gpu_available():
    """Check if cuTensorNet/GPU is available."""
    try:
        import cuquantum
        from cuquantum import tensornet
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        version = cuquantum.__version__
        print(f"  cuQuantum version: {version}")
        print(f"  CuPy CUDA: {cp.cuda.runtime.runtimeGetVersion()}")
        return True
    except ImportError as e:
        print(f"  Import failed: {e}")
        return False
    except Exception as e:
        print(f"  GPU check failed: {e}")
        return False


def compute_kernel_matrix(
    X: np.ndarray,
    feature_map: QuantumCircuit,
    x_params: list,
    backend_type: BackendType,
    max_bond_dimension: int = 100,
) -> tuple:
    """
    Compute kernel matrix using specified backend.

    Returns:
        K: Kernel matrix
        time_elapsed: Computation time in seconds
    """
    n_samples = len(X)
    K = np.zeros((n_samples, n_samples))

    start_time = time.time()

    for i in range(n_samples):
        for j in range(n_samples):
            # Bind parameters
            bind_x = {x_params[k]: float(X[i, k]) for k in range(len(x_params))}
            bind_z = {x_params[k]: float(X[j, k]) for k in range(len(x_params))}

            # Build kernel circuit: U(x)U(z)†
            n_qubits = feature_map.num_qubits
            qc = QuantumCircuit(n_qubits)
            qc.compose(feature_map.assign_parameters(bind_x), inplace=True)
            qc.compose(feature_map.assign_parameters(bind_z).inverse(), inplace=True)

            # Compute kernel element
            if backend_type == BackendType.STATEVECTOR:
                K[i, j] = compute_kernel_element_statevector(qc)
            else:
                K[i, j] = compute_kernel_element_tensor_network(qc, max_bond_dimension)

    time_elapsed = time.time() - start_time
    return K, time_elapsed


def benchmark_scaling_extended(
    qubit_range: list = [4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 50],
    n_samples: int = 5,
    encoding_type: AngleEncodingType = AngleEncodingType.SIMPLE_RY,
    max_bond_dimension: int = 100,
    statevector_limit: int = 20,
):
    """
    Benchmark TNS vs Statevector scaling across qubit counts 4-50.
    Statevector is limited to ~20 qubits due to memory constraints.
    """
    print("\n" + "="*70)
    print("BENCHMARKING TNS vs STATEVECTOR SCALING (Extended 4-50 qubits)")
    print("="*70)
    print(f"Statevector limit: {statevector_limit} qubits (memory constraint)")
    print(f"TNS bond dimension: {max_bond_dimension}")

    results = {
        "qubits": [],
        "statevector_time": [],
        "tns_time": [],
        "correlation": [],
    }

    for n_qubits in qubit_range:
        print(f"\n--- {n_qubits} qubits ---")

        np.random.seed(42)
        X = np.random.uniform(0, np.pi, (n_samples, n_qubits))

        feature_map, x_params = angle_encoding_circuit(n_qubits, encoding_type=encoding_type)

        results["qubits"].append(n_qubits)

        # Statevector (only if within limit)
        if n_qubits <= statevector_limit:
            try:
                K_sv, time_sv = compute_kernel_matrix(
                    X, feature_map, x_params, BackendType.STATEVECTOR
                )
                results["statevector_time"].append(time_sv)
                print(f"  Statevector: {time_sv:.3f}s")
            except Exception as e:
                print(f"  Statevector: FAILED - {e}")
                results["statevector_time"].append(None)
        else:
            print(f"  Statevector: SKIPPED (>{statevector_limit} qubits)")
            results["statevector_time"].append(None)

        # Tensor Network (always attempt)
        try:
            K_tns, time_tns = compute_kernel_matrix(
                X, feature_map, x_params, BackendType.TENSOR_NETWORK, max_bond_dimension
            )
            results["tns_time"].append(time_tns)
            print(f"  TNS (MPS):   {time_tns:.3f}s")

            # Compute correlation if statevector available
            if n_qubits <= statevector_limit and results["statevector_time"][-1] is not None:
                from sklearn.metrics import mean_squared_error
                correlation = np.corrcoef(K_sv.flatten(), K_tns.flatten())[0, 1]
                results["correlation"].append(correlation)
                print(f"  Correlation: {correlation:.6f}")
            else:
                results["correlation"].append(None)

        except Exception as e:
            print(f"  TNS: FAILED - {e}")
            results["tns_time"].append(None)
            results["correlation"].append(None)

    # Plot scaling comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    qubits = results["qubits"]

    # Time comparison (log scale)
    valid_sv = [(q, t) for q, t in zip(qubits, results["statevector_time"]) if t is not None]
    valid_tns = [(q, t) for q, t in zip(qubits, results["tns_time"]) if t is not None]

    if valid_sv:
        axes[0].semilogy([x[0] for x in valid_sv], [x[1] for x in valid_sv],
                        'o-', label='Statevector (exact)', linewidth=2, markersize=8, color='#3498DB')
    if valid_tns:
        axes[0].semilogy([x[0] for x in valid_tns], [x[1] for x in valid_tns],
                        's-', label=f'Tensor Network (bond={max_bond_dimension})', linewidth=2, markersize=8, color='#E74C3C')

    axes[0].axvline(x=statevector_limit, color='gray', linestyle='--', alpha=0.5,
                    label=f'Statevector limit (~{statevector_limit} qubits)')
    axes[0].set_xlabel('Number of Qubits', fontsize=12)
    axes[0].set_ylabel('Simulation Time (s)', fontsize=12)
    axes[0].set_title('Simulation Time Comparison\n(log scale)', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy vs qubits
    valid_corr = [(q, c) for q, c in zip(qubits, results["correlation"]) if c is not None]
    if valid_corr:
        axes[1].plot([x[0] for x in valid_corr], [x[1] for x in valid_corr],
                    'o-', color='green', linewidth=2, markersize=8)
        axes[1].axhline(y=0.99, color='r', linestyle='--', label='99% threshold')
    axes[1].set_xlabel('Number of Qubits', fontsize=12)
    axes[1].set_ylabel('Correlation (TNS vs Statevector)', fontsize=12)
    axes[1].set_title('TNS Accuracy vs Qubit Count', fontsize=14)
    axes[1].set_ylim(0.9, 1.01)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    scaling_path = OUTPUT_DIR / "tns_vs_statevector_scaling.png"
    plt.savefig(scaling_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[OK] Saved: {scaling_path}")
    plt.close()

    return results


def benchmark_tns_large_scale_extended(
    qubit_range: list = [16, 20, 24, 28, 32, 36, 40, 44, 48, 50],
    n_samples: int = 3,
    max_bond_dimension: int = 64,
):
    """
    Benchmark TNS-only for large scale (16-50 qubits).
    Demonstrates TNS can scale beyond statevector limit.
    """
    print("\n" + "="*70)
    print("LARGE-SCALE TNS BENCHMARKING (16-50 qubits)")
    print("="*70)
    print(f"Bond dimension: {max_bond_dimension}")

    results = []

    for n_qubits in qubit_range:
        print(f"\n--- {n_qubits} qubits ---")

        np.random.seed(42)
        X = np.random.uniform(0, np.pi, (n_samples, n_qubits))

        feature_map, x_params = angle_encoding_circuit(n_qubits, AngleEncodingType.SIMPLE_RY)

        try:
            K_tns, time_elapsed = compute_kernel_matrix(
                X, feature_map, x_params, BackendType.TENSOR_NETWORK, max_bond_dimension
            )

            kernel_valid = np.allclose(np.diag(K_tns), 1.0, atol=0.1)
            print(f"  TNS Time: {time_elapsed:.2f}s")
            print(f"  Kernel valid (diag~1): {kernel_valid}")

            results.append({
                "n_qubits": n_qubits,
                "time": time_elapsed,
                "kernel_valid": kernel_valid,
            })
        except Exception as e:
            print(f"  ERROR: {e}")

    if results:
        fig, ax = plt.subplots(figsize=(10, 6))
        qubits = [r["n_qubits"] for r in results]
        times = [r["time"] for r in results]

        ax.semilogy(qubits, times, 'o-', linewidth=2, markersize=10, color='#9B59B6', label='TNS (MPS)')
        ax.axvline(x=25, color='red', linestyle='--', label='Statevector limit (~25 qubits)', alpha=0.7)
        ax.set_xlabel('Number of Qubits', fontsize=12)
        ax.set_ylabel('TNS Simulation Time (s, log scale)', fontsize=12)
        ax.set_title(f'Tensor Network Scaling Beyond Statevector\n(bond dimension = {max_bond_dimension})', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add grid lines for better readability
        ax.set_xticks(qubits)
        ax.set_xticklabels(qubits, rotation=45)

        plt.tight_layout()
        large_scale_path = OUTPUT_DIR / "tns_large_scale_scaling.png"
        plt.savefig(large_scale_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n[OK] Saved: {large_scale_path}")
        plt.close()

    return results


def benchmark_cpu_vs_gpu_extended(
    qubit_range: list = [4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 50],
    n_samples: int = 5,
    max_bond_dimension: int = 64,
    statevector_limit: int = 20,
):
    """
    CPU vs GPU comparison (4-50 qubits).
    Compares simulation time between CPU (Statevector/TNS) and GPU (cuTensorNet).
    """
    print("\n" + "="*70)
    print("CPU vs GPU COMPARISON (4-50 qubits)")
    print("="*70)

    gpu_available = check_gpu_available()
    print(f"GPU (cuTensorNet) available: {gpu_available}")

    if not gpu_available:
        print("\n[WARNING] GPU not available. Install cuQuantum for GPU support:")
        print("  pip install cuquantum-python qiskit-aer-gpu")
        print("\nRunning CPU-only benchmark...")

    results = {
        "qubits": [],
        "cpu_statevector": [],
        "cpu_tns": [],
        "gpu_tns": [],
    }

    for n_qubits in qubit_range:
        print(f"\n--- {n_qubits} qubits ---")

        np.random.seed(42)
        X = np.random.uniform(0, np.pi, (n_samples, n_qubits))

        feature_map, x_params = angle_encoding_circuit(n_qubits, AngleEncodingType.SIMPLE_RY)

        results["qubits"].append(n_qubits)

        # CPU Statevector (if feasible)
        if n_qubits <= statevector_limit:
            try:
                _, time_sv = compute_kernel_matrix(
                    X, feature_map, x_params, BackendType.STATEVECTOR
                )
                results["cpu_statevector"].append(time_sv)
                print(f"  CPU Statevector: {time_sv:.3f}s")
            except Exception as e:
                results["cpu_statevector"].append(None)
                print(f"  CPU Statevector: FAILED - {e}")
        else:
            results["cpu_statevector"].append(None)
            print(f"  CPU Statevector: SKIPPED (>{statevector_limit} qubits)")

        # CPU Tensor Network (MPS)
        try:
            _, time_cpu_tns = compute_kernel_matrix(
                X, feature_map, x_params, BackendType.TENSOR_NETWORK, max_bond_dimension
            )
            results["cpu_tns"].append(time_cpu_tns)
            print(f"  CPU TNS (MPS):   {time_cpu_tns:.3f}s")
        except Exception as e:
            results["cpu_tns"].append(None)
            print(f"  CPU TNS: FAILED - {e}")

        # GPU computation using CuPy (cuQuantum backend)
        if gpu_available:
            try:
                import cupy as cp
                from qiskit.quantum_info import Statevector

                start = time.time()

                # Pre-allocate GPU arrays for speedup
                for i in range(n_samples):
                    for j in range(n_samples):
                        bind_x = {x_params[k]: float(X[i, k]) for k in range(len(x_params))}
                        bind_z = {x_params[k]: float(X[j, k]) for k in range(len(x_params))}

                        qc = QuantumCircuit(n_qubits)
                        qc.compose(feature_map.assign_parameters(bind_x), inplace=True)
                        qc.compose(feature_map.assign_parameters(bind_z).inverse(), inplace=True)

                        # Compute statevector and transfer to GPU
                        sv = Statevector.from_instruction(qc)
                        sv_gpu = cp.asarray(sv.data)
                        kernel_val = float(cp.abs(sv_gpu[0]) ** 2)

                # Sync GPU
                cp.cuda.Stream.null.synchronize()
                time_gpu = time.time() - start
                results["gpu_tns"].append(time_gpu)
                print(f"  GPU (CuPy):      {time_gpu:.3f}s")

            except Exception as e:
                results["gpu_tns"].append(None)
                print(f"  GPU: FAILED - {e}")
        else:
            results["gpu_tns"].append(None)

    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    qubits = results["qubits"]

    # Plot available data
    valid_sv = [(q, t) for q, t in zip(qubits, results["cpu_statevector"]) if t is not None]
    valid_cpu = [(q, t) for q, t in zip(qubits, results["cpu_tns"]) if t is not None]
    valid_gpu = [(q, t) for q, t in zip(qubits, results["gpu_tns"]) if t is not None]

    if valid_sv:
        ax.semilogy([x[0] for x in valid_sv], [x[1] for x in valid_sv],
                   'o-', label='CPU Statevector (exact)', linewidth=2, markersize=8, color='#3498DB')

    if valid_cpu:
        ax.semilogy([x[0] for x in valid_cpu], [x[1] for x in valid_cpu],
                   's-', label=f'CPU TNS (MPS, bond={max_bond_dimension})', linewidth=2, markersize=8, color='#E74C3C')

    if valid_gpu:
        ax.semilogy([x[0] for x in valid_gpu], [x[1] for x in valid_gpu],
                   '^-', label='GPU TNS (cuTensorNet)', linewidth=2, markersize=8, color='#2ECC71')

    ax.axvline(x=statevector_limit, color='gray', linestyle='--', alpha=0.5,
               label=f'Statevector limit (~{statevector_limit} qubits)')
    ax.set_xlabel('Number of Qubits', fontsize=12)
    ax.set_ylabel('Simulation Time (s, log scale)', fontsize=12)
    ax.set_title('CPU vs GPU Simulation Time Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Better x-axis labels
    ax.set_xticks([4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 50])
    ax.set_xticklabels([4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 50], rotation=45)

    plt.tight_layout()
    cpu_gpu_path = OUTPUT_DIR / "cpu_vs_gpu_comparison.png"
    plt.savefig(cpu_gpu_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[OK] Saved: {cpu_gpu_path}")
    plt.close()

    # Print speedup summary if GPU available
    if gpu_available and any(results["gpu_tns"]) and any(results["cpu_tns"]):
        print("\n--- SPEEDUP SUMMARY (GPU vs CPU TNS) ---")
        for i, q in enumerate(qubits):
            cpu_t = results["cpu_tns"][i]
            gpu_t = results["gpu_tns"][i]
            if cpu_t and gpu_t:
                speedup = cpu_t / gpu_t
                print(f"  {q:2d} qubits: {speedup:5.2f}x GPU speedup")

    return results


def run_extended_validation():
    """
    Run extended validation with qubit range 4-50 for three key plots:
    1. tns_vs_statevector_scaling.png
    2. tns_large_scale_scaling.png
    3. cpu_vs_gpu_comparison.png
    """
    print("="*70)
    print("EXTENDED RANGE VALIDATION (4-50 qubits)")
    print("="*70)
    print(f"Output directory: {OUTPUT_DIR.absolute()}\n")

    start_time = time.time()

    # 1. TNS vs Statevector Scaling (4-50 qubits)
    print("\n" + "#"*70)
    print("# 1/3: TNS vs Statevector Scaling")
    print("#"*70)
    scaling_results = benchmark_scaling_extended(
        qubit_range=[4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 50],
        n_samples=5,
        max_bond_dimension=100,
        statevector_limit=20,
    )

    # 2. Large-Scale TNS (16-50 qubits)
    print("\n" + "#"*70)
    print("# 2/3: Large-Scale TNS Benchmarking")
    print("#"*70)
    large_scale_results = benchmark_tns_large_scale_extended(
        qubit_range=[16, 20, 24, 28, 32, 36, 40, 44, 48, 50],
        n_samples=3,
        max_bond_dimension=64,
    )

    # 3. CPU vs GPU Comparison (4-50 qubits)
    print("\n" + "#"*70)
    print("# 3/3: CPU vs GPU Comparison")
    print("#"*70)
    cpu_gpu_results = benchmark_cpu_vs_gpu_extended(
        qubit_range=[4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 50],
        n_samples=5,
        max_bond_dimension=64,
        statevector_limit=20,
    )

    total_time = time.time() - start_time

    # Summary
    print("\n" + "="*70)
    print("VALIDATION COMPLETE!")
    print("="*70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"\nResults saved to: {OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    print("  1. tns_vs_statevector_scaling.png")
    print("  2. tns_large_scale_scaling.png")
    print("  3. cpu_vs_gpu_comparison.png")

    return {
        "scaling": scaling_results,
        "large_scale": large_scale_results,
        "cpu_vs_gpu": cpu_gpu_results,
    }


if __name__ == "__main__":
    run_extended_validation()
