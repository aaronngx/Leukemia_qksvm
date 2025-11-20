"""Backend configuration for quantum simulations.

Supports both exact statevector simulation and tensor network simulation (MPS)
for scaling to larger qubit counts.
"""
from enum import Enum
from typing import Any

import numpy as np
from qiskit import QuantumCircuit


class BackendType(Enum):
    """Available backend types for quantum simulation."""
    STATEVECTOR = "statevector"
    TENSOR_NETWORK = "tensor_network"  # MPS via Aer


def get_statevector_backend():
    """Get reference statevector backend from qiskit.quantum_info."""
    from qiskit.quantum_info import Statevector
    return Statevector


def get_tensor_network_backend(max_bond_dimension: int = 100):
    """
    Get tensor network (Matrix Product State) simulator from Aer.

    Parameters
    ----------
    max_bond_dimension : int
        Maximum bond dimension for MPS. Higher values give better accuracy
        but use more memory. Typical values: 50-1000.
        - 50-100: Fast, moderate accuracy
        - 100-500: Balanced
        - 500+: High accuracy, slower

    Returns
    -------
    AerSimulator configured for matrix_product_state method
    """
    try:
        from qiskit_aer import AerSimulator
        return AerSimulator(
            method='matrix_product_state',
            matrix_product_state_max_bond_dimension=max_bond_dimension
        )
    except ImportError:
        raise ImportError(
            "qiskit-aer is not installed. Please install it with:\n"
            "pip install qiskit-aer"
        )


def get_estimator_for_backend(
    backend_type: BackendType,
    max_bond_dimension: int = 100
):
    """
    Get appropriate Estimator primitive for the specified backend.

    Parameters
    ----------
    backend_type : BackendType
        Type of backend to use
    max_bond_dimension : int
        For tensor network backend, the maximum bond dimension

    Returns
    -------
    Estimator primitive
    """
    if backend_type == BackendType.STATEVECTOR:
        from qiskit.primitives import Estimator
        return Estimator()

    elif backend_type == BackendType.TENSOR_NETWORK:
        try:
            from qiskit_aer.primitives import Estimator as AerEstimator
            backend = get_tensor_network_backend(max_bond_dimension)
            return AerEstimator(backend=backend)
        except ImportError:
            raise ImportError(
                "qiskit-aer is not installed. Please install it with:\n"
                "pip install qiskit-aer"
            )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def compute_kernel_element_statevector(qc: QuantumCircuit) -> float:
    """
    Compute kernel element using statevector simulation.

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to simulate (should be U(x) U(z)†)

    Returns
    -------
    float
        Kernel value |<0|ψ>|²
    """
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_instruction(qc)
    return float(np.abs(sv[0]) ** 2)


def compute_kernel_element_tensor_network(
    qc: QuantumCircuit,
    max_bond_dimension: int = 100
) -> float:
    """
    Compute kernel element using tensor network (MPS) simulation.

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to simulate (should be U(x) U(z)†)
    max_bond_dimension : int
        Maximum bond dimension for MPS approximation

    Returns
    -------
    float
        Approximate kernel value |<0|ψ>|²
    """
    try:
        from qiskit_aer import AerSimulator

        backend = AerSimulator(
            method='matrix_product_state',
            matrix_product_state_max_bond_dimension=max_bond_dimension
        )

        # Add measurement to get probability of |0...0>
        qc_copy = qc.copy()
        qc_copy.measure_all()

        job = backend.run(qc_copy, shots=10000)
        result = job.result()
        counts = result.get_counts()

        # Get probability of all-zeros state
        n_qubits = qc.num_qubits
        zero_state = '0' * n_qubits
        prob = counts.get(zero_state, 0) / 10000

        return float(prob)
    except ImportError:
        raise ImportError(
            "qiskit-aer is not installed. Please install it with:\n"
            "pip install qiskit-aer"
        )


def get_backend_info(backend_type: BackendType, n_qubits: int) -> dict:
    """
    Get information about backend capabilities and memory requirements.

    Parameters
    ----------
    backend_type : BackendType
        Backend type
    n_qubits : int
        Number of qubits

    Returns
    -------
    dict
        Information about the backend
    """
    memory_gb = (2 ** n_qubits) * 16 / (1024 ** 3)  # Complex128 = 16 bytes

    info = {
        'backend_type': backend_type.value,
        'n_qubits': n_qubits,
        'state_dimension': 2 ** n_qubits,
    }

    if backend_type == BackendType.STATEVECTOR:
        info['memory_required_gb'] = memory_gb
        info['exact_simulation'] = True
        info['recommended_max_qubits'] = 20
        info['feasible'] = memory_gb < 32  # Assuming 32GB RAM limit
    else:  # Tensor network
        info['memory_required_gb'] = "Variable (depends on bond dimension)"
        info['exact_simulation'] = False
        info['recommended_max_qubits'] = 100
        info['feasible'] = True

    return info


def recommend_backend(n_qubits: int, available_memory_gb: float = 16.0) -> BackendType:
    """
    Recommend backend based on number of qubits and available memory.

    Parameters
    ----------
    n_qubits : int
        Number of qubits
    available_memory_gb : float
        Available RAM in GB

    Returns
    -------
    BackendType
        Recommended backend
    """
    required_memory = (2 ** n_qubits) * 16 / (1024 ** 3)

    if required_memory < available_memory_gb * 0.5:  # Use 50% safety margin
        return BackendType.STATEVECTOR
    else:
        return BackendType.TENSOR_NETWORK


def print_backend_recommendation(n_qubits: int, available_memory_gb: float = 16.0):
    """
    Print backend recommendation with explanation.

    Parameters
    ----------
    n_qubits : int
        Number of qubits
    available_memory_gb : float
        Available RAM in GB
    """
    recommended = recommend_backend(n_qubits, available_memory_gb)
    info = get_backend_info(recommended, n_qubits)

    print("="*70)
    print("BACKEND RECOMMENDATION")
    print("="*70)
    print(f"Number of qubits: {n_qubits}")
    print(f"Available memory: {available_memory_gb:.1f} GB")
    print(f"Recommended backend: {recommended.value}")
    print(f"State dimension: {info['state_dimension']:,}")

    if isinstance(info['memory_required_gb'], float):
        print(f"Memory required: {info['memory_required_gb']:.2f} GB")
        print(f"Feasible: {'Yes' if info['feasible'] else 'No (use tensor network)'}")
    else:
        print(f"Memory required: {info['memory_required_gb']}")

    print(f"Exact simulation: {'Yes' if info['exact_simulation'] else 'No (approximate)'}")
    print("="*70)
