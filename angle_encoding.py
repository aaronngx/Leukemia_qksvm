from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def angle_encoding_circuit(n_qubits: int):
    """Build angle-encoding feature map U(x) with RY rotations."""
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits, name="AngleEncoding")

    for i in range(n_qubits):
        qc.ry(x[i], i)

    return qc, list(x)