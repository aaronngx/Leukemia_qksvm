from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def angle_encoding_circuit(n_qubits: int):
    """
    Build an angle-encoding feature map U(x) with RY rotations.

    Parameters
    ----------
    n_qubits : int
        Number of qubits / features.

    Returns
    -------
    qc : QuantumCircuit
        The parameterized encoding circuit.
    x_params : list[Parameter]
        The list of parameters (one per qubit).
    """
    x = ParameterVector("x", n_qubits)
    qc = QuantumCircuit(n_qubits, name="AngleEncoding")

    for i in range(n_qubits):
        qc.ry(x[i], i)

    return qc, list(x)