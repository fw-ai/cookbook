"""Quantum circuit optimizer using iterative commutation-aware CNOT cancellation.

Exploits three commutation rules to identify and remove redundant CNOT pairs:
1. Phase/diagonal gates (Rz, T, S, Z) on the CONTROL qubit of a CNOT commute with it.
2. X-type gates (X) on the TARGET qubit of a CNOT commute with it.
3. Gates on qubits disjoint from the CNOT trivially commute.

When a pair of identical CNOTs is separated only by gates that commute with them,
the pair cancels to identity. Iterating until convergence handles cascading patterns
where removing one pair exposes new cancellable pairs.
"""

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

# Import cancellation passes with fallbacks for different Qiskit versions
try:
    from qiskit.transpiler.passes import CommutativeInverseCancellation
    _comm_pass = CommutativeInverseCancellation
except ImportError:
    from qiskit.transpiler.passes import CommutativeCancellation
    _comm_pass = CommutativeCancellation

from qiskit.transpiler.passes import CXCancellation


def _count_multi_qubit(circuit):
    return sum(1 for inst in circuit.data if inst.operation.num_qubits >= 2)


def optimize(qasm_str: str) -> str:
    """Optimize a quantum circuit to minimize multi-qubit gate count.

    Args:
        qasm_str: An OpenQASM 2.0 circuit string.

    Returns:
        An optimized OpenQASM 2.0 circuit string that is unitarily equivalent
        to the input but has fewer multi-qubit gates.
    """
    circuit = QuantumCircuit.from_qasm_str(qasm_str)

    prev_count = float("inf")
    max_iterations = 40

    for _ in range(max_iterations):
        passes = [CXCancellation(), _comm_pass()]
        pm = PassManager(passes)
        circuit = pm.run(circuit)

        count = _count_multi_qubit(circuit)
        if count >= prev_count:
            break
        prev_count = count

    # Output as QASM
    try:
        return circuit.qasm()
    except (AttributeError, DeprecationWarning):
        from qiskit.qasm2 import dumps
        return dumps(circuit)
