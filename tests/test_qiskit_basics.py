import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def test_hadamard_prepares_one_qubit_superposition():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    state = Statevector.from_instruction(circuit)

    assert np.abs(state.data) ** 2 == pytest.approx([0.5, 0.5])


def test_bell_state_has_only_correlated_basis_outcomes():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    state = Statevector.from_instruction(circuit)

    assert np.abs(state.data) ** 2 == pytest.approx([0.5, 0.0, 0.0, 0.5])


def test_four_qubit_hadamard_state_is_uniform():
    circuit = QuantumCircuit(4)
    circuit.h(range(4))
    state = Statevector.from_instruction(circuit)

    assert np.abs(state.data) ** 2 == pytest.approx(np.full(16, 1 / 16))
