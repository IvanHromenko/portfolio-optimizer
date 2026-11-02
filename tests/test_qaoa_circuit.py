import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantum_portfolio.ising import IsingModel
from quantum_portfolio.qaoa_circuit import (
    append_cost_layer,
    build_qaoa_circuit,
)
from quantum_portfolio.qaoa_results import decode_bitstring


def test_qaoa_circuit_width_depth_and_measurements():
    model = IsingModel(
        h=np.array([0.2, -0.3]),
        J={(0, 1): 0.4},
        offset=1.1,
        n_qubits=2,
    )
    circuit = build_qaoa_circuit(model, [0.1, 0.2], [0.3, 0.4], measure=True)

    assert circuit.num_qubits == 2
    assert circuit.num_clbits == 2
    assert circuit.count_ops()["h"] == 2
    assert circuit.count_ops()["rx"] == 4
    assert circuit.count_ops()["measure"] == 2
    assert circuit.count_ops()["cx"] == 4


def test_zero_parameters_preserve_uniform_state():
    model = IsingModel(
        h=np.array([0.2, -0.3]),
        J={(0, 1): 0.4},
        offset=1.1,
        n_qubits=2,
    )
    circuit = build_qaoa_circuit(model, [0.0], [0.0])
    probabilities = Statevector.from_instruction(circuit).probabilities()
    assert probabilities == pytest.approx(np.full(4, 0.25))


def test_cost_layer_basis_state_phases_match_ising_energy():
    model = IsingModel(
        h=np.array([0.25, -0.5]),
        J={(0, 1): 0.75},
        offset=-0.3,
        n_qubits=2,
    )
    gamma = 0.37

    for basis_index in range(4):
        circuit = QuantumCircuit(2)
        for qubit in range(2):
            if (basis_index >> qubit) & 1:
                circuit.x(qubit)
        append_cost_layer(circuit, model, gamma)
        amplitude = Statevector.from_instruction(circuit).data[basis_index]
        spins = np.array(
            [-1 if (basis_index >> qubit) & 1 else 1 for qubit in range(2)]
        )
        expected = np.exp(
            -1j * gamma * (model.offset + model.h @ spins + model.J[(0, 1)] * spins[0] * spins[1])
        )
        assert amplitude == pytest.approx(expected * np.exp(1j * gamma * model.offset))


@pytest.mark.parametrize(
    ("displayed", "expected"),
    [
        ("0101", [1, 0, 1, 0]),
        ("0010", [0, 1, 0, 0]),
        ("0001", [1, 0, 0, 0]),
    ],
)
def test_decode_qiskit_count_order_to_asset_order(displayed, expected):
    assert decode_bitstring(displayed, 4).tolist() == expected


def test_decoder_rejects_bad_bitstrings():
    with pytest.raises(ValueError):
        decode_bitstring("012", 3)
    with pytest.raises(ValueError):
        decode_bitstring("01", 3)


@pytest.mark.parametrize(
    ("gammas", "betas"),
    [([], []), ([0.1], []), ([np.nan], [0.2])],
)
def test_qaoa_circuit_rejects_invalid_parameters(gammas, betas):
    model = IsingModel(h=np.zeros(1), J={}, offset=0.0, n_qubits=1)
    with pytest.raises(ValueError):
        build_qaoa_circuit(model, gammas, betas)
