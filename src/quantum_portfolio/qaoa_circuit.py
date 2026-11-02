"""Qiskit circuit construction for the standard X-mixer QAOA ansatz."""

from collections.abc import Sequence

import numpy as np
from qiskit import QuantumCircuit

from .ising import IsingModel, validate_ising_model


def append_cost_layer(
    circuit: QuantumCircuit,
    model: IsingModel,
    gamma: float,
) -> None:
    """Append ``exp(-i * gamma * (h.Z + J.ZZ))`` to a circuit."""
    validate_ising_model(model)
    if circuit.num_qubits != model.n_qubits:
        raise ValueError("Circuit width must match the Ising model")
    if not np.isfinite(gamma):
        raise ValueError("gamma must be finite")

    for i, coefficient in enumerate(model.h):
        if coefficient != 0:
            circuit.rz(2 * float(gamma) * float(coefficient), i)
    for (i, j), coefficient in model.J.items():
        circuit.cx(i, j)
        circuit.rz(2 * float(gamma) * coefficient, j)
        circuit.cx(i, j)


def append_mixer_layer(
    circuit: QuantumCircuit,
    n_qubits: int,
    beta: float,
) -> None:
    """Append the standard X-mixer ``exp(-i * beta * sum_i X_i)``."""
    if circuit.num_qubits != n_qubits:
        raise ValueError("Circuit width must match the requested mixer width")
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")
    for qubit in range(n_qubits):
        circuit.rx(2 * float(beta), qubit)


def _parameter_array(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def build_qaoa_circuit(
    ising_model: IsingModel,
    gammas: Sequence[float],
    betas: Sequence[float],
    measure: bool = False,
) -> QuantumCircuit:
    """Build a depth-p QAOA circuit; displayed counts reverse asset-index order."""
    validate_ising_model(ising_model)
    if ising_model.n_qubits < 1:
        raise ValueError("QAOA requires at least one qubit")
    gammas_array = _parameter_array(gammas, "gammas")
    betas_array = _parameter_array(betas, "betas")
    if gammas_array.size != betas_array.size:
        raise ValueError("gammas and betas must have the same depth")

    circuit = QuantumCircuit(
        ising_model.n_qubits,
        ising_model.n_qubits if measure else 0,
    )
    circuit.h(range(ising_model.n_qubits))
    for gamma, beta in zip(gammas_array, betas_array):
        append_cost_layer(circuit, ising_model, float(gamma))
        append_mixer_layer(circuit, ising_model.n_qubits, float(beta))

    if measure:
        circuit.measure(range(ising_model.n_qubits), range(ising_model.n_qubits))
    return circuit
