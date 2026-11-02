"""Ising representation and conversion from the project's symmetric QUBO."""

from dataclasses import dataclass

import numpy as np

from .qubo import QUBOModel


@dataclass
class IsingModel:
    """Ising energy ``offset + h @ z + sum(J[i,j] * z[i] * z[j])``."""

    h: np.ndarray
    J: dict[tuple[int, int], float]
    offset: float
    n_qubits: int


def _validate_qubo(model: QUBOModel) -> tuple[np.ndarray, float]:
    matrix = np.asarray(model.matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("QUBO matrix must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("QUBO matrix must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-10):
        raise ValueError("QUBO matrix must be symmetric")
    offset = float(model.offset)
    if not np.isfinite(offset):
        raise ValueError("QUBO offset must be finite")
    return matrix, offset


def validate_ising_model(model: IsingModel) -> None:
    """Validate an Ising model before energy or circuit evaluation."""
    h = np.asarray(model.h, dtype=float)
    if (
        isinstance(model.n_qubits, bool)
        or not isinstance(model.n_qubits, (int, np.integer))
        or model.n_qubits < 0
    ):
        raise ValueError("n_qubits must be a non-negative integer")
    if h.shape != (model.n_qubits,):
        raise ValueError("Ising linear coefficients have incorrect shape")
    if not np.all(np.isfinite(h)):
        raise ValueError("Ising linear coefficients must be finite")
    if not np.isfinite(model.offset):
        raise ValueError("Ising offset must be finite")

    for pair, coefficient in model.J.items():
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(
                isinstance(index, bool)
                or not isinstance(index, (int, np.integer))
                for index in pair
            )
        ):
            raise ValueError("Ising coupling keys must be pairs of integer indices")
        i, j = pair
        if not 0 <= i < j < model.n_qubits:
            raise ValueError("Ising couplings must use canonical indices i < j")
        if not np.isfinite(coefficient):
            raise ValueError("Ising couplings must be finite")


def qubo_to_ising(model: QUBOModel) -> IsingModel:
    """Convert a symmetric QUBO to ``z_i in {-1, +1}`` Ising coefficients."""
    matrix, offset = _validate_qubo(model)
    n_qubits = matrix.shape[0]
    h = -0.5 * np.diag(matrix)
    couplings: dict[tuple[int, int], float] = {}
    ising_offset = offset + 0.5 * float(np.trace(matrix))

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            coupling = 0.5 * float(matrix[i, j])
            h[i] -= coupling
            h[j] -= coupling
            if coupling != 0:
                couplings[(i, j)] = coupling
                ising_offset += coupling

    result = IsingModel(
        h=h,
        J=couplings,
        offset=ising_offset,
        n_qubits=n_qubits,
    )
    validate_ising_model(result)
    return result


def ising_energy(model: IsingModel, spins: np.ndarray) -> float:
    """Evaluate the Ising polynomial for a vector containing only -1 and +1."""
    validate_ising_model(model)
    spins = np.asarray(spins)
    if spins.shape != (model.n_qubits,):
        raise ValueError("Spin vector length must match number of qubits")
    if not np.all(np.isin(spins, [-1, 1])):
        raise ValueError("Spin values must be -1 or +1")

    energy = float(model.offset + np.asarray(model.h) @ spins)
    for (i, j), coefficient in model.J.items():
        energy += coefficient * spins[i] * spins[j]
    return float(energy)
