"""Exact-statevector QAOA optimization and finite-shot Aer sampling."""

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from typing import Optional, Sequence

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from .ising import IsingModel, ising_energy, validate_ising_model
from .qaoa_circuit import build_qaoa_circuit
from .qaoa_results import QAOASamplingResult, evaluate_counts
from .data import MarketData
from .qubo import QUBOModel


@dataclass
class QAOAOptimizationResult:
    initial_gammas: np.ndarray
    initial_betas: np.ndarray
    gammas: np.ndarray
    betas: np.ndarray
    expected_energy: float
    success: bool
    status: int
    message: str
    nfev: int
    nit: int
    depth: int
    simulator: str
    shots: Optional[int]
    seed: int
    package_versions: dict[str, str]


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def _validate_depth_seed(depth: int, seed: int) -> tuple[int, int]:
    if isinstance(depth, bool) or not isinstance(depth, Integral) or depth < 1:
        raise ValueError("depth must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(depth), int(seed)


def statevector_expected_energy(
    ising_model: IsingModel,
    gammas: Sequence[float],
    betas: Sequence[float],
) -> float:
    """Compute the exact Ising expectation from an ideal Qiskit statevector."""
    validate_ising_model(ising_model)
    circuit = build_qaoa_circuit(ising_model, gammas, betas, measure=False)
    statevector = Statevector.from_instruction(circuit)
    probabilities = np.abs(np.asarray(statevector.data)) ** 2
    expected_energy = 0.0
    for basis_index, probability in enumerate(probabilities):
        if probability == 0:
            continue
        spins = np.fromiter(
            (
                -1 if (basis_index >> qubit) & 1 else 1
                for qubit in range(ising_model.n_qubits)
            ),
            dtype=int,
            count=ising_model.n_qubits,
        )
        expected_energy += probability * ising_energy(ising_model, spins)
    return float(expected_energy)


def optimize_qaoa(
    ising_model: IsingModel,
    depth: int = 1,
    seed: int = 7,
    maxiter: int = 300,
    initial_parameters: Optional[Sequence[float]] = None,
) -> QAOAOptimizationResult:
    """Minimize the exact expected energy with deterministic-seed COBYLA."""
    validate_ising_model(ising_model)
    depth, seed = _validate_depth_seed(depth, seed)
    if isinstance(maxiter, bool) or not isinstance(maxiter, Integral) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")

    if initial_parameters is None:
        rng = np.random.default_rng(seed)
        initial = np.concatenate(
            [rng.uniform(-np.pi, np.pi, depth), rng.uniform(-np.pi / 2, np.pi / 2, depth)]
        )
    else:
        initial = np.asarray(initial_parameters, dtype=float)
        if initial.shape != (2 * depth,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_parameters must contain 2 * depth finite values")

    def expected_energy(parameters: np.ndarray) -> float:
        return statevector_expected_energy(
            ising_model,
            parameters[:depth],
            parameters[depth:],
        )

    result = minimize(
        expected_energy,
        initial,
        method="COBYLA",
        options={"maxiter": int(maxiter), "rhobeg": 0.5, "tol": 1e-4},
    )
    return QAOAOptimizationResult(
        initial_gammas=np.asarray(initial[:depth], dtype=float),
        initial_betas=np.asarray(initial[depth:], dtype=float),
        gammas=np.asarray(result.x[:depth], dtype=float),
        betas=np.asarray(result.x[depth:], dtype=float),
        expected_energy=float(result.fun),
        success=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        nfev=int(result.nfev),
        nit=int(getattr(result, "nit", result.nfev)),
        depth=depth,
        simulator="Qiskit Statevector (exact)",
        shots=None,
        seed=seed,
        package_versions={
            "qiskit": _package_version("qiskit"),
            "qiskit-aer": _package_version("qiskit-aer"),
            "scipy": _package_version("scipy"),
        },
    )


def sample_qaoa(
    ising_model: IsingModel,
    optimization_result: QAOAOptimizationResult,
    shots: int,
    seed: int,
    market_data: MarketData,
    k: int,
    risk_aversion: float,
    qubo_model: QUBOModel,
    optimal_selections: Optional[list[np.ndarray]] = None,
) -> QAOASamplingResult:
    """Sample optimized angles on local Aer and evaluate portfolio outcomes."""
    validate_ising_model(ising_model)
    if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 1:
        raise ValueError("shots must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if optimization_result.depth < 1:
        raise ValueError("Optimization result depth must be positive")

    circuit = build_qaoa_circuit(
        ising_model,
        optimization_result.gammas,
        optimization_result.betas,
        measure=True,
    )
    simulator = AerSimulator()
    counts = simulator.run(
        circuit,
        shots=int(shots),
        seed_simulator=int(seed),
    ).result().get_counts()
    return evaluate_counts(
        counts=counts,
        market_data=market_data,
        k=k,
        risk_aversion=risk_aversion,
        qubo_model=qubo_model,
        optimal_selections=optimal_selections,
        simulator="Qiskit Aer (ideal shot simulation)",
        seed=int(seed),
    )
