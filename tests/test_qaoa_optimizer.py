import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from quantum_portfolio.data import MarketData
from quantum_portfolio.ising import qubo_to_ising
from quantum_portfolio.optimizer import evaluate_portfolios
from quantum_portfolio.qaoa_circuit import build_qaoa_circuit
from quantum_portfolio.qaoa_optimizer import (
    optimize_qaoa,
    sample_qaoa,
    statevector_expected_energy,
)
from quantum_portfolio.qubo import build_qubo, recommended_penalty


def _sample_problem():
    market_data = MarketData.sample_data()
    penalty = recommended_penalty(market_data, 2, 0.5)
    qubo = build_qubo(market_data, 2, 0.5, penalty)
    return market_data, qubo, qubo_to_ising(qubo)


def test_statevector_expected_energy_matches_independent_probability_sum():
    _, _, ising = _sample_problem()
    gammas = [0.21]
    betas = [-0.14]
    circuit = build_qaoa_circuit(ising, gammas, betas)
    state = Statevector.from_instruction(circuit)
    probabilities = np.abs(state.data) ** 2
    direct = 0.0
    for basis_index, probability in enumerate(probabilities):
        spins = np.array(
            [-1 if (basis_index >> q) & 1 else 1 for q in range(ising.n_qubits)]
        )
        energy = ising.offset + ising.h @ spins
        energy += sum(
            coefficient * spins[i] * spins[j]
            for (i, j), coefficient in ising.J.items()
        )
        direct += probability * energy
    assert statevector_expected_energy(ising, gammas, betas) == pytest.approx(direct)


def test_qaoa_optimization_is_reproducible_for_fixed_seed():
    _, _, ising = _sample_problem()
    first = optimize_qaoa(ising, seed=13, maxiter=20)
    second = optimize_qaoa(ising, seed=13, maxiter=20)

    assert first.gammas == pytest.approx(second.gammas)
    assert first.betas == pytest.approx(second.betas)
    assert first.expected_energy == pytest.approx(second.expected_energy)
    assert first.initial_gammas == pytest.approx(second.initial_gammas)
    assert first.initial_betas == pytest.approx(second.initial_betas)
    assert np.isfinite(first.expected_energy)
    assert first.nfev > 0
    assert first.shots is None
    assert first.package_versions["qiskit"]


def test_sample_qaoa_counts_and_portfolio_metrics():
    market_data, qubo, ising = _sample_problem()
    _, best = evaluate_portfolios(market_data, 2, 0.5)
    optimized = optimize_qaoa(ising, seed=3, maxiter=8)
    sampled = sample_qaoa(
        ising,
        optimized,
        shots=128,
        seed=3,
        market_data=market_data,
        k=2,
        risk_aversion=0.5,
        qubo_model=qubo,
        optimal_selections=[best.selection],
    )

    assert sampled.total_shots == 128
    assert 0 <= sampled.feasible_shots <= 128
    assert 0 <= sampled.feasible_probability <= 1
    assert sampled.optimal_probability is not None
    assert sampled.simulator == "Qiskit Aer (ideal shot simulation)"
    assert sampled.seed == 3
    assert sampled.unique_bitstrings == len(sampled.outcomes)
    assert sum(outcome.count for outcome in sampled.outcomes) == 128
    assert all(
        outcome.objective is None or np.isfinite(outcome.objective)
        for outcome in sampled.outcomes
    )


def test_sample_qaoa_is_reproducible_with_same_simulator_seed():
    market_data, qubo, ising = _sample_problem()
    optimized = optimize_qaoa(ising, seed=2, maxiter=5)
    args = dict(
        ising_model=ising,
        optimization_result=optimized,
        shots=64,
        seed=42,
        market_data=market_data,
        k=2,
        risk_aversion=0.5,
        qubo_model=qubo,
    )
    first = sample_qaoa(**args)
    second = sample_qaoa(**args)
    assert [(row.bitstring, row.count) for row in first.outcomes] == [
        (row.bitstring, row.count) for row in second.outcomes
    ]
