import numpy as np
import pytest

from quantum_portfolio.data import MarketData
from quantum_portfolio.optimizer import evaluate_portfolios
from quantum_portfolio.portfolio import objective
from quantum_portfolio.qubo import (
    build_qubo,
    evaluate_qubo,
    recommended_penalty,
)


def test_qubo_matches_penalized_objective_for_all_four_asset_bitstrings():
    market_data = MarketData.sample_data()
    k = 2
    risk_aversion = 0.5
    penalty = recommended_penalty(market_data, k, risk_aversion)
    model = build_qubo(market_data, k, risk_aversion, penalty)

    assert np.allclose(model.matrix, model.matrix.T)
    assert model.offset == pytest.approx(penalty * k**2)

    for bits in range(1 << len(market_data.assets)):
        selection = np.array(
            [(bits >> index) & 1 for index in range(len(market_data.assets))]
        )
        original_objective = objective(
            market_data.returns,
            market_data.covariance,
            selection,
            risk_aversion,
        )
        expected = original_objective + penalty * (selection.sum() - k) ** 2
        assert evaluate_qubo(model, selection) == pytest.approx(expected)


def test_recommended_penalty_makes_every_qubo_minimizer_feasible():
    market_data = MarketData.sample_data()
    penalty = recommended_penalty(market_data, k=2, risk_aversion=0.5)
    model = build_qubo(market_data, k=2, risk_aversion=0.5, penalty=penalty)

    energies = []
    selections = []
    for bits in range(1 << len(market_data.assets)):
        selection = np.array(
            [(bits >> index) & 1 for index in range(len(market_data.assets))]
        )
        selections.append(selection)
        energies.append(evaluate_qubo(model, selection))

    minimum_energy = min(energies)
    minimizers = [
        selection
        for selection, energy in zip(selections, energies)
        if energy == pytest.approx(minimum_energy)
    ]
    _, classical_best = evaluate_portfolios(market_data, k=2, risk_aversion=0.5)

    assert all(selection.sum() == 2 for selection in minimizers)
    assert min(energies) == pytest.approx(classical_best.objective)


@pytest.mark.parametrize(
    ("k", "risk_aversion", "penalty"),
    [(1.5, 0.5, 1.0), (2, -0.1, 1.0), (2, 0.5, 0), (2, 0.5, np.inf)],
)
def test_qubo_builder_rejects_invalid_parameters(k, risk_aversion, penalty):
    with pytest.raises(ValueError):
        build_qubo(MarketData.sample_data(), k, risk_aversion, penalty)


@pytest.mark.parametrize(
    "selection",
    [np.array([1, 0]), np.array([1, 0, 2, 0]), np.array([1, 0, np.nan, 0])],
)
def test_qubo_evaluator_rejects_invalid_selection(selection):
    model = build_qubo(MarketData.sample_data(), 2, 0.5, 1.0)
    with pytest.raises(ValueError):
        evaluate_qubo(model, selection)
