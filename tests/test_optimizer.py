import numpy as np
import pytest
from quantum_portfolio.data import MarketData
from quantum_portfolio.optimizer import evaluate_portfolios


def test_known_optimum_single_selection():
    # two assets, choose 1 -> should pick the higher return
    assets = ["A", "B"]
    returns = np.array([0.1, 0.2])
    cov = np.zeros((2, 2))
    md = MarketData(assets=assets, returns=returns, covariance=cov)

    results, best = evaluate_portfolios(md, k=1, risk_aversion=0.0)
    assert best.expected_return == 0.2
    assert best.selection.tolist() == [0, 1]


@pytest.mark.parametrize(
    "market_data",
    [
        MarketData(["A", "A"], np.array([0.1, 0.2]), np.eye(2)),
        MarketData(["A", "B"], np.array([0.1]), np.eye(2)),
        MarketData(["A", "B"], np.array([0.1, 0.2]), np.eye(1)),
        MarketData(
            ["A", "B"],
            np.array([0.1, 0.2]),
            np.array([[1.0, 0.2], [0.1, 1.0]]),
        ),
        MarketData(
            ["A"],
            np.array([np.inf]),
            np.array([[1.0]]),
        ),
        MarketData(
            ["A"],
            np.array([0.1]),
            np.array([[np.nan]]),
        ),
        MarketData(
            ["A", "B"],
            np.array([0.1, 0.2]),
            np.array([[1.0, 2.0], [2.0, 1.0]]),
        ),
    ],
)
def test_market_data_rejects_invalid_values(market_data):
    with pytest.raises(ValueError):
        market_data.validate()


def test_sample_classical_baseline():
    md = MarketData.sample_data()
    results, best = evaluate_portfolios(md, k=2, risk_aversion=0.5)

    assert len(results) == 6
    assert [result.selection.tolist() for result in results] == [
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ]
    assert np.allclose(
        [result.expected_return for result in results],
        [0.22, 0.28, 0.21, 0.30, 0.23, 0.29],
    )
    assert np.allclose(
        [result.risk for result in results],
        [0.099, 0.166, 0.110, 0.155, 0.107, 0.180],
    )
    assert np.allclose(
        [result.objective for result in results],
        [-0.1705, -0.197, -0.155, -0.2225, -0.1765, -0.200],
    )
    assert best.selection.tolist() == [0, 1, 1, 0]
    assert best.objective == pytest.approx(-0.2225)


@pytest.mark.parametrize("k", [-1, 5, 1.5, True])
def test_optimizer_rejects_invalid_k(k):
    with pytest.raises(ValueError):
        evaluate_portfolios(MarketData.sample_data(), k, 0.5)


@pytest.mark.parametrize("risk_aversion", [-0.1, np.inf, np.nan])
def test_optimizer_rejects_invalid_risk_aversion(risk_aversion):
    with pytest.raises(ValueError):
        evaluate_portfolios(MarketData.sample_data(), 2, risk_aversion)
