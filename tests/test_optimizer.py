import numpy as np
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
