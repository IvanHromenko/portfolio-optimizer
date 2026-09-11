import numpy as np
from quantum_portfolio.portfolio import (
    portfolio_return,
    portfolio_risk,
    generate_portfolios,
)


def test_portfolio_return():
    returns = np.array([0.1, 0.2])
    selection = np.array([1, 0])
    assert abs(portfolio_return(returns, selection) - 0.1) < 1e-9


def test_single_asset_risk():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    selection = np.array([1, 0])
    assert abs(portfolio_risk(cov, selection) - 0.04) < 1e-9


def test_two_asset_risk():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    selection = np.array([1, 1])
    expected = 0.04 + 0.09 + 2 * 0.01
    assert abs(portfolio_risk(cov, selection) - expected) < 1e-9


def test_generate_portfolios():
    portfolios = generate_portfolios(4, 2)
    assert len(portfolios) == 6
    for p in portfolios:
        assert p.sum() == 2
