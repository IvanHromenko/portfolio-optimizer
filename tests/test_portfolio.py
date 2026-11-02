import numpy as np
import pytest
from quantum_portfolio.portfolio import (
    portfolio_return,
    portfolio_risk,
    generate_portfolios,
    objective,
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


@pytest.mark.parametrize(
    ("n_assets", "assets_to_select"),
    [(4, -1), (4, 5), (4, 1.5), (-1, 0), (True, 1)],
)
def test_generate_portfolios_rejects_invalid_sizes(n_assets, assets_to_select):
    with pytest.raises(ValueError):
        generate_portfolios(n_assets, assets_to_select)


@pytest.mark.parametrize(
    "selection",
    [np.array([1]), np.array([1, 2]), np.array([1, np.nan])],
)
def test_portfolio_return_rejects_invalid_selection(selection):
    with pytest.raises(ValueError):
        portfolio_return(np.array([0.1, 0.2]), selection)


@pytest.mark.parametrize("risk_aversion", [-0.1, np.inf, np.nan])
def test_objective_rejects_invalid_risk_aversion(risk_aversion):
    with pytest.raises(ValueError):
        objective(
            np.array([0.1]),
            np.array([[0.04]]),
            np.array([1]),
            risk_aversion,
        )
