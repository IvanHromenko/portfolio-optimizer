from dataclasses import dataclass
from numbers import Integral, Real
from typing import List
import numpy as np


def _validate_selection(selection: np.ndarray, n_assets: int) -> np.ndarray:
    selection = np.asarray(selection)
    if selection.shape != (n_assets,):
        raise ValueError("Selection length must match number of assets")
    if not np.all(np.isin(selection, [0, 1])):
        raise ValueError("Selection values must be 0 or 1")
    return selection


def _validate_risk_aversion(risk_aversion: float) -> float:
    if (
        isinstance(risk_aversion, bool)
        or not isinstance(risk_aversion, Real)
        or not np.isfinite(risk_aversion)
        or risk_aversion < 0
    ):
        raise ValueError("Risk aversion must be finite and non-negative")
    return float(risk_aversion)


@dataclass
class PortfolioResult:
    selection: np.ndarray
    assets: List[str]
    expected_return: float
    risk: float
    objective: float


def selected_assets(assets: List[str], selection: np.ndarray) -> List[str]:
    selection = _validate_selection(selection, len(assets))

    return [a for a, s in zip(assets, selection) if int(s) == 1]


def portfolio_return(returns: np.ndarray, selection: np.ndarray) -> float:
    returns = np.asarray(returns)
    if returns.ndim != 1:
        raise ValueError("Returns must be a one-dimensional vector")
    selection = _validate_selection(selection, returns.size)
    return float(returns @ selection)


def portfolio_risk(covariance: np.ndarray, selection: np.ndarray) -> float:
    covariance = np.asarray(covariance)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance matrix must be square")
    selection = _validate_selection(selection, covariance.shape[0])
    return float(selection.T @ covariance @ selection)


def objective(
    returns: np.ndarray,
    covariance: np.ndarray,
    selection: np.ndarray,
    risk_aversion: float,
) -> float:
    risk_aversion = _validate_risk_aversion(risk_aversion)
    expected_return = portfolio_return(returns, selection)
    risk = portfolio_risk(covariance, selection)
    return -expected_return + risk_aversion * risk


def generate_portfolios(n_assets: int, assets_to_select: int) -> List[np.ndarray]:
    from itertools import combinations

    if (
        isinstance(n_assets, bool)
        or not isinstance(n_assets, Integral)
        or n_assets < 0
    ):
        raise ValueError("Number of assets must be a non-negative integer")
    if (
        isinstance(assets_to_select, bool)
        or not isinstance(assets_to_select, Integral)
        or not 0 <= assets_to_select <= n_assets
    ):
        raise ValueError("Assets to select must be an integer between 0 and n_assets")

    portfolios: List[np.ndarray] = []
    indices = range(n_assets)
    for combo in combinations(indices, assets_to_select):
        sel = np.zeros(n_assets, dtype=int)
        for i in combo:
            sel[i] = 1
        portfolios.append(sel)

    return portfolios
