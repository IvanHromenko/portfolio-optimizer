from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class PortfolioResult:
    selection: np.ndarray
    assets: List[str]
    expected_return: float
    risk: float
    objective: float


def selected_assets(assets: List[str], selection: np.ndarray) -> List[str]:
    if len(selection) != len(assets):
        raise ValueError("Selection length must match number of assets")

    if not np.all(np.isin(selection, [0, 1])):
        raise ValueError("Selection values must be 0 or 1")

    return [a for a, s in zip(assets, selection) if int(s) == 1]


def portfolio_return(returns: np.ndarray, selection: np.ndarray) -> float:
    return float(returns @ selection)


def portfolio_risk(covariance: np.ndarray, selection: np.ndarray) -> float:
    return float(selection.T @ covariance @ selection)


def objective(
    returns: np.ndarray,
    covariance: np.ndarray,
    selection: np.ndarray,
    risk_aversion: float,
) -> float:
    expected_return = portfolio_return(returns, selection)
    risk = portfolio_risk(covariance, selection)
    return -expected_return + risk_aversion * risk


def generate_portfolios(n_assets: int, assets_to_select: int) -> List[np.ndarray]:
    from itertools import combinations

    portfolios: List[np.ndarray] = []
    indices = range(n_assets)
    for combo in combinations(indices, assets_to_select):
        sel = np.zeros(n_assets, dtype=int)
        for i in combo:
            sel[i] = 1
        portfolios.append(sel)

    return portfolios
