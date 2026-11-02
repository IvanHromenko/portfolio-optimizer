"""QUBO formulation of the classical binary portfolio objective."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .data import MarketData
from .portfolio import (
    _validate_risk_aversion,
    _validate_selection,
    generate_portfolios,
    objective,
)


@dataclass
class QUBOModel:
    """QUBO using the symmetric convention ``x.T @ matrix @ x + offset``."""

    matrix: np.ndarray
    offset: float
    penalty: float
    required_assets: int


def _validate_problem_parameters(
    market_data: MarketData,
    k: int,
    risk_aversion: float,
    penalty: float,
) -> tuple[int, float, float]:
    market_data.validate()
    n_assets = len(market_data.assets)
    if (
        isinstance(k, bool)
        or not isinstance(k, Integral)
        or not 0 <= k <= n_assets
    ):
        raise ValueError("k must be an integer between 0 and n_assets")
    risk_aversion = _validate_risk_aversion(risk_aversion)
    if (
        isinstance(penalty, bool)
        or not isinstance(penalty, Real)
        or not np.isfinite(penalty)
        or penalty <= 0
    ):
        raise ValueError("Penalty must be finite and positive")
    return int(k), risk_aversion, float(penalty)


def recommended_penalty(
    market_data: MarketData,
    k: int,
    risk_aversion: float,
    safety_margin: float = 0.1,
) -> float:
    """Return a sufficient penalty using an objective lower bound.

    The bound is polynomial to compute and may be conservative. It guarantees
    that every infeasible selection has higher penalized energy than the best
    feasible selection when used by :func:`build_qubo`.
    """
    market_data.validate()
    n_assets = len(market_data.assets)
    if (
        isinstance(k, bool)
        or not isinstance(k, Integral)
        or not 0 <= k <= n_assets
    ):
        raise ValueError("k must be an integer between 0 and n_assets")
    risk_aversion = _validate_risk_aversion(risk_aversion)
    if (
        isinstance(safety_margin, bool)
        or not isinstance(safety_margin, Real)
        or not np.isfinite(safety_margin)
        or safety_margin <= 0
    ):
        raise ValueError("Safety margin must be finite and positive")

    returns = np.asarray(market_data.returns, dtype=float)
    covariance = np.asarray(market_data.covariance, dtype=float)
    linear = -returns + risk_aversion * np.diag(covariance)
    pairwise = 2 * risk_aversion * covariance[np.triu_indices(n_assets, k=1)]
    objective_lower_bound = float(
        np.minimum(linear, 0).sum() + np.minimum(pairwise, 0).sum()
    )
    feasible = generate_portfolios(n_assets, int(k))
    best_feasible = min(
        objective(returns, covariance, selection, risk_aversion)
        for selection in feasible
    )
    required = max(0.0, best_feasible - objective_lower_bound)
    margin = float(safety_margin) * max(1.0, abs(best_feasible), required)
    return required + margin


def build_qubo(
    market_data: MarketData,
    k: int,
    risk_aversion: float,
    penalty: float,
) -> QUBOModel:
    """Build the penalized objective with a symmetric QUBO matrix.

    For this convention, each off-diagonal matrix entry contributes twice to
    ``x.T @ matrix @ x``.
    """
    k, risk_aversion, penalty = _validate_problem_parameters(
        market_data, k, risk_aversion, penalty
    )
    returns = np.asarray(market_data.returns, dtype=float)
    covariance = np.asarray(market_data.covariance, dtype=float)

    matrix = risk_aversion * covariance.copy()
    matrix[np.diag_indices_from(matrix)] += (
        -returns + penalty * (1 - 2 * k)
    )
    off_diagonal = ~np.eye(len(returns), dtype=bool)
    matrix[off_diagonal] += penalty

    return QUBOModel(
        matrix=matrix,
        offset=penalty * k**2,
        penalty=penalty,
        required_assets=k,
    )


def evaluate_qubo(model: QUBOModel, selection: np.ndarray) -> float:
    """Evaluate the penalized QUBO energy for a binary selection vector."""
    matrix = np.asarray(model.matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("QUBO matrix must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("QUBO matrix must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-10):
        raise ValueError("QUBO matrix must be symmetric")
    if not np.isfinite(model.offset):
        raise ValueError("QUBO offset must be finite")
    selection = _validate_selection(selection, matrix.shape[0])
    return float(selection.T @ matrix @ selection + model.offset)
