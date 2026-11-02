"""Decode and evaluate QAOA samples in the portfolio's canonical asset order."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .data import MarketData
from .portfolio import (
    _validate_risk_aversion,
    objective,
    portfolio_return,
    portfolio_risk,
)
from .qubo import QUBOModel, evaluate_qubo


@dataclass
class MeasuredPortfolio:
    bitstring: str
    selection: tuple[int, ...]
    count: int
    probability: float
    cardinality: int
    feasible: bool
    objective: Optional[float]
    expected_return: Optional[float]
    risk: Optional[float]
    qubo_energy: float


@dataclass
class QAOASamplingResult:
    outcomes: list[MeasuredPortfolio]
    unique_bitstrings: int
    total_shots: int
    feasible_shots: int
    feasible_probability: float
    optimal_probability: Optional[float]
    best_feasible_bitstring: Optional[str]
    best_feasible_objective: Optional[float]
    mean_feasible_objective: Optional[float]
    simulator: str
    seed: Optional[int]


def decode_bitstring(bitstring: str, num_variables: int) -> np.ndarray:
    """Decode Qiskit's displayed count key into asset-index order.

    Qubit i is measured into classical bit i; Qiskit displays the highest
    classical bit at the left. Therefore the displayed string is reversed.
    """
    if (
        isinstance(num_variables, bool)
        or not isinstance(num_variables, (int, np.integer))
        or num_variables < 0
    ):
        raise ValueError("num_variables must be a non-negative integer")
    if not isinstance(bitstring, str):
        raise ValueError("bitstring must be a string")
    normalized = bitstring.replace(" ", "")
    if len(normalized) != num_variables:
        raise ValueError("Bitstring length must match num_variables")
    if any(bit not in "01" for bit in normalized):
        raise ValueError("Bitstring must contain only 0 and 1")
    return np.fromiter((int(bit) for bit in reversed(normalized)), dtype=int)


def evaluate_counts(
    counts: dict[str, int],
    market_data: MarketData,
    k: int,
    risk_aversion: float,
    qubo_model: QUBOModel,
    optimal_selections: Optional[list[np.ndarray]] = None,
    simulator: str = "provided counts",
    seed: Optional[int] = None,
) -> QAOASamplingResult:
    """Evaluate all counts; infeasible samples are never silently repaired."""
    market_data.validate()
    if not counts:
        raise ValueError("Counts must contain at least one measured outcome")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer")
    if not 0 <= k <= len(market_data.assets):
        raise ValueError("k must be between 0 and the number of assets")
    risk_aversion = _validate_risk_aversion(risk_aversion)

    total_shots = 0
    normalized_counts: list[tuple[str, np.ndarray, int]] = []
    for displayed, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError("Each count must be a non-negative integer")
        selection = decode_bitstring(displayed, len(market_data.assets))
        normalized_counts.append(("".join(map(str, selection)), selection, int(count)))
        total_shots += int(count)
    if total_shots <= 0:
        raise ValueError("Counts must sum to a positive number of shots")

    optimal_bitstrings: Optional[set[str]] = None
    if optimal_selections is not None:
        optimal_bitstrings = set()
        for selection in optimal_selections:
            selection = np.asarray(selection)
            if (
                selection.shape != (len(market_data.assets),)
                or not np.all(np.isin(selection, [0, 1]))
            ):
                raise ValueError("Optimal selections must be valid binary vectors")
            optimal_bitstrings.add("".join(str(int(bit)) for bit in selection))

    outcomes: list[MeasuredPortfolio] = []
    feasible_shots = 0
    feasible_objectives: list[tuple[str, float, int]] = []
    optimal_shots = 0
    for canonical_bits, selection, count in normalized_counts:
        cardinality = int(selection.sum())
        feasible = cardinality == k
        probability = count / total_shots
        if feasible:
            feasible_shots += count
            portfolio_objective = objective(
                market_data.returns,
                market_data.covariance,
                selection,
                risk_aversion,
            )
            expected_return = portfolio_return(market_data.returns, selection)
            risk = portfolio_risk(market_data.covariance, selection)
            feasible_objectives.append((canonical_bits, portfolio_objective, count))
        else:
            portfolio_objective = None
            expected_return = None
            risk = None
        if optimal_bitstrings is not None and canonical_bits in optimal_bitstrings:
            optimal_shots += count

        outcomes.append(
            MeasuredPortfolio(
                bitstring=canonical_bits,
                selection=tuple(int(bit) for bit in selection),
                count=count,
                probability=probability,
                cardinality=cardinality,
                feasible=feasible,
                objective=portfolio_objective,
                expected_return=expected_return,
                risk=risk,
                qubo_energy=evaluate_qubo(qubo_model, selection),
            )
        )

    best = min(feasible_objectives, key=lambda row: row[1]) if feasible_objectives else None
    feasible_count = sum(count for _, _, count in feasible_objectives)
    return QAOASamplingResult(
        outcomes=outcomes,
        unique_bitstrings=len(outcomes),
        total_shots=total_shots,
        feasible_shots=feasible_shots,
        feasible_probability=feasible_shots / total_shots,
        optimal_probability=(
            optimal_shots / total_shots if optimal_bitstrings is not None else None
        ),
        best_feasible_bitstring=best[0] if best else None,
        best_feasible_objective=best[1] if best else None,
        mean_feasible_objective=(
            sum(value * count for _, value, count in feasible_objectives)
            / feasible_count
            if feasible_count
            else None
        ),
        simulator=simulator,
        seed=seed,
    )
