from typing import List, Tuple
import numpy as np
from .data import MarketData
from .portfolio import PortfolioResult, generate_portfolios, portfolio_return, portfolio_risk, objective, selected_assets


def evaluate_portfolios(
    market_data: MarketData, k: int, risk_aversion: float
) -> Tuple[List[PortfolioResult], PortfolioResult]:
    market_data.validate()
    n = len(market_data.assets)
    portfolios = generate_portfolios(n, k)

    results: List[PortfolioResult] = []
    for sel in portfolios:
        ret = portfolio_return(market_data.returns, sel)
        risk = portfolio_risk(market_data.covariance, sel)
        obj = objective(market_data.returns, market_data.covariance, sel, risk_aversion)
        assets = selected_assets(market_data.assets, sel)
        results.append(PortfolioResult(selection=sel, assets=assets, expected_return=ret, risk=risk, objective=obj))

    best = min(results, key=lambda r: r.objective)
    return results, best
