from dataclasses import dataclass
import numpy as np
from typing import List


@dataclass
class MarketData:
    assets: List[str]
    returns: np.ndarray
    covariance: np.ndarray

    def validate(self) -> None:
        n = len(self.assets)
        if len(set(self.assets)) != n:
            raise ValueError("Asset names must be unique")

        returns = np.asarray(self.returns)
        covariance = np.asarray(self.covariance)

        if returns.shape != (n,):
            raise ValueError("Returns vector has incorrect shape")

        if covariance.shape != (n, n):
            raise ValueError("Covariance matrix has incorrect shape")

        if not np.issubdtype(returns.dtype, np.number) or not np.all(
            np.isfinite(returns)
        ):
            raise ValueError("Returns must contain only finite numeric values")

        if not np.issubdtype(covariance.dtype, np.number) or not np.all(
            np.isfinite(covariance)
        ):
            raise ValueError("Covariance must contain only finite numeric values")

        if not np.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-10):
            raise ValueError("Covariance matrix must be symmetric")

        eigenvalues = np.linalg.eigvalsh(covariance)
        covariance_scale = (
            float(np.linalg.norm(covariance, ord=2)) if n else 0.0
        )
        tolerance = 1e-10 * max(1.0, covariance_scale)
        if np.min(eigenvalues, initial=0.0) < -tolerance:
            raise ValueError("Covariance matrix must be positive semidefinite")

    @staticmethod
    def sample_data() -> "MarketData":
        assets = ["AAPL", "MSFT", "NVDA", "AMZN"]
        returns = np.array([0.10, 0.12, 0.18, 0.11])
        covariance = np.array(
            [
                [0.040, 0.012, 0.018, 0.010],
                [0.012, 0.035, 0.015, 0.011],
                [0.018, 0.015, 0.090, 0.020],
                [0.010, 0.011, 0.020, 0.050],
            ]
        )

        md = MarketData(assets=assets, returns=returns, covariance=covariance)
        md.validate()
        return md
