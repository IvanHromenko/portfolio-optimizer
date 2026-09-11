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

        if self.returns.shape != (n,):
            raise ValueError("Returns vector has incorrect shape")

        if self.covariance.shape != (n, n):
            raise ValueError("Covariance matrix has incorrect shape")

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
