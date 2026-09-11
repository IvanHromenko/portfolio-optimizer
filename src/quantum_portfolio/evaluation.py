from typing import List
import matplotlib.pyplot as plt
from .portfolio import PortfolioResult


def plot_risk_return(results: List[PortfolioResult], best: PortfolioResult) -> None:
    returns = [r.expected_return for r in results]
    risks = [r.risk for r in results]
    labels = [" + ".join(r.assets) for r in results]

    plt.figure()
    plt.scatter(returns, risks)
    for i, label in enumerate(labels):
        plt.annotate(label, (returns[i], risks[i]))
    plt.xlabel("Expected return")
    plt.ylabel("Risk")
    plt.title("Risk vs Return")
    plt.scatter([best.expected_return], [best.risk], color="red", label="Best")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_objectives(results: List[PortfolioResult]) -> None:
    labels = [" + ".join(r.assets) for r in results]
    objs = [r.objective for r in results]

    plt.figure()
    plt.bar(labels, objs)
    plt.ylabel("Objective (lower is better)")
    plt.xticks(rotation=45, ha="right")
    plt.title("Portfolio objectives")
    plt.tight_layout()
    plt.show()
