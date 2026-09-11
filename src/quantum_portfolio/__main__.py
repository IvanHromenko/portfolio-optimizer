from .data import MarketData
from .optimizer import evaluate_portfolios
from .evaluation import plot_risk_return, plot_objectives


def main() -> None:
    md = MarketData.sample_data()
    k = 2
    risk_aversion = 0.5

    results, best = evaluate_portfolios(md, k, risk_aversion)

    print("Quantum Portfolio Optimizer (classical baseline)")
    print("=============================================")
    print("Assets:")
    for a in md.assets:
        print(a)
    print(f"\nRequired assets: {k}\n")
    print(f"Evaluating {len(results)} portfolios...\n")

    print(f"{'Portfolio':30} {'Return':>8} {'Risk':>10} {'Objective':>12}")
    print('-' * 66)
    for r in results:
        name = ' + '.join(r.assets)
        print(f"{name:30} {r.expected_return:8.3f} {r.risk:10.3f} {r.objective:12.3f}")

    print("\nBest portfolio:\n")
    print(' + '.join(best.assets))

    # Optional: show plots
    try:
        plot_risk_return(results, best)
        plot_objectives(results)
    except Exception:
        pass


if __name__ == '__main__':
    main()
