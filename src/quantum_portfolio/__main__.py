import argparse

from .data import MarketData
from .optimizer import evaluate_portfolios
from .evaluation import plot_risk_return, plot_objectives


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical and QAOA portfolio optimizer")
    parser.add_argument("--qaoa", action="store_true", help="run ideal-simulator QAOA")
    parser.add_argument("--depth", type=int, default=1, help="QAOA depth (default: 1)")
    parser.add_argument("--shots", type=int, default=2000, help="measurement shots")
    parser.add_argument("--seed", type=int, default=7, help="reproducibility seed")
    parser.add_argument(
        "--maxiter", type=int, default=300, help="COBYLA maximum evaluations"
    )
    args = parser.parse_args()
    if args.qaoa:
        if args.depth < 1:
            parser.error("--depth must be positive")
        if args.shots < 1:
            parser.error("--shots must be positive")
        if args.maxiter < 1:
            parser.error("--maxiter must be positive")
        if args.seed < 0:
            parser.error("--seed must be non-negative")

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

    if args.qaoa:
        from .ising import qubo_to_ising
        from .qaoa_optimizer import optimize_qaoa, sample_qaoa
        from .qubo import build_qubo, recommended_penalty

        penalty = recommended_penalty(md, k, risk_aversion)
        qubo = build_qubo(md, k, risk_aversion, penalty)
        ising = qubo_to_ising(qubo)
        optimized = optimize_qaoa(
            ising,
            depth=args.depth,
            seed=args.seed,
            maxiter=args.maxiter,
        )
        sampled = sample_qaoa(
            ising,
            optimized,
            shots=args.shots,
            seed=args.seed,
            market_data=md,
            k=k,
            risk_aversion=risk_aversion,
            qubo_model=qubo,
            optimal_selections=[
                result.selection for result in results if result.objective == best.objective
            ],
        )
        print("\nQAOA ideal-simulator result")
        print("---------------------------")
        print(f"Depth: {optimized.depth}")
        print(f"Expected penalized energy: {optimized.expected_energy:.6f}")
        print(f"Optimizer success: {optimized.success} ({optimized.message})")
        print(f"Optimizer evaluations: {optimized.nfev}")
        print(f"Shots: {sampled.total_shots}")
        print(f"Sampling simulator: {sampled.simulator}")
        print(f"Sampling seed: {sampled.seed}")
        print(f"Unique bitstrings: {sampled.unique_bitstrings}")
        print(
            f"Feasible shots: {sampled.feasible_shots}/{sampled.total_shots}"
        )
        print(f"Feasible-shot probability: {sampled.feasible_probability:.3f}")
        print(f"Classical-optimum probability: {sampled.optimal_probability:.3f}")
        print(f"Mean feasible objective: {sampled.mean_feasible_objective}")
        if sampled.best_feasible_bitstring is None:
            print("Best feasible sample: none observed")
        else:
            print(
                f"Best feasible sample: {sampled.best_feasible_bitstring} "
                f"(objective {sampled.best_feasible_objective:.6f})"
            )
        print(f"Seed: {optimized.seed}")
        print(f"Package versions: {optimized.package_versions}")
        return

    plot_risk_return(results, best)
    plot_objectives(results)


if __name__ == '__main__':
    main()
