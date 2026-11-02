# Portfolio optimizer

The classical reference minimizes

`f(x) = -returns @ x + risk_aversion * (x.T @ covariance @ x)`

over binary selections with exactly `k` assets. The current quantum-transition
work adds a symmetric-matrix QUBO while preserving that objective:

`E(x) = f(x) + penalty * (sum(x) - k) ** 2`

In the symmetric matrix convention, off-diagonal entries contribute twice to
`x.T @ Q @ x`. `recommended_penalty` derives a sufficient penalty from a
lower bound on the unpenalized objective; it is deliberately separate from
`build_qubo`, where the penalty remains explicit and configurable.

Run the classical reference with `python -m quantum_portfolio` (with `src` on
the Python import path, or after installing the project in editable mode).

## Classical baseline

For the built-in four-asset data, `k = 2` and risk aversion `0.5` produce these
six feasible portfolios:

| Selection | Return | Risk | Objective |
|---|---:|---:|---:|
| AAPL + MSFT | 0.220 | 0.099 | -0.1705 |
| AAPL + NVDA | 0.280 | 0.166 | -0.1970 |
| AAPL + AMZN | 0.210 | 0.110 | -0.1550 |
| MSFT + NVDA | 0.300 | 0.155 | **-0.2225** |
| MSFT + AMZN | 0.230 | 0.107 | -0.1765 |
| NVDA + AMZN | 0.290 | 0.180 | -0.2000 |

The exact optimum is `MSFT + NVDA` (bitstring `0110` in asset-index order),
with objective `-0.2225`. The QUBO tests compare its energy with the penalized
classical objective for all 16 bitstrings. Qiskit is intentionally not needed
for this mathematical validation stage.