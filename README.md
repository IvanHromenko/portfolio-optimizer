# Portfolio optimizer: classical reference and QAOA experiment

This project keeps exhaustive classical search as the reference and adds a
local Qiskit simulator implementation of QAOA. The small built-in example is
for validating the mathematical mappings and learning the hybrid workflow;
it does not demonstrate quantum advantage.

## Install and run

Create/activate a virtual environment, then install the project and optional
quantum and test dependencies:

```powershell
python -m pip install -e ".[quantum,test]"
python -m pytest
```

Run the classical baseline with `python -m quantum_portfolio`. To optimize QAOA
angles using an exact statevector and then draw finite-shot samples from local
Qiskit Aer:

```powershell
python -m quantum_portfolio --qaoa --depth 1 --shots 2000 --seed 7 --maxiter 300
```

The CLI prints classical and sampled results, and avoids opening the classical
plots in QAOA mode. Qiskit, Aer and SciPy are optional dependencies; the
classical optimizer and QUBO/Ising mathematics work without them.

## Mathematical formulation

The classical reference minimizes

`f(x) = -returns @ x + risk_aversion * (x.T @ covariance @ x)`

over binary selections with exactly `k` assets. The QUBO preserves this
objective and adds a cardinality penalty:

`E(x) = f(x) + penalty * (sum(x) - k) ** 2`

The QUBO uses a symmetric matrix convention: `E(x) = x.T @ Q @ x + offset`,
so each off-diagonal matrix entry contributes twice. `recommended_penalty`
derives a sufficient penalty from a lower bound on the unpenalized objective;
`build_qubo` keeps the penalty explicit and configurable.

The Ising conversion uses `x_i = (1 - z_i) / 2`, with spins `z_i` in
`{-1, +1}`:

`E(z) = offset + sum(h_i * z_i) + sum(J_ij * z_i * z_j)`

`qubo_to_ising` is NumPy-only and `ising_energy` evaluates that polynomial.
Exhaustive tests verify equality of QUBO and Ising energies for all 16
four-asset bitstrings.

## QAOA implementation

The circuit starts in the uniform superposition, then repeats a cost and
mixer layer for each parameter pair `(gamma, beta)`. The cost Hamiltonian is
`sum(h_i Z_i) + sum(J_ij Z_i Z_j)`; its constant offset is retained in energy
calculations but omitted from gates because it contributes only global phase.
The cost terms use `RZ(2 * gamma * h_i)` and `CX-RZ(2 * gamma * J_ij)-CX`;
the standard X mixer uses `RX(2 * beta)` on each qubit. Tests compare the
resulting basis-state phases with the Ising polynomial.

`optimize_qaoa` uses SciPy COBYLA to minimize the exact statevector expected
energy. Its seed, convergence status, evaluation count, angles, and installed
initial/final angles, and installed Qiskit/Aer/SciPy versions are returned.
`sample_qaoa` then uses Aer for
finite-shot measurements. The sampled evaluator reports feasibility, exact
classical-optimum probability, best feasible objective, and average feasible
objective separately from penalized QUBO energy, along with feasible/total
shots and the number of unique measured bitstrings. It never repairs
infeasible samples. Its result includes the simulator and measurement seed.

Qiskit count strings display the highest-index classical bit on the left.
`decode_bitstring` reverses this representation into asset-index order. For
example, a canonical selection `[1, 0, 1, 0]` is displayed by Qiskit as
`"0101"` and decodes back to `[1, 0, 1, 0]`.

The current implementation uses ideal local simulation only: exact
statevector expectations for optimization and ideal finite-shot Aer sampling.
Noisy backends, hardware-provider setup, realistic portfolio weights, and
quantum-advantage claims are outside this phase. Statevector simulation scales
exponentially with qubit count; the four-asset case is a correctness
demonstration, not a practical speedup.

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