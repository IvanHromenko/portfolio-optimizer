# Phase 1 — Classical Portfolio Optimization From Scratch

## 0. Phase 1 objective

Build a small classical portfolio optimizer that will become the foundation for the later quantum-computing phases.

The end-to-end pipeline is:

```text
Asset data
   ↓
Expected returns
   ↓
Covariance matrix
   ↓
Portfolio representation
   ↓
Portfolio return
   ↓
Portfolio risk
   ↓
Objective function
   ↓
Generate all valid portfolios
   ↓
Evaluate them
   ↓
Find the optimum
   ↓
Tests + visualization
```

At the end of Phase 1, the project should be able to:

- work with 4 assets;
- represent portfolio selections as binary vectors;
- calculate expected return;
- calculate portfolio risk;
- calculate a risk/return objective;
- generate all portfolios containing exactly 2 assets;
- evaluate all valid portfolios;
- identify the classical optimum;
- verify the implementation with unit tests;
- visualize the optimization landscape;
- investigate how the optimum changes as parameters change.

**Important:** Phase 1 is completely classical. Do not use Qiskit, QAOA, QUBO libraries, IBM Quantum, or real market-data APIs yet.

---

# 1. What you should learn

Phase 1 combines four areas.

## Python

Practice:

- virtual environments;
- packages;
- modules;
- functions;
- dataclasses;
- NumPy;
- pandas;
- matplotlib;
- type hints;
- pytest;
- Git.

## Mathematics

Work with:

- vectors;
- matrices;
- matrix multiplication;
- covariance matrices;
- quadratic forms;
- binary variables;
- combinatorial search;
- optimization.

## Finance

Only the minimum required concepts:

- expected return;
- variance;
- covariance;
- portfolio risk;
- risk/return trade-off.

## Software engineering

Practice:

- separation of concerns;
- validation;
- unit testing;
- reproducible experiments;
- project structure;
- documentation;
- Git commits.

The goal is for the final project to look like a small software/algorithm project rather than a notebook full of unexplained code.

---

# 2. Repository setup

Recommended initial structure:

```text
portfolio-optimizer/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── src/
│   └── quantum_portfolio/
│       ├── __init__.py
│       ├── __main__.py
│       ├── data.py
│       ├── portfolio.py
│       ├── optimizer.py
│       └── evaluation.py
│
├── tests/
│   ├── test_portfolio.py
│   └── test_optimizer.py
│
└── notebooks/
    └── 01_classical_portfolio.ipynb
```

The exact architecture does not need to be perfect. The main idea is to separate:

```text
data
 ↓
portfolio calculations
 ↓
optimization
 ↓
evaluation / presentation
```

---

# 3. Python environment

Use a modern Python version supported by the libraries you intend to use later. Python 3.12+ is a reasonable starting point if your future quantum stack supports it.

Create a virtual environment:

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

Install the Phase 1 dependencies:

```bash
pip install numpy pandas matplotlib pytest
```

Freeze the environment:

```bash
pip freeze > requirements.txt
```

---

# 4. Verify the environment

Before starting the actual project, verify that the dependencies work.

Create a temporary script:

```python
import numpy
import pandas
import matplotlib

print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Matplotlib:", matplotlib.__version__)
```

Run it successfully before continuing.

The principle is:

> Establish that the environment works before debugging your actual project.

---

# 5. Define the mathematical problem first

Do not start with Python code. First define the optimization problem.

We have four assets:

```text
AAPL
MSFT
NVDA
AMZN
```

For each asset, define a binary decision variable:

\[
x_i \in \{0,1\}
\]

Interpretation:

```text
x_i = 1 → asset is selected
x_i = 0 → asset is not selected
```

We want exactly two assets:

\[
\boxed{x_1+x_2+x_3+x_4=2}
\]

This constraint is extremely important because it will later be transformed into a penalty term for a QUBO formulation.

---

# 6. Use a deterministic dataset

For the first implementation, do **not** download real financial data.

Use synthetic, deterministic data.

Example asset list:

```python
ASSETS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
]
```

Example expected annual returns:

```python
import numpy as np

RETURNS = np.array([
    0.10,
    0.12,
    0.18,
    0.11,
])
```

Example covariance matrix:

```python
COVARIANCE = np.array([
    [0.040, 0.012, 0.018, 0.010],
    [0.012, 0.035, 0.015, 0.011],
    [0.018, 0.015, 0.090, 0.020],
    [0.010, 0.011, 0.020, 0.050],
])
```

The exact values are not important.

Using synthetic data keeps the focus on:

- mathematics;
- Python;
- optimization;
- later quantum transformation.

Avoid distractions such as:

- API authentication;
- missing data;
- stock splits;
- time zones;
- financial-data cleaning;
- external-service failures.

---

# 7. Understand the covariance matrix

The covariance matrix is:

\[
\Sigma =
\begin{bmatrix}
\sigma_1^2 & \sigma_{12} & \sigma_{13} & \sigma_{14}\\
\sigma_{21} & \sigma_2^2 & \sigma_{23} & \sigma_{24}\\
\sigma_{31} & \sigma_{32} & \sigma_3^2 & \sigma_{34}\\
\sigma_{41} & \sigma_{42} & \sigma_{43} & \sigma_4^2
\end{bmatrix}
\]

The diagonal contains the individual asset variances:

\[
\sigma_i^2
\]

The off-diagonal elements contain covariances:

\[
\sigma_{ij}
\]

For example:

```text
COV(AAPL, MSFT)
```

represents the covariance between AAPL and MSFT.

The covariance matrix is important because it introduces **pairwise interactions** between assets. Those interactions will later become the quadratic terms of the QUBO.

---

# 8. Create a market-data model

Create `src/quantum_portfolio/data.py`.

A simple data model:

```python
from dataclasses import dataclass
import numpy as np


@dataclass
class MarketData:
    assets: list[str]
    returns: np.ndarray
    covariance: np.ndarray

    def validate(self) -> None:
        n = len(self.assets)

        if self.returns.shape != (n,):
            raise ValueError("Returns vector has incorrect shape")

        if self.covariance.shape != (n, n):
            raise ValueError("Covariance matrix has incorrect shape")
```

Then create a deterministic dataset.

The important principle is:

> Mathematical assumptions should be validated by the program.

---

# 9. Represent a portfolio as a binary vector

Represent portfolio selection using a binary vector.

Example:

```text
[1, 0, 1, 0]
```

means:

```text
AAPL → selected
MSFT → not selected
NVDA → selected
AMZN → not selected
```

Mathematically:

\[
x =
\begin{bmatrix}
1\\
0\\
1\\
0
\end{bmatrix}
\]

This represents:

```text
AAPL + NVDA
```

This representation is extremely important.

Later, the same binary vector will correspond naturally to a quantum computational-basis state.

---

# 10. Decode a portfolio

Create a function such as:

```python
def selected_assets(
    assets: list[str],
    selection: np.ndarray,
) -> list[str]:
    ...
```

For:

```python
selection = np.array([1, 0, 1, 0])
```

it should return:

```python
["AAPL", "NVDA"]
```

Also validate that:

- the selection length matches the number of assets;
- values are binary;
- the selection contains the expected number of assets when appropriate.

---

# 11. Calculate portfolio return

Define the expected portfolio return as:

\[
\boxed{R(x)=\mu^T x}
\]

where:

\[
\mu =
\begin{bmatrix}
0.10\\
0.12\\
0.18\\
0.11
\end{bmatrix}
\]

For:

\[
x =
\begin{bmatrix}
1\\
0\\
1\\
0
\end{bmatrix}
\]

we get:

\[
R(x)=0.10+0.18=0.28
\]

Implement:

```python
def portfolio_return(
    returns: np.ndarray,
    selection: np.ndarray,
) -> float:
    return float(returns @ selection)
```

Understand that:

```python
returns @ selection
```

represents the mathematical operation:

\[
\mu^T x
\]

Do not treat NumPy's `@` operator as magic.

---

# 12. Calculate portfolio risk

Use the simplified quadratic risk model:

\[
\boxed{
Risk(x)=x^T\Sigma x
}
\]

Implement:

```python
def portfolio_risk(
    covariance: np.ndarray,
    selection: np.ndarray,
) -> float:
    return float(selection.T @ covariance @ selection)
```

For at least one example, calculate this manually.

For:

```text
x = [1, 0, 1, 0]
```

calculate:

\[
x^T\Sigma x
\]

by hand and verify that Python produces the same result.

This is one of the most important exercises in Phase 1.

---

# 13. Understand why the risk is quadratic

Expand:

\[
x^T\Sigma x
\]

Conceptually, the expression contains:

\[
\sum_i \sigma_i^2x_i^2
+
2\sum_{i<j}\sigma_{ij}x_ix_j
\]

Therefore the risk consists of:

```text
individual asset terms
+
pairwise asset interaction terms
```

The pairwise terms are the reason the problem naturally leads to a **quadratic binary optimization problem**.

This is the mathematical bridge to Phase 2.

---

# 14. Define the optimization objective

We want to balance expected return and risk.

Define:

\[
\boxed{
f(x)=-\mu^Tx+\lambda x^T\Sigma x
}
\]

where:

- \(\mu\) = expected returns;
- \(\Sigma\) = covariance matrix;
- \(\lambda\) = risk-aversion parameter;
- \(x\) = binary selection vector.

For example:

```python
RISK_AVERSION = 0.5
```

Implement:

```python
def objective(
    returns: np.ndarray,
    covariance: np.ndarray,
    selection: np.ndarray,
    risk_aversion: float,
) -> float:
    expected_return = portfolio_return(
        returns,
        selection,
    )

    risk = portfolio_risk(
        covariance,
        selection,
    )

    return -expected_return + risk_aversion * risk
```

---

# 15. Understand the sign convention

We are solving:

\[
\min f(x)
\]

rather than maximizing return directly.

The objective is:

\[
f(x)=-R(x)+\lambda Risk(x)
\]

Therefore:

```text
higher return → lower objective
higher risk   → higher objective
lower objective → better portfolio
```

This formulation will be convenient later because QUBO and QAOA are naturally expressed as minimization of a cost function.

---

# 16. Generate all valid portfolios

For four assets with exactly two selected:

\[
\sum_i x_i=2
\]

there are:

\[
\binom{4}{2}=6
\]

valid portfolios.

They are:

```text
[1, 1, 0, 0] → AAPL + MSFT
[1, 0, 1, 0] → AAPL + NVDA
[1, 0, 0, 1] → AAPL + AMZN
[0, 1, 1, 0] → MSFT + NVDA
[0, 1, 0, 1] → MSFT + AMZN
[0, 0, 1, 1] → NVDA + AMZN
```

Do not hard-code them.

Generate them programmatically using:

```python
from itertools import combinations
```

Create:

```python
def generate_portfolios(
    n_assets: int,
    assets_to_select: int,
) -> list[np.ndarray]:
    ...
```

For:

```python
generate_portfolios(4, 2)
```

the function should produce six binary vectors.

---

# 17. Verify the portfolio generator

Add checks such as:

```python
portfolios = generate_portfolios(4, 2)

assert len(portfolios) == 6

for portfolio in portfolios:
    assert portfolio.sum() == 2
```

The optimizer should never receive an invalid portfolio.

This is a useful software-engineering principle:

> Validate the assumptions at the boundaries of your system.

---

# 18. Create a portfolio-result model

Create a result data structure:

```python
from dataclasses import dataclass
import numpy as np


@dataclass
class PortfolioResult:
    selection: np.ndarray
    assets: list[str]
    expected_return: float
    risk: float
    objective: float
```

Each evaluated portfolio should produce one result.

Example output:

```text
AAPL + MSFT
Return: 0.220
Risk:   ...
Score:  ...

AAPL + NVDA
Return: 0.280
Risk:   ...
Score:  ...
```

---

# 19. Evaluate every portfolio

Create a function that:

1. generates valid portfolios;
2. calculates return;
3. calculates risk;
4. calculates objective;
5. creates `PortfolioResult` objects.

Conceptually:

```text
generate portfolios
        ↓
for each portfolio:
    calculate return
    calculate risk
    calculate objective
        ↓
store result
```

Keep this separate from the mathematical calculation functions.

---

# 20. Find the classical optimum

Because there are only six possibilities, an exhaustive search is trivial.

Use:

```python
best = min(
    results,
    key=lambda result: result.objective,
)
```

The program should print something like:

```text
Best portfolio:
MSFT + AMZN

Expected return:
...

Risk:
...

Objective:
...
```

The exact winner depends on your chosen parameters.

At this point, you have built the classical baseline that later quantum algorithms must be compared against.

---

# 21. Create a command-line entry point

The project should eventually run with:

```bash
python -m quantum_portfolio
```

Example output:

```text
Quantum Portfolio Optimizer
===========================

Assets:
AAPL
MSFT
NVDA
AMZN

Required assets: 2

Evaluating 6 portfolios...

Portfolio             Return       Risk       Objective
---------------------------------------------------------
AAPL + MSFT            ...
AAPL + NVDA            ...
AAPL + AMZN            ...
MSFT + NVDA            ...
MSFT + AMZN            ...
NVDA + AMZN            ...

Best portfolio:
MSFT + AMZN
```

Even though this is a small experiment, give it a clean interface.

---

# 22. Write unit tests

Create tests for the mathematical building blocks.

## Test 1 — portfolio return

Given:

```text
returns   = [0.1, 0.2]
selection = [1, 0]
```

the expected result is:

```text
0.1
```

---

## Test 2 — single-asset risk

Given:

\[
\Sigma=
\begin{bmatrix}
0.04 & 0.01\\
0.01 & 0.09
\end{bmatrix}
\]

and:

```text
x = [1, 0]
```

the risk should be:

\[
0.04
\]

---

## Test 3 — two-asset risk

For:

```text
x = [1, 1]
```

verify:

\[
x^T\Sigma x
=
0.04+0.09+2(0.01)
\]

Therefore:

\[
=0.15
\]

---

## Test 4 — portfolio generation

Verify:

```text
number of portfolios = 6
```

and:

```text
sum(selection) = 2
```

for every generated portfolio.

---

## Test 5 — known optimum

Create a tiny artificial problem where the optimal portfolio is known in advance.

Then assert that your optimizer returns that exact portfolio.

---

# 23. Add a key mathematical consistency test

This test will become particularly important in Phase 2.

For every possible portfolio, verify that your implementation of:

```text
objective(x)
```

matches the direct mathematical calculation:

\[
-\mu^Tx+\lambda x^T\Sigma x
\]

for the same \(x\).

Later, when you construct the QUBO matrix \(Q\), you will want to verify:

\[
f(x)=x^TQx+C
\]

for every relevant binary vector.

This creates a reliable bridge between the classical and quantum implementations.

---

# 24. Visualization 1 — risk vs return

Create a scatter plot where:

- x-axis = expected return;
- y-axis = risk;
- each point = one portfolio;
- the optimum is identified.

Conceptually:

```text
Risk
 ^
 |
 |             ●
 |       ●
 |   ●
 |         ●
 | ●
 |________________> Return
```

This helps you visually understand the optimization landscape.

---

# 25. Visualization 2 — objective values

Create a bar chart for the objective of each portfolio.

Conceptually:

```text
Portfolio
AAPL+MSFT   █████████
AAPL+NVDA   █████████████
AAPL+AMZN   ███████
MSFT+NVDA   ██████████
MSFT+AMZN   ██████
NVDA+AMZN   █████████
```

The lowest bar represents the optimal solution because we are minimizing the objective.

This visualization will become useful later when comparing classical results with quantum measurement probabilities.

---

# 26. Experiment 1 — change risk aversion

Run the optimizer with several values:

```text
λ = 0
λ = 0.25
λ = 0.5
λ = 1
λ = 2
```

Observe how the optimal portfolio changes.

When:

\[
\lambda=0
\]

the objective becomes:

\[
-R(x)
\]

so the optimizer is effectively maximizing return.

As \(\lambda\) increases, risk becomes increasingly important.

Document:

- the value of \(\lambda\);
- the optimal portfolio;
- its return;
- its risk;
- its objective value.

---

# 27. Experiment 2 — change the number of selected assets

Run the optimizer with:

```text
k = 1
k = 2
k = 3
k = 4
```

For four assets:

\[
\binom41=4
\]

\[
\binom42=6
\]

\[
\binom43=4
\]

\[
\binom44=1
\]

Observe how the number of candidate portfolios changes.

---

# 28. Experiment 3 — understand combinatorial growth

Measure the number of possible portfolios for larger problems.

Example:

```text
n assets    k selected    combinations
----------------------------------------
4           2             6
8           4             70
12          6             924
16          8             12,870
20          10            184,756
```

For example:

\[
\binom{20}{10}=184756
\]

For 100 assets and 50 selections:

\[
\binom{100}{50}
\]

is enormous.

You do not need to run such a calculation by brute force.

The point is to understand why combinatorial optimization becomes difficult as the problem size grows.

This provides the motivation for investigating alternative optimization techniques.

---

# 29. Experiment 4 — brute-force scaling

Extend the experiment so that you measure:

```text
number of assets
number of valid portfolios
execution time
```

For example:

```text
n = 4
n = 8
n = 12
n = 16
n = 20
```

You are not trying to prove a formal complexity result here.

You are building intuition about the growth of the search space.

---

# 30. What not to implement in Phase 1

Do not add these yet:

```text
Qiskit
QAOA
quantum circuits
QUBO libraries
IBM Quantum
real stock APIs
machine learning
```

The purpose of Phase 1 is to understand the classical optimization problem completely.

Only after you understand the problem should you transform it into a quantum-compatible formulation.

---

# 31. Recommended development sequence

Work through Phase 1 in this order:

```text
1. Repository
   ↓
2. Python environment
   ↓
3. Synthetic data
   ↓
4. Mathematical model
   ↓
5. MarketData model
   ↓
6. Binary portfolio representation
   ↓
7. Return calculation
   ↓
8. Risk calculation
   ↓
9. Objective function
   ↓
10. Portfolio generation
   ↓
11. Portfolio evaluation
   ↓
12. Classical optimum
   ↓
13. Unit tests
   ↓
14. Visualizations
   ↓
15. Parameter experiments
   ↓
16. Documentation
```

Do not move forward just because the code runs. At each stage, make sure you understand the corresponding mathematics.

---

# 32. Suggested 3–5 day schedule

## Day 1 — Python + data model

Focus on:

- repository setup;
- virtual environment;
- NumPy;
- pandas;
- dataclasses;
- synthetic market data;
- validation;
- binary portfolio representation.

Deliverable:

```text
MarketData
+
portfolio representation
```

---

## Day 2 — Mathematics + objective

Focus on:

- expected return;
- matrix/vector multiplication;
- covariance;
- quadratic forms;
- portfolio risk;
- risk-aversion parameter;
- objective function.

Deliverable:

```text
portfolio_return()
portfolio_risk()
objective()
```

You should manually verify at least several calculations.

---

## Day 3 — Optimization + testing

Focus on:

- combinations;
- valid portfolio generation;
- exhaustive search;
- result models;
- unit tests;
- known-optimum test.

Deliverable:

```text
classical optimizer
+
test suite
```

---

## Day 4 — Visualization + experiments

Implement:

- risk/return plot;
- objective-value chart;
- risk-aversion experiment;
- different portfolio sizes;
- brute-force scaling experiment.

Deliverable:

```text
results
+
plots
+
observations
```

---

## Day 5 — Cleanup + documentation

Clean up:

- code;
- naming;
- type hints;
- tests;
- README;
- mathematical explanation;
- Git history.

The README should explain:

1. the problem;
2. the mathematical model;
3. the binary representation;
4. the objective function;
5. the classical algorithm;
6. the results;
7. limitations;
8. why the problem is interesting for quantum optimization.

---

# 33. Phase 1 final architecture

The completed classical implementation should conceptually look like:

```text
                         ┌─────────────┐
                         │ MarketData  │
                         └──────┬──────┘
                                │
                                ▼
                     ┌────────────────────┐
                     │ Portfolio generator│
                     └─────────┬──────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │ Portfolio evaluation      │
                 │                           │
                 │ return = μᵀx              │
                 │ risk   = xᵀΣx             │
                 │ objective = -return+λrisk│
                 └────────────┬──────────────┘
                              │
                              ▼
                     ┌────────────────┐
                     │ Classical      │
                     │ optimizer      │
                     └───────┬────────┘
                             │
                             ▼
                      Best portfolio
```

---

# 34. Phase 1 completion checklist

## Python

- [ ] Python virtual environment created
- [ ] NumPy installed
- [ ] pandas installed
- [ ] matplotlib installed
- [ ] pytest installed
- [ ] Git repository created
- [ ] project structure created

## Mathematics

- [ ] Understand vector \(x\)
- [ ] Understand return vector \(\mu\)
- [ ] Understand covariance matrix \(\Sigma\)
- [ ] Understand \(\mu^Tx\)
- [ ] Understand \(x^T\Sigma x\)
- [ ] Understand why risk is quadratic
- [ ] Understand binary variables
- [ ] Understand the constraint \(\sum_i x_i=k\)
- [ ] Understand why there are \(\binom{n}{k}\) possible portfolios

## Programming

- [ ] Implement `MarketData`
- [ ] Implement data validation
- [ ] Implement binary portfolio representation
- [ ] Implement portfolio decoding
- [ ] Implement return calculation
- [ ] Implement risk calculation
- [ ] Implement objective function
- [ ] Implement portfolio generation
- [ ] Implement portfolio evaluation
- [ ] Implement classical optimizer
- [ ] Implement CLI
- [ ] Write unit tests
- [ ] Generate plots

## Experiments

- [ ] Change risk-aversion parameter
- [ ] Change number of selected assets
- [ ] Measure brute-force scaling
- [ ] Document observations

---

# 35. Final learning checkpoint

Before starting Phase 2, you should be able to explain the following without looking at your code.

## Checkpoint 1 — binary representation

Explain why:

\[
x_i\in\{0,1\}
\]

can represent whether an asset is selected.

---

## Checkpoint 2 — portfolio return

Explain:

\[
R(x)=\mu^Tx
\]

and calculate it manually for a small example.

---

## Checkpoint 3 — portfolio risk

Explain:

\[
Risk(x)=x^T\Sigma x
\]

and expand it into individual and pairwise terms.

---

## Checkpoint 4 — optimization objective

Explain:

\[
f(x)=-\mu^Tx+\lambda x^T\Sigma x
\]

and why minimizing it creates a return/risk trade-off.

---

## Checkpoint 5 — combinatorial search

Explain why selecting \(k\) assets from \(n\) gives:

\[
\binom{n}{k}
\]

possible portfolios.

---

## Checkpoint 6 — the bridge to quantum computing

You should be able to explain why the objective contains terms of the form:

\[
x_i
\]

and:

\[
x_ix_j
\]

and why that makes the problem suitable for transformation into a **QUBO**.

Once these six checkpoints are solid, you are ready for:

\[
\boxed{
\text{Classical optimization}
\rightarrow
\text{QUBO}
\rightarrow
\text{Ising Hamiltonian}
\rightarrow
\text{QAOA}
}
\]

That transition is the core of the next phase.
