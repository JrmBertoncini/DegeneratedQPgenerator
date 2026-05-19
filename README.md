# randomQP

Random quadratic program generator for benchmarking and RL-based solver training.

---

## Problem class

$$
\min_{x \in \mathbb{R}^n} \quad \tfrac{1}{2} x^\top Q x + c^\top x
\qquad \text{s.t.} \quad Ax = b, \quad Gx \leq d
$$

$Q \in \mathbb{R}^{n \times n}$ is symmetric positive semidefinite,
$A \in \mathbb{R}^{m \times n}$, $G \in \mathbb{R}^{p \times n}$.

---

## Sampling distribution

Each problem is drawn by first sampling a regime:

| Regime | Weight | Characteristics |
|---|---|---|
| `standard` | 0.45 | balanced $m$, $p \sim \mathcal{U}(n, 3n)$ |
| `lp_like` | 0.25 | $m \leq 1$, dense $G$ |
| `high_p` | 0.20 | $p \in [3n, 33n]$, few equalities |
| `equality_heavy` | 0.10 | $m \in [0.4n, 0.9n]$, $p = 0$ |

Problem size is sampled as $n \sim \text{LogUniform}(5, 50)$.

