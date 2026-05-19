#!/usr/bin/env python3
"""Quick sanity-check for RandomQP_sparse problem generation."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from randomQP import RandomQP_sparse


# ---------------------------------------------------------------------------
# Problem generation (mirrors BenchmarkQP._sample_random_problems)
# ---------------------------------------------------------------------------

def sample_problems(n_problems: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    problems = []
    for i in range(n_problems):
        n = int(np.exp(rng.uniform(np.log(100), np.log(500))))

        u_m = rng.random()
        if u_m < 0.15:
            m_ratio = float(np.exp(rng.uniform(np.log(0.001), np.log(0.05))))
        elif u_m < 0.85:
            m_ratio = float(rng.uniform(0.05, 0.60))
        else:
            m_ratio = float(rng.uniform(0.60, 0.92))
        m = max(1, int(n * min(m_ratio, 0.92)))

        u_p = rng.random()
        if u_p < 0.03:
            p_ratio = rng.uniform(10.0, 130.0)
        elif u_p < 0.20:
            p_ratio = float(np.exp(rng.uniform(np.log(1.0), np.log(1.9))))
        elif u_p < 0.80:
            p_ratio = float(rng.uniform(1.9, 2.5))
        else:
            p_ratio = float(rng.uniform(2.6, 4.2))
        p = max(1, int(n * p_ratio))

        if n <= 50:    d_lo, d_hi = 0.04, 0.20
        elif n <= 150: d_lo, d_hi = 0.01, 0.08
        elif n <= 600: d_lo, d_hi = 0.004, 0.025
        else:          d_lo, d_hi = 0.0005, 0.005
        density = float(np.exp(rng.uniform(np.log(d_lo), np.log(d_hi))))

        qp_seed = int(rng.integers(0, 2**31))
        problems.append(dict(
            problem_id=f"rand{i:04d}_n{n}_m{m}_p{p}",
            n=n, m=m, p=p, density=density, seed=qp_seed,
        ))
    return problems


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    N_PROBLEMS = 10
    SEED       = 42

    specs = sample_problems(N_PROBLEMS, seed=SEED)

    ok, failed = 0, 0
    for spec in specs:
        pid = spec["problem_id"]
        try:
            qp = RandomQP_sparse(
                n=spec["n"], m=spec["m"], p=spec["p"],
                density=spec["density"], seed=spec["seed"],
            )
            mtype, cond = qp.check_matrix_type_CN()
            type_str = {0: "indef", 1: "PSD", 2: "PD"}.get(mtype, "?")
            print(f"[OK]  {pid}")
            qp.print_matrix_sizes()
            print(f"      Q type: {type_str}   cond: {cond:.2e}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {pid}: {e}")
            failed += 1

    print(f"\n{ok}/{N_PROBLEMS} generated successfully, {failed} failed.")


if __name__ == "__main__":
    main()
