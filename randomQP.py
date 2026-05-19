# randomQP.py  — Degenerate Random QP generator
# Author: Jeremy Bertoncini
# Academic use only


import numpy as np
from scipy.sparse import (
    random as sp_random, eye, vstack, hstack, diags,
    csr_matrix, coo_matrix, isspmatrix_coo
)
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _build_MM_Q(n, density, rng):
    # Rank selection: 20% chance of near-LP degenerate rank 1-2
    if rng.random() < 0.20:
        rank = int(rng.integers(1, 3))
    else:
        rank = max(1, min(int(n * rng.uniform(0.30, 0.86)), n))

    # Two-cluster spectrum: large eigenvalues + small eigenvalues + zeros
    cond_target = 10 ** rng.choice(
        [rng.uniform(0, 1), rng.uniform(1, 3), rng.uniform(3, 6)],
        p=[0.60, 0.30, 0.10]
    )
    n_large = max(1, rank // 2)
    n_small = rank - n_large
    spectrum = np.concatenate([
        rng.uniform(cond_target * 0.1, cond_target, n_large),
        rng.uniform(1.0, 10.0, n_small),
        np.zeros(n - rank)
    ])

    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Q_dense = V @ np.diag(spectrum) @ V.T

    # Sparsify
    threshold = np.percentile(np.abs(Q_dense), max(0, (1.0 - density) * 100 - 5))
    Q_dense[np.abs(Q_dense) < threshold] = 0.0
    Q_sparse = csr_matrix(Q_dense)
    Q_sparse = 0.5 * (Q_sparse + Q_sparse.T)

    return Q_sparse.tocoo(), rank, cond_target


def _build_MM_constraint_matrix(m, n, density, rng,
                                frac_exact=0.15, frac_near=0.15):
    m_base  = max(1, int(m * (1.0 - frac_exact - frac_near)))
    m_exact = int(m * frac_exact)
    m_near  = m - m_base - m_exact

    # Base block
    A_base = sp_random(m_base, n, density=density, format='csr', random_state=rng).astype(float)
    A_base.data = A_base.data * 2.0 - 1.0

    # Exact linear dependencies
    if m_exact > 0:
        coeffs  = rng.integers(-2, 3, size=(m_exact, m_base)).astype(float)
        A_exact = csr_matrix(coeffs) @ A_base
    else:
        A_exact = csr_matrix((0, n))

    # Near-dependencies (tiny perturbations)
    if m_near > 0:
        eps_vals = 10 ** rng.uniform(-12, -6, m_near)
        A_near   = csr_matrix(rng.standard_normal((m_near, m_base))) @ A_base
        for i in range(m_near):
            perturb = sp_random(1, n, density=max(density, 0.05),
                                format='csr', random_state=rng)
            perturb.data *= eps_vals[i]
            A_near[i] = A_near[i] + perturb
    else:
        A_near = csr_matrix((0, n))

    A_full = vstack([A_base, A_exact, A_near], format='csr')
    m_full = A_full.shape[0]

    # Row/column ill-conditioning
    D_r = diags(10.0 ** rng.uniform(-1, 1, m_full))
    D_c = diags(10.0 ** rng.uniform(-1, 1, n))

    return (D_r @ A_full @ D_c).tocoo(), m_full


# ---------------------------------------------------------------------------
# Dense QP  (small problems, rank-1 Q)
# ---------------------------------------------------------------------------

class RandomQP_dense:
    def __init__(self, n, m, p, scaleQ, scaleC, scaleA, scaleG, scalex, seed=None):
        rng = np.random.default_rng(seed)
        self.n, self.m, self.p = n, m, p

        q_vec  = 2.0 * rng.random(n) - 1.0
        self.Q = 0.5 * np.outer(q_vec, q_vec)
        self.C = 2.0 * rng.random(n) - 1.0
        self.A = 2.0 * rng.random((m, n)) - 1.0
        self.G = 2.0 * rng.random((p, n)) - 1.0
        self.x = 2.0 * rng.random(n) - 1.0
        self.b = self.A @ self.x
        self.d = self.G @ self.x + np.abs(rng.random(p))

    def get_QP_settings(self):
        return self.Q, self.C, self.A, self.G, self.b, self.d, self.x


# ---------------------------------------------------------------------------
# Sparse QP  (Maros-Mészáros-like)
# ---------------------------------------------------------------------------

class RandomQP_sparse:
    """
    Sparse random QP replicating the structure of the Maros-Mészáros test set:
      - Two-cluster Q spectrum with controlled rank deficiency
      - Exact + near linear dependencies in A and G
      - Row/column ill-conditioning of constraint Jacobians
      - Degenerate primal point: random subset of inequalities exactly active
      - Mixed free/bounded variables
      - Dual initialisation
    """

    def __init__(self, n, m, p, density,
                 target_rd=1.0, target_rp=1.0,
                 seed=None, targeted=False):
        self.n, self.m, self.p = n, m, p
        self._rng = np.random.default_rng(seed)

        if targeted:
            self._targeted_init(n, m, p, density, target_rd, target_rp)
        else:
            self._standard_init(n, m, p, density)

    # ------------------------------------------------------------------

    def _build_G_with_box(self, G_coo, n, rng):
        n_free       = rng.integers(0, max(1, n // 4) + 1)
        free_vars    = rng.choice(n, int(n_free), replace=False)
        bounded_vars = np.setdiff1d(np.arange(n), free_vars)
        if len(bounded_vars) > 0:
            I_b   = eye(n, format='csr')[bounded_vars, :]
            G_coo = vstack([G_coo, I_b, -I_b])
        return G_coo.tocoo()

    def _dual_solve(self, rhs):
        A_T = self.A_coo.T.tocsc()
        G_T = self.G_coo.T.tocsc()
        M   = hstack([A_T, G_T]).tocsc()
        lhs = M.T @ M + 1e-6 * eye(M.shape[1], format='csc')
        try:
            z        = spsolve(lhs, M.T @ rhs)
            self.lam = z[:self.m]
            self.mu  = np.maximum(z[self.m:], 0.0)
        except Exception:
            self.lam = np.ones(self.m, dtype=np.float64)
            self.mu  = np.ones(self.p, dtype=np.float64)

    def _standard_init(self, n, m, p, density):
        rng = self._rng

        Q_coo, q_rank, q_cond = _build_MM_Q(n, density, rng)
        self.Q_coo, self._q_rank, self._q_cond = Q_coo, q_rank, q_cond
        self.Q_data, self.Q_row_indices, self.Q_col_indices = Q_coo.data, Q_coo.row, Q_coo.col

        A_coo, m_aug = _build_MM_constraint_matrix(
            m, n, rng.uniform(0.05, 0.25), rng, frac_exact=0.05, frac_near=0.05)
        self.A_coo, self.m = A_coo, m_aug
        self.A_data, self.A_row_indices, self.A_col_indices = A_coo.data, A_coo.row, A_coo.col

        G_coo, _ = _build_MM_constraint_matrix(
            p, n, rng.uniform(0.05, 0.25), rng, frac_exact=0.05, frac_near=0.05)
        self.G_coo = self._build_G_with_box(G_coo, n, rng)
        self.p     = self.G_coo.shape[0]
        self.G_data, self.G_row_indices, self.G_col_indices = (
            self.G_coo.data, self.G_coo.row, self.G_coo.col)

        self.C = 2.0 * rng.random(n) - 1.0

        x_sol = 2.0 * rng.random(n) - 1.0
        slack = np.abs(rng.random(self.p)) + 0.1
        active_idx = rng.choice(self.p,
                                int(rng.integers(0, max(1, self.p // 3) + 1)),
                                replace=False)
        slack[active_idx] = 0.0

        self.b = self.A_coo.dot(x_sol)
        self.d = self.G_coo.dot(x_sol) + slack
        self.s = slack.copy()
        self.x = np.zeros(n, dtype=np.float64)

        self._dual_solve(-(self.Q_coo.dot(self.x) + self.C))
        self.RP, self.RD = self._compute_residuals()

    def _targeted_init(self, n, m, p, density, target_rd, target_rp):
        rng = self._rng

        Q_coo, q_rank, q_cond = _build_MM_Q(n, density, rng)
        self.Q_coo, self._q_rank, self._q_cond = Q_coo, q_rank, q_cond
        self.Q_data, self.Q_row_indices, self.Q_col_indices = Q_coo.data, Q_coo.row, Q_coo.col

        A_coo, m_aug = _build_MM_constraint_matrix(m, n, density, rng)
        self.A_coo, self.m = A_coo, m_aug
        self.A_data, self.A_row_indices, self.A_col_indices = A_coo.data, A_coo.row, A_coo.col

        G_coo, _ = _build_MM_constraint_matrix(p, n, density, rng)
        self.G_coo = self._build_G_with_box(G_coo, n, rng)
        self.p     = self.G_coo.shape[0]
        self.G_data, self.G_row_indices, self.G_col_indices = (
            self.G_coo.data, self.G_coo.row, self.G_coo.col)

        self.C = np.zeros(n, dtype=np.float64)
        x_sol  = 2.0 * rng.random(n) - 1.0

        self.b = self.A_coo.dot(x_sol) - target_rp
        self.d = self.G_coo.dot(x_sol) + (2.0 - target_rp)
        self.x = np.zeros(n, dtype=np.float64)

        self._dual_solve(target_rd - self.Q_coo.dot(x_sol) - self.C)
        self.s  = np.ones(self.p, dtype=np.float64)
        self.RP, self.RD = self._compute_residuals()

    # ------------------------------------------------------------------

    def _compute_residuals(self):
        x = self.x
        rp_eq   = np.max(np.abs(self.A_coo.dot(x) - self.b))
        rp_ineq = float(np.max(np.maximum(self.G_coo.dot(x) - self.d, 0.0))) if self.p > 0 else 0.0
        RP = float(np.maximum(rp_eq, rp_ineq))

        grad = (self.Q_coo.dot(x) + self.C
                + self.A_coo.T.dot(self.lam)
                + self.G_coo.T.dot(self.mu))
        RD = float(np.max(np.abs(grad)))
        return RP, RD

    # ------------------------------------------------------------------

    def get_QP_settings(self):
        return (
            self.n, self.m, self.p,
            self.Q_data, self.Q_row_indices, self.Q_col_indices,
            self.C,
            self.A_data, self.A_row_indices, self.A_col_indices,
            self.G_data, self.G_row_indices, self.G_col_indices,
            self.b, self.d, self.x, self.lam, self.mu, self.s,
            self.RP, self.RD,
        )

    def precompute_dual_lhs(self):
        """Cache the Tikhonov LHS for repeated dual warm-starts."""
        A_T = self.A_coo.T.tocsc()
        G_T = self.G_coo.T.tocsc()
        self._M   = hstack([A_T, G_T]).tocsc()
        self._lhs = self._M.T @ self._M + 1e-6 * eye(self._M.shape[1], format='csc')

    def resample_starting_point(self, seed=None):
        """Resample primal point and dual variables; keep Q, A, G, C fixed."""
        rng   = np.random.default_rng(seed)
        x_sol = 2.0 * rng.random(self.n) - 1.0
        slack = np.abs(rng.random(self.p)) + 0.1
        slack[rng.choice(self.p,
                         int(rng.integers(0, max(1, self.p // 3) + 1)),
                         replace=False)] = 0.0

        self.b   = self.A_coo.dot(x_sol)
        self.d   = self.G_coo.dot(x_sol) + slack
        self.s   = slack.copy()
        self.x   = 2.0 * rng.random(self.n) - 1.0
        self.lam = np.zeros(self.m, dtype=np.float64)
        self.mu  = np.zeros(self.p, dtype=np.float64)
        self.RP, self.RD = self._compute_residuals()
        return self

    def print_matrix_sizes(self):
        print(f"Q : {self.Q_coo.shape}  nnz={self.Q_coo.nnz}  rank≈{self._q_rank}  cond≈{self._q_cond:.2e}")
        print(f"A : {self.A_coo.shape}  nnz={self.A_coo.nnz}")
        print(f"G : {self.G_coo.shape}  nnz={self.G_coo.nnz}")
        print(f"RP={self.RP:.3e}   RD={self.RD:.3e}")

    def check_matrix_type_CN(self, Q_coo=None):
        """Return (type_flag, effective_condition_number) for Q."""
        if Q_coo is None:
            Q_coo = self.Q_coo
        eigs = np.real(np.linalg.eigvalsh(Q_coo.toarray()))

        if   np.all(eigs >  1e-15): mtype = 2   # PD
        elif np.all(eigs >= -1e-15): mtype = 1  # PSD
        else:                        mtype = 0  # indefinite

        nz = eigs[np.abs(eigs) > 1e-20]
        cond_eff = (np.max(nz) / np.min(np.abs(nz))) if len(nz) > 0 else np.inf
        return mtype, cond_eff

    def check_empty_rows(self, matrix, diagonal_value=1e-10):
        """Fill structurally empty rows with a small diagonal entry."""
        if not isspmatrix_coo(matrix):
            raise ValueError("Matrix must be in COO format.")
        empty_rows = np.setdiff1d(np.arange(matrix.shape[0]), np.unique(matrix.row))
        for r in empty_rows:
            matrix.row  = np.append(matrix.row,  r)
            matrix.col  = np.append(matrix.col,  r)
            matrix.data = np.append(matrix.data, diagonal_value)
        return coo_matrix((matrix.data, (matrix.row, matrix.col)), shape=matrix.shape)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Dense QP ===")
    dqp = RandomQP_dense(n=10, m=4, p=6,
                         scaleQ=1, scaleC=1, scaleA=1, scaleG=1, scalex=1, seed=42)
    Q, C, A, G, b, d, x = dqp.get_QP_settings()
    print(f"Q shape: {Q.shape}  A shape: {A.shape}  G shape: {G.shape}")

    print("\n=== Sparse QP (standard) ===")
    sqp = RandomQP_sparse(n=50, m=20, p=30, density=0.15, seed=0)
    sqp.print_matrix_sizes()
    mtype, cond = sqp.check_matrix_type_CN()
    print(f"Q type: { {0:'indefinite',1:'PSD',2:'PD'}[mtype] }   effective cond: {cond:.3e}")

    print("\n=== Sparse QP (targeted RP/RD) ===")
    tqp = RandomQP_sparse(n=40, m=15, p=20, density=0.10,
                          target_rd=1e-3, target_rp=1e-3, seed=7, targeted=True)
    tqp.print_matrix_sizes()
