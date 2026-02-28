import numpy as np
import importlib.util


def load_module():
    spec = importlib.util.spec_from_file_location("m", "experiments/finite_weil_test.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_H(M: int, L: float, N: int):
    m = load_module()
    grid = np.linspace(-L, L, N)
    basis = m.hermite_odd_basis(M, grid)
    H = m.GLE_matrix(basis, grid)
    return H


def test_shifted_operator_positive_on_orthocomp_random():
    M, L, N = 90, 15.0, 4000
    H = build_H(M, L, N)

    w, V = np.linalg.eigh(H)
    lam0 = float(w[0])
    gap = float(w[1] - w[0])
    v0 = V[:, 0].astype(float)

    Hs = H - lam0 * np.eye(H.shape[0])

    rng = np.random.default_rng(0)

    min_q = 1e9
    for _ in range(200):
        x = rng.standard_normal(H.shape[0])
        x = x - (x @ v0) * v0
        nx = np.linalg.norm(x)
        if nx == 0.0:
            continue
        x /= nx
        q = float(x @ (Hs @ x))
        min_q = min(min_q, q)

    assert min_q > 0.0
    assert min_q >= 0.25 * gap
