import numpy as np
import importlib.util


def load_module():
    spec = importlib.util.spec_from_file_location("m", "experiments/finite_weil_test.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def compute(M: int, L: float, N: int):
    m = load_module()
    grid = np.linspace(-L, L, N)
    basis = m.hermite_odd_basis(M, grid)
    H = m.GLE_matrix(basis, grid)
    w = np.linalg.eigvalsh(H)
    lam0 = float(w[0])
    lam1 = float(w[1])
    gap = float(w[1] - w[0])
    return lam0, lam1, gap


def test_spectral_floor_stable_M80_M90():
    L = 15.0
    N = 4000

    lam0_80, lam1_80, gap_80 = compute(80, L, N)
    lam0_90, lam1_90, gap_90 = compute(90, L, N)

    assert abs(lam0_90 - lam0_80) < 1e-6
    assert abs(lam1_90 - lam1_80) < 1e-6
    assert abs(gap_90 - gap_80) < 1e-6

    assert gap_90 > 0.0


def test_gap_value_regression_loose():
    L = 15.0
    N = 4000
    lam0, lam1, gap = compute(90, L, N)

    assert -4.2 < lam0 < -3.6
    assert -3.6 < lam1 < -2.6
    assert 0.6 < gap < 1.0
