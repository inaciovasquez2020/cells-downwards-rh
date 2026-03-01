import numpy as np
import importlib.util
import math

spec = importlib.util.spec_from_file_location("m", "experiments/finite_weil_test.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def smallest_positive_eigenvalue(M, tol=1e-10):
    w = np.linalg.eigvalsh(M)
    pos = w[w > tol]
    if len(pos) == 0:
        return None
    return float(pos[0])

print("Scaling enforcement test: X = exp(T) coupling")

for BAND in [1.0, 1.5, 2.0, 2.5, 3.0]:
    T = BAND * math.pi
    X = int(min(math.exp(T), 5000))

    m.PRIME_CUTOFF = X
    m.N_ZEROS = 2000

    grid = np.linspace(-m.LOG_CUTOFF, m.LOG_CUTOFF, m.GRID_N)
    basis = m.hermite_odd_basis(m.BASIS_M, grid)

    Z = m.zero_term_matrix(basis, grid)
    P = m.prime_term_matrix(basis, grid)
    G = m.GLE_matrix(basis, grid)

    H = Z - P + m.GLE_WEIGHT * G
    H = 0.5 * (H + H.T)

    gap = smallest_positive_eigenvalue(H)

    print("BAND =", BAND,
          "T =", round(T,3),
          "X =", X,
          "gap =", gap)
