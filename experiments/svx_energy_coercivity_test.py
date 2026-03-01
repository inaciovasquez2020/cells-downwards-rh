import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("m", "experiments/finite_weil_test.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def smallest_positive_eigenvalue(M, tol=1e-10):
    w = np.linalg.eigvalsh(M)
    pos = w[w > tol]
    if len(pos) == 0:
        return None
    return float(pos[0])

grid = np.linspace(-m.LOG_CUTOFF, m.LOG_CUTOFF, m.GRID_N)
basis = m.hermite_odd_basis(m.BASIS_M, grid)

Z = m.zero_term_matrix(basis, grid)
P = m.prime_term_matrix(basis, grid)
G = m.GLE_matrix(basis, grid)

H = Z - P + m.GLE_WEIGHT * G
H = 0.5 * (H + H.T)

print("Baseline gap =", smallest_positive_eigenvalue(H))
print("Energy-weighted coercivity test")

for gamma in [0.1, 0.5, 1.0, 2.0, 5.0]:
    H_mod = H + gamma * G
    H_mod = 0.5 * (H_mod + H_mod.T)
    gap = smallest_positive_eigenvalue(H_mod)
    print("gamma =", gamma, "gap =", gap)
