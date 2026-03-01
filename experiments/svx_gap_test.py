import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("m", "experiments/finite_weil_test.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def smallest_positive_eigenvalue(M, tol=1e-8):
    w = np.linalg.eigvalsh(M)
    pos = w[w > tol]
    if len(pos) == 0:
        return None
    return float(pos[0])

def SVX(H, delta_bound):
    lam = smallest_positive_eigenvalue(H)
    if lam is None:
        return None
    return lam - delta_bound

grid = np.linspace(-m.LOG_CUTOFF, m.LOG_CUTOFF, m.GRID_N)
basis = m.hermite_odd_basis(m.BASIS_M, grid)

Z = m.zero_term_matrix(basis, grid)
P = m.prime_term_matrix(basis, grid)
G = m.GLE_matrix(basis, grid)

H = Z - P + m.GLE_WEIGHT * G
H = 0.5 * (H + H.T)

delta_bound = 0.1

lam_pos = smallest_positive_eigenvalue(H)
svx_val = SVX(H, delta_bound)

print("delta_bound =", delta_bound)
print("smallest_positive_eigenvalue =", lam_pos)
print("SVX(H) =", svx_val)
