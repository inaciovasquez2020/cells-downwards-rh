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
G = 0.5 * (G + G.T)

print("Baseline ℓ² gap =", smallest_positive_eigenvalue(H))

wG, VG = np.linalg.eigh(G)
eps = 1e-12
wG_clipped = np.clip(wG, eps, None)
G_inv_sqrt = VG @ np.diag(1.0 / np.sqrt(wG_clipped)) @ VG.T

H_energy = G_inv_sqrt @ H @ G_inv_sqrt
H_energy = 0.5 * (H_energy + H_energy.T)

print("G-metric gap =", smallest_positive_eigenvalue(H_energy))
