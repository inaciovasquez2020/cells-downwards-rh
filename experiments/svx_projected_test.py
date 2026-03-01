import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("m", "experiments/finite_weil_test.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

grid = np.linspace(-m.LOG_CUTOFF, m.LOG_CUTOFF, m.GRID_N)
basis = m.hermite_odd_basis(m.BASIS_M, grid)

Z = m.zero_term_matrix(basis, grid)
P = m.prime_term_matrix(basis, grid)
G = m.GLE_matrix(basis, grid)

H = Z - P + m.GLE_WEIGHT * G
H = 0.5 * (H + H.T)

w, V = np.linalg.eigh(H)

idx = np.argsort(w)
w = w[idx]
V = V[:, idx]

lambda_min = w[0]
v_min = V[:, 0]

print("original smallest eigenvalue =", lambda_min)

P_perp = np.eye(H.shape[0]) - np.outer(v_min, v_min)
H_proj = P_perp @ H @ P_perp
H_proj = 0.5 * (H_proj + H_proj.T)

w_proj = np.linalg.eigvalsh(H_proj)
pos = w_proj[w_proj > 1e-10]

if len(pos) == 0:
    print("projected gap = None")
else:
    print("projected smallest positive eigenvalue =", float(pos[0]))
