import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("m", "experiments/finite_weil_test.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

print("Testing structural shift H -> H + alpha I (fixed projection)")

grid = np.linspace(-10.0, 10.0, 4000)

basis = m.hermite_odd_basis(m.BASIS_M, grid)
H0 = m.GLE_matrix(basis, grid)

w0, v0 = np.linalg.eigh(H0)
tol = 1e-12

null = np.where(np.abs(w0) <= tol)[0]
if len(null) > 0:
    Vnull = v0[:, null]
    P = np.eye(H0.shape[0]) - Vnull @ Vnull.T
else:
    P = np.eye(H0.shape[0])

def gap(alpha: float) -> float:
    Ha = H0 + alpha * np.eye(H0.shape[0])
    Hp = P @ Ha @ P
    w = np.linalg.eigvalsh(Hp)
    wpos = w[w > tol]
    return float(wpos.min()) if len(wpos) else float("nan")

for a in [0.0, 1e-6, 1e-5, 1e-4, 1e-3]:
    print("alpha =", a, "gap =", gap(a))
