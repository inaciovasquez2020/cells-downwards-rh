import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("m", "experiments/finite_weil_test.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

for L in [6.0, 8.0, 10.0, 12.0, 15.0]:
    grid = np.linspace(-L, L, 4000)
    basis = m.hermite_odd_basis(40, grid)
    H = m.GLE_matrix(basis, grid)
    w = np.linalg.eigvalsh(H)
    print("L =", L, "min eigenvalue =", w[0])
