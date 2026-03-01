import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("m", "experiments/finite_weil_test.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

grid = np.linspace(-10.0, 10.0, 4000)

for M in [10, 20, 30, 40, 50]:
    basis = m.hermite_odd_basis(M, grid)
    H = m.GLE_matrix(basis, grid)
    w = np.linalg.eigvalsh(H)
    print("M =", M, "min eigenvalue =", w[0])
