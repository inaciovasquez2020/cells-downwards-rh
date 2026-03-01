import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("m", "experiments/finite_weil_test.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

grid = np.linspace(-15.0, 15.0, 4000)

for M in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    basis = m.hermite_odd_basis(M, grid)
    H = m.GLE_matrix(basis, grid)
    w = np.linalg.eigvalsh(H)
    gap = w[1] - w[0]
    print("M =", M,
          "ground =", w[0],
          "first_excited =", w[1],
          "gap =", gap)
