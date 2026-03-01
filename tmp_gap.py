import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("m", "experiments/finite_weil_test.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

grid = np.linspace(-15.0, 15.0, 4000)
basis = m.hermite_odd_basis(40, grid)
H = m.GLE_matrix(basis, grid)

w = np.linalg.eigvalsh(H)

print("ground state:", w[0])
print("first excited:", w[1])
print("spectral gap:", w[1] - w[0])
