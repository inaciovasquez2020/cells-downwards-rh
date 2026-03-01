import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("m", "experiments/finite_weil_test.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

M = 40
grid = np.linspace(-15.0, 15.0, 4000)

basis = m.hermite_odd_basis(M, grid)
H = m.GLE_matrix(basis, grid)

diag = np.diag(H)
offdiag_norm = np.linalg.norm(H - np.diag(diag))

print("diag mean:", np.mean(diag))
print("diag std:", np.std(diag))
print("offdiag Frobenius norm:", offdiag_norm)
print("diag Frobenius norm:", np.linalg.norm(np.diag(diag)))
