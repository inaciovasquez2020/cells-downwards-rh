import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("m", "experiments/finite_weil_test.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

grid = np.linspace(-10.0, 10.0, 4000)
basis = m.hermite_odd_basis(40, grid)
H = m.GLE_matrix(basis, grid)

w, v = np.linalg.eigh(H)
vec = v[:, 0]

print("Most negative eigenvalue:", w[0])
print("Largest magnitude coefficient index:", np.argmax(np.abs(vec)))
print("Coefficient values (first 10):", vec[:10])
