import numpy as np
import importlib.util
from numpy.linalg import eigvalsh

spec = importlib.util.spec_from_file_location("m", "experiments/finite_weil_test.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

grid = np.linspace(-15.0, 15.0, 4000)

Ms = np.array([10, 20, 30, 40, 50, 60, 70])
gaps = []

for M in Ms:
    basis = m.hermite_odd_basis(M, grid)
    H = m.GLE_matrix(basis, grid)
    w = eigvalsh(H)
    gap = w[1] - w[0]
    gaps.append(gap)
    print("M =", M, "gap =", gap)

gaps = np.array(gaps)

# Fit gap ≈ a*(1/sqrt(M)) + b
inv_sqrtM = 1.0 / np.sqrt(Ms)
coeffs = np.polyfit(inv_sqrtM, gaps, 1)
a, b = coeffs

print("\nFit gap ≈ a*(1/sqrt(M)) + b")
print("a =", a)
print("b (limit M→∞) ≈", b)
