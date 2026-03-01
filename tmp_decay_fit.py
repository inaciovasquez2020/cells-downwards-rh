import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("m", "experiments/finite_weil_test.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

M = 60
grid = np.linspace(-15.0, 15.0, 4000)

basis = m.hermite_odd_basis(M, grid)
H = m.GLE_matrix(basis, grid)

avg_by_distance = []

for d in range(M):
    vals = []
    for i in range(M):
        j = i + d
        if j < M:
            vals.append(abs(H[i, j]))
    if vals:
        avg_by_distance.append(np.mean(vals))

dist = np.arange(1, 31)
vals = np.array(avg_by_distance[1:31])

log_vals = np.log(vals)

coeffs = np.polyfit(dist, log_vals, 1)
slope, intercept = coeffs

print("Exponential fit slope ≈", slope)
print("If exponential, decay rate ≈", -slope)
