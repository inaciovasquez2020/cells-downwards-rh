import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("m", "experiments/finite_weil_test.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def decay_rate(M):
    grid = np.linspace(-15.0, 15.0, 4000)
    basis = m.hermite_odd_basis(M, grid)
    H = m.GLE_matrix(basis, grid)

    avg = []
    for d in range(1, min(30, M)):
        vals = []
        for i in range(M - d):
            vals.append(abs(H[i, i + d]))
        avg.append(np.mean(vals))

    dist = np.arange(1, len(avg) + 1)
    log_vals = np.log(avg)

    slope, _ = np.polyfit(dist, log_vals, 1)
    return -slope

for M in [30, 40, 50, 60, 70]:
    print("M =", M, "decay rate ≈", decay_rate(M))
