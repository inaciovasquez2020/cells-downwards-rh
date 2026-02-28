import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import importlib.util


def load_module():
    spec = importlib.util.spec_from_file_location("m", "experiments/finite_weil_test.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def spectral_floor(M: int, L: float, N: int):
    m = load_module()
    grid = np.linspace(-L, L, N)
    basis = m.hermite_odd_basis(M, grid)
    H = m.GLE_matrix(basis, grid)
    w = np.linalg.eigvalsh(H)
    lam0 = float(w[0])
    lam1 = float(w[1])
    gap = float(w[1] - w[0])
    return lam0, lam1, gap


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    L = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 4000

    t0 = time.time()
    lam0, lam1, gap = spectral_floor(M=M, L=L, N=N)
    dt = time.time() - t0

    out = {
        "schema": "cells-downwards-rh.spectral-floor.v1",
        "method": {
            "basis": "hermite_odd",
            "M": M,
            "domain_L": L,
            "grid_N": N,
            "eigensolver": "numpy.linalg.eigvalsh",
        },
        "results": {
            "lambda0": lam0,
            "lambda1": lam1,
            "gap": gap,
        },
        "runtime_sec": float(dt),
        "env": {
            "python": sys.version.replace("\n", " "),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }

    Path("certificates").mkdir(parents=True, exist_ok=True)
    p = Path("certificates") / f"spectral_floor_M{M}_L{L:g}_N{N}.json"
    p.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out["results"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
