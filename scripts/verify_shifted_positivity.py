import json
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


def build_H(M: int, L: float, N: int):
    m = load_module()
    grid = np.linspace(-L, L, N)
    basis = m.hermite_odd_basis(M, grid)
    H = m.GLE_matrix(basis, grid)
    return H


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    L = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    trials = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    np.random.seed(seed)
    t0 = time.time()

    H = build_H(M, L, N)
    w, V = np.linalg.eigh(H)
    lam0 = float(w[0])
    lam1 = float(w[1])
    gap = float(w[1] - w[0])
    v0 = V[:, 0].astype(float)

    Hs = H - lam0 * np.eye(H.shape[0])

    min_q = float("inf")
    max_q = float("-inf")
    min_ratio = float("inf")

    for _ in range(trials):
        x = np.random.randn(H.shape[0])
        x = x - (x @ v0) * v0
        nx = np.linalg.norm(x)
        if nx == 0.0:
            continue
        x /= nx

        q = float(x @ (Hs @ x))
        min_q = min(min_q, q)
        max_q = max(max_q, q)

        ratio = q
        min_ratio = min(min_ratio, ratio)

    out = {
        "schema": "cells-downwards-rh.shifted-positivity.v1",
        "method": {
            "basis": "hermite_odd",
            "M": M,
            "domain_L": L,
            "grid_N": N,
            "trials": trials,
            "seed": seed,
        },
        "spectrum": {
            "lambda0": lam0,
            "lambda1": lam1,
            "gap": gap,
        },
        "checks": {
            "min_rayleigh_shifted_over_unit_vectors_orth_v0": min_q,
            "max_rayleigh_shifted_over_unit_vectors_orth_v0": max_q,
            "min_ratio_over_unit_vectors_orth_v0": min_ratio,
            "expected_lower_bound_gap": gap,
        },
        "runtime_sec": float(time.time() - t0),
    }

    Path("certificates").mkdir(parents=True, exist_ok=True)
    p = Path("certificates") / f"shifted_positivity_M{M}_L{L:g}_N{N}_tr{trials}_seed{seed}.json"
    p.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print("gap =", gap)
    print("min_q =", min_q)
    print("max_q =", max_q)


if __name__ == "__main__":
    main()
