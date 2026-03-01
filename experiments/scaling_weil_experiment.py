import numpy as np
import importlib.util
import math

spec = importlib.util.spec_from_file_location("m", "experiments/finite_weil_test.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def zeros_up_to_height(T):
    zeros = []
    k = 50
    while True:
        zs = m.critical_zeros(k)
        zeros = [g for g in zs if g <= T]
        if len(zs) < k or zs[-1] > T:
            break
        k *= 2
    return zeros

def bandlimited_basis(M, freq, band):
    B = []
    edges = np.linspace(-band, band, M+1)
    for k in range(M):
        phi_hat = np.zeros_like(freq)
        mask = (freq >= edges[k]) & (freq < edges[k+1])
        phi_hat[mask] = 1.0
        B.append(phi_hat)
    return np.array(B, dtype=np.float64)

def normalize_freq_rows(Bhat, df):
    B = Bhat.copy()
    for i in range(B.shape[0]):
        nrm2 = float(np.sum(B[i] * B[i]) * df)
        if nrm2 > 0:
            B[i] /= math.sqrt(nrm2)
    return B

def invF2pi(bhat, tgrid, xgrid, df):
    K = np.exp(2j * np.pi * np.outer(xgrid, tgrid))
    fx = (K @ bhat) * df
    return fx.real

def run_for_band(BAND):
    T = 2 * np.pi * BAND
    MAX_PRIME = 5000
    X_raw = int(math.exp(T))
    X = min(X_raw, MAX_PRIME)

    N_FREQ = 800
    M_BASIS = 40

    freq = np.linspace(-BAND, BAND, N_FREQ)
    df = float(freq[1] - freq[0])

    Bhat = bandlimited_basis(M_BASIS, freq, BAND)
    Bhat = normalize_freq_rows(Bhat, df)

    xgrid = np.linspace(-m.LOG_CUTOFF, m.LOG_CUTOFF, m.GRID_N)
    Bx = np.array([invF2pi(Bhat[i], freq, xgrid, df) for i in range(M_BASIS)], dtype=np.float64)

    zeros = zeros_up_to_height(T)
    print("  T", float(T), "zeros_used", len(zeros), "X_raw", X_raw, "X_used", X)

    Z = np.zeros((M_BASIS, M_BASIS))
    for gamma in zeros:
        t = gamma / (2 * np.pi)
        vals = np.array([np.interp(t, freq, b) for b in Bhat])
        Z += np.outer(vals, vals)
    if len(zeros) > 0:
        Z /= len(zeros)

    P = np.zeros((M_BASIS, M_BASIS))
    primes = m.simple_primes_upto(X)
    for p in primes:
        logp = np.log(p)
        t = logp / (2 * np.pi)
        vals = np.array([np.interp(t, freq, b) for b in Bhat])
        P += (logp / np.sqrt(p)) * np.outer(vals, vals)

    G = m.GLE_matrix(Bx, xgrid)

    Q = Z - P + m.GLE_WEIGHT * G
    Q = 0.5 * (Q + Q.T)

    ev = np.linalg.eigvalsh(Q)
    return float(ev[0]), float(ev[-1]), float(np.linalg.norm(Z)), float(np.linalg.norm(P))

for BAND in [2.5, 3.0, 3.5]:
    try:
        minEig, maxEig, nZ, nP = run_for_band(BAND)
        print("BAND", BAND, "minEig", minEig, "maxEig", maxEig, "||Z||", nZ, "||P||", nP)
    except Exception as e:
        print("BAND", BAND, "failed:", str(e))
