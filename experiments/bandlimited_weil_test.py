import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location("m", "experiments/finite_weil_test.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

BAND = 3.0
N_FREQ = 800
M_BASIS = 40

m.N_ZEROS = 20
m.PRIME_CUTOFF = 500

freq = np.linspace(-BAND, BAND, N_FREQ)
df = float(freq[1] - freq[0])

xgrid = np.linspace(-m.LOG_CUTOFF, m.LOG_CUTOFF, m.GRID_N)
dx = float(xgrid[1] - xgrid[0])

def bandlimited_basis(M, freq, band):
    B = []
    edges = np.linspace(-band, band, M+1)
    for k in range(M):
        phi_hat = np.zeros_like(freq)
        mask = (freq >= edges[k]) & (freq < edges[k+1])
        phi_hat[mask] = 1.0
        B.append(phi_hat)
    return np.array(B, dtype=np.float64)


def normalize_freq_rows(Bhat):
    B = Bhat.copy()
    for i in range(B.shape[0]):
        nrm2 = float(np.sum(B[i] * B[i]) * df)
        if nrm2 <= 0:
            continue
        B[i] /= np.sqrt(nrm2)
    return B

def invF2pi(bhat, tgrid, xgrid):
    K = np.exp(2j * np.pi * np.outer(xgrid, tgrid))
    fx = (K @ bhat) * df
    return fx.real

def zero_term_band(Bhat, freq):
    Q = np.zeros((Bhat.shape[0], Bhat.shape[0]), dtype=np.float64)
    zeros = m.critical_zeros(m.N_ZEROS)
    for gamma in zeros:
        t = gamma / (2 * np.pi)
        vals = np.array([np.interp(t, freq, b) for b in Bhat], dtype=np.float64)
        Q += np.outer(vals, vals)
    Q /= max(len(zeros), 1)
    return Q

def prime_term_band(Bhat, freq):
    Q = np.zeros((Bhat.shape[0], Bhat.shape[0]), dtype=np.float64)
    primes = m.simple_primes_upto(m.PRIME_CUTOFF)
    for p in primes:
        logp = np.log(p)
        t = logp / (2 * np.pi)
        vals = np.array([np.interp(t, freq, b) for b in Bhat], dtype=np.float64)
        Q += (logp / np.sqrt(p)) * np.outer(vals, vals)
    return Q

Bhat = bandlimited_basis(M_BASIS, freq, BAND)
Bhat = normalize_freq_rows(Bhat)

Bx = np.array([invF2pi(Bhat[i], freq, xgrid) for i in range(M_BASIS)], dtype=np.float64)

Z = zero_term_band(Bhat, freq)
P = prime_term_band(Bhat, freq)
G = m.GLE_matrix(Bx, xgrid)

Z = 0.5 * (Z + Z.T)
P = 0.5 * (P + P.T)
G = 0.5 * (G + G.T)

w = float(getattr(m, "GLE_WEIGHT", 0.0))
Q = Z - P + w * G
Q = 0.5 * (Q + Q.T)

ev = np.linalg.eigvalsh(Q)

print("BAND", BAND, "N_ZEROS", m.N_ZEROS, "PRIME_CUTOFF", m.PRIME_CUTOFF, "GLE_WEIGHT", w)
print("norms", float(np.linalg.norm(Z)), float(np.linalg.norm(P)), float(np.linalg.norm(G)))
print("minEig", float(ev[0]), "maxEig", float(ev[-1]))
