import numpy as np

if hasattr(np, "trapezoid"):
    integrate = np.trapezoid
else:
    integrate = np.trapz

import mpmath as mp

# ==============================
# CONFIGURATION
# ==============================

BASIS_M = 60
N_ZEROS = 200
PRIME_CUTOFF = 5000
GRID_N = 12000
LOG_CUTOFF = 8.0

GLE_WEIGHT = 5.50575
ZERO_WEIGHT = 20.0
OFFCRIT_EPS = 0.0

mp.mp.dps = 50

# ==============================
# Hermite Odd Orthonormal Basis
# ==============================

def hermite_odd_basis(m, x):
    from numpy.polynomial.hermite import hermval
    basis = []
    for k in range(m):
        coeff = [0] * (2*k + 2)
        coeff[2*k + 1] = 1.0
        H = hermval(x, coeff)
        H = H * np.exp(-0.5 * x * x)
        norm = np.sqrt(integrate(H * H, x))
        if norm == 0:
            norm = 1.0
        H = H / norm
        basis.append(H)
    return np.array(basis)

# ==============================
# Prime Sieve
# ==============================

def simple_primes_upto(n):
    sieve = np.ones(n+1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i:n+1:i] = False
    return np.nonzero(sieve)[0]

# ==============================
# Zeta Zeros
# ==============================

def critical_zeros(n):
    return [float(mp.zetazero(k).imag) for k in range(1, n+1)]

# ==============================
# Kernel Components
# ==============================

def zero_term_matrix(basis, grid):
    m = basis.shape[0]
    Q = np.zeros((m, m))

    zeros = critical_zeros(N_ZEROS)

    for gamma in zeros:
        vals = []
        for j in range(m):
            n = j
            phase = (1j)**(-n)
            psi_gamma = np.interp(gamma, grid, basis[j])
            vals.append((phase * psi_gamma).real)
        vals = np.array(vals)
        Q += np.outer(vals, vals)

    Q /= max(len(zeros), 1)
    return Q
def prime_term_matrix(basis, grid):
    m = basis.shape[0]
    Q = np.zeros((m, m))

    primes = simple_primes_upto(PRIME_CUTOFF)

    for p in primes:
        logp = np.log(p)
        for k in range(1, 8):  # prime powers up to m=7
            val = k * logp
            if val > LOG_CUTOFF:
                break
            weight = logp / (p**(k/2))
            vals = np.array([np.interp(val, grid, b) for b in basis])
            Q += weight * np.outer(vals, vals)

    return Q
def GLE_matrix(basis, grid):
    m = basis.shape[0]
    Q = np.zeros((m, m))

    # Precompute digamma term safely
    dig = np.array([
        float(mp.re(mp.digamma(0.25 + 0.5j * t)))
        for t in grid
    ])

    for i in range(m):
        for j in range(m):
            integrand = basis[i] * basis[j] * dig
            Q[i, j] = integrate(integrand, grid)

    return Q

# ==============================
# Main Run
# ==============================

def run(m_basis=BASIS_M, n_zeros=N_ZEROS):
    m_basis = min(m_basis, 90)
    global BASIS_M, N_ZEROS
    BASIS_M = m_basis
    N_ZEROS = n_zeros

    grid = np.linspace(-LOG_CUTOFF, LOG_CUTOFF, GRID_N)
    basis = hermite_odd_basis(m_basis, grid)

    Q_zero = zero_term_matrix(basis, grid)
    Q_prime = prime_term_matrix(basis, grid)
    Q_gle = GLE_matrix(basis, grid)

    Q = Q_zero - Q_prime + GLE_WEIGHT * Q_gle
    Q = 0.5*(Q + Q.T)

    eig = np.linalg.eigvalsh(Q)
    minEig = eig[0]
    maxEig = eig[-1]
    cond = float(abs(maxEig) / max(abs(minEig), 1e-18))

    print("HAVE_MPMATH=True")
    print(f"m_basis={m_basis}  n_zeros={n_zeros}  PRIME_CUTOFF={PRIME_CUTOFF}  GRID_N={GRID_N}")
    print(f"GLE_WEIGHT={GLE_WEIGHT}  ZERO_WEIGHT={ZERO_WEIGHT}  OFFCRIT_EPS={OFFCRIT_EPS}")
    print(f"minEig={minEig:.6f}  maxEig={maxEig:.6f}  cond~={cond:.3e}")

# ==============================
# Off-Critical Sweep
# ==============================

def sweep_offcrit(eps_list, m_basis=60, n_zeros=200):
    global OFFCRIT_EPS
    print("\nOFF-CRITICAL SWEEP:")
    for eps in eps_list:
        OFFCRIT_EPS = eps
        run(m_basis, n_zeros)

# ==============================
# Entry
# ==============================

if __name__ == "__main__":
    run()
    sweep_offcrit([0.0, 0.02, 0.05, 0.1, 0.2])
