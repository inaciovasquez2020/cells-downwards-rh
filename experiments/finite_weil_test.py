import numpy as np

LOG_CUTOFF = 60
PRIME_CUTOFF = 5000

ZEROS = [
    14.134725,21.022040,25.010858,30.424876,32.935061,
    37.586178,40.918719,43.327073,48.005150,49.773832,
    52.970321,56.446248,59.347044,60.831780,65.112544,
    67.079811,69.546401,72.067158,75.704690,77.144840
]

# ---- GLE / JN / PS weights (adjustable) ----
GLE_WEIGHT = 2.1261.0
JN_WEIGHT  = 1.0
PS_WEIGHT  = 1.0

# ---- utilities ----

def primes_up_to(n):
    sieve = np.ones(n+1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n**0.5)+1):
        if sieve[i]:
            sieve[i*i:n+1:i] = False
    return np.nonzero(sieve)[0]

def odd_basis(m, grid):
    return np.array([np.sin((k+1)*grid) for k in range(m)])

# ---- PRIME TERM ----
def prime_term_matrix(basis, grid):
    m = basis.shape[0]
    Q = np.zeros((m,m))
    primes = primes_up_to(PRIME_CUTOFF)
    for p in primes:
        weight = np.log(p)/np.sqrt(p)
        t = np.log(p)
        idx = np.argmin(np.abs(grid - t))
        v = basis[:,idx]
        Q += weight * np.outer(v,v)
    return Q

# ---- ZERO TERM ----
def zero_term_matrix(basis, grid):
    m = basis.shape[0]
    Q = np.zeros((m,m))
    ZERO_WEIGHT = 20.0
    for gamma in ZEROS:
        idx = np.argmin(np.abs(grid - gamma))
        v = basis[:,idx]
        Q += ZERO_WEIGHT * np.outer(v, v)
    return Q

# ---- GLE (Gamma-like even weight) ----
def GLE_matrix(basis, grid):
    m = basis.shape[0]
    Q = np.zeros((m,m))
    weight = 1.0 / (1.0 + grid**2)   # placeholder decay
    for i in range(m):
        for j in range(m):
            Q[i,j] = np.sum(weight * basis[i]*basis[j])
    return Q

# ---- JN (symmetry projector-like correction) ----
def JN_matrix(basis):
    m = basis.shape[0]
    return np.eye(m)  # identity correction placeholder

# ---- PS (prime scaling correction) ----
def PS_matrix(basis, grid):
    m = basis.shape[0]
    Q = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            Q[i,j] = 0.01 * np.dot(basis[i], basis[j])
    return Q

def run_test(m):
    grid = np.linspace(-LOG_CUTOFF,LOG_CUTOFF,10000)
    basis = odd_basis(m,grid)

    Q = (
        zero_term_matrix(basis,grid)
        - prime_term_matrix(basis,grid)
        + GLE_WEIGHT * GLE_matrix(basis,grid)
        + JN_WEIGHT  * JN_matrix(basis)
        - PS_WEIGHT  * PS_matrix(basis,grid)
    )

    eigvals = np.linalg.eigvalsh(Q)
    print(f"m={m}, min eigenvalue={eigvals[0]}")

if __name__ == "__main__":
    for m in [5,10,15,20]:
        run_test(m)
