import numpy as np

# ---- parameters ----
LOG_CUTOFF = 50
PRIME_CUTOFF = 2000

# ---- utilities ----

def primes_up_to(n):
    sieve = np.ones(n+1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n**0.5)+1):
        if sieve[i]:
            sieve[i*i:n+1:i] = False
    return np.nonzero(sieve)[0]

def von_mangoldt(n):
    # crude implementation
    for p in range(2, int(n**0.5)+1):
        k = 1
        while p**k <= n:
            if p**k == n:
                return np.log(p)
            k += 1
    if n > 1:
        return np.log(n)
    return 0.0

# ---- basis ----

def odd_basis(m, grid):
    basis = []
    for k in range(1, m+1):
        basis.append(np.sin(k * grid))
    return np.array(basis)

# ---- prime term contribution ----

def prime_term_matrix(basis, grid):
    m = basis.shape[0]
    Q = np.zeros((m, m))
    primes = primes_up_to(PRIME_CUTOFF)
    logs = np.log(primes)
    for p in primes:
        weight = np.log(p) / np.sqrt(p)
        t = np.log(p)
        idx = np.argmin(np.abs(grid - t))
        for i in range(m):
            for j in range(m):
                Q[i,j] += weight * basis[i,idx] * basis[j,idx]
    return Q

# ---- zero term (toy symmetric) ----

def zero_term_matrix(basis):
    m = basis.shape[0]
    Q = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            Q[i,j] = np.dot(basis[i], basis[j])
    return Q

# ---- test ----

def test_coercivity(m=10, N=4000):
    grid = np.linspace(-LOG_CUTOFF, LOG_CUTOFF, N)
    basis = odd_basis(m, grid)

    Q_zero = zero_term_matrix(basis)
    Q_prime = prime_term_matrix(basis, grid)

    Q = Q_zero - Q_prime  # explicit-form structure: zeros - primes

    eigvals = np.linalg.eigvalsh(Q)

    print("Eigenvalues:", eigvals)
    print("Min eigenvalue:", eigvals[0])

if __name__ == "__main__":
    test_coercivity()
