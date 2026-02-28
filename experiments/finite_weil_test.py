import numpy as np

# Toy odd basis on symmetric grid
def odd_basis(m, grid):
    basis = []
    for k in range(1, m+1):
        basis.append(np.sin(k * grid))
    return np.array(basis)

# Toy quadratic form model
# Replace with real explicit-form evaluation later
def Q_matrix(basis):
    m = basis.shape[0]
    Q = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            Q[i,j] = np.dot(basis[i], basis[j])
    return Q

def test_coercivity(m=6, N=2000):
    grid = np.linspace(-10, 10, N)
    basis = odd_basis(m, grid)
    Q = Q_matrix(basis)
    eigvals = np.linalg.eigvalsh(Q)
    print("Eigenvalues:", eigvals)
    print("Min eigenvalue:", eigvals[0])

if __name__ == "__main__":
    test_coercivity()
