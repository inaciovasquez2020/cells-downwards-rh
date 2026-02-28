import numpy as np
import math

N = 256
alpha = 0.4
sigma_spike = 0.15
A = 1.0
zero_positions = [5, 13, 29, 47]

def make_multiplier(N: int, sigma: float, A: float) -> np.ndarray:
    m = np.zeros(N, dtype=float)
    c = A / (sigma * math.sqrt(2.0 * math.pi))
    for j in range(N):
        k = j if j < N//2 else j - N
        smooth = 1.0 / (1.0 + k*k)
        spike = 0.0
        for z in zero_positions:
            spike += c * math.exp(-((k - z)**2)/(2.0*sigma*sigma))
            spike += c * math.exp(-((k + z)**2)/(2.0*sigma*sigma))
        m[j] = smooth + spike
    return m

def min_eigenvalue(alpha: float, theta: float, m: np.ndarray) -> float:
    lam = 1.0 - 2.0*alpha*m + theta*(m*m)
    return float(np.min(lam))

def analytic_min(alpha: float, theta: float) -> float:
    return 1.0 - (alpha*alpha)/theta

def main():
    m = make_multiplier(N, sigma_spike, A)

    print("alpha =", alpha)
    print("alpha^2 =", alpha*alpha)
    print("theta    numeric_min     analytic_min")

    for theta in [0.10, 0.14, 0.16, 0.18, 0.20, 0.25]:
        num = min_eigenvalue(alpha, theta, m)
        ana = analytic_min(alpha, theta)
        print(f"{theta:<8.3f} {num:<14.12e} {ana:.12e}")

if __name__ == "__main__":
    main()
