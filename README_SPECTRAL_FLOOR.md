Spectral Floor Normalization (GSCM)

Define H on the Hermite-odd basis truncation.

Numerically:
lambda0 = inf spectrum(H) at truncation
gap = lambda1 - lambda0

The coercive operator is H - lambda0 I, positive on the orthogonal complement of the ground state.

Reproduce:
python3 scripts/compute_spectral_floor.py 90 15.0 4000
pytest -q
