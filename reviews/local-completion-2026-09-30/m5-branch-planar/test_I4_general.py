import numpy as np
rng = np.random.default_rng(1)
maxerr = 0
for trial in range(200):
    n = rng.integers(3, 8)
    A = rng.normal(size=(n, n)); L = A + A.T
    b = rng.normal(size=n); b *= np.sqrt(2) / np.linalg.norm(b)
    c = rng.normal(size=n); c *= np.sqrt(2) / np.linalg.norm(c)
    B, C = np.outer(b, b), np.outer(c, c)
    def I(j):
        mp = np.linalg.matrix_power
        return np.trace(mp(L-B-C, j) - mp(L-B, j) - mp(L-C, j) + mp(L, j))
    s = b @ c; u = b @ L @ c; v = b @ L @ L @ c; al = b @ L @ b; be = c @ L @ c
    f2 = 2*s*s; f3 = 6*s*u - 12*s*s
    f4 = 8*s*v + 4*u*u - 32*s*u - 4*s*s*(al+be) + 48*s*s + 2*s**4
    err = max(abs(I(2)-f2), abs(I(3)-f3), abs(I(4)-f4)) / (1 + abs(I(4)))
    maxerr = max(maxerr, err)
print("max relative error of (1),(2) on random symmetric L with |b|^2=|c|^2=2:", maxerr)
