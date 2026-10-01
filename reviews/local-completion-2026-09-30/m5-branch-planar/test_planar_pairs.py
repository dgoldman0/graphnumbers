import numpy as np
from math import factorial
from fractions import Fraction

def grid_laplacian(side, torus=False):
    n = side * side
    L = np.zeros((n, n), dtype=np.int64)
    idx = lambda x, y: (y % side) * side + (x % side)
    for y in range(side):
        for x in range(side):
            for dx, dy in ((1, 0), (0, 1)):
                xx, yy = x + dx, y + dy
                if not torus and (xx >= side or yy >= side):
                    continue
                u, v = idx(x, y), idx(xx, yy)
                L[u, v] -= 1; L[v, u] -= 1; L[u, u] += 1; L[v, v] += 1
    return L, idx

def edge_lap(n, u, v):
    B = np.zeros((n, n), dtype=np.int64)
    B[u, u] = B[v, v] = 1; B[u, v] = B[v, u] = -1
    return B

def mixed_moments(L, B, C, order):
    mats = {'LBC': L - B - C, 'LB': L - B, 'LC': L - C, 'L': L}
    signs = {'LBC': 1, 'LB': -1, 'LC': -1, 'L': 1}
    out = [0] * (order + 1)
    for key, M in mats.items():
        P = np.eye(L.shape[0], dtype=np.int64)
        for j in range(order + 1):
            out[j] += signs[key] * int(np.trace(P))
            P = P @ M
    return out

rows = [("perp adjacent", ((0, 0), (0, 1)), (2, 24, 226, 1980, 16826)),
        ("collinear adjacent", ((1, 0), (2, 0)), (2, 24, 218, 1840, 15212)),
        ("opposite square", ((0, 1), (1, 1)), (0, 0, 16, 320, 4176)),
        ("parallel dist 2", ((0, 2), (1, 2)), (0, 0, 0, 0, 24)),
        ("separated collinear", ((3, 0), (4, 0)), (0, 0, 0, 0, 6))]
for side, torus in ((23, False), (12, True), (10, True)):
    L, idx = grid_laplacian(side, torus)
    n = L.shape[0]; c0 = side // 2
    pt = lambda p: idx(c0 + p[0], c0 + p[1])
    print(f"--- side {side} torus={torus}")
    for name, (p, q), expected in rows:
        bvec = np.zeros(n, dtype=np.int64); bvec[pt((0, 0))] = 1; bvec[pt((1, 0))] = -1
        cvec = np.zeros(n, dtype=np.int64); cvec[pt(p)] = 1; cvec[pt(q)] = -1
        B = np.outer(bvec, bvec); C = np.outer(cvec, cvec)
        I = mixed_moments(L, B, C, 8)
        s = int(bvec @ cvec); u = int(bvec @ L @ cvec); v = int(bvec @ L @ L @ cvec)
        al = int(bvec @ L @ bvec); be = int(cvec @ L @ cvec)
        I4f = 8*s*v + 4*u*u - 32*s*u - 4*s*s*(al+be) + 48*s*s + 2*s**4
        I3f = 6*s*u - 12*s*s
        cross = [int(bvec @ np.linalg.matrix_power(L, a) @ cvec) for a in range(5)]
        m = next(a for a, x in enumerate(cross) if x)
        heat = [Fraction((-1)**j * I[j], factorial(j)) for j in range(9)]
        first = next(j for j in range(9) if I[j])
        print(f"{name:22s} I2..I8={I[2:]}, matches table={tuple(I[2:7])==expected}, I3 formula ok={I3f==I[3]}, I4 formula ok={I4f==I[4]}, "
              f"s={s},u={u},v={v},su={s*u},sv={s*v},alpha={al},beta={be}, (m,cross)={(m,cross[m])}, "
              f"first order={first}, (2m+2)cross^2={(2*m+2)*cross[m]**2}, leading heat={heat[first]}")
