"""Independent implementation of the note's (7)-(12) coordinates."""
from fractions import Fraction as F
from itertools import combinations


def root_squares(adj):
    nb = list(adj[0])
    total = 0
    for a, b in combinations(nb, 2):
        total += len((adj[a] & adj[b]) - {0})
    return total


def sphere(adj, dist, r):
    s = [0] * (r + 1)
    for v in adj:
        s[dist[v]] += 1
    return s


def series_mul(a, b, n):
    out = [F(0)] * (n + 1)
    for i, x in enumerate(a[:n + 1]):
        if x:
            for j, y in enumerate(b[:n + 1 - i]):
                out[i + j] += x * y
    return out


def series_log(a, n):
    """log of a power series with a[0]=1, mod z^{n+1}."""
    a = [F(x) for x in a] + [F(0)] * (n + 1 - len(a))
    assert a[0] == 1
    # log(1+u) = sum (-1)^{k-1} u^k / k
    u = [F(0)] + a[1:n + 1]
    out = [F(0)] * (n + 1)
    p = [F(1)] + [F(0)] * n
    for k in range(1, n + 1):
        p = series_mul(p, u, n)
        for i in range(n + 1):
            out[i] += F((-1) ** (k - 1), k) * p[i]
    return out


def series_from_rational(num, den, n):
    """num/den power series mod z^{n+1}; den[0] = 1."""
    num = [F(x) for x in num] + [F(0)] * (n + 1)
    den = [F(x) for x in den] + [F(0)] * (n + 1)
    out = [F(0)] * (n + 1)
    for i in range(n + 1):
        s = num[i] - sum(den[j] * out[i - j] for j in range(1, i + 1))
        out[i] = s / den[0]
    return out


def tree_coords(adj, dist, r, d):
    """Return (N_d, c_0..c_{r-1}, c_R) per (8),(9),(11)."""
    assert r >= 2
    delta = len(adj[0])
    Qv = root_squares(adj)
    N = F((2 * d - 1) * delta - delta * delta + 2 * Qv, d * (d - 1))
    logS = series_log(sphere(adj, dist, r), r)
    Fd = series_from_rational([1, 1], [1, -(d - 1)], r)
    logF = series_log(Fd, r)
    U = [x - N * y for x, y in zip(logS, logF)]
    cs = []
    for j in range(r):
        # ell_j = log(1 - z^{j+1}/(1+z))
        frac = series_from_rational([0] * (j + 1) + [1], [1, 1], r)
        ell = series_log([1 - frac[0]] + [-x for x in frac[1:]], r)
        assert ell[j + 1] == -1 and all(x == 0 for x in ell[:j + 1])
        c = -U[j + 1]
        cs.append(c)
        U = [x - c * y for x, y in zip(U, ell)]
    assert all(x == 0 for x in U), U
    return (N,) + tuple(cs) + (N - sum(cs),)
