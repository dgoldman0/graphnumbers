import random, mpmath
from fractions import Fraction as Q
from graphlocal import *
from graphlocal.interaction_bounds import interaction_geometry, interaction_heat_bound, GeometricInteraction
import oracle

random.seed(11)
def random_graph(n, p):
    return graph(n, [(u, v) for u in range(n) for v in range(u + 1, n) if random.random() < p])

fails = cases = 0
for trial in range(60):
    n = random.randint(2, 7)
    g = random_graph(n, random.choice([0.4, 0.6]))
    present = [(u, v) for u in range(n) for v in range(u + 1, n) if g.rows[u] >> v & 1]
    absent = [(u, v) for u in range(n) for v in range(u + 1, n) if not g.rows[u] >> v & 1]
    k = random.randint(1, 3)
    pool = [(u, v, -1) for u, v in present] + [(u, v, 1) for u, v in absent]
    if len(pool) < k: continue
    edits = random.sample(pool, k)
    normalize = random.random() < 0.3
    x = EdgeInteraction(g, edits, normalize=normalize)
    scale = Q(1, n) if normalize else Q(1)
    for t in (Q(1, 8), Q(1, 2), Q(2)):
        cases += 1
        true = scale * oracle.heat(g.rows, edits, t)
        hb = interaction_heat_bound(x, t, "1e-12")
        if abs(true) > oracle.mpq(hb.magnitude_bound) * (1 + mpmath.mpf(10) ** -40):
            fails += 1; print("MAG FAIL", g.rows, edits, t, true, float(hb.magnitude_bound))
        for M in (0, 1, 3, 6):
            tb = interaction_heat_bound(x, t, "1e-12", after_step=M)
            D = x.degree_bound
            lz = [scale * m for m in oracle.lazy_moments(g.rows, edits, D, M)]
            lam = t * D
            retained = mpmath.e ** (-oracle.mpq(lam)) * sum(oracle.mpq(lam ** j / __import__('math').factorial(j) * m) for j, m in enumerate(lz))
            if abs(true - retained) > oracle.mpq(tb.magnitude_bound) * (1 + mpmath.mpf(10) ** -40):
                fails += 1; print("TAIL FAIL", g.rows, edits, t, M, float(true - retained), float(tb.magnitude_bound))
        for X in (x, GeometricInteraction(x)):
            ch = controlled_heat(X, t, "1e-9")
            lo, hi = oracle.mpq(ch.interval.lower), oracle.mpq(ch.interval.upper)
            if not (lo <= true <= hi) or ch.interval.radius > Q("1e-9"):
                fails += 1; print("CONTROLLED FAIL", type(X).__name__, g.rows, edits, t, true, float(lo), float(hi))
print("cases", cases, "fails", fails)
