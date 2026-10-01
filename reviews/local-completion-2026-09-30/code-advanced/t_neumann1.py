from fractions import Fraction as Q
from math import comb
import itertools
from graphlocal import *
from graphlocal.graphs import IsoGraph, ball

# Oracle: hypercube balls. T_r(H^n) = [B_r(Q_n)] (normalized, mass 1).
def hypercube(n):
    N = 1 << n
    return graph(N, [(u, u ^ (1 << i)) for u in range(N) for i in range(n) if u < u ^ (1 << i)])

def zn_ball_size(n, r):
    return sum(2**j * comb(n, j) * comb(r, j) for j in range(min(n, r) + 1))

def qn_ball_size(n, r):
    return sum(comb(n, j) for j in range(min(n, r) + 1))

H = Finite.from_graph(path(2), normalize=True)
for c in (Q(4), Q(2), Q(3, 2)):
    for r in (0, 1, 2, 3):
        for k in (1, 2, 3):
            inv = NeumannInverse(H / c)
            try:
                cert = inv.approximation_certificate(r, k, "1e-6")
            except Exception as e:
                print("c", c, "r", r, "k", k, "EXC", type(e).__name__, e)
                continue
            N = cert.truncation_degree
            h = cert.approximation.histogram
            q = 1 / c
            # check coefficients by type: for r>=1, type determined by root degree n
            got = {}
            for key, coef in h.values.items():
                got[key.graph.rows[0].bit_count()] = got.get(key.graph.rows[0].bit_count(), 0) + coef
            if r >= 1:
                exp_coefs = {n: q**n for n in range(N + 1)}
                assert got == exp_coefs, (c, r, k, got, exp_coefs)
                # exact weighted tail
                tail = sum(q**n * qn_ball_size(n, r)**k for n in range(N + 1, N + 3000))
            else:
                assert h.mass == 1 / (1 - q)
                tail = 0
            ok = tail <= cert.approximation.error <= Q("1e-6")
            print(f"c={c} r={r} k={k} N={N} err={float(cert.approximation.error):.3e} exact_tail~{float(tail):.3e} ok={ok}")
