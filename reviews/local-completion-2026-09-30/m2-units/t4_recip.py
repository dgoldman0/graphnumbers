from gl import *
E = REG.id(G(1, []))
def rdeg(t): return REG.reps[t].deg(0)

def recip_recursion(a, r, maxdeg):
    """formula (15): b(e)=1/c, b(B) = -1/c sum_{C*D=B, C!=e} a(C) b(D), by root-degree levels."""
    c = a[E]; b = {E: 1 / Q(c)}
    acc = defaultdict(Q)
    levels = defaultdict(set); levels[0].add(E)
    supp = [(C, v) for C, v in a.items() if C != E]
    for d in range(0, maxdeg + 1):
        for B in sorted(levels[d]):
            if B != E:
                b[B] = -acc[B] / c
            for C, v in supp:
                P = star(C, B, r)
                if rdeg(P) <= maxdeg:
                    acc[P] += v * b[B]; levels[rdeg(P)].add(P)
    return {t: v for t, v in b.items() if v != 0}

def recip_geometric(a, r, maxdeg):
    c = Q(a[E]); z = {t: -Q(v) / c for t, v in a.items() if t != E}
    out = defaultdict(Q); pw = {E: Q(1)}
    for j in range(maxdeg + 1):
        for t, v in pw.items(): out[t] += v / c
        pw = {t: v for t, v in conv(pw, z, r).items() if rdeg(t) <= maxdeg}
    return {t: v for t, v in out.items() if v != 0}

K1, K2 = G(1, []), complete(2)
for r in (1, 2):
    a = lin_hist([(1, K1), (1, K2)], r)
    b1 = recip_recursion(a, r, 7); b2 = recip_geometric(a, r, 7)
    print(f"K1+K2 r={r}: recursion==geometric {b1 == b2};",
          sorted((rdeg(t), REG.reps[t].n, str(v)) for t, v in b1.items()))
    prod = {t: v for t, v in conv(a, b1, r).items() if rdeg(t) <= 7}
    print("   a*b = delta_e up to degree 7:", prod == {E: 1})
# line L: T_r(L)=delta(centered path P_{2r+1}); 1 - tL reciprocal coefficients t^n on Z^n balls
for r in (1, 2):
    ell = REG.id(path(2 * r + 1, root=r))
    t = Q(3, 4)
    a = {E: Q(1), ell: -t}
    b = recip_recursion(a, r, 8)
    print(f"1-(3/4)L r={r}:", sorted((rdeg(x), REG.reps[x].n, str(v)) for x, v in b.items()))
# X_3 = 1 - 1/2(C6/6 - C3/3) at radius 1: reciprocal l1 mass by degree
a = lin_hist([(1, K1), (Q(-1, 12), cycle(6)), (Q(1, 6), cycle(3))], 1)
print("T_1(X_3) =", {(rdeg(t), REG.reps[t].n, len(REG.reps[t].edges())): str(v) for t, v in a.items()})
b = recip_recursion(a, 1, 12)
mass = defaultdict(Q)
for t, v in b.items(): mass[rdeg(t)] += abs(v)
print("X_3 reciprocal l1 mass per root degree:", {d: str(m) for d, m in sorted(mass.items())})
