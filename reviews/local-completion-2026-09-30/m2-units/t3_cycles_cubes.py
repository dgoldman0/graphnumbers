import math, cmath
from gl import *
# odd-cycle characters
for N in range(3, 26, 2):
    r = (N-1)//2
    qC = cumulants_from_moments(walks(cycle(N), N, True))[N]
    q2C = cumulants_from_moments(walks(cycle(2*N), N, True))[N]
    # X_N = 1 - 1/2 (C_{2N}/(2N) - C_N/N)
    XN = lin_hist([(1, G(1, [])), (Q(-1, 4*N), cycle(2*N)), (Q(1, 2*N), cycle(N))], r)
    chi = 0
    for t, v in XN.items():
        q = cumulants_from_moments(walks(REG.reps[t], N, True))[N]
        chi += v * cmath.exp(1j*math.pi*q/2)
    # local equality with 1 below detecting radius
    eq = all(lin_hist([(1, G(1, [])), (Q(-1, 4*N), cycle(2*N)), (Q(1, 2*N), cycle(N))], R) == {REG.id(G(1,[])): 1} for R in range(0, r))
    # degree generating function
    print(f"N={N:2d} r={r:2d} q_N(C_N)={qC} q_N(C_2N)={q2C} chi_N(X_N)={abs(chi):.1e} T_R(X_N)=T_R(1) for R<r: {eq}; T_r(X_N) support size {len(XN)}")
# hypercube balls
cube = G(1, [])
K2 = complete(2)
for n in range(0, 9):
    for r in range(1, 5):
        sz = ball(cube, 0, r).n
        assert sz == sum(math.comb(n, j) for j in range(min(r, n)+1)), (n, r, sz)
    cube = cart(cube, K2)
print("hypercube (21) ok n<=8, r<=4")
viol = [(n, r) for n in range(0, 400) for r in range(1, 12) if sum(math.comb(n, j) for j in range(min(r, n)+1)) > (n+1)**r]
print("b_r(n) <= (n+1)^r violations:", viol[:5])
