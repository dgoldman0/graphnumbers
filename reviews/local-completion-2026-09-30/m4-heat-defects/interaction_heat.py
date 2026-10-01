import mpmath as mp, numpy as np, networkx as nx, itertools
from fractions import Fraction as F
from math import comb
mp.mp.dps = 40
def h(t): t=mp.mpf(t); return mp.e**(-2*t)*mp.besseli(0,2*t)
def e(t): t=mp.mpf(t); return (1-mp.e**(-4*t))/2
def p(ell,t): t=mp.mpf(t); return mp.fsum(mp.e**(-t*(2-2*mp.cos(mp.pi*a/ell))) for a in range(ell))
def J3(ell,t): return p(ell,t)-ell*h(t)-e(t)
def b(a,t):
    t=mp.mpf(t); return mp.nsum(lambda j: t**(a+2*j)/(mp.factorial(j)*mp.factorial(a+j)), [0, mp.inf])
def J5(ell,t):
    t=mp.mpf(t); return 2*ell*mp.e**(-2*t)*mp.nsum(lambda k: b(2*ell*int(k),t), [1, mp.inf])
def lap(G):
    return nx.laplacian_matrix(G, nodelist=sorted(G.nodes())).toarray().astype(float)
def trheat(G,t):
    ev = np.linalg.eigvalsh(lap(G)); return float(np.exp(-t*ev).sum())
def finite_I(ell, n, t):
    G2 = nx.cycle_graph(n); G2.remove_edge(n-1,0); G2.remove_edge(ell-1,ell)
    G1 = nx.cycle_graph(n); G1.remove_edge(n-1,0)
    return trheat(G2,t)+trheat(nx.cycle_graph(n),t)-2*trheat(G1,t)
print("ell  t    J(3)              J(5)              finite n=240       J<=J1<1/2  J<=Pr[Pois(2t)>=2ell]")
for ell in (1,2,3,5,8):
    for t in ('0.1','0.5','1','2','4'):
        a3, a5 = J3(ell,t), J5(ell,t)
        fin = finite_I(ell, 240, float(t))
        pois = 1 - mp.fsum(mp.e**(-2*mp.mpf(t))*(2*mp.mpf(t))**j/mp.factorial(j) for j in range(2*ell))
        print(ell, t, mp.nstr(a3,14), mp.nstr(a5,14), '%.12e'%fin, bool(0 < a5 <= J5(1,t) < 0.5), bool(a5 <= pois))
# moments (6) from exact walk counting on finite cycles, D=2: d_j = tr P^j differences
def trP_powers(G, jmax, D=2):
    nodes = sorted(G.nodes()); n=len(nodes)
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=object).astype(int)
    deg = A.sum(axis=1)
    M = [[(D-deg[i]) if i==j else A[i][j] for j in range(n)] for i in range(n)]  # D*P integer
    M = np.array(M, dtype=object)
    P = np.identity(n, dtype=object); out=[]
    for j in range(jmax+1):
        out.append(F(int(sum(P[i][i] for i in range(n))), D**j))
        P = P.dot(M)
    return out
ok=True
for ell in (1,2,3,4):
    n = 4*ell + 14; jmax = 12
    G2 = nx.cycle_graph(n); G2.remove_edge(n-1,0); G2.remove_edge(ell-1,ell)
    G1 = nx.cycle_graph(n); G1.remove_edge(n-1,0)
    a, c, d = trP_powers(G2,jmax), trP_powers(nx.cycle_graph(n),jmax), trP_powers(G1,jmax)
    for j in range(jmax+1):
        dj = a[j]+c[j]-2*d[j]
        formula = 0 if j%2 else F(2*ell, 2**j)*sum(comb(j, j//2+ell*k) for k in range(1, j+1) if j//2+ell*k <= j)
        if dj != formula: ok=False; print("moment mismatch", ell, j, dj, formula)
print("moments (6) match exact finite walk counts (n large):", ok)
# (7) series coefficients
import sympy as sp
T = sp.symbols('t')
for ell in (1,2,3):
    ser = 2*ell*sp.exp(-2*T)*sum(sum(T**(2*ell*k+2*j)/(sp.factorial(j)*sp.factorial(2*ell*k+j)) for j in range(4)) for k in range(1,3))
    s = sp.series(ser, T, 0, 2*ell+2).removeO()
    print(f"ell={ell}: coeff t^{2*ell}={s.coeff(T,2*ell)} vs 1/(2ell-1)!={sp.Rational(1,sp.factorial(2*ell-1))}; coeff t^{2*ell+1}={s.coeff(T,2*ell+1)} vs -2/(2ell-1)!={-2*sp.Rational(1,sp.factorial(2*ell-1))}")
# (9) and asymptotics
for ell in (1,2,3):
    for t in (50, 400):
        val = J3(ell,t); approx = mp.mpf(1)/2 - ell/mp.sqrt(4*mp.pi*t)
        print(f"large t: ell={ell} t={t} J={mp.nstr(val,12)} 1/2-ell/sqrt(4pi t)={mp.nstr(approx,12)} diff*t^1.5={mp.nstr((val-approx)*mp.mpf(t)**1.5,6)}")
# resolvent (11) vs (12) vs finite cycles
def R11(ell,s):
    s=mp.mpf(s); return mp.fsum(1/(s+2-2*mp.cos(mp.pi*a/ell)) for a in range(ell)) - ell/mp.sqrt(s*(s+4)) - 2/(s*(s+4))
def R12(ell,s):
    s=mp.mpf(s); r=(s+2-mp.sqrt(s*(s+4)))/2; return 2*ell/mp.sqrt(s*(s+4))*r**(2*ell)/(1-r**(2*ell))
def Rfin(ell,s,n=400):
    G2 = nx.cycle_graph(n); G2.remove_edge(n-1,0); G2.remove_edge(ell-1,ell)
    G1 = nx.cycle_graph(n); G1.remove_edge(n-1,0)
    f = lambda G: float(np.sum(1/(s+np.linalg.eigvalsh(lap(G)))))
    return f(G2)+f(nx.cycle_graph(n))-2*f(G1)
for ell in (1,2,3,6):
    for s in ('0.05','0.5','2'):
        print(f"resolvent ell={ell} s={s}: (11)={mp.nstr(R11(ell,s),14)} (12)={mp.nstr(R12(ell,s),14)} finite n=400={Rfin(ell,float(s)):.12e}")
# Laplace transform consistency: integral e^{-st}J(t) dt = R_s
for ell in (1,2):
    s = mp.mpf('0.7')
    lt = mp.quad(lambda t: mp.e**(-s*t)*J3(ell,t), [0, 5, 20, mp.inf])
    print("Laplace check ell",ell, mp.nstr(lt,14), mp.nstr(R12(ell,s),14))
