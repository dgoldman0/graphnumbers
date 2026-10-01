import sys, random
sys.path.insert(0, '/home/user/graphnumbers/python/src')
import numpy as np, mpmath as mp, networkx as nx
from fractions import Fraction as Q
from graphlocal import Finite, Line, cycle, path, graph
from graphlocal.defects import CutLineDefect, SparseEdgeDifference, relative_heat
from graphlocal.interactions import TwoCutLineDefect, CutInteraction, LineCutDefect, connected_cut_interaction
mp.mp.dps = 40
def M(x): return mp.mpf(x.numerator)/x.denominator if isinstance(x, Q) else mp.mpf(x)
def h(t): t=M(t); return mp.e**(-2*t)*mp.besseli(0,2*t)
def e(t): t=M(t); return (1-mp.e**(-4*t))/2
def p(ell,t): t=M(t); return mp.fsum(mp.e**(-t*(2-2*mp.cos(mp.pi*a/ell))) for a in range(ell))
def J(ell,t): return p(ell,t)-ell*h(t)-e(t)
def lapG(g):
    n=g.n; L=np.zeros((n,n))
    for u in range(n):
        L[u,u]=g.rows[u].bit_count()
        for v in g.neighbors(u): L[u,v]=-1
    return L
def trh(g,t):
    ev = mp.eigsy(mp.matrix(lapG(g).tolist()), eigvals_only=True)
    return mp.fsum(mp.e**(-M(t)*x) for x in ev)
def check(name, X, t, truth, eps="1e-8"):
    c = relative_heat(X, t, eps)
    lo, hi = M(c.interval.lower), M(c.interval.upper)
    ok = lo <= truth <= hi and c.interval.radius <= Q(eps)
    if not ok: print("FAIL", name, t, mp.nstr(lo,15), mp.nstr(truth,15), mp.nstr(hi,15))
    return ok
times = [Q(0), Q(1,20), Q(1,2), Q(1), Q(2), Q(4)]
res = []
for t in times:
    res.append(check('E', CutLineDefect(), t, e(t)))
    for ell in (1,2,3,5):
        res.append(check(f'I{ell}', CutInteraction(ell), t, J(ell,t)))
        res.append(check(f'D{ell}', TwoCutLineDefect(ell), t, e(t)+p(ell,t)-ell*h(t)))
    for pos in [(0,1,2), (0,2,7), (-3,0,4,5)]:
        k=len(pos); ps=sorted(pos); gaps=[b-a for a,b in zip(ps,ps[1:])]
        res.append(check(f'cuts{pos}', LineCutDefect(pos), t, k*e(t)+sum(J(g,t) for g in gaps)))
        res.append(check(f'conn{pos}', connected_cut_interaction(pos), t, (-1)**k*J(ps[-1]-ps[0],t)))
    if t <= 2:
        res.append(check('E*L', CutLineDefect()*Line(), t, e(t)*h(t)))
        for H in (path(2), cycle(3)):
            res.append(check(f'E*U{H.rows}', CutLineDefect()*Finite.from_graph(H, True), t, e(t)*trh(H,t)/H.n))
    if t <= 1:
        res.append(check('E*L^2', CutLineDefect()*Line()*Line(), t, e(t)*h(t)**2))
print("structured relative-heat checks:", len(res), "all ok:", all(res))
# random sparse edits on random graphs, with mixed signs, unnormalized and normalized
rng = random.Random(11); r2=[]
for trial in range(30):
    n = rng.randint(5, 14)
    edges = set((i,i+1) for i in range(n-1)) | set(tuple(sorted(rng.sample(range(n),2))) for _ in range(rng.randint(0,n)))
    g = graph(n, edges)
    present = sorted(edges); absent = [(u,v) for u in range(n) for v in range(u+1,n) if (u,v) not in edges]
    edits = [(u,v,-1) for (u,v) in rng.sample(present, min(len(present), rng.randint(0,2)))]
    edits += [(u,v,1) for (u,v) in rng.sample(absent, min(len(absent), rng.randint(0,2)))]
    if not edits: continue
    norm = rng.random() < 0.5
    X = SparseEdgeDifference(g, edits, normalize=norm)
    for t in (Q(1,10), Q(1,2), Q(3,2)):
        truth = (trh(X.after,t) - trh(X.before,t)) * (mp.mpf(1)/n if norm else 1)
        r2.append(check(f'sparse{trial}', X, t, truth))
        # also verify cap |H| <= q*min(1,2t) and sign rules independently
        q = len(edits) * (mp.mpf(1)/n if norm else 1)
        cap = q*min(1, 2*M(t))
        if abs(truth) > cap + mp.mpf('1e-30'): print("CAP VIOLATION", trial, t, truth, cap)
        if all(s==-1 for *_ ,s in edits) and truth < 0: print("SIGN VIOLATION deletion", trial)
        if all(s==1 for *_ ,s in edits) and truth > 0: print("SIGN VIOLATION addition", trial)
print("random sparse checks:", len(r2), "all ok:", all(r2))
