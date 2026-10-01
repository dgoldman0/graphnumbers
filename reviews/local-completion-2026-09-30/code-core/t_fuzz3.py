from fractions import Fraction as Q
import random, itertools, time, sys
import networkx as nx
import mpmath as mp
from common import *
from oracles import *
from graphlocal import *
random.seed(21)
bad = 0; done = 0; rejected = 0
SL = mp.mpf(10) ** -40

def rand_finite():
    """random finite element and oracle function t -> H_t"""
    n = random.randint(1, 5)
    G = nx.gnp_random_graph(n, random.uniform(0.3, 0.9), seed=random.randint(0, 10**9))
    norm = random.random() < 0.6
    c = Q(random.choice([1, 1, 2, -1, -3]), random.choice([1, 2, 3]))
    X = Finite.from_graph(from_nx(G), normalize=norm) * c
    return X, (lambda t, G=G, n=n, norm=norm, c=c: mpq(c) * heat_trace(G, t) / (n if norm else 1))

def rand_bg():
    k = random.randint(0, 1)
    X, f = Finite.scalar(1), (lambda t: mp.mpf(1))
    for _ in range(k):
        if random.random() < 0.5:
            X, f = X * Line(), (lambda t, f=f: f(t) * hline(t))
        else:
            Y, g = rand_finite()
            X, f = X * Y, (lambda t, f=f, g=g: f(t) * g(t))
    return X, f

def rand_defect():
    kind = random.randint(0, 4)
    if kind == 0:
        return CutLineDefect(), (lambda t: cuts_value((0,), t))
    if kind == 1:
        pos = tuple(sorted(random.sample(range(0, 9), random.randint(1, 3))))
        return LineCutDefect(pos), (lambda t, pos=pos: cuts_value(pos, t))
    if kind == 2:
        ell = random.randint(1, 5)
        return TwoCutLineDefect(ell), (lambda t, ell=ell: cuts_value((0, ell), t))
    if kind == 3:
        ell = random.randint(1, 5)
        return CutInteraction(ell), (lambda t, ell=ell: cuts_value((0, ell), t) - 2 * cuts_value((0,), t))
    n = random.randint(3, 8)
    G = nx.gnp_random_graph(n, 0.5, seed=random.randint(0, 10**9))
    pairs = list(itertools.combinations(range(n), 2))
    ch = random.sample(pairs, random.randint(1, 3))
    edits = [(u, v, -1 if G.has_edge(u, v) else 1) for u, v in ch]
    H = G.copy()
    for u, v, s in edits: (H.remove_edge if s < 0 else H.add_edge)(u, v)
    norm = random.random() < 0.3
    return SparseEdgeDifference(from_nx(G), edits, norm), (lambda t, G=G, H=H, n=n, norm=norm: (heat_trace(H, t) - heat_trace(G, t)) / (n if norm else 1))

for trial in range(160):
    terms = []
    for _ in range(random.randint(1, 3)):
        D, f = rand_defect()
        B, g = rand_bg()
        c = Q(random.choice([1, -1, 2, -2, 3]), random.choice([1, 2, 5]))
        left = random.random() < 0.5
        X = (D * B if left else B * D) * c
        terms.append((X, (lambda t, f=f, g=g, c=c: mpq(c) * f(t) * g(t))))
    X = terms[0][0]
    for Y, _ in terms[1:]:
        X = X + Y
    t = random.choice([Q(1, 4), Q(1, 2), Q(1)])
    eps = random.choice(["1e-6", "1e-10"])
    print('start rel', trial, 't=', t, 'deg', X.degree_bound, flush=True)
    try:
        t0 = time.time(); cert = relative_heat(X, t, eps, max_steps=600)
        print('rel', trial, 't=', t, 'deg', X.degree_bound, 'steps', cert.steps, f'{time.time()-t0:.1f}s', flush=True)
    except (ValueError, BudgetExceeded) as e:
        print('rej', trial, type(e).__name__, str(e)[:60], flush=True)
        rejected += 1
        continue
    truth = mp.fsum(f(mpq(t)) for _, f in terms)
    lo, hi = mpq(cert.interval.lower), mpq(cert.interval.upper)
    done += 1
    if not (lo - SL <= truth <= hi + SL):
        bad += 1; print("FAIL relative", trial, float(lo), float(truth), float(hi))
print("relative fuzz: checked", done, "rejected", rejected, "bad", bad)

done = rejected = 0
for trial in range(160):
    X, f = rand_finite()
    Xs, fs = [X], [f]
    for _ in range(random.randint(0, 3)):
        B, g = rand_bg()
        c = Q(random.choice([1, -1, 2, -3]), random.choice([1, 2, 7]))
        Xs.append(B * c); fs.append(lambda t, g=g, c=c: mpq(c) * g(t))
    X = Xs[0]
    for Y in Xs[1:]: X = X + Y
    t = random.choice([Q(0), Q(1, 3), Q(1), Q(3, 2)])
    eps = random.choice(["1e-6", "1e-10"])
    try:
        t0 = time.time(); cert = heat_return(X, t, eps, max_steps=600)
        print('heat', trial, 't=', t, 'deg', X.degree_bound, 'steps', cert.steps, f'{time.time()-t0:.1f}s', flush=True)
    except (ValueError, BudgetExceeded) as e:
        print('rej', trial, type(e).__name__, str(e)[:60], flush=True)
        rejected += 1; continue
    truth = mp.fsum(g(mpq(t)) for g in fs)
    lo, hi = mpq(cert.interval.lower), mpq(cert.interval.upper)
    done += 1
    if not (lo - SL <= truth <= hi + SL):
        bad += 1; print("FAIL heat", trial, float(lo), float(truth), float(hi))
print("heat fuzz: checked", done, "rejected", rejected, "bad", bad)
