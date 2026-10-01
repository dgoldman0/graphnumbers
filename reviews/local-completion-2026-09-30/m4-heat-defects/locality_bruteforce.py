"""Independent brute force for the heat-locality lemma.

For rooted graph (G,o) with max degree <= D, B = induced R-ball:
claim (P_G^j)_oo == (P_B^j)_oo for j <= 2R, with P = I - Lap/D (B uses its own degrees).
Also check Laplacian powers (Lap^m)_oo. Find the minimal j where they can differ.
"""
from fractions import Fraction as F
import itertools, networkx as nx
from networkx.generators.atlas import graph_atlas_g

def ball_nodes(G, o, R):
    return set(nx.single_source_shortest_path_length(G, o, cutoff=R).keys())

def lazy_diag(G, o, D, jmax):
    # exact (P^j)_oo, P = I - L/D, using integer vector of D^j P^j e_o
    nodes = list(G.nodes()); idx = {v:i for i,v in enumerate(nodes)}
    deg = {v: G.degree(v) for v in nodes}
    vec = {v: 0 for v in nodes}; vec[o] = 1
    out = [F(1)]
    for j in range(1, jmax+1):
        new = {}
        for v in nodes:
            s = (D - deg[v]) * vec[v] + sum(vec[w] for w in G.neighbors(v))
            new[v] = s
        vec = new
        out.append(F(vec[o], D**j))
    return out

def lap_diag(G, o, mmax):
    nodes = list(G.nodes())
    deg = {v: G.degree(v) for v in nodes}
    vec = {v: 0 for v in nodes}; vec[o] = 1
    out = [1]
    for m in range(1, mmax+1):
        vec = {v: deg[v]*vec[v] - sum(vec[w] for w in G.neighbors(v)) for v in nodes}
        out.append(vec[o])
    return out

graphs = [g for g in graph_atlas_g()[1:] if nx.is_connected(g)]
print("connected graphs in atlas (<=7 vertices):", len(graphs))
viol = []   # violations of j<=2R
first_diff_at = {}  # R -> min j where some example differs (lazy)
first_diff_lap = {}
count=0
for G in graphs:
    D0 = max(dict(G.degree()).values()) if G.number_of_nodes()>1 else 0
    for o in G.nodes():
        for R in range(0, 4):
            Bn = ball_nodes(G, o, R)
            B = G.subgraph(Bn).copy()
            if len(Bn) == G.number_of_nodes():
                continue  # ball is whole graph: trivial
            jmax = 2*R + 3
            for D in sorted(set([max(D0,1), D0+1, D0+3])):
                if D == 0: continue
                a = lazy_diag(G, o, D, jmax); b = lazy_diag(B, o, D, jmax)
                count += 1
                for j in range(jmax+1):
                    if a[j] != b[j]:
                        if j <= 2*R:
                            viol.append((G.edges(), o, R, D, j))
                        first_diff_at[R] = min(first_diff_at.get(R, 99), j)
                        break
            la = lap_diag(G, o, 2*R+3); lb = lap_diag(B, o, 2*R+3)
            for m in range(2*R+4):
                if la[m] != lb[m]:
                    if m <= 2*R: viol.append(('lap', list(G.edges()), o, R, m))
                    first_diff_lap[R] = min(first_diff_lap.get(R, 99), m)
                    break
print("comparisons:", count)
print("violations of j<=2R:", len(viol), viol[:5])
print("minimal differing order by R (lazy P):", first_diff_at)
print("minimal differing order by R (Laplacian):", first_diff_lap)

# Explicit sharpness witnesses for each R: cycle C_n vs its R-ball (path)
for R in range(0, 6):
    n = 2*R + 3
    C = nx.cycle_graph(n); B = C.subgraph(ball_nodes(C, 0, R)).copy()
    a = lazy_diag(C, 0, 2, 2*R+1); b = lazy_diag(B, 0, 2, 2*R+1)
    print(f"R={R}: C_{n} vs induced ball: agree through 2R? {a[:2*R+1]==b[:2*R+1]}; order 2R+1: {a[2*R+1]} vs {b[2*R+1]}")

# Two genuinely different rooted graphs with identical R-balls whose order-(2R+1) coefficient differs
R=2
G1 = nx.path_graph(7)  # root 3: ball = P5 centred, vertex at dist 2 have outer neighbours
G2 = nx.path_graph(5)  # root 2: ball = whole P5 (ends have degree 1)
a = lazy_diag(G1, 3, 2, 6); b = lazy_diag(G2, 2, 2, 6)
print("P7 root 3 vs P5 root 2 (same induced 2-ball):", [str(x) for x in a], [str(x) for x in b])
