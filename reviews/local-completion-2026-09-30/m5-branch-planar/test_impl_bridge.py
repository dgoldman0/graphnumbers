import random, itertools
import networkx as nx
from collections import defaultdict
from gcore import Registry, clean
from test_bridge_blocks_lib import block_tree_graph
import graphlocal as gl
from graphlocal.edge_interactions import EdgeInteraction, bridge_cut_reduction

reg = Registry()
def to_gl(G):
    nodes = sorted(G.nodes()); idx = {v: i for i, v in enumerate(nodes)}
    return gl.graph(len(nodes), [(idx[u], idx[v]) for u, v in G.edges()]), idx

def gl_to_nx(g):
    H = nx.Graph(); H.add_nodes_from(range(g.n))
    for u in range(g.n):
        for v in g.neighbors(u):
            if u < v: H.add_edge(u, v)
    return H

def brute(G, F):
    acc = defaultdict(int); k = len(F)
    for m in range(k + 1):
        for S in itertools.combinations(F, m):
            H = G.copy(); H.remove_edges_from(S)
            for c in nx.connected_components(H):
                acc[reg.key(H.subgraph(c))] += (-1) ** (k - m)
    return clean(acc)

rng = random.Random(99)
bad = 0; n = 0; reduced = 0
for trial in range(300):
    G, _ = block_tree_graph(rng, rng.randint(2, 6))
    br = list(nx.bridges(G))
    g, idx = to_gl(G)
    if rng.random() < 0.7 and br:
        F = rng.sample(br, rng.randint(1, min(5, len(br))))
    else:
        E = list(G.edges()); F = rng.sample(E, rng.randint(1, min(5, len(E))))
    edits = [(idx[u], idx[v], -1) for u, v in F]
    val = EdgeInteraction(g, edits, max_edits=12)
    fin = val.finite()
    got = defaultdict(int)
    for c, h in fin.terms:
        got[reg.key(gl_to_nx(h))] += c
    got = clean(got)
    want = brute(nx.convert_node_labels_to_integers(G, ordering='sorted'), [(idx[u], idx[v]) for u, v in F])
    n += 1; reduced += val.bridge_reduction_applies and len(val.active_edits) < len(F)
    if got != want:
        bad += 1; print("MISMATCH", trial)
print("implementation EdgeInteraction.finite() vs brute force:", n, "cases,", bad, "mismatches,", reduced, "proper reductions applied")
