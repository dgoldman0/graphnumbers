"""Independently check recorded certificates using only networkx-based histograms."""
import json, networkx as nx, indep
from fractions import Fraction as Fr
def from_rows(rows):
    G = nx.Graph(); G.add_nodes_from(range(len(rows)))
    for u, row in enumerate(rows):
        for v in range(len(rows)):
            if row >> v & 1:
                assert rows[v] >> u & 1
                G.add_edge(u, v)
    return G
def check(result, maxn, maxdeg):
    R = result['radius']
    reg = indep.Registry()
    tid = lambda rows: reg.register(indep.rooted(from_rows(rows), 0))
    target = {tid(t['rows']): Fr(t['value']) for t in result['target']}
    dual = {tid(t['rows']): Fr(t['value']) for t in result['dual']}
    cat = [(from_rows(c['rows']), Fr(c['coefficient'])) for c in result['catalog']]
    # catalog completeness vs atlas
    atlas = indep.connected_graphs(maxn, maxdeg)
    assert len(atlas) == len(cat), (len(atlas), len(cat))
    for A in atlas:
        assert sum(nx.is_isomorphic(A, G) for G, _ in cat) == 1
    # primal feasibility
    h = indep.lin_hist([(c, G) for G, c in cat if c], R, reg)
    primal_ok = h == {t: v for t, v in target.items() if v}
    cost = sum(abs(c) * G.number_of_nodes() for G, c in cat)
    dual_val = sum(dual.get(t, 0) * v for t, v in target.items())
    feas = all(abs(sum(dual.get(t, 0) * m for t, m in indep.hist(G, R, reg).items())) <= G.number_of_nodes() for G, _ in cat)
    neg = sum(-c * G.number_of_nodes() for G, c in cat if c < 0)
    return primal_ok, feas, cost, dual_val, Fr(result['cost']), neg, Fr(result['negative_mass'])
res = json.load(open('/home/user/graphnumbers/research/local-completion/reconstruction_example_result.json'))
print("example:", check(res, 5, 2))
spec = json.load(open('/home/user/graphnumbers/research/local-completion/spectral_approximation_results.json'))
for ex in spec['reconstruction_examples']:
    print(ex['max_vertices'], ex['max_degree'], ex['radius'], check(ex, ex['max_vertices'], ex['max_degree']))
