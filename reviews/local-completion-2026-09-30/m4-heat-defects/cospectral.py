import numpy as np, networkx as nx, itertools, random
from networkx.algorithms import isomorphism as iso
Z4 = [(a,b) for a in range(4) for b in range(4)]
R = nx.Graph(); R.add_nodes_from(Z4)
for x,y in itertools.combinations(Z4,2):
    if (x[0]==y[0]) != (x[1]==y[1]): R.add_edge(x,y)
Dset = {(1,0),(3,0),(0,1),(0,3),(1,1),(3,3)}
S = nx.Graph(); S.add_nodes_from(Z4)
for x,y in itertools.combinations(Z4,2):
    if ((x[0]-y[0])%4,(x[1]-y[1])%4) in Dset: S.add_edge(x,y)
for name,G in (('R',R),('S',S)):
    A = nx.to_numpy_array(G, nodelist=Z4, dtype=int)
    print(name, G.number_of_nodes(), G.number_of_edges(), set(dict(G.degree()).values()),
          'A^2==4I+2J:', np.array_equal(A@A, 4*np.eye(16,dtype=int)+2*np.ones((16,16),dtype=int)),
          'spec(L):', sorted(np.round(np.linalg.eigvalsh(6*np.eye(16)-A)).astype(int).tolist()) == sorted([0]+[4]*6+[8]*9))
    nb = G.subgraph(list(G.neighbors((0,0))))
    print('  neighbourhood of root:', sorted(d for _,d in nb.degree()), 'components', [len(c) for c in nx.connected_components(nb)],
          'isomorphic to C6:', nx.is_isomorphic(nb, nx.cycle_graph(6)), 'to 2K3:', nx.is_isomorphic(nb, nx.disjoint_union(nx.complete_graph(3),nx.complete_graph(3))))
    print('  4-cliques:', sum(1 for c in nx.enumerate_all_cliques(G) if len(c)==4), ' vertex transitive (Cayley/rook): yes by construction')
# rooted 1-balls differ
def rooted_ball(G,o,r):
    d=nx.single_source_shortest_path_length(G,o,cutoff=r); B=G.subgraph(d).copy(); nx.set_node_attributes(B,{v:v==o for v in B},'root'); return B
bR, bS = rooted_ball(R,(0,0),1), rooted_ball(S,(0,0),1)
print('rooted 1-balls isomorphic?', nx.is_isomorphic(bR,bS,node_match=lambda a,b:a['root']==b['root']), 'sizes', bR.number_of_nodes(), bS.number_of_nodes())
# every rooted 1-ball within each graph identical (vertex transitive)
print('all R balls iso:', all(nx.is_isomorphic(rooted_ball(R,v,1),bR,node_match=lambda a,b:a['root']==b['root']) for v in Z4),
      'all S balls iso:', all(nx.is_isomorphic(rooted_ball(S,v,1),bS,node_match=lambda a,b:a['root']==b['root']) for v in Z4))
# Cartesian products: spectra equal, Q4 counts
def q4(G): return sum(1 for c in nx.enumerate_all_cliques(G) if len(c)==4)
for H in (nx.path_graph(2), nx.cycle_graph(5), nx.complete_graph(4), nx.star_graph(3)):
    RH, SH = nx.cartesian_product(R,H), nx.cartesian_product(S,H)
    sR = np.sort(np.linalg.eigvalsh(nx.laplacian_matrix(RH).toarray().astype(float)))
    sS = np.sort(np.linalg.eigvalsh(nx.laplacian_matrix(SH).toarray().astype(float)))
    nH=H.number_of_nodes()
    print(f'H={nH} vertices: spectra equal {np.allclose(sR,sS)}; Q4(R□H)={q4(RH)} Q4(S□H)={q4(SH)}; predicted {nH*8+16*q4(H)},{16*q4(H)}; normalized diff {(q4(RH)-q4(SH))/(16*nH)}')
# random check of Q4(G□H)=|H|Q4(G)+|G|Q4(H)
rng=random.Random(2); ok=True
for _ in range(40):
    G=nx.gnp_random_graph(rng.randint(2,7),0.7,seed=rng.randint(0,9999)); H=nx.gnp_random_graph(rng.randint(2,6),0.7,seed=rng.randint(0,9999))
    if q4(nx.cartesian_product(G,H)) != H.number_of_nodes()*q4(G)+G.number_of_nodes()*q4(H): ok=False
print('Q4 Leibniz on random pairs:', ok)
