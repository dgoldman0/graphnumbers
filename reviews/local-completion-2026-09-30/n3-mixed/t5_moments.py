import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from mm import *
import numpy as np
def moments(G, m):
    L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes, key=str)).toarray().astype(object)
    out=[]; P=np.identity(L.shape[0], dtype=object)
    for k in range(m+1):
        out.append(int(np.trace(P))); P = P.dot(L)
    return out
for name, (G, e) in (("B4", tree_with_edge(4, 4)), ("P", grid_with_edge(5)), ("B3", tree_with_edge(3,5))):
    c = Cut(name, G, e)
    a, b = moments(c.after, 4), moments(c.before, 4)
    print(name, [x-y for x, y in zip(a, b)])
