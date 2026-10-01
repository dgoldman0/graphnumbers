from check4_laplacian import local_traces, a_k
from canon import delete_edge, edge_centered_tree
import itertools
def box(n, dim):
    adj = {}
    for p in itertools.product(range(-n, n + 1), repeat=dim):
        s = set()
        for i in range(dim):
            for e in (1, -1):
                q = list(p); q[i] += e
                if -n <= q[i] <= n: s.add(tuple(q))
        adj[p] = s
    return adj
def pred(d, c):
    return [0, -2, -4*d, -6*d*d - 6*d + 4, -8*d**3 - 24*d*d + 16*d - 8*c]
G = box(5, 3); u, v = (0,0,0), (1,0,0)
print("Z^3:", local_traces(G, delete_edge(G, u, v), u, v, 5), "pred", pred(6, 4))
for d in (3, 5):
    G = edge_centered_tree(d, 5)
    print(f"tree d={d}:", local_traces(G, delete_edge(G, 0, 1), 0, 1, 5), "pred", pred(d, 0))
