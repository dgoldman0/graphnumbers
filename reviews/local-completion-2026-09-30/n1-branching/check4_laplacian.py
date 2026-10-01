"""Check (30)-(34): relative Laplacian trace moments, tree d=4 vs square lattice."""
import numpy as np
import scipy.sparse as sp
from fractions import Fraction as F
from math import factorial
from canon import edge_centered_tree, delete_edge, grid_box, bfs_dist


def lap(adj):
    nodes = list(adj)
    idx = {v: i for i, v in enumerate(nodes)}
    rows, cols, vals = [], [], []
    for v in nodes:
        rows.append(idx[v]); cols.append(idx[v]); vals.append(len(adj[v]))
        for w in adj[v]:
            rows.append(idx[v]); cols.append(idx[w]); vals.append(-1)
    n = len(nodes)
    return sp.csr_matrix((np.array(vals, dtype=np.int64), (rows, cols)), shape=(n, n)), idx


def full_traces(adj, M):
    L, _ = lap(adj)
    P = sp.identity(L.shape[0], dtype=np.int64, format='csr')
    out = [L.shape[0]]
    for _ in range(M):
        P = (P @ L).tocsr()
        out.append(int(P.diagonal().sum()))
    return out


def local_traces(adj_before, adj_after, u, v, M):
    """Exact sum of diagonal differences over all vertices within distance M+1 of the edge."""
    near = set(bfs_dist(adj_before, u, M + 1)) | set(bfs_dist(adj_before, v, M + 1))
    Lb, idx = lap(adj_before)
    La, _ = lap(adj_after)
    out = [0] * (M + 1)
    for w in near:
        e = np.zeros(Lb.shape[0], dtype=np.int64); e[idx[w]] = 1
        xb, xa = e.copy(), e.copy()
        for m in range(1, M + 1):
            xb = Lb @ xb; xa = La @ xa
            out[m] += int(xa[idx[w]] - xb[idx[w]])
    return out


def a_k(adj, u, v, K):
    L, idx = lap(adj)
    b = np.zeros(L.shape[0], dtype=np.int64); b[idx[u]] = 1; b[idx[v]] = -1
    out, x = [], b.copy()
    for k in range(K + 1):
        out.append(int(b @ x)); x = L @ x
    return out


def _main():
    # ---- degree-4 tree ----
    res = {}
    for Ldepth in (6, 7):
        G = edge_centered_tree(4, Ldepth)
        Gc = delete_edge(G, 0, 1)
        if Ldepth == 6:
            tb, ta = full_traces(G, 6), full_traces(Gc, 6)
            res['tree_full_L6'] = [x - y for x, y in zip(ta, tb)]
        res[f'tree_local_L{Ldepth}'] = local_traces(G, Gc, 0, 1, 8)
        res[f'tree_a_L{Ldepth}'] = a_k(G, 0, 1, 3)
    # ---- square lattice ----
    for n in (8, 10, 12):
        G = grid_box(n)
        Gc = delete_edge(G, (0, 0), (1, 0))
        if n == 10:
            tb, ta = full_traces(G, 6), full_traces(Gc, 6)
            res['grid_full_n10'] = [x - y for x, y in zip(ta, tb)]
        res[f'grid_local_n{n}'] = local_traces(G, Gc, (0, 0), (1, 0), 8)
        res[f'grid_a_n{n}'] = a_k(G, (0, 0), (1, 0), 3)
    # ---- square lattice torus (independent boundary) ----
    N = 25
    tor = {(x, y): {((x + 1) % N, y), ((x - 1) % N, y), (x, (y + 1) % N), (x, (y - 1) % N)}
           for x in range(N) for y in range(N)}
    torc = delete_edge(tor, (0, 0), (1, 0))
    res['torus_local_N25'] = local_traces(tor, torc, (0, 0), (1, 0), 8)
    res['torus_full_N25'] = [x - y for x, y in zip(full_traces(torc, 6), full_traces(tor, 6))]
    for k, v in res.items():
        print(k, v)

    T = res['tree_local_L7']; P = res['grid_local_n12']
    assert res['tree_local_L6'][:7] == T[:7] and res['tree_full_L6'] == T[:7]
    assert res['grid_local_n10'] == P and res['grid_full_n10'] == P[:7] and res['grid_local_n8'][:7] == P[:7]
    assert res['torus_local_N25'] == P and res['torus_full_N25'] == P[:7]
    d = 4
    for c_e, D in ((0, T), (2, P)):
        pred = [0, -2, -4 * d, -6 * d * d - 6 * d + 4, -8 * d ** 3 - 24 * d * d + 16 * d - 8 * c_e]
        assert D[:5] == pred, (D[:5], pred)
        a = [2, 2 * d + 2, 2 * d * d + 6 * d, 2 * d ** 3 + 12 * d * d + 4 * d - 2 + 2 * c_e]
        print("a_k predicted", a)
    print("tree  Delta_0..8:", T)
    print("grid  Delta_0..8:", P)
    diff = [x - y for x, y in zip(T, P)]
    print("tree-grid       :", diff)
    coef = [F((-1) ** m * x, factorial(m)) for m, x in enumerate(diff)]
    print("H_t(B4)-H_t(P) Taylor coefficients t^0..t^8:", [str(c) for c in coef])

    # numerical heat check on finite graphs (boundary effects only at high order)
    import numpy.linalg as la
    def heat_diff(adj_b, adj_a, ts):
        Lb = lap(adj_b)[0].toarray().astype(float); La = lap(adj_a)[0].toarray().astype(float)
        eb, ea = la.eigvalsh(Lb), la.eigvalsh(La)
        return [np.sum(np.exp(-t * ea)) - np.sum(np.exp(-t * eb)) for t in ts]
    ts = [0.02, 0.05, 0.1, 0.2]
    G = edge_centered_tree(4, 6); Gc = delete_edge(G, 0, 1)
    ht = heat_diff(G, Gc, ts)
    G = grid_box(10); Gc = delete_edge(G, (0, 0), (1, 0))
    hp = heat_diff(G, Gc, ts)
    for t, x, y in zip(ts, ht, hp):
        series = sum(float(c) * t ** m for m, c in enumerate(coef))
        print(f"t={t}: finite heat diff tree-grid = {x - y:.6e}; series(<=8) = {series:.6e}; (2/3)t^4 = {2/3*t**4:.6e}")


if __name__ == "__main__":
    _main()
