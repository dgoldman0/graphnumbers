from canon import *


def S(r, d):
    return sum((d - 1) ** j for j in range(r))


def tree_cut(reg, d, r, L=None):
    """T_r(B_d) from a finite buffered tree (roots near the edge; others cancel trivially)."""
    L = 2 * r + 1 if L is None else L
    G = edge_centered_tree(d, L)
    Gc = delete_edge(G, 0, 1)
    roots = set(bfs_dist(G, 0, r + 2)) | set(bfs_dist(G, 1, r + 2))
    h = histogram(reg, Gc, r, 1, roots)
    h = histogram(reg, G, r, -1, roots, h)
    h = clean(h)
    atoms = {}
    for c, v in h.items():
        adj, dist = reg.reps[c]
        if v < 0:
            atoms['R'] = c
        else:
            defic = [x for x in adj if dist[x] < r and len(adj[x]) != d]
            atoms[dist[defic[0]]] = c
    return h, atoms


def cut_line(reg, r):
    n = 4 * r + 6
    h = histogram(reg, path_graph(n), r, 1)
    return clean(histogram(reg, cycle_graph(n), r, -1, None, h))
