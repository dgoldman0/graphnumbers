from canon import *
reg = Registry()
for r in (1, 2, 3, 4):
    hs = []
    for n in (2 * r + 2, 2 * r + 4):
        G = grid_box(n); Gc = delete_edge(G, (0, 0), (1, 0))
        roots = set(bfs_dist(G, (0, 0), r + 2)) | set(bfs_dist(G, (1, 0), r + 2))
        h = histogram(reg, Gc, r, 1, roots); h = clean(histogram(reg, G, r, -1, roots, h))
        hs.append(h)
    assert hs[0] == hs[1]
    h = hs[0]
    print(f"P r={r}: variation {norm(reg, h)} (4r^2={4*r*r}), types {len(h)}")
