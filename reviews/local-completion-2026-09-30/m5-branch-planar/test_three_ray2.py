import networkx as nx
from collections import defaultdict
from gcore import Registry, rooted_ball, interaction
import test_three_ray as t3
G = t3.spider(4); F = [(0, n) for n in G.neighbors(0)]
h = interaction(G, F, lambda H: t3.hist(H, 1))
for k, v in h.items():
    H, root = t3.reg.reps[k]
    print(v, "graph with", H.number_of_nodes(), "vertices, root degree", H.degree(root), "edges", H.number_of_edges())
