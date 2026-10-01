import networkx as nx
from math import comb
from collections import defaultdict
from gcore import REG, interaction, component_vector, clean
for k in range(1, 7):
    G = nx.star_graph(k); F = list(G.edges())
    lhs = interaction(G, F, component_vector)
    rhs = defaultdict(int)
    for j in range(k + 1):
        rhs[REG.key(nx.star_graph(j) if j else nx.empty_graph(1))] += (-1) ** j * comb(k, j)
    rhs = clean(rhs)
    print(k, "C_F == sum_j (-1)^j C(k,j) K_{1,j}:", lhs == rhs)
