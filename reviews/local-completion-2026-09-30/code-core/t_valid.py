from fractions import Fraction as Q
from graphlocal import *
from graphlocal.graphs import IsoGraph
import networkx as nx
cases = {
 "graph with duplicate/reversed edge (multigraph list)": lambda: graph(2, [(0, 1), (1, 0)]).edges,
 "from_networkx self-loop": lambda: from_networkx(nx.Graph([(0, 0), (0, 1)])),
 "from_networkx multigraph": lambda: from_networkx(nx.MultiGraph([(0, 1), (0, 1)])),
 "Finite float coeff": lambda: Finite([(0.5, path(2))]),
 "approximate k=0": lambda: Line().approximate(1, 0, "1e-3"),
 "approximate eps=0": lambda: Line().approximate(1, 1, 0),
 "approximate radius=-1": lambda: Line().approximate(-1, 1, "1e-3"),
 "approximate radius=True": lambda: Line().approximate(True, 1, "1e-3"),
 "heat max_steps=True": lambda: heat_return(Line(), "1/2", "1e-3", max_steps=True),
 "heat eps float": lambda: heat_return(Line(), "1/2", 1e-3),
 "relative eps 0": lambda: relative_heat(CutLineDefect(), "1/2", 0),
 "PreparedLocal radius -1": lambda: PreparedLocal(Line(), -1),
 "walk_observable(-1)": lambda: walk_observable(-1),
 "catalog max_degree True": lambda: catalog(3, max_degree=True),
 "reconstruct empty graph": lambda: reconstruct(Line().local(1), [graph(0)]),
 "CutInteraction(True)": lambda: CutInteraction(True),
 "LineCutDefect float": lambda: LineCutDefect([0, 1.0]),
 "SED bool sign": lambda: SparseEdgeDifference(cycle(5), [(0, 1, True)]),
 "LocalHistogram disconnected key": lambda: LocalHistogram(2, [(graph(2), 1)]),
 "Interval float": lambda: Interval(0.1, 0.2),
 "lazy cache bypass of validation (Fraction degree)": lambda: __import__("graphlocal.heat", fromlist=["x"]).lazy_returns(path(2), Q(2), 3),
 "Scale(Line,0).local(-1)": lambda: (Line() * 0).local(-1),
 "exp max_terms=0": lambda: exp(Line(), max_terms=0),
 "reconstruct non-histogram target": lambda: reconstruct(Finite.from_graph(path(2)), [path(2)]),
}
for name, f in cases.items():
    try:
        r = f()
        print(f"ACCEPTED  {name}: {r!r}"[:140])
    except Exception as e:
        print(f"rejected  {name}: {type(e).__name__}: {e}"[:140])
