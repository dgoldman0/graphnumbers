import json, traceback
from fractions import Fraction as Q
from graphlocal import *
from graphlocal.interaction_bounds import interaction_geometry, interaction_heat_bound, GeometricInteraction
from graphlocal.interaction_moments import interaction_moments, tree_interaction_leading
from graphlocal.nonspectral import certified_jet, ROOT_DEGREE, rooted_cliques, link_components, JointDistribution, MomentJet

def run(name, f):
    try:
        r = f()
        print(name, "->", r)
    except Exception as e:
        print(name, "EXC", type(e).__name__, e)

H = Finite.from_graph(path(2), normalize=True)
run("neumann zero to_data", lambda: json.dumps(NeumannInverse(0).approximation_certificate(2, 1).to_data()))
run("neumann scalar r=3", lambda: NeumannInverse(Q(1,3)).approximate(3, 2, "1e-9").histogram.to_data())
run("neumann mass None r=0", lambda: None)
class NoMass(Element):
    degree_bound, variation_bound = 1, Q(1, 4)
    def local(self, r): return (H / 4).local(r)
run("neumann nomass r=0", lambda: NeumannInverse(NoMass()).approximate(0, 1, "1e-9"))
run("neumann nomass r=1", lambda: NeumannInverse(NoMass()).approximate(1, 1, "1e-6").error)
run("cutexp t=0 r=0", lambda: CutLineExponential(0).approximation_certificate(0).to_data())
run("cutexp k=0", lambda: CutLineExponential(1).approximate(1, 0))
run("cutexp r=-1", lambda: CutLineExponential(1).approximate(-1, 1))
run("cutexp eps float", lambda: CutLineExponential(1).approximate(1, 1, 1e-3))
run("cutexp norm_bound r=0", lambda: CutLineExponential(1).norm_bound(0, 1))
run("local_inverse r0", lambda: local_inverse_certificate(3, LocalHistogram(0, [(graph(1), Q(1, 3))])).to_data())
run("geometry k=1 max_cycles=0", lambda: interaction_geometry(EdgeInteraction(path(2), [(0, 1, -1)]), max_cycles=0))
run("geometry empty to_data", lambda: interaction_geometry(EdgeInteraction(path(2), [])).to_data())
run("heat bound empty to_data", lambda: interaction_heat_bound(EdgeInteraction(path(2), []), 1).to_data())
run("heat bound after_step", lambda: interaction_heat_bound(EdgeInteraction(path(3), [(0, 1, -1)]), 1, after_step=0).to_data())
run("moments order 0 to_data", lambda: interaction_moments(EdgeInteraction(path(3), [(0, 1, -1)]), 0).to_data())
run("controlled to_data zero", lambda: controlled_heat(EdgeInteraction(path(2), []), 1).to_data())
run("controlled degree0", lambda: controlled_heat(Finite.scalar(Q(3, 2)), 5).interval)
run("tree k=1", lambda: tree_interaction_leading(EdgeInteraction(path(2), [(0, 1, -1)])))
run("geom wrapper heat", lambda: interaction_heat_bound(GeometricInteraction(EdgeInteraction(path(4), [(0,1,-1),(2,3,-1)])), 1))
run("jet order0", lambda: certified_jet(Finite.scalar(2), (ROOT_DEGREE,), 0).to_data())
run("rooted_cliques kw", lambda: rooted_cliques(3) is rooted_cliques(size=3))
run("link kw", lambda: link_components(path(2)) is link_components(path(2), None))
run("jd compat kw", lambda: JointDistribution((rooted_cliques(3),), {(1,): 1}) + JointDistribution((rooted_cliques(size=3),), {(1,): 1}))
run("jet float", lambda: MomentJet((ROOT_DEGREE,), 1, {(1,): 0.5}))
