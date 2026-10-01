import random, time, cProfile, pstats
exec(open("t_fuzz.py").read().split("for trial in range(160):")[0])  # defs + seed
terms = []
for _ in range(random.randint(1, 3)):
    D, f = rand_defect(); B, g = rand_bg()
    c = Q(random.choice([1, -1, 2, -2, 3]), random.choice([1, 2, 5])); left = random.random() < 0.5
    X = (D * B if left else B * D) * c
    terms.append(X)
    def show(e, depth=0):
        name = type(e).__name__
        extra = ""
        if name == "Finite": extra = str([(str(c), g.n, g.edges) for c, g in e.terms])
        if name == "SparseEdgeDifference": extra = f"n={e.before.n} edits={e.edits}"
        if name == "LineCutDefect": extra = str(e.positions)
        print("  " * depth + name, extra, "deg", e.degree_bound)
        for attr in ("left", "right", "value"):
            if hasattr(e, attr) and hasattr(getattr(e, attr), "degree_bound"):
                show(getattr(e, attr), depth + 1)
    show(X)
from graphlocal.defects import _relative_plan
Xs = terms[0]
for Y in terms[1:]: Xs = Xs + Y
t = random.choice([Q(1, 3), Q(1), Q(2)])
plan = _relative_plan(t, Xs.degree_bound, Q(Xs.edit_bound), Q("1e-6"), 600)
print("t", t, "steps", plan[0], "radius", (plan[0] + 1) // 2, "edit", Xs.edit_bound)
for r in range(1, (plan[0] + 1) // 2 + 1, 2):
    t0 = time.time()
    try:
        h = Xs.local(r)
        print(f"r={r}: {len(h.values)} types, largest {max((k.graph.n for k in h.values), default=0)} vertices, {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print(f"r={r}: {type(e).__name__} {e} after {time.time()-t0:.1f}s", flush=True); break
    if time.time() - t0 > 60:
        pr = cProfile.Profile(); pr.enable(); Xs.local(r + 1); pr.disable()
        pstats.Stats(pr).sort_stats("cumulative").print_stats(8); break
