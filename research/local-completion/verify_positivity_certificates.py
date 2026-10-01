"""Exact finite mechanisms for POSITIVITY_CERTIFICATES.md.

The fixtures check the pieces the note's proofs rely on, on explicit finite
graphs: divergences of local transports sum to zero, two explicit
certificates (a degree inequality and the radius-one P3 obstruction at
degree three) hold pointwise, positive but unbalanced arrays violate them,
finite permutation actions satisfy the trace identity as a rerooting
identity, and the product decomposition used for the robustness bound has
the stated masses. The general theorems are not checked here.
No graphlocal or third-party modules are imported.
"""
from fractions import Fraction as Q
from itertools import combinations
from pathlib import Path
import argparse
import hashlib
import json
import random


def require(condition, message):
    if not condition:
        raise ArithmeticError(message)


def graph(n, edges):
    g = [set() for _ in range(n)]
    for u, v in edges:
        g[u].add(v)
        g[v].add(u)
    return g


def random_graph(rng, n, p, cap=None):
    g = [set() for _ in range(n)]
    pairs = list(combinations(range(n), 2))
    rng.shuffle(pairs)
    for u, v in pairs:
        if rng.random() < p and (cap is None or (len(g[u]) < cap and len(g[v]) < cap)):
            g[u].add(v)
            g[v].add(u)
    return g


def link(g, o):
    """Neighbour link of o: its vertex list and component structure."""
    nb = sorted(g[o])
    comps, seen = [], set()
    for a in nb:
        if a in seen:
            continue
        comp, stack = [], [a]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack += [y for y in g[x] if y in g[o] and y not in seen]
        edges = sum(1 for x in comp for y in g[x] if y in comp)//2
        comps.append((len(comp), edges, comp))
    return comps


def is_p3_link(g, o):
    comps = link(g, o)
    return len(comps) == 1 and comps[0][0] == 3 and comps[0][1] == 2


def link_kind(g, o):
    shape = sorted((size, edges) for size, edges, _ in link(g, o))
    return {((2, 1),): "K2", ((1, 0), (2, 1)): "K2+K1"}.get(tuple(shape), "other")


def p3_ends(g, v):
    """The two path ends of a P3 link at v."""
    nb = list(g[v])
    mids = [b for b in nb if sum(1 for c in nb if c in g[b]) == 2]
    require(len(mids) == 1, "P3 link should have one middle vertex")
    return [a for a in nb if a != mids[0]]


def transports(g):
    """Divergences of two local transports; each must sum to zero over V(G)."""
    deg = [len(row) for row in g]
    div_deg = [sum(deg[v] for v in g[o])-deg[o]**2 for o in range(len(g))]
    div_far = []
    for o in range(len(g)):
        out = sum(1 for v in range(len(g)) if v != o and v not in g[o]
                  and any(v in g[w] for w in g[o]) and deg[o] == 3)
        inc = sum(1 for v in range(len(g)) if v != o and v not in g[v]
                  and o not in g[v] and any(o in g[w] for w in g[v]) and deg[v] == 3)
        div_far.append(out-inc)
    return div_deg, div_far


def divergence_fixtures(rng):
    checked = 0
    for n in range(2, 13):
        for _ in range(25):
            g = random_graph(rng, n, rng.choice([0.2, 0.4, 0.7]))
            for div in transports(g):
                require(sum(div) == 0, "divergence does not sum to zero")
            checked += 1
    return {"graphs_checked": checked,
            "transports": ["F(o,v)=deg(v)[v~o]", "F(o,v)=[dist(o,v)=2][deg(o)=3]"]}


def degree_certificate(rng):
    # f = sum_{v~o} deg v - 2 deg o + 1 = (deg o - 1)^2 + div F, F(o,v)=deg(v)[v~o].
    checked = 0
    for n in range(1, 13):
        for _ in range(25):
            g = random_graph(rng, n, rng.choice([0.15, 0.35, 0.6]))
            deg = [len(row) for row in g]
            div_deg, _ = transports(g)
            values = [sum(deg[v] for v in g[o])-2*deg[o]+1 for o in range(n)]
            for o in range(n):
                require(values[o] == (deg[o]-1)**2+div_deg[o], "certificate identity")
            require(sum(values) >= 0, "finite graph violates a certified inequality")
            checked += 1
    star = graph(4, [(0, 1), (0, 2), (0, 3)])
    centre_value = sum(len(star[v]) for v in star[0])-2*len(star[0])+1
    uniform = Q(sum(sum(len(star[v]) for v in star[o])-2*len(star[o])+1 for o in range(4)), 4)
    require(centre_value < 0 <= uniform, "balance is what the certificate uses")
    return {"inequality": "E[sum_{v~o} deg v] >= E[2 deg o - 1] on every unimodular law",
            "certificate": "(deg o - 1)^2 + divergence of F(o,v)=deg(v)[v~o]",
            "graphs_checked": checked,
            "unbalanced_point_mass_at_star_centre": centre_value,
            "uniform_root_on_the_same_star": str(uniform)}


def p3_certificate(rng):
    # At degree <= 3: 2[link in {K2, K2+K1}] - 2[link = P3] = g + div F, g >= 0,
    # where F(o,v)=[link(v)=P3 and o is an end of that path], so
    # div F(o) = received(o) - 2[link(o)=P3].
    checked, p3_roots = 0, 0
    graphs = []
    for n in range(1, 7):
        pairs = list(combinations(range(n), 2))
        for mask in range(1 << len(pairs)):
            g = graph(n, [pairs[i] for i in range(len(pairs)) if mask >> i & 1])
            if max((len(row) for row in g), default=0) <= 3:
                graphs.append(g)
    for _ in range(3000):
        graphs.append(random_graph(rng, rng.randint(7, 14), rng.choice([0.3, 0.5, 0.8]), cap=3))
    for g in graphs:
        n = len(g)
        p3 = [is_p3_link(g, o) for o in range(n)]
        received = [0]*n
        for v in range(n):
            if p3[v]:
                for a in p3_ends(g, v):
                    received[a] += 1
        for o in range(n):
            good = link_kind(g, o) in ("K2", "K2+K1")
            lhs = 2*good-2*p3[o]
            div = received[o]-2*p3[o]
            slack = lhs-div
            require(slack == 2*good-received[o] and slack >= 0, "P3 certificate is not pointwise")
            p3_roots += p3[o]
        checked += 1
    return {"inequality": "Pr[link = P3] <= Pr[link in {K2, K2+K1}] on unimodular laws of degree <= 3",
            "certificate": "slack 2[link in {K2,K2+K1}] - (P3 roots using o as an end) >= 0, plus a divergence",
            "graphs_checked": checked, "p3_roots_seen": p3_roots,
            "unbalanced_point_mass_at_cone_over_P3": -2}


def permutation_traces(rng):
    def compose(p, q):            # apply p, then q
        return [q[p[i]] for i in range(len(p))]

    def inverse(p):
        out = [0]*len(p)
        for i, x in enumerate(p):
            out[x] = i
        return out
    checked = 0
    for n in (1, 2, 3, 5, 8, 13, 21):
        for _ in range(40):
            a, b = list(range(n)), list(range(n))
            rng.shuffle(a)
            rng.shuffle(b)
            ab, ba = compose(a, b), compose(b, a)
            fixed_ab = [o for o in range(n) if ab[o] == o]
            fixed_ba = [o for o in range(n) if ba[o] == o]
            require(len(fixed_ab) == len(fixed_ba), "trace identity")
            # Rerooting form: o is fixed by ab exactly when o.a is fixed by ba.
            require(sorted(a[o] for o in fixed_ab) == fixed_ba, "rerooting along the a-edge")
            word = compose(compose(a, b), compose(inverse(a), b))
            conj = compose(compose(b, a), compose(word, inverse(compose(b, a))))
            require(sum(word[o] == o for o in range(n)) == sum(conj[o] == o for o in range(n)),
                    "conjugation invariance")
            checked += 1
    return {"identity": "phi(ab) = phi(ba): o is fixed by ab iff o.a is fixed by ba",
            "actions_checked": checked}


def product_decomposition():
    records = []
    for m, n in ((Q(0), Q(0)), (Q(1, 3), Q(0)), (Q(1, 2), Q(2, 5)), (Q(3), Q(7, 4))):
        positive = (1+m)*(1+n)+m*n
        negative = (1+m)*n+m*(1+n)
        require(positive-negative == 1 and negative == m+n+2*m*n, "product masses")
        require(1+2*negative == (1+2*m)*(1+2*n), "1+2R is multiplicative on decompositions")
        records.append({"m": str(m), "n": str(n), "negative_mass_of_product": str(negative)})
    return {"claim": "(1+m)s-mq times (1+n)s'-nq' has negative mass m+n+2mn", "records": records}


def serialize(value):
    if isinstance(value, Q):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("positivity_certificates_results.json"))
    args = parser.parse_args()
    rng = random.Random(20261001)
    results = {
        "status": "passed",
        "arithmetic": "Python standard library; exact integers and rationals; seeded random graphs",
        "divergences_sum_to_zero": divergence_fixtures(rng),
        "degree_certificate": degree_certificate(rng),
        "p3_certificate": p3_certificate(rng),
        "permutation_traces": permutation_traces(rng),
        "product_decomposition": product_decomposition(),
        "scope": "Finite mechanisms only; the certificate, duality and product theorems are proved in the note.",
        "proof_review": "The note's arguments remain unreviewed.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(serialize(results), indent=2, sort_keys=True)+"\n")
    print("PASS: positivity certificate fixtures:", args.output)


if __name__ == "__main__":
    main()
