"""Conventional floating-point comparators for local heat-trace corrections.

These examples require the optional NumPy/SciPy benchmark dependencies. They
never produce a graphlocal rational certificate. Timings include conversion of
the supplied Graph objects into matrices, but exclude construction of those
Graph objects; an experiment driver must report graph construction separately.

The common-subspace block Krylov method is Algorithm 1 specialized to symmetric
polynomial Krylov spaces in Cortinovis, Kressner and Massei, "Divide-and-conquer
methods for functions of matrices with banded or hierarchical low-rank
structure", arXiv:2107.04337. Their Theorem 2 establishes trace exactness for
polynomials of degree <= 2m in exact arithmetic. Numerical rank truncation,
roundoff, and approximation of the exponential remain uncertified here.
"""
from __future__ import annotations

import math
import time as clock

import numpy as np
from scipy.linalg import eigh, svd
from scipy.sparse import csr_matrix, eye
from scipy.special import gammainc, gammaln


def _validate(g, time, steps=None):
    if not g.n:
        raise ValueError("Heat comparisons require a nonempty graph")
    time = float(time)
    if not math.isfinite(time) or time < 0:
        raise ValueError("time must be finite and nonnegative")
    if steps is not None and (isinstance(steps, bool)
                              or not isinstance(steps, int) or steps < 0):
        raise ValueError("steps must be a nonnegative integer")
    return time


def _laplacian(g):
    rows, columns, values = [], [], []
    for u in range(g.n):
        degree = g.rows[u].bit_count()
        if degree:
            rows.append(u)
            columns.append(u)
            values.append(float(degree))
        for v in g.neighbors(u):
            rows.append(u)
            columns.append(v)
            values.append(-1.0)
    return csr_matrix((values, (rows, columns)), shape=(g.n, g.n))


def dense_heat(g, time, *, normalize=True):
    """Return (heat trace, metrics) using dense symmetric diagonalization.

    The trace is divided by |V| when normalize=True. Construction of the dense
    Laplacian is included in the timings. Floating-point error is unbounded by
    this routine; the value is an independent numerical comparison only.
    """
    start = clock.perf_counter()
    time = _validate(g, time)
    laplacian = _laplacian(g).toarray()
    built = clock.perf_counter()
    eigenvalues = eigh(laplacian, eigvals_only=True, check_finite=False)
    scale = g.n if normalize else 1
    value = math.fsum(np.exp(-time * eigenvalues)) / scale
    done = clock.perf_counter()
    return float(value), {
        "method": "dense_symmetric_eigendecomposition",
        "normalized": normalize,
        "matrix_seconds": built - start,
        "eigensolve_seconds": done - built,
        "total_seconds": done - start,
        "matrix_bytes": int(laplacian.nbytes),
        "minimum_computed_eigenvalue": float(eigenvalues[0]),
        "error_status": "Floating-point roundoff is not certified.",
    }


def sparse_uniformized_heat(g, time, steps, degree_bound=None, *, normalize=True):
    """Deterministic sparse-power heat trace, including powers 0 through steps.

    For D >= max degree, P=I-L/D is stochastic and
    exp(-tL)=exp(-Dt) sum_j (Dt)^j P^j/j!. The omitted normalized trace is at
    most the Poisson tail recorded in truncation_bound (multiplied by |V| for
    normalize=False). This mathematical truncation bound excludes floating
    roundoff. All sparse powers are computed, so this comparator has no sampling
    noise and may become expensive as those matrices fill in.
    """
    start = clock.perf_counter()
    time = _validate(g, time, steps)
    if degree_bound is None:
        degree_bound = g.max_degree
    if (isinstance(degree_bound, bool) or not isinstance(degree_bound, int)
            or degree_bound < g.max_degree):
        raise ValueError("degree_bound must be an integer >= graph maximum degree")
    laplacian = _laplacian(g)
    p = eye(g.n, format="csr")
    if degree_bound:
        p = p - laplacian / degree_bound
        p.eliminate_zeros()
    built = clock.perf_counter()
    intensity = degree_bound * time
    scale = g.n if normalize else 1
    power = eye(g.n, format="csr")
    peak_nonzeros, sparse_multiplies = int(power.nnz), 0
    contributions = []
    for j in range(steps + 1):
        if j:
            power = power @ p
            sparse_multiplies += 1
            peak_nonzeros = max(peak_nonzeros, int(power.nnz))
        if not intensity:
            weight = 1.0 if j == 0 else 0.0
        else:
            weight = math.exp(-intensity + j * math.log(intensity) - gammaln(j + 1))
        contributions.append(weight * float(power.diagonal().sum()) / scale)
    tail = float(gammainc(steps + 1, intensity)) * g.n / scale
    done = clock.perf_counter()
    return math.fsum(contributions), {
        "method": "deterministic_sparse_uniformization",
        "normalized": normalize,
        "degree_bound": degree_bound,
        "steps": steps,
        "matrix_seconds": built - start,
        "powers_seconds": done - built,
        "total_seconds": done - start,
        "sparse_nonzeros": int(p.nnz),
        "peak_power_nonzeros": peak_nonzeros,
        "sparse_matrix_multiplies": sparse_multiplies,
        "truncation_bound": tail,
        "error_status": "Poisson truncation bound excludes floating-point roundoff.",
    }


def _edits(before, after, edits):
    if before.n != after.n:
        raise ValueError("The two graphs must have the same vertex labels and count")
    if edits is None:
        edits = []
        for u, (old, new) in enumerate(zip(before.rows, after.rows)):
            changed = old ^ new
            while changed:
                bit = changed & -changed
                v = bit.bit_length() - 1
                changed -= bit
                if u < v:
                    edits.append((u, v, 1 if new & bit else -1))
    edits = tuple(edits)
    rows, seen = list(before.rows), set()
    for edit in edits:
        if len(edit) != 3:
            raise ValueError("Each edit must be (u, v, +1 or -1)")
        u, v, sign = edit
        if (isinstance(u, bool) or isinstance(v, bool)
                or not isinstance(u, int) or not isinstance(v, int)
                or not 0 <= u < before.n or not 0 <= v < before.n or u == v):
            raise ValueError("Edited edge endpoints must be distinct vertex labels")
        if isinstance(sign, bool) or sign not in (-1, 1):
            raise ValueError("Edit sign must be +1 for addition or -1 for deletion")
        key = tuple(sorted((u, v)))
        if key in seen:
            raise ValueError("Each edited edge must occur exactly once")
        seen.add(key)
        exists = bool(rows[u] & (1 << v))
        if exists != (sign == -1):
            raise ValueError("Edit sign disagrees with the original graph")
        rows[u] ^= 1 << v
        rows[v] ^= 1 << u
    if tuple(rows) != after.rows:
        raise ValueError("The edit list does not exactly describe the graph difference")
    return edits


def _orthogonal_block(block, basis, threshold):
    # Two complete orthogonalization passes suppress loss of independence.
    if basis.shape[1]:
        for _ in range(2):
            block = block - basis @ (basis.T @ block)
    if not block.shape[1]:
        return block
    left, singular, _ = svd(block, full_matrices=False, check_finite=False)
    return left[:, singular > threshold]


def _krylov_projection(before, after, steps, edits=None, rank_tolerance=1e-12):
    """Shared implementation exposed privately for moment-exactness tests."""
    start = clock.perf_counter()
    _validate(before, 0, steps)
    if steps < 1:
        raise ValueError("Krylov steps must be at least one")
    if not math.isfinite(rank_tolerance) or not 0 < rank_tolerance < 1:
        raise ValueError("rank_tolerance must lie strictly between zero and one")
    edits = _edits(before, after, edits)
    laplacian = _laplacian(before)
    incidence = np.zeros((before.n, len(edits)))
    signs = np.empty(len(edits))
    for j, (u, v, sign) in enumerate(edits):
        incidence[u, j], incidence[v, j], signs[j] = 1, -1, sign
    built = clock.perf_counter()
    basis = np.empty((before.n, 0))
    threshold = rank_tolerance * max(1.0, float(np.linalg.norm(incidence)))
    frontier = _orthogonal_block(incidence, basis, threshold)
    basis = frontier.copy()
    initial_rank, matvec_columns = basis.shape[1], 0
    block_dimensions = [int(initial_rank)]
    breakdown = not initial_rank
    for _ in range(1, steps):
        if not frontier.shape[1] or basis.shape[1] == before.n:
            breakdown = True
            break
        candidate = laplacian @ frontier
        matvec_columns += frontier.shape[1]
        threshold = rank_tolerance * max(1.0, float(np.linalg.norm(candidate)))
        frontier = _orthogonal_block(candidate, basis, threshold)
        block_dimensions.append(int(frontier.shape[1]))
        if not frontier.shape[1]:
            breakdown = True
            break
        basis = np.column_stack((basis, frontier))
    formed = clock.perf_counter()
    action = laplacian @ basis
    matvec_columns += basis.shape[1]
    original = basis.T @ action
    original = (original + original.T) / 2
    seed = basis.T @ incidence
    modified = original + (seed * signs) @ seed.T
    modified = (modified + modified.T) / 2
    residual = float(np.linalg.norm(action - basis @ original))
    orthogonality = float(np.linalg.norm(basis.T @ basis - np.eye(basis.shape[1])))
    projected = clock.perf_counter()
    metrics = {
        "method": "defect_seeded_common_subspace_block_krylov",
        "vertices": before.n,
        "edited_edges": len(edits),
        "initial_block_rank": int(initial_rank),
        "requested_steps": steps,
        "block_dimensions": block_dimensions,
        "subspace_dimension": int(basis.shape[1]),
        "moment_exact_through_degree": 2 * steps,
        "moment_exactness_qualification": "Exact arithmetic; numerical rank truncation may alter the space.",
        "numerical_breakdown": breakdown,
        "rank_tolerance": rank_tolerance,
        "matvec_columns": int(matvec_columns),
        "sparse_nonzeros": int(laplacian.nnz),
        "estimated_sparse_scalar_products": int(laplacian.nnz * matvec_columns),
        "basis_bytes": int(basis.nbytes),
        "basis_orthogonality_error": orthogonality,
        "invariant_residual": residual,
        "matrix_seconds": built - start,
        "krylov_seconds": formed - built,
        "projection_seconds": projected - formed,
        "projection_total_seconds": projected - start,
    }
    return original, modified, metrics


def defect_krylov_heat(before, after, time, steps, edits=None,
                       rank_tolerance=1e-12, *, normalize=True):
    """Return (heat-trace correction, metrics) using an edge-seeded Krylov space.

    Estimates tr(exp(-t L_after)-exp(-t L_before)), divided by vertex count if
    normalize=True. Graph labels must agree. Edits are (u,v,+1) additions or
    (u,v,-1) deletions; None discovers them. A supplied list is fully checked.

    m=steps builds span{B,L_before B,...,L_before**(m-1) B}, where B contains
    edited-edge incidence vectors. Both Laplacians are projected into the SAME
    orthonormal basis. In exact arithmetic this preserves trace differences of
    polynomials of degree <=2m. Deflated two-pass block Arnoldi and SVD rank
    detection build the basis. Values and projection errors remain numerical;
    there is no rigorous error bound for the returned exponential estimate.
    """
    start = clock.perf_counter()
    time = _validate(before, time)
    original, modified, metrics = _krylov_projection(
        before, after, steps, edits, rank_tolerance)
    projected = clock.perf_counter()
    old = eigh(original, eigvals_only=True, check_finite=False)
    new = eigh(modified, eigvals_only=True, check_finite=False)
    # Pair sorted eigenvalues and use expm1 to avoid subtracting two O(dim)
    # traces. Select the larger exponential as prefactor to avoid overflow.
    differences = []
    for a, b in zip(old, new):
        if b >= a:
            differences.append(math.exp(-time * a) * math.expm1(-time * (b - a)))
        else:
            differences.append(-math.exp(-time * b) * math.expm1(-time * (a - b)))
    unnormalized = math.fsum(differences)
    value = unnormalized / before.n if normalize else unnormalized
    done = clock.perf_counter()
    metrics.update({
        "normalized": normalize,
        "unnormalized_estimate": unnormalized,
        "normalized_estimate": unnormalized / before.n,
        "eigensolve_seconds": done - projected,
        "total_seconds": done - start,
        "error_status": "Projection and floating-point errors are not certified.",
    })
    return float(value), metrics
