"""
Operator-linear 'optimal morphing' experiment between two graphs.

Requirements:
    pip install networkx numpy matplotlib
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Graph construction
# ----------------------------------------------------------------------

def build_grid_graph(n_side: int = 10, seed: int = 0) -> nx.Graph:
    """
    Build a 2D grid graph with n_side * n_side nodes,
    then relabel nodes to integers 0..n-1.
    """
    G = nx.grid_2d_graph(n_side, n_side)
    # Convert (i, j) labels to 0..n-1
    G = nx.convert_node_labels_to_integers(G)
    return G


def build_sbm_graph(n: int = 100,
                    sizes=(50, 50),
                    p_in: float = 0.3,
                    p_out: float = 0.02,
                    seed: int = 1) -> nx.Graph:
    """
    Build a simple 2-block stochastic block model (SBM) graph.
    """
    if sum(sizes) != n:
        raise ValueError("sum(sizes) must equal n")

    probs = [
        [p_in, p_out],
        [p_out, p_in]
    ]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    G = nx.convert_node_labels_to_integers(G)
    return G


# ----------------------------------------------------------------------
# 2. Operators and morphing path
# ----------------------------------------------------------------------

def normalized_adjacency(G: nx.Graph, eps: float = 1e-6) -> np.ndarray:
    """
    Symmetric normalized adjacency:
        T = D^{-1/2} A D^{-1/2}
    where D is the degree matrix plus a small epsilon on the diagonal
    to regularize isolated vertices.
    """
    A = nx.to_numpy_array(G)
    deg = A.sum(axis=1) + eps
    D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    T = D_inv_sqrt @ A @ D_inv_sqrt
    # T is symmetric by construction
    return T


def create_operator_geodesic(T0: np.ndarray, T1: np.ndarray):
    """
    Return a function T_t(t) = (1 - t)*T0 + t*T1
    representing the operator-linear geodesic.
    """
    def T_t(t: float) -> np.ndarray:
        return (1.0 - t) * T0 + t * T1
    return T_t


# ----------------------------------------------------------------------
# 3. Sampling simple graphs from an operator kernel
# ----------------------------------------------------------------------

def sample_graph_from_operator(T: np.ndarray,
                               rng: np.random.Generator,
                               rescale: bool = True) -> nx.Graph:
    """
    Interpret a symmetric matrix T as a kernel and sample a simple
    undirected graph from it.

    Steps:
    - (Optionally) shift and rescale T into [0, 1].
    - Use entries as Bernoulli parameters to sample adjacency.
    - Enforce symmetry and zero diagonal.
    """
    T_rescaled = T.copy()

    if rescale:
        mn = T_rescaled.min()
        T_rescaled -= mn
        mx = T_rescaled.max()
        if mx > 0:
            T_rescaled /= mx

    # Clamp to [0, 1] for safety
    P = np.clip(T_rescaled, 0.0, 1.0)

    n = P.shape[0]
    R = rng.random(size=(n, n))
    A = (R < P).astype(int)

    # Symmetrize and remove self-loops
    A = np.triu(A, 1)
    A = A + A.T
    np.fill_diagonal(A, 0)

    G = nx.from_numpy_array(A)
    return G


# ----------------------------------------------------------------------
# 4. Graph statistics along the morph
# ----------------------------------------------------------------------

def triangle_count(G: nx.Graph) -> int:
    """
    Total number of triangles in G.
    """
    tri_dict = nx.triangles(G)
    return sum(tri_dict.values()) // 3


def avg_clustering(G: nx.Graph) -> float:
    """
    Average clustering coefficient.
    """
    return nx.average_clustering(G)


def avg_shortest_path_largest_cc(G: nx.Graph) -> float:
    """
    Average shortest path length on the largest connected component.
    Returns NaN for an empty graph.
    """
    if G.number_of_nodes() == 0:
        return float("nan")

    components = list(nx.connected_components(G))
    if not components:
        return float("nan")

    largest_cc_nodes = max(components, key=len)
    H = G.subgraph(largest_cc_nodes).copy()

    # If the component has a single node, the path length is trivially zero
    if H.number_of_nodes() <= 1:
        return 0.0

    return nx.average_shortest_path_length(H)


def spectral_features(T: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Top-k eigenvalues of a symmetric matrix T, sorted in descending order.
    """
    vals = np.linalg.eigvalsh(T)  # for symmetric matrices
    vals_sorted = np.sort(vals)[::-1]
    k = min(k, len(vals_sorted))
    return vals_sorted[:k]


def frobenius_norm(A: np.ndarray) -> float:
    """
    Frobenius norm of a matrix.
    """
    return float(np.linalg.norm(A, "fro"))


# ----------------------------------------------------------------------
# 5. Experiment runner
# ----------------------------------------------------------------------

def run_morph_experiment(
    n: int = 100,
    n_side_grid: int = 10,
    ts: np.ndarray = None,
    seed: int = 42,
    rescale_sampling: bool = True,
    k_eigs: int = 10
):
    """
    Run a complete morphing experiment and produce several plots.

    Parameters
    ----------
    n : int
        Total number of nodes.
    n_side_grid : int
        Side length of the grid graph. Product n_side_grid^2 should equal n.
    ts : np.ndarray
        Array of time points t in [0, 1] at which to evaluate the morph.
    seed : int
        Random seed for reproducibility.
    rescale_sampling : bool
        Whether to rescale the operator T_t to [0, 1] before sampling edges.
    k_eigs : int
        Number of top eigenvalues to track.
    """

    if ts is None:
        ts = np.linspace(0.0, 1.0, 21)

    if n_side_grid * n_side_grid != n:
        raise ValueError("n_side_grid^2 must equal n for this setup.")

    rng = np.random.default_rng(seed)

    # 1) Build graphs
    print("Building graphs...")
    G0 = build_grid_graph(n_side=n_side_grid, seed=seed)
    G1 = build_sbm_graph(
        n=n,
        sizes=(n // 2, n - n // 2),
        p_in=0.3,
        p_out=0.02,
        seed=seed,
    )

    # 2) Build operators
    print("Building operators...")
    T0 = normalized_adjacency(G0)
    T1 = normalized_adjacency(G1)

    T_t = create_operator_geodesic(T0, T1)

    # Prepare containers for statistics
    tri_counts = []
    clustering_vals = []
    avg_path_vals = []
    d_frob_to_0 = []
    d_frob_to_1 = []
    eigs_over_time = []  # list of arrays of length k_eigs

    # 3) Iterate along the morph
    print("Sampling along operator-linear path...")
    for t in ts:
        T = T_t(t)

        # Frobenius distances to endpoints
        d_frob_to_0.append(frobenius_norm(T - T0))
        d_frob_to_1.append(frobenius_norm(T - T1))

        # Spectral features
        eigs = spectral_features(T, k=k_eigs)
        eigs_over_time.append(eigs)

        # Sample a simple graph from T
        G_sample = sample_graph_from_operator(T, rng=rng, rescale=rescale_sampling)

        # Graph statistics
        tri_counts.append(triangle_count(G_sample))
        clustering_vals.append(avg_clustering(G_sample))
        avg_path_vals.append(avg_shortest_path_largest_cc(G_sample))

    eigs_over_time = np.array(eigs_over_time)  # shape: (len(ts), k_eigs)
    tri_counts = np.array(tri_counts)
    clustering_vals = np.array(clustering_vals)
    avg_path_vals = np.array(avg_path_vals)
    d_frob_to_0 = np.array(d_frob_to_0)
    d_frob_to_1 = np.array(d_frob_to_1)

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------

    print("Plotting results...")

    # 6.1 Eigenvalues vs t
    plt.figure(figsize=(8, 5))
    for i in range(min(k_eigs, eigs_over_time.shape[1])):
        plt.plot(ts, eigs_over_time[:, i], label=f"eig {i+1}")
    plt.xlabel("t")
    plt.ylabel("Eigenvalue")
    plt.title("Top-k eigenvalues along operator-linear morph")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 6.2 Triangles and clustering vs t
    fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax[0].plot(ts, tri_counts, marker="o")
    ax[0].set_ylabel("Triangle count")
    ax[0].set_title("Triangle count along morph")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(ts, clustering_vals, marker="o", color="tab:orange")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("Average clustering")
    ax[1].set_title("Average clustering along morph")
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 6.3 Average shortest path (largest CC) vs t
    plt.figure(figsize=(8, 5))
    plt.plot(ts, avg_path_vals, marker="o")
    plt.xlabel("t")
    plt.ylabel("Avg shortest path (largest CC)")
    plt.title("Average path length along morph")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 6.4 Frobenius distance to endpoints vs t
    plt.figure(figsize=(8, 5))
    plt.plot(ts, d_frob_to_0, label="||T_t - T0||_F", marker="o")
    plt.plot(ts, d_frob_to_1, label="||T_t - T1||_F", marker="s")
    plt.xlabel("t")
    plt.ylabel("Frobenius distance")
    plt.title("Frobenius distances to endpoints along morph")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Done.")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Example parameters:
    # n = 100, 10x10 grid, 21 time steps from 0 to 1
    ts = np.linspace(0.0, 1.0, 21)
    run_morph_experiment(
        n=100,
        n_side_grid=10,
        ts=ts,
        seed=123,
        rescale_sampling=True,
        k_eigs=10,
    )

