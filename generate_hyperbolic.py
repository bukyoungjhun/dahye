"""Generate an S1 random graph and save only its largest connected component.

Produces a reindexed edgelist (0..N_lcc-1) and matching latent coordinates
[x, kappa, z] for LCC nodes only.

Usage:
    python generate_s1_lcc.py
    python generate_s1_lcc.py --N 1024 --beta 2 --k-mean 4 --seed 42 \
        --edge-out edgelist.txt --coord-out coordinate.txt
"""

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
from numba import njit
from numba.typed import List


@njit
def _generate_edges(N, beta, mu, x, kappa, seed):
    np.random.seed(seed)
    out_i = List()
    out_j = List()
    for i in range(N):
        for j in range(i):
            dx = abs(x[i] - x[j])
            dx = min(dx, N - dx)
            pij = 1.0 / (1.0 + (dx / mu / kappa[i] / kappa[j]) ** beta)
            if np.random.random() < pij:
                out_i.append(i)
                out_j.append(j)
    return out_i, out_j


def generate_s1_lcc(N, beta, k_mean, seed):
    """Return (edgelist, coordinate) for the LCC of an S1 graph."""
    mu = beta / (2.0 * np.pi * k_mean) * np.sin(np.pi / beta)
    rng = np.random.default_rng(seed)
    x = N * rng.random(N)
    kappa = np.arange(1, N + 1, dtype=np.float64) ** -0.5
    kappa = N * k_mean * kappa / kappa.sum()

    edge_i, edge_j = _generate_edges(N, beta, mu, x, kappa, seed)
    edges = np.column_stack((np.array(edge_i), np.array(edge_j)))

    g = nx.Graph()
    g.add_nodes_from(range(N))
    g.add_edges_from(edges.tolist())

    lcc_nodes = sorted(max(nx.connected_components(g), key=len))
    new_idx = {old: i for i, old in enumerate(lcc_nodes)}

    edgelist = np.array(
        [[new_idx[n], new_idx[m]] for n, m in edges if n in new_idx and m in new_idx],
        dtype=np.int64,
    )
    if edgelist.size == 0:
        edgelist = np.empty((0, 2), dtype=np.int64)

    coordinate = np.zeros((len(lcc_nodes), 3), dtype=np.float64)
    for old, new in new_idx.items():
        coordinate[new, 0] = x[old]
        coordinate[new, 1] = kappa[old]
        coordinate[new, 2] = 0.0

    return edgelist, coordinate, mu


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate an S1 graph and save its largest connected component."
    )
    p.add_argument("--N", type=int, default=128, help="Number of nodes before LCC.")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--k-mean", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--edge-out", type=str,  default="edgelist.txt")
    p.add_argument("--coord-out", type=str, default="coordinate.txt")
    return p.parse_args()


def main():
    args = parse_args()
    edgelist, coordinate, mu = generate_s1_lcc(
        args.N, args.beta, args.k_mean, args.seed
    )

    edge_out = Path(args.edge_out)
    coord_out = Path(args.coord_out)
    edge_out.parent.mkdir(parents=True, exist_ok=True)
    coord_out.parent.mkdir(parents=True, exist_ok=True)

    np.savetxt(edge_out, edgelist, fmt="%d")
    np.savetxt(coord_out, coordinate)

if __name__ == "__main__":
    main()
