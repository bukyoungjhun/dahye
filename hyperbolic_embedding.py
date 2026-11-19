import collections
import numpy as np
import networkx as nx
import scipy.special
from scipy.sparse import csr_matrix, identity, diags
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.optimize import minimize_scalar

try:
    import _hyperbolic_embedding as _native
    _GradStateCore = _native.GradState
except ImportError:
    _native = None
    _GradStateCore = None

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def dec(f):
            return f
        return dec if not args else dec(args[0])


def set_seed(seed):
    if _native is not None:
        _native.set_seed(seed)
    np.random.seed(seed)


class GradState:
    def __init__(self, N=1):
        self._state = _GradStateCore(N)
        self.edgelist = np.empty((0, 2), dtype=np.int64)

    def __getattr__(self, name):
        return getattr(self._state, name)

    def set_edgelist(self, edgelist):
        self._state.set_edgelist(edgelist)
        self.edgelist = np.array(self._state.get_edges(), dtype=np.int64)
        self.g = nx.Graph()
        self.g.add_nodes_from(range(self.N))
        self.g.add_edges_from(self.edgelist.tolist())
        n_edges = self.g.number_of_edges()
        self.k_mean = 2 * n_edges / self.N if self.N else 0.0

    def __setattr__(self, name, value):
        if name == "edgelist" and hasattr(self, "_state"):
            self._state.set_edgelist(value)
            value = np.array(self._state.get_edges(), dtype=np.int64)
        object.__setattr__(self, name, value)

    def infer_beta(self, n_sample=600, beta_bounds=(0.01, 25.0), n_grid=None,
                   kappa_max_iter=500, max_beta_iter=60, bisect_iter=40,
                   seed=42, verbose=False):
        if not hasattr(self, "g") or self.g.number_of_edges() == 0:
            raise ValueError("set_edgelist must be called before infer_beta")

        N = self.N
        k_mean = self.k_mean
        nb_points = n_sample
        beta_min_floor = beta_bounds[0]
        beta_abs_max = beta_bounds[1]
        NUM_ZERO = 1e-10
        THRESH = 1e-3

        degree_counts = collections.Counter(int(d) for _, d in self.g.degree() if d > 0)
        degrees = np.array(sorted(degree_counts.keys()), dtype=np.float64)
        counts = np.array([degree_counts[int(k)] for k in degrees], dtype=np.float64)
        n_deg = len(degrees)

        local_c = nx.clustering(self.g)
        deg_map = dict(self.g.degree())
        gt1 = [local_c[v] for v in self.g if deg_map[v] > 1]
        nb_deg_gt_one = len(gt1)
        if nb_deg_gt_one == 0:
            raise ValueError("no nodes with degree >= 2; cannot infer beta")
        target = sum(gt1) / nb_deg_gt_one

        def mercator_mu(beta):
            if beta > 1:
                return beta * np.sin(np.pi / beta) / (2.0 * np.pi * k_mean)
            if beta < 1:
                return (1 - beta) * 2.0 ** (-beta) * N ** (beta - 1) / k_mean
            return 1.0 / (2 * k_mean * np.log(N))

        def hyp2f1a(beta, z1, z2):
            if beta >= 1:
                arg = -(z1 / z2) ** beta
                return scipy.special.hyp2f1(1.0, 1.0 / beta, 1.0 + 1.0 / beta, arg)
            arg = -(np.asarray(z1, dtype=float) ** beta) / z2
            x = scipy.special.hyp2f1(1.0, 1.0 / beta, 1.0 + 1.0 / beta, arg)
            return np.where(np.isnan(x), 1.0 / (1.0 + 1.0 / z2), x)

        def invert_bisect(u, kA, kB, beta, mu):
            R = N / (2.0 * np.pi)
            z2 = mu * kA * kB
            hmax = hyp2f1a(beta, np.pi * R, z2)
            zmin = np.zeros_like(u)
            zmax = np.full_like(u, np.pi)
            for _ in range(bisect_iter):
                z = 0.5 * (zmin + zmax)
                Fz = (z / np.pi) * hyp2f1a(beta, z * R, z2) / hmax
                hi = Fz > u
                zmax = np.where(hi, z, zmax)
                zmin = np.where(hi, zmin, z)
            return 0.5 * (zmin + zmax)

        def infer_kappas(beta, mu, rng):
            kappa = degrees.copy()
            for _ in range(kappa_max_iter):
                P = hyp2f1a(beta, N / 2.0, mu * np.outer(kappa, kappa))
                exp_deg = P.dot(counts) - np.diag(P)
                if np.all(np.abs(exp_deg - degrees) <= THRESH):
                    break
                kappa = np.abs(kappa + (degrees - exp_deg) * rng.random(n_deg))
            return kappa

        def clustering_at_beta(beta, rng):
            mu = mercator_mu(beta)
            kappa = infer_kappas(beta, mu, rng)
            P = hyp2f1a(beta, N / 2.0, mu * np.outer(kappa, kappa))
            ge2 = np.where(degrees >= 2)[0]
            if ge2.size == 0:
                return 0.0
            j2_list, j3_list, i_list = [], [], []
            for i in ge2:
                w = counts * P[i, :]
                w = np.where(w > NUM_ZERO, w, 0.0)
                s = w.sum()
                prob = (w / s) if s > 0 else (counts / counts.sum())
                j2_list.append(rng.choice(n_deg, size=nb_points, p=prob))
                j3_list.append(rng.choice(n_deg, size=nb_points, p=prob))
                i_list.append(np.full(nb_points, i, dtype=np.int64))
            i_idx = np.concatenate(i_list)
            j2, j3 = np.concatenate(j2_list), np.concatenate(j3_list)
            kA, kB2, kB3 = kappa[i_idx], kappa[j2], kappa[j3]
            z12 = invert_bisect(rng.random(i_idx.size), kA, kB2, beta, mu)
            z13 = invert_bisect(rng.random(i_idx.size), kA, kB3, beta, mu)
            flip = rng.random(i_idx.size) < 0.5
            da = np.where(flip, np.abs(z12 + z13), np.abs(z12 - z13))
            da = np.minimum(da, 2.0 * np.pi - da)
            daR = N / 2.0 / np.pi * da  # da * R, R = N / (2 pi)
            if beta >= 1:
                conn = 1.0 / (1.0 + (daR / (mu * kB2 * kB3)) ** beta)
            else:
                conn = 1.0 / (1.0 + daR ** beta / (mu * kB2 * kB3))
            p23 = np.where(da < NUM_ZERO, 1.0, conn)
            ck = p23.reshape(ge2.size, nb_points).mean(axis=1)
            return float(np.sum(ck * counts[ge2]) / nb_deg_gt_one)

        rng = np.random.default_rng(seed)
        beta = 2.0 + rng.random()
        lo, hi = 0.0, -1.0  # beta_min, beta_max (-1 means upper bound not yet found)
        for _ in range(max_beta_iter):
            c = clustering_at_beta(beta, rng)
            if verbose:
                print(beta, c)
            if np.isnan(c):
                break
            if abs(c - target) < THRESH:
                break
            if c > target:
                hi = beta
                beta = (hi + lo) / 2.0
                if beta < beta_min_floor:
                    beta = beta_min_floor
                    break
            else:
                lo = beta
                beta = beta * 1.5 if hi == -1.0 else (hi + lo) / 2.0
            if beta > beta_abs_max:
                break
        self.set_beta(beta)

    def _analytic_mu(self, beta):
        N, k_mean = self.N, self.k_mean
        if beta > 1:
            return beta * np.sin(np.pi / beta) / (2.0 * np.pi * k_mean)
        if beta == 1:
            return 1.0 / (2.0 * k_mean * np.log(N))
        return (1.0 - beta) / (2.0 ** beta * k_mean * N ** (1.0 - beta))

    def infer_mu(self, numeric=True, max_iter=100, tol=5e-5):
        beta, k_mean, N = self.beta, self.k_mean, self.N
        mu = self._analytic_mu(beta)

        if numeric and beta > 0:
            degree_counts = collections.Counter(int(d) for _, d in self.g.degree() if d > 0)
            k = np.array(sorted(degree_counts.keys()), dtype=np.float64)
            Nk = np.array([degree_counts[int(v)] for v in k], dtype=np.float64)
            kk = np.outer(k, k)      # k1 * k2
            NN = np.outer(Nk, Nk)    # N_{k1} * N_{k2}

            converged = False
            for _ in range(max_iter):
                z = mu * kk
                # int1: expected average degree; int2: helper used in Newton's step.
                # hyp2f1a argument matches Mercator (beta < 1: -z1^beta/z2).
                if beta >= 1:
                    arg = -(N / 2.0 / z) ** beta
                else:
                    arg = -((N / 2.0) ** beta) / z
                P = scipy.special.hyp2f1(1.0, 1.0 / beta, 1.0 + 1.0 / beta, arg)
                if beta < 1:
                    P = np.where(np.isnan(P), 1.0 / (1.0 + 1.0 / z), P)
                int1 = float((NN * P).sum() / N)
                if beta > 1:
                    d = 1.0 / (1.0 + (N / (2.0 * z)) ** beta)
                else:
                    d = 1.0 / (1.0 + (N / 2.0) ** beta / z)
                int2 = float((NN * d).sum() / N)

                if abs(int1 - k_mean) < tol:
                    converged = True
                    break
                denom = int1 - int2
                if denom == 0:
                    break
                if beta >= 1:
                    mu -= mu * (int1 - k_mean) / denom
                else:
                    mu -= beta * mu * (int1 - k_mean) / denom
                if mu <= 0:
                    mu = 1e-10

            if not converged:
                mu = self._analytic_mu(beta)

        self.set_mu(mu)

    def init_embedding(self, method="random", **kwargs):
        """Initialize positions randomly or from Laplacian spectral embedding."""
        if method == "random":
            for i in range(self.N):
                x = np.random.rand() * self.N
                y = 16 * np.random.rand()
                z = np.random.rand() * 2 - 1
                self.set_pos(i, [x, y, z])
        elif method == "laplacian":
            beta, mu, N = self.beta, self.mu, self.N
            z1 = N / 2.0

            def hyp2f1a_arg(z2):
                if beta >= 1:
                    return -(z1 / z2) ** beta
                return -(z1 ** beta) / z2

            def hyp2f1_safe(a, b, c, z, z2):
                x = scipy.special.hyp2f1(a, b, c, z)
                if beta < 1 and (np.isnan(x) or not np.isfinite(x)):
                    return 1.0 / (1.0 + 1.0 / z2)
                return x

            degree = np.fromiter(
                (self.g.degree[i] for i in range(N)), dtype=np.float64, count=N
            )
            degree = np.maximum(degree, 1.0)

            vec_dx2 = []
            for n, m in self.edgelist:
                k1 = degree[int(n)]
                k2 = degree[int(m)]
                z2 = mu * k1 * k2
                arg = hyp2f1a_arg(z2)
                num = hyp2f1_safe(1.0, 2.0 / beta, 1.0 + 2.0 / beta, arg, z2)
                den = hyp2f1_safe(1.0, 1.0 / beta, 1.0 + 1.0 / beta, arg, z2)
                dtheta = np.pi / 2 * num / den
                dx = 2 * np.sin(dtheta / 2)
                vec_dx2.append(dx ** 2)
            t_val = np.mean(vec_dx2)
            vec_omega = np.exp(-np.asarray(vec_dx2, dtype=float) / t_val)
            W = csr_matrix(
                (vec_omega, (self.edgelist[:, 0], self.edgelist[:, 1])),
                shape=(N, N),
                dtype=float,
            )
            W = W + W.T
            # Symmetric normalized Laplacian L_sym = I - D^{-1/2} W D^{-1/2}.
            strength = np.asarray(W.sum(axis=1)).ravel()
            d_inv_sqrt = np.zeros_like(strength)
            nz = strength > 0
            d_inv_sqrt[nz] = 1.0 / np.sqrt(strength[nz])
            D_inv_sqrt = diags(d_inv_sqrt)
            ident = identity(N, format="csr", dtype=float)
            L_sym = (ident - D_inv_sqrt @ W @ D_inv_sqrt).tocsc()
            L_sym = (L_sym + L_sym.T) * 0.5
            k = min(3, N - 1)
            if k < 1:
                raise ValueError("Laplacian initialization needs N >= 2")
            try:
                vals, vecs = eigsh(L_sym, k=k, sigma=-1e-6, which="LM")
            except (ArpackNoConvergence, RuntimeError):
                w, v = np.linalg.eigh(L_sym.toarray())
                vals, vecs = w[:k], v[:, :k]
            order = np.argsort(vals)
            vecs = vecs[:, order]
            # Convert symmetric eigenvectors back to random-walk eigenvectors.
            evec = d_inv_sqrt[:, None] * vecs
            # Skip the trivial constant mode (eigenvalue 0); use the two
            # smallest non-trivial eigenvectors as the angular axes.
            if evec.shape[1] >= 3:
                ax_a, ax_b = 2, 1
            else:
                ax_a, ax_b = evec.shape[1] - 1, 0
            for i in range(N):
                x = N / (2.0 * np.pi) * np.arctan2(evec[i, ax_a], evec[i, ax_b])
                y = degree[i]
                z = np.random.rand() * 2 - 1
                self.set_pos(i, [x, y, z])
        else:
            raise ValueError(f"Unknown init_embedding method: {method}")

GradStateCore = _GradStateCore
