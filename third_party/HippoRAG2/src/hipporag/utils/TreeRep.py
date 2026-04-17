import numpy as np
import torch
import networkx as nx


class TreeRep:
    """Class for running the TreeRep Algorithm.

    The algorithm takes in an n by n symmetric matrix
    with positive entries that should represent a metric.

    It outputs a weighted tree, such that the all pairs
    shortest path metric on the tree is an approximation
    of the input metric.

    If the input metric is 0-hyperbolic, then we get a
    tree that exactly represents the input metric.

    Parameters
    ----------
    d : numpy.ndarray or torch.Tensor
        n by n symmetric matrix whose entries correspond
        to the input metric. That is d[i,j] is the distance
        between the ith and the jth data point. Required.

    theta : n x p tensor. p is the dimension of embeddings.

    tol : float
        If any distances in the output tree are smaller than
        tol then we round the distances to 0. Optional, default=1e-5
    """

    def __init__(self, d, theta=None, tol=1e-5):
        if isinstance(d, np.ndarray):
            self.d = torch.tensor(d)
        else:
            self.d = d
        self.n = self.d.shape[0]
        self.S = int(1.3 * self.n)
        self.tol = float(tol)
        self.nextroots = list(range(self.n, 2 * self.n))
        self.nextroots.reverse()
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n))
        self.debug = False

        if theta is not None:
            self.theta = torch.cat((theta, torch.zeros_like(theta)), dim=0)
        else:
            self.theta = torch.zeros(2 * self.n, 1)

    # --------- SAFE GRAPH UTILS ---------
    def _safe_remove_edge(self, u, v):
        """Remove edge (u,v) only if it exists and u != v."""
        if u == v:
            return
        if self.G.has_edge(u, v):
            self.G.remove_edge(u, v)

    # --------- GEOMETRIC UTIL ---------
    def gid(self, D, w, x, y):
        """Gromov product of x and y with respect to base w for metric D."""
        return 0.5 * (D[w, x] + D[w, y] - D[x, y])

    # --------- CORE OPS ---------
    def contract_ra(self, r, a, b, c, V):
        """Try to contract edge (r, a). Return (replaced_root: bool, new_center)."""
        if self.W[r, a].abs() < self.tol:
            for v in V:
                self.W[r, v] = 0
                self.W[v, r] = 0

            # SAFE removals
            self._safe_remove_edge(a, r)
            self._safe_remove_edge(b, r)
            self._safe_remove_edge(c, r)

            if self.G.has_node(r):
                self.G.remove_node(r)
            self.nextroots.append(r)
            self.theta[r, :] = 0

            self.G.add_edge(a, b)
            self.G.add_edge(a, c)

            if self.debug:
                print("Contracting")
                print(r, a, b, c)
                print()

            return True, a
        return False, r

    def sort_into_zones(self, V, r, x, y, z, replaced_root=False):
        n = self.n
        X1, X2, Y1, Y2, Z1, Z2, R1 = [], [], [], [], [], [], []

        for w in V:
            a = self.gid(self.W, w, x, y)
            b = self.gid(self.W, w, y, z)
            c = self.gid(self.W, w, x, z)

            if np.abs(a - b) < self.tol and np.abs(b - c) < self.tol and np.abs(c - a) < self.tol:
                if a < self.tol and b < self.tol and c < self.tol and not replaced_root:
                    replaced_root = True

                    # move row/col from r to w
                    self.W[w, n:] = self.W[r, n:]
                    self.W[n:, w] = self.W[n:, r]
                    self.W[r, n:] = 0
                    self.W[n:, r] = 0

                    # SAFE removals
                    self._safe_remove_edge(x, r)
                    self._safe_remove_edge(y, r)
                    self._safe_remove_edge(z, r)

                    if self.G.has_node(r):
                        self.G.remove_node(r)
                    self.theta[r, :] = 0

                    self.nextroots.append(r)
                    r = w

                    self.G.add_edge(x, r)
                    self.G.add_edge(y, r)
                    self.G.add_edge(z, r)
                else:
                    R1.append(w)
                    val = (a + b + c) / 3
                    self.W[w, r] = val
                    self.W[r, w] = val
            elif np.abs(a - np.max([a, b, c])) < 1e-10:
                if np.abs(self.W[w, z] - b) < self.tol or np.abs(self.W[w, z] - c) < self.tol:
                    Z1.append(w)
                else:
                    Z2.append(w)
                self.W[w, r] = a
                self.W[r, w] = a
            elif np.abs(b - np.max([a, b, c])) < 1e-10:
                if np.abs(self.W[w, x] - a) < self.tol or np.abs(self.W[w, x] - c) < self.tol:
                    X1.append(w)
                else:
                    X2.append(w)
                self.W[w, r] = b
                self.W[r, w] = b
            elif np.abs(c - np.max([a, b, c])) < 1e-10:
                if np.abs(self.W[w, y] - b) < self.tol or np.abs(self.W[w, y] - a) < self.tol:
                    Y1.append(w)
                else:
                    Y2.append(w)
                self.W[w, r] = c
                self.W[r, w] = c
            else:
                if self.debug:
                    print("[warn] unexpected a,b,c:", a, b, c)

        Zones = [
            (R1, 1, r, r),
            (X1, 1, x, x), (X2, 2, x, r),
            (Y1, 1, y, y), (Y2, 2, y, r),
            (Z1, 1, z, z), (Z2, 2, z, r),
        ]
        return Zones

    def add_steiner_node(self, x, y, z):
        r = self.nextroots.pop(-1)
        self.theta[r, :] = (self.theta[x, :] + self.theta[y, :] + self.theta[z, :]) / 3

        # expand W if needed
        if r >= self.S:
            new_s = int(1.3 * self.S) + 1
            new_w = torch.zeros(new_s, new_s, dtype=self.W.dtype, device=self.W.device)
            new_w[:self.S, :self.S] = self.W
            self.S = new_s
            self.W = new_w

        self.G.add_node(r)
        self.G.add_edge(x, r)
        self.G.add_edge(y, r)
        self.G.add_edge(z, r)

        if self.debug:
            print("standard steiner")
            print(r, x, y, z)
            print()

        # set W symmetrically
        vx = self.gid(self.W, x, y, z)
        self.W[r, x] = vx
        self.W[x, r] = vx
        replaced_root, r = self.contract_ra(r, x, y, z, [x])

        vy = self.gid(self.W, y, x, z)
        self.W[r, y] = vy
        self.W[y, r] = vy
        if not replaced_root:
            replaced_root, r = self.contract_ra(r, y, x, z, [x, y])

        vz = self.gid(self.W, z, x, y)
        self.W[r, z] = vz
        self.W[z, r] = vz
        if not replaced_root:
            replaced_root, r = self.contract_ra(r, z, x, y, [x, y, z])

        if not replaced_root:
            Z = (-self.W[r, x]).exp() + (-self.W[r, y]).exp() + (-self.W[r, z]).exp()
            self.theta[r, :] = (
                (-self.W[r, x]).exp() * self.theta[x, :]
                + (-self.W[r, y]).exp() * self.theta[y, :]
                + (-self.W[r, z]).exp() * self.theta[z, :]
            ) / Z

        return replaced_root, r

    def zone1_helper(self, V, x):
        if len(V) == 0:
            return []

        if len(V) == 1:
            self.G.add_edge(x, V[0])
            if self.debug:
                print("Zone 1")
                print(x, V[0])
                print()
            return []

        p = torch.randperm(len(V))
        y = V[p[0]]
        z = V[p[1]]

        V_rem = [V[p[i]] for i in range(2, len(V))]

        replaced_root, r = self.add_steiner_node(x, y, z)
        Zones = self.sort_into_zones(V_rem, r, x, y, z, replaced_root)
        return Zones

    def zone2_helper(self, V, x, y):
        if len(V) == 0:
            return []

        idx = self.W[y, V].argmin().item()
        p = torch.arange(0, len(V))
        p[0] = idx
        p[idx] = 0

        z = V[p[0]]
        V_rem = [V[p[i]] for i in range(1, len(V))]

        # SAFE removal instead of direct remove_edge
        self._safe_remove_edge(x, y)

        replaced_root, r = self.add_steiner_node(x, y, z)
        Zones = self.sort_into_zones(V_rem, r, x, y, z, replaced_root)
        return Zones

    # --------- OPTIONAL: robust distance matrix from built tree ---------
    def to_distance_matrix(self, n=None):
        """Shortest-path distance matrix from current tree (weighted).
        - finite enforcement
        - symmetrization
        - zero diagonal
        - non-negative clamp
        """
        if n is None:
            n = self.n

        dist = dict(nx.all_pairs_dijkstra_path_length(self.G, weight="weight"))

        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dij = dist.get(i, {}).get(j, np.inf)
                D[i, j] = dij

        finite = np.isfinite(D)
        if finite.any():
            max_finite = D[finite].max()
            D = np.where(finite, D, max_finite)
        else:
            D[:] = 0.0

        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
        D = np.maximum(D, 0.0)
        return D

    # --------- MAIN ENTRY ---------
    def learn_tree(self):
        # sanitize: remove any self-loops before we start
        self.G.remove_edges_from(nx.selfloop_edges(self.G))

        # Create the weight matrix for the output tree (with room for steiner nodes)
        self.W = torch.zeros(self.S, self.S, dtype=self.d.dtype, device=self.d.device)
        self.W[:self.n, :self.n] = self.d

        # Pick the initial 3 points to start
        p = torch.randperm(self.n)
        x = p[0].item()
        y = p[1].item()
        z = p[2].item()

        V = [p[i].item() for i in range(3, len(p))]

        replaced_root, r = self.add_steiner_node(x, y, z)
        Zones = self.sort_into_zones(V, r, x, y, z, replaced_root)

        while len(Zones) > 0:
            V, zt, a, b = Zones.pop(0)
            if zt == 1:
                new_zones = self.zone1_helper(V, a)
                Zones.extend(new_zones)
            else:
                new_zones = self.zone2_helper(V, a, b)
                Zones.extend(new_zones)

        # write back edge weights from W with safety (non-negative & finite)
        for u, v in list(self.G.edges()):
            w = self.W[u, v]
            w = float(w.item()) if hasattr(w, "item") else float(w)
            if not np.isfinite(w):
                w = 0.0
            if w < self.tol:
                w = self.tol
            self.G[u][v]["weight"] = w
