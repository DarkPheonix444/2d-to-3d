import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
import math

Point = Tuple[int, int]


class GraphBuilder:

    def __init__(self, snap_ratio=0.01, min_snap=3, max_snap=10, debug=True):
        self.snap_ratio = snap_ratio
        self.min_snap = min_snap
        self.max_snap = max_snap
        self.debug = debug
        self.snap_tol = None

    # ===================== MAIN =====================

    def build(self, lines_with_votes: List[Dict]) -> nx.Graph:

        if not lines_with_votes:
            return nx.Graph()

        pts = [p for d in lines_with_votes for p in d["line"]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        scale = np.hypot(max(xs) - min(xs), max(ys) - min(ys))
        self.snap_tol = min(max(self.snap_ratio * scale, self.min_snap), self.max_snap)

        if self.debug:
            print(f"[Graph] scale={scale:.2f}, snap_tol={self.snap_tol:.2f}")

        # STEP 1 — snap points
        snap_map = self._build_snap_map(pts)

        G = nx.Graph()

        for d in lines_with_votes:
            (x1, y1), (x2, y2) = d["line"]
            votes = max(int(d.get("votes", 1)), 1)

            p1 = snap_map.get((x1, y1), (x1, y1))
            p2 = snap_map.get((x2, y2), (x2, y2))

            if p1 == p2:
                continue

            # STEP 2 — enforce axis alignment (CRITICAL)
            p1, p2 = self._axis_align_safe(p1, p2)

            length = self._dist((x1, y1), (x2, y2))

            if length < self.snap_tol * 0.5:
                continue  # remove micro edges

            if G.has_edge(p1, p2):
                if votes > G[p1][p2]["votes"]:
                    G[p1][p2]["votes"] = votes
                    G[p1][p2]["length"] = length
            else:
                G.add_edge(p1, p2, votes=votes, length=length)

        # STEP 3 — merge collinear edges
        G = self._merge_collinear(G)

        if self.debug:
            self._debug_stats(G)

        return G

    # ===================== AXIS ALIGN =====================

    def _axis_align_safe(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        dx = abs(x1 - x2)
        dy = abs(y1 - y2)

        if dx < self.snap_tol:
            return (x1, y1), (x1, y2)

        if dy < self.snap_tol:
            return (x1, y1), (x2, y1)

        return p1, p2  # DO NOT FORCE

    # ===================== SNAP =====================

    def _build_snap_map(self, points):

        unique_points = sorted(set(points))
        reps = []
        snap_map = {}

        for p in unique_points:
            best = None
            best_d = float("inf")

            for r in reps:
                d = self._dist(p, r)
                if d < self.snap_tol and d < best_d:
                    best = r
                    best_d = d

            if best is None:
                reps.append(p)
                snap_map[p] = p
            else:
                snap_map[p] = best

        return snap_map

    # ===================== MERGE =====================

    def _merge_collinear(self, G):

        def angle(u, v):
            dx = v[0] - u[0]
            dy = v[1] - u[1]
            return abs(math.degrees(math.atan2(dy, dx))) % 180

        changed = True

        while changed:
            changed = False

            for node in list(G.nodes):
                nbrs = list(G.neighbors(node))

                if len(nbrs) != 2:
                    continue

                u, v = nbrs

                a1 = angle(node, u)
                a2 = angle(node, v)

                if abs(a1 - a2) < 5:
                    if not G.has_edge(u, v):
                        length = G[u][node]["length"] + G[node][v]["length"]
                        votes = max(G[u][node]["votes"], G[node][v]["votes"])
                        G.add_edge(u, v, length=length, votes=votes)

                    G.remove_node(node)
                    changed = True
                    break

        return G

    # ===================== DIST =====================

    def _dist(self, a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    # ===================== DEBUG =====================

    def _debug_stats(self, G):

        if len(G.nodes) == 0:
            print("[Graph] Empty")
            return

        deg = [G.degree(n) for n in G.nodes]

        print(f"[Graph] Nodes: {len(G.nodes)}")
        print(f"[Graph] Edges: {len(G.edges)}")
        print(f"[Graph] Avg Degree: {np.mean(deg):.2f}")
        print(f"[Graph] Dangling: {sum(1 for d in deg if d == 1)}")