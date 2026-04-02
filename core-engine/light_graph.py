import numpy as np
import networkx as nx
from typing import List, Dict, Tuple

Point = Tuple[int, int]


class GraphBuilder:

    def __init__(self, snap_ratio=0.01, min_snap=3, max_snap=10, debug=True):
        """
        snap_ratio: relative snapping tolerance (scale-aware)
        min_snap: safety floor in pixels
        max_snap: safety ceiling in pixels (prevents over-merge on large plans)
        """
        self.snap_ratio = snap_ratio
        self.min_snap = min_snap
        self.max_snap = max_snap
        self.debug = debug
        self.snap_tol = None

    # ===================== MAIN =====================

    def build(self, lines_with_votes: List[Dict]) -> nx.Graph:

        if not lines_with_votes:
            return nx.Graph()

        # ---- SCALE (global, consistent) ----
        pts = [
            p
            for d in lines_with_votes
            if "line" in d and len(d["line"]) == 2
            for p in d["line"]
        ]

        if not pts:
            return nx.Graph()

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        scale = np.hypot(max(xs) - min(xs), max(ys) - min(ys))
        self.snap_tol = min(
            max(self.snap_ratio * scale, self.min_snap),
            self.max_snap
        )

        if self.debug:
            print(f"[Graph] scale={scale:.2f}, snap_tol={self.snap_tol:.2f}")

        # Build snap map once from raw endpoints to avoid order-dependent graph growth.
        snap_map = self._build_snap_map(pts)

        G = nx.Graph()
        skipped_degenerate = 0
        merged_duplicates = 0

        # ===================== BUILD =====================
        for d in lines_with_votes:

            if "line" not in d or len(d["line"]) != 2:
                continue

            (x1, y1), (x2, y2) = d["line"]
            votes = max(int(d.get("votes", 1)), 1)

            # ---- SNAP ENDPOINTS (deterministic) ----
            p1 = snap_map.get((x1, y1), (x1, y1))
            p2 = snap_map.get((x2, y2), (x2, y2))

            # ---- avoid degenerate edges ----
            if p1 == p2:
                skipped_degenerate += 1
                continue

            # Keep geometric length from original segment for gate thresholds.
            length = self._dist((x1, y1), (x2, y2))

            # ---- merge duplicates (strongest wins) ----
            if G.has_edge(p1, p2):
                merged_duplicates += 1
                curr_votes = G[p1][p2]["votes"]
                curr_len = G[p1][p2]["length"]

                if votes > curr_votes or (votes == curr_votes and length > curr_len):
                    G[p1][p2]["votes"] = votes
                    G[p1][p2]["length"] = length
            else:
                G.add_edge(p1, p2, votes=votes, length=length)

        if self.debug:
            print(f"[Graph] degenerate_edges_skipped={skipped_degenerate}")
            print(f"[Graph] duplicate_edges_merged={merged_duplicates}")
            self._debug_stats(G)

        return G

    # ===================== SNAP =====================

    def _build_snap_map(self, points: List[Point]) -> Dict[Point, Point]:
        """
        Build deterministic snapping map from raw endpoints.
        Uses sorted unique points so output does not depend on input edge order.
        """
        unique_points = sorted(set(points), key=lambda p: (p[0], p[1]))
        representatives: List[Point] = []
        snap_map: Dict[Point, Point] = {}

        for p in unique_points:
            nearest = None
            best_dist = float("inf")

            for rep in representatives:
                d = self._dist(p, rep)
                if d <= self.snap_tol and d < best_dist:
                    nearest = rep
                    best_dist = d

            if nearest is None:
                representatives.append(p)
                snap_map[p] = p
            else:
                snap_map[p] = nearest

        return snap_map

    def _snap(self, p: Point, G: nx.Graph) -> Point:
        """
        Snap point to nearest existing node within tolerance.
        """
        nearest = None
        best_dist = float("inf")

        for node in G.nodes:
            d = self._dist(p, node)
            if d < self.snap_tol and d < best_dist:
                nearest = node
                best_dist = d

        return nearest if nearest is not None else p

    # ===================== DIST =====================

    def _dist(self, a: Point, b: Point) -> float:
        return np.hypot(a[0] - b[0], a[1] - b[1])

    # ===================== DEBUG =====================

    def _debug_stats(self, G: nx.Graph):

        if len(G.nodes) == 0:
            print("[Graph] Empty graph")
            return

        degrees = [G.degree(n) for n in G.nodes]

        avg_deg = sum(degrees) / len(degrees)
        dangling = sum(1 for d in degrees if d == 1)

        print(f"[Graph] Nodes: {len(G.nodes)}")
        print(f"[Graph] Edges: {len(G.edges)}")
        print(f"[Graph] Avg Degree: {avg_deg:.2f}")
        print(f"[Graph] Dangling nodes: {dangling} ({dangling/len(G.nodes):.2%})")