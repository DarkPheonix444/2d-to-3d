import networkx as nx
from collections import deque
import math
import numpy as np


class Gate2:
    """
    Gate 2: Structural Importance Propagation (NO REMOVAL)

    Purpose:
    - Assign structural importance scores to nodes and edges
    - Based on propagation from backbone (largest component)

    Output:
    - Graph with added edge attribute: "score"
    """

    def __init__(
        self,
        decay=0.85,
        angle_boost=True,
        debug=True
    ):
        self.decay = decay
        self.angle_boost = angle_boost
        self.debug = debug

    # 🔥 angle helper
    def _edge_angle(self, u, v):
        dx = v[0] - u[0]
        dy = v[1] - u[1]
        angle = abs(math.degrees(math.atan2(dy, dx)))
        return angle % 180

    # 🔥 angle alignment check (0° / 90°)
    def _is_aligned(self, angle):
        return (
            abs(angle - 0) < 10 or
            abs(angle - 90) < 10
        )

    def apply(self, G: nx.Graph) -> nx.Graph:

        if G.number_of_edges() == 0:
            return G.copy()

        G_out = G.copy()

        # 🔥 STEP 1 — find backbone (largest component)
        components = list(nx.connected_components(G_out))
        backbone = max(components, key=len)

        # 🔥 STEP 2 — initialize scores
        node_score = {n: 0.0 for n in G_out.nodes}

        # Seed only strong structural nodes (not whole backbone),
        # otherwise a connected graph gets uniform score=1 everywhere.
        backbone_sub = G_out.subgraph(backbone)
        edge_votes = [data.get("votes", 1) for _, _, data in backbone_sub.edges(data=True)]

        seeds = set()
        if edge_votes:
            vote_thr = float(np.percentile(edge_votes, 70))
            for u, v, data in backbone_sub.edges(data=True):
                votes = data.get("votes", 1)
                if votes >= vote_thr:
                    seeds.add(u)
                    seeds.add(v)

        # Fallback if vote-based seeds are sparse.
        if not seeds:
            high_deg = [n for n in backbone if G_out.degree(n) >= 3]
            if high_deg:
                seeds.update(high_deg)

        # Last-resort fallback: one central node.
        if not seeds and backbone:
            center = max(backbone, key=lambda n: G_out.degree(n))
            seeds.add(center)

        for n in seeds:
            node_score[n] = 1.0

        # 🔥 STEP 3 — BFS propagation
        visited = set(seeds)
        queue = deque(seeds)

        while queue:
            u = queue.popleft()

            for v in G_out.neighbors(u):

                if v in visited:
                    continue

                # base decay
                score = node_score[u] * self.decay

                # 🔥 optional angle boost
                if self.angle_boost:
                    angle = self._edge_angle(u, v)
                    if self._is_aligned(angle):
                        score *= 1.1  # small boost

                # assign
                node_score[v] = score

                visited.add(v)
                queue.append(v)

        # 🔥 STEP 4 — assign edge scores
        for u, v in G_out.edges():
            G_out[u][v]["score"] = (node_score[u] + node_score[v]) / 2

        if self.debug:
            scores = list(node_score.values())
            print(f"[Gate2] Seeds used: {len(seeds)}")
            print(f"[Gate2] Node score range: {min(scores):.3f} → {max(scores):.3f}")

        return G_out