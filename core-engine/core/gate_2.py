import networkx as nx
import numpy as np
import math


class Gate2:
    """
    Gate 2: Multi-Signal Structural Scoring (NO REMOVAL)

    Signals:
    - Graph (edge betweenness centrality)
    - Local (length + node degree)
    - Geometric (orthogonality)

    Output:
    - edge["score"]
    - edge["label"] ∈ {"strong", "weak", "uncertain"}
    """

    def __init__(
        self,
        w_graph=0.4,
        w_local=0.3,
        w_ortho=0.3,
        angle_tol=10,
        debug=True
    ):
        self.w_graph = w_graph
        self.w_local = w_local
        self.w_ortho = w_ortho
        self.angle_tol = angle_tol
        self.debug = debug

    # -------------------------
    # Angle computation
    # -------------------------
    def _edge_angle(self, u, v):
        dx = v[0] - u[0]
        dy = v[1] - u[1]
        angle = abs(math.degrees(math.atan2(dy, dx))) % 180
        return angle

    def _ortho_score(self, angle):
        if abs(angle - 0) < self.angle_tol or abs(angle - 90) < self.angle_tol:
            return 1.0
        return 0.0

    # -------------------------
    # Main
    # -------------------------
    def apply(self, G: nx.Graph) -> nx.Graph:

        if G.number_of_edges() == 0:
            return G.copy()

        G_out = G.copy()

        # =========================
        # STEP 1 — Graph signal (betweenness)
        # =========================
        betweenness = nx.edge_betweenness_centrality(G_out)

        max_b = max(betweenness.values()) if betweenness else 1.0

        # =========================
        # STEP 2 — Length normalization
        # =========================
        lengths = [data.get("length", 0.0) for _, _, data in G_out.edges(data=True)]
        max_length = max(lengths) if lengths else 1.0

        # =========================
        # STEP 3 — Scoring
        # =========================
        strong = weak = uncertain = 0

        for u, v, data in G_out.edges(data=True):

            edge_key = (u, v) if (u, v) in betweenness else (v, u)

            # ---- Graph score ----
            b = betweenness.get(edge_key, 0.0)
            graph_score = b / max_b if max_b > 0 else 0.0

            # ---- Local score ----
            length = data.get("length", 0.0)
            length_score = length / max_length if max_length > 0 else 0.0

            deg_u = G_out.degree[u]
            deg_v = G_out.degree[v]

            degree_score = min((deg_u + deg_v) / 4.0, 1.0)

            local_score = 0.6 * length_score + 0.4 * degree_score

            # ---- Orthogonal score ----
            angle = self._edge_angle(u, v)
            ortho_score = self._ortho_score(angle)

            # ---- Final score ----
            score = (
                self.w_graph * graph_score +
                self.w_local * local_score +
                self.w_ortho * ortho_score
            )

            # =========================
            # Labeling
            # =========================
            if score > 0.65:
                label = "strong"
                strong += 1
            elif score < 0.25:
                label = "weak"
                weak += 1
            else:
                label = "uncertain"
                uncertain += 1

            # store attributes
            G_out[u][v]["score"] = score
            G_out[u][v]["label"] = label
            G_out[u][v]["graph_score"] = graph_score
            G_out[u][v]["local_score"] = local_score
            G_out[u][v]["ortho_score"] = ortho_score

        # =========================
        # Debug
        # =========================
        if self.debug:
            print(f"[Gate2] Total edges: {G_out.number_of_edges()}")
            print(f"[Gate2] Strong: {strong}")
            print(f"[Gate2] Weak: {weak}")
            print(f"[Gate2] Uncertain: {uncertain}")

        return G_out