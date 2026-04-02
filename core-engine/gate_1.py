import networkx as nx
import numpy as np


class Gate1:

    def __init__(
        self,
        length_ratio: float = 0.03,
        vote_percentile: float = 30,
        min_vote_floor: int = 1,
        debug: bool = True,
    ):
        self.length_ratio = length_ratio
        self.vote_percentile = vote_percentile
        self.min_vote_floor = min_vote_floor
        self.debug = debug

    def apply(self, G: nx.Graph) -> nx.Graph:

        if G.number_of_edges() == 0:
            return G.copy()

        G_clean = G.copy()

        iteration = 0

        while True:
            iteration += 1

            lengths = []
            votes_list = []

            for _, _, data in G_clean.edges(data=True):
                lengths.append(data.get("length", 0.0))
                votes_list.append(data.get("votes", 1))

            # Graph may become empty after a previous iteration's removals.
            if not votes_list:
                break

            scale = np.median(lengths) if lengths else 1.0

            L = scale * self.length_ratio
            V = max(
                np.percentile(votes_list, self.vote_percentile),
                self.min_vote_floor
            )

            deg = dict(G_clean.degree())

            edges_to_remove = []

            for u, v, data in G_clean.edges(data=True):

                votes = data.get("votes", 1)
                length = data.get("length", 0.0)

                deg_u = deg[u]
                deg_v = deg[v]

                # dangling weak
                if votes <= V and min(deg_u, deg_v) == 1:
                    edges_to_remove.append((u, v))
                    continue

                # weak micro
                if (
                    votes <= V
                    and length <= L
                    and deg_u <= 2
                    and deg_v <= 2
                ):
                    edges_to_remove.append((u, v))

            if not edges_to_remove:
                break

            G_clean.remove_edges_from(edges_to_remove)
            G_clean.remove_nodes_from(list(nx.isolates(G_clean)))

        if self.debug:
            print(f"[Gate1] iterations={iteration}")
            print(f"[Gate1] remaining_edges={G_clean.number_of_edges()}")

        return G_clean