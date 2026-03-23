from typing import Dict, List, Tuple

Point = Tuple[int, int]


class GapBridger:

    def __init__(self, max_dist=30, align_tol=5):
        self.max_dist = max_dist
        self.align_tol = align_tol

    # ===================== MAIN =====================

    def bridge(self, graph: Dict[Point, List[Point]]) -> Dict[Point, List[Point]]:

        nodes = list(graph.keys())
        new_edges = []

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):

                n1 = nodes[i]
                n2 = nodes[j]

                # skip if already connected
                if n2 in graph[n1]:
                    continue

                # check alignment
                if self._aligned(n1, n2):

                    dist = self._distance(n1, n2)

                    if dist < self.max_dist:
                        new_edges.append((n1, n2))

        # add edges
        for a, b in new_edges:
            graph[a].append(b)
            graph[b].append(a)

        return graph

    # ===================== HELPERS =====================

    def _aligned(self, a: Point, b: Point) -> bool:
        return (
            abs(a[0] - b[0]) < self.align_tol or  # vertical
            abs(a[1] - b[1]) < self.align_tol     # horizontal
        )

    def _distance(self, a: Point, b: Point) -> float:
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5