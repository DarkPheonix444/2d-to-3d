import math
from typing import Dict, List, Set, Tuple

Point = Tuple[int, int]
Line = Tuple[Point, Point]


class LayoutGraph:

    def __init__(self, snap_tolerance: int = 5):
        self.snap_tolerance = snap_tolerance

    def build(self, floors: List[List[Line]]) -> List[Dict]:

        if not isinstance(floors, list):
            raise ValueError("Input must be a list of floors.")

        layouts = []

        for walls in floors:

            self._validate_walls(walls)

            nodes = self._extract_nodes(walls)

            nodes = self._snap_nodes(nodes)

            graph = self._build_graph(nodes, walls)

            rooms = self._detect_rooms(graph)

            layouts.append({
                "nodes": nodes,
                "walls": walls,
                "graph": graph,
                "rooms": rooms
            })

        return layouts

    def _validate_walls(self, walls: List[Line]) -> None:

        if not isinstance(walls, list):
            raise TypeError("Walls must be a list")

        for wall in walls:

            if not isinstance(wall, tuple) or len(wall) != 2:
                raise ValueError("Invalid wall segment")

            p1, p2 = wall

            if len(p1) != 2 or len(p2) != 2:
                raise ValueError("Wall endpoints must be coordinates")

    def _extract_nodes(self, walls: List[Line]) -> Set[Point]:

        nodes: Set[Point] = set()

        for p1, p2 in walls:
            nodes.add(p1)
            nodes.add(p2)

        return nodes

    def _snap_nodes(self, nodes: Set[Point]) -> Set[Point]:
        nodes = list(nodes)
        visited = set()
        clusters = []

        for i, n1 in enumerate(nodes):
            if i in visited:
                continue

            cluster = [n1]
            visited.add(i)

            for j, n2 in enumerate(nodes):
                if j in visited:
                    continue

                if self._dist(n1, n2) <= self.snap_tolerance:
                    cluster.append(n2)
                    visited.add(j)

            # average cluster
            cx = int(sum(p[0] for p in cluster) / len(cluster))
            cy = int(sum(p[1] for p in cluster) / len(cluster))

            clusters.append((cx, cy))

        return set(clusters)
    def _build_graph(
        self,
        nodes: Set[Point],
        walls: List[Line]
    ) -> Dict[Point, List[Point]]:

        graph: Dict[Point, List[Point]] = {n: [] for n in nodes}

        for p1, p2 in walls:

            n1 = self._nearest_node(p1, nodes)
            n2 = self._nearest_node(p2, nodes)

            if n2 not in graph[n1]:
                graph[n1].append(n2)

            if n1 not in graph[n2]:
                graph[n2].append(n1)

        return graph

    def _detect_rooms(self, graph: Dict[Point, List[Point]]) -> List[List[Point]]:

        visited: Set[Point] = set()
        rooms: List[List[Point]] = []

        for node in graph:

            if node in visited:
                continue

            cycle = self._dfs_cycle(node, graph, visited)

            if cycle and len(cycle) >= 4:
                rooms.append(cycle)

        return rooms

    def _dfs_cycle(
        self,
        start: Point,
        graph: Dict[Point, List[Point]],
        visited: Set[Point]
    ):

        stack = [(start, None)]
        parent = {}

        while stack:

            node, prev = stack.pop()

            if node in visited:
                continue

            visited.add(node)
            parent[node] = prev

            for nb in graph[node]:

                if nb == prev:
                    continue

                if nb in parent:

                    cycle = [nb, node]
                    p = parent[node]

                    while p and p != nb:
                        cycle.append(p)
                        p = parent[p]

                    cycle.reverse()
                    return cycle

                stack.append((nb, node))

        return None

    def _nearest_node(self, p: Point, nodes: Set[Point]) -> Point:

        best = None
        best_dist = float("inf")

        for n in nodes:

            d = self._dist(p, n)

            if d < best_dist:
                best = n
                best_dist = d

        return best

    def _dist(self, a: Point, b: Point) -> float:

        return math.hypot(a[0] - b[0], a[1] - b[1])