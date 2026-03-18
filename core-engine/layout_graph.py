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
        node_list = list(nodes)

        if not node_list:
            return set()

        visited = [False] * len(node_list)
        clustered: Set[Point] = set()

        for i, seed in enumerate(node_list):
            if visited[i]:
                continue

            stack = [i]
            visited[i] = True
            cluster = [seed]

            while stack:
                idx = stack.pop()
                p = node_list[idx]

                for j, q in enumerate(node_list):
                    if visited[j]:
                        continue

                    if self._dist(p, q) <= self.snap_tolerance:
                        visited[j] = True
                        stack.append(j)
                        cluster.append(q)

            cx = int(round(sum(p[0] for p in cluster) / len(cluster)))
            cy = int(round(sum(p[1] for p in cluster) / len(cluster)))
            clustered.add((cx, cy))

        return clustered

    def _build_graph(
        self,
        nodes: Set[Point],
        walls: List[Line]
    ) -> Dict[Point, List[Point]]:

        graph = {n: set() for n in nodes}

        for p1, p2 in walls:

            n1 = self._nearest_node(p1, nodes)
            n2 = self._nearest_node(p2, nodes)

            if n1 == n2:
                continue

            graph[n1].add(n2)
            graph[n2].add(n1)

        return {k: list(v) for k, v in graph.items()}

    def _detect_rooms(self, graph: Dict[Point, List[Point]]) -> List[List[Point]]:
        rooms: List[List[Point]] = []
        seen_cycles: Set[Tuple[Point, ...]] = set()
        max_cycle_len = 14

        def dfs(start: Point, current: Point, path: List[Point]) -> None:
            if len(path) > max_cycle_len:
                return

            for neighbor in graph[current]:
                if neighbor == start and len(path) >= 4:
                    cycle = path[:]
                    key = self._canonical_cycle(cycle)

                    if key in seen_cycles:
                        continue

                    if not self._is_chordless_cycle(cycle, graph):
                        continue

                    if self._is_valid_room(cycle):
                        seen_cycles.add(key)
                        rooms.append(cycle)
                    continue

                if neighbor in path:
                    continue

                dfs(start, neighbor, path + [neighbor])

        for node in graph:
            dfs(node, node, [node])

        return self._remove_container_cycles(rooms)

    def _is_valid_room(self, room: List[Point]) -> bool:

        room = list(dict.fromkeys(room))

        if len(room) < 4:
            return False

        if self._has_duplicate_or_near_duplicate_vertices(room):
            return False

        if self._has_self_intersection(room):
            return False

        area = 0
        for i in range(len(room)):
            x1, y1 = room[i]
            x2, y2 = room[(i + 1) % len(room)]
            area += x1 * y2 - x2 * y1

        area = abs(area) / 2

        return area > 800

    def _canonical_cycle(self, cycle: List[Point]) -> Tuple[Point, ...]:
        seq = cycle[:]
        n = len(seq)

        variants = []
        for i in range(n):
            variants.append(tuple(seq[i:] + seq[:i]))

        rev = list(reversed(seq))
        for i in range(n):
            variants.append(tuple(rev[i:] + rev[:i]))

        return min(variants)

    def _is_chordless_cycle(self, cycle: List[Point], graph: Dict[Point, List[Point]]) -> bool:
        cycle_set = set(cycle)
        n = len(cycle)

        cycle_edges = set()
        for i in range(n):
            a = cycle[i]
            b = cycle[(i + 1) % n]
            cycle_edges.add(tuple(sorted((a, b))))

        for a in cycle:
            for b in graph[a]:
                if b not in cycle_set:
                    continue

                edge = tuple(sorted((a, b)))
                if edge in cycle_edges:
                    continue

                return False

        return True

    def _has_duplicate_or_near_duplicate_vertices(self, room: List[Point]) -> bool:
        for i in range(len(room)):
            for j in range(i + 1, len(room)):
                if self._dist(room[i], room[j]) < self.snap_tolerance:
                    return True
        return False

    def _has_self_intersection(self, polygon: List[Point]) -> bool:
        n = len(polygon)

        def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
            def orient(p: Point, q: Point, r: Point) -> int:
                val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
                if abs(val) < 1e-9:
                    return 0
                return 1 if val > 0 else 2

            def on_seg(p: Point, q: Point, r: Point) -> bool:
                return (
                    min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                    min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
                )

            o1 = orient(a1, a2, b1)
            o2 = orient(a1, a2, b2)
            o3 = orient(b1, b2, a1)
            o4 = orient(b1, b2, a2)

            if o1 != o2 and o3 != o4:
                return True

            if o1 == 0 and on_seg(a1, b1, a2):
                return True
            if o2 == 0 and on_seg(a1, b2, a2):
                return True
            if o3 == 0 and on_seg(b1, a1, b2):
                return True
            if o4 == 0 and on_seg(b1, a2, b2):
                return True

            return False

        for i in range(n):
            a1 = polygon[i]
            a2 = polygon[(i + 1) % n]

            for j in range(i + 1, n):
                if j == i:
                    continue

                if (j == (i + 1) % n) or (i == (j + 1) % n):
                    continue

                b1 = polygon[j]
                b2 = polygon[(j + 1) % n]

                if segments_intersect(a1, a2, b1, b2):
                    return True

        return False

    def _remove_container_cycles(self, rooms: List[List[Point]]) -> List[List[Point]]:
        if not rooms:
            return rooms

        def area(poly: List[Point]) -> float:
            s = 0
            for i in range(len(poly)):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % len(poly)]
                s += x1 * y2 - x2 * y1
            return abs(s) / 2

        def point_in_poly(pt: Point, poly: List[Point]) -> bool:
            x, y = pt
            inside = False
            j = len(poly) - 1

            for i in range(len(poly)):
                xi, yi = poly[i]
                xj, yj = poly[j]

                if ((yi > y) != (yj > y)):
                    x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
                    if x < x_cross:
                        inside = not inside
                j = i

            return inside

        keep = [True] * len(rooms)
        areas = [area(r) for r in rooms]

        for i in range(len(rooms)):
            for j in range(len(rooms)):
                if i == j:
                    continue

                if areas[i] <= areas[j]:
                    continue

                if all(point_in_poly(p, rooms[i]) for p in rooms[j]):
                    keep[i] = False
                    break

        return [rooms[i] for i in range(len(rooms)) if keep[i]]

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