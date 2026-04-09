import math
from typing import Dict, List, Tuple

import cv2
import networkx as nx

Point = Tuple[int, int]
Line = Tuple[Point, Point]


class LayoutGraphNX:

    def __init__(self, snap=10, debug=True, min_room_area=500):
        self.snap = snap
        self.debug = debug
        self.min_room_area = min_room_area

    # ===================== MAIN =====================

    def build(self, floors: List[List[Line]], images=None) -> List[Dict]:

        layouts = []

        for i, walls in enumerate(floors):

            if not walls:
                layouts.append({"graph": nx.Graph(), "rooms": []})
                continue

            dynamic_snap = self._compute_snap(walls)
            walls = [self._canonical_line(w, dynamic_snap) for w in walls]
            walls = self._deduplicate_lines(walls)

            G = self._build_graph(walls)
            G = self._prune_tiny_components(G, min_edges=3)

            cycles = self._collect_cycles(G)

            # 🔥 improved extraction
            rooms = self._extract_rooms(cycles)

            if self.debug and images is not None:
                self._show(images[i], walls, rooms)

            layouts.append({
                "graph": G,
                "rooms": rooms
            })

        return layouts

    # ===================== GRAPH =====================

    def _build_graph(self, walls: List[Line]):

        G = nx.Graph()

        graph_snap = self._compute_snap(walls)

        for (x1, y1), (x2, y2) in walls:

            p1 = self._snap_point((x1, y1), graph_snap)
            p2 = self._snap_point((x2, y2), graph_snap)

            if p1 == p2:
                continue

            G.add_edge(p1, p2)

        return G

    # ===================== 🔥 ROOM EXTRACTION =====================

    def _extract_rooms(self, cycles):

        candidates = []

        for cycle in cycles:

            if len(cycle) < 4:
                continue

            cycle = self._order_polygon(cycle)

            if not self._is_axis_aligned_cycle(cycle):
                continue

            area = self._polygon_area(cycle)

            if area < self.min_room_area:
                continue

            if self._has_self_intersection(cycle):
                continue

            if not self._passes_room_shape(cycle, area):
                continue

            score = area

            candidates.append((cycle, score, area))

        candidates.sort(key=lambda x: x[1], reverse=True)

        final_rooms = []

        for cycle, score, area in candidates:

            if any(self._is_inside(cycle, r) for r in final_rooms):
                continue

            if any(self._overlap_ratio(cycle, r) > 0.7 for r in final_rooms):
                continue

            final_rooms.append(cycle)

        return final_rooms

    # ===================== GEOMETRY =====================

    def _polygon_area(self, poly):
        area = 0
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2

    def _is_inside(self, small, big):

        cx = sum(p[0] for p in small) / len(small)
        cy = sum(p[1] for p in small) / len(small)

        return self._point_in_polygon((cx, cy), big)

    def _point_in_polygon(self, point, poly):

        x, y = point
        inside = False

        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]

            if ((y1 > y) != (y2 > y)) and \
               (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-6) + x1):
                inside = not inside

        return inside

    def _overlap_ratio(self, poly1, poly2):

        x1_min = min(p[0] for p in poly1)
        x1_max = max(p[0] for p in poly1)
        y1_min = min(p[1] for p in poly1)
        y1_max = max(p[1] for p in poly1)

        x2_min = min(p[0] for p in poly2)
        x2_max = max(p[0] for p in poly2)
        y2_min = min(p[1] for p in poly2)
        y2_max = max(p[1] for p in poly2)

        inter_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        inter_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

        inter_area = inter_w * inter_h
        area1 = (x1_max - x1_min) * (y1_max - y1_min)

        if area1 == 0:
            return 0

        return inter_area / area1

    def _snap_point(self, p: Point, snap: int) -> Point:
        if snap <= 1:
            return int(p[0]), int(p[1])
        return ((int(round(p[0] / snap)) * snap), (int(round(p[1] / snap)) * snap))

    def _canonical_line(self, line: Line, snap: int) -> Line:
        a = self._snap_point(line[0], snap)
        b = self._snap_point(line[1], snap)
        return (a, b) if a <= b else (b, a)

    def _compute_snap(self, walls: List[Line]) -> int:
        if not walls:
            return max(1, int(self.snap))

        pts = [p for line in walls for p in line]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        span = max(max(xs) - min(xs), max(ys) - min(ys), 1)
        adaptive = int(round(0.005 * span))
        adaptive = max(3, min(25, adaptive))

        if self.snap is None:
            return adaptive

        return max(1, int(self.snap))

    def _deduplicate_lines(self, walls: List[Line]) -> List[Line]:
        out = []
        seen = set()

        for a, b in walls:
            if a == b:
                continue
            key = (a, b) if a <= b else (b, a)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)

        return out

    def _collect_cycles(self, G: nx.Graph):
        cycles = []
        seen = set()

        for collector in (nx.cycle_basis, nx.minimum_cycle_basis):
            try:
                found = collector(G)
            except Exception:
                found = []

            for cycle in found:
                if len(cycle) < 3:
                    continue
                key = frozenset(cycle)
                if key in seen:
                    continue
                seen.add(key)
                cycles.append(cycle)

        return cycles

    def _prune_tiny_components(self, G: nx.Graph, min_edges=3):
        if G.number_of_edges() == 0:
            return G

        remove_nodes = []
        for comp in nx.connected_components(G):
            sg = G.subgraph(comp)
            if sg.number_of_edges() < min_edges:
                remove_nodes.extend(list(comp))

        if remove_nodes:
            G = G.copy()
            G.remove_nodes_from(remove_nodes)

        return G

    def _order_polygon(self, pts):

        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        return sorted(pts, key=lambda p: math.atan2(p[1]-cy, p[0]-cx))

    def _is_axis_aligned_cycle(self, cycle):

        for i in range(len(cycle)):
            x1, y1 = cycle[i]
            x2, y2 = cycle[(i + 1) % len(cycle)]

            if not (x1 == x2 or y1 == y2):
                return False

        return True

    def _has_self_intersection(self, poly):

        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        def intersects(a, b, c, d):
            return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

        n = len(poly)
        for i in range(n):
            a1 = poly[i]
            a2 = poly[(i + 1) % n]
            for j in range(i + 1, n):
                if abs(i - j) <= 1 or abs(i - j) == n - 1:
                    continue
                b1 = poly[j]
                b2 = poly[(j + 1) % n]
                if intersects(a1, a2, b1, b2):
                    return True
        return False

    def _passes_room_shape(self, cycle, area):
        xs = [p[0] for p in cycle]
        ys = [p[1] for p in cycle]

        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        if w <= 0 or h <= 0:
            return False

        bbox_area = float(w * h)
        fill_ratio = area / bbox_area if bbox_area > 0 else 0.0
        if fill_ratio < 0.55:
            return False

        aspect = max(w / h, h / w)
        if aspect > 15.0:
            return False

        return True

    # ===================== VISUAL =====================

    def _show(self, img, lines, rooms):

        if len(img.shape) == 2:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            vis = img.copy()

        for (x1, y1), (x2, y2) in lines:
            cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        for room in rooms:
            for i in range(len(room)):
                p1 = room[i]
                p2 = room[(i+1) % len(room)]
                cv2.line(vis, p1, p2, (0, 255, 0), 3)

        cv2.imshow("rooms_detected", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()