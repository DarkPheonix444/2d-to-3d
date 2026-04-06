import math
from typing import List, Tuple, Dict
import networkx as nx
import cv2

Point = Tuple[int, int]
Line = Tuple[Point, Point]


class LayoutGraphNX:

    def __init__(self, snap=10, debug=True):
        self.snap = snap
        self.debug = debug

    # ===================== MAIN =====================

    def build(self, floors: List[List[Line]], images=None) -> List[Dict]:

        layouts = []

        for i, walls in enumerate(floors):

            walls = [self._canonical_line(w) for w in walls]

            G = self._build_graph(walls)

            cycles = nx.cycle_basis(G)

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

        for (x1, y1), (x2, y2) in walls:

            p1 = self._snap_point((x1, y1))
            p2 = self._snap_point((x2, y2))

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

            # 🔥 LOWER threshold → allow imperfect large structures
            if area < 500:
                continue

            # 🔥 SOFT SCORE (favor bigger structures)
            score = area

            candidates.append((cycle, score, area))

        # ---------- SORT BY SCORE (DOMINANCE) ----------
        candidates.sort(key=lambda x: x[1], reverse=True)

        final_rooms = []

        for cycle, score, area in candidates:

            # 🔥 containment suppression
            if any(self._is_inside(cycle, r) for r in final_rooms):
                continue

            # 🔥 avoid near-duplicate overlapping cycles
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

    # 🔥 NEW: overlap check
    def _overlap_ratio(self, poly1, poly2):

        # approximate using bounding boxes
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

    def _snap_point(self, p: Point) -> Point:
        return ((p[0] // self.snap) * self.snap,
                (p[1] // self.snap) * self.snap)

    def _canonical_line(self, line: Line) -> Line:
        return self._snap_point(line[0]), self._snap_point(line[1])

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

    # ===================== VISUAL =====================

    def _show(self, img, lines, rooms):

        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

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