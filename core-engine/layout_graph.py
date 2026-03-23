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

            rooms = self._filter_rooms(cycles)

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

    # ===================== ROOM FILTER =====================

    def _filter_rooms(self, cycles):

        rooms = []

        for cycle in cycles:

            if len(cycle) < 4:
                continue

            cycle = self._order_polygon(cycle)

            if not self._is_axis_aligned_cycle(cycle):
                continue

            if self._is_valid_room(cycle):
                rooms.append(cycle)

        return rooms

    def _is_valid_room(self, room):

        if len(room) < 4:
            return False

        if self._has_self_intersection(room):
            return False

        # area check
        area = 0
        for i in range(len(room)):
            x1, y1 = room[i]
            x2, y2 = room[(i + 1) % len(room)]
            area += x1 * y2 - x2 * y1

        area = abs(area) / 2

        return area > 500

    # ===================== GEOMETRY =====================

    def _snap_point(self, p: Point) -> Point:
        return ((p[0] // self.snap) * self.snap,
                (p[1] // self.snap) * self.snap)

    def _canonical_line(self, line: Line) -> Line:
        return self._snap_point(line[0]), self._snap_point(line[1])

    def _order_polygon(self, pts):

        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)

        return sorted(pts, key=lambda p: math.atan2(p[1]-cy, p[0]-cx))

    def _has_self_intersection(self, poly):

        def ccw(A, B, C):
            return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        n = len(poly)

        for i in range(n):
            for j in range(i+1, n):

                if abs(i-j) <= 1 or abs(i-j) == n-1:
                    continue

                if intersect(poly[i], poly[(i+1)%n],
                             poly[j], poly[(j+1)%n]):
                    return True

        return False
    
    def _is_axis_aligned_cycle(self, cycle):

        for i in range(len(cycle)):
            x1, y1 = cycle[i]
            x2, y2 = cycle[(i + 1) % len(cycle)]

            # must be horizontal or vertical
            if not (x1 == x2 or y1 == y2):
                return False

        return True

    # ===================== VISUAL =====================

    def _show(self, img, lines, rooms):

        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # draw walls
        for (x1, y1), (x2, y2) in lines:
            cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # draw rooms
        for room in rooms:
            for i in range(len(room)):
                p1 = room[i]
                p2 = room[(i+1) % len(room)]
                cv2.line(vis, p1, p2, (0, 255, 0), 3)

        cv2.imshow("rooms_detected", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()