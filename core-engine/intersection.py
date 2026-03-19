from typing import List, Tuple, Optional, Set

Point = Tuple[int, int]
Line = Tuple[Point, Point]


class IntersectionDetector:

    def __init__(self, tolerance: int = 4, snap_grid: int = 10):
        self.tolerance = tolerance
        self.snap_grid = snap_grid

    # ===================== MAIN =====================

    def process(self, floors: List[List[Line]]) -> List[List[Line]]:
        results = []

        for walls in floors:
            intersections = self._find_intersections(walls)
                      
            split_walls = self._split_walls(walls, intersections)

            results.append(split_walls)

        return results

    # ===================== CORE =====================

    def _find_intersections(self, walls: List[Line]) -> List[Point]:

        points: Set[Point] = set()

        # CROSS INTERSECTIONS
        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                p = self._line_intersection(walls[i], walls[j])
                if p:
                    points.add(p)

        # T-JUNCTIONS (single clean pass)
        for i, wall in enumerate(walls):
            for endpoint in [wall[0], wall[1]]:
                p = self._snap_point(endpoint)

                for j, other in enumerate(walls):
                    if i == j:
                        continue

                    if not self._are_orthogonal(wall, other):
                        continue

                    if self._point_on_line(p, other) and not self._is_near_any_endpoint(p, other):
                        points.add(self._snap_point(p))

        return list(points)

    def _line_intersection(self, a: Line, b: Line) -> Optional[Point]:

        if self._is_horizontal(a) and self._is_vertical(b):
            h, v = a, b
        elif self._is_vertical(a) and self._is_horizontal(b):
            h, v = b, a
        else:
            return None

        (hx1, hy1), (hx2, hy2) = h
        (vx1, vy1), (vx2, vy2) = v

        h_y = hy1
        v_x = vx1

        if not self._on_segment(v_x, h_y, h):
            return None
        if not self._on_segment(v_x, h_y, v):
            return None

        return self._snap_point((v_x, h_y))

    # ===================== SPLIT =====================

    def _split_walls(self, walls: List[Line], points: List[Point]) -> List[Line]:

        new_walls: List[Line] = []

        for wall in walls:

            pts: Set[Point] = set()
            pts.add(wall[0])
            pts.add(wall[1])

            (x1, y1), (x2, y2) = wall

            for px, py in points:

                if self._is_horizontal(wall):
                    if abs(py - y1) <= self.tolerance and min(x1, x2) <= px <= max(x1, x2):
                        pts.add(self._snap_point((px, y1)))

                elif self._is_vertical(wall):
                    if abs(px - x1) <= self.tolerance and min(y1, y2) <= py <= max(y1, y2):
                        pts.add(self._snap_point((x1, py)))

            pts = list(pts)
            if self._is_horizontal(wall):
                pts.sort(key=lambda p: p[0])  # sort by x
            else:
                pts.sort(key=lambda p: p[1])  # sort by y
            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i + 1]

                if p1 == p2:
                    continue

                if self._length(p1, p2) < self.snap_grid*2:
                    continue

                new_walls.append((p1, p2))

        cleaned = self._deduplicate(new_walls)
        return self._merge_collinear(cleaned)

    # ===================== HELPERS =====================

    def _snap_point(self, p: Point) -> Point:
        g = self.snap_grid
        return (round(p[0] / g) * g, round(p[1] / g) * g)

    def _length(self, a: Point, b: Point) -> float:
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

    def _is_horizontal(self, l: Line):
        return l[0][1] == l[1][1]

    def _is_vertical(self, l: Line):
        return l[0][0] == l[1][0]

    def _on_segment(self, x: int, y: int, l: Line):
        (x1, y1), (x2, y2) = l
        return min(x1, x2) - self.tolerance <= x <= max(x1, x2) + self.tolerance and \
               min(y1, y2) - self.tolerance <= y <= max(y1, y2) + self.tolerance

    def _point_on_line(self, p: Point, l: Line):
        if self._is_horizontal(l):
            return abs(p[1] - l[0][1]) <= self.tolerance and self._on_segment(p[0], p[1], l)
        if self._is_vertical(l):
            return abs(p[0] - l[0][0]) <= self.tolerance and self._on_segment(p[0], p[1], l)
        return False

    def _are_orthogonal(self, a: Line, b: Line):
        return (self._is_horizontal(a) and self._is_vertical(b)) or \
               (self._is_vertical(a) and self._is_horizontal(b))

    def _is_near_any_endpoint(self, p: Point, l: Line):
        return self._length(p, l[0]) <= self.tolerance or self._length(p, l[1]) <= self.tolerance

    def _deduplicate(self, walls: List[Line]) -> List[Line]:
        seen = set()
        res = []
        for a, b in walls:
            key = tuple(sorted([a, b]))
            if key not in seen:
                seen.add(key)
                res.append((a, b))
        return res

    def _merge_collinear(self, walls: List[Line]) -> List[Line]:
        from collections import defaultdict

        h = defaultdict(list)
        v = defaultdict(list)

        for (x1,y1),(x2,y2) in walls:
            if y1 == y2:
                h[y1].append((min(x1,x2), max(x1,x2)))
            else:
                v[x1].append((min(y1,y2), max(y1,y2)))

        merged = []

        for y, segs in h.items():
            segs.sort()
            s,e = segs[0]
            for ns,ne in segs[1:]:
                gap = ns - e
                if ns <= e + self.snap_grid//2:
                    e = max(e, ne)
                else:
                    merged.append(((s,y),(e,y)))
                    s,e = ns,ne
            merged.append(((s,y),(e,y)))

        for x, segs in v.items():
            segs.sort()
            s,e = segs[0]
            for ns,ne in segs[1:]:
                gap = ns - e
                if ns <= e + self.snap_grid//2:
                    e = max(e, ne)
                else:
                    merged.append(((x,s),(x,e)))
                    s,e = ns,ne
            merged.append(((x,s),(x,e)))

        return merged