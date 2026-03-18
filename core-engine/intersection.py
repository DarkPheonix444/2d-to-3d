from typing import List, Tuple, Optional, Set

Point = Tuple[int, int]
Line = Tuple[Point, Point]


class IntersectionDetector:

    def __init__(self, tolerance: int = 5, snap_grid: int = 5):
        self.tolerance = tolerance
        self.snap_grid = snap_grid

    # ===================== MAIN =====================

    def process(self, floors: List[List[Line]]) -> List[List[Line]]:
        results = []

        for walls in floors:

            # STEP 1: SNAP ALL INPUT WALLS
            walls = [self._snap_line(w) for w in walls]

            intersections = self._find_intersections(walls)
            print(f"[IntersectionDetector] total intersections found: {len(intersections)}")

            split_walls = self._split_walls(walls, intersections)
            print(f"[IntersectionDetector] total split segments: {len(split_walls)}")

            results.append(split_walls)

        return results

    # ===================== CORE =====================

    def _find_intersections(self, walls: List[Line]) -> List[Point]:

        cross_points: Set[Point] = set()

        # 1. NORMAL CROSS INTERSECTIONS (+)
        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                p = self._line_intersection(walls[i], walls[j])
                if p:
                    cross_points.add(self._snap_point(p))

        # 2. T-JUNCTIONS (endpoint of one wall touches body of another)
        t_points = self._detect_t_junctions(walls)

        # 3. MERGE
        all_points = cross_points.union(t_points)

        return list(all_points)

    def _detect_t_junctions(self, walls: List[Line]) -> Set[Point]:
        junctions: Set[Point] = set()

        for i, wall in enumerate(walls):
            for endpoint in [self._snap_point(wall[0]), self._snap_point(wall[1])]:
                for j, other in enumerate(walls):
                    if i == j:
                        continue

                    # T-junctions must be between orthogonal walls.
                    if not self._are_orthogonal(wall, other):
                        continue

                    # Endpoint must lie on the interior/body of the other wall.
                    if self._point_on_line(endpoint, other) and not self._is_near_any_endpoint(endpoint, other):
                        junctions.add(self._snap_point(endpoint))

        # Extra tolerance-safe pass: detect near-miss endpoint/body cases that may
        # appear after wall snapping and preserve true junctions.
        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                a = walls[i]
                b = walls[j]

                if not self._are_orthogonal(a, b):
                    continue

                for p in [self._snap_point(a[0]), self._snap_point(a[1])]:
                    if self._point_on_line(p, b) and not self._is_near_any_endpoint(p, b):
                        junctions.add(self._snap_point(p))

                for p in [self._snap_point(b[0]), self._snap_point(b[1])]:
                    if self._point_on_line(p, a) and not self._is_near_any_endpoint(p, a):
                        junctions.add(self._snap_point(p))

        return junctions

    def _line_intersection(self, a: Line, b: Line) -> Optional[Point]:

        a_h = self._is_horizontal(a)
        a_v = self._is_vertical(a)
        b_h = self._is_horizontal(b)
        b_v = self._is_vertical(b)

        if a_h and b_v:
            h, v = a, b
        elif a_v and b_h:
            h, v = b, a
        else:
            return None

        (hx1, hy1), (hx2, hy2) = h
        (vx1, vy1), (vx2, vy2) = v

        h_y = int(round((hy1 + hy2) / 2))
        v_x = int(round((vx1 + vx2) / 2))

        # STRICT VALIDATION
        if not self._on_segment(v_x, h_y, h):
            return None
        if not self._on_segment(v_x, h_y, v):
            return None

        return (v_x, h_y)

    # ===================== SPLIT =====================

    def _split_walls(self, walls: List[Line], points: List[Point]) -> List[Line]:

        new_walls: List[Line] = []

        for wall in walls:

            if not (self._is_horizontal(wall) or self._is_vertical(wall)):
                continue

            pts: Set[Point] = set()
            p0 = self._project_point_to_wall(self._snap_point(wall[0]), wall)
            p1 = self._project_point_to_wall(self._snap_point(wall[1]), wall)
            pts.add(p0)
            pts.add(p1)

            for p in points:
                if self._point_on_line(p, wall):
                    pts.add(self._project_point_to_wall(self._snap_point(p), wall))

            pts = list(pts)

            if self._is_horizontal(wall):
                pts.sort(key=lambda x: x[0])
            else:
                pts.sort(key=lambda x: x[1])

            for i in range(len(pts) - 1):
                p1 = pts[i]
                p2 = pts[i + 1]

                if p1 == p2:
                    continue

                # REMOVE MICRO SEGMENTS
                if self._length(p1, p2) < 5:
                    continue

                # Keep segment axis-aligned to preserve layout graph consistency.
                p1, p2 = self._axis_aligned_segment(p1, p2, wall)

                new_walls.append((p1, p2))

        return self._deduplicate(new_walls)

    # ===================== HELPERS =====================

    def _snap_point(self, p: Point) -> Point:
        x, y = p
        g = self.snap_grid
        return (round(x / g) * g, round(y / g) * g)

    def _snap_line(self, line: Line) -> Line:
        return (self._snap_point(line[0]), self._snap_point(line[1]))

    def _length(self, a: Point, b: Point) -> float:
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

    def _is_horizontal(self, line: Line) -> bool:
        (x1, y1), (x2, y2) = line
        return abs(y1 - y2) <= self.tolerance

    def _is_vertical(self, line: Line) -> bool:
        (x1, y1), (x2, y2) = line
        return abs(x1 - x2) <= self.tolerance

    def _on_segment(self, x: int, y: int, line: Line) -> bool:
        (x1, y1), (x2, y2) = line
        return (
            min(x1, x2) - self.tolerance <= x <= max(x1, x2) + self.tolerance and
            min(y1, y2) - self.tolerance <= y <= max(y1, y2) + self.tolerance
        )

    def _point_on_line(self, p: Point, line: Line) -> bool:
        x, y = p
        (x1, y1), (x2, y2) = line

        if self._is_horizontal(line):
            return abs(y - y1) <= self.tolerance and self._on_segment(x, y, line)

        if self._is_vertical(line):
            return abs(x - x1) <= self.tolerance and self._on_segment(x, y, line)

        return False

    def _are_orthogonal(self, a: Line, b: Line) -> bool:
        return (self._is_horizontal(a) and self._is_vertical(b)) or (
            self._is_vertical(a) and self._is_horizontal(b)
        )

    def _is_near_any_endpoint(self, p: Point, line: Line) -> bool:
        return self._length(p, line[0]) <= self.tolerance or self._length(p, line[1]) <= self.tolerance

    def _project_point_to_wall(self, p: Point, wall: Line) -> Point:
        x, y = p
        (x1, y1), (x2, y2) = wall

        if self._is_horizontal(wall):
            y_axis = int(round((y1 + y2) / 2))
            return (x, y_axis)

        if self._is_vertical(wall):
            x_axis = int(round((x1 + x2) / 2))
            return (x_axis, y)

        return p

    def _axis_aligned_segment(self, a: Point, b: Point, wall: Line) -> Line:
        if self._is_horizontal(wall):
            y_axis = int(round((wall[0][1] + wall[1][1]) / 2))
            x1, x2 = sorted([a[0], b[0]])
            return (x1, y_axis), (x2, y_axis)

        if self._is_vertical(wall):
            x_axis = int(round((wall[0][0] + wall[1][0]) / 2))
            y1, y2 = sorted([a[1], b[1]])
            return (x_axis, y1), (x_axis, y2)

        return a, b

    def _deduplicate(self, walls: List[Line]) -> List[Line]:
        seen = set()
        result = []

        for w in walls:
            a, b = sorted(w)
            key = (a, b)
            if key not in seen:
                seen.add(key)
                result.append((a, b))

        return result