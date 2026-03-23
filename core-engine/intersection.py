from typing import List, Tuple, Optional, Set
import cv2

Point = Tuple[int, int]
Line = Tuple[Point, Point]


class IntersectionDetector:

    def __init__(self, tolerance: int = 8, debug: bool = True):
        self.tolerance = tolerance
        self.debug = debug

    # ===================== MAIN =====================

    def process(self, floors: List[List[Line]], images=None) -> List[List[Line]]:
        results = []

        for i, walls in enumerate(floors):

            intersections = self._find_intersections(walls)
            print(f"[DEBUG] intersections found ({i}): {len(intersections)}")

            split_walls = self._split_walls(walls, intersections)

            if self.debug and images is not None:
                self._show(images[i], split_walls, f"intersection_{i}")

            results.append(split_walls)

        return results

    # ===================== INTERSECTION =====================

    def _find_intersections(self, walls: List[Line]) -> List[Point]:

        points: Set[Point] = set()

        # ---- CROSS INTERSECTIONS ----
        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                p = self._intersect(walls[i], walls[j])
                if p:
                    points.add(p)

        # ---- T-JUNCTIONS ----
        for i, wall in enumerate(walls):
            for endpoint in wall:
                for j, other in enumerate(walls):
                    if i == j:
                        continue

                    if not self._are_orthogonal(wall, other):
                        continue

                    if self._point_on_line(endpoint, other):
                        points.add(endpoint)

        return list(points)

    def _intersect(self, a: Line, b: Line) -> Optional[Point]:

        # Determine orientation
        if self._is_horizontal(a) and self._is_vertical(b):
            h, v = a, b
        elif self._is_vertical(a) and self._is_horizontal(b):
            h, v = b, a
        else:
            return None

        (hx1, hy1), (hx2, hy2) = h
        (vx1, vy1), (vx2, vy2) = v

        # Compute intersection
        x = int(round((vx1 + vx2) / 2))
        y = int(round((hy1 + hy2) / 2))

        # Check bounds
        if not (min(hx1, hx2) - self.tolerance <= x <= max(hx1, hx2) + self.tolerance):
            return None

        if not (min(vy1, vy2) - self.tolerance <= y <= max(vy1, vy2) + self.tolerance):
            return None

        return (x, y)

    # ===================== SPLITTING =====================

    def _split_walls(self, walls: List[Line], points: List[Point]) -> List[Line]:

        new_walls: List[Line] = []

        for wall in walls:

            pts: Set[Point] = set()
            pts.add(wall[0])
            pts.add(wall[1])

            # ---- collect intersection points ON this wall ----
            for p in points:
                if self._point_on_line(p, wall):

                    px, py = p

                    # SNAP TO AXIS (CRITICAL)
                    if self._is_horizontal(wall):
                        py = int((wall[0][1] + wall[1][1]) / 2)
                    else:
                        px = int((wall[0][0] + wall[1][0]) / 2)

                    pts.add((px, py))

            # ---- convert to list BEFORE sorting ----
            pts = list(pts)

            # ---- sort along line ----
            if self._is_horizontal(wall):
                pts.sort(key=lambda p: p[0])
            else:
                pts.sort(key=lambda p: p[1])

            # ---- split into segments ----
            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i + 1]

                if p1 == p2:
                    continue

                if self._length(p1, p2) < 5:
                    continue

                new_walls.append((p1, p2))

        return self._deduplicate(new_walls)

    # ===================== HELPERS =====================

    def _is_horizontal(self, l: Line):
        return abs(l[0][1] - l[1][1]) <= self.tolerance

    def _is_vertical(self, l: Line):
        return abs(l[0][0] - l[1][0]) <= self.tolerance

    def _are_orthogonal(self, a: Line, b: Line):
        return (self._is_horizontal(a) and self._is_vertical(b)) or \
               (self._is_vertical(a) and self._is_horizontal(b))

    def _point_on_line(self, p: Point, l: Line):
        (x1, y1), (x2, y2) = l
        px, py = p

        # bounding check
        if not (min(x1, x2) - self.tolerance <= px <= max(x1, x2) + self.tolerance and
                min(y1, y2) - self.tolerance <= py <= max(y1, y2) + self.tolerance):
            return False

        if self._is_horizontal(l):
            return abs(py - (y1 + y2) // 2) <= self.tolerance

        elif self._is_vertical(l):
            return abs(px - (x1 + x2) // 2) <= self.tolerance

        return False

    def _length(self, a: Point, b: Point):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

    def _deduplicate(self, walls: List[Line]) -> List[Line]:
        seen = set()
        res = []

        for a, b in walls:
            key = tuple(sorted([a, b]))
            if key not in seen:
                seen.add(key)
                res.append((a, b))

        return res

    # ===================== VISUAL =====================

    def _show(self, img, lines, title="intersection"):
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for (x1, y1), (x2, y2) in lines:
            cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow(title, vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()