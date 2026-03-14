from typing import List, Tuple, Optional
import math


Point = Tuple[int, int]
Line = Tuple[Point, Point]


class IntersectionDetector:

    def __init__(self, tolerance: int = 3):
        self.tolerance = tolerance


    def process(self, floors: List[List[Line]]) -> List[List[Line]]:

        results = []

        for walls in floors:

            intersections = self._find_intersections(walls)

            split_walls = self._split_walls(walls, intersections)

            results.append(split_walls)

        return results


    def _find_intersections(self, walls: List[Line]) -> List[Point]:

        intersections = []

        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):

                p = self._line_intersection(walls[i], walls[j])

                if p is not None:
                    intersections.append(p)

        return intersections


    def _line_intersection(self, a: Line, b: Line) -> Optional[Point]:

        (x1, y1), (x2, y2) = a
        (x3, y3), (x4, y4) = b

        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)

        if denom == 0:
            return None

        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom

        if self._on_segment(px, py, a) and self._on_segment(px, py, b):
            return (int(px), int(py))

        return None


    def _on_segment(self, x: float, y: float, line: Line) -> bool:

        (x1, y1), (x2, y2) = line

        return (
            min(x1, x2) - self.tolerance <= x <= max(x1, x2) + self.tolerance
            and
            min(y1, y2) - self.tolerance <= y <= max(y1, y2) + self.tolerance
        )


    def _split_walls(self, walls: List[Line], points: List[Point]) -> List[Line]:

        new_walls = []

        for wall in walls:

            pts = [wall[0], wall[1]]

            for p in points:

                if self._point_on_line(p, wall):
                    pts.append(p)

            pts = sorted(pts, key=lambda x: (x[0], x[1]))

            for i in range(len(pts)-1):

                if pts[i] != pts[i+1]:
                    new_walls.append((pts[i], pts[i+1]))

        return new_walls


    def _point_on_line(self, p: Point, line: Line) -> bool:

        (x, y) = p
        (x1, y1), (x2, y2) = line

        cross = abs((y - y1)*(x2 - x1) - (x - x1)*(y2 - y1))

        if cross > self.tolerance:
            return False

        dot = (x-x1)*(x2-x1) + (y-y1)*(y2-y1)

        if dot < 0:
            return False

        sq_len = (x2-x1)**2 + (y2-y1)**2

        if dot > sq_len:
            return False

        return True