import math
from typing import List, Tuple, Dict, Set

Point = Tuple[int, int]
Line = Tuple[Point, Point]


class ParallelWallMerger:

    def __init__(
        self,
        distance_tol: float = 15.0,
        angle_tol: float = 5.0,
        overlap_ratio: float = 0.3
    ):
        self.distance_tol = distance_tol
        self.angle_tol = math.radians(angle_tol)
        self.overlap_ratio = overlap_ratio

    def merge(self, floors: List[List[Line]]) -> List[List[Dict]]:

        results: List[List[Dict]] = []

        for walls in floors:

            normalized = [self._normalize_line(w) for w in walls]

            merged = self._merge_floor(normalized)

            results.append(merged)

        return results

    def _merge_floor(self, walls: List[Line]) -> List[Dict]:

        used: Set[int] = set()
        merged_walls: List[Dict] = []

        for i in range(len(walls)):

            if i in used:
                continue

            w1 = walls[i]
            pair_found = False

            for j in range(i + 1, len(walls)):

                if j in used:
                    continue

                w2 = walls[j]

                if not self._is_parallel(w1, w2):
                    continue

                dist = self._line_distance(w1, w2)

                if dist > self.distance_tol:
                    continue

                if not self._overlap(w1, w2):
                    continue

                center = self._center_line(w1, w2)

                merged_walls.append({
                    "center_line": center,
                    "thickness": dist
                })

                used.add(i)
                used.add(j)

                pair_found = True
                break

            if not pair_found:

                merged_walls.append({
                    "center_line": w1,
                    "thickness": 1.0
                })

                used.add(i)

        return merged_walls

    def _normalize_line(self, line: Line) -> Line:

        (x1, y1), (x2, y2) = line

        if (x1, y1) <= (x2, y2):
            return line

        return ((x2, y2), (x1, y1))

    def _is_parallel(self, a: Line, b: Line) -> bool:

        (x1, y1), (x2, y2) = a
        (x3, y3), (x4, y4) = b

        dx1 = x2 - x1
        dy1 = y2 - y1

        dx2 = x4 - x3
        dy2 = y4 - y3

        dot = dx1 * dx2 + dy1 * dy2

        mag1 = math.hypot(dx1, dy1)
        mag2 = math.hypot(dx2, dy2)

        if mag1 == 0 or mag2 == 0:
            return False

        cos_angle = abs(dot / (mag1 * mag2))

        angle = math.acos(min(1.0, cos_angle))

        return angle < self.angle_tol

    def _line_distance(self, a: Line, b: Line) -> float:

        (x1, y1), (x2, y2) = a
        (x3, y3), _ = b

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 and dy == 0:
            return float("inf")

        return abs(dy * x3 - dx * y3 + x2 * y1 - y2 * x1) / math.hypot(dx, dy)

    def _overlap(self, a: Line, b: Line) -> bool:

        (x1, y1), (x2, y2) = a
        (x3, y3), (x4, y4) = b

        if abs(x1 - x2) > abs(y1 - y2):

            a_min, a_max = sorted([x1, x2])
            b_min, b_max = sorted([x3, x4])

        else:

            a_min, a_max = sorted([y1, y2])
            b_min, b_max = sorted([y3, y4])

        overlap = max(0, min(a_max, b_max) - max(a_min, b_min))

        a_len = abs(a_max - a_min)

        if a_len == 0:
            return False

        return (overlap / a_len) >= self.overlap_ratio

    def _center_line(self, a: Line, b: Line) -> Line:

        (x1, y1), (x2, y2) = a
        (x3, y3), (x4, y4) = b

        cx1 = (x1 + x3) / 2
        cy1 = (y1 + y3) / 2

        cx2 = (x2 + x4) / 2
        cy2 = (y2 + y4) / 2

        return ((cx1, cy1), (cx2, cy2))