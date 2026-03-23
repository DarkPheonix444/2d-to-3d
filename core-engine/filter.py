from typing import List, Tuple
import numpy as np

Line = Tuple[Tuple[int, int], Tuple[int, int]]


class WallFilter:

    def __init__(self, min_length=40, angle_tol=10):
        self.min_length = min_length
        self.angle_tol = angle_tol

    def filter(self, lines: List[Line]) -> List[Line]:

        result = []

        for l in lines:
            (x1, y1), (x2, y2) = l

            dx = x2 - x1
            dy = y2 - y1

            length = np.hypot(dx, dy)

            # ---- LENGTH CHECK ----
            if length < self.min_length:
                continue

            # ---- ORIENTATION CHECK ----
            angle = abs(np.degrees(np.arctan2(dy, dx)))

            is_horizontal = angle < self.angle_tol or abs(angle - 180) < self.angle_tol
            is_vertical = abs(angle - 90) < self.angle_tol

            if not (is_horizontal or is_vertical):
                continue

            # ---- SNAP TO GRID (CLEAN LINES) ----
            if is_horizontal:
                y = int((y1 + y2) / 2)
                result.append(((min(x1, x2), y), (max(x1, x2), y)))
            else:
                x = int((x1 + x2) / 2)
                result.append(((x, min(y1, y2)), (x, max(y1, y2))))

        return result