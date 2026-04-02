from typing import List, Tuple
import numpy as np

Line = Tuple[Tuple[int, int], Tuple[int, int]]


class LineNormalizer:

    def __init__(self, grid_ratio=0.02):
        self.grid_ratio = grid_ratio

    def normalize(self, lines: List[Line]) -> List[Line]:

        if not lines:
            return []

        pts = [p for l in lines for p in l]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        scale = np.hypot(max(xs) - min(xs), max(ys) - min(ys))
        grid = max(2, int(self.grid_ratio * scale))

        normalized = []

        for (x1, y1), (x2, y2) in lines:

            x1, y1 = self._snap(x1, y1, grid)
            x2, y2 = self._snap(x2, y2, grid)

            # enforce strict axis alignment
            if abs(x1 - x2) < abs(y1 - y2):
                x2 = x1
            else:
                y2 = y1

            if (x1, y1) > (x2, y2):
                x1, y1, x2, y2 = x2, y2, x1, y1

            normalized.append(((x1, y1), (x2, y2)))

        return normalized

    def _snap(self, x, y, grid):
        return int(round(x / grid) * grid), int(round(y / grid) * grid)