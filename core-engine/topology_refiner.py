from typing import List, Tuple, Dict, Set
from collections import defaultdict

Point = Tuple[int, int]
Line = Tuple[Point, Point]


class TopologyRefiner:

    def __init__(self, intersection_detector, grid: int = 10, max_dist: int = 100):
        self.intersection = intersection_detector
        self.grid = grid
        self.max_dist = max_dist

    # ===================== PUBLIC =====================

    def refine(self, walls: List[Line]) -> List[Line]:

        current = self._deduplicate_lines(walls)

        for _ in range(8):  # convergence loop

            split = self.intersection.process([current])[0]
            extended = self._extend_lines(split)

            cleaned = self._deduplicate_lines(extended)

            if set(cleaned) == set(current):
                break

            current = cleaned

        # final stabilization
        final = self.intersection.process([current])[0]
        final = self._deduplicate_lines(final)

        return final

    # ===================== EXTENSION =====================

    def _extend_lines(self, lines: List[Line]) -> List[Line]:
        """
        Extend ONLY dangling endpoints (degree == 1) to connect walls.
        Rules:
        - Only extend degree == 1 endpoints
        - Extension must follow wall direction
        - Extension must not cross existing walls
        - Only connect to orthogonal walls
        """
        
        # Count endpoint degrees
        deg = defaultdict(int)
        for a, b in lines:
            deg[a] += 1
            deg[b] += 1

        # Build spatial index by orientation
        horiz: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        vert: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

        for (x1, y1), (x2, y2) in lines:
            if y1 == y2:
                horiz[y1].append((min(x1, x2), max(x1, x2)))
            else:
                vert[x1].append((min(y1, y2), max(y1, y2)))

        new_lines = []

        for (x1, y1), (x2, y2) in lines:

            start_pt = (x1, y1)
            end_pt = (x2, y2)
            is_horizontal = (y1 == y2)

            # Try to extend start endpoint
            if deg[start_pt] == 1:
                ext = self._find_extension(start_pt, end_pt, is_horizontal, 
                                          horiz, vert, lines)
                if ext:
                    start_pt = ext

            # Try to extend end endpoint
            if deg[end_pt] == 1:
                ext = self._find_extension(end_pt, start_pt, is_horizontal,
                                          horiz, vert, lines)
                if ext:
                    end_pt = ext

            new_lines.append((start_pt, end_pt))

        return self._deduplicate_lines(new_lines)

    def _find_extension(self, endpoint: Point, other_end: Point, is_horizontal: bool,
                       horiz: Dict[int, List[Tuple[int, int]]],
                       vert: Dict[int, List[Tuple[int, int]]],
                       all_lines: List[Line]) -> Point:
        """
        Try to extend a single degree-1 endpoint.
        Returns extended endpoint or None if no valid target found.
        """
        
        x, y = endpoint
        ox, oy = other_end

        best = None
        best_dist = 1e9

        if is_horizontal:
            # Extending horizontal endpoint - look for vertical walls
            dx = x - ox

            for vx in vert:
                # DIRECTION CHECK: extend only in forward direction
                if dx > 0 and vx <= x:
                    continue
                if dx < 0 and vx >= x:
                    continue

                dist = abs(vx - x)
                if dist > self.max_dist:
                    continue

                # BLOCKING CHECK: path must be clear
                if self._blocks_path(x, y, vx, y, all_lines):
                    continue

                # Find valid target on vertical wall
                for y_start, y_end in vert[vx]:
                    if y_start - self.grid <= y <= y_end + self.grid:
                        if 0 < dist < best_dist:
                            best = (vx, y)
                            best_dist = dist

        else:
            # Extending vertical endpoint - look for horizontal walls
            dy = y - oy

            for hy in horiz:
                # DIRECTION CHECK: extend only in forward direction
                if dy > 0 and hy <= y:
                    continue
                if dy < 0 and hy >= y:
                    continue

                dist = abs(hy - y)
                if dist > self.max_dist:
                    continue

                # BLOCKING CHECK: path must be clear
                if self._blocks_path(x, y, x, hy, all_lines):
                    continue

                # Find valid target on horizontal wall
                for x_start, x_end in horiz[hy]:
                    if x_start - self.grid <= x <= x_end + self.grid:
                        if 0 < dist < best_dist:
                            best = (x, hy)
                            best_dist = dist

        return best

    def _blocks_path(self, x1: int, y1: int, x2: int, y2: int,
                     all_lines: List[Line]) -> bool:
        """
        Check if path from (x1,y1) to (x2,y2) crosses any existing walls.
        Returns True if path is blocked, False if clear.
        """
        
        if y1 == y2:
            # Horizontal path - check for vertical walls crossing
            min_x, max_x = min(x1, x2), max(x1, x2)
            for (wx1, wy1), (wx2, wy2) in all_lines:
                if wx1 == wx2 and wy1 != wy2:
                    # This is a vertical wall
                    if min_x < wx1 < max_x:
                        if min(wy1, wy2) <= y1 <= max(wy1, wy2):
                            return True
        else:
            # Vertical path - check for horizontal walls crossing
            min_y, max_y = min(y1, y2), max(y1, y2)
            for (wx1, wy1), (wx2, wy2) in all_lines:
                if wy1 == wy2 and wx1 != wx2:
                    # This is a horizontal wall
                    if min_y < wy1 < max_y:
                        if min(wx1, wx2) <= x1 <= max(wx1, wx2):
                            return True

        return False

    # ===================== UTIL =====================

    def _deduplicate_lines(self, lines: List[Line]) -> List[Line]:
        seen: Set[Tuple[Point, Point]] = set()
        result: List[Line] = []

        for a, b in lines:

            if a == b:
                continue

            key = (a, b) if a <= b else (b, a)

            if key in seen:
                continue

            seen.add(key)
            result.append(key)

        return result