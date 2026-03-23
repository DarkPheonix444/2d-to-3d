from typing import List, Tuple, Dict, Set
from collections import defaultdict
import cv2

Point = Tuple[int, int]
Line = Tuple[Point, Point]


class TopologyRefiner:

    def __init__(self, intersection_detector, tol=6, max_dist=20, debug=True):
        self.intersection = intersection_detector
        self.tol = tol
        self.max_dist = max_dist
        self.debug = debug

    # ===================== MAIN =====================

    def refine(self, walls: List[Line], img=None) -> List[Line]:

        current = self._snap_lines(self._deduplicate(walls))

        for _ in range(5):

            # ensure clean splits
            split = self.intersection.process([current])[0]

            # snap again after split
            split = self._snap_lines(split)

            # extend
            extended = self._extend_lines(split)

            # clean
            cleaned = self._deduplicate(extended)

            if set(cleaned) == set(current):
                break

            current = cleaned

        final = self._snap_lines(current)

        if self.debug and img is not None:
            self._show(img, final, "topology")

        return final

    # ===================== SNAP =====================

    def _snap_lines(self, lines: List[Line]) -> List[Line]:

        snapped = []

        for (x1, y1), (x2, y2) in lines:

            # horizontal
            if abs(y1 - y2) <= self.tol:
                y = int((y1 + y2) / 2)
                snapped.append(((x1, y), (x2, y)))

            # vertical
            elif abs(x1 - x2) <= self.tol:
                x = int((x1 + x2) / 2)
                snapped.append(((x, y1), (x, y2)))

            else:
                snapped.append(((x1, y1), (x2, y2)))

        return snapped

    # ===================== EXTENSION =====================

    def _extend_lines(self, lines: List[Line]) -> List[Line]:

        deg = defaultdict(int)
        for a, b in lines:
            deg[a] += 1
            deg[b] += 1

        horiz = defaultdict(list)
        vert = defaultdict(list)

        for (x1, y1), (x2, y2) in lines:

            if abs(y1 - y2) <= self.tol:
                y = int((y1 + y2) / 2)
                horiz[y].append((min(x1, x2), max(x1, x2)))

            elif abs(x1 - x2) <= self.tol:
                x = int((x1 + x2) / 2)
                vert[x].append((min(y1, y2), max(y1, y2)))

        new_lines = []

        for (x1, y1), (x2, y2) in lines:

            start = (x1, y1)
            end = (x2, y2)

            is_horizontal = abs(y1 - y2) <= self.tol

            # extend start
            if deg[start] == 1:
                ext = self._find_extension(start, end, is_horizontal, horiz, vert, lines)
                if ext:
                    start = ext

            # extend end
            if deg[end] == 1:
                ext = self._find_extension(end, start, is_horizontal, horiz, vert, lines)
                if ext:
                    end = ext

            new_lines.append((start, end))

        return new_lines

    def _find_extension(self, endpoint, other_end, is_horizontal, horiz, vert, lines):

        x, y = endpoint
        ox, oy = other_end

        best = None
        best_dist = 1e9

        if is_horizontal:
            dx = x - ox

            for vx in vert:

                if dx > 0 and vx <= x:
                    continue
                if dx < 0 and vx >= x:
                    continue

                dist = abs(vx - x)
                if dist > self.max_dist:
                    continue

                for y1, y2 in vert[vx]:
                    if y1 - self.tol <= y <= y2 + self.tol:
                        if dist < best_dist:
                            best = (vx, y)
                            best_dist = dist

        else:
            dy = y - oy

            for hy in horiz:

                if dy > 0 and hy <= y:
                    continue
                if dy < 0 and hy >= y:
                    continue

                dist = abs(hy - y)
                if dist > self.max_dist:
                    continue

                for x1, x2 in horiz[hy]:
                    if x1 - self.tol <= x <= x2 + self.tol:
                        if dist < best_dist:
                            best = (x, hy)
                            best_dist = dist

        return best

    # ===================== UTIL =====================

    def _deduplicate(self, lines: List[Line]) -> List[Line]:

        seen = set()
        result = []

        for a, b in lines:

            if a == b:
                continue

            key = (a, b) if a <= b else (b, a)

            if key not in seen:
                seen.add(key)
                result.append(key)

        return result

    # ===================== VISUAL =====================

    def _show(self, img, lines, title="topology"):
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for (x1, y1), (x2, y2) in lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.imshow(title, vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()