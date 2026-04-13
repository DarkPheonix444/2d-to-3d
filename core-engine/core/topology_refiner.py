from typing import List, Tuple
from collections import defaultdict
import numpy as np
import cv2

Point = Tuple[int, int]
Line = Tuple[Point, Point]


class TopologyRefiner:

    def __init__(self, intersection_detector, tol=8, debug=True):
        self.intersection = intersection_detector
        self.tol = tol
        self.debug = debug

    # ===================== MAIN =====================

    def refine(self, walls: List[Line], img=None) -> List[Line]:

        current = self._snap_lines(self._deduplicate(walls))

        snap_tol = int(self.tol * 1.2)

        for _ in range(5):

            self.current_lines = current

            # --- intersection split ---
            split_data = self.intersection.split(
                [{"line": l, "votes": 1} for l in current]
            )
            split = [d["line"] for d in split_data]

            split = self._snap_lines(split)

            # --- compute avg length (adaptive scaling) ---
            lengths = [
                ((x2-x1)**2 + (y2-y1)**2)**0.5
                for (x1,y1),(x2,y2) in split
            ]
            avg_len = np.mean(lengths) if lengths else 1

            # --- controlled extension ---
            extended = self._extend_lines(split, avg_len)

            # --- safe gap connection ---
            connected = self._safe_connect(extended)

            # --- global snapping ---
            connected = self._snap_endpoints_global(connected, snap_tol)

            cleaned = self._deduplicate(connected)

            if set(cleaned) == set(current):
                break

            current = cleaned

        final = self._snap_lines(current)

        return final

    # ===================== SAFE GAP CONNECT =====================

    def _safe_connect(self, lines):
        new_lines = list(lines)

        for i in range(len(lines)):
            (x1, y1), (x2, y2) = lines[i]
            len1 = ((x2-x1)**2 + (y2-y1)**2)**0.5

            for j in range(i + 1, len(lines)):
                (x3, y3), (x4, y4) = lines[j]
                len2 = ((x4-x3)**2 + (y4-y3)**2)**0.5

                is_h1 = abs(y1 - y2) <= self.tol
                is_h2 = abs(y3 - y4) <= self.tol

                is_v1 = abs(x1 - x2) <= self.tol
                is_v2 = abs(x3 - x4) <= self.tol

                # ---------- HORIZONTAL ----------
                if is_h1 and is_h2:

                    if abs(y1 - y3) > self.tol:
                        continue

                    seg1 = sorted([x1, x2])
                    seg2 = sorted([x3, x4])

                    # 🔥 correct gap direction
                    if seg1[1] <= seg2[0]:
                        gap = seg2[0] - seg1[1]
                        p, q = (seg1[1], y1), (seg2[0], y1)
                    elif seg2[1] <= seg1[0]:
                        gap = seg1[0] - seg2[1]
                        p, q = (seg2[1], y1), (seg1[0], y1)
                    else:
                        continue

                    local_thresh = min(len1, len2) * 0.2

                    if 0 < gap <= local_thresh:
                        new_lines.append((p, q))

                # ---------- VERTICAL ----------
                elif is_v1 and is_v2:

                    if abs(x1 - x3) > self.tol:
                        continue

                    seg1 = sorted([y1, y2])
                    seg2 = sorted([y3, y4])

                    if seg1[1] <= seg2[0]:
                        gap = seg2[0] - seg1[1]
                        p, q = (x1, seg1[1]), (x1, seg2[0])
                    elif seg2[1] <= seg1[0]:
                        gap = seg1[0] - seg2[1]
                        p, q = (x1, seg2[1]), (x1, seg1[0])
                    else:
                        continue

                    local_thresh = min(len1, len2) * 0.2

                    if 0 < gap <= local_thresh:
                        new_lines.append((p, q))

        return new_lines

    # ===================== EXTENSION =====================

    def _extend_lines(self, lines: List[Line], avg_len):

        deg = defaultdict(int)
        for a, b in lines:
            deg[a] += 1
            deg[b] += 1

        new_lines = []

        for (x1, y1), (x2, y2) in lines:

            start = (x1, y1)
            end = (x2, y2)

            length = ((x2-x1)**2 + (y2-y1)**2)**0.5

            is_horizontal = abs(y1 - y2) <= self.tol

            # 🔥 adaptive filtering
            if length < 0.1 * avg_len:
                new_lines.append((start, end))
                continue

            if deg[start] == 1:
                ext = self._find_extension(start, end, is_horizontal)
                if ext:
                    start = ext

            if deg[end] == 1:
                ext = self._find_extension(end, start, is_horizontal)
                if ext:
                    end = ext

            if start != end:
                new_lines.append((start, end))

        return new_lines

    # ===================== EXTENSION HELPER =====================

    def _find_extension(self, endpoint, other_end, is_horizontal):

        x, y = endpoint

        best = None
        best_dist = 1e9

        for (x1, y1), (x2, y2) in self.current_lines:

            if is_horizontal:
                if abs(y1 - y2) > self.tol:
                    continue

                if not (min(x1, x2) <= x <= max(x1, x2)):
                    continue

                dist = abs(y1 - y)

                if dist < best_dist and dist <= self.tol * 1.5:
                    best = (x, y1)
                    best_dist = dist

            else:
                if abs(x1 - x2) > self.tol:
                    continue

                if not (min(y1, y2) <= y <= max(y1, y2)):
                    continue

                dist = abs(x1 - x)

                if dist < best_dist and dist <= self.tol * 1.5:
                    best = (x1, y)
                    best_dist = dist

        return best

    # ===================== GLOBAL SNAP =====================

    def _snap_endpoints_global(self, lines, snap_tol):

        points = list({p for l in lines for p in l})
        parent = {}

        def find(p):
            while parent[p] != p:
                parent[p] = parent[parent[p]]
                p = parent[p]
            return p

        def union(a, b):
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pb] = pa

        for p in points:
            parent[p] = p

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p, q = points[i], points[j]
                if abs(p[0] - q[0]) <= snap_tol and abs(p[1] - q[1]) <= snap_tol:
                    union(p, q)

        rep_map = {p: find(p) for p in points}

        new_lines = []
        for a, b in lines:
            a = rep_map[a]
            b = rep_map[b]
            if a != b:
                new_lines.append((a, b))

        return new_lines

    # ===================== UTIL =====================

    def _snap_lines(self, lines):
        snapped = []
        for (x1, y1), (x2, y2) in lines:
            if abs(y1 - y2) <= self.tol:
                y = int((y1 + y2) / 2)
                snapped.append(((x1, y), (x2, y)))
            elif abs(x1 - x2) <= self.tol:
                x = int((x1 + x2) / 2)
                snapped.append(((x, y1), (x, y2)))
            else:
                snapped.append(((x1, y1), (x2, y2)))
        return snapped

    def _deduplicate(self, lines):
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