from typing import List, Tuple
from collections import defaultdict
import cv2

Point = Tuple[int, int]
Line = Tuple[Point, Point]


class TopologyRefiner:

    def __init__(self, intersection_detector, tol=8, max_dist=20, debug=True):
        self.intersection = intersection_detector
        self.tol = tol
        self.max_dist = max_dist
        self.debug = debug

    # ===================== MAIN =====================

    def refine(self, walls: List[Line], img=None) -> List[Line]:

        current = self._snap_lines(self._deduplicate(walls))

        pts = [p for l in current for p in l]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        scale = max(max(xs)-min(xs), max(ys)-min(ys))

        self.max_dist = int(0.03 * scale)
        snap_tol = int(self.tol * 2)

        for _ in range(5):

            # --- intersection split ---
            split_data = self.intersection.split(
                [{"line": l, "votes": 1} for l in current]
            )
            split = [d["line"] for d in split_data]

            split = self._snap_lines(split)

            # --- controlled extension ---
            extended = self._extend_lines(split)

            # --- SAFE GAP CONNECTION (NEW CORE FIX) ---
            connected = self._safe_connect(extended)

            # --- global snapping ---
            connected = self._snap_endpoints_global(connected, snap_tol)

            cleaned = self._deduplicate(connected)

            if set(cleaned) == set(current):
                break

            current = cleaned

        final = self._snap_lines(current)

        if self.debug:
            self._analyze_topology(final)

        if self.debug and img is not None:
            self._show(img, final)

        return final

    # ===================== SAFE GAP CONNECT =====================

    def _safe_connect(self, lines):
        new_lines = list(lines)

        for i in range(len(lines)):
            (x1, y1), (x2, y2) = lines[i]

            for j in range(i + 1, len(lines)):
                (x3, y3), (x4, y4) = lines[j]

                # orientation
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

                    gap = max(seg1[0], seg2[0]) - min(seg1[1], seg2[1])

                    if 0 < gap <= self.max_dist:
                        p = (seg1[1], y1)
                        q = (seg2[0], y1)
                        new_lines.append((p, q))

                # ---------- VERTICAL ----------
                elif is_v1 and is_v2:
                    if abs(x1 - x3) > self.tol:
                        continue

                    seg1 = sorted([y1, y2])
                    seg2 = sorted([y3, y4])

                    gap = max(seg1[0], seg2[0]) - min(seg1[1], seg2[1])

                    if 0 < gap <= self.max_dist:
                        p = (x1, seg1[1])
                        q = (x1, seg2[0])
                        new_lines.append((p, q))

        return new_lines

    # ===================== EXTENSION (RESTRICTED) =====================

    def _extend_lines(self, lines: List[Line]):

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

            # 🔥 ONLY extend true endpoints
            if deg[start] == 1:
                ext = self._find_extension(start, end, is_horizontal, horiz, vert)
                if ext:
                    start = ext

            if deg[end] == 1:
                ext = self._find_extension(end, start, is_horizontal, horiz, vert)
                if ext:
                    end = ext

            if start != end:
                new_lines.append((start, end))

        return new_lines

    # ===================== EXTENSION HELPER =====================

    def _find_extension(self, endpoint, other_end, is_horizontal, horiz, vert):

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