from typing import List, Tuple, Dict
import numpy as np

Line = Tuple[Tuple[int, int], Tuple[int, int]]


class IntersectionSplitter:

    def __init__(self, tol_ratio=0.01, min_seg_ratio=0.01, debug=True):
        self.tol_ratio = tol_ratio
        self.min_seg_ratio = min_seg_ratio
        self.debug = debug

        self.tol = None
        self.min_seg = None

    # ===================== MAIN =====================

    def split(self, lines_with_votes: List[Dict]) -> List[Dict]:

        if not lines_with_votes:
            return []

        # ---- SCALE COMPUTATION ----
        pts = [p for d in lines_with_votes for p in d["line"]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        scale = np.hypot(max(xs) - min(xs), max(ys) - min(ys))

        self.tol = self.tol_ratio * scale
        self.min_seg = self.min_seg_ratio * scale

        if self.debug:
            print(f"[Intersection] tol={self.tol:.2f}, min_seg={self.min_seg:.2f}")

        # ---- FIND INTERSECTIONS ----
        intersections = self._find_intersections(lines_with_votes)

        # ---- SPLIT LINES ----
        new_lines = []

        for d in lines_with_votes:
            line = d["line"]
            votes = d["votes"]

            split_pts = []

            for p in intersections:
                if self._on_segment(line, p):
                    if not self._near_endpoint(line, p):
                        split_pts.append(p)

            if not split_pts:
                new_lines.append(d)
                continue

            pts = [line[0]] + sorted(split_pts, key=lambda p: self._dist(line[0], p)) + [line[1]]

            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i + 1]

                if self._dist(p1, p2) < self.min_seg:
                    continue

                new_lines.append({
                    "line": (p1, p2),
                    "votes": votes
                })

        # ---- REMOVE DUPLICATES ----
        final = self._deduplicate(new_lines)

        if self.debug:
            print(f"[Intersection] Before: {len(lines_with_votes)}, After: {len(final)}")

        return final

    # ===================== INTERSECTION =====================

    def _find_intersections(self, lines):

        points = []

        for i in range(len(lines)):
            l1 = lines[i]["line"]

            for j in range(i + 1, len(lines)):
                l2 = lines[j]["line"]

                p = self._intersection_point(l1, l2)

                if p is None:
                    continue

                if self._on_segment(l1, p) and self._on_segment(l2, p):
                    points.append(p)

        return points

    def _intersection_point(self, l1, l2):
        (x1, y1), (x2, y2) = l1
        (x3, y3), (x4, y4) = l2

        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)

        if abs(denom) < 1e-6:
            return None

        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom

        return (int(px), int(py))

    # ===================== HELPERS =====================

    def _on_segment(self, l, p):
        (x1, y1), (x2, y2) = l
        px, py = p

        return (
            min(x1, x2) - self.tol <= px <= max(x1, x2) + self.tol and
            min(y1, y2) - self.tol <= py <= max(y1, y2) + self.tol
        )

    def _near_endpoint(self, l, p):
        return (
            self._dist(l[0], p) < self.tol or
            self._dist(l[1], p) < self.tol
        )

    def _dist(self, a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    # ===================== DEDUP =====================

    def _deduplicate(self, lines):

        unique = []

        for d in lines:
            l = d["line"]

            exists = False
            for u in unique:
                if self._same_line(l, u["line"]):
                    exists = True
                    break

            if not exists:
                unique.append(d)

        return unique

    def _same_line(self, l1, l2):
        return (
            self._dist(l1[0], l2[0]) < self.tol and
            self._dist(l1[1], l2[1]) < self.tol
        ) or (
            self._dist(l1[0], l2[1]) < self.tol and
            self._dist(l1[1], l2[0]) < self.tol
        )