import cv2
import numpy as np
from typing import List, Tuple, Optional
from collections import defaultdict

Line = Tuple[Tuple[int, int], Tuple[int, int]]


class WallDetector:

    GRID = 10
    THICKNESS_TOL = 15  # max distance to consider parallel walls

    def __init__(
        self,
        canny_low=50,
        canny_high=150,
        hough_threshold=50,
        min_line_length=25,
        max_line_gap=20,
        orientation_tol=15,
        min_wall_length=40,
        use_adaptive_canny=True
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.orientation_tol = orientation_tol
        self.min_wall_length = min_wall_length
        self.use_adaptive_canny = use_adaptive_canny

    # ===================== CORE =====================

    def detect(self, images: List[np.ndarray]) -> List[List[Line]]:
        results = []

        for img in images:
            gray = self._to_gray(img)
            gray = self._preprocess_structure(gray)
            edges = self._detect_edges(gray)
            lines = self._detect_lines(edges)

            walls = self._filter_walls(lines)

            # 🔥 FIX 1: SNAP EARLY
            walls = self._snap_all(walls)

            # 🔥 FIX 2: COLLAPSE THICK WALLS
            walls = self._collapse_parallel(walls)
            walls = self._snap_all(walls)
            # 🔥 FIX 3: MERGE CLEANLY
            walls = self._merge_walls(walls)
            walls= self._filter_small(walls)

            results.append(walls)

        return results

    # ===================== BASIC =====================

    def _snap(self, v: int) -> int:
        return int(round(v / self.GRID) * self.GRID)

    def _snap_all(self, walls: List[Line]) -> List[Line]:
        snapped = []
        for (x1, y1), (x2, y2) in walls:
            snapped.append((
                (self._snap(x1), self._snap(y1)),
                (self._snap(x2), self._snap(y2))
            ))
        return snapped

    def _to_gray(self, image):
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _detect_edges(self, image):
        if not self.use_adaptive_canny:
            return cv2.Canny(image, self.canny_low, self.canny_high)

        median = np.median(image)
        low = int(max(0, 0.66 * median))
        high = int(min(255, 1.33 * median))
        return cv2.Canny(image, low, high)

    def _detect_lines(self, edges):
        return cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

    # ===================== FILTER =====================

    def _filter_walls(self, lines):
        walls = []
        if lines is None:
            return walls

        for l in lines:
            x1, y1, x2, y2 = l[0]

            dx = x2 - x1
            dy = y2 - y1

            angle = abs(np.degrees(np.arctan2(dy, dx)))
            if angle > 180:
                angle -= 180

            is_h = angle < self.orientation_tol or abs(angle - 180) < self.orientation_tol
            is_v = abs(angle - 90) < self.orientation_tol

            if not (is_h or is_v):
                continue

            length = np.hypot(dx, dy)
            if length < self.min_wall_length:
                continue

            if is_h:
                y = int(round((y1 + y2) / 2))
                x1, x2 = sorted([x1, x2])
                walls.append(((x1, y), (x2, y)))
            else:
                x = int(round((x1 + x2) / 2))
                y1, y2 = sorted([y1, y2])
                walls.append(((x, y1), (x, y2)))

        return walls
    # proces structure 
    def _preprocess_structure(self, img):
        _, bin_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

        # 🔥 horizontal closing (fix divider)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        h_closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, h_kernel)

        # 🔥 vertical closing
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        v_closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, v_kernel)

        combined = cv2.bitwise_or(h_closed, v_closed)

        return combined

    # ===================== 🔥 KEY FIX =====================

    def _collapse_parallel(self, walls: List[Line]) -> List[Line]:
        """
        Convert double-line walls → single centerline
        """

        horizontal = []
        vertical = []

        for (x1, y1), (x2, y2) in walls:
            if y1 == y2:
                horizontal.append(((x1, y1), (x2, y2)))
            else:
                vertical.append(((x1, y1), (x2, y2)))

        collapsed = []

        # ---- Horizontal collapse ----
        used = set()
        for i in range(len(horizontal)):
            if i in used:
                continue

            (x1, y1), (x2, y2) = horizontal[i]
            group = [(x1, x2, y1)]
            used.add(i)

            for j in range(i + 1, len(horizontal)):
                if j in used:
                    continue

                (xx1, yy1), (xx2, yy2) = horizontal[j]

                current_center = np.mean([g[2] for g in group])
                if abs(yy1 - current_center) <= self.THICKNESS_TOL:
                    # Overlap check
                    if not (xx2 < x1 or xx1 > x2):
                        group.append((xx1, xx2, yy1))
                        used.add(j)

            # Collapse group → centerline
            xs = []
            ys = []
            for gx1, gx2, gy in group:
                xs.append(gx1)
                xs.append(gx2)
                ys.append(gy)

            y_center = int(round(np.mean(ys)))
            collapsed.append(((min(xs), y_center), (max(xs), y_center)))

        # ---- Vertical collapse ----
        used = set()
        for i in range(len(vertical)):
            if i in used:
                continue

            (x1, y1), (x2, y2) = vertical[i]
            group = [(y1, y2, x1)]
            used.add(i)

            for j in range(i + 1, len(vertical)):
                if j in used:
                    continue

                (xx1, yy1), (xx2, yy2) = vertical[j]

                current_center = np.mean([g[2] for g in group])
                if abs(xx1 - current_center) <= self.THICKNESS_TOL:
                    if not (yy2 < y1 or yy1 > y2):
                        group.append((yy1, yy2, xx1))
                        used.add(j)

            ys = []
            xs = []
            for gy1, gy2, gx in group:
                ys.append(gy1)
                ys.append(gy2)
                xs.append(gx)

            x_center = int(round(np.mean(xs)))
            collapsed.append(((x_center, min(ys)), (x_center, max(ys))))

        return collapsed

    # ===================== MERGE =====================

    def _merge_walls(self, walls):
        horizontal = defaultdict(list)
        vertical = defaultdict(list)

        # group
        for (x1, y1), (x2, y2) in walls:
            if y1 == y2:
                horizontal[y1].append((min(x1, x2), max(x1, x2)))
            else:
                vertical[x1].append((min(y1, y2), max(y1, y2)))

        merged = []

        GAP = self.GRID  # 🔥 control knob (10 px)

        # ---------- HORIZONTAL ----------
        for y, segs in horizontal.items():
            segs.sort()
            s, e = segs[0]

            for ns, ne in segs[1:]:
                gap = ns - e

                # ✅ merge only if overlap OR small gap
                if gap <= GAP:
                    e = max(e, ne)
                else:
                    merged.append(((s, y), (e, y)))
                    s, e = ns, ne

            merged.append(((s, y), (e, y)))

        # ---------- VERTICAL ----------
        for x, segs in vertical.items():
            segs.sort()
            s, e = segs[0]

            for ns, ne in segs[1:]:
                gap = ns - e

                if gap <= GAP:
                    e = max(e, ne)
                else:
                    merged.append(((x, s), (x, e)))
                    s, e = ns, ne

            merged.append(((x, s), (x, e)))

        return merged
    
    def _filter_small(self, walls):
        return [
            w for w in walls
            if abs(w[0][0]-w[1][0]) + abs(w[0][1]-w[1][1]) >= self.GRID * 2
        ]