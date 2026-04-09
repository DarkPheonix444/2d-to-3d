import numpy as np
import cv2
from typing import List, Tuple

Point = Tuple[int, int]
Line = Tuple[Point, Point]


class RegionRefiner:

    def __init__(self, min_area=3000, debug=True):
        self.min_area = min_area
        self.debug = debug

    # ===================== MAIN =====================

    def refine(self, regions: List[np.ndarray], walls: List[Line], shape):

        H, W = shape[:2]

        # ------------------ 1. CONVERT TO MASKS ------------------
        region_masks = []
        for cnt in regions:
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            region_masks.append(mask)

        # ------------------ 2. REMOVE SMALL ------------------
        region_masks = [
            m for m in region_masks if np.sum(m) > self.min_area
        ]

        # ------------------ 3. ITERATIVE MERGE ------------------
        merged = True

        while merged:
            merged = False
            new_regions = []
            used = [False] * len(region_masks)

            for i in range(len(region_masks)):

                if used[i]:
                    continue

                current = region_masks[i]

                for j in range(i + 1, len(region_masks)):

                    if used[j]:
                        continue

                    if self._are_adjacent(current, region_masks[j]):

                        if not self._is_wall_supported_boundary(
                            current, region_masks[j], walls
                        ):
                            current = cv2.bitwise_or(current, region_masks[j])
                            used[j] = True
                            merged = True

                used[i] = True
                new_regions.append(current)

            region_masks = new_regions

        # ------------------ 4. BACK TO CONTOURS ------------------
        rooms = []

        for m in region_masks:
            contours, _ = cv2.findContours(
                m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                rooms.append(contours[0])

        if self.debug:
            self._visualize(rooms, shape)

        return rooms

    # ===================== ADJACENCY =====================

    def _are_adjacent(self, m1, m2):
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(m1, kernel, iterations=1)
        overlap = cv2.bitwise_and(dilated, m2)
        return np.any(overlap > 0)

    # ===================== CORE FIX =====================

    def _is_wall_supported_boundary(self, m1, m2, walls):

        # ---- 1. get shared boundary ----
        kernel = np.ones((3, 3), np.uint8)

        b1 = cv2.dilate(m1, kernel, iterations=1) - m1
        b2 = cv2.dilate(m2, kernel, iterations=1) - m2

        shared = cv2.bitwise_and(b1, b2)

        ys, xs = np.where(shared > 0)

        if len(xs) == 0:
            return False  # no boundary → merge

        # ---- 2. check wall support ----
        support = 0

        for (x, y) in zip(xs, ys):
            if self._point_near_any_wall((x, y), walls):
                support += 1

        ratio = support / len(xs)

        if self.debug:
            print(f"[Boundary] support={support}, total={len(xs)}, ratio={ratio:.2f}")

        # ---- 3. decision ----
        return ratio > 0.6

    # ===================== POINT → WALL =====================

    def _point_near_any_wall(self, p, walls, tol=5):

        for (x1, y1), (x2, y2) in walls:
            if self._point_near_line(p, (x1, y1), (x2, y2), tol):
                return True

        return False

    def _point_near_line(self, p, a, b, tol):

        px, py = p
        x1, y1 = a
        x2, y2 = b

        dx = x2 - x1
        dy = y2 - y1

        if dx == dy == 0:
            return False

        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))

        proj_x = x1 + t * dx
        proj_y = y1 + t * dy

        dist = np.hypot(px - proj_x, py - proj_y)

        return dist < tol

    # ===================== DEBUG =====================

    def _visualize(self, rooms, shape):

        vis = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

        for cnt in rooms:
            color = (
                np.random.randint(50, 255),
                np.random.randint(50, 255),
                np.random.randint(50, 255)
            )
            cv2.drawContours(vis, [cnt], -1, color, -1)

        cv2.imshow("Refined Rooms", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()