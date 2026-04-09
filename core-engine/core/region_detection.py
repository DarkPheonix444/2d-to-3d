import numpy as np
import cv2
from typing import List, Tuple

Point = Tuple[int, int]
Line = Tuple[Point, Point]


class RegionDetector:

    def __init__(self, thickness=3, debug=True):
        self.thickness = thickness
        self.debug = debug

    # ===================== MAIN =====================

    def detect(self, walls: List[Line], image_shape):

        H, W = image_shape[:2]

        # ------------------ 1. WALL MASK ------------------
        wall_mask = np.zeros((H, W), dtype=np.uint8)

        for (x1, y1), (x2, y2) in walls:
            cv2.line(wall_mask, (x1, y1), (x2, y2), 255, 6)  # 🔥 thicker walls

        # ------------------ 2. STRONG GAP CLOSING ------------------
        kernel = np.ones((5, 5), np.uint8)

        wall_mask = cv2.dilate(wall_mask, kernel, iterations=3)
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 🔥 balance (prevent overgrowth)
        wall_mask = cv2.erode(wall_mask, kernel, iterations=1)

        # ------------------ 3. FREE SPACE ------------------
        free_space = cv2.bitwise_not(wall_mask)

        # ------------------ 4. CONNECTED COMPONENTS ------------------
        num_labels, labels = cv2.connectedComponents(free_space)

        # ------------------ 5. AREA COMPUTE ------------------
        areas = [np.sum(labels == i) for i in range(num_labels)]

        outside = np.argmax(areas)

        rooms = []

        for i in range(num_labels):

            if i == outside:
                continue

            # 🔥 better threshold (scale-aware)
            if areas[i] < 3000:
                continue

            mask = (labels == i).astype(np.uint8) * 255

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                rooms.append(contours[0])

        if self.debug:
            self._visualize(wall_mask, free_space, rooms)

        return rooms
    # ===================== DEBUG =====================

    def _visualize(self, wall_mask, rooms):

        vis = cv2.cvtColor(wall_mask, cv2.COLOR_GRAY2BGR)

        for cnt in rooms:
            cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)

        cv2.imshow("Region Rooms", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()