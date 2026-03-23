import cv2
import numpy as np
from typing import List, Tuple

Line = Tuple[Tuple[int, int], Tuple[int, int]]


class WallDetector:

    def __init__(self, debug=True):
        self.debug = debug

    # ===================== MAIN =====================

    def detect(self, images: List[np.ndarray]) -> List[List[Line]]:
        results = []

        for idx, img in enumerate(images):

            # ---- PREPROCESS (LIGHT, NOT AGGRESSIVE) ----
            kernel = np.ones((3, 3), np.uint8)
            processed = cv2.dilate(img, kernel, iterations=1)

            detected = []

            # ===================== STAGE 1 (STRICT) =====================
            lines1 = cv2.HoughLinesP(
                processed,
                rho=1,
                theta=np.pi / 180,
                threshold=80,
                minLineLength=30,
                maxLineGap=15
            )

            # ===================== STAGE 2 (RELAXED) =====================
            lines2 = cv2.HoughLinesP(
                processed,
                rho=1,
                theta=np.pi / 180,
                threshold=40,
                minLineLength=30,
                maxLineGap=30
            )

            # ---- COMBINE BOTH ----
            for lines in [lines1, lines2]:
                if lines is None:
                    continue

                for l in lines:
                    x1, y1, x2, y2 = l[0]

                    dx, dy = x2 - x1, y2 - y1
                    angle = abs(np.degrees(np.arctan2(dy, dx)))

                    # ---- ORIENTATION FILTER ----
                    if not (
                        angle < 10 or
                        abs(angle - 90) < 10 or
                        abs(angle - 180) < 10
                    ):
                        continue

                    detected.append(((x1, y1), (x2, y2)))

            # ===================== REMOVE SMALL LINES =====================
            detected = self._remove_small(detected, min_len=15)

            if self.debug:
                print(f"[{idx}] detected (after cleanup): {len(detected)}")
                self._show(img, detected, idx)

            results.append(detected)

        return results

    # ===================== HELPERS =====================

    def _remove_small(self, lines: List[Line], min_len=15) -> List[Line]:
        res = []

        for (x1, y1), (x2, y2) in lines:
            length = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
            if length >= min_len:
                res.append(((x1, y1), (x2, y2)))

        return res

    # ===================== VISUAL =====================

    def _show(self, img, lines, idx):
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for (x1, y1), (x2, y2) in lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow(f"detected_{idx}", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()