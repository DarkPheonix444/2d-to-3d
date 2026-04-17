import numpy as np
import cv2


class LightCleaner:

    def __init__(self, tol=8, debug=True):
        self.tol = tol
        self.debug = debug

    # ===================== MAIN =====================

    def clean(self, lines, base_img=None):

        if not lines:
            return []

        # ---- STEP 1: remove non-orthogonal ----
        ortho = []
        for l in lines:
            (x1, y1), (x2, y2) = l

            if abs(x1 - x2) <= self.tol or abs(y1 - y2) <= self.tol:
                ortho.append(l)

        # ---- STEP 2: remove small segments ----
        lengths = [self._length(l) for l in ortho]
        avg_len = np.mean(lengths) if lengths else 1

        filtered = [l for l in ortho if self._length(l) > 0.08 * avg_len]

        # ---- STEP 2.5: remove short orthogonal clutter ----
        filtered2 = []
        for l in filtered:
            if self._length(l) < 0.15 * avg_len:
                continue
            filtered2.append(l)

        filtered = filtered2

        # ---- STEP 3: weak connectivity ----
        points = [p for l in filtered for p in l]

        def is_connected(p):
            count = 0
            for q in points:
                if abs(p[0] - q[0]) <= self.tol * 2 and abs(p[1] - q[1]) <= self.tol * 2:
                    count += 1
            return count > 1

        cleaned = []
        for l in filtered:
            if is_connected(l[0]) or is_connected(l[1]):
                cleaned.append(l)

        # ---- DEBUG ----
        if self.debug:
            print(f"[LightCleaner] in={len(lines)}, ortho={len(ortho)}, out={len(cleaned)}")

        # ---- VISUALIZATION ----
        if self.debug and base_img is not None:
            self._visualize(base_img, lines, cleaned)

        return cleaned

    # ===================== VISUALIZATION =====================

    def _visualize(self, base_img, before, after):

        if len(base_img.shape) == 2:
            vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            vis = base_img.copy()

        # ---- BEFORE (RED) ----
        for (x1, y1), (x2, y2) in before:
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

        # ---- AFTER (GREEN) ----
        for (x1, y1), (x2, y2) in after:
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        h, w = vis.shape[:2]
        scale = min(900 / w, 700 / h, 1.0)
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)))

        cv2.imshow("Light Cleaner Debug (Red=Before, Green=After)", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ===================== UTIL =====================

    def _length(self, l):
        (x1, y1), (x2, y2) = l
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5