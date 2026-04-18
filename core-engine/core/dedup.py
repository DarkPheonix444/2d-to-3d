import numpy as np
import cv2


# ============================================================
# 🔥 ENDPOINT SNAPPING (CRITICAL FIX)
# ============================================================

def snap_endpoints(lines, tol=6):

    new_lines = []

    for i, ((x1, y1), (x2, y2)) in enumerate(lines):

        for j, ((x3, y3), (x4, y4)) in enumerate(lines):

            # snap start point
            if abs(x1 - x3) <= tol and abs(y1 - y3) <= tol:
                x1, y1 = x3, y3

            # snap end point
            if abs(x2 - x4) <= tol and abs(y2 - y4) <= tol:
                x2, y2 = x4, y4

        new_lines.append(((x1, y1), (x2, y2)))

    return new_lines


# ============================================================
# 🔥 DEDUPLICATOR
# ============================================================

class Deduplicator:

    def __init__(self, tol=6, debug=True):
        self.tol = tol
        self.debug = debug

    # ===================== MAIN =====================

    def process(self, lines, base_img=None):

        if not lines:
            return []

        # ---- STEP 1: normalize ----
        norm_lines = [self._normalize(l) for l in lines]

        # ---- STEP 2: deduplicate ----
        unique = self._dedup(norm_lines)

        # ---- STEP 3: snap endpoints (IMPORTANT) ----
        snapped = snap_endpoints(unique, tol=self.tol)

        # ---- STEP 4: recompute length ----
        result = []
        for l in snapped:
            result.append({
                "line": l,
                "length": self._length(l)
            })

        if self.debug:
            print(f"[Deduplicator] in={len(lines)}, out={len(snapped)}")

        # ---- VISUALIZATION ----
        if self.debug and base_img is not None:
            self._visualize(base_img, lines, snapped)

        return result

    # ===================== NORMALIZE =====================

    def _normalize(self, l):
        (x1, y1), (x2, y2) = l

        def snap(v):
            return int(round(v / self.tol) * self.tol)

        x1, y1 = snap(x1), snap(y1)
        x2, y2 = snap(x2), snap(y2)

        # enforce straight lines
        if abs(x1 - x2) < self.tol:
            x2 = x1
        if abs(y1 - y2) < self.tol:
            y2 = y1

        return ((x1, y1), (x2, y2))

    # ===================== DEDUP =====================

    def _dedup(self, lines):
        seen = set()
        result = []

        for l in lines:
            a, b = l
            key = (a, b) if a <= b else (b, a)

            if key not in seen:
                seen.add(key)
                result.append(key)

        return result

    # ===================== VISUALIZATION =====================

    def _visualize(self, base_img, before, after):

        if len(base_img.shape) == 2:
            vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            vis = base_img.copy()

        # BEFORE (RED)
        for (x1, y1), (x2, y2) in before:
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

        # AFTER (GREEN)
        for (x1, y1), (x2, y2) in after:
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        h, w = vis.shape[:2]
        scale = min(900 / w, 700 / h, 1.0)
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)))

        cv2.imshow("Dedup + Snap (Red=Before, Green=After)", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ===================== UTIL =====================

    def _length(self, l):
        (x1, y1), (x2, y2) = l
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5