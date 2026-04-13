import numpy as np
import cv2


class StructuralCleaner:

    def __init__(self, debug=True):
        self.debug = debug
        self.initial_median = None
        self.final_median = None

    def clean(self, walls):
        if not walls:
            self.final_median = 0
            return []

        # ============================================================
        # 🔴 STEP 1: INITIAL MEDIAN
        # ============================================================
        lengths = [w["length"] for w in walls]
        self.initial_median = np.median(lengths)

        # ============================================================
        # 🔴 STEP 2: REMOVE VERY SMALL EDGES
        # ============================================================
        min_len = 0.03 * self.initial_median
        walls = [w for w in walls if w["length"] >= min_len]

        if self.debug:
            print(f"[Cleaner] after small removal: {len(walls)}")

        # ============================================================
        # 🔴 STEP 3: REMOVE ISOLATED EDGES (NO GRAPH)
        # ============================================================
        walls = self._remove_isolated(walls)

        if self.debug:
            print(f"[Cleaner] after isolated removal: {len(walls)}")

        # ============================================================
        # 🔴 STEP 4: REMOVE SMALL DANGLING (NON-RECURSIVE)
        # ============================================================
        walls = self._remove_small_dangling(walls)

        if self.debug:
            print(f"[Cleaner] after dangling removal: {len(walls)}")

        # ============================================================
        # 🔴 STEP 5: FINAL MEDIAN (FOR NEXT MODULES)
        # ============================================================
        if walls:
            self.final_median = np.median([w["length"] for w in walls])
        else:
            self.final_median = 0

        if self.debug:
            print(f"[Cleaner] final median: {self.final_median:.2f}")

        return walls

    # ============================================================
    # 🔴 REMOVE ISOLATED EDGES
    # ============================================================
    def _remove_isolated(self, walls):

        def dist(p1, p2):
            return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

        points = []
        for w in walls:
            points.append(w["line"][0])
            points.append(w["line"][1])

        new_walls = []

        for w in walls:
            p1, p2 = w["line"]

            connected = False

            for p in points:
                if p != p1 and dist(p1, p) < 10:
                    connected = True
                if p != p2 and dist(p2, p) < 10:
                    connected = True

            if connected:
                new_walls.append(w)

        return new_walls

    # ============================================================
    # 🔴 REMOVE SMALL DANGLING (ONE PASS ONLY)
    # ============================================================
    def _remove_small_dangling(self, walls):

        def dist(p1, p2):
            return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

        threshold = 0.05 * self.initial_median

        points = []
        for w in walls:
            points.append(w["line"][0])
            points.append(w["line"][1])

        new_walls = []

        for w in walls:
            p1, p2 = w["line"]

            connections = 0

            for p in points:
                if p != p1 and dist(p1, p) < 10:
                    connections += 1
                if p != p2 and dist(p2, p) < 10:
                    connections += 1

            # dangling if <=1 connection AND small
            if connections <= 1 and w["length"] < threshold:
                continue

            new_walls.append(w)

        return new_walls

    # ============================================================
    # 🔴 VISUALIZATION (KEEP SAME)
    # ============================================================
    def visualize_cleaner(self, base_img, before_walls, after_walls):

        if base_img is None:
            return

        if len(base_img.shape) == 2:
            vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            vis = base_img.copy()

        def norm(w):
            (x1, y1), (x2, y2) = w["line"]
            return (int(x1), int(y1), int(x2), int(y2))

        before_set = set(norm(w) for w in before_walls)
        after_set = set(norm(w) for w in after_walls)

        removed = before_set - after_set
        kept = after_set

        for (x1, y1, x2, y2) in removed:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for (x1, y1, x2, y2) in kept:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        h, w = vis.shape[:2]
        scale = min(900 / w, 700 / h, 1.0)
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)))

        cv2.imshow("Pre-Cleaner Debug", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()