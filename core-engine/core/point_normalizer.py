# point_normalizer.py

from typing import List, Dict, Tuple
import numpy as np
import cv2
from collections import defaultdict

Line = Tuple[Tuple[int, int], Tuple[int, int]]


class PointNormalizer:

    def __init__(self, tol: float, debug: bool = True, visualize: bool = False):
        self.tol = tol
        self.debug = debug
        self.visualize = visualize

    def normalize(self, merged_data: List[Dict], base_img=None) -> List[Dict]:

        if not merged_data:
            return []

        # ===================== COLLECT POINTS =====================
        points = [p for d in merged_data for p in d["line"]]

        # ===================== CLUSTER POINTS =====================
        clusters = []

        for p in points:
            assigned = False

            for c in clusters:
                cx, cy = c["center"]

                if np.hypot(p[0] - cx, p[1] - cy) <= self.tol:
                    c["points"].append(p)

                    xs = [pt[0] for pt in c["points"]]
                    ys = [pt[1] for pt in c["points"]]

                    c["center"] = (
                        int(round(sum(xs) / len(xs))),
                        int(round(sum(ys) / len(ys)))
                    )

                    assigned = True
                    break

            if not assigned:
                clusters.append({
                    "center": p,
                    "points": [p]
                })
        

        new_clusters = []

        for c in clusters:
            merged = False
            for nc in new_clusters:
                cx, cy = nc["center"]
                x, y = c["center"]

                if np.hypot(x - cx, y - cy) <= self.tol:
                    nc["points"].extend(c["points"])

                    xs = [pt[0] for pt in nc["points"]]
                    ys = [pt[1] for pt in nc["points"]]

                    nc["center"] = (
                        int(round(sum(xs) / len(xs))),
                        int(round(sum(ys) / len(ys)))
                    )

                    merged = True
                    break

            if not merged:
                new_clusters.append(c)

        clusters = new_clusters
        # ===================== BUILD MAPPING =====================
        mapping = {}
        for c in clusters:
            for p in c["points"]:
                mapping[p] = c["center"]

        # ===================== SNAP LINES =====================
        snapped = []
        collapse_count = 0

        for d in merged_data:
            (x1, y1), (x2, y2) = d["line"]

            p1 = mapping[(x1, y1)]
            p2 = mapping[(x2, y2)]

            if p1 == p2:
                collapse_count += 1
                continue

            snapped.append({
                "line": (p1, p2),
                "votes": d["votes"]
            })

        # ===================== DEBUG =====================
        if self.debug:
            before_unique = len(set(points))
            after_points = [p for d in snapped for p in d["line"]]
            after_unique = len(set(after_points))

            cluster_sizes = [len(c["points"]) for c in clusters]

            print("\n========== POINT NORMALIZER ==========")
            print(f"tol={self.tol:.2f}")
            print(f"points_before={before_unique}")
            print(f"points_after={after_unique}")
            print(f"clusters={len(clusters)}")
            print(f"collapsed_lines={collapse_count}")

            print("\n[CLUSTER DISTRIBUTION]")
            print(f"size_1={sum(1 for s in cluster_sizes if s == 1)}")
            print(f"size_2={sum(1 for s in cluster_sizes if s == 2)}")
            print(f"size_3+={sum(1 for s in cluster_sizes if s >= 3)}")

        # ===================== VISUALIZATION =====================
        if self.visualize and base_img is not None:
            self._visualize(points, mapping, base_img)

        return snapped

    # ===================== VISUALIZER =====================

    def _visualize(self, original_points, mapping, base_img):

        vis = cv2.cvtColor(base_img.copy(), cv2.COLOR_GRAY2BGR)

        # Draw original points (RED)
        for p in original_points:
            cv2.circle(vis, p, 2, (0, 0, 255), -1)

        # Draw snapped points (GREEN)
        snapped_points = set(mapping.values())
        for p in snapped_points:
            cv2.circle(vis, p, 3, (0, 255, 0), -1)

        # Draw movement vectors (BLUE lines)
        for p in original_points:
            new_p = mapping[p]
            if p != new_p:
                cv2.line(vis, p, new_p, (255, 0, 0), 1)

        vis = self._resize(vis)

        cv2.imshow("Point Normalization (Red=Original, Green=Snapped)", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _resize(self, img, max_w=900, max_h=700):
        h, w = img.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        return cv2.resize(img, (int(w * scale), int(h * scale)))