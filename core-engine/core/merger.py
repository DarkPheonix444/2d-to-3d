from typing import List, Tuple, Dict
import numpy as np
import cv2

Line = Tuple[Tuple[int, int], Tuple[int, int]]


class MergeSystem:

    def __init__(self, debug=True):
        self.debug = debug
        self.align_tol = None
        self.overlap_tol = None
        self._offset = (0, 0)

    # ===================== MAIN =====================

    def merge(self, line_sets: List[List[Line]]) -> List[Dict]:

  
        # ---- SCALE COMPUTATION ----
        all_points = [p for s in line_sets for l in s for p in l]

        if not all_points:
            return []

        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]

        scale = np.hypot(max(xs) - min(xs), max(ys) - min(ys))

        # ---- SCALE-AWARE TOLERANCE ----
        self.align_tol = 0.01 * scale
        self.overlap_tol = 0.01 * scale

        if self.debug:
            print(f"[Merge] scale={scale:.2f}, tol={self.align_tol:.2f}")

        # ---- FLATTEN ----
        all_lines = [l for s in line_sets for l in s]

        clusters = []

        # ===================== CLUSTERING =====================
        for line in all_lines:
            placed = False

            for cluster in clusters:
                if any(self._similar(line, l) for l in cluster):
                    cluster.append(line)
                    placed = True
                    break

            if not placed:
                clusters.append([line])

        # ===================== BUILD OUTPUT WITH VOTES =====================

        merged_data = []

        for cluster in clusters:
            rep = self._representative(cluster)

            merged_data.append({
                "line": rep,
                "votes": len(cluster)
            })

        if self.debug:
            print(f"[Merge] input_lines={len(all_lines)}")
            print(f"[Merge] clusters_before_length_filter={len(merged_data)}")

        # ===================== LENGTH FILTER =====================

        min_len = 0.03 * scale

        before_filter = len(merged_data)
        merged_data = [
            d for d in merged_data
            if self._length(d["line"]) >= min_len
        ]

        if self.debug:
            removed_short = before_filter - len(merged_data)
            vote_vals = [d["votes"] for d in merged_data]
            if vote_vals:
                print(f"[Merge] min_len={min_len:.2f}")
                print(f"[Merge] removed_short={removed_short}")
                print(
                    f"[Merge] votes: min={min(vote_vals)}, max={max(vote_vals)}, avg={np.mean(vote_vals):.2f}"
                )
                print(
                    "[Merge] vote colors: "
                    "v=1 Red, v=2 Orange, v=3-4 Yellow, v>=5 Green"
                )

        # ===================== VISUALIZATION =====================

        if self.debug:
            self._visualize(all_lines, merged_data)

        return merged_data

    # ===================== SIMILARITY =====================

    def _similar(self, l1: Line, l2: Line) -> bool:
        (x1, y1), (x2, y2) = l1
        (xx1, yy1), (xx2, yy2) = l2

        angle1 = self._angle(l1)
        angle2 = self._angle(l2)

        # --- robust angle check ---
        if self._angle_diff(angle1, angle2) > 10:
            return False

        # --- classify orientation using angle ---
        is_horizontal = (
            abs(angle1 - 0) < 10 or abs(angle1 - 180) < 10
        )
        is_vertical = abs(angle1 - 90) < 10

        # --- horizontal case ---
        if is_horizontal:
            if abs(y1 - yy1) < self.align_tol:
                return self._overlap(x1, x2, xx1, xx2)

        # --- vertical case ---
        if is_vertical:
            if abs(x1 - xx1) < self.align_tol:
                return self._overlap(y1, y2, yy1, yy2)

        return False

    def _overlap(self, a1, a2, b1, b2):
        return not (
            max(a1, a2) < min(b1, b2) - self.overlap_tol or
            max(b1, b2) < min(a1, a2) - self.overlap_tol
        )

    # ===================== REPRESENTATIVE =====================

    def _representative(self, cluster: List[Line]) -> Line:
        return max(
            cluster,
            key=lambda l: (l[1][0] - l[0][0]) ** 2 + (l[1][1] - l[0][1]) ** 2
        )

    # ===================== LENGTH =====================

    def _length(self, l: Line):
        (x1, y1), (x2, y2) = l
        return np.hypot(x2 - x1, y2 - y1)
    
    def _angle(self, l):
        (x1, y1), (x2, y2) = l
        return abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180


    def _angle_diff(self, a, b):
        diff = abs(a - b) % 180
        return min(diff, 180 - diff)

    # ===================== VISUALIZATION =====================

    def _visualize(self, all_lines, merged_data):

        canvas = self._create_canvas(all_lines)
        vis = canvas.copy()

        ox, oy = self._offset

        # ALL lines (gray)
        for (x1, y1), (x2, y2) in all_lines:
            cv2.line(vis, (x1 - ox, y1 - oy), (x2 - ox, y2 - oy), (100, 100, 100), 1)

        # MERGED (color based on votes)
        for d in merged_data:
            (x1, y1), (x2, y2) = d["line"]
            votes = d["votes"]

            color, _ = self._vote_color(votes)

            cv2.line(vis, (x1 - ox, y1 - oy), (x2 - ox, y2 - oy), color, 2)

        self._draw_legend(vis)

        vis = self._resize_for_display(vis)

        cv2.imshow("Merge Debug (Votes)", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _vote_color(self, votes: int):
        """Return BGR color and label for vote-strength bucket."""
        if votes <= 1:
            return (0, 0, 255), "v=1 weak"
        if votes == 2:
            return (0, 140, 255), "v=2 medium"
        if votes <= 4:
            return (0, 255, 255), "v=3-4 strong"
        return (0, 255, 0), "v>=5 very strong"

    def _draw_legend(self, vis):
        """Draw color legend directly on merge debug image."""
        items = [
            ((100, 100, 100), "Gray: all raw detected lines"),
            ((0, 0, 255), "Red: vote=1 (weak, single detector)"),
            ((0, 140, 255), "Orange: vote=2"),
            ((0, 255, 255), "Yellow: vote=3-4"),
            ((0, 255, 0), "Green: vote>=5 (very stable)"),
        ]

        x = 12
        y = 20
        line_h = 22

        for color, text in items:
            cv2.line(vis, (x, y), (x + 28, y), color, 4)
            cv2.putText(
                vis,
                text,
                (x + 36, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            y += line_h

    def _create_canvas(self, lines):
        xs = [p[0] for l in lines for p in l]
        ys = [p[1] for l in lines for p in l]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        w = max_x - min_x + 50
        h = max_y - min_y + 50

        self._offset = (min_x - 25, min_y - 25)

        return np.zeros((h, w, 3), dtype=np.uint8)

    def _resize_for_display(self, img, max_w=900, max_h=700):
        h, w = img.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    