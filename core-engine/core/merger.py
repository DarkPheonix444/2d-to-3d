from typing import List, Tuple, Dict
import numpy as np
import cv2
from collections import Counter

Line = Tuple[Tuple[int, int], Tuple[int, int]]


class MergeSystem:

    def __init__(self, debug=True):
        self.debug = debug
        self.align_tol = None
        self.overlap_tol = None
        self._offset = (0, 0)
        self.last_debug_stats = {}

    # ===================== MAIN =====================

    def merge(self, line_sets: List[List[Line]]) -> List[Dict]:

        self.last_debug_stats = {}

  
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
            self._print_cluster_stats(clusters)

        cluster_diagnostics, cluster_summary = self._analyze_clusters(clusters)
        if self.debug:
            self._print_cluster_diagnostics(cluster_diagnostics, cluster_summary)

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
                print(f"[Merge] median_votes={np.median(vote_vals):.2f}")
                print(
                    "[Merge] vote colors: "
                    "v=1 Red, v=2 Orange, v=3-4 Yellow, v>=5 Green"
                )
                self._print_vote_hist(vote_vals)

            self._print_line_stats("raw", all_lines)
            self._print_line_stats("merged", [d["line"] for d in merged_data])

        # ===================== VISUALIZATION =====================

        if self.debug:
            self._visualize(all_lines, merged_data)

        # ===================== ENDPOINT CONSISTENCY =====================

        points = [p for d in merged_data for p in d["line"]]

        total = len(points)
        unique = len(set(points))

        near_endpoint_stats = self._near_endpoint_cluster_stats(points, self.align_tol)

        self.last_debug_stats = {
            "scale": float(scale),
            "align_tol": float(self.align_tol),
            "overlap_tol": float(self.overlap_tol),
            "input_lines": int(len(all_lines)),
            "clusters_before_length_filter": int(before_filter),
            "merged_count": int(len(merged_data)),
            "removed_short": int(before_filter - len(merged_data)),
            "votes": [int(d["votes"]) for d in merged_data],
            "raw_lengths": [float(self._length(line)) for line in all_lines],
            "merged_lengths": [float(self._length(d["line"])) for d in merged_data],
            "endpoint_total": int(total),
            "endpoint_unique": int(unique),
            "endpoint_duplicates": int(total - unique),
            "near_endpoint_cluster": near_endpoint_stats,
            "cluster_diagnostics": cluster_diagnostics,
            "cluster_summary": cluster_summary,
        }

        if self.debug:
            print("\n========== MERGE ENDPOINT CHECK ==========")
            print(f"total_points={total}")
            print(f"unique_points={unique}")
            print(f"duplicates={total - unique}")
            self._print_endpoint_stats(points)

        return merged_data

    # ===================== SIMILARITY =====================

    def _similar(self, l1: Line, l2: Line) -> bool:
        (x1, y1), (x2, y2) = l1
        (xx1, yy1), (xx2, yy2) = l2

        angle1 = self._angle(l1)
        angle2 = self._angle(l2)

        # --- robust angle check ---
        if self._angle_diff(angle1, angle2) > 20:
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

    def _orientation_bucket(self, line: Line) -> str:
        angle = self._angle(line)
        if angle <= 10 or angle >= 170:
            return "horizontal"
        if 80 <= angle <= 100:
            return "vertical"
        return "diagonal"

    def _print_cluster_stats(self, clusters: List[List[Line]]):
        if not clusters:
            print("[Merge] cluster_sizes: none")
            return

        sizes = [len(cluster) for cluster in clusters]
        print(
            "[Merge] cluster_sizes: "
            f"min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.2f}, median={np.median(sizes):.2f}"
        )
        print(f"[Merge] largest_clusters={sorted(sizes, reverse=True)[:8]}")

    def _analyze_clusters(self, clusters: List[List[Line]]):
        diagnostics = []

        for cluster_id, cluster in enumerate(clusters, start=1):
            if not cluster:
                continue

            orientation = self._cluster_orientation(cluster)
            axis = "x" if orientation == "V" else "y"
            cross_values = []
            spans = []
            intervals = []

            for line in cluster:
                (x1, y1), (x2, y2) = line
                mid_x = (x1 + x2) / 2.0
                mid_y = (y1 + y2) / 2.0

                if axis == "x":
                    cross_values.append(mid_x)
                    span_a = min(y1, y2)
                    span_b = max(y1, y2)
                else:
                    cross_values.append(mid_y)
                    span_a = min(x1, x2)
                    span_b = max(x1, x2)

                spans.extend([span_a, span_b])
                intervals.append((span_a, span_b))

            drift_std = float(np.std(cross_values)) if cross_values else 0.0
            drift_flag = bool(drift_std > float(self.align_tol or 0.0))

            intervals.sort(key=lambda pair: pair[0])
            gaps = []
            overlap_pairs = 0
            for idx in range(len(intervals) - 1):
                curr_start, curr_end = intervals[idx]
                next_start, next_end = intervals[idx + 1]
                gap = next_start - curr_end
                if gap > 0:
                    gaps.append(float(gap))
                if next_start <= curr_end + float(self.overlap_tol or 0.0):
                    overlap_pairs += 1

            gaps_gt_align_tol = int(sum(1 for g in gaps if g > float(self.align_tol or 0.0)))
            fragmented = bool(gaps_gt_align_tol > 0)

            span_min = float(min(spans)) if spans else 0.0
            span_max = float(max(spans)) if spans else 0.0

            votes = [len(cluster)]
            diagnostics.append({
                "cluster_id": int(cluster_id),
                "num_lines": int(len(cluster)),
                "orientation": orientation,
                "axis_drift_std": drift_std,
                "drift_flag": drift_flag,
                "drift_metric": f"std({axis})",
                "span_min": span_min,
                "span_max": span_max,
                "vote_min": int(min(votes) if votes else 0),
                "vote_max": int(max(votes) if votes else 0),
                "vote_avg": float(np.mean(votes) if votes else 0.0),
                "gap_min": float(min(gaps)) if gaps else 0.0,
                "gap_max": float(max(gaps)) if gaps else 0.0,
                "gap_avg": float(np.mean(gaps)) if gaps else 0.0,
                "gap_count": int(len(gaps)),
                "gaps_gt_align_tol": gaps_gt_align_tol,
                "fragmented": fragmented,
                "overlap_pairs": int(overlap_pairs),
            })

        cluster_sizes = [d["num_lines"] for d in diagnostics]
        summary = {
            "total_clusters": int(len(diagnostics)),
            "clusters_flagged_drift": int(sum(1 for d in diagnostics if d["drift_flag"])),
            "clusters_fragmented": int(sum(1 for d in diagnostics if d["fragmented"])),
            "avg_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0.0,
        }

        return diagnostics, summary

    def _cluster_orientation(self, cluster: List[Line]) -> str:
        counts = Counter()
        for line in cluster:
            bucket = self._orientation_bucket(line)
            if bucket == "horizontal":
                counts["H"] += 1
            elif bucket == "vertical":
                counts["V"] += 1

        if counts["H"] >= counts["V"]:
            return "H"
        return "V"

    def _print_cluster_diagnostics(self, diagnostics: List[Dict], summary: Dict):
        if not diagnostics:
            print("\n========== MERGE CLUSTER DIAGNOSTICS ==========")
            print("No clusters available.")
            return

        print("\n========== MERGE CLUSTER DIAGNOSTICS ==========")
        for d in diagnostics:
            axis_label = "x" if d["orientation"] == "V" else "y"
            span_label = "y-range" if d["orientation"] == "V" else "x-range"
            print(
                f"cluster_id={d['cluster_id']} lines={d['num_lines']} orientation={d['orientation']} "
                f"std({axis_label})={d['axis_drift_std']:.2f} span({span_label})=[{d['span_min']:.2f},{d['span_max']:.2f}] "
                f"votes(min/max/avg)={d['vote_min']}/{d['vote_max']}/{d['vote_avg']:.2f} "
                f"gaps(min/max/avg/count/>tol)={d['gap_min']:.2f}/{d['gap_max']:.2f}/{d['gap_avg']:.2f}/{d['gap_count']}/{d['gaps_gt_align_tol']} "
                f"overlap_pairs={d['overlap_pairs']} drift_flag={d['drift_flag']} fragmented={d['fragmented']}"
            )

        print("\n========== MERGE CLUSTER SUMMARY ==========")
        print(f"total_clusters={summary.get('total_clusters', 0)}")
        print(f"clusters_flagged_drift={summary.get('clusters_flagged_drift', 0)}")
        print(f"clusters_fragmented={summary.get('clusters_fragmented', 0)}")
        print(f"avg_cluster_size={summary.get('avg_cluster_size', 0.0):.2f}")

    def _print_vote_hist(self, votes: List[int]):
        if not votes:
            print("[Merge] vote_hist=none")
            return

        vote_hist = Counter(votes)
        ordered = ", ".join(f"{vote}:{vote_hist[vote]}" for vote in sorted(vote_hist))
        print(f"[Merge] vote_hist={ordered}")

    def _print_line_stats(self, name: str, lines: List[Line]):
        if not lines:
            print(f"[Merge] {name}_lengths: none")
            return

        lengths = [self._length(line) for line in lines]
        orientation = Counter(self._orientation_bucket(line) for line in lines)

        print(
            f"[Merge] {name}_lengths: "
            f"min={min(lengths):.2f}, max={max(lengths):.2f}, avg={np.mean(lengths):.2f}, median={np.median(lengths):.2f}"
        )
        print(
            f"[Merge] {name}_orientation: "
            f"h={orientation['horizontal']}, v={orientation['vertical']}, d={orientation['diagonal']}"
        )

    def _print_endpoint_stats(self, points: List[Tuple[int, int]]):
        if not points:
            print("[Merge] endpoint_reuse: none")
            return

        counts = Counter(points)
        reused = [cnt for cnt in counts.values() if cnt > 1]
        if not reused:
            print("[Merge] endpoint_reuse: all endpoints unique")
            return

        print(
            "[Merge] endpoint_reuse: "
            f"reused_points={len(reused)}, max_reuse={max(reused)}, avg_reuse={np.mean(reused):.2f}"
        )

    def _near_endpoint_cluster_stats(self, points: List[Tuple[int, int]], tol: float) -> Dict[str, float]:
        # Group close-by endpoints into local clusters to reveal near-miss connectivity.
        endpoint_clusters = []

        for point in points:
            found = False
            for cluster in endpoint_clusters:
                cx, cy = cluster["center"]
                if abs(point[0] - cx) <= tol and abs(point[1] - cy) <= tol:
                    cluster["points"].append(point)
                    xs = [pt[0] for pt in cluster["points"]]
                    ys = [pt[1] for pt in cluster["points"]]
                    cluster["center"] = (sum(xs) / len(xs), sum(ys) / len(ys))
                    found = True
                    break

            if not found:
                endpoint_clusters.append({
                    "center": point,
                    "points": [point],
                })

        if not endpoint_clusters:
            stats = {
                "num_clusters": 0,
                "avg_cluster_size": 0.0,
                "max_cluster_size": 0,
                "clusters_gt_2": 0,
                "clusters_size_1": 0,
                "clusters_size_2": 0,
                "clusters_size_3_plus": 0,
            }
        else:
            cluster_sizes = [len(cluster["points"]) for cluster in endpoint_clusters]
            stats = {
                "num_clusters": int(len(endpoint_clusters)),
                "avg_cluster_size": float(np.mean(cluster_sizes)),
                "max_cluster_size": int(max(cluster_sizes)),
                "clusters_gt_2": int(sum(1 for size in cluster_sizes if size >= 2)),
                "clusters_size_1": int(sum(1 for size in cluster_sizes if size == 1)),
                "clusters_size_2": int(sum(1 for size in cluster_sizes if size == 2)),
                "clusters_size_3_plus": int(sum(1 for size in cluster_sizes if size >= 3)),
            }

        if self.debug:
            print("\n========== NEAR ENDPOINT CLUSTERS ==========")
            print(f"num_clusters={stats['num_clusters']}")
            print(f"avg_cluster_size={stats['avg_cluster_size']:.2f}")
            print(f"max_cluster_size={stats['max_cluster_size']}")
            print(f"clusters_gt_2={stats['clusters_gt_2']}")
            print("\n========== NEAR CLUSTER DISTRIBUTION ==========")
            print(f"clusters_size_1={stats['clusters_size_1']}")
            print(f"clusters_size_2={stats['clusters_size_2']}")
            print(f"clusters_size_3+={stats['clusters_size_3_plus']}")

        return stats

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
    