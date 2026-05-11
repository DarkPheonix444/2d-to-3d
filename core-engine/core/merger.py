from typing import List, Tuple, Dict
import numpy as np
import cv2
from collections import Counter
from collections import defaultdict

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

        #new approach: use representative-based clustering to group similar lines, then apply purification and voting within clusters
        
        horiz = []
        vert = []

        for (x1, y1), (x2, y2) in all_lines:
            if abs(y1 - y2) <= self.align_tol:
                y = int((y1 + y2) / 2)
                horiz.append((y, min(x1, x2), max(x1, x2)))
            elif abs(x1 - x2) <= self.align_tol:
                x = int((x1 + x2) / 2)
                vert.append((x, min(y1, y2), max(y1, y2)))


       

        h_groups = defaultdict(list)
        for y, x1, x2 in horiz:
            key = int(y / self.align_tol)
            h_groups[key].append((y, x1, x2))

        v_groups = defaultdict(list)
        for x, y1, y2 in vert:
            key = int(x / self.align_tol)
            v_groups[key].append((x, y1, y2))


        h_clusters = []
        v_clusters = []

        GAP_THRESHOLD = 0.02 * scale   # tune later

        # horizontal
        for group in h_groups.values():
            group.sort(key=lambda t: t[1])  # sort by start

            cur_cluster = []
            cur_start, cur_end = None, None

            for y, s, e in group:
                if cur_cluster == []:
                    cur_cluster = [(y, s, e)]
                    cur_start, cur_end = s, e
                    continue

                gap = s - cur_end

                if gap <= GAP_THRESHOLD:
                    cur_cluster.append((y, s, e))
                    cur_end = max(cur_end, e)
                else:
                    h_clusters.append(cur_cluster)
                    cur_cluster = [(y, s, e)]
                    cur_start, cur_end = s, e

            if cur_cluster:
                h_clusters.append(cur_cluster)

        # vertical
        for group in v_groups.values():
            group.sort(key=lambda t: t[1])  # sort by start

            cur_cluster = []
            cur_start, cur_end = None, None

            for x, s, e in group:
                if cur_cluster == []:
                    cur_cluster = [(x, s, e)]
                    cur_start, cur_end = s, e
                    continue

                gap = s - cur_end

                if gap <= GAP_THRESHOLD:
                    cur_cluster.append((x, s, e))
                    cur_end = max(cur_end, e)
                else:
                    v_clusters.append(cur_cluster)
                    cur_cluster = [(x, s, e)]
                    cur_start, cur_end = s, e

            if cur_cluster:
                v_clusters.append(cur_cluster)

        merged_data = []

# ---------------- HORIZONTAL ----------------
        for cluster in h_clusters:
            ys = [y for y, _, _ in cluster]
            intervals = [(s, e) for _, s, e in cluster]

            y = int(np.median(ys))

            intervals.sort()
            s, e = intervals[0]

            for ns, ne in intervals[1:]:
                if ns <= e + GAP_THRESHOLD:
                    e = max(e, ne)
                else:
                    merged_data.append({
                        "line": ((int(s), int(y)), (int(e), int(y))),
                        "votes": len(cluster)
                    })
                    s, e = ns, ne

            merged_data.append({
                "line": ((int(s), int(y)), (int(e), int(y))),
                "votes": len(cluster)
            })


        # ---------------- VERTICAL ----------------
        for cluster in v_clusters:
            xs = [x for x, _, _ in cluster]
            intervals = [(s, e) for _, s, e in cluster]

            x = int(np.median(xs))

            intervals.sort()
            s, e = intervals[0]

            for ns, ne in intervals[1:]:
                if ns <= e + GAP_THRESHOLD:
                    e = max(e, ne)
                else:
                    merged_data.append({
                        "line": ((int(x), int(s)), (int(x), int(e))),
                        "votes": len(cluster)
                    })
                    s, e = ns, ne

            merged_data.append({
                "line": ((int(x), int(s)), (int(x), int(e))),
                "votes": len(cluster)
            })
        
        if self.debug:
            print(f"[Merge] input_lines={len(all_lines)}")
            print(f"[Merge] clusters_before_length_filter={len(merged_data)}")
            print(f"[Merge] horiz_clusters={len(h_clusters)}")
            print(f"[Merge] vert_clusters={len(v_clusters)}")

        cluster_diagnostics, cluster_summary = self.analyze_interval_clusters(h_clusters, v_clusters)
        if self.debug:
            self._print_cluster_diagnostics(cluster_diagnostics, cluster_summary)

                # clusters = []

        # # ===================== CLUSTERING =====================
        # for line in all_lines:
        #     placed = False

        #     for cluster in clusters:

        #         rep = cluster[0]

        #         if self._similar(line, rep):
        #             cluster.append(line)
        #             placed = True
        #             break

        #     if not placed:
        #         clusters.append([line])

        # ===================== BUILD OUTPUT WITH VOTES =====================

        # 'merged_data = []

        # for cluster in clusters:

        #     # --- DRIFT CHECK ---
        #     diag = self._analyze_clusters([cluster])[0][0]
        #     if diag["axis_drift_std"] > self.align_tol * 1.2:
        #         continue

        #     # --- PURIFICATION ---
        #     clean_cluster = self._filter_dominant_band(cluster)

        #     if len(clean_cluster) == 0:
        #         continue

        #     # --- HANDLE SINGLE LINE ---
        #     if len(clean_cluster) == 1:
        #         merged_data.append({
        #             "line": clean_cluster[0],
        #             "votes": 1
        #         })
        #         continue

        #     # --- SPREAD ---
        #     spread = self._axis_spread(clean_cluster)

        #     # --- CLASSIFICATION ---
        #     if spread < 3:
        #         # duplicate collapse
        #         segments = self._build_segments_from_cluster(clean_cluster)

        #     elif spread < self.align_tol * 1.5:

        #         diag = self._analyze_clusters([clean_cluster])[0][0]

        #         if diag["fragmented"]:
        #             # IMPORTANT: do NOT collapse fragmented cluster
        #             segments = self._build_segments_from_cluster(clean_cluster)
        #         else:
        #             # only collapse when truly one segment
        #             segments = self._collapse_to_centerline(clean_cluster)

        #     else:
        #         # impure cluster
        #         continue

        #     if self.debug:
        #         print(f"[Cluster] raw={len(cluster)} clean={len(clean_cluster)} spread={spread:.2f}")

        #     for seg in segments:
        #         merged_data.append({
        #             "line": seg,
        #             "votes": len(clean_cluster)
        #         })


        # ===================== LENGTH FILTER =====================

        min_len = 0.03 * scale

        before_filter = len(merged_data)
        merged_data = [
            d for d in merged_data
            if self._length(d["line"]) >= min_len
        ]

        # ===================== VOTES FILTER =====================

        merged_data = [d for d in merged_data if d["votes"] >= 2]

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
        if self._angle_diff(angle1, angle2) > 8:
            return False

        # --- classify orientation using angle ---
        is_horizontal = (
            abs(angle1 - 0) < 10 or abs(angle1 - 180) < 10
        )
        is_vertical = abs(angle1 - 90) < 10

        # --- horizontal case ---
        if is_horizontal:
            if abs(y1 - yy1) < self.align_tol *0.3:
                overlap = self._overlap(x1, x2, xx1, xx2)
                gap = max(0, max(xx1, xx2) - min(x1, x2), max(x1, x2) - min(xx1, xx2))
                if overlap:
                    return True

                if gap < self.align_tol * 0.5:
                    return True

                return False

        # --- vertical case ---`
        if is_vertical:
            if abs(x1 - xx1) < self.align_tol*0.3:
                overlap = self._overlap(y1, y2, yy1, yy2)
                gap = max(0, max(yy1, yy2) - min(y1, y2), max(y1, y2) - min(yy1, yy2))
                if overlap:
                    return True

                if gap < self.align_tol * 0.5:
                    return True

                return False

        return False

    def _overlap(self, a1, a2, b1, b2):
        return not (
            max(a1, a2) < min(b1, b2) - self.overlap_tol or
            max(b1, b2) < min(a1, a2) - self.overlap_tol
        )

    # ===================== REPRESENTATIVE and  cluster purification  =====================

    def _representative(self, cluster: List[Line]) -> Line:
        return max(
            cluster,
            key=lambda l: (l[1][0] - l[0][0]) ** 2 + (l[1][1] - l[0][1]) ** 2
        )
    
   # def _filter_dominant_band(self, cluster: List[Line]) -> List[Line]:
        orientation = self._cluster_orientation(cluster)

        if orientation == "H":
            vals = [(y1 + y2) / 2 for (x1,y1),(x2,y2) in cluster]
        else:
            vals = [(x1 + x2) / 2 for (x1,y1),(x2,y2) in cluster]

        # scale-aware bin size
        bin_size = max(2, self.align_tol * 0.15)

        bins = {}
        for i, v in enumerate(vals):
            key = int(round(v / bin_size)) * bin_size
            bins.setdefault(key, []).append(cluster[i])

        dominant = max(bins.values(), key=len)
        return dominant
    
    def _collapse_to_centerline(self, cluster: List[Line]) -> List[Line]:
        orientation = self._cluster_orientation(cluster)

        intervals = []

        if orientation == "H":
            ys = [(y1 + y2)/2 for (x1,y1),(x2,y2) in cluster]
            center_y = int(round(np.mean(ys)))

            for (x1,y1),(x2,y2) in cluster:
                intervals.append((min(x1,x2), max(x1,x2)))

            intervals.sort()
            merged = []
            gap_tol = self.align_tol * 1.5
            s, e = intervals[0]

            for ns, ne in intervals[1:]:
                if ns <= e + gap_tol:
                    e = max(e, ne)
                else:
                    merged.append((s, e))
                    s, e = ns, ne
            merged.append((s, e))

            return [((int(s), center_y),(int(e), center_y)) for s,e in merged]

        else:
            xs = [(x1 + x2)/2 for (x1,y1),(x2,y2) in cluster]
            center_x = int(round(np.mean(xs)))

            for (x1,y1),(x2,y2) in cluster:
                intervals.append((min(y1,y2), max(y1,y2)))

            intervals.sort()
            merged = []
            gap_tol = self.align_tol * 1.5
            s, e = intervals[0]

            for ns, ne in intervals[1:]:
                if ns <= e + gap_tol:
                    e = max(e, ne)
                else:
                    merged.append((s, e))
                    s, e = ns, ne
            merged.append((s, e))

            return [((center_x, int(s)),(center_x, int(e))) for s,e in merged]
        

   # def _merge_similar_lines(self, merged_data):

        final = []
        used = [False] * len(merged_data)

        for i, d1 in enumerate(merged_data):
            if used[i]:
                continue

            (x1, y1), (x2, y2) = d1["line"]

            group = [d1]
            used[i] = True

            for j, d2 in enumerate(merged_data):
                if used[j]:
                    continue

                (xx1, yy1), (xx2, yy2) = d2["line"]

                # orientation check
                if self._angle_diff(self._angle(d1["line"]), self._angle(d2["line"])) > 5:
                    continue

                # SAME AXIS (tight)
                if abs(y1 - yy1) < self.align_tol * 0.2 or abs(x1 - xx1) < self.align_tol * 0.2:
                    group.append(d2)
                    used[j] = True

            # merge group
            lines = [g["line"] for g in group]
            votes = sum(g["votes"] for g in group)

            orientation = self._cluster_orientation(lines)

            if orientation == "H":
                y = int(round(np.mean([(l[0][1] + l[1][1]) / 2 for l in lines])))
                xs = [p[0] for l in lines for p in l]

                final.append({
                    "line": ((min(xs), y), (max(xs), y)),
                    "votes": votes
                })

            else:
                x = int(round(np.mean([(l[0][0] + l[1][0]) / 2 for l in lines])))
                ys = [p[1] for l in lines for p in l]

                final.append({
                    "line": ((x, min(ys)), (x, max(ys))),
                    "votes": votes
                })

        return final

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
    

    def _axis_spread(self, cluster):
        orientation = self._cluster_orientation(cluster)

        if orientation == "H":
            vals = [(y1 + y2) / 2 for (x1,y1),(x2,y2) in cluster]
        else:
            vals = [(x1 + x2) / 2 for (x1,y1),(x2,y2) in cluster]

        return max(vals) - min(vals)

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
    

    def _build_segments_from_cluster(self, cluster: List[Line]) -> List[Line]:
        orientation = self._cluster_orientation(cluster)

        intervals = []
        axis_vals = []

        for (x1, y1), (x2, y2) in cluster:
            if orientation == "H":
                intervals.append((min(x1, x2), max(x1, x2)))
                axis_vals.append((y1 + y2) / 2.0)
            else:
                intervals.append((min(y1, y2), max(y1, y2)))
                axis_vals.append((x1 + x2) / 2.0)

        # sort intervals
        intervals.sort(key=lambda x: x[0])

        merged = []
        cur_start, cur_end = intervals[0]
        gap_tol = self.align_tol * 1.5

        for start, end in intervals[1:]:
            if start <= cur_end + gap_tol:
                cur_end = max(cur_end, end)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = start, end

        merged.append((cur_start, cur_end))

        # build lines
        axis = int(round(np.median(axis_vals)))
        lines = []

        for start, end in merged:
            if orientation == "H":
                lines.append(((int(start), axis), (int(end), axis)))
            else:
                lines.append(((axis, int(start)), (axis, int(end))))

        return lines


    def analyze_interval_clusters(self, h_clusters, v_clusters):

        diagnostics = []

        def process(cluster, orientation, cluster_id):
            axis_vals = []
            intervals = []

            if orientation == "H":
                for y, s, e in cluster:
                    axis_vals.append(y)
                    intervals.append((s, e))
                axis_name = "y"
            else:
                for x, s, e in cluster:
                    axis_vals.append(x)
                    intervals.append((s, e))
                axis_name = "x"

            # ---- DRIFT ----
            drift_std = float(np.std(axis_vals))
            drift_flag = drift_std > self.align_tol

            # ---- INTERVAL ANALYSIS ----
            intervals.sort()
            gaps = []

            for i in range(len(intervals) - 1):
                cur_s, cur_e = intervals[i]
                next_s, next_e = intervals[i + 1]

                gap = next_s - cur_e
                if gap > 0:
                    gaps.append(gap)

            fragmented = any(g > self.align_tol for g in gaps)

            span_min = min(s for s, _ in intervals)
            span_max = max(e for _, e in intervals)

            diagnostics.append({
                "cluster_id": cluster_id,
                "orientation": orientation,
                "num_segments": len(cluster),
                "axis": axis_name,
                "axis_drift_std": drift_std,
                "drift_flag": drift_flag,
                "span": (span_min, span_max),
                "gap_count": len(gaps),
                "gap_max": max(gaps) if gaps else 0,
                "gap_avg": float(np.mean(gaps)) if gaps else 0,
                "fragmented": fragmented
            })

        # process horizontal
        cid = 1
        for cluster in h_clusters:
            process(cluster, "H", cid)
            cid += 1

        # process vertical
        for cluster in v_clusters:
            process(cluster, "V", cid)
            cid += 1

        # ---- SUMMARY ----
        summary = {
            "total_clusters": len(diagnostics),
            "fragmented_clusters": sum(1 for d in diagnostics if d["fragmented"]),
            "drifted_clusters": sum(1 for d in diagnostics if d["drift_flag"]),
            "avg_cluster_size": np.mean([d["num_segments"] for d in diagnostics]) if diagnostics else 0
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

    def _print_cluster_diagnostics(self, diagnostics, summary):
        print("\n========== MERGE CLUSTER DIAGNOSTICS ==========")

        for d in diagnostics:
            axis = "x" if d["orientation"] == "V" else "y"

            span = d.get("span", (0, 0))

            print(
                f"cluster_id={d['cluster_id']} "
                f"num_segments={d['num_segments']} "
                f"orientation={d['orientation']} "
                f"std({axis})={d['axis_drift_std']:.2f} "
                f"span=[{span[0]:.2f},{span[1]:.2f}] "
                f"gap_max={d['gap_max']:.2f} "
                f"gap_avg={d['gap_avg']:.2f} "
                f"gap_count={d['gap_count']} "
                f"fragmented={d['fragmented']} "
                f"drift_flag={d['drift_flag']}"
            )

        print("\n========== MERGE CLUSTER SUMMARY ==========")
        print(f"total_clusters={summary.get('total_clusters', 0)}")
        print(f"fragmented_clusters={summary.get('fragmented_clusters', 0)}")
        print(f"drifted_clusters={summary.get('drifted_clusters', 0)}")
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
    