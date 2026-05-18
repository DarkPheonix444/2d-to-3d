from typing import List, Tuple, Dict
import numpy as np
from collections import Counter
from collections import defaultdict
import cv2



Point = Tuple[int, int]
Line = Tuple[Point, Point]


class MergeSystemV2:

    def __init__(self, align_tol=5, debug=False):
        self.align_tol = align_tol
        self.debug = debug

    # ---------------------------
    # MAIN ENTRY
    # ---------------------------
    def merge(self, wall_hypotheses, scale):

        # -------- STEP 1: flatten --------
        all_lines = []

        for h in wall_hypotheses:
            if isinstance(h, dict) and "lines" in h:
                all_lines.extend(h["lines"])
            elif isinstance(h, list):
                all_lines.extend(h)
            else:
                raise TypeError(f"Unsupported hypothesis format: {type(h)}")

        if not all_lines:
            if self.debug:
                return [], [], []
            return []

        GAP_THRESHOLD = 0.02 * scale

        # -------- STEP 2: classify + project --------
        horiz = []
        vert = []

        for (x1, y1), (x2, y2) in all_lines:

            if abs(y1 - y2) <= self.align_tol:
                y = int((y1 + y2) / 2)
                s, e = sorted([x1, x2])
                horiz.append((y, s, e))

            elif abs(x1 - x2) <= self.align_tol:
                x = int((x1 + x2) / 2)
                s, e = sorted([y1, y2])
                vert.append((x, s, e))

        # -------- STEP 3: bucket by axis --------
        def bucketize(lines):
            groups = []

            for axis, s, e in lines:
                placed = False

                for g in groups:
                    if abs(axis - g["axis_mean"]) <= self.align_tol:
                        g["lines"].append((axis, s, e))
                        g["axis_vals"].append(axis)
                        g["axis_mean"] = sum(g["axis_vals"]) / len(g["axis_vals"])
                        placed = True
                        break

                if not placed:
                    groups.append({
                        "axis_mean": axis,
                        "axis_vals": [axis],
                        "lines": [(axis, s, e)]
                    })

            return [g["lines"] for g in groups]

        h_groups = bucketize(horiz)
        v_groups = bucketize(vert)

        


        # -------- STEP 4: split by large gaps --------
        def split_by_large_gaps(group):
            group.sort(key=lambda t: t[1])

            subgroups = []
            current = [group[0]]

            for i in range(1, len(group)):
                prev = current[-1]
                curr = group[i]

                gap = curr[1] - prev[2]

                if gap > GAP_THRESHOLD * 2:
                    subgroups.append(current)
                    current = [curr]
                else:
                    current.append(curr)

            subgroups.append(current)
            return subgroups

        # -------- STEP 5: sequential clustering --------
        def cluster_group(group):
            group.sort(key=lambda t: t[1])

            clusters = []
            current = [group[0]]
            cur_end = group[0][2]

            for i in range(1, len(group)):
                axis, s, e = group[i]

                gap = s - cur_end

                if gap <= GAP_THRESHOLD:
                    current.append((axis, s, e))
                    cur_end = max(cur_end, e)
                else:
                    clusters.append(current)
                    current = [(axis, s, e)]
                    cur_end = e

            clusters.append(current)
            return clusters

        h_clusters = []
        v_clusters = []

        # process horizontal
        for group in h_groups:
            subgroups = split_by_large_gaps(group)
            for sg in subgroups:
                h_clusters.extend(cluster_group(sg))

        # process vertical
        for group in v_groups:
            subgroups = split_by_large_gaps(group)
            for sg in subgroups:
                v_clusters.extend(cluster_group(sg))

        # -------- STEP 6: reconstruct lines --------
        merged_data = []

        # HORIZONTAL
      

        for cluster in h_clusters:
                line = self.construct_from_cluster(cluster, "H")
                # line = self.collapse_to_centerline(lines_input)

                length = abs(line[1][0] - line[0][0])

                merged_data.append({
                    "id": len(merged_data),

                    "line": line,

                    "orientation": "H",

                    "axis": line[0][1],

                    "length": length,

                    "support_count": len(cluster),

                    "supporting_segments": cluster,

                    "natural_intersections": [],

                    "neighbors": [],

                    "continuity_links": [],

                    "confidence": float(len(cluster))
                })

        # VERTICAL
       

        for cluster in v_clusters:
            line = self.construct_from_cluster(cluster, "V")
            # line = self.collapse_to_centerline(lines_input)
            length = abs(line[1][1] - line[0][1])

            merged_data.append({
                "id": len(merged_data),

                "line": line,

                "orientation": "V",

                "axis": line[0][0],

                "length": length,

                "support_count": len(cluster),

                "supporting_segments": cluster,

                "natural_intersections": [],

                "neighbors": [],

                "continuity_links": [],

                "confidence": float(len(cluster))
            })
        final_data = self.remove_same_axis_redundancy(merged_data,)
        
        # -------- DEBUG --------



        if self.debug:
            print("\n========== MERGER V2 DEBUG ==========")
            print(f"[MergeV2] input_lines={len(all_lines)}")
            print(f"[MergeV2] horiz_groups={len(h_groups)}")
            print(f"[MergeV2] vert_groups={len(v_groups)}")
            print(f"[MergeV2] horiz_clusters={len(h_clusters)}")
            print(f"[MergeV2] vert_clusters={len(v_clusters)}")
            print(f"[MergeV2] merged_lines={len(merged_data)}")
            print(f"[MergeV2] gap_threshold={GAP_THRESHOLD:.2f}")

            print("\n====== CLUSTER INTERNAL DEBUG ======")

            # def debug_cluster(cluster, orientation, cid):
            #     if orientation == "H":
            #         intervals = [(s, e) for _, s, e in cluster]
            #         axis_vals = [y for y, _, _ in cluster]
            #     else:
            #         intervals = [(s, e) for _, s, e in cluster]
            #         axis_vals = [x for x, _, _ in cluster]

            #     intervals.sort()

            #     print(f"\n[Cluster {cid} | {orientation}]")
            #     print(f"size={len(cluster)} axis_std={np.std(axis_vals):.2f}")

            #     print("intervals:")
            #     for s, e in intervals:
            #         print(f"  [{s:.1f} → {e:.1f}]")

            #     print("gaps:")
            #     for i in range(len(intervals)-1):
            #         gap = intervals[i+1][0] - intervals[i][1]
            #         print(f"  gap[{i}] = {gap:.1f}")

            # cid = 1
            # for c in h_clusters:
            #     debug_cluster(c, "H", cid)
            #     cid += 1

            # for c in v_clusters:
            #     debug_cluster(c, "V", cid)
            #     cid += 1

            # if self.debug:
            #     return final_data, h_clusters, v_clusters

            if self.debug:
                print("\n====== FINAL LINE OVERLAP DEBUG ======")

                def check_overlap(l1, l2):
                    (x1,y1),(x2,y2) = l1
                    (x3,y3),(x4,y4) = l2

                    # horizontal
                    if abs(y1 - y2) < 2 and abs(y3 - y4) < 2:
                        if abs(y1 - y3) <= self.align_tol:
                            overlap = min(x2, x4) - max(x1, x3)
                            return overlap

                    # vertical
                    if abs(x1 - x2) < 2 and abs(x3 - x4) < 2:
                        if abs(x1 - x3) <= self.align_tol:
                            overlap = min(y2, y4) - max(y1, y3)
                            return overlap

                    return 0

                n = len(final_data)

                for i in range(n):
                    for j in range(i+1, n):
                        l1 = final_data[i]["line"]
                        l2 = final_data[j]["line"]

                        overlap = check_overlap(l1, l2)

                        if overlap > 0:
                            print(f"\n⚠ OVERLAP DETECTED:")
                            print(f"  L1: {l1}")
                            print(f"  L2: {l2}")
                            print(f"  overlap = {overlap:.2f}")

            print("\n====== PARALLEL DUPLICATE CHECK ======")

            for i in range(len(final_data)):
                for j in range(i+1, len(final_data)):

                    (x1,y1),(x2,y2) = final_data[i]["line"]
                    (x3,y3),(x4,y4) = final_data[j]["line"]

                    # horizontal
                    if abs(y1 - y2) < 2 and abs(y3 - y4) < 2:
                        axis_diff = abs(y1 - y3)
                        overlap = min(x2, x4) - max(x1, x3)

                        if axis_diff <= self.align_tol and overlap > 0:
                            print(f"\n⚠ PARALLEL DUPLICATE:")
                            print(f"  L1: {final_data[i]['line']}")
                            print(f"  L2: {final_data[j]['line']}")
                            print(f"  axis_diff = {axis_diff}, overlap = {overlap}")

                    # vertical
                    if abs(x1 - x2) < 2 and abs(x3 - x4) < 2:
                        axis_diff = abs(x1 - x3)
                        overlap = min(y2, y4) - max(y1, y3)

                        if axis_diff <= self.align_tol and overlap > 0:
                            print(f"\n⚠ PARALLEL DUPLICATE:")
                            print(f"  L1: {final_data[i]['line']}")
                            print(f"  L2: {final_data[j]['line']}")
                            print(f"  axis_diff = {axis_diff}, overlap = {overlap}")
        if self.debug:
            return final_data, h_clusters, v_clusters

        return final_data
    def collapse_intervals(self,intervals, tol):
            intervals.sort()
            merged = []

            for s, e in intervals:
                if not merged:
                    merged.append([s, e])
                    continue

                ps, pe = merged[-1]

                if s <= pe + tol:
                    merged[-1][1] = max(pe, e)
                else:
                    merged.append([s, e])

            return merged

    def remove_same_axis_redundancy(self, lines):

        used = [False] * len(lines)
        final = []

        for i in range(len(lines)):

            if used[i]:
                continue

            best = lines[i]
            used[i] = True

            l1 = lines[i]["line"]
            o1 = lines[i]["orientation"]

            # ---------- LINE 1 ----------
            if o1 == "H":

                axis1 = l1[0][1]

                s1, e1 = sorted([l1[0][0], l1[1][0]])

            else:

                axis1 = l1[0][0]

                s1, e1 = sorted([l1[0][1], l1[1][1]])

            len1 = e1 - s1

            # ---------- COMPARE ----------
            for j in range(i + 1, len(lines)):

                if used[j]:
                    continue

                l2 = lines[j]["line"]
                o2 = lines[j]["orientation"]

                # orientation mismatch
                if o1 != o2:
                    continue

                # ---------- LINE 2 ----------
                if o2 == "H":

                    axis2 = l2[0][1]

                    # ONLY exact same axis
                    if axis1 != axis2:
                        continue

                    s2, e2 = sorted([l2[0][0], l2[1][0]])

                else:

                    axis2 = l2[0][0]

                    if axis1 != axis2:
                        continue

                    s2, e2 = sorted([l2[0][1], l2[1][1]])

                len2 = e2 - s2

                # ---------- OVERLAP ----------
                overlap = min(e1, e2) - max(s1, s2)

                if overlap <= 0:
                    continue

                overlap_ratio = overlap / min(len1, len2)

                # strong containment / redundancy
                if overlap_ratio >= 0.8:

                    # keep larger representative
                    if len2 > len1:
                        best = lines[j]
                        len1 = len2
                        s1, e1 = s2, e2

                    used[j] = True

            final.append(best)

        return final
    
    def construct_from_cluster(self, cluster, orientation):

        if orientation == "H":

            axis_vals = [y for y, _, _ in cluster]
            intervals = [(s, e) for _, s, e in cluster]

            axis = int(np.median(axis_vals))

            start = min(s for s, _ in intervals)
            end = max(e for _, e in intervals)

            return ((start, axis), (end, axis))

        else:

            axis_vals = [x for x, _, _ in cluster]
            intervals = [(s, e) for _, s, e in cluster]

            axis = int(np.median(axis_vals))

            start = min(s for s, _ in intervals)
            end = max(e for _, e in intervals)

            return ((axis, start), (axis, end))
    

    def collapse_axis_duplicates(self, lines, tol=5):

        used = [False] * len(lines)
        final = []

        for i in range(len(lines)):
            if used[i]:
                continue

            (x1, y1), (x2, y2) = lines[i]["line"]
            best = lines[i]
            used[i] = True

            for j in range(i + 1, len(lines)):
                if used[j]:
                    continue
  
                (x3, y3), (x4, y4) = lines[j]["line"]

                # -------- HORIZONTAL --------
                if abs(y1 - y3) <= tol:
                    overlap = min(x2, x4) - max(x1, x3)

                    if overlap > 0:
                        used[j] = True

                        # choose better representative
                        best_len = abs(best["line"][1][0] - best["line"][0][0])
                        curr_len = abs(x4 - x3)

                        if curr_len > best_len or lines[j]["votes"] > best["votes"]:
                            best = lines[j]

                # -------- VERTICAL --------
                elif abs(x1 - x3) <= tol:
                    overlap = min(y2, y4) - max(y1, y3)

                    if overlap > 0:
                        used[j] = True

                        best_len = abs(best["line"][1][1] - best["line"][0][1])
                        curr_len = abs(y4 - y3)

                        if curr_len > best_len or lines[j]["votes"] > best["votes"]:
                            best = lines[j]

            final.append(best)

        return final
    

   

# ---------------------------
# VISUALIZATION (OUTSIDE CLASS)
# ---------------------------
def visualize_merger_v2(image, h_clusters, v_clusters, merged_data):

    vis = image.copy()

    # Draw clusters (light grey)
    for cluster in h_clusters:
        for y, s, e in cluster:
            cv2.line(vis, (int(s), int(y)), (int(e), int(y)), (200, 200, 200), 1)

    for cluster in v_clusters:
        for x, s, e in cluster:
            cv2.line(vis, (int(x), int(s)), (int(x), int(e)), (200, 200, 200), 1)

    # Draw merged walls
    for d in merged_data:
        (x1, y1), (x2, y2) = d["line"]
        votes = d["support_count"]

        if votes <= 2:
            color = (0, 0, 255)
        elif votes <= 4:
            color = (0, 165, 255)
        elif votes <= 7:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        thickness = 2 if votes < 5 else 3

        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    return vis