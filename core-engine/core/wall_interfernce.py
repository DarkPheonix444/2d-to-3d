import cv2
import numpy as np


class ThicknessInference:

    def __init__(
        self,
        axis_tol=8,
        overlap_ratio_thresh=0.6,
        debug=True
    ):

        self.axis_tol = axis_tol
        self.overlap_ratio_thresh = overlap_ratio_thresh
        self.debug = debug

    # ==================================================
    # MAIN
    # ==================================================

    def infer_walls(self, edge_candidates):

        used = set()
        consumed_wall=set()

        wall_objects = []

        wall_id = 0

        if self.debug:

            print("\n========== THICKNESS INFERENCE ==========")

            print(f"[ThicknessInference] input_edges={len(edge_candidates)}")

        # --------------------------------------------------
        # MATCH PARALLEL EDGES
        # --------------------------------------------------

        for i in range(len(edge_candidates)):

            if i in used:
                continue

            e1 = edge_candidates[i]

            best_match = None
            best_score = 0

            for j in range(i + 1, len(edge_candidates)):

                if j in used:
                    continue

                e2 = edge_candidates[j]

                # ------------------------------------------
                # ORIENTATION CHECK
                # ------------------------------------------

                if e1["orientation"] != e2["orientation"]:
                    continue

                orientation = e1["orientation"]

                # ==========================================
                # HORIZONTAL
                # ==========================================

                if orientation == "H":

                    axis_diff = abs(
                        e1["axis"] - e2["axis"]
                    )

                    if axis_diff > self.axis_tol:
                        continue

                    s1, e1x = sorted([
                        e1["line"][0][0],
                        e1["line"][1][0]
                    ])

                    s2, e2x = sorted([
                        e2["line"][0][0],
                        e2["line"][1][0]
                    ])

                    overlap = min(e1x, e2x) - max(s1, s2)

                    if overlap <= 0:
                        continue

                # ==========================================
                # VERTICAL
                # ==========================================

                else:

                    axis_diff = abs(
                        e1["axis"] - e2["axis"]
                    )

                    if axis_diff > self.axis_tol:
                        continue

                    s1, e1y = sorted([
                        e1["line"][0][1],
                        e1["line"][1][1]
                    ])

                    s2, e2y = sorted([
                        e2["line"][0][1],
                        e2["line"][1][1]
                    ])

                    overlap = min(e1y, e2y) - max(s1, s2)

                    if overlap <= 0:
                        continue

                # ==========================================
                # OVERLAP RATIO
                # ==========================================

                len1 = e1["length"]
                len2 = e2["length"]

                overlap_ratio = overlap / min(len1, len2)

                if overlap_ratio < self.overlap_ratio_thresh:
                    continue

                # ==========================================
                # SCORE
                # ==========================================

                score = (
                    overlap_ratio *
                    (
                        (e1["support_count"] +
                         e2["support_count"]) / 2
                    )
                ) / (axis_diff + 1)

                # ==========================================
                # DEBUG
                # ==========================================

                if self.debug:

                    print("\n------ PARALLEL CANDIDATE ------")

                    print(f"edge_a        : {e1['id']}")
                    print(f"edge_b        : {e2['id']}")

                    print(f"orientation   : {orientation}")

                    print(f"axis_diff     : {axis_diff}")

                    print(f"overlap       : {overlap:.2f}")

                    print(f"overlap_ratio : {overlap_ratio:.3f}")

                    print(f"score          : {score:.3f}")

                # ==========================================
                # BEST MATCH
                # ==========================================

                if score > best_score:

                    best_score = score

                    best_match = j

            # --------------------------------------------------
            # CREATE WALL OBJECT
            # --------------------------------------------------

            if best_match is not None:

                e2 = edge_candidates[best_match]

                # ==========================================
                # HORIZONTAL WALL
                # ==========================================

                if e1["orientation"] == "H":

                    center_y = int(round(
                        (e1["axis"] + e2["axis"]) / 2
                    ))

                    x1 = min(
                        e1["line"][0][0],
                        e1["line"][1][0],
                        e2["line"][0][0],
                        e2["line"][1][0]
                    )

                    x2 = max(
                        e1["line"][0][0],
                        e1["line"][1][0],
                        e2["line"][0][0],
                        e2["line"][1][0]
                    )

                    centerline = (
                        (x1, center_y),
                        (x2, center_y)
                    )

                # ==========================================
                # VERTICAL WALL
                # ==========================================

                else:

                    center_x = int(round(
                        (e1["axis"] + e2["axis"]) / 2
                    ))

                    y1 = min(
                        e1["line"][0][1],
                        e1["line"][1][1],
                        e2["line"][0][1],
                        e2["line"][1][1]
                    )

                    y2 = max(
                        e1["line"][0][1],
                        e1["line"][1][1],
                        e2["line"][0][1],
                        e2["line"][1][1]
                    )

                    centerline = (
                        (center_x, y1),
                        (center_x, y2)
                    )

                # ==========================================
                # WALL OBJECT
                # ==========================================

                wall_objects.append({

                    "wall_id": wall_id,

                    "centerline": centerline,

                    "orientation": e1["orientation"],

                    "thickness": abs(
                        e1["axis"] - e2["axis"]
                    ),

                    "edge_ids": [
                        e1["id"],
                        e2["id"]
                    ],

                    "confidence": best_score,

                    "generator": "thickness_inference",

                    "parent_ids": [],

                    "consumed": False
                })

                used.add(i)

                used.add(best_match)

                wall_id += 1

        # ==================================================
        # DEBUG SUMMARY
        # ==================================================

        if self.debug:

            print("\n========== WALL SUMMARY ==========")

            print(f"[ThicknessInference] inferred_walls={len(wall_objects)}")

            for w in wall_objects:

                print(f"\n[Wall {w['wall_id']}]")

                print(f"  orientation : {w['orientation']}")

                print(f"  thickness   : {w['thickness']}")

                print(f"  confidence  : {w['confidence']:.3f}")

                print(f"  edge_ids    : {w['edge_ids']}")

                print(f"  centerline  : {w['centerline']}")

        reconstructed_walls = self.consolidate_wall_chains(
    wall_objects,
    consumed_wall
)

        final_walls = []

        # ------------------------------------------
        # KEEP UNCONSUMED ORIGINAL WALLS
        # ------------------------------------------

        for w in wall_objects:

            if w["wall_id"] not in consumed_wall:
                final_walls.append(w)

        # ------------------------------------------
        # ADD RECONSTRUCTED WALLS
        # ------------------------------------------

        final_walls.extend(reconstructed_walls)

        if self.debug:

            print("\n========== FINAL WALL SET ==========")

            print(f"[ThicknessInference] final_walls={len(final_walls)}")

        return final_walls
    
    def _merge_intervals(self, intervals, gap_tol=6):

        if not intervals:
            return []

        intervals.sort(key=lambda x: x[0])

        merged = [list(intervals[0])]

        for start, end in intervals[1:]:

            prev_start, prev_end = merged[-1]

            # overlap OR tiny gap
            if start <= prev_end + gap_tol:
                merged[-1][1] = max(prev_end, end)

            else:
                merged.append([start, end])

        return merged
    
    def consolidate_wall_chains(
    self,
    wall_objects,
    consumed_wall):

        reconstructed = []

        used = set()

        new_id = 10000

        for i in range(len(wall_objects)):

            if i in used:
                continue

            w1 = wall_objects[i]

            chain = [w1]
            used.add(i)

            for j in range(i + 1, len(wall_objects)):

                if j in used:
                    continue

                w2 = wall_objects[j]

                # --------------------------------------
                # ORIENTATION
                # --------------------------------------

                if w1["orientation"] != w2["orientation"]:
                    continue

                # --------------------------------------
                # HORIZONTAL
                # --------------------------------------

                if w1["orientation"] == "H":

                    y1 = w1["centerline"][0][1]
                    y2 = w2["centerline"][0][1]

                    if abs(y1 - y2) > self.axis_tol:
                        continue

                    x1s = min(
                        w1["centerline"][0][0],
                        w1["centerline"][1][0]
                    )

                    x1e = max(
                        w1["centerline"][0][0],
                        w1["centerline"][1][0]
                    )

                    x2s = min(
                        w2["centerline"][0][0],
                        w2["centerline"][1][0]
                    )

                    x2e = max(
                        w2["centerline"][0][0],
                        w2["centerline"][1][0]
                    )

                    gap = min(
                        abs(x1e - x2s),
                        abs(x2e - x1s)
                    )

                    overlap = min(x1e, x2e) - max(x1s, x2s)

                    if gap > 40 and overlap <= 0:
                        continue

                # --------------------------------------
                # VERTICAL
                # --------------------------------------

                else:

                    x1 = w1["centerline"][0][0]
                    x2 = w2["centerline"][0][0]

                    if abs(x1 - x2) > self.axis_tol:
                        continue

                    y1s = min(
                        w1["centerline"][0][1],
                        w1["centerline"][1][1]
                    )

                    y1e = max(
                        w1["centerline"][0][1],
                        w1["centerline"][1][1]
                    )

                    y2s = min(
                        w2["centerline"][0][1],
                        w2["centerline"][1][1]
                    )

                    y2e = max(
                        w2["centerline"][0][1],
                        w2["centerline"][1][1]
                    )

                    gap = min(
                        abs(y1e - y2s),
                        abs(y2e - y1s)
                    )

                    overlap = min(y1e, y2e) - max(y1s, y2s)

                    if gap > 40 and overlap <= 0:
                        continue

                # --------------------------------------
                # CHAIN MATCH
                # --------------------------------------
                thickness_diff = abs(w1["thickness"] - w2["thickness"])

                if thickness_diff > 4:
                    continue
                chain.append(w2)
                
                used.add(j)

            # ------------------------------------------
            # RECONSTRUCT CHAIN
            # ------------------------------------------

            if len(chain) <= 1:
                continue

            consumed_ids = []

            orientation = chain[0]["orientation"]

            if orientation == "H":

                ys = [
                    w["centerline"][0][1]
                    for w in chain
                ]

                center_y = int(round(np.mean(ys)))

                intervals = []

                for w in chain:

                    x1 = min(
                        w["centerline"][0][0],
                        w["centerline"][1][0]
                    )

                    x2 = max(
                        w["centerline"][0][0],
                        w["centerline"][1][0]
                    )

                    intervals.append((x1, x2))

                merged_intervals = self._merge_intervals(
                    intervals,
                    gap_tol=self.axis_tol
                )

                # choose longest merged interval
                best = max(
                    merged_intervals,
                    key=lambda x: x[1] - x[0]
                )

                centerline = (
                    (int(best[0]), center_y),
                    (int(best[1]), center_y)
                )

            else:

                xs = [
                    w["centerline"][0][0]
                    for w in chain
                ]

                center_x = int(round(np.mean(xs)))

                intervals = []

                for w in chain:

                    y1 = min(
                        w["centerline"][0][1],
                        w["centerline"][1][1]
                    )

                    y2 = max(
                        w["centerline"][0][1],
                        w["centerline"][1][1]
                    )

                    intervals.append((y1, y2))

                merged_intervals = self._merge_intervals(
                    intervals,
                    gap_tol=self.axis_tol
                )

                best = max(
                    merged_intervals,
                    key=lambda x: x[1] - x[0]
                )

                centerline = (
                    (center_x, int(best[0])),
                    (center_x, int(best[1]))
                )

            thickness = int(round(np.mean([
                w["thickness"]
                for w in chain
            ])))

            confidence = np.mean([
                w["confidence"]
                for w in chain
            ])

            for w in chain:
                w["consumed"] = True
                consumed_wall.add(
                    w["wall_id"]
                )

                consumed_ids.append(
                    w["wall_id"]
                )
            merged_edge_ids = []

            for wall in chain:
                merged_edge_ids.extend(
                    wall["edge_ids"]
                )
            reconstructed.append({

                "wall_id": new_id,

                "centerline": centerline,

                "orientation": orientation,

                "thickness": thickness,

                "confidence": confidence,

                "parent_ids": consumed_ids,

                "edge_ids": merged_edge_ids,

                "generator": "chain_reconstruction",

                "reconstructed": True,

                "consumed": False
            })

            if self.debug:

                print("\n====== RECONSTRUCTED WALL ======")

                print(f"new_wall_id : {new_id}")

                print(f"orientation : {orientation}")

                print(f"source_walls: {consumed_ids}")

                print(f"centerline  : {centerline}")

            new_id += 1

        return reconstructed

    # ==================================================
    # VISUALIZATION
    # ==================================================

    def visualize(
        self,
        image,
        edge_candidates,
        wall_objects
    ):

        vis = image.copy()

        # --------------------------------------------------
        # DRAW WALL OBJECTS
        # --------------------------------------------------

        edge_lookup = {
            e["id"]: e
            for e in edge_candidates
        }

        for w in wall_objects:

            # ------------------------------------------
            # DRAW SOURCE EDGES
            # ------------------------------------------

            if "edge_ids" in w:

                edge_a = edge_lookup.get(w["edge_ids"][0])
                edge_b = edge_lookup.get(w["edge_ids"][1])

                if edge_a:
                    (x1,y1),(x2,y2) = edge_a["line"]

                    cv2.line(
                        vis,
                        (int(x1),int(y1)),
                        (int(x2),int(y2)),
                        (255,0,0),      # BLUE
                        2
                    )

                if edge_b:
                    (x1,y1),(x2,y2) = edge_b["line"]

                    cv2.line(
                        vis,
                        (int(x1),int(y1)),
                        (int(x2),int(y2)),
                        (0,0,255),      # RED
                        2
                    )

            # ------------------------------------------
            # DRAW CENTERLINE
            # ------------------------------------------

            (x1,y1),(x2,y2) = w["centerline"]

            cv2.line(
                vis,
                (int(x1),int(y1)),
                (int(x2),int(y2)),
                (0,255,0),             # GREEN
                2
            )

            # ------------------------------------------
            # LABEL
            # ------------------------------------------

            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)

            label = (
                f"{w['wall_id']} "
                f"T:{w['thickness']}"
            )

            cv2.putText(
                vis,
                label,
                (cx,cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0,255,255),
                1
            )
            

        h, w = vis.shape[:2]

        scale = min(
            1400 / w,
            900 / h,
            1.0
        )

        vis = cv2.resize(
            vis,
            (
                int(w * scale),
                int(h * scale)
            )
        )

     
        cv2.imshow(
            "Thickness Inference",
            vis
        )

        cv2.waitKey(0)

        cv2.destroyAllWindows()