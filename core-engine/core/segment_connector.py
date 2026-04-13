import numpy as np
from collections import defaultdict
import cv2




class SegmentConnector:

        def __init__(self, debug=True):
            self.debug = debug
            self.align_tol = None
            self.gap_tol = None

        def connect(self, merged_data):
            if not merged_data:
                return []

            # ---- compute scale ----
            pts = [p for d in merged_data for p in d["line"]]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]

            scale = np.hypot(max(xs) - min(xs), max(ys) - min(ys))

            self.align_tol = 0.005 * scale
            self.gap_tol = 0.015 * scale

            # ---- classify ----
            horizontal = []
            vertical = []
            passthrough = []

            for d in merged_data:
                (x1, y1), (x2, y2) = d["line"]
                votes = int(d.get("votes", 1))

                if abs(y1 - y2) <= self.align_tol:
                    y = int((y1 + y2) / 2)
                    horizontal.append((y, min(x1, x2), max(x1, x2), votes))

                elif abs(x1 - x2) <= self.align_tol:
                    x = int((x1 + x2) / 2)
                    vertical.append((x, min(y1, y2), max(y1, y2), votes))
                else:
                    passthrough.append({
                        "line": ((int(x1), int(y1)), (int(x2), int(y2))),
                        "length": float(np.hypot(x2 - x1, y2 - y1)),
                        "orientation": "D",
                        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                        "votes": votes,
                    })

            # ---- group by axis ----
            h_groups = defaultdict(list)
            for y, x1, x2, votes in horizontal:
                key = int(y / self.align_tol)
                h_groups[key].append((y, x1, x2, votes))

            v_groups = defaultdict(list)
            for x, y1, y2, votes in vertical:
                key = int(x / self.align_tol)
                v_groups[key].append((x, y1, y2, votes))

            result = []

            # ---- merge horizontal ----
            for group in h_groups.values():
                if not group:
                    continue

                group.sort(key=lambda x: x[1])
                cur_y, cur_start, cur_end, cur_votes = group[0]
                vote_bucket = [cur_votes]

                for y, s, e, votes in group[1:]:
                    gap = s - cur_end

                    if gap <= self.gap_tol and s <= cur_end + self.gap_tol:
                        cur_end = max(cur_end, e)
                        vote_bucket.append(votes)
                    else:
                        result.append(self._build_wall((cur_start, cur_y), (cur_end, cur_y), "H", votes=max(vote_bucket)))
                        cur_y, cur_start, cur_end, cur_votes = y, s, e, votes
                        vote_bucket = [cur_votes]

                result.append(self._build_wall((cur_start, cur_y), (cur_end, cur_y), "H", votes=max(vote_bucket)))

            # ---- merge vertical ----
            for group in v_groups.values():
                if not group:
                    continue

                group.sort(key=lambda x: x[1])
                cur_x, cur_start, cur_end, cur_votes = group[0]
                vote_bucket = [cur_votes]

                for x, s, e, votes in group[1:]:
                    gap = s - cur_end

                    if gap <= self.gap_tol and s <= cur_end + self.gap_tol:
                        cur_end = max(cur_end, e)
                        vote_bucket.append(votes)
                    else:
                        result.append(self._build_wall((cur_x, cur_start), (cur_x, cur_end), "V", votes=max(vote_bucket)))
                        cur_x, cur_start, cur_end, cur_votes = x, s, e, votes
                        vote_bucket = [cur_votes]

                result.append(self._build_wall((cur_x, cur_start), (cur_x, cur_end), "V", votes=max(vote_bucket)))

            result.extend(passthrough)

            # ---- stats ----
            if self.debug and result:
                lengths = [r["length"] for r in result]
                print(f"[SegmentConnector] in={len(merged_data)}, out={len(result)}")
                print(f"[SegmentConnector] avg={np.mean(lengths):.2f}, median={np.median(lengths):.2f}")
                if passthrough:
                    print(f"[SegmentConnector] passthrough_non_orthogonal={len(passthrough)}")

            return result

        # ---- helper ----
        def _build_wall(self, p1, p2, orientation, votes=1):
            (x1, y1), (x2, y2) = p1, p2

            length = abs(x2 - x1) if orientation == "H" else abs(y2 - y1)

            return {
                "line": ((int(x1), int(y1)), (int(x2), int(y2))),
                "length": float(length),
                "orientation": orientation,
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                "votes": int(votes),
            }
        def visualize_segment_connector(self, base_img, merged_data, connected_data):

            # convert to BGR if needed
            if len(base_img.shape) == 2:
                vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
            else:
                vis = base_img.copy()

            # ---- draw BEFORE (merged) in gray ----
            for d in merged_data:
                (x1, y1), (x2, y2) = d["line"]
                cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (120,120,120), 1)

            # ---- draw AFTER (connected) in green ----
            for d in connected_data:
                (x1, y1), (x2, y2) = d["line"]
                cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

            # resize for display
            h, w = vis.shape[:2]
            scale = min(900 / w, 700 / h, 1.0)
            vis = cv2.resize(vis, (int(w*scale), int(h*scale)))

            cv2.imshow("Segment Connector Debug", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            