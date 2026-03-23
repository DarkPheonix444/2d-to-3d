from typing import List, Tuple
from collections import defaultdict
import cv2

Line = Tuple[Tuple[int, int], Tuple[int, int]]


class LineMerger:

    def __init__(self, gap=15, align_tol=10, debug=True):
        self.gap = gap
        self.align_tol = align_tol
        self.debug = debug

    # ===================== CLUSTER =====================

    def _cluster_keys(self, keys):
        clusters = []
        for k in sorted(keys):
            placed = False
            for cluster in clusters:
                if abs(cluster[0] - k) <= self.align_tol:
                    cluster.append(k)
                    placed = True
                    break
            if not placed:
                clusters.append([k])
        return clusters

    # ===================== REMOVE PARALLEL =====================

    def _remove_parallel_duplicates(self, lines):
        final = []
        used = [False] * len(lines)

        for i in range(len(lines)):
            if used[i]:
                continue

            (x1, y1), (x2, y2) = lines[i]

            group = [lines[i]]
            used[i] = True

            for j in range(i + 1, len(lines)):
                if used[j]:
                    continue

                (xx1, yy1), (xx2, yy2) = lines[j]

                # horizontal
                if abs(y1 - y2) <= self.align_tol and abs(yy1 - yy2) <= self.align_tol:
                    if abs(y1 - yy1) <= self.align_tol:
                        group.append(lines[j])
                        used[j] = True

                # vertical
                elif abs(x1 - x2) <= self.align_tol and abs(xx1 - xx2) <= self.align_tol:
                    if abs(x1 - xx1) <= self.align_tol:
                        group.append(lines[j])
                        used[j] = True

            # pick longest
            best = max(group, key=lambda l: (l[1][0]-l[0][0])**2 + (l[1][1]-l[0][1])**2)
            final.append(best)

        return final

    # ===================== VISUAL =====================

    def _show(self, img, lines, title="merged"):
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for (x1, y1), (x2, y2) in lines:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow(title, vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ===================== MAIN =====================

    def merge(self, lines: List[Line], img=None) -> List[Line]:

        horiz = defaultdict(list)
        vert = defaultdict(list)

        # ---- GROUP ----
        for (x1, y1), (x2, y2) in lines:

            if abs(y1 - y2) <= self.align_tol:
                y = int((y1 + y2) / 2)
                horiz[y].append((min(x1, x2), max(x1, x2)))

            elif abs(x1 - x2) <= self.align_tol:
                x = int((x1 + x2) / 2)
                vert[x].append((min(y1, y2), max(y1, y2)))

        merged = []

        # ---- HORIZONTAL ----
        for cluster in self._cluster_keys(horiz.keys()):
            segs = []
            y_avg = int(sum(cluster) / len(cluster))

            for y in cluster:
                segs.extend(horiz[y])

            segs.sort()
            s, e = segs[0]

            for ns, ne in segs[1:]:
                if ns <= e + self.gap:
                    e = max(e, ne)
                else:
                    merged.append(((s, y_avg), (e, y_avg)))
                    s, e = ns, ne

            merged.append(((s, y_avg), (e, y_avg)))

        # ---- VERTICAL ----
        for cluster in self._cluster_keys(vert.keys()):
            segs = []
            x_avg = int(sum(cluster) / len(cluster))

            for x in cluster:
                segs.extend(vert[x])

            segs.sort()
            s, e = segs[0]

            for ns, ne in segs[1:]:
                if ns <= e + self.gap:
                    e = max(e, ne)
                else:
                    merged.append(((x_avg, s), (x_avg, e)))
                    s, e = ns, ne

            merged.append(((x_avg, s), (x_avg, e)))

        # 🔥 KEY FIX
        merged = self._remove_parallel_duplicates(merged)

        # ---- VISUAL ----
        if self.debug and img is not None and len(merged) > 0:
            self._show(img, merged, "merged_final")

        return merged