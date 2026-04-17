import numpy as np
import cv2


def collapse_wall_thickness(lines, tol=10):

    used = [False] * len(lines)
    result = []

    for i in range(len(lines)):
        if used[i]:
            continue

        (x1, y1), (x2, y2) = lines[i]

        paired = False

        for j in range(i + 1, len(lines)):
            if used[j]:
                continue

            (x3, y3), (x4, y4) = lines[j]

            # both vertical
            if abs(x1 - x2) <= tol and abs(x3 - x4) <= tol:

                if abs(x1 - x3) <= tol * 3:

                    seg1 = sorted([y1, y2])
                    seg2 = sorted([y3, y4])

                    overlap = min(seg1[1], seg2[1]) - max(seg1[0], seg2[0])

                    if overlap > tol:
                        x_mid = int((x1 + x3) / 2)

                        result.append(((x_mid, overlap + max(seg1[0], seg2[0])),
                                       (x_mid, overlap + min(seg1[1], seg2[1]))))

                        used[i] = used[j] = True
                        paired = True
                        break

            # both horizontal
            elif abs(y1 - y2) <= tol and abs(y3 - y4) <= tol:

                if abs(y1 - y3) <= tol * 3:

                    seg1 = sorted([x1, x2])
                    seg2 = sorted([x3, x4])

                    overlap = min(seg1[1], seg2[1]) - max(seg1[0], seg2[0])

                    if overlap > tol:
                        y_mid = int((y1 + y3) / 2)

                        result.append(((overlap + max(seg1[0], seg2[0]), y_mid),
                                       (overlap + min(seg1[1], seg2[1]), y_mid)))

                        used[i] = used[j] = True
                        paired = True
                        break

        if not paired:
            result.append(lines[i])

    return result


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

        # ---- STEP 3: recompute length ----
        result = []
        for l in unique:
            result.append({
                "line": l,
                "length": self._length(l)
            })

        if self.debug:
            print(f"[Deduplicator] in={len(lines)}, out={len(unique)}")

        if self.debug and base_img is not None:
            self._visualize(base_img, lines, unique)

        return result

    # ===================== NORMALIZE =====================

    def _normalize(self, l):
        (x1, y1), (x2, y2) = l

        x1 = int(round(x1 / self.tol) * self.tol)
        y1 = int(round(y1 / self.tol) * self.tol)
        x2 = int(round(x2 / self.tol) * self.tol)
        y2 = int(round(y2 / self.tol) * self.tol)

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

    # ===================== PARALLEL COLLAPSE =====================

    def _collapse_parallel(self, lines):

        horizontal = []
        vertical = []

        for l in lines:
            (x1,y1),(x2,y2) = l

            if abs(y1-y2) <= self.tol:
                y = int((y1+y2)/2)
                horizontal.append((y, min(x1,x2), max(x1,x2)))

            elif abs(x1-x2) <= self.tol:
                x = int((x1+x2)/2)
                vertical.append((x, min(y1,y2), max(y1,y2)))

        merged = []

        # ---- HORIZONTAL CLUSTER ----
        horizontal.sort()

        used = [False]*len(horizontal)

        for i in range(len(horizontal)):
            if used[i]:
                continue

            y_i, s_i, e_i = horizontal[i]

            group = [(y_i, s_i, e_i)]
            used[i] = True

            for j in range(i+1, len(horizontal)):
                if used[j]:
                    continue

                y_j, s_j, e_j = horizontal[j]

                # close parallel lines
                if abs(y_i - y_j) <= self.tol:

                    overlap = min(e_i, e_j) - max(s_i, s_j)

                    if overlap >= -self.tol:
                        group.append((y_j, s_j, e_j))
                        used[j] = True

            # collapse group → single line
            ys = [g[0] for g in group]
            ss = [g[1] for g in group]
            ee = [g[2] for g in group]

            y = int(sum(ys)/len(ys))
            merged.append(((min(ss), y), (max(ee), y)))

        # ---- VERTICAL CLUSTER ----
        vertical.sort()

        used = [False]*len(vertical)

        for i in range(len(vertical)):
            if used[i]:
                continue

            x_i, s_i, e_i = vertical[i]

            group = [(x_i, s_i, e_i)]
            used[i] = True

            for j in range(i+1, len(vertical)):
                if used[j]:
                    continue

                x_j, s_j, e_j = vertical[j]

                if abs(x_i - x_j) <= self.tol:

                    overlap = min(e_i, e_j) - max(s_i, s_j)

                    if overlap >= -self.tol:
                        group.append((x_j, s_j, e_j))
                        used[j] = True

            xs = [g[0] for g in group]
            ss = [g[1] for g in group]
            ee = [g[2] for g in group]

            x = int(sum(xs)/len(xs))
            merged.append(((x, min(ss)), (x, max(ee))))

        return merged

    # ===================== VISUALIZATION =====================

    def _visualize(self, base_img, before, after):

        if len(base_img.shape) == 2:
            vis = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            vis = base_img.copy()

        # ---- BEFORE (RED) ----
        for (x1, y1), (x2, y2) in before:
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

        # ---- AFTER (GREEN) ----
        for (x1, y1), (x2, y2) in after:
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        h, w = vis.shape[:2]
        scale = min(900 / w, 700 / h, 1.0)
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)))

        cv2.imshow("Deduplicator Debug (Red=Before, Green=After)", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ===================== UTIL =====================

    def _length(self, l):
        (x1, y1), (x2, y2) = l
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5