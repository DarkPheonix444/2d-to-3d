import cv2
import numpy as np
from typing import List, Tuple
from collections import defaultdict
from shapely.geometry import LineString
from shapely.ops import polygonize

Line = Tuple[Tuple[int, int], Tuple[int, int]]


class WallFilter:

    ANGLE_TOL = 10
    MIN_LEN = 20

    PARALLEL_DIST = 20
    DENSITY_BAND = 50
    DENSITY_GAP = 5

    SMALL_BOX = 120
    SMALL_AREA = 10000

    WALL_LEN = 120

    def __init__(self, debug=True):
        self.debug = debug

    # ===================== MAIN =====================

    def process(self, floors: List[List[Line]]) -> List[List[Line]]:
        results = []

        for idx, lines in enumerate(floors):

            if self.debug:
                print(f"\n=== IMAGE {idx+1} ===")
                self._debug_show(lines, f"{idx+1}_RAW")

            lines = self._orientation_filter(lines)
            if self.debug:
                print(f"After orientation: {len(lines)}")
                self._debug_show(lines, f"{idx+1}_F1_orientation")

            lines = self._length_filter(lines)
            if self.debug:
                print(f"After length: {len(lines)}")
                self._debug_show(lines, f"{idx+1}_F2_length")

            lines = self._density_filter(lines)
            if self.debug:
                print(f"After density: {len(lines)}")
                self._debug_show(lines, f"{idx+1}_F3_density")

            lines = self._parallel_filter(lines)
            if self.debug:
                print(f"After parallel: {len(lines)}")
                self._debug_show(lines, f"{idx+1}_F4_parallel")

            lines = self._remove_small_polygons(lines)
            if self.debug:
                print(f"After polygons: {len(lines)}")
                self._debug_show(lines, f"{idx+1}_F5_polygons")

            lines = self._wall_dominance_filter(lines)
            if self.debug:
                print(f"After dominance: {len(lines)}")
                self._debug_show(lines, f"{idx+1}_F6_dominance")

            results.append(lines)

        return results

    # ===================== FILTERS =====================

    def _orientation_filter(self, lines):
        result = []

        for (x1, y1), (x2, y2) in lines:
            dx, dy = x2 - x1, y2 - y1
            angle = abs(np.degrees(np.arctan2(dy, dx)))

            is_h = angle < self.ANGLE_TOL or abs(angle - 180) < self.ANGLE_TOL
            is_v = abs(angle - 90) < self.ANGLE_TOL

            if not (is_h or is_v):
                continue

            if is_h:
                y = int((y1 + y2) / 2)
                result.append(((min(x1, x2), y), (max(x1, x2), y)))
            else:
                x = int((x1 + x2) / 2)
                result.append(((x, min(y1, y2)), (x, max(y1, y2))))

        return result

    def _length_filter(self, lines):
        return [
            l for l in lines
            if np.hypot(l[1][0]-l[0][0], l[1][1]-l[0][1]) >= self.MIN_LEN
        ]

    def _density_filter(self, lines):
        groups = defaultdict(list)

        for l in lines:
            (x1, y1), (x2, y2) = l

            if y1 == y2:
                band = y1 // self.DENSITY_BAND
                groups[("h", band)].append(l)
            else:
                band = x1 // self.DENSITY_BAND
                groups[("v", band)].append(l)

        result = []

        for _, segs in groups.items():
            segs.sort()

            keep = []

            for l in segs:
                if not keep:
                    keep.append(l)
                    continue

                last = keep[-1]

                if self._line_distance(l, last) < self.DENSITY_GAP:
                    if self._length(l) > self._length(last):
                        keep[-1] = l
                else:
                    keep.append(l)

            result.extend(keep)

        return result

    def _parallel_filter(self, lines):

        used = set()
        result = []

        for i in range(len(lines)):
            if i in used:
                continue

            l1 = LineString([lines[i][0], lines[i][1]])
            keep = lines[i]

            for j in range(i+1, len(lines)):
                if j in used:
                    continue

                l2 = LineString([lines[j][0], lines[j][1]])

                # 🔥 BOTH conditions required
                close = l1.distance(l2) < self.PARALLEL_DIST
                overlap = l1.buffer(2).intersects(l2)

                if close and overlap:
                    if l2.length > l1.length:
                        keep = lines[j]
                        used.add(i)
                    else:
                        used.add(j)

            result.append(keep)

        return result

    def _remove_small_polygons(self, lines):

        shapely_lines = [LineString([l[0], l[1]]) for l in lines]
        polygons = list(polygonize(shapely_lines))

        remove_pts = set()

        for poly in polygons:
            minx, miny, maxx, maxy = poly.bounds
            w, h = maxx - minx, maxy - miny

            if w < self.SMALL_BOX and h < self.SMALL_BOX and poly.area < self.SMALL_AREA:
                for coord in poly.exterior.coords:
                    remove_pts.add((int(coord[0]), int(coord[1])))

        result = []
        for l in lines:
            if l[0] in remove_pts and l[1] in remove_pts:
                continue
            result.append(l)

        return result

    def _wall_dominance_filter(self, lines):

        shapely_lines = [LineString([l[0], l[1]]) for l in lines]

        result = []

        for i, l in enumerate(lines):

            line_geom = shapely_lines[i]

            if line_geom.length > self.WALL_LEN:
                result.append(l)
                continue

            remove = False

            for j, other in enumerate(shapely_lines):
                if i == j:
                    continue

                if other.length > self.WALL_LEN:
                    if line_geom.intersects(other):
                        remove = True
                        break

            if not remove:
                result.append(l)

        return result

    # ===================== DEBUG =====================

    def _debug_show(self, lines, title):

        canvas = np.zeros((1000, 1000, 3), dtype=np.uint8)

        for (x1, y1), (x2, y2) in lines:
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.circle(canvas, (x1, y1), 2, (255, 0, 0), -1)
            cv2.circle(canvas, (x2, y2), 2, (255, 0, 0), -1)

        cv2.imshow(title, canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ===================== UTILS =====================

    def _length(self, l):
        return np.hypot(l[1][0]-l[0][0], l[1][1]-l[0][1])

    def _line_distance(self, a, b):
        la = LineString([a[0], a[1]])
        lb = LineString([b[0], b[1]])
        return la.distance(lb)