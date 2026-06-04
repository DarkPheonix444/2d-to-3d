import cv2
import numpy as np


class ParallelProcessor:
    def __init__(self, tol, debug=False):
        self.debug = debug
        self.tol=tol

    def process(self, wall_segments):

        consumed_walls = set()
        resolved_walls = []

        for i in range(len(wall_segments)):

            if i in consumed_walls:
                continue

            w1 = wall_segments[i]

            best_match = None
            best_overlap = 0

            for j in range(i + 1, len(wall_segments)):

                if j in consumed_walls:
                    continue

                w2 = wall_segments[j]

                # same orientation
                if w1["orientation"] != w2["orientation"]:
                    continue

                # same thickness
                if abs(
                    w1["thickness"] -
                    w2["thickness"]
                ) > 3:
                    continue

                # horizontal
                if w1["orientation"] == "H":

                    y1 = w1["centerline"][0][1]
                    y2 = w2["centerline"][0][1]

                    axis_diff = abs(y1 - y2)

                    if axis_diff > self.tol:
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

                    overlap = min(x1e, x2e) - max(x1s, x2s)

                else:

                    x1 = w1["centerline"][0][0]
                    x2 = w2["centerline"][0][0]

                    axis_diff = abs(x1 - x2)

                    if axis_diff > self.tol:
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

                    overlap = min(y1e, y2e) - max(y1s, y2s)

                if overlap <= 0:
                    continue

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = j