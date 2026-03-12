import cv2
import numpy as np
from typing import List, Tuple, Optional


Line = Tuple[Tuple[int, int], Tuple[int, int]]


class WallDetector:

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 100,
        min_line_length: int = 50,
        max_line_gap: int = 10,
        orientation_tol: int = 10,
        min_wall_length: int = 40
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.orientation_tol = orientation_tol
        self.min_wall_length = min_wall_length


    def detect(self, images: List[np.ndarray]) -> List[List[Line]]:

        if not isinstance(images, list):
            raise ValueError("Input must be a list of images.")

        results: List[List[Line]] = []

        for img in images:

            self._validate_image(img)

            edges = self._detect_edges(img)

            lines = self._detect_lines(edges)

            walls = self._filter_walls(lines)

            merged_walls = self._merge_walls(walls)

            results.append(merged_walls)

        return results


    def _validate_image(self, image: np.ndarray) -> None:

        if image is None:
            raise ValueError("Input image is None")

        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be numpy.ndarray")

        if image.size == 0:
            raise ValueError("Empty image provided")


    def _detect_edges(self, image: np.ndarray) -> np.ndarray:

        return cv2.Canny(image, self.canny_low, self.canny_high)


    def _detect_lines(self, edges: np.ndarray) -> Optional[np.ndarray]:

        return cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )


    def _filter_walls(self, lines: Optional[np.ndarray]) -> List[Line]:

        walls: List[Line] = []

        if lines is None:
            return walls

        for line in lines:

            x1, y1, x2, y2 = line[0]

            length = np.hypot(x2 - x1, y2 - y1)

            if length < self.min_wall_length:
                continue

            if abs(y1 - y2) < self.orientation_tol:
                walls.append(((x1, y1), (x2, y2)))

            elif abs(x1 - x2) < self.orientation_tol:
                walls.append(((x1, y1), (x2, y2)))

        return walls


    def _merge_walls(self, walls: List[Line]) -> List[Line]:

        if not walls:
            return []

        merged: List[Line] = []

        walls = sorted(walls, key=lambda x: (x[0][0], x[0][1]))

        for line in walls:

            if not merged:
                merged.append(line)
                continue

            prev = merged[-1]

            if self._are_collinear(prev, line):
                merged[-1] = self._merge_pair(prev, line)

            else:
                merged.append(line)

        return merged


    def _are_collinear(self, line1: Line, line2: Line) -> bool:

        (x1, y1), (x2, y2) = line1
        (x3, y3), (x4, y4) = line2

        if abs(y1 - y2) < self.orientation_tol and abs(y3 - y4) < self.orientation_tol:
            return abs(y1 - y3) < self.orientation_tol

        if abs(x1 - x2) < self.orientation_tol and abs(x3 - x4) < self.orientation_tol:
            return abs(x1 - x3) < self.orientation_tol

        return False


    def _merge_pair(self, a: Line, b: Line) -> Line:

        points = [a[0], a[1], b[0], b[1]]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x1 = min(xs)
        x2 = max(xs)
        y1 = min(ys)
        y2 = max(ys)

        return ((x1, y1), (x2, y2))