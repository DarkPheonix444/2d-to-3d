

import cv2
import numpy as np
from typing import List, Tuple, Optional


Line = Tuple[Tuple[int, int], Tuple[int, int]]

class wall_detector:


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


    def detect(self,images:List[np.array])->List[list[Line]]:
        if not isinstance(images, list):
            raise ValueError("Input must be a list of images.")
        
        result:List[list[Line]] = []

        for img in images:
            self._validate_image(img)

            edges=self._detect_edges(img)

            lines=self._detect_lines(edges)

            walls=self._filter_walls(lines)

            merged_walls=self._merge_walls(walls)

            result.append(merged_walls)

        return result

    def _validate_image(self, image: np.ndarray) -> None:

        if image is None:
            raise ValueError("Input image is None.")
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be numpy.ndarray")

        if image.size == 0:
            raise ValueError("Empty image provided") 


    def _detect_edges(self, image: np.ndarray) -> np.ndarray:

        edges=cv2.Canny(image, self.canny_low, self.canny_high)

        return edges

    def _detect_lines(self, edges: np.ndarray) -> Optional[np.ndarray]:

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        return lines


    def _filter_walls(self, lines: Optional[np.ndarray]) -> List[Line]:

        walls: List[Line] = []

        if lines is None:
            return walls
        
        for line in lines:

            x1,y1,x2,y2=line[0]

            length = np.hypot(x2 - x1, y2 - y1)

            if length < self.min_wall_length:
                continue

            if abs(y1-y2)<self.orientation_tol:
                walls.append(((x1,y1),(x2,y2)))

            elif abs(x1-x2)<self.orientation_tol:
                walls.append(((x1,y1),(x2,y2)))

        return walls
    

    



