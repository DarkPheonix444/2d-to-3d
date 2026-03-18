import math
from typing import List, Tuple, Dict

Point = Tuple[float, float]
Line = Tuple[Point, Point]

class WindowDoorDetector:

    def __init__(
        self,
        door_gap_min: float = 20,
        door_gap_max: float = 120,
        window_parallel_tol: float = 8
    ):
        self.door_gap_min = door_gap_min
        self.door_gap_max = door_gap_max
        self.window_parallel_tol = window_parallel_tol

    def detect(self,Layout:List[Dict]):

        for floor in Layout:
            walls=floor['walls']

            door=self._detect_doors(walls)
            window=self._detect_windows(walls)

            floor['doors']=door
            floor['windows']=window

        return Layout
    
    def _detect_doors(self,walls:List[Line])->List[Line]:

        doors=[]

        for i in range(len(walls)):
            for j in range(i+1,len(walls)):

                w1=self._normalize_line(walls[i])
                w2=self._normalize_line(walls[j])

                if not self._same_axis_line(w1,w2):
                    continue

                gap_segment = self._door_gap_segment(w1, w2)

                if gap_segment is None:
                    continue

                gap=self._segment_length(gap_segment)

                if self.door_gap_min<=gap<=self.door_gap_max:

                    doors.append(gap_segment)

        return self._dedup_lines(doors)
    
    def _detect_windows(self,walls:List[Line])->List[Line]:

        windows=[]

        for i in range(len(walls)):
            for j in range(i+1,len(walls)):

                w1 = self._normalize_line(walls[i])
                w2 = self._normalize_line(walls[j])

                if self._parallel(w1, w2):

                    dist = self._line_distance(w1, w2)

                    if 1 <= dist < self.window_parallel_tol and self._has_meaningful_overlap(w1, w2):

                        windows.append(w1)

        return self._dedup_lines(windows)

    def _normalize_line(self, line: Line) -> Line:
        a, b = line
        return (a, b) if a <= b else (b, a)

    def _segment_length(self, line: Line) -> float:
        (x1, y1), (x2, y2) = line
        return math.hypot(x2 - x1, y2 - y1)

    def _is_horizontal(self, line: Line) -> bool:
        (x1, y1), (x2, y2) = line
        return abs(y1 - y2) <= 3

    def _is_vertical(self, line: Line) -> bool:
        (x1, y1), (x2, y2) = line
        return abs(x1 - x2) <= 3

    def _same_axis_line(self, a: Line, b: Line) -> bool:
        if self._is_horizontal(a) and self._is_horizontal(b):
            return abs(a[0][1] - b[0][1]) <= 4
        if self._is_vertical(a) and self._is_vertical(b):
            return abs(a[0][0] - b[0][0]) <= 4
        return False

    def _door_gap_segment(self, a: Line, b: Line):
        if self._is_horizontal(a) and self._is_horizontal(b):
            y = int(round((a[0][1] + b[0][1]) / 2))
            a_start, a_end = sorted([a[0][0], a[1][0]])
            b_start, b_end = sorted([b[0][0], b[1][0]])

            if a_end <= b_start:
                return ((a_end, y), (b_start, y))
            if b_end <= a_start:
                return ((b_end, y), (a_start, y))
            return None

        if self._is_vertical(a) and self._is_vertical(b):
            x = int(round((a[0][0] + b[0][0]) / 2))
            a_start, a_end = sorted([a[0][1], a[1][1]])
            b_start, b_end = sorted([b[0][1], b[1][1]])

            if a_end <= b_start:
                return ((x, a_end), (x, b_start))
            if b_end <= a_start:
                return ((x, b_end), (x, a_start))
            return None

        return None

    def _has_meaningful_overlap(self, a: Line, b: Line) -> bool:
        if self._is_horizontal(a) and self._is_horizontal(b):
            a0, a1 = sorted([a[0][0], a[1][0]])
            b0, b1 = sorted([b[0][0], b[1][0]])
            overlap = max(0, min(a1, b1) - max(a0, b0))
            shorter = min(abs(a1 - a0), abs(b1 - b0))
            return shorter >= 15 and overlap >= 0.6 * shorter

        if self._is_vertical(a) and self._is_vertical(b):
            a0, a1 = sorted([a[0][1], a[1][1]])
            b0, b1 = sorted([b[0][1], b[1][1]])
            overlap = max(0, min(a1, b1) - max(a0, b0))
            shorter = min(abs(a1 - a0), abs(b1 - b0))
            return shorter >= 15 and overlap >= 0.6 * shorter

        return False

    def _dedup_lines(self, lines: List[Line]) -> List[Line]:
        seen = set()
        unique = []

        for line in lines:
            norm = self._normalize_line(line)
            key = (tuple(map(float, norm[0])), tuple(map(float, norm[1])))
            if key in seen:
                continue
            seen.add(key)
            unique.append(norm)

        return unique
    

    def _collinear(self, a: Line, b: Line):

        (x1,y1),(x2,y2) = a
        (x3,y3),(x4,y4) = b

        return abs((y2-y1)*(x4-x3) - (x2-x1)*(y4-y3)) < 5

    def _gap_distance(self, a: Line, b: Line):

        (x1,y1),(x2,y2) = a
        (x3,y3),(x4,y4) = b

        return math.hypot(x3-x2, y3-y2)
    
    def _parallel(self, a: Line, b: Line):

        (x1,y1),(x2,y2) = a
        (x3,y3),(x4,y4) = b

        dx1 = x2-x1
        dy1 = y2-y1

        dx2 = x4-x3
        dy2 = y4-y3

        return abs(dx1*dy2 - dy1*dx2) < 5

    def _line_distance(self, a: Line, b: Line):

        (x1,y1),(x2,y2) = a
        (x3,y3),_ = b

        dx = x2-x1
        dy = y2-y1

        return abs(dy*x3 - dx*y3 + x2*y1 - y2*x1) / math.hypot(dx,dy)

