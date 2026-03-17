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

    def detect(self,Layout:Dict):
        walls=Layout['walls']

        door=self._detect_doors(walls)
        window=self._detect_windows(walls)

        Layout['doors']=door
        Layout['windows']=window

        return Layout
    
    def _detect_doors(self,walls:List[Line])->List[Line]:

        doors=[]

        for i in range(len(walls)):
            for j in range(i+1,len(walls)):

                w1=walls[i]
                w2=walls[j]

                if not self._collinear(w1,w2):
                    continue

                gap=self._gap_distance(w1,w2)

                if self.door_gap_min<=gap<=self.door_gap_max:

                    p1=w1[0]
                    p2=w1[1]

                    doors.append((p1,p2))

        return doors
    
    def _detect_windows(self,walls:List[Line])->List[Line]:

        windows=[]

        for i in range(len(walls)):
            for j in range(i+1,len(walls)):

                w1 = walls[i]
                w2 = walls[j]

                if self._parallel(w1, w2):

                    dist = self._line_distance(w1, w2)

                    if dist < self.window_parallel_tol:

                        windows.append((w1[0], w1[1]))

        return windows
    

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

