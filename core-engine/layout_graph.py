import math
from typing import List, Set, Tuple  


Point=Tuple[int, int]
line=Tuple[Point, Point]



class LayoutGraph:


    def __init__(self,snap_tolerance:int=5):
        self.snap_tolerance=snap_tolerance


    def build(self,floors:List[list[line]])->List[dict]:


        if not isinstance(floors, list):
            raise ValueError("Input must be a list of floors.")
        

        layouts=[]

        for walls in floors:

            self._vaidate_walls(walls)

            nodes=self.extract_nodes(walls)

            nodes = self._snap_nodes(nodes)

            graph = self._build_graph(nodes, walls)

            rooms = self._find_cycles(graph)

            layouts.append({
                "nodes": nodes,
                "walls": walls,
                "graph": graph,
                "rooms": rooms
            })

        return layouts
    

    def _vaidate_walls(self,walls:List[line])->None:

        if not isinstance(walls, list):
            raise TypeError("Walls must be a list")

        for wall in walls:

            if not isinstance(wall, tuple) or len(wall) != 2:
                raise ValueError("Invalid wall segment")

            p1, p2 = wall

            if len(p1) != 2 or len(p2) != 2:
                raise ValueError("Wall endpoints must be coordinates")
            

    def _extract_nodes(self, walls: List[line]) -> Set[Point]:

        nodes: Set[Point] = set()

        for p1, p2 in walls:
            nodes.add(p1)
            nodes.add(p2)

        return nodes


    def _snap_nodes(self, nodes: Set[Point]) -> Set[Point]:
        snapped: Set[Point] = set()

        for node in nodes:

            merged = False

            for existing in snapped:

                if self._dist(node, existing) <= self.snap_tol:
                    merged = True
                    break

            if not merged:
                snapped.add(node)

        return snapped
    
    