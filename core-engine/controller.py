from core.image_manager import InputController
from core.normalizer import Normalizer
from core.wall_detector import WallDetector
from intersection import IntersectionDetector
from layout_graph import LayoutGraph
from topology_refiner import TopologyRefiner


from parallel_wall import ParallelWallMerger
from windowdoor import WindowDoorDetector
from typing import List, Dict


class CoreEngine:

    def __init__(self):
        self.manager = InputController()
        self.normalizer = Normalizer()
        self.wall_detector = WallDetector()
        self.intersection = IntersectionDetector()
        self.layout_graph = LayoutGraph()
        self.refiner = TopologyRefiner(self.intersection)
        # self.parallel_wall=ParallelWallMerger()
        # self.window_door_detector=WindowDoorDetector()
    def process(self, path: str) -> List[Dict]:

        images = self.manager.process(path)

        normalized = self.normalizer.normalize(images)

        walls = self.wall_detector.detect(normalized)

        split_walls = self.intersection.process(walls)
        
        refined_walls = []
        for floor in split_walls:
            refined = self.refiner.refine(floor)
            refined_walls.append(refined)

        layout = self.layout_graph.build(refined_walls)

        # parallel_wall=self.parallel_wall.merge(walls)

        # window_door=self.window_door_detector.detect(layout)


        return {
            "walls": walls,
            "split_walls": split_walls,
            "refined_walls": refined_walls,
            "layout": layout
        }