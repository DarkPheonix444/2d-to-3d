from image_manager import InputController
from normalizer import Normalizer
from wall_detector import WallDetector
from intersection import IntersectionDetector
from layout_graph import LayoutGraph

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

        self.parallel_wall=ParallelWallMerger()
        self.window_door_detector=WindowDoorDetector()
    def process(self, path: str) -> List[Dict]:

        images = self.manager.process(path)

        normalized = self.normalizer.normalize(images)

        walls = self.wall_detector.detect(normalized)

        split_walls = self.intersection.process(walls)

        layout = self.layout_graph.build(split_walls)

        parallel_wall=self.parallel_wall.merge(walls)

        window_door=self.window_door_detector.detect(layout)


        return {
            "walls": walls,
            "split_walls": split_walls,
            "layout": layout,
            "parallel_wall":parallel_wall,
            "window_door":window_door
        }