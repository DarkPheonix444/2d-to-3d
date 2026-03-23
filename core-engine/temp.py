from image_manager import InputController
from normalizer import Normalizer
from wall_detector import WallDetector
from filter import WallFilter
from line_merger import LineMerger
from intersection import IntersectionDetector
from topology_refiner import TopologyRefiner
from topology_visualizer import show_refined_on_original
from layout_graph import LayoutGraphNX

def testing():
    input_controller = InputController()
    normalizer = Normalizer(debug=False)
    wall_detector = WallDetector(debug=True)
    wall_filter = WallFilter()
    line_merger = LineMerger(debug=True)
    intersection = IntersectionDetector(debug=True)
    topology_refiner = TopologyRefiner(intersection_detector=intersection)
    layout=LayoutGraphNX(debug=True)


    images = input_controller.process('core-engine/images/floor.jpg')

    if not isinstance(images, list):
        raise TypeError("InputController must return a list")

    print(f"[DEBUG] Total images: {len(images)}")

    # ---- NORMALIZE ----
    normalized = normalizer.normalize(images)
    print(f"[DEBUG] Normalized count: {len(normalized)}")

    # ---- DETECT ----
    walls = wall_detector.detect(normalized)

    for i, w in enumerate(walls):
        print(f"[DEBUG] Raw walls ({i}): {len(w)}")

    # ---- FILTER ----
    filtered_walls = []
    for i, w in enumerate(walls):
        filtered = wall_filter.filter(w)
        filtered_walls.append(filtered)
        print(f"[DEBUG] Filtered walls ({i}): {len(filtered)}")

    # ---- MERGE ----
    merged_walls = []
    for i, w in enumerate(filtered_walls):
        merged = line_merger.merge(w, normalized[i])
        merged_walls.append(merged)
        print(f"[DEBUG] Merged walls ({i}): {len(merged)}")

    # ---- FIND INTERSECTIONS ----
    intersection_walls = intersection.process(merged_walls, images=normalized)

    # ---- REFINE TOPOLOGY ----
    refined_walls = []
    for i, floor in enumerate(intersection_walls):
        refined = topology_refiner.refine(floor)
        refined_walls.append(refined)
        print(f"[DEBUG] Refined walls ({i}): {len(refined)}")

    # ---- VISUALIZE FIRST FLOOR (REFINED LINES ON ORIGINAL IMAGE) ----
    if images and refined_walls:
        show_refined_on_original(
            original_image=images[0],
            refined_lines=refined_walls[0],
            title="TOPOLOGY REFINED ON ORIGINAL",
            line_color=(255, 0, 0),
            line_thickness=2,
        )
    result = layout.build(refined_walls, images=normalized)

    return {
        "merged_walls": merged_walls,
        "split_walls": intersection_walls,
        "refined_walls": refined_walls,
        "layout": result

    }

if __name__ == "__main__":
    testing()