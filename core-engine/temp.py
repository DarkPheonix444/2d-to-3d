from image_manager import InputController
from normalizer import Normalizer
from wall_detector import WallDetector
from filter import WallFilter   # 🔥 YOU MISSED THIS


def testing():
    input_controller = InputController()
    normalizer = Normalizer(debug=False)
    wall_detector = WallDetector(debug=False)
    wall_filter = WallFilter(debug=True)   # 🔥 DEBUG HERE

    images = input_controller.process('core-engine/images/test1.jpg')

    # ---- VALIDATE INPUT ----
    if not isinstance(images, list):
        raise TypeError("InputController must return a list")

    print(f"[DEBUG] Total images: {len(images)}")

    # ---- NORMALIZE ----
    normalized = normalizer.normalize(images)

    print(f"[DEBUG] Normalized count: {len(normalized)}")

    # ---- WALL DETECTION ----
    walls = wall_detector.detect(normalized)

    print(f"[DEBUG] Raw walls: {len(walls[0])}")

    # ---- FILTER (🔥 THIS WAS MISSING) ----
    filtered_walls = wall_filter.process(walls)

    print(f"[DEBUG] Filtered walls: {len(filtered_walls[0])}")


testing()