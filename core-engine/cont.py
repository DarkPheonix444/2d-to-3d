import cv2
import numpy as np
from PIL import Image

from core.image_manager import InputController
from core.normalizer import Normalizer
from core.wall_detector import WallDetector
from core.merge_system import MergeSystemV2, visualize_merger_v2
from core.wall_interfernce import ThicknessInference


def to_numpy(image):
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    return image


def resize_for_display(img, max_width=700, max_height=700):
    height, width = img.shape[:2]
    if height > max_height or width > max_width:
        return cv2.resize(img, (max_width, max_height))

    return img


def show(img, name="preview"):
    if img is None:
        print(f"[preview skipped] {name} (no image)")
        return

    preview = resize_for_display(img)

    try:
        cv2.imshow(name, preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        Image.fromarray(preview).show()


def temp_merger_controller(image_path, debug=True, visualize=True):

    print("\n===== TEMP MERGER CONTROLLER =====\n")

    # -------- LOAD --------
    ic = InputController()
    img_pil = ic.process(image_path)

    if isinstance(img_pil, list):
        img_pil = img_pil[0]

    img = to_numpy(img_pil)
    scale = max(img.shape[:2])

    # -------- NORMALIZE --------
    normalizer = Normalizer(debug=False)
    norm = normalizer.normalize([img_pil])

    # -------- DETECT --------
    wall_detector = WallDetector(debug=False)
    det_inputs = [
        {"skeleton": x["skeleton"], "stabilized": x["stabilized"]}
        for x in norm
    ]

    detections = wall_detector.detect(det_inputs)

    # -------- MERGE --------
    merger = MergeSystemV2(debug=debug)

    result = merger.merge(detections, scale=scale)

    # -------- HANDLE RETURN CONTRACT --------
    if merger.debug:
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError("MergeSystemV2 must return (final_data, h_clusters, v_clusters) in debug mode")

        final_data, h_clusters, v_clusters = result
    else:
        final_data = result
        h_clusters, v_clusters = [], []

    print(f"[Controller] final_lines = {len(final_data)}")

    # -------- THICKNESS INFERENCE --------
    thickness_inference = ThicknessInference(debug=debug)
    wall_objects = thickness_inference.infer_walls(final_data)

    print(f"[Controller] inferred_walls = {len(wall_objects)}")

    # -------- VISUALIZE --------
    if visualize:
        vis = visualize_merger_v2(img, h_clusters, v_clusters, final_data)
        show(vis, "MERGER V2 DEBUG VIEW")

        thickness_inference.visualize(img, final_data, wall_objects)
    else:
        print("[Controller] visualization disabled")

    print("\n===== DONE =====\n")

    return final_data


if __name__ == "__main__":
    temp_merger_controller("core-engine/v1images/test10.png")