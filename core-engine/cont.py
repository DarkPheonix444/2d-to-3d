import cv2
import numpy as np
from PIL import Image

from core.image_manager import InputController
from core.normalizer import Normalizer
from core.wall_detector import WallDetector

from core.merge_system import (
    MergeSystemV2,
    visualize_merger_v2
)

from core.wall_interfernce import ThicknessInference
from core.appender import WallAppender


def to_numpy(image):
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    return image


def resize_for_display(img, max_width=700, max_height=700):

    h, w = img.shape[:2]

    if h > max_height or w > max_width:
        return cv2.resize(img, (max_width, max_height))

    return img


def show(img, name="preview"):

    if img is None:
        print(f"[preview skipped] {name}")
        return

    preview = resize_for_display(img)

    try:
        cv2.imshow(name, preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception:
        Image.fromarray(preview).show()


def temp_merger_controller(
    image_path,
    debug=True,
    visualize=True
):

    print("\n===== TEMP MERGER CONTROLLER =====\n")

    # --------------------------------------------------
    # LOAD
    # --------------------------------------------------
    ic = InputController()

    img_pil = ic.process(image_path)

    if isinstance(img_pil, list):
        img_pil = img_pil[0]

    img = to_numpy(img_pil)

    scale = max(img.shape[:2])

    # --------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------
    normalizer = Normalizer(debug=False)

    norm = normalizer.normalize([img_pil])

    # --------------------------------------------------
    # DETECT
    # --------------------------------------------------
    wall_detector = WallDetector(debug=False)

    det_inputs = [

        {
            "skeleton": x["skeleton"],
            "stabilized": x["stabilized"]
        }

        for x in norm
    ]

    detections = wall_detector.detect(det_inputs)

    # --------------------------------------------------
    # MERGE
    # --------------------------------------------------
    merger = MergeSystemV2(debug=debug)

    result = merger.merge(
        detections,
        scale=scale
    )

    if merger.debug:

        if (
            not isinstance(result, tuple)
            or len(result) != 4
        ):
            raise ValueError(
                "Expected "
                "(final_data,h_clusters,v_clusters,parallel_lines)"
            )

        (
            final_data,
            h_clusters,
            v_clusters,
            parallel_lines
        ) = result

    else:

        final_data, parallel_lines = result

        h_clusters = []
        v_clusters = []

    print(
        f"[Controller] final_lines = {len(final_data)}"
    )

    # --------------------------------------------------
    # THICKNESS INFERENCE
    # --------------------------------------------------
    thickness_inference = ThicknessInference(
        debug=debug
    )

    wall_objects = thickness_inference.infer_walls(
        final_data
    )

    print(
        f"[Controller] inferred_walls = "
        f"{len(wall_objects)}"
        )

    print("\n===== WALL OBJECTS =====")

    for wall in wall_objects:

        print(
            f"wall_id={wall['wall_id']}"
        )

        print(
            f"edge_ids={wall['edge_ids']}"
        )

        print(
            f"parent_ids={wall.get('parent_ids', [])}"
        )

        print("----------------")

    # --------------------------------------------------
    # BUILD APPENDER OPS
    # --------------------------------------------------
    operations = []

    next_id = (
        max(w["id"] for w in final_data) + 1
        if final_data else 0
    )

    for wall in wall_objects:

        new_wall = {

            "id": next_id,

            "line": wall["centerline"],

            "orientation": wall["orientation"],

            "generator":
                "thickness_inference",

            "parent_ids":
                wall["edge_ids"]
        }

        next_id += 1

        operations.append({

            "consumed_ids":
                wall["edge_ids"],

            "new_walls":
                [new_wall]
        })

    # --------------------------------------------------
    # APPENDER
    # --------------------------------------------------
    appender = WallAppender()

    updated_walls = appender.apply_batch(
        final_data,
        operations
    )

    print(
        f"[Controller] appended_walls = "
        f"{len(updated_walls)}"
    )

    # --------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------
    if visualize:

        overlap_vis, parallel_vis = visualize_merger_v2(
            img,
            h_clusters,
            v_clusters,
            final_data,
            parallel_lines
        )

        show(
            overlap_vis,
            "MERGER V2 OVERLAP VIEW"
        )

        show(
            parallel_vis,
            "MERGER V2 PARALLEL VIEW"
        )

        thickness_inference.visualize(
            img,
            final_data,
            wall_objects
        )

        app_vis = appender.visualize_appender(
            img,
            final_data,
            updated_walls
        )

        show(
            app_vis,
            "APPENDER DEBUG"
        )
        # later:
        # appender.visualize(...)
        # parallel_resolver.visualize(...)

    else:

        print(
            "[Controller] visualization disabled"
        )

    print("\n===== DONE =====\n")

    return updated_walls


if __name__ == "__main__":

    temp_merger_controller(
        "core-engine/v1images/test5.png"
    )