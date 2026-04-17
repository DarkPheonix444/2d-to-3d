import cv2
import numpy as np
from PIL import Image
import os
import argparse

from core.image_manager import InputController
from core.normalizer import Normalizer
from core.wall_detector import WallDetector
from core.merger import MergeSystem
from core.segment_connector import SegmentConnector
from core.intersection_split import IntersectionSplitter
from core.cleaner import LightCleaner
from core.dedup import Deduplicator
from core.region_detection import RegionDetector
from core.region_refiner import RegionRefiner
from core.topology_refiner import TopologyRefiner


# ===================== HELPERS =====================

def to_numpy(image):
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    return image


def resize_for_display(img, max_width=700, max_height=700):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def show_image(window_name, img):
    try:
        cv2.imshow(window_name, resize_for_display(img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        try:
            preview = img
            if len(preview.shape) == 3:
                preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            Image.fromarray(preview).show(title=window_name)
        except Exception:
            print(f"[preview skipped] {window_name}")


def overlay_lines(base_img, lines, color, thickness=2):
    viz = base_img.copy()
    for line in lines:
        if len(line) != 2:
            continue
        (x1, y1), (x2, y2) = line
        cv2.line(viz, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return resize_for_display(viz)


def overlay_merge_votes(base_img, merged_data, thickness=2):
    viz = base_img.copy()

    for item in merged_data:
        (x1, y1), (x2, y2) = item["line"]
        votes = int(item.get("votes", 1))

        if votes <= 1:
            color = (0, 0, 255)
        elif votes == 2:
            color = (0, 140, 255)
        elif votes <= 4:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        cv2.line(viz, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    return resize_for_display(viz)


def overlay_region_rooms(base_img, walls, rooms, wall_color=(255, 140, 0), room_color=(0, 255, 0)):
    viz = base_img.copy()

    if len(viz.shape) == 2:
        viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)

    for (x1, y1), (x2, y2) in walls:
        cv2.line(viz, (int(x1), int(y1)), (int(x2), int(y2)), wall_color, 2)

    for room in rooms:
        if room is None or len(room) == 0:
            continue
        cv2.drawContours(viz, [room], -1, room_color, 2)

    return resize_for_display(viz)


def overlay_segment_connector(base_img, merged_data, connected_data):
    viz = base_img.copy()

    if len(viz.shape) == 2:
        viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)

    for item in merged_data:
        (x1, y1), (x2, y2) = item["line"]
        cv2.line(viz, (int(x1), int(y1)), (int(x2), int(y2)), (120, 120, 120), 1)

    for item in connected_data:
        (x1, y1), (x2, y2) = item["line"]
        cv2.line(viz, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    return resize_for_display(viz)


def room_stats(rooms, label=""):
    print(f"\n===== {label} =====")
    print(f"Rooms: {len(rooms)}")
    if not rooms:
        return

    areas = [cv2.contourArea(room) for room in rooms if room is not None and len(room) > 0]
    if not areas:
        return

    print(f"Avg Area: {np.mean(areas):.2f}")
    print(f"Largest Area: {np.max(areas):.2f}")
    print(f"Smallest Area: {np.min(areas):.2f}")


def build_adaptive_config(img):
    h, w = img.shape[:2]
    span = max(h, w)

    topo_tol = max(4, min(12, int(round(span * 0.003))))
    topo_max_dist = max(topo_tol * 2, min(40, int(round(span * 0.012))))

    region_thickness = max(2, min(8, int(round(span * 0.003))))

    return {
        "topo_tol": topo_tol,
        "topo_max_dist": topo_max_dist,
        "region_thickness": region_thickness,
    }


def parse_topology_output(refine_output):
    """Support both legacy list return and structured dict return from TopologyRefiner."""
    if isinstance(refine_output, dict):
        lines = refine_output.get("lines") or refine_output.get("refined_lines") or []
        stats = refine_output.get("stats")
        return lines, stats

    if isinstance(refine_output, list):
        return refine_output, None

    raise TypeError(
        "TopologyRefiner.refine() must return list[Line] or dict with 'lines'/'refined_lines'."
    )


class IntersectionSplitAdapter:
    """Bridge IntersectionSplitter.process(...) to a split(...) interface."""

    def __init__(self, splitter):
        self.splitter = splitter

    def split(self, lines_with_votes):
        if not lines_with_votes:
            return []

        line_list = []
        for item in lines_with_votes:
            if isinstance(item, dict) and "line" in item:
                line_list.append(item["line"])
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                line_list.append(item)

        split_lines = self.splitter.process([line_list])[0] if line_list else []
        return [{"line": line, "votes": 1} for line in split_lines]


def testing(image_path="core-engine/images/test7.jpg", visualize=True):
    print("\n========== PIPELINE START ==========\n")

    input_controller = InputController()
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    img_pil = input_controller.process(image_path)
    if isinstance(img_pil, list):
        img_pil = img_pil[0]

    img = to_numpy(img_pil)
    cfg = build_adaptive_config(img)
    print(
        f"topo_tol={cfg['topo_tol']}, topo_max_dist={cfg['topo_max_dist']}, "
        f"region_thickness={cfg['region_thickness']}"
    )

    print("[1] Normalizing...")
    normalizer = Normalizer(debug= False
                            )
    normalized = normalizer.normalize([img_pil])

    print("[2] Detecting walls...")
    wall_detector = WallDetector(debug=False)
    det_inputs = [{"skeleton": x["skeleton"], "stabilized": x["stabilized"]} for x in normalized]
    detections = wall_detector.detect(det_inputs)

    print("[3] Merge hypotheses...")
    merger = MergeSystem(debug=True)
    merged = merger.merge(detections)

    if visualize:
        show_image("MERGED - Vote View (Red weak / Orange medium / Yellow strong / Green very strong)", overlay_merge_votes(img, merged, thickness=2))

    print("[4] Segment connector...")
    connector = SegmentConnector(debug=True)
    connected = connector.connect(merged)
    print(f"[SegmentConnector] lines={len(connected)}")
    if visualize:
        connector_viz = overlay_segment_connector(img, merged, connected)
        show_image("SEGMENT CONNECTOR - Gray merged / Green connected", connector_viz)

    splitter = IntersectionSplitter(tolerance=cfg["topo_tol"], debug=False)
    split_adapter = IntersectionSplitAdapter(splitter)

    print("[5] Split lines...")
    split_lines = split_adapter.split(connected)
    split_line_list = [d["line"] for d in split_lines]
    if visualize:
        show_image("SPLIT - After SegmentConnector", overlay_lines(img, split_line_list, (0, 150, 255), 2))

    print("[6] Cleaner...")
    cleaner = LightCleaner(tol=cfg["topo_tol"], debug=False)
    cleaned_lines = cleaner.clean(split_line_list, base_img=normalized[0]["stabilized"])
    print(f"[LightCleaner] lines={len(cleaned_lines)}")
    if visualize:
        show_image("CLEANED - Before Topology", overlay_lines(img, cleaned_lines, (0, 255, 255), 2))

    print("[7] Deduplicate...")
    deduplicator = Deduplicator(tol=cfg["topo_tol"], debug=False)
    deduped_data = deduplicator.process(cleaned_lines, base_img=normalized[0]["stabilized"])
    deduped_lines = [item["line"] for item in deduped_data]
    print(f"[Deduplicator] lines={len(deduped_lines)}")
    if visualize:
        show_image("DEDUPED - Before Topology", overlay_lines(img, deduped_lines, (255, 0, 255), 2))

    print("[8] Topology refine...")
    topo_refiner = TopologyRefiner(
        split_adapter,
        tol=cfg["topo_tol"],
        debug=False,
    )
    topology_output = topo_refiner.refine(deduped_lines, normalized[0]["stabilized"])
    refined_topology_lines, topology_stats = parse_topology_output(topology_output)
    print(f"[TopologyRefiner] lines={len(refined_topology_lines)}")
    if topology_stats:
        print(f"[TopologyRefiner] stats={topology_stats}")
    if visualize:
        show_image("TOPOLOGY REFINED - Before Final Layout", overlay_lines(img, refined_topology_lines, (255, 255, 0), 2))

    print("[9] Region detection...")
    region_walls = refined_topology_lines or deduped_lines
    if not refined_topology_lines:
        print("[RegionDetector] TopologyRefiner returned no lines; falling back to deduplicated output.")

    image_shape = img.shape
    region_detector = RegionDetector(thickness=cfg["region_thickness"], debug=False)
    region_rooms = region_detector.detect(region_walls, image_shape)

    print(f"[RegionDetector] walls={len(region_walls)}, rooms={len(region_rooms)}")
    room_stats(region_rooms, "After RegionDetector")
    if visualize:
        region_viz = overlay_region_rooms(img, region_walls, region_rooms)
        show_image("REGION DETECTION OUTPUT - Orange walls / Green rooms", region_viz)

    print("[10] Region refine...")
    region_refiner = RegionRefiner(min_area=3000, debug=False)
    refined_rooms = region_refiner.refine(region_rooms, region_walls, image_shape)

    print(f"[RegionRefiner] walls={len(region_walls)}, rooms={len(refined_rooms)}")
    room_stats(refined_rooms, "After RegionRefiner")
    if visualize:
        refined_region_viz = overlay_region_rooms(img, region_walls, refined_rooms)
        show_image("REGION REFINER OUTPUT - Orange walls / Green rooms", refined_region_viz)

    print("\n========== DONE ==========\n")
    return {
        "connected_lines": [d["line"] for d in connected],
        "cleaned_lines": cleaned_lines,
        "deduped_lines": deduped_lines,
        "topology_lines": refined_topology_lines,
        "topology_stats": topology_stats,
        "region_walls": region_walls,
        "region_rooms": region_rooms,
        "regions": refined_rooms,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run robust 2D plan graph and room extraction pipeline.")
    parser.add_argument("--image-path", default="core-engine/images/test7.jpg", help="Input image path")
    parser.add_argument("--no-viz", action="store_true", help="Disable OpenCV visualization windows")
    args = parser.parse_args()

    testing(image_path=args.image_path, visualize=(not args.no_viz))