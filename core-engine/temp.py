import cv2
import numpy as np
import networkx as nx
from PIL import Image

from core.image_manager import InputController
from core.normalizer import Normalizer
from core.wall_detector import WallDetector
from core.merger import MergeSystem
from core.intersection_split import IntersectionSplitter
from core.light_graph import GraphBuilder
from core.layout_graph import LayoutGraphNX
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


def overlay_plain_graph(base_img, G, color=(50, 255, 50), thickness=2):
    viz = base_img.copy()
    for u, v in G.edges():
        cv2.line(viz, (int(u[0]), int(u[1])), (int(v[0]), int(v[1])), color, thickness)
    return resize_for_display(viz)


def overlay_layout_rooms(base_img, layout, wall_color=(255, 140, 0), room_color=(0, 255, 0)):
    viz = base_img.copy()

    if len(viz.shape) == 2:
        viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)

    G = layout.get("graph")
    rooms = layout.get("rooms", [])

    if G is not None:
        for u, v in G.edges():
            cv2.line(viz, (int(u[0]), int(u[1])), (int(v[0]), int(v[1])), wall_color, 2)

    for room in rooms:
        for i in range(len(room)):
            p1 = room[i]
            p2 = room[(i + 1) % len(room)]
            cv2.line(viz, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), room_color, 3)

    return resize_for_display(viz)


def graph_stats(G, label=""):
    print(f"\n===== {label} =====")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    if G.number_of_nodes() == 0:
        return
    degrees = [G.degree(n) for n in G.nodes()]
    print(f"Avg Degree: {np.mean(degrees):.2f}")
    print(f"Dangling: {sum(1 for d in degrees if d == 1)}")


def testing():
    print("\n========== PIPELINE START ==========\n")

    input_controller = InputController()
    img_pil = input_controller.process("core-engine/images/floor.jpg")
    if isinstance(img_pil, list):
        img_pil = img_pil[0]

    img = to_numpy(img_pil)

    print("[1] Normalizing...")
    normalizer = Normalizer(debug=False)
    normalized = normalizer.normalize([img_pil])

    print("[2] Detecting walls...")
    wall_detector = WallDetector(debug=False)
    det_inputs = [{"skeleton": x["skeleton"], "stabilized": x["stabilized"]} for x in normalized]
    detections = wall_detector.detect(det_inputs)

    print("[3] Merge hypotheses...")
    merger = MergeSystem(debug=True)
    merged = merger.merge(detections)

    show_image("MERGED - Vote View (Red weak / Orange medium / Yellow strong / Green very strong)", overlay_merge_votes(img, merged, thickness=2))

    print("[4] Gate 1...")
    split_lines = IntersectionSplitter(debug=False).split(merged)
    split_line_list = [d["line"] for d in split_lines]
    show_image("SPLIT - Before Gates", overlay_lines(img, split_line_list, (0, 150, 255), 2))

    G = GraphBuilder(snap_ratio=0.01, debug=False).build(split_lines)
    graph_stats(G, "After GraphBuilder")
    print("After GraphBuilder:", nx.number_connected_components(G))
    show_image("GRAPH - Plain Structure View", overlay_plain_graph(img, G, color=(50, 255, 50), thickness=2))

    # G1 = Gate1(debug=False).apply(G)
    # graph_stats(G1, "After Gate1")
    # print("After Gate1:", nx.number_connected_components(G1))
    # show_image("GATE 1 - Kept Structure (green)", overlay_plain_graph(img, G1, color=(0, 255, 0), thickness=2))

    # Gate1 filtering is disabled for now.
    G1 = G

    print("[5] Topology refine (after Gate1)...")
    gate1_lines = [
        ((int(u[0]), int(u[1])), (int(v[0]), int(v[1])))
        for u, v in G1.edges()
    ]
    topo_refiner = TopologyRefiner(IntersectionSplitter(debug=False), debug=False)
    refined_topology_lines = topo_refiner.refine(gate1_lines, normalized[0]["stabilized"])
    print(f"[TopologyRefiner] lines={len(refined_topology_lines)}")
    show_image("TOPOLOGY REFINED - Before Final Layout", overlay_lines(img, refined_topology_lines, (255, 255, 0), 2))

    print("[6] Layout graph + room extraction...")
    layout_lines = refined_topology_lines
    layout_engine = LayoutGraphNX(snap=10, debug=False)
    layouts = layout_engine.build([layout_lines])
    layout = layouts[0]

    print(f"[LayoutGraph] nodes={layout['graph'].number_of_nodes()}, edges={layout['graph'].number_of_edges()}, rooms={len(layout['rooms'])}")
    layout_viz = overlay_layout_rooms(img, layout)
    show_image("LAYOUT GRAPH OUTPUT - Orange walls / Green rooms", layout_viz)

    print("\n========== DONE ==========\n")
    return {
        "gate1_graph": G1,
        "topology_lines": refined_topology_lines,
        "layout": layout,
    }


if __name__ == "__main__":
    testing()