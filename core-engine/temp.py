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
from core.gate_1 import Gate1
from core.gate_2 import Gate2
from text_remover.fallback_ocr import remove_text_with_fallback


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


def stack_text_debug(original, mask, cleaned):
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if len(cleaned.shape) == 2:
        cleaned = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    original = resize_for_display(original)
    mask = resize_for_display(mask)
    cleaned = resize_for_display(cleaned)

    return np.hstack((original, mask, cleaned))


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


def overlay_gate2_labels(base_img, G, thickness=2):
    viz = base_img.copy()
    color_map = {
        "strong": (0, 255, 0),
        "uncertain": (0, 255, 255),
        "weak": (0, 0, 255),
    }

    for u, v, data in G.edges(data=True):
        color = color_map.get(data.get("label", "uncertain"), (180, 180, 180))
        cv2.line(viz, (int(u[0]), int(u[1])), (int(v[0]), int(v[1])), color, thickness)

    legend_items = [
        ("strong", color_map["strong"]),
        ("uncertain", color_map["uncertain"]),
        ("weak", color_map["weak"]),
    ]
    legend_x = 15
    legend_y = 24
    for index, (label, color) in enumerate(legend_items):
        y = legend_y + index * 24
        cv2.rectangle(viz, (legend_x, y - 12), (legend_x + 16, y + 4), color, -1)
        cv2.putText(
            viz,
            label,
            (legend_x + 24, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return resize_for_display(viz)


def overlay_plain_graph(base_img, G, color=(50, 255, 50), thickness=2):
    viz = base_img.copy()
    for u, v in G.edges():
        cv2.line(viz, (int(u[0]), int(u[1])), (int(v[0]), int(v[1])), color, thickness)
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
    img_pil = input_controller.process("core-engine/images/test7.jpg")
    if isinstance(img_pil, list):
        img_pil = img_pil[0]

    img = to_numpy(img_pil)

    print("[0] Removing text...")
    cleaned_img, text_mask = remove_text_with_fallback(img, debug=True)
    show_image("TEXT REMOVAL - Original | Mask | Cleaned", stack_text_debug(img, text_mask, cleaned_img))

    cleaned_pil = Image.fromarray(cleaned_img)

    print("[1] Normalizing...")
    normalizer = Normalizer(debug=True)
    normalized = normalizer.normalize([cleaned_pil])

    print("[2] Detecting walls...")
    wall_detector = WallDetector(debug=True)
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

    G1 = Gate1(debug=False).apply(G)
    graph_stats(G1, "After Gate1")
    print("After Gate1:", nx.number_connected_components(G1))
    show_image("GATE 1 - Kept Structure (green)", overlay_plain_graph(img, G1, color=(0, 255, 0), thickness=2))

    gate2 = Gate2(w_graph=0.4, w_local=0.3, w_ortho=0.3, debug=True)
    G2 = gate2.apply(G1)
    graph_stats(G2, "After Gate2")
    print("After Gate2:", nx.number_connected_components(G2))

    strong_count = weak_count = uncertain_count = 0
    for _, _, data in G2.edges(data=True):
        if data.get("label") == "strong":
            strong_count += 1
        elif data.get("label") == "weak":
            weak_count += 1
        else:
            uncertain_count += 1

    print(f"[Gate2 Verify] strong={strong_count}, uncertain={uncertain_count}, weak={weak_count}")

    gate2_viz = overlay_gate2_labels(img, G2, thickness=2)
    show_image("GATE 2 VERIFY - Green strong / Yellow uncertain / Red weak", gate2_viz)

    print("\n========== DONE ==========\n")
    return G2


if __name__ == "__main__":
    testing()