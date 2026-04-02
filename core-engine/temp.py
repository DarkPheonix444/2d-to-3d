import cv2
import numpy as np
import networkx as nx
from PIL import Image

from image_manager import InputController
from normalizer import Normalizer
from wall_detector import WallDetector
from merger import MergeSystem
from intersection_split import IntersectionSplitter
from light_graph import GraphBuilder
from gate_1 import Gate1
from gate_2 import Gate2


# ===================== HELPERS =====================

def to_numpy(image):
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    return image


def normalize_edge(e):
    return tuple(sorted(e))


def resize_for_display(img, max_width=700, max_height=700):
    """Resize image to 700px max while maintaining aspect ratio"""
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h))
    return img


def overlay_lines_on_image_resized(base_img, lines, color=(0, 255, 0), thickness=3):
    """Overlay lines on base image, then resize to 700px for display"""
    viz = base_img.copy()
    
    for line in lines:
        if len(line) != 2:
            continue
        (x1, y1), (x2, y2) = line
        cv2.line(viz, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    
    return resize_for_display(viz)


def overlay_graph_on_image_resized(base_img, G, votes_image=False, thickness=2):
    """Overlay graph edges on base image, then resize to 700px for display"""
    viz = base_img.copy()
    
    if G.number_of_edges() == 0:
        return resize_for_display(viz)
    
    if votes_image:
        votes_vals = [G[u][v].get("votes", 1) for u, v in G.edges()]
        if votes_vals:
            min_v, max_v = min(votes_vals), max(votes_vals)
    
    for u, v in G.edges():
        x1, y1 = u
        x2, y2 = v
        
        if votes_image and votes_vals:
            votes = G[u][v].get("votes", 1)
            norm_votes = (votes - min_v) / (max_v - min_v + 1e-6) if max_v > min_v else 0.5
            # Vibrant Blue -> Cyan -> Green -> Yellow -> Red gradient
            if norm_votes < 0.25:
                color = (255, int(norm_votes * 1020), 0)  # Blue to Cyan
            elif norm_votes < 0.5:
                color = (int(255 - (norm_votes - 0.25) * 1020), 255, 0)  # Cyan to Green
            elif norm_votes < 0.75:
                color = (0, 255, int((norm_votes - 0.5) * 1020))  # Green to Yellow
            else:
                color = (0, int(255 - (norm_votes - 0.75) * 1020), 255)  # Yellow to Red
        else:
            color = (50, 255, 50)  # Bright Lime Green
        
        cv2.line(viz, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    
    return resize_for_display(viz)


def show_image(window_name, img):
    """Display image and wait for key press"""
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def overlay_edge_groups_on_image_resized(base_img, keep_edges, other_edges, thickness=2):
    """
    Visual verification for structure confidence:
    - keep_edges: high-confidence structural edges (green)
    - other_edges: lower-confidence/non-wall candidate edges (red)
    """
    viz = base_img.copy()

    for u, v in other_edges:
        cv2.line(viz, (int(u[0]), int(u[1])), (int(v[0]), int(v[1])), (0, 0, 255), thickness)

    for u, v in keep_edges:
        cv2.line(viz, (int(u[0]), int(u[1])), (int(v[0]), int(v[1])), (0, 255, 0), thickness)

    return resize_for_display(viz)


def graph_stats(G, label=""):
    print(f"\n===== {label} =====")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    if G.number_of_nodes() == 0:
        return

    degrees = [G.degree(n) for n in G.nodes()]
    dangling = sum(1 for d in degrees if d == 1)

    print(f"Avg Degree: {np.mean(degrees):.2f}")
    print(f"Dangling: {dangling} ({dangling/len(degrees):.2%})")


# ===================== MAIN PIPELINE =====================

def testing():

    print("\n========== PIPELINE START ==========\n")

    # ---- LOAD (KEEP ORIGINAL SIZE) ----
    print("[1] Loading image (original size)...")
    input_controller = InputController()
    img_pil = input_controller.process("core-engine/images/test7.jpg")
    if isinstance(img_pil, list):
        img_pil = img_pil[0]

    img = to_numpy(img_pil)
    h, w = img.shape[:2]
    print(f"✓ Loaded at original size: {w}x{h}")

    # ---- NORMALIZE (WITH BUILT-IN VISUALIZATION) ----
    print("\n[2] Normalizing (shows its own visualization)...")
    normalizer = Normalizer(debug=False)  # Shows Gray, Binary, Closed, Stabilized, Skeleton
    normalized = normalizer.normalize([img_pil])
    print("✓ Normalized")

    # ---- DETECT (WITH BUILT-IN VISUALIZATION) ----
    print("\n[3] Detecting walls (shows its own visualization)...")
    wall_detector = WallDetector(debug=False)  # Shows detected lines
    images = [{"skeleton": x["skeleton"], "stabilized": x["stabilized"]} for x in normalized]
    detections = wall_detector.detect(images)
    print(f"✓ Detections: {[len(d) for d in detections]} lines per hypothesis")

    # ---- MERGE (WITH BUILT-IN VISUALIZATION) ----
    print("\n[4] Merging with voting (shows its own visualization)...")
    merger = MergeSystem(debug=True)  # Shows merge debug window
    merged = merger.merge(detections)
    print(f"✓ Merged: {len(merged)} lines")

    # ---- SPLIT (temp.py visualization) ----
    print("\n[5] Splitting at intersections (temp.py shows output)...")
    splitter = IntersectionSplitter(debug=False)
    split_lines = splitter.split(merged)
    print(f"✓ Split: {len(split_lines)} segments")
    
    # Visualization at 700px
    split_line_list = [d["line"] for d in split_lines]
    split_img_450 = overlay_lines_on_image_resized(img, split_line_list, color=(0, 150, 255), thickness=3)
    show_image("SPLIT - Intersection Segments (700px)", split_img_450)

    # ---- GRAPH (temp.py visualization) ----
    print("\n[6] Building graph (temp.py shows output)...")
    builder = GraphBuilder(snap_ratio=0.01, debug=False)
    G = builder.build(split_lines)
    graph_stats(G, "Initial Graph")
    print("After GraphBuilder:", nx.number_connected_components(G))
    
    graph_img_votes_450 = overlay_graph_on_image_resized(img, G, votes_image=True, thickness=2)
    show_image("GRAPH - With Vote Coloring (700px)", graph_img_votes_450)

    # ---- GATE 1 (temp.py visualization) ----
    print("\n[7] Applying Gate 1 (short unsupported edges) (temp.py shows output)...")
    gate1 = Gate1(debug=False)
    G1 = gate1.apply(G)
    graph_stats(G1, "After Gate1")
    print("After Gate1:", nx.number_connected_components(G1))
    removed_g1 = G.number_of_edges() - G1.number_of_edges()
    print(f"✓ Removed {removed_g1} edges")
    
    gate1_img_450 = overlay_graph_on_image_resized(img, G1, votes_image=False, thickness=2)
    show_image("GATE 1 - After Short Edge Removal (700px)", gate1_img_450)

    # ---- GATE 2 (temp.py visualization) ----
    print("\n[8] Applying Gate 2 (structural confidence scoring) (temp.py shows output)...")

    bridges_before = list(nx.bridges(G1))
    total_edges = G1.number_of_edges()

    print(f"Bridges found: {len(bridges_before)}")
    print(f"Bridge Ratio: {(len(bridges_before)/total_edges)*100:.2f}%" if total_edges else 0)

    gate2 = Gate2(
        decay=0.85,
        angle_boost=True,
        debug=True
    )

    G2 = gate2.apply(G1)
    graph_stats(G2, "After Gate2")
    print("After Gate2:", nx.number_connected_components(G2))

    # ---- SCORE ANALYSIS (confidence structure vs non-wall candidates) ----
    scored_edges = []
    for u, v, data in G2.edges(data=True):
        scored_edges.append((u, v, float(data.get("score", 0.0))))

    if scored_edges:
        score_vals = [s for _, _, s in scored_edges]
        score_thr = float(np.percentile(score_vals, 40))

        keep_edges = [(u, v) for u, v, s in scored_edges if s >= score_thr]
        other_edges = [(u, v) for u, v, s in scored_edges if s < score_thr]

        print(f"[Gate2 Verify] score_range: {min(score_vals):.3f} -> {max(score_vals):.3f}")
        print(f"[Gate2 Verify] score_threshold(p40): {score_thr:.3f}")
        print(f"[Gate2 Verify] confident_structure_edges: {len(keep_edges)}")
        print(f"[Gate2 Verify] non_wall_candidate_edges: {len(other_edges)}")
    else:
        keep_edges = []
        other_edges = []
        print("[Gate2 Verify] No edges available for score analysis")

    # ---- VISUAL VERIFY: confident vs other structures ----
    verify_img_700 = overlay_edge_groups_on_image_resized(
        img,
        keep_edges=keep_edges,
        other_edges=other_edges,
        thickness=2,
    )
    show_image("GATE 2 VERIFY - Green: confident structure, Red: other/non-wall candidates", verify_img_700)

    # ---- COMPONENT CHECK ----
    comp_before = nx.number_connected_components(G1)
    comp_after = nx.number_connected_components(G2)

    print(f"Components before: {comp_before}")
    print(f"Components after : {comp_after}")
    
    gate2_img_450 = overlay_graph_on_image_resized(img, G2, votes_image=False, thickness=2)
    show_image("GATE 2 - Score Annotated Graph (700px)", gate2_img_450)

    print("\n========== DONE ==========\n")

    return G2


if __name__ == "__main__":
    testing()