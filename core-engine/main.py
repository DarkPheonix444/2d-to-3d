from controller import CoreEngine
import cv2
import numpy as np
from collections import defaultdict

control = CoreEngine()


def compute_degree(lines):
    deg = defaultdict(int)
    for a, b in lines:
        deg[a] += 1
        deg[b] += 1
    return deg


def _all_points(*line_groups):
    pts = []
    for lines in line_groups:
        for a, b in lines:
            pts.append(a)
            pts.append(b)
    return pts


def _build_transform(points, canvas_w, canvas_h, margin=60):
    if not points:
        return 1.0, 0.0, 0.0

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span_x = max(1, max_x - min_x)
    span_y = max(1, max_y - min_y)

    usable_w = max(1, canvas_w - 2 * margin)
    usable_h = max(1, canvas_h - 2 * margin)

    scale = min(usable_w / span_x, usable_h / span_y)

    off_x = margin - min_x * scale
    off_y = margin - min_y * scale
    return scale, off_x, off_y


def draw_overlay_debug(
    layers,
    size=(1000, 760),
    grid=10,
    title="WALL PIPELINE OVERLAY"
):
    canvas = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255

    all_pts = _all_points(*[lines for _, lines, _ in layers])
    scale, off_x, off_y = _build_transform(all_pts, size[0], size[1])

    # Draw background grid in world-space alignment.
    if all_pts:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        gx0 = (min_x // grid) * grid
        gy0 = (min_y // grid) * grid

        x = gx0
        while x <= max_x + grid:
            sx = int(round(x * scale + off_x))
            cv2.line(canvas, (sx, 0), (sx, size[1]), (230, 230, 230), 1)
            x += grid

        y = gy0
        while y <= max_y + grid:
            sy = int(round(y * scale + off_y))
            cv2.line(canvas, (0, sy), (size[0], sy), (230, 230, 230), 1)
            y += grid

    # Draw selected line sets on the same canvas.
    for _, lines, color in layers:
        for (x1, y1), (x2, y2) in lines:
            p1 = (int(round(x1 * scale + off_x)), int(round(y1 * scale + off_y)))
            p2 = (int(round(x2 * scale + off_x)), int(round(y2 * scale + off_y)))
            cv2.line(canvas, p1, p2, color, 2)

    # Draw dangling endpoints for each layer with outlined circles.
    for _, lines, color in layers:
        deg = compute_degree(lines)
        for (x, y), d in deg.items():
            if d != 1:
                continue
            p = (int(round(x * scale + off_x)), int(round(y * scale + off_y)))
            cv2.circle(canvas, p, 4, (0, 0, 255), -1)
            cv2.circle(canvas, p, 6, color, 1)

    # Simple legend and counts.
    y0 = 30
    layer_names = " + ".join([name for name, _, _ in layers])
    cv2.putText(canvas, f"Overlay: {layer_names}", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    y0 += 28

    for name, lines, color in layers:
        text = f"{name}: {len(lines)} segments"
        cv2.line(canvas, (22, y0 - 5), (62, y0 - 5), color, 3)
        cv2.putText(canvas, text, (70, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        y0 += 24

    cv2.putText(canvas, "Red center = dangling endpoint (degree 1)", (20, y0 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)

    cv2.imshow(title, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    path = "core-engine/floor2.png"

    result = control.process(path)

    walls = result["walls"][0]
    split = result["split_walls"][0]
    refined = result["refined_walls"][0]

    print("\n===== COUNTS =====")
    print("RAW:", len(walls))
    print("SPLIT:", len(split))
    print("REFINED:", len(refined))

    coords = {
        "raw": sorted([(a, b) for a, b in walls]),
        "split": sorted([(a, b) for a, b in split]),
        "refined": sorted([(a, b) for a, b in refined]),
    }

    print("\n===== COORDINATES =====")
    print("RAW COORDS:")
    print(coords["raw"])
    print("\nSPLIT COORDS:")
    print(coords["split"])
    print("\nREFINED COORDS:")
    print(coords["refined"])

    raw_split_layers = [
        ("RAW", walls, (255, 120, 0)),
        ("SPLIT", split, (0, 120, 255)),
    ]
    refined_layer = [
        ("REFINED", refined, (40, 180, 40)),
    ]

    draw_overlay_debug(raw_split_layers, title="RAW vs SPLIT")
    draw_overlay_debug(refined_layer, title="REFINED")

    return coords


if __name__ == "__main__":
    main()